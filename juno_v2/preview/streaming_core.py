"""Whisper streaming preview core — LocalAgreement-2 over the local Whisper decoder.

This module owns the per-utterance audio buffer, VAD gate, and HypothesisBuffer.
On each preview chunk it:

1. Admits/drops the chunk via the VAD gate
2. Appends admitted audio to a growing buffer
3. Decides (wall-clock cadence) whether to run a fresh mlx_whisper.transcribe
4. Promotes agreement-prefix words to ``committed`` via LocalAgreement-2
5. Trims the buffer at confirmed Whisper segment ends (or force-trims at 25 s)
6. Returns ``{committed_text, tail_text, metadata}``

Invariants:
- LocalAgreement-2's ``flush`` only extends ``HypothesisBuffer.committed`` —
  never shrinks it. The append-only property holds inside ``live_agreement.py``.
- ``tail_text`` is the volatile last hypothesis past the agreement; renderers
  MUST treat it as flicker-allowed and visually distinct.
- Three controlled mutations of ``state.hypothesis.committed`` happen in this
  file, each at a documented site:
    1. Trailing-BoH strip after each commit (silence-hallucination defense).
    2. Tail-quarantine rollback when the last commit equals the prior
       quarantined tail with no new audio evidence.
    3. Silence-tail-quarantine rollback under the same condition during a
       silence-confirmation decode when the repeated tail is still unsafe.
    4. Isolated boundary-letter strip for unsupported single-letter commits
       at the start/end of a segment ("D Hey", "better D").
  These are the ONLY places that may shrink ``committed``.

This replaces the prior heuristic merge stack with a single LocalAgreement-2
contract: stable prefixes commit, volatile tails remain quarantinable display
evidence until they are backed by agreement or finalization.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from juno_v2.contracts.preview import PreviewDecodeRequest
from juno_v2.preview.live_agreement import (
    HypothesisBuffer,
    absolute_words,
    force_trim_audio,
    trim_audio_at_segment_end,
)
from juno_v2.preview.orthography import normalize_preview_orthography
from juno_v2.preview.personalization_repair import repair_preview_word_dicts
from juno_v2.preview.vad_gate import VadGate
from juno_v2.runtime.mlx_lock import mlx_decode_guard


# ---------------------------------------------------------------------------
# Decoder protocol — anything that turns audio bytes into Whisper output.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WhisperDecodeOutput:
    """Normalized output of one Whisper decode pass on a buffer."""

    text: str
    language: str | None
    decode_ms: float
    word_dicts: list[dict]            # raw word records from the model, relative timestamps
    segment_ends: list[float]         # confirmed segment-end times, buffer-relative seconds
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingPreviewDecoder(Protocol):
    def warm(self) -> None: ...

    def decode(
        self,
        *,
        audio: np.ndarray,
        sample_rate_hz: int,
        language: str | None,
        allowed_languages: list[str],
        language_policy: str | None,
        initial_prompt: str | None,
        bias_phrases: list[str],
        context_payload: dict,
    ) -> WhisperDecodeOutput: ...


# ---------------------------------------------------------------------------
# Per-utterance state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StreamingPreviewState:
    """Per-utterance state owned by ``StreamingPreviewSessionManager``."""

    utterance_id: str
    buffer_audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    buffer_start_t: float = 0.0
    hypothesis: HypothesisBuffer = field(default_factory=HypothesisBuffer)
    vad: VadGate = field(default_factory=VadGate)
    last_decode_wall_ns: int = 0
    last_emit_committed_text: str = ""
    last_emit_tail_text: str = ""
    last_language: str | None = None
    decode_attempts: int = 0
    decode_skipped_inflight: int = 0
    decode_skipped_cadence: int = 0
    decode_skipped_too_short: int = 0
    decode_skipped_vad_silence: int = 0
    decode_silence_confirmations: int = 0
    commit_events: int = 0
    segment_trim_events: int = 0
    force_trim_events: int = 0
    decodes_since_last_commit: int = 0
    tail_suppressed_events: int = 0
    tail_repeat_drop_events: int = 0
    committed_replay_suppressed_events: int = 0
    committed_replay_agreement_drops: int = 0
    committed_burst_budget_events: int = 0
    commit_draft_horizon_demotions: int = 0
    tail_commit_quarantine_events: int = 0
    committed_boh_strips: int = 0
    committed_boundary_letter_strips: int = 0
    tail_final_promotion_events: int = 0
    tail_final_promotion_blocked_events: int = 0
    last_tail_no_speech_prob: float | None = None
    quarantined_tail_norms: list[str] = field(default_factory=list)
    quarantined_tail_reason: str | None = None
    silence_tail_norms: list[str] = field(default_factory=list)
    silence_tail_quarantine_events: int = 0
    # Fingerprint of the audio buffer at the most recent silence-confirmation
    # decode. We refuse to re-run a silence-confirmation pass over identical
    # audio: it cannot produce new agreement (the HypothesisBuffer state has
    # not advanced) and each call gives the replay-prefix matcher another
    # chance to fire. ``None`` means we have not yet done a silence pass.
    last_silence_decode_buffer_sig: tuple[int, float] | None = None
    created_at_ns: int = field(default_factory=time.perf_counter_ns)
    # Last client activity for this utterance. This is intentionally updated
    # on every chunk, not just VAD-admitted speech: a long pause is still an
    # active utterance while the shell continues streaming chunks.
    updated_at_ns: int = field(default_factory=time.perf_counter_ns)


@dataclass(slots=True)
class StreamingPreviewResponse:
    """Result emitted to the broker per preview chunk."""

    utterance_id: str
    committed_text: str
    tail_text: str
    is_final: bool
    language: str | None
    decode_ms: float
    metadata: dict[str, Any]

    @property
    def text(self) -> str:
        """Combined string for legacy callers. Renderers should prefer the
        committed/tail split."""
        if self.committed_text and self.tail_text:
            return f"{self.committed_text} {self.tail_text}"
        return self.committed_text or self.tail_text


# ---------------------------------------------------------------------------
# The session manager
# ---------------------------------------------------------------------------


class StreamingPreviewSessionManager:
    """Owns per-utterance LocalAgreement-2 state for the preview lane.

    Decode cadence: a fresh ``decoder.decode(...)`` runs at most every
    ``decode_cadence_ms`` of wall-clock time. Audio chunks arriving faster
    than the cadence are buffered and counted as ``decode_skipped_cadence``.
    The single-thread decode owner outside this class already serializes,
    so we never need an explicit in-flight lock here.

    Buffer trim:
    - On every successful decode, if a confirmed Whisper segment end falls
      inside ``committed`` we trim the audio buffer near that timestamp while
      retaining a small carry window for the next preview decode.
    - As a safety, if the buffer ever exceeds ``force_trim_max_seconds`` we
      drop everything but the last ``force_trim_carry_seconds``.
    """

    def __init__(
        self,
        decoder: StreamingPreviewDecoder,
        *,
        sample_rate_hz: int = 16_000,
        decode_cadence_ms: float = 1_000.0,
        min_decode_audio_ms: float = 500.0,
        segment_trim_carry_seconds: float = 2.0,
        force_trim_max_seconds: float = 25.0,
        force_trim_carry_seconds: float = 5.0,
        max_session_idle_seconds: float = 30.0,
        vad_enabled: bool = True,
        commit_draft_horizon_ms: float | None = None,
    ) -> None:
        self.decoder = decoder
        self.sample_rate_hz = sample_rate_hz
        self.decode_cadence_ms = decode_cadence_ms
        self.min_decode_audio_ms = min_decode_audio_ms
        self.segment_trim_carry_seconds = max(0.0, float(segment_trim_carry_seconds))
        self.force_trim_max_seconds = force_trim_max_seconds
        self.force_trim_carry_seconds = force_trim_carry_seconds
        self.max_session_idle_ns = int(max_session_idle_seconds * 1_000_000_000)
        self.vad_enabled = vad_enabled
        # Words whose timestamps end inside the last N ms of buffered audio
        # sit in Whisper's truncated-decode zone — that is where stable
        # hallucinated continuations come from ("…listens to hey Juno" +
        # cut-off speech → "I don't know."). Agreement commits inside the
        # horizon are demoted back to the tail until more audio confirms
        # them. 0 disables (default; production enables via env).
        if commit_draft_horizon_ms is None:
            try:
                commit_draft_horizon_ms = float(
                    os.environ.get("JUNO_V2_PREVIEW_COMMIT_DRAFT_HORIZON_MS", "0") or 0.0
                )
            except ValueError:
                commit_draft_horizon_ms = 0.0
        self.commit_draft_horizon_ms = max(0.0, float(commit_draft_horizon_ms))
        self._states: dict[str, StreamingPreviewState] = {}

    def warm(self) -> None:
        self.decoder.warm()

    # ------------------------------------------------------------------ main

    def process(self, req: PreviewDecodeRequest) -> StreamingPreviewResponse:
        self._prune_idle_states(active_utterance_id=req.utterance_id)
        context_payload = req.context_payload if isinstance(req.context_payload, dict) else {}
        retain_state_after_final = bool(context_payload.get("retain_state_after_final"))
        preview_segment_final = bool(
            context_payload.get("preview_segment_final")
            or retain_state_after_final
        )
        semantic_final = bool(req.is_final and not preview_segment_final)
        preserve_zero_reset = bool(
            context_payload.get("preserve_state_on_decode_seq_zero")
        )
        if req.reset_decoder_state or (req.decode_seq == 0 and not preserve_zero_reset):
            self._states.pop(req.utterance_id, None)

        state = self._states.get(req.utterance_id)
        if state is None:
            state = StreamingPreviewState(utterance_id=req.utterance_id)
            if not self.vad_enabled:
                state.vad.silero_enabled = False
            self._states[req.utterance_id] = state
        state.updated_at_ns = time.perf_counter_ns()

        # 1. VAD gate
        chunk_audio = np.asarray(req.audio, dtype=np.float32).reshape(-1)
        chunk_abs = np.abs(chunk_audio) if chunk_audio.size else np.zeros(0, dtype=np.float32)
        chunk_rms = float(np.sqrt(np.mean(np.square(chunk_audio)))) if chunk_audio.size else 0.0
        chunk_peak = float(np.max(chunk_abs)) if chunk_abs.size else 0.0
        admit_reason = "speech_observed"
        if self.vad_enabled:
            admit, admit_reason = state.vad.should_admit(chunk_audio)
        else:
            admit = chunk_audio.size > 0
            admit_reason = "vad_disabled" if admit else "empty_chunk"

        if admit:
            if state.buffer_audio.size == 0:
                state.buffer_audio = chunk_audio.copy()
            else:
                state.buffer_audio = np.concatenate([state.buffer_audio, chunk_audio])
        else:
            state.decode_skipped_vad_silence += 1

        # 2. Decide whether to decode this tick.
        decoded: WhisperDecodeOutput | None = None
        decode_skip_reason: str | None = None
        decode_on_silence = False
        final_agreement_committed = False
        final_agreement_budget_tail = False
        previous_committed_display_words = _display_word_count(state.hypothesis.committed)
        newly_committed_display_words = 0
        if not admit and not semantic_final:
            # Do not append silence to the audio buffer, but do allow a cadence
            # decode over the existing speech buffer. LocalAgreement needs this
            # second pass to confirm the last spoken words when the user pauses.
            # The previous code skipped decode entirely on VAD silence, which
            # starved commits exactly in the "speak, pause, continue" case.
            decode_on_silence = (
                state.buffer_audio.size > 0
                and (bool(state.hypothesis.tail) or bool(state.hypothesis.committed))
            )
            if not decode_on_silence:
                decode_skip_reason = "vad_silence"
            else:
                # Skip the silence-confirmation decode when the audio buffer
                # is byte-for-byte identical to the buffer we already decoded
                # over in the prior silence pass. Repeating the decode cannot
                # produce new LocalAgreement evidence and just gives the
                # replay-prefix matcher more chances to fire on identical
                # output.
                buffer_sig = (int(state.buffer_audio.size), float(state.buffer_start_t))
                if state.last_silence_decode_buffer_sig == buffer_sig:
                    decode_skip_reason = "silence_decode_already_done_for_buffer"
                else:
                    decode_skip_reason = self._cadence_skip_reason(state, is_final=False)
                    if decode_skip_reason is None:
                        state.decode_silence_confirmations += 1
                        state.last_silence_decode_buffer_sig = buffer_sig
                        decoded = self._run_decode(state, req)
        else:
            decode_skip_reason = self._cadence_skip_reason(state, is_final=semantic_final)
            if decode_skip_reason is None:
                decoded = self._run_decode(state, req)

        # 3. If we decoded, run LocalAgreement-2 + buffer trim.
        tail_suppress_reason: str | None = None
        if decoded is not None:
            decoded.word_dicts, variant_meta = _canonicalize_common_preview_word_variants(
                decoded.word_dicts
            )
            decoded.metadata = {**decoded.metadata, **variant_meta}
            if variant_meta.get("preview_word_variant_repairs"):
                decoded.text = " ".join(
                    str(w.get("word") or w.get("text") or "")
                    for w in decoded.word_dicts
                ).strip()
            repaired_word_dicts, repair_meta = repair_preview_word_dicts(
                decoded.word_dicts,
                context_payload=req.context_payload,
                bias_phrases=req.bias_phrases,
            )
            decoded.word_dicts = repaired_word_dicts
            decoded.metadata = {**decoded.metadata, **repair_meta}
            if repair_meta.get("preview_repair_applied"):
                decoded.text = " ".join(str(w.get("word") or w.get("text") or "") for w in repaired_word_dicts).strip()
            new_words = absolute_words(decoded.word_dicts, state.buffer_start_t)
            previous_quarantined_tail_norms = list(state.quarantined_tail_norms)
            previous_quarantined_tail_reason = state.quarantined_tail_reason
            previous_silence_tail_norms = list(state.silence_tail_norms)
            state.quarantined_tail_norms = []
            state.quarantined_tail_reason = None
            if not decode_on_silence:
                state.silence_tail_norms = []
            replay_drop_reason = state.hypothesis.insert(new_words)
            if replay_drop_reason is not None:
                state.committed_replay_suppressed_events += 1
            newly_committed = state.hypothesis.flush()
            if state.hypothesis.last_flush_replay_drop_reason is not None:
                state.committed_replay_agreement_drops += 1
            live_commit_budget = _MAX_LIVE_COMMIT_WORDS
            budget_prefix_len = _display_budget_prefix_len(newly_committed, live_commit_budget)
            if budget_prefix_len < len(newly_committed):
                excess = newly_committed[budget_prefix_len:]
                if excess:
                    del state.hypothesis.committed[-len(excess):]
                    state.hypothesis.tail = list(excess) + list(state.hypothesis.tail)
                    if semantic_final:
                        final_agreement_budget_tail = True
                    if state.hypothesis.committed:
                        state.hypothesis.last_committed_time = state.hypothesis.committed[-1].end
                    else:
                        state.hypothesis.last_committed_time = 0.0
                    newly_committed = newly_committed[:budget_prefix_len]
                    state.committed_burst_budget_events += 1
            # Draft-horizon demotion: agreed words ending inside the last
            # N ms of buffered audio are truncated-decode territory and may
            # be stable hallucinations ("…hey Juno" + cut-off speech →
            # "I don't know."). Hold them in the tail until more audio
            # arrives. True utterance stop and pause-confirmation decodes
            # bypass the guard — there the buffer end IS the speech end.
            if (
                newly_committed
                and self.commit_draft_horizon_ms > 0
                and not semantic_final
                and not decode_on_silence
            ):
                sr = float(req.sample_rate_hz or self.sample_rate_hz)
                buffer_end_t = state.buffer_start_t + (state.buffer_audio.size / sr)
                horizon_start_t = buffer_end_t - (self.commit_draft_horizon_ms / 1000.0)
                # Real decoder timestamps never exceed the audio length; a
                # word claiming to end past the buffer means inconsistent
                # timing (synthetic feeds, clock drift) — fail open rather
                # than demote on data we cannot reason about.
                epsilon_s = 0.05
                horizon_cut = len(newly_committed)
                while horizon_cut > 0:
                    word_end = newly_committed[horizon_cut - 1].end
                    if word_end > buffer_end_t + epsilon_s or word_end <= horizon_start_t:
                        break
                    horizon_cut -= 1
                if horizon_cut < len(newly_committed):
                    excess = newly_committed[horizon_cut:]
                    del state.hypothesis.committed[-len(excess):]
                    state.hypothesis.tail = list(excess) + list(state.hypothesis.tail)
                    if state.hypothesis.committed:
                        state.hypothesis.last_committed_time = state.hypothesis.committed[-1].end
                    else:
                        state.hypothesis.last_committed_time = 0.0
                    newly_committed = newly_committed[:horizon_cut]
                    state.commit_draft_horizon_demotions += 1
            newly_committed_norms = [w.normalized for w in newly_committed if w.normalized]
            quarantined_tail_commit_safe = _silence_agreement_commit_safe(
                newly_committed,
                decoded=decoded,
                previous_tail_reason=previous_quarantined_tail_reason,
                decode_on_silence=False,
            )
            silence_tail_commit_safe = _silence_agreement_commit_safe(
                newly_committed,
                decoded=decoded,
                previous_tail_reason=None,
                decode_on_silence=decode_on_silence,
            )
            if (
                newly_committed
                and previous_quarantined_tail_norms
                and not state.hypothesis.tail
                and newly_committed_norms == previous_quarantined_tail_norms
                and not quarantined_tail_commit_safe
            ):
                del state.hypothesis.committed[-len(newly_committed):]
                state.hypothesis.tail = list(newly_committed)
                if state.hypothesis.committed:
                    state.hypothesis.last_committed_time = state.hypothesis.committed[-1].end
                else:
                    state.hypothesis.last_committed_time = 0.0
                state.tail_commit_quarantine_events += 1
                state.tail_suppressed_events += 1
                newly_committed = []
                newly_committed_norms = []
            if semantic_final and newly_committed:
                final_agreement_committed = True
            if (
                newly_committed
                and previous_silence_tail_norms
                and decode_on_silence
                and not req.is_final
                and newly_committed_norms
                and newly_committed_norms == previous_silence_tail_norms[: len(newly_committed_norms)]
                and not silence_tail_commit_safe
            ):
                del state.hypothesis.committed[-len(newly_committed):]
                state.hypothesis.tail = list(newly_committed) + list(state.hypothesis.tail)
                if state.hypothesis.committed:
                    state.hypothesis.last_committed_time = state.hypothesis.committed[-1].end
                else:
                    state.hypothesis.last_committed_time = 0.0
                state.silence_tail_quarantine_events += 1
                state.tail_suppressed_events += 1
                newly_committed = []
            if newly_committed:
                state.commit_events += 1
                state.decodes_since_last_commit = 0
                strip_reason = _strip_trailing_committed_boh(state)
                if strip_reason is not None:
                    state.committed_boh_strips += 1
                    state.tail_suppressed_events += 1
                boundary_letter_strip_reason = _strip_unsupported_boundary_letters(state)
                if boundary_letter_strip_reason is not None:
                    state.committed_boundary_letter_strips += 1
                    state.tail_suppressed_events += 1
                # Trim at the latest segment end that falls "inside" committed.
                # Whisper segment ends typically include up to ~300 ms of
                # trailing silence past the last word; we allow 500 ms slack so
                # we don't strand sentence-end pauses inside the buffer.
                committed_end_t = state.hypothesis.last_committed_time
                abs_segment_ends = [state.buffer_start_t + s for s in decoded.segment_ends]
                segment_ends_in_committed = [s for s in abs_segment_ends if s <= committed_end_t + 0.5]
                if segment_ends_in_committed:
                    cut_t = max(segment_ends_in_committed)
                    new_buf, new_start = trim_audio_at_segment_end(
                        state.buffer_audio,
                        state.buffer_start_t,
                        self.sample_rate_hz,
                        cut_t,
                        carry_over_seconds=self.segment_trim_carry_seconds,
                    )
                    if new_buf is not state.buffer_audio:
                        state.buffer_audio = new_buf
                        state.buffer_start_t = new_start
                        state.segment_trim_events += 1
            else:
                state.decodes_since_last_commit += 1
            newly_committed_display_words = max(
                0,
                _display_word_count(state.hypothesis.committed) - previous_committed_display_words,
            )

            # Force-trim safety (long-utterance cap)
            new_buf, new_start, did_force = force_trim_audio(
                state.buffer_audio,
                state.buffer_start_t,
                self.sample_rate_hz,
                max_seconds=self.force_trim_max_seconds,
                carry_over_seconds=self.force_trim_carry_seconds,
            )
            if did_force:
                state.buffer_audio = new_buf
                state.buffer_start_t = new_start
                state.force_trim_events += 1

            if decoded.language:
                state.last_language = decoded.language

        # 4. Build response. Committed/tail come from the HypothesisBuffer.
        committed_text = state.hypothesis.committed_text()
        tail_text = state.hypothesis.tail_text()
        tail_repeat_drop_reason = state.hypothesis.drop_repeated_tail_suffix()
        if tail_repeat_drop_reason is not None:
            state.tail_repeat_drop_events += 1
            state.tail_suppressed_events += 1
            tail_text = state.hypothesis.tail_text()

        # Tail hallucination guards. The committed text is protected by
        # LocalAgreement-2's two-pass invariant, but the tail is the most
        # recent single hypothesis — Whisper sometimes ships a long repeating
        # loop there ("X the same as the same as the same as..."), especially
        # on growing buffers that contain silence. Suppress those.
        tail_text, tail_suppress_reason = _guard_tail_hallucination(tail_text)
        if tail_suppress_reason is not None:
            state.tail_suppressed_events += 1
            if tail_suppress_reason == "tail_boh_suffix" and tail_text:
                keep_count = len(tail_text.split())
                state.hypothesis.tail = state.hypothesis.tail[:keep_count]
            else:
                state.hypothesis.tail = []
        tail_display_suppress_reason: str | None = None
        tail_no_speech_prob: float | None = None
        if tail_text:
            # Pass the tail-most segment's no_speech_prob from this decode
            # so confident single-word utterances render immediately rather
            # than waiting for the full quarantine cycle.
            if decoded is not None:
                raw_nsp = decoded.metadata.get("last_segment_no_speech_prob") if decoded.metadata else None
                if isinstance(raw_nsp, (int, float)):
                    tail_no_speech_prob = float(raw_nsp)
                    state.last_tail_no_speech_prob = tail_no_speech_prob
            if tail_no_speech_prob is None:
                tail_no_speech_prob = state.last_tail_no_speech_prob
            tail_display_suppress_reason = _guard_tail_display(
                tail_text,
                request_audio_ms=max(req.audio_duration_ms, _tail_audio_span_ms(state)),
                no_speech_prob=tail_no_speech_prob,
                vad_admit_reason=admit_reason,
                audio_rms=chunk_rms,
                audio_peak=chunk_peak,
                decode_on_silence=decode_on_silence,
                is_final=semantic_final,
            )
            if tail_display_suppress_reason is not None:
                state.tail_suppressed_events += 1
                # The HUD must not render untrusted tails. Keep the words only
                # as internal evidence so a later real continuation can
                # confirm them. Only audio-risk tails are commit-quarantined:
                # active-speech single words and tiny phrases should be hidden
                # from the HUD until agreement, but must still be able to
                # become the user's final word once repeated by real speech.
                if tail_display_suppress_reason in _TAIL_COMMIT_QUARANTINE_REASONS:
                    state.quarantined_tail_norms = [
                        w.normalized for w in state.hypothesis.tail if w.normalized
                    ]
                    state.quarantined_tail_reason = tail_display_suppress_reason
                else:
                    state.quarantined_tail_norms = []
                    state.quarantined_tail_reason = None
                tail_text = ""
            else:
                state.quarantined_tail_norms = []
                state.quarantined_tail_reason = None
        else:
            state.last_tail_no_speech_prob = None

        # Note: trailing-BoH strip already runs adjacent to each commit above.
        # Nothing between that strip and here mutates ``state.hypothesis.committed``
        # (the tail guards above only touch ``state.hypothesis.tail``), so a
        # second strip at this point would always be a no-op. The post-final-
        # drain strip below remains because the drain DOES extend committed.

        if preview_segment_final:
            tail_promotion_max_words = _MAX_SEGMENT_FINAL_TAIL_PROMOTION_WORDS
        elif semantic_final and not state.hypothesis.committed:
            tail_promotion_max_words = _MAX_FINAL_EMPTY_COMMITTED_TAIL_PROMOTION_WORDS
        else:
            tail_promotion_max_words = _MAX_FINAL_TAIL_PROMOTION_WORDS

        # On final, only drain a tail that survived the same trust gates used
        # for display. A hidden/quarantined tail is explicitly untrusted: making
        # it opaque here is how silence guesses like "Session here" and stray
        # boundary letters leaked into the HUD. Segment-final flushes are
        # different from utterance-final flushes: keep unsafe tails as internal
        # evidence so later audio can still confirm them.
        tail_final_promotion_status = "agreement_committed" if final_agreement_committed else "none"
        tail_final_promotion_reason: str | None = None
        if semantic_final and state.hypothesis.tail:
            unsafe_tail_reason = tail_suppress_reason or tail_display_suppress_reason
            confirmed_budget_tail = (
                final_agreement_budget_tail
                and unsafe_tail_reason == "tail_single_word_quarantine"
                and tail_suppress_reason is None
            )
            if unsafe_tail_reason is not None and not confirmed_budget_tail:
                tail_final_promotion_status = "blocked"
                tail_final_promotion_reason = unsafe_tail_reason
                state.tail_final_promotion_blocked_events += 1
                if not retain_state_after_final:
                    state.hypothesis.tail = []
                tail_text = ""
            elif _display_word_count(state.hypothesis.tail) > tail_promotion_max_words:
                tail_final_promotion_status = "deferred"
                tail_final_promotion_reason = "tail_promotion_chunk_budget"
                state.tail_final_promotion_blocked_events += 1
                tail_text = state.hypothesis.tail_text()
            else:
                tail_final_promotion_status = "promoted"
                state.tail_final_promotion_events += 1
                promoted_tail = list(state.hypothesis.tail)
                promoted_tail, boundary_revision_reason = (
                    state.hypothesis.resolve_boundary_revision_before_append(promoted_tail)
                )
                if boundary_revision_reason is not None:
                    state.committed_replay_agreement_drops += 1
                old_len = len(state.hypothesis.committed)
                state.hypothesis.committed.extend(promoted_tail)
                promoted_tail, duplicate_drop_reason = (
                    state.hypothesis.drop_adjacent_duplicate_boundary_after_append(
                        old_len,
                        promoted_tail,
                        allow_committed_suffix_drop=True,
                    )
                )
                if duplicate_drop_reason is not None:
                    state.committed_replay_agreement_drops += 1
                if state.hypothesis.committed:
                    state.hypothesis.last_committed_time = state.hypothesis.committed[-1].end
                else:
                    state.hypothesis.last_committed_time = 0.0
                state.hypothesis.tail = []
                committed_text = state.hypothesis.committed_text()
                # Apply strip again on the post-final drain (in case the tail
                # contained a BoH phrase or unsupported boundary letter that
                # survived the per-tick guards).
                final_strip_reason = _strip_trailing_committed_boh(state)
                if final_strip_reason is not None:
                    state.committed_boh_strips += 1
                    state.tail_suppressed_events += 1
                    committed_text = state.hypothesis.committed_text()
                final_letter_strip_reason = _strip_unsupported_boundary_letters(state)
                if final_letter_strip_reason is not None:
                    state.committed_boundary_letter_strips += 1
                    state.tail_suppressed_events += 1
                    committed_text = state.hypothesis.committed_text()
                tail_text = ""
            state.last_tail_no_speech_prob = None
        elif req.is_final and preview_segment_final and state.hypothesis.tail:
            tail_final_promotion_status = "deferred"
            tail_final_promotion_reason = "segment_boundary"

        if bool(context_payload.get("preview_display_orthography")):
            committed_text, tail_text, orthography_meta = normalize_preview_orthography(
                committed_text,
                tail_text,
                protected_terms=_preview_orthography_protected_terms(context_payload, req.bias_phrases),
            )
        else:
            orthography_meta = {
                "preview_orthography_applied": 0,
                "preview_orthography_committed_changed": False,
                "preview_orthography_tail_changed": False,
            }
        state.last_emit_committed_text = committed_text
        state.last_emit_tail_text = tail_text

        if req.is_final:
            pass
        elif decoded is not None and decode_on_silence:
            state.silence_tail_norms = [w.normalized for w in state.hypothesis.tail if w.normalized]
        elif decoded is not None:
            state.silence_tail_norms = []

        metadata: dict[str, Any] = {
            "decode_seq": req.decode_seq,
            "model_streaming_mode": "local_agreement_2",
            "model_is_native_streaming": False,
            "vad_admit_reason": admit_reason,
            "decode_skip_reason": decode_skip_reason,
            "buffer_audio_ms": (state.buffer_audio.size / self.sample_rate_hz) * 1000.0,
            "buffer_start_t": state.buffer_start_t,
            "committed_chars": len(committed_text),
            "tail_chars": len(tail_text),
            "committed_word_count": len(state.hypothesis.committed),
            "tail_word_count": len(state.hypothesis.tail),
            "decode_attempts": state.decode_attempts,
            "decode_skipped_cadence": state.decode_skipped_cadence,
            "decode_skipped_inflight": state.decode_skipped_inflight,
            "decode_skipped_too_short": state.decode_skipped_too_short,
            "decode_skipped_vad_silence": state.decode_skipped_vad_silence,
            "decode_silence_confirmations": state.decode_silence_confirmations,
            "decode_on_silence": decode_on_silence,
            "commit_events": state.commit_events,
            "segment_trim_events": state.segment_trim_events,
            "segment_trim_carry_seconds": self.segment_trim_carry_seconds,
            "force_trim_events": state.force_trim_events,
            "tail_suppressed_events": state.tail_suppressed_events,
            "tail_repeat_drop_events": state.tail_repeat_drop_events,
            "tail_repeat_drop_reason": tail_repeat_drop_reason,
            "committed_replay_suppressed_events": state.committed_replay_suppressed_events,
            "committed_replay_agreement_drops": state.committed_replay_agreement_drops,
            "committed_burst_budget_events": state.committed_burst_budget_events,
            "commit_draft_horizon_demotions": state.commit_draft_horizon_demotions,
            "committed_burst_max_words": _MAX_LIVE_COMMIT_WORDS,
            "tail_commit_quarantine_events": state.tail_commit_quarantine_events,
            "silence_tail_quarantine_events": state.silence_tail_quarantine_events,
            "committed_boh_strips": state.committed_boh_strips,
            "committed_boundary_letter_strips": state.committed_boundary_letter_strips,
            "tail_suppress_reason": tail_suppress_reason or tail_display_suppress_reason,
            "tail_display_suppress_reason": tail_display_suppress_reason,
            "tail_quarantine_reason": state.quarantined_tail_reason,
            "tail_no_speech_prob": tail_no_speech_prob,
            "request_audio_rms": chunk_rms,
            "request_audio_peak": chunk_peak,
            "tail_final_promotion_status": tail_final_promotion_status,
            "tail_final_promotion_reason": tail_final_promotion_reason,
            "tail_final_promotion_max_words": tail_promotion_max_words,
            "newly_committed_display_words": newly_committed_display_words,
            "tail_final_promotion_events": state.tail_final_promotion_events,
            "tail_final_promotion_blocked_events": state.tail_final_promotion_blocked_events,
            "silence_tail_quarantine_active": bool(state.silence_tail_norms),
            "decodes_since_last_commit": state.decodes_since_last_commit,
            **orthography_meta,
        }
        decode_ms = 0.0
        if decoded is not None:
            decode_ms = decoded.decode_ms
            metadata.update(decoded.metadata)

        if req.is_final:
            metadata["finalized"] = not retain_state_after_final
            if retain_state_after_final:
                metadata["segment_finalized"] = True
            # P3.2: include the full final committed text in the metadata so
            # post-utterance audits don't have to grep the engine log.
            metadata["committed_text_final"] = committed_text[:2000]
            if not retain_state_after_final:
                self._states.pop(req.utterance_id, None)

        return StreamingPreviewResponse(
            utterance_id=req.utterance_id,
            committed_text=committed_text,
            tail_text=tail_text,
            is_final=req.is_final,
            language=state.last_language or req.language,
            decode_ms=decode_ms,
            metadata=metadata,
        )

    # ----------------------------------------------------------------- helpers

    def _cadence_skip_reason(
        self, state: StreamingPreviewState, *, is_final: bool
    ) -> str | None:
        if is_final:
            return None
        if state.buffer_audio.size == 0:
            state.decode_skipped_too_short += 1
            return "buffer_empty"
        buffer_ms = (state.buffer_audio.size / self.sample_rate_hz) * 1000.0
        if buffer_ms < self.min_decode_audio_ms:
            state.decode_skipped_too_short += 1
            return "buffer_too_short"
        if state.last_decode_wall_ns > 0:
            elapsed_ms = (time.perf_counter_ns() - state.last_decode_wall_ns) / 1_000_000.0
            if elapsed_ms < self.decode_cadence_ms:
                state.decode_skipped_cadence += 1
                return "cadence_not_elapsed"
        return None

    def _run_decode(
        self, state: StreamingPreviewState, req: PreviewDecodeRequest
    ) -> WhisperDecodeOutput:
        initial_prompt = self._build_initial_prompt(state, req)
        state.decode_attempts += 1
        state.last_decode_wall_ns = time.perf_counter_ns()
        return self.decoder.decode(
            audio=state.buffer_audio,
            sample_rate_hz=req.sample_rate_hz or self.sample_rate_hz,
            language=req.language,
            allowed_languages=list(req.allowed_languages),
            language_policy=req.language_policy,
            initial_prompt=initial_prompt,
            bias_phrases=list(req.bias_phrases),
            context_payload=dict(req.context_payload),
        )

    def _build_initial_prompt(
        self, state: StreamingPreviewState, req: PreviewDecodeRequest
    ) -> str | None:
        payload = req.context_payload if isinstance(req.context_payload, dict) else {}
        prompt_mode = str(payload.get("preview_prompt_mode") or "").strip().lower()
        if prompt_mode in {"", "acoustic_first", "personalization_repair_only"}:
            return None
        # Whisper's initial_prompt is capped at ~224 tokens. We approximate
        # token budget at ~3.5 chars per token and stay well under.
        #
        # Critical: strip trailing Tier-A BoH phrases from the committed
        # history before feeding it as prompt. If "Thank you" or
        # "Thanks for watching" sneaked into ``committed`` (despite Layers
        # 1-3), including it in the prompt biases the next decode toward
        # continuing the YouTube-corpus pattern (this is exactly the
        # lock-in we observed in round 4: once a hallucination committed,
        # every subsequent decode regenerated it from the prompt).
        from juno_v2.preview.hallucination_blocklist import strip_trailing_boh

        parts: list[str] = []
        if state.hypothesis.committed:
            committed_words = [w.text for w in state.hypothesis.committed[-150:]]
            committed_str = " ".join(committed_words)
            cleaned, _removed = strip_trailing_boh(committed_str)
            if cleaned:
                parts.append(cleaned)
        if req.bias_phrases:
            parts.append("bias: " + ", ".join(req.bias_phrases[:16]))
        candidate_entities = (
            req.context_payload.get("candidate_entities")
            if isinstance(req.context_payload, dict)
            else None
        )
        if isinstance(candidate_entities, list) and candidate_entities:
            parts.append(
                "context: " + ", ".join(str(x) for x in candidate_entities[:16])
            )
        if req.initial_prompt:
            parts.append(req.initial_prompt)
        if not parts:
            return None
        prompt = " | ".join(parts)
        # 600 chars ≈ 170 tokens — safe margin under the 224-token cap.
        return prompt[-600:]

    def _prune_idle_states(self, *, active_utterance_id: str | None = None) -> None:
        now_ns = time.perf_counter_ns()
        stale = [
            uid
            for uid, st in self._states.items()
            if uid != active_utterance_id and now_ns - st.updated_at_ns > self.max_session_idle_ns
        ]
        for uid in stale:
            self._states.pop(uid, None)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


class MlxWhisperStreamingDecoder:
    """Whisper-large-v3-turbo wrapped for LocalAgreement-2 use on Apple Silicon.

    Holds the MLX Whisper model resident. Each ``decode(...)`` call runs one
    pass with ``word_timestamps=True`` so the upstream LocalAgreement state
    machine can do agreement on word boundaries and trim audio at segment ends.

    Decode parameters are fixed for live use:
        beam_size=1 (greedy), temperature=0.0, condition_on_previous_text=True,
        word_timestamps=False, fp16=True. The fallback temperature ladder is
        explicitly disabled. We use beam_size=1 + no-word-timestamps because
        on short audio (< 1 s padded to 30 s) mlx-whisper's beam=5 + word
        alignment can take 10+ seconds — long enough to time out the 15 s
        broker → preview-service HTTP call. Greedy decode + segment-level
        timestamps gives the same LocalAgreement-2 behavior at < 200 ms per
        decode on M-series.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        hf_repo_id: str | None = None,
        beam_size: int = 1,
    ) -> None:
        self.model_path = Path(str(model_path))
        self.hf_repo_id = (hf_repo_id or "").strip() or None
        self.beam_size = beam_size
        self._mlx_whisper = None
        self._loaded_model = None
        self._model_ref: str | None = None
        self._model_source_type: str | None = None
        self._decode_primed = False

    def warm(self) -> None:
        if self._mlx_whisper is not None:
            self._prime_decode()
            return
        try:
            import mlx_whisper  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "mlx-whisper is required for the MLX Whisper streaming preview service"
            ) from exc
        from juno_v2.final.backends.mlx_whisper import MlxWhisperFinalBackend

        MlxWhisperFinalBackend._patch_model_dimensions(mlx_whisper)
        model_ref, source_type = self._resolve_model_ref()
        self._mlx_whisper = mlx_whisper
        self._model_ref = model_ref
        self._model_source_type = source_type
        try:
            from mlx_whisper import load_models  # type: ignore

            load_kwargs: dict[str, Any] = {"path_or_hf_repo": model_ref}
            model_dtype = MlxWhisperFinalBackend._transcribe_fp16_dtype(mlx_whisper)
            if model_dtype is not None:
                load_kwargs["dtype"] = model_dtype
            self._loaded_model = load_models.load_model(**load_kwargs)
            try:
                from mlx_whisper.transcribe import ModelHolder  # type: ignore

                ModelHolder.model = self._loaded_model
                ModelHolder.model_path = model_ref
            except Exception:
                pass
        except Exception as exc:
            self._loaded_model = None
            raise RuntimeError(
                f"MLX Whisper streaming preview failed to load model "
                f"(source_type={source_type}, ref={model_ref}): {type(exc).__name__}: {exc}"
            ) from exc
        self._prime_decode()

    def _prime_decode(self) -> None:
        if self._decode_primed or self._mlx_whisper is None:
            return
        model_ref = self._model_ref or self._resolve_model_ref()[0]
        # JIT the decode path on startup so the first user utterance does not
        # pay for kernel compilation.
        audio = np.zeros(16_000, dtype=np.float32)
        with mlx_decode_guard():
            self._mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=model_ref,
                language="en",
                fp16=True,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
        self._decode_primed = True

    def decode(
        self,
        *,
        audio: np.ndarray,
        sample_rate_hz: int,
        language: str | None,
        allowed_languages: list[str],
        language_policy: str | None,
        initial_prompt: str | None,
        bias_phrases: list[str],
        context_payload: dict,
    ) -> WhisperDecodeOutput:
        self.warm()
        assert self._mlx_whisper is not None
        model_ref = self._model_ref or self._resolve_model_ref()[0]
        started = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        with mlx_decode_guard():
            result = self._mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=model_ref,
                language=language,
                initial_prompt=initial_prompt,
                fp16=True,
                word_timestamps=False,
                # condition_on_previous_text=True is what every reference
                # streaming implementation uses (whisper_streaming line 249,
                # faster-whisper default, WhisperX default, WhisperLive
                # default). Without it, Whisper sees our 1.5-s audio padded
                # to 30 s with zeros and has no acoustic prior — every
                # segment trips its built-in `no_speech_prob > 0.6 AND
                # avg_logprob < -1.0` filter, and `transcribe()` returns
                # empty text on quiet-but-real speech (observed in round 5
                # trace: RMS 0.02-0.04 = normal voice, but raw='' for 8 s
                # straight).
                #
                # The hallucination lock-in this was trying to prevent is
                # instead handled by stripping BoH-blocklisted phrases from
                # the trailing committed words before they become Whisper's
                # ``initial_prompt`` (see ``_build_initial_prompt`` and
                # the ``_strip_trailing_boh_words`` helper in this file).
                # That cleans the prompt while leaving the algorithm's
                # append-only HypothesisBuffer invariant intact.
                condition_on_previous_text=True,
                temperature=0.0,
                no_speech_threshold=0.6,
            )
        decode_ms = (time.perf_counter() - started) * 1000.0
        text = str(result.get("text", "") or "").strip()
        detected_language = result.get("language") or language
        # We deliberately do NOT request word_timestamps=True from mlx-whisper:
        # on short padded audio that pass takes 10+ seconds, longer than the
        # 15 s broker→subprocess HTTP timeout. Instead we build synthetic
        # word records by splitting each segment's text on whitespace and
        # linearly interpolating timestamps inside the segment span. This is
        # less precise than the model's own word alignment but is plenty
        # accurate for LocalAgreement-2 (which only needs monotonic time
        # anchors for n-gram dedupe and segment-end audio trim).
        #
        # Two hallucination defences operate per-segment, BEFORE any words
        # reach LocalAgreement-2:
        #   Layer 2 — no_speech_prob filter: Whisper's own per-segment
        #     no_speech_prob signal. The library's built-in filter requires
        #     `no_speech_prob > threshold AND avg_logprob < log_prob_threshold`
        #     (transcribe.py); we drop on no_speech_prob > 0.6 ALONE because
        #     the AND-gate is what lets "Thank you" through (Whisper has high
        #     avg_logprob for its frequent training-set hallucinations).
        #   Layer 3 — BoH (Bag of Hallucinations) blocklist: arXiv:2501.11378
        #     showed 24.76% of Whisper non-speech outputs are "thank you",
        #     10.32% are "thanks for watching". We drop segments whose entire
        #     text matches a known hallucination phrase, gated by
        #     no_speech_prob > 0.3 so we still admit a user genuinely
        #     dictating "thank you" inside real speech.
        from juno_v2.preview.hallucination_blocklist import (
            is_whisper_silence_hallucination,
        )
        word_dicts: list[dict] = []
        segment_ends: list[float] = []
        dropped_segments: list[dict[str, Any]] = []
        # Track the no_speech_prob of the LAST kept segment. The tail words
        # come from the most-recent segment in time; gating the tail-display
        # quarantine on this value lets real one-word utterances ("yes",
        # "stop", "play") render without the conservative single-word block
        # while still suppressing Whisper's silence hallucinations.
        last_segment_no_speech_prob: float | None = None
        segments = result.get("segments") or []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            no_speech_prob = seg.get("no_speech_prob")
            try:
                no_speech_prob_f = float(no_speech_prob) if no_speech_prob is not None else None
            except (TypeError, ValueError):
                no_speech_prob_f = None
            seg_text_raw = str(seg.get("text") or "").strip()

            # Layer 2: no_speech_prob > 0.6 → drop the segment outright.
            if no_speech_prob_f is not None and no_speech_prob_f > 0.6:
                dropped_segments.append({
                    "text": seg_text_raw[:120],
                    "no_speech_prob": no_speech_prob_f,
                    "reason": "no_speech_prob_above_0_6",
                })
                continue

            # Layer 3: BoH blocklist with no_speech_prob > 0.3 conditional gate.
            # The conditional gate prevents legitimate "Thank you." utterances
            # (where Whisper is confident, no_speech_prob ~0.05) from being
            # dropped, while still catching silence hallucinations (where
            # Whisper itself signals doubt, no_speech_prob ~0.5–0.95).
            if (
                seg_text_raw
                and is_whisper_silence_hallucination(seg_text_raw)
                and no_speech_prob_f is not None
                and no_speech_prob_f > 0.3
            ):
                dropped_segments.append({
                    "text": seg_text_raw[:120],
                    "no_speech_prob": no_speech_prob_f,
                    "reason": "boh_blocklist",
                })
                continue

            # Segment passed both filters — collect its segment_end and words.
            seg_end = seg.get("end")
            if seg_end is not None:
                try:
                    segment_ends.append(float(seg_end))
                except (TypeError, ValueError):
                    pass
            try:
                seg_start = float(seg.get("start") or 0.0)
                seg_end_f = float(seg.get("end") or seg_start)
            except (TypeError, ValueError):
                continue
            if not seg_text_raw:
                continue
            tokens = seg_text_raw.split()
            if not tokens:
                continue
            # Record this segment's no_speech_prob; iteration is in time order
            # so the final value is the tail-most segment's confidence.
            if no_speech_prob_f is not None:
                last_segment_no_speech_prob = no_speech_prob_f
            span = max(seg_end_f - seg_start, 0.01)
            step = span / max(len(tokens), 1)
            for idx, tok in enumerate(tokens):
                w_start = seg_start + idx * step
                w_end = seg_start + (idx + 1) * step
                word_dicts.append({"word": tok, "start": w_start, "end": w_end})
        text, word_dicts, leading_letter_stripped = _strip_leading_hallucinated_letter(
            text,
            word_dicts,
        )
        return WhisperDecodeOutput(
            text=text,
            language=detected_language,
            decode_ms=decode_ms,
            word_dicts=word_dicts,
            segment_ends=segment_ends,
            metadata={
                "mlx_whisper_decode_ms": decode_ms,
                "mlx_whisper_model_ref": model_ref,
                "mlx_whisper_model_source_type": self._model_source_type,
                "leading_hallucination_stripped": leading_letter_stripped,
                "prompt_bias_phrase_count": len(bias_phrases),
                "prompt_context_entity_count": _context_entity_count(context_payload),
                "requested_language": language,
                "allowed_languages": list(allowed_languages),
                "language_policy": language_policy,
                "segment_count": len(segments),
                "segments_dropped": len(dropped_segments),
                "dropped_segments_preview": dropped_segments[:4],
                "word_count": len(word_dicts),
                "last_segment_no_speech_prob": last_segment_no_speech_prob,
            },
        )

    @staticmethod
    def _has_mlx_weights(model_path: Path) -> bool:
        return (model_path / "weights.safetensors").exists() or (model_path / "weights.npz").exists()

    def _resolve_model_ref(self) -> tuple[str, str]:
        if self.model_path.exists() and self._has_mlx_weights(self.model_path):
            return str(self.model_path), "local_path"
        if self.hf_repo_id:
            return self.hf_repo_id, "hf_repo"
        model_ref = str(self.model_path)
        if "/" in model_ref and not self.model_path.exists():
            return model_ref, "hf_repo"
        return model_ref, "unknown"


class FasterWhisperStreamingDecoder:
    """faster-whisper backend; used for Linux CI parity and as a portable
    fallback. Not the production path on Apple Silicon (that's MLX above).
    """

    def __init__(self, whisper_model: object, beam_size: int = 5) -> None:
        self.whisper_model = whisper_model
        self.beam_size = beam_size

    def warm(self) -> None:
        return None

    def decode(
        self,
        *,
        audio: np.ndarray,
        sample_rate_hz: int,
        language: str | None,
        allowed_languages: list[str],
        language_policy: str | None,
        initial_prompt: str | None,
        bias_phrases: list[str],
        context_payload: dict,
    ) -> WhisperDecodeOutput:
        started = time.perf_counter()
        segments_iter, info = self.whisper_model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=self.beam_size,
            best_of=1,
            vad_filter=False,
            condition_on_previous_text=True,
            word_timestamps=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        word_dicts: list[dict] = []
        segment_ends: list[float] = []
        text_parts: list[str] = []
        for seg in segments_iter:
            seg_text = getattr(seg, "text", "")
            if seg_text:
                text_parts.append(seg_text.strip())
            seg_end = getattr(seg, "end", None)
            if seg_end is not None:
                segment_ends.append(float(seg_end))
            for word in getattr(seg, "words", None) or []:
                word_dicts.append(
                    {
                        "word": getattr(word, "word", "").strip(),
                        "start": float(getattr(word, "start", 0.0)),
                        "end": float(getattr(word, "end", 0.0)),
                    }
                )
        text = " ".join(text_parts).strip()
        decode_ms = (time.perf_counter() - started) * 1000.0
        return WhisperDecodeOutput(
            text=text,
            language=getattr(info, "language", language),
            decode_ms=decode_ms,
            word_dicts=word_dicts,
            segment_ends=segment_ends,
            metadata={
                "faster_whisper_decode_ms": decode_ms,
                "prompt_bias_phrase_count": len(bias_phrases),
                "prompt_context_entity_count": _context_entity_count(context_payload),
                "requested_language": language,
                "allowed_languages": list(allowed_languages),
                "language_policy": language_policy,
                "segment_count": len(segment_ends),
                "word_count": len(word_dicts),
            },
        )


def _context_entity_count(context_payload: dict) -> int:
    candidate_entities = (
        context_payload.get("candidate_entities") if isinstance(context_payload, dict) else None
    )
    return len(candidate_entities) if isinstance(candidate_entities, list) else 0


def _preview_orthography_protected_terms(
    context_payload: dict[str, Any],
    bias_phrases: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if not isinstance(context_payload, dict):
        return list(bias_phrases or ())[:64]
    out: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            out.append(text)

    for key in ("candidate_entities", "recent_screen_terms", "preview_personalization_terms"):
        raw = context_payload.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw[:64]:
            if isinstance(item, dict):
                add(item.get("text") or item.get("canonical") or item.get("term"))
                for alias in item.get("aliases") or item.get("spoken_forms") or ():
                    add(alias)
            else:
                add(item)
    packet = context_payload.get("memory_serving_packet")
    if isinstance(packet, dict):
        for value in (packet.get("lexicon_terms") or [])[:64]:
            add(value)
        for row in (packet.get("corrections") or [])[:64]:
            if isinstance(row, dict):
                add(row.get("corrected"))
        for row in (packet.get("replacements") or [])[:64]:
            if isinstance(row, dict):
                add(row.get("replacement"))
    for phrase in bias_phrases or ():
        add(phrase)
    return out[:160]


# Maximum word count we'll let through in the unstable tail as a last resort.
# The shell coalesces chunks under preview backpressure, so a legitimate tail
# can exceed the old 14-word limit. Repetition / BoH checks carry the real
# hallucination defense; length alone fires only on absurd runaway output.
_TAIL_MAX_WORDS = 60

# When checking for repetitive-loop hallucination, look at this many words from
# the end of the tail. If those words contain a short n-gram repeated this many
# times, we treat the tail as a loop and suppress it.
_TAIL_LOOP_WINDOW = 30
_TAIL_LOOP_NGRAM_REPETITIONS = 3


_VALID_LEADING_SINGLE_LETTERS = frozenset({"a", "i", "o"})
_COMMON_PREVIEW_WORD_VARIANTS = {
    "oke": "Okay",
}


def _canonicalize_common_preview_word_variants(
    word_dicts: list[dict],
) -> tuple[list[dict], dict[str, object]]:
    """Normalize known ASR spelling variants before LocalAgreement.

    This is intentionally tiny and lexical. It is not personalization or fuzzy
    correction; it only maps invalid/common preview spellings to the canonical
    word so LocalAgreement can recognize later corrected decodes as overlap
    instead of committing both forms (for example ``Oke Okay``).
    """
    if not word_dicts:
        return list(word_dicts or []), {"preview_word_variant_repairs": []}
    repaired = [dict(w) for w in word_dicts]
    applied: list[dict[str, object]] = []
    for idx, word in enumerate(repaired):
        raw = str(word.get("word") or word.get("text") or "")
        if not raw:
            continue
        match = re.match(r"^(\W*)([A-Za-z]+)(\W*)$", raw.strip())
        if match is None:
            continue
        prefix, core, suffix = match.groups()
        replacement = _COMMON_PREVIEW_WORD_VARIANTS.get(core.casefold())
        if replacement is None:
            continue
        fixed = f"{prefix}{replacement}{suffix}"
        word["word"] = fixed
        word["text"] = fixed
        applied.append({"index": idx, "raw": raw, "replacement": fixed})
    return repaired, {"preview_word_variant_repairs": applied[:8]}



def _strip_leading_hallucinated_letter(
    text: str,
    word_dicts: list[dict],
) -> tuple[str, list[dict], str | None]:
    """Drop a phantom leading B-Z token before it can reach LocalAgreement."""
    if not word_dicts:
        return text, word_dicts, None
    first = str(word_dicts[0].get("word") or "").strip()
    core = first.strip(".,!?;:'\"-").casefold()
    if len(core) != 1 or not core.isalpha() or core in _VALID_LEADING_SINGLE_LETTERS:
        return text, word_dicts, None

    remaining = word_dicts[1:]
    next_norm = ""
    if remaining:
        next_norm = re.sub(r"[^a-z0-9']", "", str(remaining[0].get("word") or "").casefold())
    if _letter_context_allows(prev_norm="", next_norm=next_norm):
        return text, word_dicts, None

    new_text = re.sub(r"^\s*" + re.escape(first) + r"\b\s*", "", text or "", count=1).lstrip()
    if new_text == (text or ""):
        new_text = " ".join(str(w.get("word") or "").strip() for w in remaining).strip()
    return new_text, remaining, first


def _guard_tail_hallucination(tail_text: str) -> tuple[str, str | None]:
    """Suppress the unstable tail when it shows Whisper hallucination patterns.

    Returns ``(tail_or_empty, reason_or_None)``. ``reason`` is non-None only
    when the tail was suppressed; the caller uses it for telemetry.

    Two guards:

    1. **Length cap** — the tail is the hypothesis past the LocalAgreement-2
       agreement point. On a buffer that's been trimmed correctly, the tail
       should represent at most ~1-2 s of audio = ~10 words. Anything past
       ``_TAIL_MAX_WORDS`` is almost certainly hallucination.
    2. **Loop detector** — Whisper on padded silence with
       ``condition_on_previous_text=True`` produces n-gram loops ("the same
       as the same as the same…"). We scan the trailing window for any 2-4
       word n-gram that appears at least ``_TAIL_LOOP_NGRAM_REPETITIONS``
       times consecutively-ish, and if found, drop the whole tail.

    The committed text is NEVER touched by these guards — it has the
    two-pass agreement guarantee.
    """
    tail_text = (tail_text or "").strip()
    if not tail_text:
        return tail_text, None

    from juno_v2.preview.hallucination_blocklist import (
        is_whisper_silence_hallucination,
        strip_trailing_boh,
    )

    if is_whisper_silence_hallucination(tail_text):
        return "", "tail_boh_blocklist"
    _cleaned_tail, removed_suffix = strip_trailing_boh(tail_text)
    if removed_suffix is not None:
        # If a tail ends in a known signoff/subtitle hallucination, suppress the
        # unsafe signoff prefix ("I'm gonna" from "I'm gonna see you next
        # time") too. When the cleaned prefix has real words ("this works Thank
        # you"), keep that prefix internally and only strip the hallucinated
        # suffix.
        if _tail_prefix_is_unsafe_after_boh_strip(_cleaned_tail):
            return "", "tail_boh_suffix"
        return _cleaned_tail, "tail_boh_suffix"

    # Loop detector: take the last _TAIL_LOOP_WINDOW words (or all if shorter),
    # look for any n-gram of size 2..4 that occurs >= _TAIL_LOOP_NGRAM_REPETITIONS
    # times in that window.
    words = tail_text.split()
    window = [w.lower().strip(".,!?;:'\"") for w in words[-_TAIL_LOOP_WINDOW:]]
    window = [w for w in window if w]
    if len(window) < 4:
        return tail_text, None
    for n in (2, 3, 4):
        if len(window) < n * _TAIL_LOOP_NGRAM_REPETITIONS:
            continue
        seen: dict[tuple[str, ...], int] = {}
        for i in range(len(window) - n + 1):
            gram = tuple(window[i : i + n])
            # Skip n-grams that are all stopwords/articles — those repeat
            # legitimately in normal speech ("the", "of", "and").
            if all(w in {"the", "a", "an", "of", "and", "or", "to", "in", "on"} for w in gram):
                continue
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= _TAIL_LOOP_NGRAM_REPETITIONS:
                return "", f"tail_loop_n{n}"
    if len(words) > _TAIL_MAX_WORDS:
        return "", "tail_length_cap"
    return tail_text, None


_TAIL_LOW_SIGNAL_RMS = 0.006
_TAIL_LOW_SIGNAL_PEAK = 0.03
_TAIL_LOW_SIGNAL_MAX_WORDS = 8
_TAIL_RISKY_NO_SPEECH_PROB = 0.3
_TAIL_COMMIT_QUARANTINE_REASONS = {
    "tail_fragment_quarantine",
    "tail_warmup_quarantine",
    "tail_silence_decode_quarantine",
    "tail_post_speech_grace_quarantine",
    "tail_low_signal_quarantine",
    "tail_low_confidence_single_word_quarantine",
}
_MAX_LIVE_COMMIT_WORDS = 4
_MAX_SEGMENT_FINAL_TAIL_PROMOTION_WORDS = 0
_MAX_FINAL_TAIL_PROMOTION_WORDS = 12
_MAX_FINAL_EMPTY_COMMITTED_TAIL_PROMOTION_WORDS = 32
_SILENCE_ONLY_SINGLE_WORD_BLOCKLIST = frozenset(
    {
        "and",
        "hello",
        "next",
        "okay",
        "right",
        "so",
        "thanks",
        "the",
        "um",
        "uh",
        "yeah",
        "yep",
        "you",
    }
)


def _display_word_count(words: list[Word]) -> int:
    return sum(len(re.findall(r"[A-Za-z0-9]+", word.text or "")) for word in words if word.normalized)


def _display_budget_prefix_len(words: list[Word], max_words: int) -> int:
    total = 0
    for idx, word in enumerate(words):
        count = len(re.findall(r"[A-Za-z0-9]+", word.text or "")) if word.normalized else 0
        if idx > 0 and total + count > max_words:
            return idx
        total += count
    return len(words)


def _silence_agreement_commit_safe(
    words: list[Word],
    *,
    decoded: WhisperDecodeOutput | None,
    previous_tail_reason: str | None,
    decode_on_silence: bool,
) -> bool:
    """Allow a repeated silence tail to become committed only with bounded risk.

    The HUD still never renders untrusted tail text. This only covers the case
    where a word appeared as hidden tail, then appeared again in the next decode.
    Without this gate the final word can remain invisible until the user keeps
    talking; with a blanket rollback removal, silence guesses like "hello" can
    become opaque text. The criteria below keep the LocalAgreement evidence but
    reject known silence/corpus artifacts and risky single-token guesses.
    """
    if not words:
        return False
    reason = (previous_tail_reason or "").strip()
    if reason != "tail_silence_decode_quarantine" and not decode_on_silence:
        return False
    if _display_word_count(words) > _MAX_LIVE_COMMIT_WORDS:
        return False
    norms = [w.normalized for w in words if w.normalized]
    if not norms:
        return False
    if any(_is_unsupported_boundary_letter(norm) for norm in norms):
        return False
    if len(norms) == 1 and norms[0] in _SILENCE_ONLY_SINGLE_WORD_BLOCKLIST:
        return False
    text = " ".join(w.text for w in words).strip()
    if not text:
        return False
    guarded, reason = _guard_tail_hallucination(text)
    if reason is not None or guarded.strip() != text:
        return False
    try:
        nsp_raw = decoded.metadata.get("last_segment_no_speech_prob") if decoded is not None else None
        nsp = float(nsp_raw) if nsp_raw is not None else None
    except (TypeError, ValueError):
        nsp = None
    if nsp is not None and nsp >= _TAIL_RISKY_NO_SPEECH_PROB:
        return False
    return True


def _guard_tail_display(
    tail_text: str,
    *,
    request_audio_ms: float | None,
    no_speech_prob: float | None = None,
    vad_admit_reason: str | None = None,
    audio_rms: float | None = None,
    audio_peak: float | None = None,
    decode_on_silence: bool = False,
    is_final: bool = False,
) -> str | None:
    """Quarantine volatile tails that are unsafe to render immediately.

    Single-word, tiny-audio, and fragment tails are too weak to show as trusted
    live text. The caller keeps them in ``HypothesisBuffer.tail`` as unstable
    evidence. Audio-risk reasons also commit-quarantine exact
    no-continuation repeats so a hidden silence guess cannot become committed
    text on the next pass.

    ``no_speech_prob`` is intentionally not used as a confidence bypass. Live
    logs showed MLX Whisper can assign near-zero ``no_speech_prob`` to
    short-window hallucinations like "Hello.". A high value is still useful as
    negative evidence for commit quarantine.
    """
    text = (tail_text or "").strip()
    if not text:
        return None
    if re.search(r"[-–—]\s*$", text):
        return "tail_fragment_quarantine"
    words = text.split()
    reason = str(vad_admit_reason or "").strip().lower()
    try:
        nsp = float(no_speech_prob) if no_speech_prob is not None else None
    except (TypeError, ValueError):
        nsp = None
    try:
        rms = float(audio_rms) if audio_rms is not None else 0.0
    except (TypeError, ValueError):
        rms = 0.0
    try:
        peak = float(audio_peak) if audio_peak is not None else 0.0
    except (TypeError, ValueError):
        peak = 0.0
    if not is_final:
        if reason == "warmup":
            return "tail_warmup_quarantine"
        if decode_on_silence:
            return "tail_silence_decode_quarantine"
        if reason == "vad_post_speech_grace" and len(words) <= _TAIL_LOW_SIGNAL_MAX_WORDS:
            return "tail_post_speech_grace_quarantine"
        if (
            0.0 < peak < _TAIL_LOW_SIGNAL_PEAK
            and rms < _TAIL_LOW_SIGNAL_RMS
            and len(words) <= _TAIL_LOW_SIGNAL_MAX_WORDS
        ):
            return "tail_low_signal_quarantine"
    if len(words) <= 1:
        if nsp is not None and nsp >= _TAIL_RISKY_NO_SPEECH_PROB:
            return "tail_low_confidence_single_word_quarantine"
        return "tail_single_word_quarantine"
    try:
        req_ms = float(request_audio_ms) if request_audio_ms is not None else 0.0
    except (TypeError, ValueError):
        req_ms = 0.0
    if req_ms > 0 and req_ms < 500.0 and len(words) <= 3 and not is_final:
        return "tail_short_audio_quarantine"
    return None


def _tail_audio_span_ms(state: StreamingPreviewState) -> float:
    if not state.hypothesis.tail:
        return 0.0
    try:
        start = min(float(w.start) for w in state.hypothesis.tail)
        end = max(float(w.end) for w in state.hypothesis.tail)
    except Exception:
        return 0.0
    return max(0.0, (end - start) * 1000.0)


def _tail_prefix_is_unsafe_after_boh_strip(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s']", "", (text or "").casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in {
        "im gonna",
        "i'm gonna",
        "im going to",
        "i'm going to",
        "i am going to",
    }


def _strip_trailing_committed_boh(state: StreamingPreviewState) -> str | None:
    """Remove Tier-A Whisper signoff hallucinations from committed history.

    A display-only strip leaves the bad words inside
    ``HypothesisBuffer.committed``. The next decode prompt can then see the
    hallucination and lock into it. This mutates the source-of-truth committed
    list so the HUD, prompt, and LocalAgreement state stay aligned.
    """
    if not state.hypothesis.committed:
        return None
    from juno_v2.preview.hallucination_blocklist import strip_trailing_boh

    text = state.hypothesis.committed_text()
    cleaned, reason = strip_trailing_boh(text)
    if reason is None:
        return None
    keep_count = len(cleaned.split()) if cleaned.strip() else 0
    if keep_count < len(state.hypothesis.committed):
        state.hypothesis.committed = state.hypothesis.committed[:keep_count]
        if state.hypothesis.committed:
            state.hypothesis.last_committed_time = state.hypothesis.committed[-1].end
        else:
            state.hypothesis.last_committed_time = 0.0
    return reason


_UNTRUSTED_SINGLE_LETTERS = frozenset("bcdefghjklmnopqrstuvwxyz")
_LETTER_CONTEXT_WORDS = frozenset(
    {
        "button",
        "choice",
        "column",
        "command",
        "control",
        "drive",
        "grade",
        "hit",
        "hold",
        "key",
        "keys",
        "letter",
        "option",
        "part",
        "press",
        "row",
        "section",
        "suite",
        "type",
        "vitamin",
    }
)


def _strip_unsupported_boundary_letters(state: StreamingPreviewState) -> str | None:
    """Remove isolated B-Z letters only at committed segment boundaries.

    The retained user traces show MLX Whisper inserting unsupported edge letters
    ("D Hey", "better D") during quiet/pause-heavy preview windows. We keep the
    rule boundary-only and context-gated so real letter dictation such as
    "press D", "letter D", or "D D D" survives.
    """
    words = state.hypothesis.committed
    if not words:
        return None
    reasons: list[str] = []

    while words and _is_unsupported_boundary_letter(words[0].normalized):
        next_norm = words[1].normalized if len(words) > 1 else ""
        if _letter_context_allows(prev_norm="", next_norm=next_norm):
            break
        words = words[1:]
        reasons.append("leading_single_letter")

    while words and _is_unsupported_boundary_letter(words[-1].normalized):
        prev_norm = words[-2].normalized if len(words) > 1 else ""
        if _letter_context_allows(prev_norm=prev_norm, next_norm=""):
            break
        words = words[:-1]
        reasons.append("trailing_single_letter")

    if not reasons:
        return None
    state.hypothesis.committed = words
    if words:
        state.hypothesis.last_committed_time = words[-1].end
    else:
        state.hypothesis.last_committed_time = 0.0
    return ",".join(reasons)


def _is_unsupported_boundary_letter(norm: str) -> bool:
    return len(norm or "") == 1 and norm in _UNTRUSTED_SINGLE_LETTERS


def _letter_context_allows(*, prev_norm: str, next_norm: str) -> bool:
    if _is_unsupported_boundary_letter(prev_norm) or _is_unsupported_boundary_letter(next_norm):
        return True
    return prev_norm in _LETTER_CONTEXT_WORDS or next_norm in _LETTER_CONTEXT_WORDS
