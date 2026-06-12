"""One-shot dictation pipeline.

The Mac shell and compatible clients hit a single HTTP endpoint
(``POST /api/broker/dictation/ingest_wav`` or the OpenAI-compatible
``POST /v1/audio/transcriptions``) with a recorded WAV and expect back a
single transcript string to paste. That endpoint historically called
:class:`~juno_core_v3.dictation.transcriber.DictationTranscriber`
directly, which meant **every** competitive advantage Juno has --
context, recognition bias, personalization, writer normalization,
multilingual routing -- was silently skipped on the primary user path.

This module restores the full engine pipeline on the one-shot path. It
composes the same building blocks used by the streaming
:class:`~juno_v2.engine.session.DictationSessionRunner` but without a
VAD / utterance buffer, because the caller has already decided the
utterance boundaries (the record button). Every stage is optional: if a
component is not wired (e.g. no writer service, no context provider),
the pipeline degrades gracefully and still returns a transcript. This
means tests and the standalone workbench keep working while production
deployments get the full treatment automatically.

Lifecycle:

1. Capability gate (optional). Checks the frontmost app / focused field
   and short-circuits with ``error_code=capability_blocked`` if the
   surface is a secure input.
2. Context snapshot (optional). Captures selected text, focused text,
   app name, window title, clipboard ring. Stamped onto the ASR request
   as ``initial_prompt`` + ``bias_phrases`` via the bias engine.
3. Memory serving packet (optional). Lexicon terms, replacement rules,
   correction pairs, session entities feed the bias engine.
4. Language decision (optional). Script-sniff over the context to pick a
   language hint on ``auto_supported`` policy.
5. Bias plan. ``RecognitionBiasEngine.build_plan`` produces the
   ``initial_prompt`` + ``bias_phrases`` the ASR backend will honour.
6. ASR decode. Any
   :class:`~juno_core_v3.dictation.transcriber.DictationTranscriber`.
7. Memory normalization. Applies corrections, replacements, and lexicon
   canonicalization to the raw transcript.
8. Language-aware normalization. Script-specific punctuation / spacing.
9. Writer processing (optional). Handles voice commands like
   ``"bullet mode"``, ``"add correction foo to bar"``, etc. The output
   is what the surface should paste; state-mutation intents return an
   empty ``output_text`` and the surface should NOT paste anything.
10. Retain utterance record. The raw transcript + plan are cached keyed
    on ``utterance_id`` so that when the Mac shell later hits
    ``/api/broker/insertion/committed`` (or ``observe_correction``) we
    can call ``memory_store.record_correction`` and
    ``upsert_session_entities`` with real (raw, committed) pairs.

The pipeline is deliberately framework-free: no Flask / aiohttp / etc.
It's a plain function you pass a WAV byte string to.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache
import re
import difflib
import os
from datetime import datetime, timezone
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from juno_core_v3.context.clipboard_ring import ClipboardRingBuffer
    from juno_v2.context.compiler import CompiledContext
    from juno_v2.context.provider import ContextProvider
    from juno_v2.contracts.memory import MemoryServingPacket
    from juno_v2.contracts.modes import ModePolicy, ModeSelection
    from juno_v2.modes.store import CustomModeStore
    from juno_v2.transcript.contracts import TranscriptAdjudicationResult
    from juno_v2.writer.service import WriterService

from juno_v2.context.app_classifier import classify_app_category
from juno_v2.context.clipboard_enrichment import inject_clipboard_ring
from juno_v2.context.compiler import compile_context
from juno_v2.context.frozen_merge import merge_frozen_capability_into_bundle
from juno_v2.audio.diagnostics import analyze_wav_bytes
from juno_v2.contracts.context import RecognitionBiasPlan, TypedContextBundle
from juno_core_v3.actions.contracts import Action, ActionKind
from juno_core_v3.actions.pipeline_hook import detect_actions_for_pipeline
from juno_core_v3.actions.grammar import strip_wake
from juno_core_v3.context.plane import ContextPlane, ContextPlaneConfig
from juno_v2.contracts.memory import MemorySnapshot, TranscriptNormalization
from juno_v2.contracts.tracing import TraceKind
from juno_v2.contracts.workbench import ClientSelection, CommitMode
from juno_v2.contracts.writer import WriterActionKind, WriterOutcome
from juno_v2.itn.engine import ITNEngine
from juno_v2.itn.format_policy import resolve_itn_format_policy
from juno_v2.presets.surface_presets import (
    SurfacePresetStore,
    build_surface_context_line,
    default_surface_presets_path,
    merge_prompt_parts,
    resolve_mode_with_surface_presets,
)
from juno_v2.language.normalize import LanguageAwareNormalizer
from juno_v2.language.policy import LanguagePlanner
from juno_v2.memory.ai_dictionary import AI_GLOSSARY
from juno_v2.memory.bias import RecognitionBiasEngine
from juno_v2.memory.entity_policy import (
    common_english_single_word,
    commit_session_entity_allowed,
    term_present_in_text,
)
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.memory.hallucination import (
    looks_like_hallucination,
    looks_like_low_yield_garbage,
    looks_like_silence_hallucination,
    strip_leading_prompt_echo,
    strip_trailing_silence_hallucination,
)
from juno_v2.memory.repetition import collapse_tail_repetition
from juno_v2.memory.term_policy import learned_term_allowed
from juno_v2.personalization.seed.runtime import JunoSeedPersonalizationRuntime
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.transcript.adjudicator import TranscriptAdjudicator, TranscriptAdjudicatorConfig
from juno_v2.transcript.validators import validate_adjudication_result
from juno_v2.turn_plan import actions_from_turn_plan, validate_turn_plan

from juno_core_v3.dictation.self_corrections import (
    MID_UTTERANCE_EDIT_MARKER_RE,
    apply_unambiguous_retakes,
)
from juno_core_v3.dictation.transcriber import (
    DictationTranscriber,
    TranscribeResult,
    TranscribeUnavailable,
)


# ------------------------------------------------------------------ #
# Capability gate protocol
# ------------------------------------------------------------------ #

class CapabilityGate(Protocol):
    """Minimum surface of :class:`~juno_core_v3.context.capability_probe.CapabilityChecker`.

    We depend on the narrow protocol (rather than the concrete class) so
    tests can inject a fake that always allows / always blocks without
    needing a compiled Swift helper.
    """

    def decide(self, *, app_bundle_id: str | None, window_title: str | None) -> "CapabilityDecisionLike": ...


class CapabilityDecisionLike(Protocol):
    blocked: bool
    reason: str
    mode: str  # "allow" | "block" | "warn"


# ------------------------------------------------------------------ #
# Utterance record cache
# ------------------------------------------------------------------ #

@dataclass(slots=True)
class UtteranceRecord:
    """Retained across transcribe -> commit so we can learn corrections."""

    utterance_id: str
    raw_text: str
    normalized_text: str
    adjudicated_text: str
    writer_text: str
    plan: RecognitionBiasPlan | None
    context: TypedContextBundle | None
    literal_text: str = ""
    final_text: str = ""
    writer_action: str | None = None
    learn_from_commit: bool = True
    created_at: float = field(default_factory=time.time)
    # Set when the utterance was parsed as a Juno action ("hey juno take a
    # note…"). Memory learning must not observe action wrappers as
    # raw→committed correction pairs — wake phrases and command verbs would
    # contaminate the lexicon. See record_insertion below.
    is_action: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptEditResult:
    text: str
    changed: bool
    edits: tuple[dict[str, Any], ...] = ()


class UtteranceRecordCache:
    """Bounded, thread-safe cache of recent utterance records.

    Keyed on ``utterance_id``. Also tracks the most recent key so the
    insertion_committed endpoint can fall back to "whatever the user
    just dictated" when the surface forgot to echo the id.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, UtteranceRecord]" = OrderedDict()
        self._max = max(1, int(max_entries))

    def put(self, record: UtteranceRecord) -> None:
        with self._lock:
            self._entries[record.utterance_id] = record
            self._entries.move_to_end(record.utterance_id)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def get(self, utterance_id: str) -> UtteranceRecord | None:
        with self._lock:
            return self._entries.get(utterance_id)

    def pop(self, utterance_id: str) -> UtteranceRecord | None:
        with self._lock:
            return self._entries.pop(utterance_id, None)

    def last(self) -> UtteranceRecord | None:
        with self._lock:
            if not self._entries:
                return None
            key = next(reversed(self._entries))
            return self._entries[key]


# ------------------------------------------------------------------ #
# Pipeline result
# ------------------------------------------------------------------ #

@dataclass(slots=True)
class OneShotDictationResult:
    """Wire-level result returned to the HTTP layer.

    The HTTP layer turns this into JSON. ``transcript`` is the text the
    surface should paste (empty string for state-mutation voice commands
    so the surface shows a toast instead of typing). All other fields
    are observability metadata — surfaces are free to ignore them.
    """

    utterance_id: str
    ok: bool
    transcript: str
    raw_transcript: str
    backend_name: str
    audio_duration_ms: float
    decode_ms: float
    language: str | None = None
    # Provenance: absolute path / repo id of the ASR checkpoint that
    # produced this transcript. Lets the Mac history view attribute a
    # given paste to the exact model without re-scanning trace logs.
    model_path: str = ""
    writer_action: str | None = None
    writer_deterministic: bool = False
    memory_updated: bool = False
    normalization_applied: list[dict[str, Any]] = field(default_factory=list)
    bias_phrase_count: int = 0
    context_present: bool = False
    capability_mode: str | None = None
    error: str | None = None
    error_code: str | None = None
    # How the macOS shell should paste: ``insert`` (caret), ``replace``
    # (selection still highlighted), ``none`` (state mutation / noop /
    # ambiguous — do not synthesize Cmd+V).
    paste_kind: str = "insert"
    noop_reason: str | None = None
    # Rejected action turns suppress the paste, but the spoken words must
    # remain recoverable by the shell (copyable) — never silently erased.
    recoverable_transcript: str = ""
    degraded_writer: bool = False
    frozen_context_merged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    transcript_stage: str = "final_delivery"
    # Parsed Juno actions (notes/reminders). ``None`` for the overwhelming
    # majority of utterances — only set when the wake-word + verb parser
    # detects a hit. The macOS shell consumes this list to dispatch sinks
    # (EKEventStore for reminders, AppleScript for notes) and posts results
    # back via ``POST /api/broker/history/{utterance_id}/actions``.
    actions: list[dict[str, Any]] | None = None
    # Explicit "this utterance was parsed as an action" flag. Previously the
    # shell had to infer this from ``paste_kind == "none"`` plus the presence
    # of ``actions``, which is implicit and breaks when an utterance is a
    # noop / state-mutation that is *not* an action (paste_kind="none" with
    # actions=None). Setting this field unambiguously tells the shell to
    # suppress paste because of an action, not because of some other reason.
    is_action: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "utterance_id": self.utterance_id,
            "transcript": self.transcript,
            "raw_transcript": self.raw_transcript,
            "backend_name": self.backend_name,
            "model_path": self.model_path,
            "audio_duration_ms": self.audio_duration_ms,
            "decode_ms": self.decode_ms,
            "language": self.language,
            "writer_action": self.writer_action,
            "writer_deterministic": self.writer_deterministic,
            "memory_updated": self.memory_updated,
            "normalization_applied": list(self.normalization_applied),
            "bias_phrase_count": self.bias_phrase_count,
            "context_present": self.context_present,
            "capability_mode": self.capability_mode,
            "paste_kind": self.paste_kind,
            "noop_reason": self.noop_reason,
            "recoverable_transcript": self.recoverable_transcript,
            "degraded_writer": self.degraded_writer,
            "frozen_context_merged": self.frozen_context_merged,
            "metadata": dict(self.metadata),
            "stage": self.transcript_stage,
            "transcript_stage": self.transcript_stage,
            "is_action": self.is_action,
        }
        # Only include the actions field when present; absent on ~99% of
        # utterances, which keeps the wire payload byte-identical to today
        # for non-action use and makes downstream consumers' "does this
        # response have actions" check trivially `actions in payload`.
        if self.actions:
            payload["actions"] = [dict(a) for a in self.actions]
        if not self.ok:
            payload["error"] = self.error
            payload["error_code"] = self.error_code
        return payload


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #

@dataclass
class OneShotDictationPipeline:
    """Compose the full Juno stack into a single
    wav-bytes-in / transcript-out call.

    Only ``transcriber`` is required; everything else is optional. The
    pipeline degrades gracefully — missing components are skipped with
    a structured trace event so ops can see exactly which stage ran.
    """

    transcriber: DictationTranscriber
    recorder: TraceRecorder
    context_provider: ContextProvider | None = None
    memory_store: JsonMemoryStore | None = None
    bias_engine: RecognitionBiasEngine = field(default_factory=RecognitionBiasEngine)
    writer_service: WriterService | None = None
    language_planner: LanguagePlanner | None = None
    transcript_normalizer: LanguageAwareNormalizer = field(
        default_factory=LanguageAwareNormalizer
    )
    capability_gate: CapabilityGate | None = None
    clipboard_ring: ClipboardRingBuffer | None = None
    itn_engine: ITNEngine = field(default_factory=ITNEngine)
    writer_enabled: bool = True
    itn_enabled: bool = True
    context_plane: ContextPlane = field(default_factory=lambda: ContextPlane(ContextPlaneConfig()))
    records: UtteranceRecordCache = field(default_factory=UtteranceRecordCache)
    # Bounded audio retention for replay/rerun. When set, successful
    # utterances are saved under ``audio_save_dir/YYYY/MM/DD/{id}.wav``.
    # Only the most recent ``audio_retention_limit`` files are kept
    # tree-wide; ``audio_retention_days`` > 0 also drops WAVs older than
    # that many days (mtime). ``audio_retention_days == 0`` skips the
    # time-based prune (used when broker policy is ``forever`` / ``off``).
    audio_save_dir: Path | None = None
    audio_retention_limit: int = 20
    audio_retention_days: int = 30
    utterance_id_factory: Callable[[], str] = field(
        default_factory=lambda: _default_utterance_id_factory
    )
    custom_mode_store: CustomModeStore | None = None
    surface_preset_store: SurfacePresetStore | None = None
    juno_seed_runtime: JunoSeedPersonalizationRuntime | None = None
    transcript_adjudicator: TranscriptAdjudicator | None = None
    live_transcript_adjudicator: TranscriptAdjudicator | None = None
    transcript_adjudicator_config: TranscriptAdjudicatorConfig = field(
        default_factory=TranscriptAdjudicatorConfig
    )

    def run(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        app_bundle_id: str | None = None,
        window_title_hint: str | None = None,
        utterance_id: str | None = None,
        manual_writer_mode: str | None = None,
        custom_writer_mode: str | None = None,
        frozen_context: dict[str, Any] | None = None,
        degraded_writer_lane: bool = False,
        save_history: bool = True,
        save_audio: bool = True,
        transcript_stage: str = "final_delivery",
        session_context_tape: dict[str, Any] | list[Any] | None = None,
        transcript_hint: str | None = None,
        language_mode: str | None = None,
    ) -> OneShotDictationResult:
        started_ns = time.perf_counter_ns()
        uid = utterance_id or self._new_utterance_id()
        frozen_context_merged = False
        stage = _normalize_transcript_stage(transcript_stage)
        live_adjudication = stage == "live_adjudication"
        live_transcript_hint = _usable_transcript_hint_fallback(transcript_hint)
        use_live_transcript_for_final = False
        capability_mode: str | None = None
        if live_adjudication:
            save_history = False
            save_audio = False

        audio_diag = None
        if not wav_bytes:
            if save_history:
                from juno_v2.observability.history_store import append_history_record

                append_history_record(
                    self.recorder.log_dir,
                    {
                        "utterance_id": uid,
                        "ts_unix_ms": int(time.time() * 1000),
                        "transcript": "",
                        "raw_transcript": "",
                        "mode": "unknown",
                        "final_backend": str(getattr(self.transcriber, "backend_name", "unknown")),
                        "model_path": "",
                        "context": {},
                        "failure_reason": "empty_audio",
                        "session_class": "insert",
                        "processing_ms": 0,
                        "words": 0,
                        "replay_available": False,
                    },
                )
            return OneShotDictationResult(
                utterance_id=uid,
                ok=False,
                transcript="",
                raw_transcript="",
                backend_name=getattr(self.transcriber, "backend_name", "unknown"),
                audio_duration_ms=0.0,
                decode_ms=0.0,
                error="empty_audio",
                error_code="empty_audio",
                paste_kind="none",
                noop_reason="empty_audio",
                degraded_writer=degraded_writer_lane,
                frozen_context_merged=False,
                transcript_stage=stage,
            )

        if not live_adjudication:
            audio_diag = analyze_wav_bytes(wav_bytes)
            if audio_diag is not None and audio_diag.low_signal and not use_live_transcript_for_final:
                diag_payload = audio_diag.to_dict()
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_low_signal_audio_rejected",
                    {
                        "utterance_id": uid,
                        "audio_diagnostics": diag_payload,
                    },
                )
                if save_history:
                    from juno_v2.observability.history_store import append_history_record

                    append_history_record(
                        self.recorder.log_dir,
                        {
                            "utterance_id": uid,
                            "ts_unix_ms": int(time.time() * 1000),
                            "transcript": "",
                            "raw_transcript": "",
                            "mode": "unknown",
                            "final_backend": str(getattr(self.transcriber, "backend_name", "unknown")),
                            "model_path": "",
                            "context": {"app_bundle_id": app_bundle_id, "window_title": window_title_hint},
                            "failure_reason": "low_signal_audio",
                            "session_class": "insert",
                            "processing_ms": int((time.perf_counter_ns() - started_ns) / 1_000_000.0),
                            "words": 0,
                            "replay_available": False,
                        },
                    )
                return OneShotDictationResult(
                    utterance_id=uid,
                    ok=False,
                    transcript="",
                    raw_transcript="",
                    backend_name=getattr(self.transcriber, "backend_name", "unknown"),
                    audio_duration_ms=audio_diag.duration_ms,
                    decode_ms=0.0,
                    language=language,
                    capability_mode=capability_mode,
                    error="low_signal_audio",
                    error_code="low_signal_audio",
                    paste_kind="none",
                    noop_reason="low_signal_audio",
                    degraded_writer=degraded_writer_lane,
                    frozen_context_merged=False,
                    metadata={"audio_diagnostics": diag_payload},
                    transcript_stage=stage,
                )

        # 1. Capability gate -------------------------------------------------
        if self.capability_gate is not None:
            try:
                decision = self.capability_gate.decide(
                    app_bundle_id=app_bundle_id,
                    window_title=window_title_hint,
                )
                capability_mode = getattr(decision, "mode", None)
                if getattr(decision, "blocked", False):
                    self.recorder.record(
                        TraceKind.SYSTEM,
                        "oneshot_capability_blocked",
                        {
                            "utterance_id": uid,
                            "reason": getattr(decision, "reason", "blocked"),
                            "mode": capability_mode,
                            "app_bundle_id": app_bundle_id,
                        },
                    )
                    if save_history:
                        from juno_v2.observability.history_store import append_history_record

                        append_history_record(
                            self.recorder.log_dir,
                            {
                                "utterance_id": uid,
                                "ts_unix_ms": int(time.time() * 1000),
                                "transcript": "",
                                "raw_transcript": "",
                                "mode": "unknown",
                                "final_backend": str(getattr(self.transcriber, "backend_name", "unknown")),
                                "model_path": "",
                                "context": {"app_bundle_id": app_bundle_id, "window_title": window_title_hint},
                                "failure_reason": "capability_blocked",
                                "session_class": "insert",
                                "processing_ms": int((time.perf_counter_ns() - started_ns) / 1_000_000.0),
                                "words": 0,
                                "replay_available": False,
                            },
                        )
                    return OneShotDictationResult(
                        utterance_id=uid,
                        ok=False,
                        transcript="",
                        raw_transcript="",
                        backend_name=getattr(self.transcriber, "backend_name", "unknown"),
                        audio_duration_ms=0.0,
                        decode_ms=0.0,
                        capability_mode=capability_mode,
                        error=getattr(decision, "reason", "capability_blocked"),
                        error_code="capability_blocked",
                        paste_kind="none",
                        noop_reason="capability_blocked",
                        degraded_writer=degraded_writer_lane,
                        frozen_context_merged=False,
                        transcript_stage=stage,
                    )
            except Exception as exc:  # noqa: BLE001 — gate is best-effort
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_capability_error",
                    {"utterance_id": uid, "error": str(exc)},
                )

        # 2. Context snapshot -----------------------------------------------
        context: TypedContextBundle = self._snapshot_context()
        # The Mac shell knows which app was frontmost at hotkey press;
        # that view is more authoritative than whatever the context
        # provider probed a few ms later (the shell's HUD animation can
        # steal focus for a frame). Merge the hints in without
        # overwriting anything the provider already found.
        self._apply_surface_hints(
            context,
            app_bundle_id=app_bundle_id,
            window_title_hint=window_title_hint,
        )

        if frozen_context:
            frozen_context_merged = merge_frozen_capability_into_bundle(context, frozen_context)
            if frozen_context_merged:
                self.recorder.record(
                    TraceKind.CONTEXT,
                    "oneshot_frozen_context_merged",
                    {
                        "utterance_id": uid,
                        "frozen_keys": sorted(frozen_context.keys()),
                        "selection_chars": len(context.selected_text or ""),
                    },
                )

        tape_meta = self._apply_session_context_tape(
            context,
            session_context_tape,
            transcript_hint=transcript_hint,
        )

        # 2b. Context plane — canonical context packet -----------------------
        # Run the raw bundle through ContextPlane so budgets, suppression,
        # and degradation rules are applied and recorded.  Back-apply the
        # (potentially truncated) field values so downstream stages read
        # the plane-authoritative text rather than the raw snapshot.
        ctx_packet = self.context_plane.build_from_typed_bundle(
            context,
            surface_id=app_bundle_id or "workbench_dev",
        )
        suppression = ctx_packet.metadata.get("suppression", "none")
        degradation = ctx_packet.metadata.get("degradation", "none")
        # Reflect truncated/suppressed field values back onto context.
        context.selected_text = ctx_packet.selected_text
        context.focused_text_before = ctx_packet.focused_text_before
        context.focused_text_after = ctx_packet.focused_text_after
        context.clipboard_text = ctx_packet.clipboard_text
        context.field_text_excerpt = ctx_packet.field_text_excerpt
        context.app_name = ctx_packet.app_name
        context.window_title = ctx_packet.window_title
        self.recorder.record(
            TraceKind.CONTEXT,
            "oneshot_context_plane_applied",
            {
                "utterance_id": uid,
                "suppression": suppression,
                "degradation": degradation,
                "truncation_applied": dict(ctx_packet.truncation_applied),
                "provenance": {k: v.value if hasattr(v, "value") else v
                               for k, v in ctx_packet.provenance.items()},
                "budget_exceeded": ctx_packet.metadata.get("budget_exceeded", False),
            },
        )

        seed_attachment = None
        if self.juno_seed_runtime is not None:
            seed_attachment = self.juno_seed_runtime.build_seed_attachment(
                snapshot=self.memory_store.snapshot()
                if self.memory_store is not None
                else MemorySnapshot(schema_version=1),
                context=context,
                context_plane_suppression=suppression if isinstance(suppression, str) else None,
            )

        # 3. Memory serving packet ------------------------------------------
        memory_snapshot: MemorySnapshot = (
            self.memory_store.snapshot() if self.memory_store is not None
            else MemorySnapshot(schema_version=1)
        )

        # 4. Language decision ----------------------------------------------
        requested_language = language
        language_decision = None
        if self.language_planner is not None:
            language_decision = self.language_planner.plan_utterance(
                utterance_id=uid,
                context=context,
                configured_preview_language=language,
                configured_final_language=language,
            )
            requested_language = language_decision.request_language or language
            language_policy = language_decision.policy_name
        else:
            language_policy = "auto" if language is None else "fixed"

        # 4b. Mode + compiled context packet ---------------------------------
        custom_rec = None
        if self.custom_mode_store is not None and custom_writer_mode:
            custom_rec = self.custom_mode_store.get(custom_writer_mode)
        preset_store = self.surface_preset_store or SurfacePresetStore(
            default_surface_presets_path(self.recorder.log_dir)
        )
        mode_store = self.custom_mode_store
        br = (context.metadata or {}).get("app_bundle_id")
        bundle_id = br.strip() if isinstance(br, str) and br.strip() else None
        mode_sel, mode_pol, active_preset = resolve_mode_with_surface_presets(
            manual_mode_name=manual_writer_mode,
            custom_mode_name=custom_writer_mode,
            custom_record=custom_rec,
            surface_hint=context.app_category,
            surface_bundle_id=bundle_id,
            preset_store=preset_store,
            custom_mode_store=mode_store,
        )
        compiled_context = compile_context(
            utterance_id=uid,
            context=context,
            memory_snapshot=memory_snapshot,
            mode_selection=mode_sel,
            mode_policy=mode_pol,
            transcript_hint=transcript_hint,
            session_terms=tape_meta.get("candidate_terms") if isinstance(tape_meta, dict) else None,
            language=requested_language,
            stage=stage,
            seed_attachment=seed_attachment,
        )
        memory_packet = compiled_context.memory_packet
        asr_bias_packet = compiled_context.asr_bias_packet(max_prompt_chars=self.bias_engine.max_prompt_chars)
        self.recorder.record(
            TraceKind.CONTEXT,
            "context_compiler_built",
            {
                "utterance_id": uid,
                "stage": stage,
                "term_count": len(compiled_context.terms),
                "mode": mode_sel.effective_mode,
                "language": requested_language,
            },
        )
        self.recorder.record(
            TraceKind.CONTEXT,
            "context_compiler_asr_packet",
            {
                "utterance_id": uid,
                "bias_phrase_count": len(asr_bias_packet.bias_phrases),
                "max_prompt_chars": asr_bias_packet.max_prompt_chars,
                "initial_prompt_chars": len(asr_bias_packet.initial_prompt or ""),
            },
        )

        lang_initial = language_decision.initial_prompt if language_decision is not None else None
        include_title = bool(active_preset and active_preset.include_window_title_in_asr)
        surf_line = build_surface_context_line(
            app_name=context.app_name,
            window_title=context.window_title,
            app_category=context.app_category,
            include_window_title=include_title,
        )
        preset_addon = (active_preset.asr_addon or "").strip() if active_preset else None
        base_prompt = merge_prompt_parts(
            lang_initial,
            (mode_pol.prompt_prefix or "").strip() or None,
            preset_addon,
            surf_line,
            max_chars=self.bias_engine.max_prompt_chars,
        )

        # 5. Bias plan -------------------------------------------------------
        plan = self.bias_engine.build_plan(
            utterance_id=uid,
            snapshot=memory_snapshot,
            context=context,
            base_prompt=base_prompt,
            memory_packet=memory_packet,
            mode_policy=mode_pol,
            effective_mode=mode_sel.effective_mode,
            seed_attachment=seed_attachment,
        )
        plan.metadata["mode_selection"] = mode_sel.to_dict()
        plan.metadata["mode_policy_snapshot"] = mode_pol.to_dict()
        if language_decision is not None:
            plan.metadata["language_decision"] = language_decision.to_dict()
        if active_preset is not None:
            plan.metadata["surface_preset"] = {
                "id": active_preset.id,
                "bundle_id": active_preset.bundle_id,
                "asr_addon_len": len(active_preset.asr_addon or ""),
            }
            if (active_preset.writer_tone_addon or "").strip():
                plan.metadata["surface_preset_writer_tone"] = active_preset.writer_tone_addon.strip()

        self.recorder.record(
            TraceKind.CONTEXT,
            "oneshot_plan_built",
            {
                "utterance_id": uid,
                "bias_phrase_count": len(plan.bias_phrases),
                "has_prompt": bool(plan.initial_prompt),
                "language": requested_language,
                "language_policy": language_policy,
                "app_name": context.app_name,
                "window_title": context.window_title,
                "selection_chars": len(context.selected_text),
                "clipboard_chars": len(context.clipboard_text),
                "surface_preset_id": active_preset.id if active_preset else None,
                "mode_source": mode_sel.mode_source.value,
            },
        )
        self.recorder.record(
            TraceKind.SYSTEM,
            "oneshot_final_transcription_inputs",
            {
                "utterance_id": uid,
                "transcriber_backend": str(getattr(self.transcriber, "backend_name", "unknown")),
                "initial_prompt": plan.initial_prompt,
                "bias_phrases": list(plan.bias_phrases),
                "memory_snapshot_counts": _memory_snapshot_counts(memory_snapshot),
                "context": _context_debug_payload(context),
            },
        )

        # 6. ASR decode ------------------------------------------------------
        # Live transcript snapshots are useful context, but final_delivery must
        # be backed by final ASR. The hint shortcut is reserved for explicit
        # text-only live adjudication and hidden opt-in experiments.
        skip_whisper_for_live = bool(
            live_adjudication and transcript_hint and transcript_hint.strip()
        )
        try:
            if use_live_transcript_for_final:
                result = TranscribeResult(
                    transcript=live_transcript_hint,
                    language=requested_language,
                    backend_name="live_transcript_hint_final",
                    audio_duration_ms=audio_diag.duration_ms if audio_diag is not None else 0.0,
                    decode_ms=0.0,
                    model_path="live_transcript_hint_final",
                )
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_final_asr_skipped_for_live_transcript",
                    {
                        "utterance_id": uid,
                        "hint_chars": len(live_transcript_hint),
                        "audio_duration_ms": result.audio_duration_ms,
                    },
                )
            elif skip_whisper_for_live:
                result = TranscribeResult(
                    transcript=transcript_hint,
                    language=requested_language,
                    backend_name="live_preview_hint",
                    audio_duration_ms=0.0,
                    decode_ms=0.0,
                    model_path="live_preview_hint",
                )
            else:
                result = self.transcriber.transcribe_wav(
                    wav_bytes,
                    language=requested_language,
                    language_policy=language_policy,
                    initial_prompt=plan.initial_prompt,
                    bias_phrases=list(plan.bias_phrases),
                )
        except TranscribeUnavailable as exc:
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_transcribe_unavailable",
                {"utterance_id": uid, "code": exc.code, "reason": str(exc)},
            )
            if save_history:
                from juno_v2.observability.history_store import append_history_record

                append_history_record(
                    self.recorder.log_dir,
                    {
                        "utterance_id": uid,
                        "ts_unix_ms": int(time.time() * 1000),
                        "transcript": "",
                        "raw_transcript": "",
                        "mode": "unknown",
                        "final_backend": str(getattr(self.transcriber, "backend_name", "unknown")),
                        "model_path": "",
                        "context": {"app_bundle_id": app_bundle_id, "window_title": window_title_hint},
                        "failure_reason": str(exc.code or "transcribe_unavailable"),
                        "session_class": "insert",
                        "processing_ms": int((time.perf_counter_ns() - started_ns) / 1_000_000.0),
                        "words": 0,
                        "replay_available": False,
                    },
                )
            return OneShotDictationResult(
                utterance_id=uid,
                ok=False,
                transcript="",
                raw_transcript="",
                backend_name=getattr(self.transcriber, "backend_name", "unknown"),
                audio_duration_ms=0.0,
                decode_ms=0.0,
                language=requested_language,
                capability_mode=capability_mode,
                error=str(exc),
                error_code=exc.code,
                paste_kind="none",
                noop_reason=exc.code,
                degraded_writer=degraded_writer_lane,
                frozen_context_merged=frozen_context_merged,
                transcript_stage=stage,
            )
        except Exception as exc:  # noqa: BLE001 — surface never sees stacktraces
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_transcribe_error",
                {"utterance_id": uid, "error": str(exc), "type": type(exc).__name__},
            )
            if save_history:
                from juno_v2.observability.history_store import append_history_record

                append_history_record(
                    self.recorder.log_dir,
                    {
                        "utterance_id": uid,
                        "ts_unix_ms": int(time.time() * 1000),
                        "transcript": "",
                        "raw_transcript": "",
                        "mode": "unknown",
                        "final_backend": str(getattr(self.transcriber, "backend_name", "unknown")),
                        "model_path": "",
                        "context": {"app_bundle_id": app_bundle_id, "window_title": window_title_hint},
                        "failure_reason": "transcribe_failed",
                        "session_class": "insert",
                        "processing_ms": int((time.perf_counter_ns() - started_ns) / 1_000_000.0),
                        "words": 0,
                        "replay_available": False,
                    },
                )
            return OneShotDictationResult(
                utterance_id=uid,
                ok=False,
                transcript="",
                raw_transcript="",
                backend_name=getattr(self.transcriber, "backend_name", "unknown"),
                audio_duration_ms=0.0,
                decode_ms=0.0,
                language=requested_language,
                capability_mode=capability_mode,
                error=f"transcribe_failed: {exc}",
                error_code="transcribe_failed",
                paste_kind="none",
                noop_reason="transcribe_failed",
                degraded_writer=degraded_writer_lane,
                frozen_context_merged=frozen_context_merged,
                transcript_stage=stage,
            )

        raw_text = (result.transcript or "").strip()
        stripped_tail_text = raw_text
        if not live_adjudication and raw_text:
            stripped_tail_text = strip_trailing_silence_hallucination(
                raw_text,
                segments=getattr(result, "segments", ()),
                audio_duration_ms=result.audio_duration_ms,
            ).strip()
            if stripped_tail_text and stripped_tail_text != raw_text:
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_trailing_silence_hallucination_stripped",
                    {
                        "utterance_id": uid,
                        "raw_text_chars": len(raw_text),
                        "cleaned_text_chars": len(stripped_tail_text),
                        "backend": result.backend_name,
                    },
                )
                raw_text = stripped_tail_text
            stock_tail_text, stock_tail_phrase = _strip_unconfirmed_stock_tail(
                raw_text,
                transcript_hint=transcript_hint,
                audio_duration_ms=result.audio_duration_ms,
            )
            if stock_tail_phrase is not None and stock_tail_text != raw_text:
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_unconfirmed_stock_tail_stripped",
                    {
                        "utterance_id": uid,
                        "phrase": stock_tail_phrase,
                        "raw_text_chars": len(raw_text),
                        "cleaned_text_chars": len(stock_tail_text),
                        "backend": result.backend_name,
                    },
                )
                raw_text = stock_tail_text
            repetition_text, repetition_diag = collapse_tail_repetition(
                raw_text,
                audio_duration_ms=result.audio_duration_ms,
            )
            if repetition_diag.collapsed and repetition_text != raw_text:
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_final_repetition_collapsed",
                    {
                        "utterance_id": uid,
                        **repetition_diag.to_dict(),
                        "raw_text_chars": len(raw_text),
                        "cleaned_text_chars": len(repetition_text),
                        "backend": result.backend_name,
                    },
                )
                raw_text = repetition_text
        if not live_adjudication and raw_text:
            deleaked_text, echo_removed = strip_leading_prompt_echo(raw_text)
            if echo_removed is not None and deleaked_text != raw_text:
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_leading_prompt_echo_stripped",
                    {
                        "utterance_id": uid,
                        "removed_prefix": echo_removed[:80],
                        "raw_text_chars": len(raw_text),
                        "cleaned_text_chars": len(deleaked_text),
                        "backend": result.backend_name,
                    },
                )
                raw_text = deleaked_text
        self.recorder.record(
            TraceKind.SYSTEM,
            "oneshot_final_transcription_result",
            {
                "utterance_id": uid,
                "backend": result.backend_name,
                "model_path": str(getattr(result, "model_path", "") or ""),
                "language": result.language,
                "audio_duration_ms": result.audio_duration_ms,
                "decode_ms": result.decode_ms,
                "raw_text": _text_debug_payload(raw_text),
            },
        )
        final_asr_hallucination_fallback: dict[str, Any] | None = None

        # 6b. Hallucination guard ------------------------------------------
        #
        # mlx_whisper occasionally emits long autoregressive loops on
        # silence or low-information audio ("nuclear nuclear ..." × 250,
        # "to to to ..." × N, "thank you thank you ..."). The streaming
        # ``DictationSessionRunner`` runs ``looks_like_hallucination`` at
        # commit time (``juno_v2/commit/controller.py:219``) but the
        # one-shot path that the live macOS shell hits used to skip that
        # guard, so the loop reached insertion AND poisoned the memory
        # store via ``observe_transcript_for_context_entities`` and the
        # downstream correction-learning step. Both effects are
        # destructive: a single bad utterance can pollute the bias
        # prompt for every subsequent session, which makes whisper more
        # likely to hallucinate the same loop next time. Catching here
        # — before any memory write — breaks that feedback cycle.
        confidence_hint = getattr(result, "avg_logprob", None)
        if not isinstance(confidence_hint, (int, float)):
            confidence_hint = None
        # Silence-phrase guard: catches whisper-on-silence one-shots that the
        # structural hallucination guard above doesn't. The streaming path in
        # ``juno_v2/commit/controller.py`` runs this check; OneShot used to
        # skip it, so a 6-second silent dictation could pass "!" or
        # ".../okay/thank you" through to history and paste. The punctuation-
        # only short-circuit catches the "!" class unconditionally because a
        # user cannot intentionally dictate just punctuation; stock phrases
        # like "thank you" need audio-side corroboration, which OneShot only
        # has via avg_logprob and audio_duration_ms (no_speech_prob isn't on
        # TranscribeResult — extend the dataclass if we later want broader
        # silence-phrase coverage here).
        if not live_adjudication and raw_text and looks_like_silence_hallucination(
            raw_text,
            avg_logprob=confidence_hint,
            audio_duration_ms=result.audio_duration_ms,
        ):
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_silence_hallucination_rejected",
                {
                    "utterance_id": uid,
                    "raw_text": raw_text,
                    "audio_duration_ms": result.audio_duration_ms,
                    "avg_logprob": confidence_hint,
                    "backend": result.backend_name,
                },
            )
            return OneShotDictationResult(
                utterance_id=uid,
                ok=False,
                transcript="",
                raw_transcript=raw_text,
                backend_name=result.backend_name,
                audio_duration_ms=result.audio_duration_ms,
                decode_ms=result.decode_ms,
                language=result.language or requested_language,
                capability_mode=capability_mode,
                error="silence_hallucination",
                error_code="silence_hallucination",
                paste_kind="none",
                noop_reason="silence_hallucination",
                degraded_writer=degraded_writer_lane,
                frozen_context_merged=frozen_context_merged,
                transcript_stage=stage,
            )
        if not live_adjudication and raw_text and (
            looks_like_hallucination(raw_text, confidence=confidence_hint)
            or looks_like_low_yield_garbage(
                raw_text,
                confidence=confidence_hint,
                audio_duration_ms=result.audio_duration_ms,
            )
        ):
            fallback_hint = _usable_transcript_hint_fallback(transcript_hint)
            if fallback_hint and not live_adjudication:
                final_asr_hallucination_fallback = {
                    "reason": "hallucinated_transcript",
                    "rejected_backend": result.backend_name,
                    "rejected_model_path": str(getattr(result, "model_path", "") or ""),
                    "rejected_raw_chars": len(raw_text),
                    "rejected_raw_preview": raw_text[:120],
                    "fallback_source": "transcript_hint",
                    "fallback_chars": len(fallback_hint),
                }
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_hallucination_fallback_to_live_hint",
                    {
                        "utterance_id": uid,
                        "raw_text_chars": len(raw_text),
                        "raw_text_preview": raw_text[:120],
                        "confidence_hint": confidence_hint,
                        "backend": result.backend_name,
                        "audio_duration_ms": result.audio_duration_ms,
                        "fallback_chars": len(fallback_hint),
                    },
                )
                result = TranscribeResult(
                    transcript=fallback_hint,
                    language=result.language or requested_language,
                    backend_name="live_transcript_hint_fallback",
                    audio_duration_ms=result.audio_duration_ms,
                    decode_ms=result.decode_ms,
                    model_path=str(getattr(result, "model_path", "") or ""),
                )
                raw_text = fallback_hint
            else:
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_hallucination_rejected",
                    {
                        "utterance_id": uid,
                        "raw_text_chars": len(raw_text),
                        "raw_text_preview": raw_text[:120],
                        "confidence_hint": confidence_hint,
                        "backend": result.backend_name,
                        "audio_duration_ms": result.audio_duration_ms,
                    },
                )
                return OneShotDictationResult(
                    utterance_id=uid,
                    ok=False,
                    transcript="",
                    raw_transcript=raw_text,
                    backend_name=result.backend_name,
                    audio_duration_ms=result.audio_duration_ms,
                    decode_ms=result.decode_ms,
                    language=result.language or requested_language,
                    capability_mode=capability_mode,
                    error="hallucinated_transcript",
                    error_code="hallucinated_transcript",
                    paste_kind="none",
                    noop_reason="hallucinated_transcript",
                    degraded_writer=degraded_writer_lane,
                    frozen_context_merged=frozen_context_merged,
                    transcript_stage=stage,
                )
        # 6c. Post-ASR context enrichment ----------------------------------
        #
        # The ASR prompt must be built before Whisper runs, but Qwen's final
        # speech-resolution pass should see terms ranked against what Whisper
        # actually heard. Recompile the serving packet/term list after final ASR
        # without changing the already-used ASR bias plan.
        if not live_adjudication and raw_text:
            prior_term_count = len(compiled_context.terms)
            prior_memory_packet = memory_packet.to_dict()
            compiled_context = compile_context(
                utterance_id=uid,
                context=context,
                memory_snapshot=memory_snapshot,
                mode_selection=mode_sel,
                mode_policy=mode_pol,
                transcript_hint=transcript_hint,
                final_transcript_text=raw_text,
                session_terms=tape_meta.get("candidate_terms") if isinstance(tape_meta, dict) else None,
                language=requested_language,
                stage=stage,
                seed_attachment=seed_attachment,
            )
            memory_packet = compiled_context.memory_packet
            plan.metadata["post_asr_memory_serving_packet"] = memory_packet.to_dict()
            self.recorder.record(
                TraceKind.CONTEXT,
                "context_compiler_post_asr_enriched",
                {
                    "utterance_id": uid,
                    "prior_term_count": prior_term_count,
                    "term_count": len(compiled_context.terms),
                    "memory_packet_changed": prior_memory_packet != memory_packet.to_dict(),
                    "raw_text_chars": len(raw_text),
                },
            )

        # 7. Memory-aware normalization -------------------------------------
        memory_norm: TranscriptNormalization = self.bias_engine.normalize_transcript(
            raw_text, snapshot=memory_snapshot, plan=plan, scope="oneshot"
        )

        # 8. Language-aware normalization ----------------------------------
        language_norm: TranscriptNormalization = self.transcript_normalizer.normalize_transcript(
            memory_norm.normalized_text,
            requested_language=requested_language,
            observed_language=result.language,
            policy_name=language_policy,
            scope="oneshot",
        )
        normalized_text = language_norm.normalized_text
        all_applied = [c.to_dict() for c in memory_norm.applied] + [
            c.to_dict() for c in language_norm.applied
        ]

        # 8b. ITN plane -------------------------------------------------------
        # Whole-utterance formatting commands ("New paragraph", "Next bullet")
        # must reach the writer's command parser as WORDS. ITN's spoken-
        # punctuation pass would collapse them ("New paragraph" → "\n\n"),
        # after which the parser sees an empty surface and the turn dies as
        # unsupported_intent (production 2026-06-11).
        if self.itn_enabled and not live_adjudication and _is_pure_command_utterance(normalized_text):
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_itn_skipped_for_command",
                {"utterance_id": uid, "text_preview": normalized_text[:60]},
            )
        elif self.itn_enabled and not live_adjudication:
            itn_profile = self.itn_engine.profile_for_category(
                getattr(context, "app_category", None)
            )
            itn_fmt = resolve_itn_format_policy(context)
            itn_result = self.itn_engine.run(
                normalized_text, profile=itn_profile, format_policy=itn_fmt
            )
            normalized_text = itn_result.text
            if itn_result.changed:
                all_applied.append({
                    "rule": "itn",
                    "profile": itn_result.profile,
                    "rules_applied": itn_result.rules_applied,
                    "scope": "oneshot",
                })
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_itn_applied",
                    {
                        "utterance_id": uid,
                        "profile": itn_result.profile,
                        "rules_applied": itn_result.rules_applied,
                        "original_chars": len(itn_result.original_text),
                        "output_chars": len(normalized_text),
                        "format": dict(itn_result.format_snapshot),
                    },
                )
        alphabet_edits = _repair_alphabet_sequence_runs(normalized_text)
        if alphabet_edits.changed:
            before_alpha = normalized_text
            normalized_text = alphabet_edits.text
            all_applied.append({
                "rule": "alphabet_sequence_repair",
                "scope": "oneshot",
                "edits": list(alphabet_edits.edits),
            })
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_alphabet_sequence_repaired",
                {
                    "utterance_id": uid,
                    "original_chars": len(before_alpha),
                    "output_chars": len(normalized_text),
                    "edit_count": len(alphabet_edits.edits),
                    "edits": list(alphabet_edits.edits[:8]),
                },
            )
        protected_term_values = tuple(
            t.canonical or t.text for t in compiled_context.terms if getattr(t, "protected", False)
        )
        context_term_values = tuple(t.canonical or t.text for t in compiled_context.terms)
        context_proper_term_values = _context_proper_repair_terms(context)
        repair_term_values = _dedupe_term_values((
            *protected_term_values,
            *context_term_values,
            *context_proper_term_values,
        ))
        explicit_candidate_terms = _explicit_candidate_terms(context)
        reconciled_text, proper_noun_replacements = _reconcile_proper_nouns_from_live_hint(
            live_hint=transcript_hint,
            final_text=normalized_text,
            protected_terms=protected_term_values,
        )
        if proper_noun_replacements:
            normalized_text = reconciled_text
            all_applied.append({
                "rule": "live_hint_proper_noun_reconciliation",
                "scope": "oneshot",
                "replacements": proper_noun_replacements,
            })
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_live_hint_proper_nouns_reconciled",
                {
                    "utterance_id": uid,
                    "replacement_count": len(proper_noun_replacements),
                    "replacements": proper_noun_replacements[:8],
                },
            )
        protected_text, protected_term_replacements = _reconcile_explicit_candidate_term_confusions(
            text=normalized_text,
            explicit_candidate_terms=explicit_candidate_terms,
            protected_terms=protected_term_values,
        )
        if protected_term_replacements:
            normalized_text = protected_text
            all_applied.append({
                "rule": "explicit_candidate_term_reconciliation",
                "scope": "oneshot",
                "replacements": protected_term_replacements,
            })
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_explicit_candidate_terms_reconciled",
                {
                    "utterance_id": uid,
                    "stage": "pre_adjudication",
                    "replacement_count": len(protected_term_replacements),
                    "replacements": protected_term_replacements[:8],
                },
            )
        protected_text, protected_near_miss_replacements = _reconcile_protected_term_near_misses(
            text=normalized_text,
            protected_terms=repair_term_values,
        )
        if protected_near_miss_replacements:
            normalized_text = protected_text
            all_applied.append({
                "rule": "protected_term_near_miss_reconciliation",
                "scope": "oneshot",
                "replacements": protected_near_miss_replacements,
            })
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_protected_term_near_misses_reconciled",
                {
                    "utterance_id": uid,
                    "stage": "pre_adjudication",
                    "replacement_count": len(protected_near_miss_replacements),
                    "replacements": protected_near_miss_replacements[:8],
                },
            )
        terminal_text, terminal_command_replacements = _repair_terminal_protected_command_terms(
            text=normalized_text,
            explicit_candidate_terms=explicit_candidate_terms,
            protected_terms=protected_term_values,
            app_category=context.app_category,
        )
        if terminal_command_replacements:
            normalized_text = terminal_text
            all_applied.append({
                "rule": "terminal_protected_command_term_repair",
                "scope": "oneshot",
                "replacements": terminal_command_replacements,
            })
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_terminal_command_terms_repaired",
                {
                    "utterance_id": uid,
                    "replacement_count": len(terminal_command_replacements),
                    "replacements": terminal_command_replacements[:8],
                },
            )
        self_correction_cues = _collect_self_correction_cues(normalized_text)
        if self_correction_cues:
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_self_correction_cues_detected",
                {
                    "utterance_id": uid,
                    "cue_count": len(self_correction_cues),
                    "cues": list(self_correction_cues[:8]),
                },
            )
        # Deterministic retake application. The model lanes get cue diagnostics
        # too, but they pass through on a large share of real utterances. This
        # pass is intentionally conservative and also catches ASR variants of a
        # spoken cue, for example "scratch that" heard as "scratched at" between
        # two clock expressions.
        retake_text, retakes_applied = apply_unambiguous_retakes(normalized_text)
        if retakes_applied:
            normalized_text = retake_text
            all_applied.append({
                "rule": "self_correction_retakes",
                "scope": "oneshot",
                "replacements": retakes_applied,
            })
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_self_corrections_applied",
                {
                    "utterance_id": uid,
                    "applied_count": len(retakes_applied),
                    "retakes": retakes_applied[:8],
                },
            )
            self_correction_cues = _collect_self_correction_cues(normalized_text)
        if self_correction_cues:
            compiled_context.metadata["self_correction_cues"] = list(self_correction_cues)
        self.recorder.record(
            TraceKind.SYSTEM,
            "oneshot_final_transcription_postprocess",
            {
                "utterance_id": uid,
                "final_normalized_text": _text_debug_payload(normalized_text),
                "memory_normalized_text": _text_debug_payload(memory_norm.normalized_text),
                "language_normalized_text": _text_debug_payload(language_norm.normalized_text),
                "normalization_applied": list(all_applied),
                "itn_enabled": bool(self.itn_enabled),
                "live_adjudication": bool(live_adjudication),
            },
        )

        # 8c. Transcript adjudication ----------------------------------------
        # Qwen is the transcript intelligence layer. It runs after acoustic
        # decode + bounded memory/language/ITN candidates and before actions
        # or final formatting. Live adjudication remains disabled by default
        # and falls back to today's normalized text until explicitly enabled.
        adjudication_result = None
        adjudicated_text = normalized_text
        transcript_policy = getattr(mode_pol, "transcript_correction_policy", "standard")
        live_correction_policy = getattr(
            mode_pol, "live_correction_policy", "stable_span_standard"
        )
        adjudication_skip_reason: str | None = None
        fast_skip_reason: str | None = None
        # Non-wake dictation is corrected by the dictation editor (one 4B
        # pass with full context); running the 0.6B final adjudication too
        # would be redundant model latency. Wake/action turns keep it so
        # commands are cleaned before extraction.
        editor_owns_final_correction = (
            not live_adjudication
            and self.writer_enabled
            and self.writer_service is not None
            and bool(getattr(getattr(self.writer_service, "config", None), "dictation_editor_enabled", False))
            and not leading_wake_status(raw_text, normalized_text).verified
        )
        if editor_owns_final_correction:
            adjudication_skip_reason = "dictation_editor_lane"
        if not editor_owns_final_correction and _should_run_transcript_adjudication(
            transcript_policy=transcript_policy,
            live_correction_policy=live_correction_policy,
            app_category=context.app_category,
            live_adjudication=live_adjudication,
            config=self.transcript_adjudicator_config,
        ) and not (
            fast_skip_reason := _final_adjudication_fast_skip_reason(
                live_adjudication=live_adjudication,
                transcript_hint=transcript_hint,
                raw_text=raw_text,
                normalized_text=normalized_text,
                normalization_applied=all_applied,
                audio_duration_ms=result.audio_duration_ms,
            )
        ):
            packet = compiled_context.transcript_packet(
                stage="live" if live_adjudication else "final",
                base_visible_text=transcript_hint or "",
                base_visible_revision=None,
                live_preview_text=transcript_hint or "",
                whisper_text=raw_text,
                memory_candidate_text=normalized_text,
                raw_text=raw_text,
            )
            self.recorder.record(
                TraceKind.CONTEXT,
                "context_compiler_transcript_packet",
                {
                    "utterance_id": uid,
                    "stage": packet.stage,
                    "term_count": len(packet.context_terms),
                    "protected_terms": len(packet.protected_terms),
                    "no_touch": packet.no_touch,
                },
            )
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_transcript_adjudication_started",
                {
                    "utterance_id": uid,
                    "stage": packet.stage,
                    "policy": transcript_policy,
                },
            )
            adjudicator = (
                self._active_live_transcript_adjudicator()
                if live_adjudication
                else self._active_transcript_adjudicator()
            )
            adjudication_result = adjudicator.adjudicate(packet)
            if adjudication_result.rejected:
                salvaged = False
                if not live_adjudication and str(adjudication_result.corrected_text or "").strip():
                    original_reason = adjudication_result.rejected_reason
                    original_text = adjudication_result.corrected_text
                    repaired_text = original_text
                    validation_repair_replacements: list[dict[str, str]] = []

                    protected_text, protected_term_replacements = _reconcile_explicit_candidate_term_confusions(
                        text=repaired_text,
                        explicit_candidate_terms=explicit_candidate_terms,
                        protected_terms=protected_term_values,
                    )
                    if protected_term_replacements:
                        repaired_text = protected_text
                        validation_repair_replacements.extend(protected_term_replacements)

                    protected_text, protected_near_miss_replacements = _reconcile_protected_term_near_misses(
                        text=repaired_text,
                        protected_terms=repair_term_values,
                    )
                    if protected_near_miss_replacements:
                        repaired_text = protected_text
                        validation_repair_replacements.extend(protected_near_miss_replacements)

                    if validation_repair_replacements and repaired_text != original_text:
                        adjudication_result.corrected_text = repaired_text
                        ok, validation_reason = validate_adjudication_result(packet, adjudication_result)
                        if ok:
                            adjudication_result.rejected = False
                            adjudication_result.rejected_reason = None
                            adjudication_result.metadata = dict(adjudication_result.metadata or {})
                            adjudication_result.metadata["salvaged_after_validation_repair"] = {
                                "original_rejected_reason": original_reason,
                                "replacement_count": len(validation_repair_replacements),
                            }
                            adjudicated_text = repaired_text
                            all_applied.append({
                                "rule": "adjudication_validation_repair",
                                "scope": "oneshot_adjudication_salvage",
                                "replacements": validation_repair_replacements,
                            })
                            self.recorder.record(
                                TraceKind.SYSTEM,
                                "oneshot_transcript_adjudication_salvaged",
                                {
                                    "utterance_id": uid,
                                    "stage": packet.stage,
                                    "original_reason": original_reason,
                                    "replacement_count": len(validation_repair_replacements),
                                    "replacements": validation_repair_replacements[:8],
                                },
                            )
                            salvaged = True
                        else:
                            adjudication_result.corrected_text = original_text
                            adjudication_result.rejected_reason = original_reason or validation_reason
                if not salvaged:
                    fallback_text = _fallback_adjudicated_text(
                        raw_text=raw_text,
                        normalized_text=normalized_text,
                        memory_candidate_text=normalized_text,
                    )
                    adjudicated_text = fallback_text
                    self.recorder.record(
                        TraceKind.SYSTEM,
                        "oneshot_transcript_adjudication_fallback",
                        {
                            "utterance_id": uid,
                            "stage": packet.stage,
                            "reason": adjudication_result.rejected_reason,
                            "fallback_chars": len(fallback_text),
                        },
                    )
            else:
                adjudicated_text = adjudication_result.corrected_text
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_transcript_adjudication_ok",
                    {
                        "utterance_id": uid,
                        "stage": packet.stage,
                        "confidence": adjudication_result.confidence,
                        "ops": len(adjudication_result.ops),
                        "backend": adjudication_result.backend_name,
                    },
                )
        else:
            adjudication_skip_reason = fast_skip_reason or "policy_or_live_disabled"
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_transcript_adjudication_rejected",
                {
                    "utterance_id": uid,
                    "stage": "live" if live_adjudication else "final",
                    "reason": adjudication_skip_reason,
                    "policy": transcript_policy,
                    "live_correction_policy": live_correction_policy,
                },
            )

        if not live_adjudication and adjudicated_text:
            protected_text, protected_term_replacements = _reconcile_explicit_candidate_term_confusions(
                text=adjudicated_text,
                explicit_candidate_terms=explicit_candidate_terms,
                protected_terms=protected_term_values,
            )
            protected_text, protected_near_miss_replacements = _reconcile_protected_term_near_misses(
                text=protected_text,
                protected_terms=repair_term_values,
            )
            all_protected_replacements = protected_term_replacements + protected_near_miss_replacements
            if all_protected_replacements:
                adjudicated_text = protected_text
                if adjudication_result is not None and not adjudication_result.rejected:
                    adjudication_result.corrected_text = protected_text
                all_applied.append({
                    "rule": "protected_term_reconciliation",
                    "scope": "oneshot_final",
                    "replacements": all_protected_replacements,
                })
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_explicit_candidate_terms_reconciled",
                    {
                        "utterance_id": uid,
                        "stage": "post_adjudication",
                        "replacement_count": len(all_protected_replacements),
                        "replacements": all_protected_replacements[:8],
                    },
                )
            duplicate_result = _collapse_adjacent_duplicate_phrases(adjudicated_text)
            if duplicate_result.changed:
                adjudicated_text = duplicate_result.text
                if adjudication_result is not None and not adjudication_result.rejected:
                    adjudication_result.corrected_text = duplicate_result.text
                all_applied.append({
                    "rule": "adjacent_duplicate_phrase_collapse",
                    "scope": "oneshot_final",
                    "edits": list(duplicate_result.edits),
                })
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "oneshot_adjacent_duplicate_phrases_collapsed",
                    {
                        "utterance_id": uid,
                        "edit_count": len(duplicate_result.edits),
                        "edits": list(duplicate_result.edits[:8]),
                    },
                )
            # Self-correction resolution belongs to Qwen's final speech
            # resolver. Do not run the old deterministic edit pass after Qwen:
            # it can delete literal spoken content such as "the words blank
            # space" and creates traces where Qwen's decision no longer matches
            # the final paste text.

        if live_adjudication:
            resolved_model_path = str(getattr(result, "model_path", "") or "")
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_live_adjudicated",
                {
                    "utterance_id": uid,
                    "raw_chars": len(raw_text),
                    "normalized_chars": len(normalized_text),
                    "adjudicated_chars": len(adjudicated_text),
                    "duration_ms": result.audio_duration_ms,
                    "backend": result.backend_name,
                    "model_path": resolved_model_path,
                    "bias_phrase_count": len(plan.bias_phrases),
                    "session_context_terms": len(tape_meta.get("candidate_terms", [])) if isinstance(tape_meta, dict) else 0,
                },
            )
            meta_out = {
                "language_policy": language_policy,
                "has_initial_prompt": bool(plan.initial_prompt),
                "paste_kind": "none",
                "live_adjudication": True,
                "session_context_tape": tape_meta,
            }
            if adjudication_result is not None and not adjudication_result.rejected:
                meta_out["transcript_patch"] = adjudication_result.to_dict()
            _record_transcript_decision(
                self.recorder,
                utterance_id=uid,
                stage="live",
                raw_text=raw_text,
                normalized_text=normalized_text,
                adjudicated_text=adjudicated_text,
                writer_text="",
                context=context,
                compiled_context=compiled_context,
                memory_packet=memory_packet,
                plan=plan,
                mode_selection=mode_sel,
                mode_policy=mode_pol,
                language_policy=language_policy,
                requested_language=requested_language,
                backend_name=result.backend_name,
                model_path=resolved_model_path,
                audio_duration_ms=result.audio_duration_ms,
                decode_ms=result.decode_ms,
                tape_meta=tape_meta,
                adjudication_result=adjudication_result,
                adjudication_skip_reason=adjudication_skip_reason,
                patch_included="transcript_patch" in meta_out,
                paste_kind="none",
                noop_reason="live_adjudication",
                parsed_actions_payload=None,
                action_attempt_rejected=False,
                wake_status=None,
            )
            return OneShotDictationResult(
                utterance_id=uid,
                ok=True,
                transcript=adjudicated_text,
                raw_transcript=raw_text,
                backend_name=result.backend_name,
                model_path=resolved_model_path,
                audio_duration_ms=result.audio_duration_ms,
                decode_ms=result.decode_ms,
                language=result.language or requested_language,
                writer_action=None,
                writer_deterministic=False,
                memory_updated=False,
                normalization_applied=all_applied,
                bias_phrase_count=len(plan.bias_phrases),
                context_present=bool(
                    context.app_name or context.selected_text or context.focused_text_before or context.candidate_entities
                ),
                capability_mode=capability_mode,
                paste_kind="none",
                noop_reason="live_adjudication",
                degraded_writer=degraded_writer_lane,
                frozen_context_merged=frozen_context_merged,
                metadata=meta_out,
                actions=None,
                is_action=False,
                transcript_stage=stage,
            )

        # 9a. Actions detection ----------------------------------------------
        # Wake verification is based only on raw/normalized ASR before Qwen.
        # The action extractor then receives the adjudicated post-wake content.
        # This prevents transcript adjudication from inventing "Juno" and
        # accidentally turning ordinary dictation into an action.
        parsed_actions_payload: list[dict[str, Any]] | None = None
        wake_status = leading_wake_status(raw_text, normalized_text)
        self.recorder.record(
            TraceKind.SYSTEM,
            "action_wake_gate_checked",
            {
                "utterance_id": uid,
                "raw_wake_detected": wake_status.raw_wake_detected,
                "normalized_wake_detected": wake_status.normalized_wake_detected,
            },
        )
        candidate_action_source_text = _post_wake_from_adjudicated(adjudicated_text) if wake_status.verified else adjudicated_text
        hinted_action_source_text = _action_source_from_live_hint(
            candidate_action_source_text,
            transcript_hint=transcript_hint,
            wake_verified=wake_status.verified,
        )
        if hinted_action_source_text is not None:
            self.recorder.record(
                TraceKind.SYSTEM,
                "action_source_recovered_from_live_hint",
                {
                    "utterance_id": uid,
                    "final_post_wake_chars": len(candidate_action_source_text or ""),
                    "hint_post_wake_chars": len(hinted_action_source_text),
                },
            )
            candidate_action_source_text = hinted_action_source_text
        action_source_text = candidate_action_source_text if wake_status.verified else adjudicated_text
        action_attempt_rejected = False
        action_now = datetime.now().astimezone()
        writer_mode_policy = _mode_policy_for_final_delivery(
            mode_pol,
            context=context,
            raw_text=raw_text,
            adjudicated_text=adjudicated_text,
            adjudication_result=adjudication_result,
        )
        turn_plan_result = None
        turn_plan_validation_payload: dict[str, Any] | None = None
        turn_plan_allows_mixed_paste = False
        turn_plan_controls_actions = False
        turn_plan_source_text = action_source_text if wake_status.verified else adjudicated_text
        # When the dictation editor owns non-wake dictation, the pipeline
        # only pre-plans for wake (actions) and selection (transform) turns;
        # plain dictation goes straight to the editor inside the writer
        # service, with the planner as its internal floor.
        editor_owns_dictation = (
            self.writer_enabled
            and self.writer_service is not None
            and bool(getattr(getattr(self.writer_service, "config", None), "dictation_editor_enabled", False))
            and not wake_status.verified
            and not (getattr(context, "selected_text", "") or "").strip()
        )
        if (
            self.writer_enabled
            and self.writer_service is not None
            and turn_plan_source_text
            and not editor_owns_dictation
        ):
            try:
                wt = plan.metadata.get("surface_preset_writer_tone")
                writer_tone = wt.strip() if isinstance(wt, str) and wt.strip() else None
                turn_plan_result = self.writer_service.plan_turn(
                    utterance_id=uid,
                    final_text=turn_plan_source_text,
                    raw_text=raw_text,
                    context=context,
                    memory_store=self.memory_store,
                    memory_snapshot=memory_snapshot,
                    memory_packet=memory_packet.to_dict(),
                    language_hint=requested_language,
                    mode_policy=writer_mode_policy,
                    mode_selection=mode_sel,
                    partial_text=transcript_hint or "",
                    writer_tone_addon=writer_tone,
                    wake_verified=wake_status.verified,
                    now_iso=action_now.isoformat(),
                )
            except Exception as exc:  # noqa: BLE001
                self.recorder.record(
                    TraceKind.WRITER,
                    "turn_plan_pipeline_error",
                    {"utterance_id": uid, "error": str(exc)},
                )
                turn_plan_result = None

        if wake_status.verified and action_source_text:
            should_try_action_fallback = False
            action_fallback_reason: str | None = None

            def _route_action_to_fallback(reason: str, **extra: Any) -> None:
                nonlocal should_try_action_fallback, action_fallback_reason
                should_try_action_fallback = True
                action_fallback_reason = reason
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "turn_plan_action_routed_to_fallback",
                    {
                        "utterance_id": uid,
                        "reason": reason,
                        **extra,
                    },
                )

            if turn_plan_result is not None:
                turn_plan_controls_actions = True
                if turn_plan_result.ok and isinstance(turn_plan_result.plan, dict):
                    validation = validate_turn_plan(turn_plan_result.plan, source_text=turn_plan_source_text, context=context)
                    turn_plan_validation_payload = {
                        "ok": validation.ok,
                        "errors": list(validation.errors),
                        "warnings": list(validation.warnings),
                    }
                    self.recorder.record(
                        TraceKind.SYSTEM,
                        "turn_plan_action_validation",
                        {
                            "utterance_id": uid,
                            **turn_plan_validation_payload,
                            "utterance_kind": turn_plan_result.plan.get("utterance_kind"),
                        },
                    )
                    if validation.ok:
                        plan_kind = str(turn_plan_result.plan.get("utterance_kind") or "").strip()
                        turn_plan_allows_mixed_paste = _turn_plan_allows_mixed_paste(turn_plan_result.plan)
                        planned = actions_from_turn_plan(
                            turn_plan_result.plan,
                            source_text=turn_plan_source_text,
                            now=action_now,
                        )
                        if planned.actions and not _looks_like_action_attempt(action_source_text):
                            # Wake word alone is not consent to reroute
                            # dictation into Notes/Reminders. Without an
                            # explicit action verb in the spoken text the
                            # planner's actions are treated as hallucinated
                            # and the utterance stays dictation (pasted).
                            self.recorder.record(
                                TraceKind.SYSTEM,
                                "turn_plan_actions_ignored_without_verb",
                                {
                                    "utterance_id": uid,
                                    "count": len(planned.actions),
                                    "kinds": [a.kind.value for a in planned.actions],
                                },
                            )
                        elif planned.actions:
                            parsed_actions_payload = [a.to_dict() for a in planned.actions]
                            self.recorder.record(
                                TraceKind.SYSTEM,
                                "turn_plan_actions_detected",
                                {
                                    "utterance_id": uid,
                                    "count": len(planned.actions),
                                    "kinds": [a.kind.value for a in planned.actions],
                                    "mixed_paste_allowed": turn_plan_allows_mixed_paste,
                                    "missing_fields": list(planned.missing_fields),
                                    "skipped_reasons": list(planned.skipped_reasons),
                                },
                            )
                            if planned.skipped_reasons:
                                # Valid sibling actions shipped; these did not.
                                # Distinct event so partially dropped commands
                                # are findable without diffing counts.
                                self.recorder.record(
                                    TraceKind.SYSTEM,
                                    "turn_plan_actions_partially_skipped",
                                    {
                                        "utterance_id": uid,
                                        "skipped_count": len(planned.skipped_reasons),
                                        "skipped_reasons": list(planned.skipped_reasons),
                                        "accepted_count": len(planned.actions),
                                        "missing_fields": list(planned.missing_fields),
                                    },
                                )
                        elif planned.rejected_reason and (
                            plan_kind in {"actions", "mixed"}
                            or bool(turn_plan_result.plan.get("actions"))
                        ):
                            if "unsupported_operation" in str(planned.rejected_reason):
                                # The turn-plan lane only creates. Operations
                                # on existing actions (complete / update /
                                # delete / snooze) belong to the extractor
                                # lane with the actions-index reference
                                # resolver — route there instead of failing
                                # the turn.
                                should_try_action_fallback = True
                                action_fallback_reason = "turn_plan_unsupported_operation"
                                self.recorder.record(
                                    TraceKind.SYSTEM,
                                    "turn_plan_operation_routed_to_fallback",
                                    {
                                        "utterance_id": uid,
                                        "reason": planned.rejected_reason,
                                    },
                                )
                            else:
                                if _looks_like_action_attempt(action_source_text):
                                    _route_action_to_fallback(
                                        str(planned.rejected_reason or "turn_plan_action_rejected"),
                                        missing_fields=list(planned.missing_fields),
                                    )
                                else:
                                    action_attempt_rejected = True
                                    self.recorder.record(
                                        TraceKind.SYSTEM,
                                        "turn_plan_action_rejected",
                                        {
                                            "utterance_id": uid,
                                            "reason": planned.rejected_reason,
                                            "missing_fields": list(planned.missing_fields),
                                        },
                                    )
                        elif plan_kind in {"actions", "mixed"} or bool(turn_plan_result.plan.get("actions")):
                            if _looks_like_action_attempt(action_source_text):
                                _route_action_to_fallback(
                                    str(planned.rejected_reason or "no_valid_actions"),
                                    missing_fields=list(planned.missing_fields),
                                )
                            else:
                                action_attempt_rejected = True
                                self.recorder.record(
                                    TraceKind.SYSTEM,
                                    "turn_plan_action_rejected",
                                    {
                                        "utterance_id": uid,
                                        "reason": planned.rejected_reason or "no_valid_actions",
                                        "missing_fields": list(planned.missing_fields),
                                    },
                                )
                    elif _looks_like_action_attempt(action_source_text):
                        _route_action_to_fallback(
                            "turn_plan_validation_failed",
                            validation_errors=list(validation.errors),
                        )
                elif _looks_like_action_attempt(action_source_text):
                    _route_action_to_fallback(
                        str(getattr(turn_plan_result, "status", None) or "turn_plan_invalid"),
                        errors=list(getattr(turn_plan_result, "errors", ()) or ()),
                    )
            else:
                should_try_action_fallback = _looks_like_action_attempt(action_source_text)
                action_fallback_reason = "turn_plan_unavailable" if should_try_action_fallback else None

            if should_try_action_fallback and not parsed_actions_payload:
                parsed_actions = detect_actions_for_pipeline(
                    utterance_id=uid,
                    normalized_text=action_source_text,
                    recorder=self.recorder,
                    trace_kind=TraceKind.SYSTEM,
                    now=action_now,
                    wake_verified=True,
                    raw_wake_text=wake_status.raw_wake_text,
                    context_packet=compiled_context.action_packet(
                        corrected_text=action_source_text,
                        raw_or_normalized_text_for_wake_gate=wake_status.raw_wake_text or normalized_text,
                        now_iso=action_now.isoformat(),
                    ),
                )
                if parsed_actions:
                    parsed_actions, skipped_fallback_actions = _dispatchable_actions_for_pipeline(parsed_actions)
                    if skipped_fallback_actions:
                        self.recorder.record(
                            TraceKind.SYSTEM,
                            "turn_plan_action_fallback_partially_skipped",
                            {
                                "utterance_id": uid,
                                "reason": action_fallback_reason,
                                "skipped_reasons": skipped_fallback_actions,
                                "accepted_count": len(parsed_actions or []),
                            },
                        )
                if parsed_actions:
                    parsed_actions_payload = [a.to_dict() for a in parsed_actions]
                    turn_plan_controls_actions = False
                    self.recorder.record(
                        TraceKind.SYSTEM,
                        "turn_plan_action_fallback_used",
                        {
                            "utterance_id": uid,
                            "reason": action_fallback_reason,
                            "count": len(parsed_actions),
                            "kinds": [a.kind.value for a in parsed_actions],
                        },
                    )
                elif _looks_like_action_attempt(action_source_text):
                    action_attempt_rejected = True

        # Seed context-entity observations are durable memory signals. They are
        # recorded only after the final text is actually committed by the user
        # (see record_insertion). Pre-paste ASR observations polluted memory with
        # HUD/ASR noise such as generic starts and one-off mishears.
        if self.juno_seed_runtime is not None:
            self.recorder.record(
                TraceKind.MEMORY,
                "oneshot_seed_observation_deferred",
                {
                    "utterance_id": uid,
                    "reason": "awaiting_successful_commit",
                    "action_utterance": bool(parsed_actions_payload or action_attempt_rejected),
                },
            )

        # 9. Writer service --------------------------------------------------
        writer_outcome: WriterOutcome | None = None
        if (
            not parsed_actions_payload
            and not action_attempt_rejected
            and self.writer_enabled
            and self.writer_service is not None
            and adjudicated_text
        ):
            try:
                wt = plan.metadata.get("surface_preset_writer_tone")
                writer_tone = wt.strip() if isinstance(wt, str) and wt.strip() else None
                writer_outcome = self.writer_service.process_transcript(
                    utterance_id=uid,
                    final_text=action_source_text if wake_status.verified else adjudicated_text,
                    raw_text=raw_text,
                    context=context,
                    anchor_selection=_anchor_from_context(context),
                    memory_store=self.memory_store,
                    memory_snapshot=memory_snapshot,
                    memory_packet=memory_packet.to_dict(),
                    language_hint=requested_language,
                    mode_policy=writer_mode_policy,
                    mode_selection=mode_sel,
                    partial_text=transcript_hint or "",
                    writer_tone_addon=writer_tone,
                    turn_plan_result=turn_plan_result,
                    wake_verified=wake_status.verified,
                    now_iso=action_now.isoformat(),
                )
            except Exception as exc:  # noqa: BLE001 — never fail the whole turn on writer errors
                self.recorder.record(
                    TraceKind.WRITER,
                    "oneshot_writer_error",
                    {"utterance_id": uid, "error": str(exc)},
                )
                writer_outcome = None

        writer_fallback_text = (
            action_source_text
            if wake_status.verified and turn_plan_result is not None
            else adjudicated_text
        )
        writer_text, writer_action, writer_deterministic, memory_updated = _writer_to_surface_text(
            writer_outcome, fallback=writer_fallback_text
        )
        writer_text, writer_term_repairs = _preserve_writer_context_terms(
            writer_text,
            fallback_text=writer_fallback_text,
            compiled_context=compiled_context,
        )
        if writer_term_repairs:
            self.recorder.record(
                TraceKind.WRITER,
                "writer_context_terms_preserved",
                {
                    "utterance_id": uid,
                    "repairs": writer_term_repairs[:8],
                    "repair_count": len(writer_term_repairs),
                },
            )
        paste_kind, noop_reason = _compute_oneshot_paste_kind(
            writer_outcome, fallback_text=writer_fallback_text
        )
        writer_safety_reason = _unsafe_writer_surface_reason(
            writer_text,
            fallback_text=writer_fallback_text,
            raw_text=raw_text,
            writer_outcome=writer_outcome,
        )
        if writer_safety_reason is not None:
            self.recorder.record(
                TraceKind.WRITER,
                "oneshot_writer_surface_fallback",
                {
                    "utterance_id": uid,
                    "reason": writer_safety_reason,
                    "writer_action": writer_action,
                    "writer_chars": len(writer_text or ""),
                    "fallback_chars": len(writer_fallback_text or ""),
                },
            )
            writer_text = writer_fallback_text
        if writer_outcome is not None:
            writer_outcome_payload = writer_outcome.to_dict()
            if writer_safety_reason is not None:
                writer_outcome_payload["surface_fallback_reason"] = writer_safety_reason
            self.recorder.record(
                TraceKind.WRITER,
                "writer_outcome",
                writer_outcome_payload,
            )

        # Action utterances ("hey juno, remind me to call mom at 6pm")
        # are voice commands, not dictation — they must never insert
        # the spoken text into whatever field the user happens to have
        # focused. The action parser only returns a non-empty list when
        # BOTH the Juno wake-word AND at least one action verb match,
        # so this is a high-confidence "addressed to Juno" signal.
        # Suppressing the paste here is what lets users fire actions
        # while their cursor is in Slack/Notes/anywhere without
        # polluting that field with the command text.
        recoverable_transcript = ""
        if parsed_actions_payload:
            paste_kind = "none"
            if not noop_reason:
                noop_reason = "action_only"
            writer_text = ""
            self.recorder.record(
                TraceKind.SYSTEM,
                "action_paste_suppressed",
                {
                    "utterance_id": uid,
                    "action_count": len(parsed_actions_payload),
                    "turn_plan_controls_actions": turn_plan_controls_actions,
                },
            )
        elif action_attempt_rejected:
            paste_kind = "none"
            noop_reason = "action_rejected"
            writer_text = ""
            recoverable_transcript = (adjudicated_text or "").strip()
            self.recorder.record(
                TraceKind.SYSTEM,
                "action_extraction_rejected",
                {"utterance_id": uid, "reason": "validator_or_required_fields_failed", "confidence": None},
            )
        elif wake_status.verified:
            self.recorder.record(
                TraceKind.SYSTEM,
                "action_fallthrough_to_dictation",
                {"utterance_id": uid, "reason": "no_valid_action"},
            )

        _record_transcript_decision(
            self.recorder,
            utterance_id=uid,
            stage="final",
            raw_text=raw_text,
            normalized_text=normalized_text,
            adjudicated_text=adjudicated_text,
            writer_text=writer_text,
            context=context,
            compiled_context=compiled_context,
            memory_packet=memory_packet,
            plan=plan,
            mode_selection=mode_sel,
            mode_policy=writer_mode_policy,
            language_policy=language_policy,
            requested_language=requested_language,
            backend_name=result.backend_name,
            model_path=str(getattr(result, "model_path", "") or ""),
            audio_duration_ms=result.audio_duration_ms,
            decode_ms=result.decode_ms,
            tape_meta=tape_meta,
            adjudication_result=adjudication_result,
            adjudication_skip_reason=adjudication_skip_reason,
            patch_included=False,
            paste_kind=paste_kind,
            noop_reason=noop_reason,
            parsed_actions_payload=parsed_actions_payload,
            action_attempt_rejected=action_attempt_rejected,
            wake_status=wake_status,
        )

        # 10. Retain utterance record for insertion_committed / learning ----
        record_raw_text = action_source_text if wake_status.verified and not (parsed_actions_payload or action_attempt_rejected) else raw_text
        record = UtteranceRecord(
            utterance_id=uid,
            raw_text=record_raw_text,
            normalized_text=normalized_text,
            adjudicated_text=adjudicated_text,
            writer_text=writer_text,
            plan=plan,
            context=context,
            literal_text=adjudicated_text,
            final_text=writer_text,
            writer_action=writer_action,
            learn_from_commit=_should_learn_from_oneshot_record(writer_outcome, adjudicated_text, writer_text),
            is_action=bool(parsed_actions_payload or action_attempt_rejected),
        )
        self.records.put(record)

        # Bounded audio retention — save WAV for replay/rerun when configured.
        audio_path_rel: str | None = None
        audio_expires_at_ms: int | None = None
        if save_audio and self.audio_save_dir is not None:
            audio_path_rel, audio_expires_at_ms = self._save_audio_bounded(uid, wav_bytes)
            self.recorder.record(
                TraceKind.SYSTEM,
                "oneshot_audio_retained",
                {
                    "utterance_id": uid,
                    "replay_available": self.replay_available(uid),
                    "rerun_available": self.replay_available(uid),
                    "retention_limit": max(1, int(self.audio_retention_limit)),
                    "storage": "local_bounded",
                },
            )

        # Persistent history (P0). Append a compact per-utterance record to
        # <log_dir>/history.jsonl so UI surfaces survive broker restarts.
        from juno_v2.observability.history_store import append_history_record

        # Read back the retained context/mode from the record cache (authoritative).
        cached = self.records.get(uid)
        cached_ctx = cached.context if cached is not None else context
        ms = plan.metadata.get("mode_selection", {}) if isinstance(plan.metadata, dict) else {}
        cached_mode = (ms.get("effective_mode") or ms.get("selected_mode") or ms.get("mode")) if isinstance(ms, dict) else None

        processing_ms = int((time.perf_counter_ns() - started_ns) / 1_000_000.0)
        transcript_out = (writer_text or "").strip()
        if not transcript_out and action_attempt_rejected:
            # A rejected action attempt suppresses the paste, but the spoken
            # words must stay recoverable from History — silent loss of the
            # transcript is never acceptable.
            transcript_out = (adjudicated_text or "").strip()
        raw_out = (raw_text or "").strip()
        words = len([tok for tok in transcript_out.split() if tok])
        app_name = getattr(cached_ctx, "app_name", None)
        window_title = getattr(cached_ctx, "window_title", None)
        app_category = getattr(cached_ctx, "app_category", None)
        bundle_id = (getattr(cached_ctx, "metadata", None) or {}).get("app_bundle_id")

        if save_history:
            append_history_record(
                self.recorder.log_dir,
                {
                    "utterance_id": uid,
                    "ts_unix_ms": int(time.time() * 1000),
                    "transcript": transcript_out,
                    "raw_transcript": raw_out,
                    "mode": str(cached_mode or mode_sel.effective_mode),
                    "final_backend": str(getattr(result, "backend_name", "") or getattr(self.transcriber, "backend_name", "")),
                    "model_path": str(getattr(result, "model_path", "") or ""),
                    "language": result.language,
                    "language_mode": language_mode or language,
                    "context": {
                        "app_name": app_name,
                        "app_bundle_id": bundle_id,
                        "window_title": window_title,
                        "app_category": app_category,
                    },
                    "failure_reason": None if paste_kind != "none" else (noop_reason or None),
                    "session_class": "insert",
                    "processing_ms": processing_ms,
                    "words": words,
                    "replay_available": bool(self.replay_available(uid)),
                    "audio_path": audio_path_rel,
                    "audio_expires_at": audio_expires_at_ms,
                    # Parsed actions (Phase 3). Stored as the *intent*; the
                    # macOS shell overwrites with execution outcomes via
                    # POST /api/broker/history/{uid}/actions once it's
                    # finished talking to EKEventStore / Notes.
                    "actions": parsed_actions_payload,
                },
            )

        resolved_model_path = str(getattr(result, "model_path", "") or "")
        self.recorder.record(
            TraceKind.SYSTEM,
            "oneshot_transcribed",
            {
                "utterance_id": uid,
                "raw_chars": len(raw_text),
                "normalized_chars": len(normalized_text),
                "adjudicated_chars": len(adjudicated_text),
                "writer_chars": len(writer_text),
                "duration_ms": result.audio_duration_ms,
                "backend": result.backend_name,
                "model_path": resolved_model_path,
                "writer_action": writer_action,
                "memory_packet_summary": {
                    "lexicon_terms": len(memory_packet.lexicon_terms),
                    "replacements": len(memory_packet.replacements),
                    "corrections": len(memory_packet.corrections),
                    "session_entities": len(memory_packet.session_entities),
                },
            },
        )

        meta_out = {
            "language_policy": language_policy,
            "has_initial_prompt": bool(plan.initial_prompt),
            "paste_kind": paste_kind,
            "session_context_tape": tape_meta,
            "normalized_text": normalized_text,
            "adjudicated_text": adjudicated_text,
        }
        if audio_diag is not None:
            meta_out["audio_diagnostics"] = audio_diag.to_dict()
        if final_asr_hallucination_fallback is not None:
            meta_out["final_asr_hallucination_fallback"] = final_asr_hallucination_fallback
        if adjudication_result is not None:
            meta_out["transcript_adjudication"] = adjudication_result.to_dict()
        if turn_plan_result is not None:
            meta_out["turn_plan"] = _turn_plan_result_summary(
                turn_plan_result,
                validation=turn_plan_validation_payload,
                mixed_paste_allowed=turn_plan_allows_mixed_paste,
                controls_actions=turn_plan_controls_actions,
            )
        if noop_reason:
            meta_out["noop_reason"] = noop_reason
        if writer_safety_reason is not None:
            meta_out["writer_safety_fallback"] = {
                "reason": writer_safety_reason,
                "writer_action": writer_action,
            }
        if writer_outcome is not None:
            outcome_meta = dict(writer_outcome.metadata or {})
            writer_outcome_meta = {
                "action": writer_outcome.action.value,
                "commit_mode": writer_outcome.commit_mode.value if writer_outcome.commit_mode else None,
                "target": outcome_meta.get("target"),
                "target_text_chars": outcome_meta.get("target_text_chars"),
                "deterministic_used": bool(writer_outcome.deterministic_used),
                "model_used": bool(writer_outcome.model_used),
            }
            for key in (
                "snippet_expanded",
                "dictation_cleanup",
                "grammar_postpass",
                "transform_kind",
                "reason",
                "structure",
            ):
                if key in outcome_meta:
                    writer_outcome_meta[key] = outcome_meta.get(key)
            meta_out["writer_outcome"] = writer_outcome_meta
            if "snippet_expanded" in outcome_meta:
                meta_out["snippet_expanded"] = bool(outcome_meta.get("snippet_expanded"))

        return OneShotDictationResult(
            utterance_id=uid,
            ok=True,
            transcript=writer_text,
            raw_transcript=raw_text,
            backend_name=result.backend_name,
            model_path=resolved_model_path,
            audio_duration_ms=result.audio_duration_ms,
            decode_ms=result.decode_ms,
            language=result.language or requested_language,
            writer_action=writer_action,
            writer_deterministic=writer_deterministic,
            memory_updated=memory_updated,
            normalization_applied=all_applied,
            bias_phrase_count=len(plan.bias_phrases),
            context_present=bool(
                context.app_name or context.selected_text or context.focused_text_before
            ),
            capability_mode=capability_mode,
            paste_kind=paste_kind,
            noop_reason=noop_reason,
            recoverable_transcript=recoverable_transcript,
            degraded_writer=degraded_writer_lane,
            frozen_context_merged=frozen_context_merged,
            metadata=meta_out,
            actions=parsed_actions_payload,
            is_action=bool(parsed_actions_payload or action_attempt_rejected),
            transcript_stage=stage,
        )

    def _active_transcript_adjudicator(self) -> TranscriptAdjudicator:
        if self.transcript_adjudicator is not None:
            return self.transcript_adjudicator
        return TranscriptAdjudicator(
            backend=self.writer_service if self.writer_service is not None else None,
            recorder=self.recorder,
            config=self.transcript_adjudicator_config,
        )

    def _active_live_transcript_adjudicator(self) -> TranscriptAdjudicator:
        if self.live_transcript_adjudicator is not None:
            return self.live_transcript_adjudicator
        return self._active_transcript_adjudicator()

    # ---- learning hooks used by insertion_committed / observe_correction ----

    def record_insertion(
        self,
        *,
        utterance_id: str | None,
        committed_text: str | None,
    ) -> dict[str, Any]:
        """Learn from a successful paste.

        Called by the workbench's ``/api/broker/insertion/committed``
        endpoint. When both a raw (what ASR heard) and committed (what
        the user kept) text are available, we record a correction pair
        and upsert extracted session entities. Returns a small payload
        describing what was learned so the caller can log it.
        """
        if self.memory_store is None:
            return {"learned": False, "reason": "no_memory_store"}

        record: UtteranceRecord | None = None
        if utterance_id:
            record = self.records.pop(utterance_id)
        if record is None:
            record = self.records.last()
        if record is None:
            return {"learned": False, "reason": "no_utterance_record"}

        sup_meta = None
        if record.plan is not None and self.juno_seed_runtime is not None:
            sup_meta = self.juno_seed_runtime.context_plane_suppression_value(record.plan.metadata)
        durable_suppressed = False
        if self.juno_seed_runtime is not None:
            durable_suppressed = self.juno_seed_runtime.durable_memory_suppressed(
                record.context or TypedContextBundle(),
                context_plane_suppression=sup_meta,
            )

        if record.is_action:
            # The utterance fired as a Juno action ("hey juno take a note…").
            # Memory learning would otherwise record the wake phrase + verb
            # wrapper as a raw→committed correction pair and corrupt the
            # lexicon. Action commits are observed via the actions payload,
            # not the insertion-learning path.
            self.recorder.record(
                TraceKind.MEMORY,
                "oneshot_memory_learn_suppressed",
                {
                    "utterance_id": record.utterance_id,
                    "reason": "action_utterance",
                },
            )
            return {
                "learned": False,
                "reason": "action_utterance",
                "utterance_id": record.utterance_id,
            }

        correction_suppressed_reason: str | None = None
        if not record.learn_from_commit:
            correction_suppressed_reason = "writer_transform_or_structural_rewrite"
            self.recorder.record(
                TraceKind.MEMORY,
                "oneshot_memory_learn_suppressed",
                {
                    "utterance_id": record.utterance_id,
                    "reason": correction_suppressed_reason,
                    "writer_action": record.writer_action,
                },
            )

        raw_text = record.raw_text
        committed = (committed_text or record.writer_text or record.adjudicated_text or record.normalized_text).strip()
        if not committed:
            return {"learned": False, "reason": "empty_committed_text"}

        if durable_suppressed:
            self.recorder.record(
                TraceKind.MEMORY,
                "oneshot_memory_learn_suppressed",
                {
                    "utterance_id": record.utterance_id,
                    "reason": "durable_memory_suppressed",
                },
            )
            return {
                "learned": False,
                "reason": "durable_memory_suppressed",
                "utterance_id": record.utterance_id,
            }

        learned_correction = False
        correction_promo: dict[str, Any] = {}
        if correction_suppressed_reason is None and raw_text and raw_text != committed:
            learned_correction = bool(self.memory_store.record_correction(raw_text, committed))
            if learned_correction and self.juno_seed_runtime is not None:
                correction_promo = self.juno_seed_runtime.promotion.maybe_promote_correction_to_lexicon(
                    observed=raw_text,
                    corrected=committed,
                    durable_memory_suppressed=False,
                )

        committed_lower = committed.casefold()
        raw_or_adjudicated = " ".join(
            part
            for part in (
                raw_text,
                record.normalized_text,
                record.adjudicated_text,
                record.literal_text,
            )
            if part
        )
        if correction_suppressed_reason is None:
            entities = [
                token
                for token in self.bias_engine.extract_session_entities(committed)
                if _commit_session_entity_allowed(
                    token,
                    committed_text=committed,
                    spoken_evidence_text=raw_or_adjudicated,
                    context=record.context,
                )
            ]
        else:
            entities = []
        if record.plan is not None:
            for candidate in record.plan.context.candidate_entities[:8]:
                token = (candidate or "").strip()
                if (
                    self.bias_engine.is_session_entity_candidate(token)
                    and token.casefold() in committed_lower
                    and (
                        correction_suppressed_reason is None
                        or term_present_in_text(token, raw_or_adjudicated)
                    )
                    and _commit_session_entity_allowed(
                        token,
                        committed_text=committed,
                        spoken_evidence_text=raw_or_adjudicated,
                        context=record.context,
                        context_backed=True,
                    )
                ):
                    entities.append(token)
        entities = _canonicalize_session_entities_against_memory(entities, self.memory_store.snapshot())
        entities = [
            token
            for token in dict.fromkeys(entities)
            if self.bias_engine.is_session_entity_candidate(token)
        ]
        self.memory_store.upsert_session_entities(entities, source="oneshot_commit")

        context_promotions: list[dict[str, Any]] = []
        if self.juno_seed_runtime is not None:
            for token in dict.fromkeys(entities):
                t = (token or "").strip()
                if not learned_term_allowed(t) or t.casefold() not in committed_lower:
                    continue
                self.juno_seed_runtime.learned_store.increment_observation(
                    t,
                    from_suppressed_context=False,
                )
                self.juno_seed_runtime.learned_store.increment_acceptance(
                    t,
                    from_suppressed_context=False,
                )
                context_promotions.append(
                    self.juno_seed_runtime.promotion.maybe_promote_context_entity_to_lexicon(
                        token=t,
                        durable_memory_suppressed=False,
                    )
                )

        self.recorder.record(
            TraceKind.MEMORY,
            "oneshot_memory_updated_from_commit",
            {
                "utterance_id": record.utterance_id,
                "raw_text": raw_text,
                "committed_text": committed,
                "entities": entities,
                "correction_recorded": learned_correction,
                "correction_suppressed_reason": correction_suppressed_reason,
                "correction_promotion": correction_promo,
                "context_promotions": context_promotions,
            },
        )
        learned_anything = learned_correction or bool(entities)
        out = {
            "learned": learned_anything,
            "correction_recorded": learned_correction,
            "entity_count": len(entities),
            "utterance_id": record.utterance_id,
            "correction_promotion": correction_promo,
            "context_promotions": context_promotions,
        }
        if correction_suppressed_reason is not None:
            out["reason"] = correction_suppressed_reason
            out["correction_suppressed_reason"] = correction_suppressed_reason
        return out

    # ---- internals ----

    def _snapshot_context(self) -> TypedContextBundle:
        if self.context_provider is None:
            return TypedContextBundle()
        try:
            bundle = self.context_provider.snapshot()
        except Exception as exc:  # noqa: BLE001
            self.recorder.record(
                TraceKind.CONTEXT,
                "oneshot_context_snapshot_error",
                {"error": str(exc)},
            )
            return TypedContextBundle()
        # Feed the clipboard ring into the bundle so writer & bias engine
        # can reference recent pastes without the context provider
        # needing to know about the workbench-level clipboard ring.
        inject_clipboard_ring(bundle, self.clipboard_ring, limit=5)
        return bundle

    def _apply_surface_hints(
        self,
        context: TypedContextBundle,
        *,
        app_bundle_id: str | None,
        window_title_hint: str | None,
    ) -> None:
        """Merge shell-provided hints into the context bundle.

        Only fills fields that are still empty after the context
        provider ran. We also stamp ``app_category`` whenever we can
        classify it from the (possibly newly filled) app identity,
        because downstream writer / bias logic branches on category
        and a missing value silently disables a lot of personalization.
        """
        if app_bundle_id and not context.metadata.get("app_bundle_id"):
            context.metadata["app_bundle_id"] = app_bundle_id
        if app_bundle_id and not context.app_name:
            # Best-effort fallback: use the bundle id as app name so
            # downstream bias / category code has something to key on
            # when the provider returned nothing at all.
            context.app_name = app_bundle_id
        if window_title_hint and not context.window_title:
            context.window_title = window_title_hint
        if not context.app_category and (context.app_name or app_bundle_id):
            context.app_category = classify_app_category(
                context.app_name,
                context.window_title,
                app_bundle_id=app_bundle_id or context.metadata.get("app_bundle_id"),
            )

    def _apply_session_context_tape(
        self,
        context: TypedContextBundle,
        tape: dict[str, Any] | list[Any] | None,
        *,
        transcript_hint: str | None,
    ) -> dict[str, Any]:
        """Fold a bounded macOS session context tape into candidate terms.

        The tape is serving context only. It must never make durable memory
        writes and it must stay small enough for ASR/writer prompts.
        """
        if tape is None:
            return {"snapshot_count": 0, "candidate_terms": []}
        raw_items: list[dict[str, Any]]
        if isinstance(tape, dict):
            maybe = tape.get("snapshots")
            raw_items = maybe if isinstance(maybe, list) else [tape]
        elif isinstance(tape, list):
            raw_items = tape
        else:
            raw_items = []

        try:
            from juno_v2.context.provider import _context_candidate_allowed, _extract_candidates
        except ImportError:
            _extract_candidates = None
            _context_candidate_allowed = None

        chunks: list[str] = []
        explicit_terms: list[str] = []
        seen_snapshot_keys: set[tuple[str, str, str]] = set()
        for item in raw_items[:12]:
            if not isinstance(item, dict):
                continue
            app = _bounded_text(item.get("app_name") or item.get("frontmost_app_name"), 80)
            title = _bounded_text(item.get("window_title"), 120)
            selected = _bounded_text(item.get("selected_text"), 240)
            before = _bounded_text(item.get("focused_text_before") or item.get("focused_text"), 240)
            after = _bounded_text(item.get("focused_text_after"), 240)
            field_excerpt = _bounded_text(item.get("field_text_excerpt"), 480)
            doc = _bounded_text(item.get("focused_document_path") or item.get("focused_file_path"), 160)
            key = (
                app.casefold(),
                title.casefold(),
                (selected or before or after or field_excerpt).casefold(),
            )
            if key in seen_snapshot_keys:
                continue
            seen_snapshot_keys.add(key)
            chunks.extend([app, title, selected, before, after, field_excerpt, doc])
            raw_candidates = item.get("candidate_entities") or item.get("candidate_terms")
            if isinstance(raw_candidates, list):
                for raw in raw_candidates[:24]:
                    value = _bounded_text(raw, 80)
                    if value and (_context_candidate_allowed is None or _context_candidate_allowed(value)):
                        explicit_terms.append(value)

        terms: list[str] = list(explicit_terms)
        if _extract_candidates is not None:
            terms.extend(_extract_candidates([c for c in chunks if c]))

        transcript_hint_terms: list[str] = []
        if transcript_hint and _extract_candidates is not None:
            # The live HUD text is useful evidence for adjudication, but it is
            # not context truth. Do not promote its extracted tokens into
            # candidate_entities; otherwise a volatile preview misspelling such
            # as "Nobq" can canonicalize/protect the final ASR output.
            transcript_hint_terms = _extract_candidates([_bounded_text(transcript_hint, 360)])[:24]

        existing = list(context.candidate_entities or [])
        seen: set[str] = {e.casefold() for e in existing if isinstance(e, str)}
        added: list[str] = []
        for term in terms:
            if len(existing) + len(added) >= 40:
                break
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            added.append(term)
        if added:
            context.candidate_entities = existing + added
        context.metadata = dict(context.metadata or {})
        if explicit_terms:
            explicit_existing = list(context.metadata.get("explicit_candidate_entities") or [])
            explicit_seen = {str(item).casefold() for item in explicit_existing if str(item).strip()}
            for term in explicit_terms:
                key = term.casefold()
                if key in explicit_seen:
                    continue
                explicit_seen.add(key)
                explicit_existing.append(term)
            context.metadata["explicit_candidate_entities"] = explicit_existing[:40]
        context.metadata["session_context_tape_terms"] = added
        context.metadata["session_context_tape_live_hint_terms"] = transcript_hint_terms
        context.metadata["session_context_tape_snapshot_count"] = len(seen_snapshot_keys)
        return {
            "snapshot_count": len(seen_snapshot_keys),
            "candidate_terms": added,
            "live_hint_candidate_terms": transcript_hint_terms,
        }

    def _new_utterance_id(self) -> str:
        return self.utterance_id_factory()

    def _audio_day_dir(self, root: Path) -> Path:
        now = datetime.now(timezone.utc)
        return root / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"

    def _iter_retained_wavs(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        return [p for p in root.rglob("*.wav") if p.is_file()]

    def _wav_paths_for_utterance(self, utterance_id: str) -> list[Path]:
        """Resolve on-disk paths for *utterance_id* (partitioned tree + legacy flat)."""
        name = f"{utterance_id}.wav"
        found: list[Path] = []
        if self.audio_save_dir is not None:
            root = Path(self.audio_save_dir)
            for p in self._iter_retained_wavs(root):
                if p.name == name:
                    found.append(p)
        try:
            legacy = Path(self.recorder.log_dir) / "audio_rerun" / name
            if legacy.is_file() and legacy not in found:
                found.append(legacy)
        except OSError:
            pass
        return found

    def _save_audio_bounded(self, utterance_id: str, wav_bytes: bytes) -> tuple[str | None, int | None]:
        """Save WAV under ``audio_save_dir/UTC/YYYY/MM/DD/`` and prune.

        Returns ``(path relative to recorder.log_dir, expires_at_ms)`` for
        product history, or ``(None, None)`` when nothing was written.
        """
        save_root = self.audio_save_dir
        if save_root is None:
            return None, None
        save_root = Path(save_root)
        day_dir = self._audio_day_dir(save_root)
        day_dir.mkdir(parents=True, exist_ok=True)
        out = day_dir / f"{utterance_id}.wav"
        out.write_bytes(wav_bytes)
        # Prune: keep only the retention_limit most recent files (whole tree).
        files = sorted(self._iter_retained_wavs(save_root), key=lambda f: f.stat().st_mtime)
        limit = max(1, int(self.audio_retention_limit))
        for old in files[:-limit]:
            try:
                old.unlink()
            except OSError:
                pass
        # Time-based prune (skip when days == 0 → forever / broker "off" handling).
        keep_days = int(self.audio_retention_days)
        if keep_days > 0:
            cutoff = time.time() - (keep_days * 86400)
            for f in self._iter_retained_wavs(save_root):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass
        log_dir = Path(self.recorder.log_dir)
        try:
            rel = out.resolve().relative_to(log_dir.resolve())
            rel_s = rel.as_posix()
        except ValueError:
            rel_s = str(out)
        exp_ms: int | None = None
        if keep_days > 0:
            exp_ms = int(time.time() * 1000) + keep_days * 86400 * 1000
        return rel_s, exp_ms

    def get_audio_for_rerun(self, utterance_id: str) -> bytes | None:
        """Return stored WAV bytes for *utterance_id*, or None if not retained."""
        paths = self._wav_paths_for_utterance(utterance_id)
        if not paths:
            return None
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            return paths[0].read_bytes()
        except OSError:
            return None

    def replay_available(self, utterance_id: str) -> bool:
        """Return True when stored audio is available for rerun."""
        return bool(self._wav_paths_for_utterance(utterance_id))

    def delete_audio_for_rerun(self, utterance_id: str) -> bool:
        """Delete retained WAV for *utterance_id*. Returns True when removed."""
        paths = self._wav_paths_for_utterance(utterance_id)
        if not paths:
            return False
        removed = False
        for path in paths:
            try:
                path.unlink()
                removed = True
            except OSError:
                continue
        return removed


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _normalize_transcript_stage(value: str | None) -> str:
    stage = (value or "final_delivery").strip().lower()
    if stage in {"live", "adjudication", "live_correction"}:
        return "live_adjudication"
    if stage in {"history_reprocess", "reprocess"}:
        return "history_reprocess"
    if stage != "live_adjudication":
        return "final_delivery"
    return stage


@dataclass(frozen=True, slots=True)
class WakeStatus:
    verified: bool
    raw_wake_detected: bool
    normalized_wake_detected: bool
    raw_wake_text: str | None
    post_wake_text: str


def leading_wake_status(raw_text: str, normalized_text: str) -> WakeStatus:
    raw_post = strip_wake(raw_text or "")
    norm_post = strip_wake(normalized_text or "")
    if raw_post is not None:
        return WakeStatus(
            verified=True,
            raw_wake_detected=True,
            normalized_wake_detected=norm_post is not None,
            raw_wake_text=raw_text,
            post_wake_text=raw_post,
        )
    if norm_post is not None:
        return WakeStatus(
            verified=True,
            raw_wake_detected=False,
            normalized_wake_detected=True,
            raw_wake_text=normalized_text,
            post_wake_text=norm_post,
        )
    return WakeStatus(
        verified=False,
        raw_wake_detected=False,
        normalized_wake_detected=False,
        raw_wake_text=None,
        post_wake_text=normalized_text,
    )


def _post_wake_from_adjudicated(adjudicated_text: str) -> str:
    stripped = strip_wake(adjudicated_text or "")
    return (stripped if stripped is not None else adjudicated_text or "").strip()


def _looks_like_action_attempt(text: str) -> bool:
    from juno_v2.turn_plan.actions import native_action_signal_present

    return native_action_signal_present(text)


def _dispatchable_actions_for_pipeline(actions: list[Action] | None) -> tuple[list[Action] | None, list[str]]:
    if not actions:
        return None, []
    accepted: list[Action] = []
    skipped: list[str] = []
    for idx, action in enumerate(actions):
        body = str(getattr(action, "body", "") or "").strip()
        kind = getattr(action, "kind", None)
        if kind is ActionKind.ALARM and getattr(action, "when", None) is None:
            skipped.append(f"action_{idx}_alarm_missing_schedule")
            continue
        if kind in {ActionKind.NOTE, ActionKind.REMINDER} and not body:
            skipped.append(f"action_{idx}_missing_body")
            continue
        accepted.append(action)
    return accepted or None, skipped


@lru_cache(maxsize=1)
def _command_probe_parser():
    from juno_v2.writer.parser import WriterIntentParser

    return WriterIntentParser()


def _is_pure_command_utterance(text: str) -> bool:
    """Short utterance the writer parser claims as a command result."""
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned.split()) > 6:
        return False
    try:
        from juno_v2.contracts.writer import WriterIntentKind

        intent = _command_probe_parser().parse(cleaned)
        return getattr(intent, "kind", None) is WriterIntentKind.COMMAND_RESULT
    except Exception:  # noqa: BLE001 — probe must never break the turn
        return False


def _turn_plan_allows_mixed_paste(plan: dict[str, Any]) -> bool:
    del plan
    return False


def _action_source_from_live_hint(
    final_post_wake_text: str,
    *,
    transcript_hint: str | None,
    wake_verified: bool,
) -> str | None:
    if not wake_verified or not transcript_hint:
        return None
    final_source = str(final_post_wake_text or "").strip()
    if _looks_like_action_attempt(final_source):
        return None
    hint_status = leading_wake_status(transcript_hint, transcript_hint)
    hinted = _post_wake_from_adjudicated(transcript_hint) if hint_status.verified else str(transcript_hint or "").strip()
    if not hinted or not _looks_like_action_attempt(hinted):
        return None
    if not _action_hint_compatible(final_source, hinted):
        return None
    return hinted


def _action_hint_compatible(final_post_wake_text: str, hinted_post_wake_text: str) -> bool:
    final_tokens = _action_source_content_tokens(final_post_wake_text)
    hint_tokens = _action_source_content_tokens(hinted_post_wake_text)
    if not final_tokens or not hint_tokens:
        return False
    overlap = final_tokens.intersection(hint_tokens)
    return len(overlap) >= min(2, len(final_tokens), len(hint_tokens))


_ACTION_SOURCE_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "is",
    "it",
    "juno",
    "me",
    "note",
    "remind",
    "reminder",
    "set",
    "take",
    "that",
    "the",
    "to",
}


def _action_source_content_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text or "").casefold())
        if len(token) >= 3 and token not in _ACTION_SOURCE_TOKEN_STOPWORDS
    }
    return tokens


def _record_transcript_decision(
    recorder: TraceRecorder,
    *,
    utterance_id: str,
    stage: str,
    raw_text: str,
    normalized_text: str,
    adjudicated_text: str,
    writer_text: str,
    context: TypedContextBundle,
    compiled_context: CompiledContext,
    memory_packet: MemoryServingPacket,
    plan: RecognitionBiasPlan,
    mode_selection: ModeSelection,
    mode_policy: ModePolicy,
    language_policy: str,
    requested_language: str | None,
    backend_name: str,
    model_path: str,
    audio_duration_ms: float,
    decode_ms: float,
    tape_meta: dict[str, Any],
    adjudication_result: TranscriptAdjudicationResult | None,
    adjudication_skip_reason: str | None,
    patch_included: bool,
    paste_kind: str,
    noop_reason: str | None,
    parsed_actions_payload: list[dict[str, Any]] | None,
    action_attempt_rejected: bool,
    wake_status: WakeStatus | None,
) -> None:
    try:
        if adjudication_result is None:
            adjudication_payload = {
                "ran": False,
                "rejected": True,
                "rejected_reason": adjudication_skip_reason,
                "corrected_text": None,
                "ops": [],
                "confidence": 0.0,
                "backend": None,
                "decode_ms": 0.0,
                "protected_terms_used": [],
            }
        else:
            adjudication_payload = {
                "ran": True,
                "rejected": bool(getattr(adjudication_result, "rejected", False)),
                "rejected_reason": getattr(adjudication_result, "rejected_reason", None),
                "corrected_text": getattr(adjudication_result, "corrected_text", None),
                "ops": [
                    op.to_dict() if hasattr(op, "to_dict") else dict(op)
                    for op in list(getattr(adjudication_result, "ops", ()) or ())
                ],
                "confidence": float(getattr(adjudication_result, "confidence", 0.0) or 0.0),
                "backend": getattr(adjudication_result, "backend_name", None),
                "decode_ms": float(getattr(adjudication_result, "decode_ms", 0.0) or 0.0),
                "protected_terms_used": list(getattr(adjudication_result, "protected_terms_used", ()) or ()),
            }
        context_terms = [
            term.to_dict() if hasattr(term, "to_dict") else dict(term)
            for term in list(getattr(compiled_context, "terms", ()) or ())[:40]
        ]
        memory_payload = (
            memory_packet.to_dict()
            if hasattr(memory_packet, "to_dict")
            else dict(memory_packet or {})
        )
        context_payload = {
            "app_name": context.app_name,
            "app_category": context.app_category,
            "app_bundle_id": (context.metadata or {}).get("app_bundle_id"),
            "window_title": context.window_title,
            "selected_text": _bounded_text(context.selected_text, 1200),
            "focused_text_before": _bounded_text(context.focused_text_before, 1200),
            "focused_text_after": _bounded_text(context.focused_text_after, 600),
            "field_text_excerpt": _bounded_text(context.field_text_excerpt, 1200),
            "candidate_entities": list(context.candidate_entities[:40]),
            "focused_file_path": context.focused_file_path,
            "symbol_under_cursor": context.symbol_under_cursor,
        }
        wake_payload = None
        if wake_status is not None:
            wake_payload = {
                "verified": wake_status.verified,
                "raw_wake_detected": wake_status.raw_wake_detected,
                "normalized_wake_detected": wake_status.normalized_wake_detected,
                "raw_wake_text": wake_status.raw_wake_text,
            }
        recorder.record(
            TraceKind.SYSTEM,
            "transcript_decision",
            {
                "utterance_id": utterance_id,
                "stage": stage,
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "adjudicated_text": adjudicated_text,
                "writer_text": writer_text,
                "text_lengths": {
                    "raw": len(raw_text or ""),
                    "normalized": len(normalized_text or ""),
                    "adjudicated": len(adjudicated_text or ""),
                    "writer": len(writer_text or ""),
                },
                "policies": {
                    "mode": getattr(mode_selection, "effective_mode", None),
                    "mode_source": str(getattr(getattr(mode_selection, "mode_source", None), "value", getattr(mode_selection, "mode_source", "")) or ""),
                    "transcript_correction_policy": getattr(mode_policy, "transcript_correction_policy", "standard"),
                    "final_formatting_policy": getattr(mode_policy, "final_formatting_policy", "minimal"),
                    "language_policy": language_policy,
                    "requested_language": requested_language,
                },
                "asr": {
                    "backend": backend_name,
                    "model_path": model_path,
                    "audio_duration_ms": audio_duration_ms,
                    "decode_ms": decode_ms,
                    "bias_phrase_count": len(plan.bias_phrases),
                    "bias_phrases": list(plan.bias_phrases[:24]),
                    "has_initial_prompt": bool(plan.initial_prompt),
                },
                "adjudication": adjudication_payload,
                "patch_included": bool(patch_included),
                "patch_op_count": len(adjudication_payload.get("ops") or []) if patch_included else 0,
                "memory_packet": memory_payload,
                "context_terms": context_terms,
                "context": context_payload,
                "session_context_tape": dict(tape_meta or {}),
                "paste": {
                    "paste_kind": paste_kind,
                    "noop_reason": noop_reason,
                },
                "actions": {
                    "wake": wake_payload,
                    "parsed": list(parsed_actions_payload or []),
                    "action_attempt_rejected": bool(action_attempt_rejected),
                },
            },
        )
    except Exception:
        pass


def _turn_plan_result_summary(
    result: Any,
    *,
    validation: dict[str, Any] | None,
    mixed_paste_allowed: bool,
    controls_actions: bool,
) -> dict[str, Any]:
    plan = getattr(result, "plan", None)
    payload: dict[str, Any] = {
        "status": getattr(result, "status", None),
        "backend": getattr(result, "backend_name", None),
        "decode_ms": float(getattr(result, "decode_ms", 0.0) or 0.0),
        "errors": list(getattr(result, "errors", ()) or ()),
        "repair_attempted": bool(getattr(result, "repair_attempted", False)),
        "repair_status": getattr(result, "repair_status", None),
        "initial_status": getattr(result, "initial_status", None),
        "initial_errors": list(getattr(result, "initial_errors", ()) or ()),
        "validation_errors_before_repair": list(
            getattr(result, "validation_errors_before_repair", ()) or ()
        ),
        "validation_warnings_before_repair": list(
            getattr(result, "validation_warnings_before_repair", ()) or ()
        ),
        "normalization_notes": list(getattr(result, "normalization_notes", ()) or ()),
        "raw_output_chars": len(str(getattr(result, "raw_output", "") or "")),
        "validation": dict(validation or {}),
        "mixed_paste_allowed": bool(mixed_paste_allowed),
        "controls_actions": bool(controls_actions),
    }
    if isinstance(plan, dict):
        render = plan.get("render_plan") if isinstance(plan.get("render_plan"), dict) else {}
        safety = plan.get("safety") if isinstance(plan.get("safety"), dict) else {}
        payload.update({
            "utterance_kind": plan.get("utterance_kind"),
            "render_kind": render.get("render_kind"),
            "action_count": len(plan.get("actions") or []) if isinstance(plan.get("actions"), list) else 0,
            "uncertainty_count": len(plan.get("uncertainties") or []) if isinstance(plan.get("uncertainties"), list) else 0,
            "commit_policy": safety.get("commit_policy"),
            "execute_policy": safety.get("execute_policy"),
        })
    return payload


def _should_run_transcript_adjudication(
    *,
    transcript_policy: str,
    live_correction_policy: str = "stable_span_standard",
    app_category: str | None,
    live_adjudication: bool,
    config: TranscriptAdjudicatorConfig,
) -> bool:
    if live_adjudication:
        # The live preview lane honors ``live_correction_policy`` from the
        # active ``ModePolicy``. Modes that explicitly opt out of preview
        # corrections (e.g. ``verbatim`` and ``command_mode`` declare
        # ``live_correction_policy='none'``) suppress live adjudication
        # even when ``transcript_correction_policy`` is permissive.
        if (live_correction_policy or "").strip().lower() == "none":
            return False
        return bool(config.live_enabled) and transcript_policy not in {"none"}
    if transcript_policy == "none":
        return False
    if (app_category or "").strip().lower() == "terminal" and transcript_policy not in {"exact_only"}:
        return False
    return True


def _mode_policy_for_final_delivery(
    mode_policy: "ModePolicy",
    *,
    context: TypedContextBundle,
    raw_text: str,
    adjudicated_text: str,
    adjudication_result: "TranscriptAdjudicationResult | None",
) -> "ModePolicy":
    current_policy = str(getattr(mode_policy, "final_formatting_policy", "") or "minimal")
    if current_policy not in {"", "minimal"}:
        return mode_policy
    cat = (context.app_category or "").strip().lower()
    if cat in {"code", "terminal"}:
        return mode_policy

    requested_policy = _formatting_policy_from_qwen_plan(adjudication_result)
    if requested_policy is None:
        return mode_policy

    prefix = (getattr(mode_policy, "prompt_prefix", "") or "").strip()
    if requested_policy == "explicit_rewrite":
        nudge = (
            "Explicit rewrite request detected: preserve every content unit and apply "
            "only the requested rewrite style."
        )
    else:
        nudge = (
            "Spoken structure detected: preserve every content unit and apply only the "
            "requested list, section, or email structure."
        )
    prompt_prefix = f"{prefix} {nudge}".strip() if prefix else nudge
    return replace(
        mode_policy,
        final_formatting_policy=requested_policy,
        prompt_prefix=prompt_prefix,
    )


def _formatting_policy_from_qwen_plan(
    adjudication_result: "TranscriptAdjudicationResult | None",
) -> str | None:
    if adjudication_result is None or adjudication_result.rejected:
        return None
    metadata = getattr(adjudication_result, "metadata", {}) or {}
    plan = metadata.get("formatting_plan") if isinstance(metadata, dict) else None
    if isinstance(plan, dict):
        values = " ".join(str(v) for v in plan.values() if isinstance(v, (str, int, float, bool))).casefold()
    else:
        values = str(plan or "").casefold()
    if not values:
        return None
    if (
        "no_format" in values
        or "no formatting" in values
        or "verbatim" in values
        or re.search(r"\b(?:no|without|avoid)\s+(?:bullet|bullets|list|numbered|formatting|structure)\b", values)
        or re.search(r"\bdo\s+not\b.{0,48}\b(?:bullet|bullets|list|numbered|format|structure)\b", values)
    ):
        return None
    if "email" in values:
        return "email"
    if any(token in values for token in ("bullet", "number", "section", "list", "heading", "structured")):
        return "structured_notes"
    if any(
        token in values
        for token in (
            "checklist",
            "concise",
            "formal",
            "polish",
            "rewrite",
            "status update",
            "summary",
        )
    ):
        return "explicit_rewrite"
    return None


def _collect_self_correction_cues(text: str) -> tuple[dict[str, Any], ...]:
    current = re.sub(r"\s+", " ", (text or "").strip())
    if not current:
        return ()
    cues: list[dict[str, Any]] = []
    for match in _MID_UTTERANCE_EDIT_MARKER_RE.finditer(current):
        start, end = match.span()
        cues.append({
            "marker": match.group(0).strip(),
            "before_excerpt": current[max(0, start - 96):start].strip(),
            "after_excerpt": current[end:end + 160].strip(" ,.;:!?-"),
            "char_start": start,
            "char_end": end,
        })
        if len(cues) >= 12:
            break
    for match in re.finditer(
        r"(?<![A-Za-z0-9])(?P<correct>[A-Za-z][A-Za-z'-]{2,31})\s+not\s+(?P<wrong>[A-Za-z][A-Za-z'-]{2,31})(?![A-Za-z0-9])",
        current,
        flags=re.IGNORECASE,
    ):
        correct = match.group("correct")
        wrong = match.group("wrong")
        if not _inline_not_correction_allowed(correct, wrong):
            continue
        cues.append({
            "marker": "not",
            "kept_candidate": correct,
            "removed_candidate": wrong,
            "before_excerpt": current[max(0, match.start() - 80):match.start()].strip(),
            "after_excerpt": current[match.end():match.end() + 120].strip(" ,.;:!?-"),
            "char_start": match.start(),
            "char_end": match.end(),
        })
        if len(cues) >= 16:
            break
    return tuple(cues)


def _final_adjudication_fast_skip_reason(
    *,
    live_adjudication: bool,
    transcript_hint: str | None,
    raw_text: str,
    normalized_text: str,
    normalization_applied: list[str],
    audio_duration_ms: float | None = None,
) -> str | None:
    # Final transcript adjudication is a product contract, not an optional
    # latency knob. The preview lane can be volatile, but the final paste must
    # pass through the cleanup/adjudication layer so long dictations and
    # wake-gated action commands resolve corrections, punctuation, and
    # formatting intent before turn planning. Wake verification still happens
    # from raw/normalized ASR and cannot be created by Qwen.
    return None


def _reconcile_proper_nouns_from_live_hint(
    *,
    live_hint: str | None,
    final_text: str,
    protected_terms: tuple[str, ...],
) -> tuple[str, list[dict[str, str]]]:
    del protected_terms
    live_terms = _proper_noun_terms(live_hint or "")
    live_terms = list(dict.fromkeys(t for t in live_terms if _proper_noun_candidate(t)))
    if not live_terms or not final_text:
        return final_text, []

    live_case_by_fold = {term.casefold(): term for term in live_terms}
    replacements: list[dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        folded = token.casefold()
        exact_case = live_case_by_fold.get(folded)
        if exact_case:
            if token != exact_case:
                replacements.append({"from": token, "to": exact_case, "source": "live_hint_case"})
                return exact_case
            return token
        return token

    out = re.sub(r"(?<!\w)[A-Z][A-Za-z]{2,}(?!\w)", repl, final_text)
    return out, replacements


def _explicit_candidate_terms(context: TypedContextBundle) -> tuple[str, ...]:
    raw = (context.metadata or {}).get("explicit_candidate_entities")
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for item in raw[:40]:
        term = str(item or "").strip()
        key = term.casefold()
        if not term or key in seen:
            continue
        seen.add(key)
        out.append(term)
    return tuple(out)


def _dedupe_term_values(values: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        term = str(item or "").strip()
        key = term.casefold()
        if not term or key in seen:
            continue
        seen.add(key)
        out.append(term)
    return tuple(out)


def _context_proper_repair_terms(context: TypedContextBundle) -> tuple[str, ...]:
    chunks: list[str] = [
        context.window_title or "",
        context.selected_text or "",
        context.focused_text_before or "",
        context.focused_text_after or "",
        context.field_text_excerpt or "",
    ]
    chunks.extend(str(item or "") for item in (context.candidate_entities or ())[:40])
    terms: list[str] = []
    for chunk in chunks:
        terms.extend(_proper_noun_terms(chunk))
    return _dedupe_term_values(tuple(terms))


def _reconcile_explicit_candidate_term_confusions(
    *,
    text: str,
    explicit_candidate_terms: tuple[str, ...],
    protected_terms: tuple[str, ...],
) -> tuple[str, list[dict[str, str]]]:
    if not text or not explicit_candidate_terms or not protected_terms:
        return text, []
    protected = {str(term or "").strip().casefold() for term in protected_terms if str(term or "").strip()}
    terms = [
        term.strip()
        for term in explicit_candidate_terms
        if term.strip().casefold() in protected and _phonetic_candidate_term_allowed(term)
    ]
    if not terms:
        return text, []

    out = text
    replacements: list[dict[str, str]] = []
    for alias, replacement in _explicit_candidate_sequence_aliases(terms):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){alias}(?![A-Za-z0-9])", flags=re.IGNORECASE)

        def sequence_repl(match: re.Match[str]) -> str:
            observed = match.group(0)
            replacements.append({"from": observed, "to": replacement, "source": "explicit_candidate_phrase"})
            return _case_phrase_like(observed, replacement)

        out = pattern.sub(sequence_repl, out)

    for term in terms[:16]:
        required_count = 2 if _explicit_candidate_phrase_aliases(term) else 1
        if _count_word_occurrences(out, term) >= required_count:
            continue
        for alias in _explicit_candidate_phrase_aliases(term):
            pattern = re.compile(rf"(?<![A-Za-z0-9]){alias}(?![A-Za-z0-9])", flags=re.IGNORECASE)

            def alias_repl(match: re.Match[str]) -> str:
                observed = match.group(0)
                replacements.append({"from": observed, "to": term, "source": "explicit_candidate_phrase"})
                return _case_like(observed, term)

            out = pattern.sub(alias_repl, out)
        split_text, split_replacements = _reconcile_split_candidate_term(out, term)
        if split_replacements:
            out = split_text
            replacements.extend(split_replacements)
        if _contains_word(out, term):
            continue
        soundex = _soundex(term)
        if not soundex:
            continue

        def repl(match: re.Match[str]) -> str:
            observed = match.group(0)
            if observed.casefold() == term.casefold():
                return observed
            if not _observed_candidate_repair_allowed(observed):
                return observed
            if not _candidate_term_phonetic_confusion(term, observed, soundex):
                return observed
            replacements.append({"from": observed, "to": term, "source": "explicit_candidate"})
            return _case_like(observed, term)

        out = re.sub(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z'-]{1,31}(?![A-Za-z0-9])", repl, out)
    return out, replacements


def _reconcile_split_candidate_term(
    text: str,
    term: str,
    *,
    source: str = "explicit_candidate_split_phrase",
) -> tuple[str, list[dict[str, str]]]:
    target = re.sub(r"[^A-Za-z0-9]+", "", term or "")
    if len(target) < 5 or not _phonetic_candidate_term_allowed(target):
        return text, []
    spans = [
        (match.group(0), match.start(), match.end())
        for match in re.finditer(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z'-]{1,31}(?![A-Za-z0-9])", text or "")
    ]
    edits: list[tuple[int, int, str, dict[str, str]]] = []
    target_folded = target.casefold()
    for width in (3, 2):
        for idx in range(0, len(spans) - width + 1):
            observed_tokens = [item[0] for item in spans[idx : idx + width]]
            if not _split_candidate_tokens_safe_for_term(observed_tokens, target):
                continue
            joined = re.sub(r"[^A-Za-z0-9]+", "", "".join(observed_tokens))
            folded = joined.casefold()
            if not folded or folded[:1] != target_folded[:1]:
                continue
            if folded == target_folded:
                ratio = 1.0
            else:
                if abs(len(folded) - len(target_folded)) > 2:
                    continue
                ratio = difflib.SequenceMatcher(a=folded, b=target_folded, autojunk=False).ratio()
                if ratio < 0.78:
                    continue
            start = spans[idx][1]
            end = spans[idx + width - 1][2]
            observed = " ".join(observed_tokens)
            edits.append(
                (
                    start,
                    end,
                    _case_phrase_like(observed, term),
                    {"from": observed, "to": term, "source": source},
                )
            )
            break
        if edits:
            break
    if not edits:
        return text, []
    out = text
    replacements: list[dict[str, str]] = []
    for start, end, replacement, meta in sorted(edits, key=lambda row: row[0], reverse=True):
        out = out[:start] + replacement + out[end:]
        replacements.append(meta)
    return out, list(reversed(replacements))


_SPLIT_CANDIDATE_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _split_candidate_tokens_safe_for_term(observed_tokens: list[str], target: str) -> bool:
    token_norms = [
        re.sub(r"[^A-Za-z0-9]+", "", token or "").casefold()
        for token in observed_tokens
    ]
    token_norms = [token for token in token_norms if token]
    if len(token_norms) != len(observed_tokens):
        return False
    target_folded = (target or "").casefold()
    if not target_folded:
        return False
    if any(token in _SPLIT_CANDIDATE_FUNCTION_WORDS for token in token_norms):
        return False
    if any(token == target_folded for token in token_norms) and "".join(token_norms) != target_folded:
        return False
    return True


def _observed_candidate_repair_allowed(observed: str) -> bool:
    token = re.sub(r"[^A-Za-z0-9]+", "", observed or "")
    if len(token) <= 2:
        return False
    if token.casefold() in _SPLIT_CANDIDATE_FUNCTION_WORDS:
        return False
    if common_english_single_word(token):
        return False
    return True


def _repair_terminal_protected_command_terms(
    *,
    text: str,
    explicit_candidate_terms: tuple[str, ...],
    protected_terms: tuple[str, ...],
    app_category: str | None,
) -> tuple[str, list[dict[str, str]]]:
    if not text or (app_category or "").strip().lower() not in {"terminal", "code"}:
        return text, []
    protected = {str(term or "").strip().casefold() for term in protected_terms if str(term or "").strip()}
    explicit = {
        str(term or "").strip().casefold()
        for term in explicit_candidate_terms
        if str(term or "").strip()
    }
    if "pytest" not in protected or "pytest" not in explicit:
        return text, []

    pattern = re.compile(
        r"(?<![A-Za-z0-9])python(?:\s*-\s*|\s+dash\s+|\s+)m\s*"
        r"(?:py\s*dest|py\s*test|pi\s*dest|pi\s*test|pydest|pytest|pitest|pidest)"
        r"(?P<tests>\s*tests\s*(?:v\s*)?(?:2|two)|testsv(?:2|two))?",
        flags=re.IGNORECASE,
    )
    replacements: list[dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        observed = match.group(0)
        replacement = "python -m pytest"
        if match.group("tests"):
            replacement += " tests v2"
        replacements.append({
            "from": observed,
            "to": replacement,
            "source": "terminal_command_protected_term",
        })
        return replacement

    out = pattern.sub(repl, text)
    explicit_original = [
        str(term or "").strip()
        for term in explicit_candidate_terms
        if str(term or "").strip().casefold() in protected
    ]
    for term in explicit_original[:16]:
        term_words = re.findall(r"[A-Za-z][A-Za-z0-9]*", term)
        if len(term_words) != 2:
            continue
        first, second = term_words
        canonical = " ".join(term_words)
        sep_pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(first)}\s+"
            rf"(?:underscore|dash|hyphen)\s+{re.escape(second)}(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )

        def sep_repl(match: re.Match[str]) -> str:
            observed = match.group(0)
            replacements.append({
                "from": observed,
                "to": canonical,
                "source": "terminal_protected_phrase_separator",
            })
            return canonical

        out = sep_pattern.sub(sep_repl, out)
        glued = f"{first}{second}"
        glue_pattern = re.compile(re.escape(glued), flags=re.IGNORECASE)

        def glue_repl(match: re.Match[str]) -> str:
            observed = match.group(0)
            replacements.append({
                "from": observed,
                "to": canonical,
                "source": "terminal_protected_phrase_glued",
            })
            return f" {canonical} "

        out = glue_pattern.sub(glue_repl, out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out).strip()
    return out, replacements


def _reconcile_protected_term_near_misses(
    *,
    text: str,
    protected_terms: tuple[str, ...],
    ai_glossary: frozenset[str] = AI_GLOSSARY,
) -> tuple[str, list[dict[str, str]]]:
    """Repair near-misses against protected terms and the AI/ML glossary.

    The static glossary is not gated by app category. Instead, static glossary
    candidates use a stricter observed-word check so ordinary real words are not
    rewritten just because a domain term exists. User/context protected terms
    remain stronger evidence and can repair in any app surface.
    """

    if not text:
        return text, []

    candidates: list[tuple[str, str, bool]] = []
    seen: set[str] = set()

    def _add(term: str, *, source: str, allow_common_target: bool = False) -> None:
        token = (term or "").strip()
        if not token or not _protected_near_miss_term_allowed(token, allow_common_target=allow_common_target):
            return
        key = f"{source}:{token.casefold()}"
        if key in seen:
            return
        seen.add(key)
        candidates.append((token, source, allow_common_target))

    ai_glossary_keys = {term.casefold() for term in ai_glossary}
    for term in protected_terms[:32]:
        _add(term, source="protected", allow_common_target=str(term or "").casefold() in ai_glossary_keys)
        for token in _protected_phrase_repair_tokens(term):
            _add(token, source="protected", allow_common_target=True)
    for term in ai_glossary:
        _add(term, source="ai_glossary", allow_common_target=True)

    if not candidates:
        return text, []

    # Single pass over the text. For each token, evaluate ALL candidates and
    # pick the highest-ratio match — this avoids iteration-order bugs (e.g.
    # "Mixxtral" matching the lower-similarity "Mistral" before "Mixtral" had
    # a chance) and lets the targets be a frozenset (unordered) safely.
    #
    # Tokens that already match a candidate verbatim are short-circuited by
    # ``_protected_term_near_miss`` itself, which returns False when observed
    # casefolds to a candidate. So we don't need a per-term "already present"
    # pre-skip; near-miss repair should still work when the correct term
    # appears elsewhere in the same text.

    replacements: list[dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        observed = match.group(0)
        best_target: str | None = None
        best_ratio = 0.0
        for term, source, _allow_common_target in candidates:
            if not _protected_term_near_miss(
                term,
                observed,
                static_glossary=source == "ai_glossary",
            ):
                continue
            ratio = difflib.SequenceMatcher(
                a=observed.casefold(), b=term.casefold(), autojunk=False
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_target = term
        if best_target is None:
            return observed
        replacements.append(
            {"from": observed, "to": best_target, "source": "protected_term_near_miss"}
        )
        return best_target

    out = re.sub(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z'-]{2,31}(?![A-Za-z0-9])", repl, text)
    for term, source, _allow_common_target in candidates[:32]:
        if source == "ai_glossary":
            continue
        split_text, split_replacements = _reconcile_split_candidate_term(
            out,
            term,
            source="protected_term_split_phrase",
        )
        if split_replacements:
            out = split_text
            replacements.extend(split_replacements)
    return out, replacements


def _protected_phrase_repair_tokens(term: str) -> tuple[str, ...]:
    """Rare-looking tokens inside protected phrases can repair ASR spelling.

    A screen/memory phrase should help final ASR repair a rare token inside
    that phrase without making the whole utterance deterministic. Short phrase
    fragments are intentionally ignored so ordinary words do not get rewritten
    to short protected-token fragments.
    """

    raw = str(term or "").strip()
    if not raw or " " not in raw:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,31}", raw):
        if len(token) < 5:
            continue
        if token.casefold() in _COMMON_PHONETIC_REPAIR_WORDS:
            continue
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return tuple(out)


def _protected_near_miss_term_allowed(term: str, *, allow_common_target: bool = False) -> bool:
    """Gate for what tokens may serve as a near-miss reconciliation target.

    The rule is intentionally generic: any token that is long enough, shaped
    like an identifier, and not in the common-English filter qualifies. Token
    *shape* (uppercase, digits, camelCase, q-not-u) is NOT a discriminator —
    eligibility comes from being on the user's protected list or in the AI
    glossary, both checked at the call site.
    """

    token = (term or "").strip()
    if len(token) < 3 or len(token) > 32:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9'-]{2,31}", token):
        return False
    if token.casefold() in _COMMON_PHONETIC_REPAIR_WORDS:
        return False
    if not allow_common_target and common_english_single_word(token) and not _single_token_has_identifier_shape(token):
        return False
    return True


def _single_token_has_identifier_shape(value: str) -> bool:
    token = str(value or "")
    return bool(
        re.fullmatch(r"[A-Z0-9]{2,}(?:[._-][A-Z0-9]{2,})*", token)
        or re.search(r"[a-z][A-Z]", token)
        or (any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token))
    )


def _single_letter_inflection_pair(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are the same word ± a trailing inflection
    suffix ("action"/"actions", "branch"/"branches")."""
    for x, y in ((a, b), (b, a)):
        if x == y + "s" or x == y + "es":
            return True
    return False


def _protected_term_near_miss(term: str, observed: str, *, static_glossary: bool = False) -> bool:
    """True if ``observed`` is a near-miss for ``term``.

    Constraints (all must hold):
    - Casefolded observed != target (no identity match).
    - Observed not in the common-English filter.
    - Static glossary repairs also reject observed tokens that are real/common
      English words; explicit protected context is stronger evidence and can
      still repair those.
    - Same first letter as target.
    - Length differs by at most 1 character. This is tight on purpose: with
      diff ≤ 2 a 3-letter glossary term like "MoE" matches a 5-letter common
      word like "moves" (LCS m,o,e → ratio 0.75 ≥ 0.74). Real near-misses
      from ASR are insert/delete/substitute of one character, which all stay
      within ±1 length.
    - SequenceMatcher ratio ≥ 0.74.
    """

    obs = (observed or "").strip()
    target = (term or "").strip()
    if not obs or obs.casefold() == target.casefold():
        return False
    obs_folded = obs.casefold()
    target_folded = target.casefold()
    if obs_folded in _COMMON_PHONETIC_REPAIR_WORDS:
        return False
    if static_glossary and common_english_single_word(obs_folded):
        return False
    if (
        common_english_single_word(obs_folded)
        and _single_letter_inflection_pair(obs_folded, target_folded)
        and not _single_token_has_identifier_shape(target)
    ):
        # A common word and its own plural/singular are not an ASR
        # near-miss — they are the same word inflected, and "repairing"
        # one into the other rewrites the user's grammar. Screen-term
        # phrase tokens made Juno's own sidebar label eligible here and
        # "take a note, action items…" became "Actions items…" — which
        # then broke turn-plan span grounding for the note body
        # (production 2026-06-11). Distinct words that merely look alike
        # ("gamma" → protected "Gemma") still repair.
        return False
    if (
        len(obs_folded) > len(target_folded)
        and obs_folded.startswith(target_folded)
        and not _single_token_has_identifier_shape(target)
    ):
        return False
    if obs_folded[:1] != target_folded[:1] or abs(len(obs_folded) - len(target_folded)) > 1:
        return False
    return difflib.SequenceMatcher(a=obs_folded, b=target_folded, autojunk=False).ratio() >= 0.74


def _canonicalize_session_entities_against_memory(
    entities: list[str],
    snapshot: MemorySnapshot,
) -> list[str]:
    if not entities:
        return []
    rare_terms = _rare_memory_subterms(snapshot)
    if not rare_terms:
        return entities
    out: list[str] = []
    for entity in entities:
        token = (entity or "").strip()
        replacement = None
        for term in rare_terms:
            if _protected_term_near_miss(term, token):
                replacement = term
                break
        out.append(replacement or token)
    return out


def _commit_session_entity_allowed(
    token: str,
    *,
    committed_text: str,
    spoken_evidence_text: str,
    context: TypedContextBundle | None,
    context_backed: bool = False,
) -> bool:
    context_text = ""
    if context is not None:
        context_text = " ".join(
            part
            for part in (
                context.selected_text,
                context.focused_text_before,
                context.focused_text_after,
                context.field_text_excerpt,
                context.window_title,
            )
            if part
        )
    return commit_session_entity_allowed(
        token,
        committed_text=committed_text,
        spoken_evidence_text=spoken_evidence_text,
        context_text=context_text,
        context_backed=context_backed,
    )


def _rare_memory_subterms(snapshot: MemorySnapshot) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for entry in snapshot.lexicon:
        surfaces = [
            str(getattr(entry, "canonical_form", "") or ""),
            str(getattr(entry, "term", "") or ""),
            *(str(alias or "") for alias in getattr(entry, "aliases", ()) or ()),
        ]
        for surface in surfaces:
            for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,31}", surface):
                if not _protected_near_miss_term_allowed(token):
                    continue
                key = token.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(token)
    return tuple(out)


def _explicit_candidate_phrase_aliases(term: str) -> tuple[str, ...]:
    return ()


def _explicit_candidate_sequence_aliases(terms: list[str]) -> tuple[tuple[str, str], ...]:
    del terms
    return ()


def _case_phrase_like(observed: str, replacement: str) -> str:
    words = replacement.split()
    if not words:
        return replacement
    if observed.isupper():
        return replacement.upper()
    if observed[:1].isupper():
        return " ".join(word[:1].upper() + word[1:] for word in words)
    return replacement


_COMMON_PHONETIC_REPAIR_WORDS = {
    "about", "after", "again", "all", "also", "and", "are", "ask", "back",
    "bar", "be", "because", "before", "being", "bird", "birds", "but", "call",
    "can", "case", "chat", "check", "clear", "clearly", "clause", "code",
    "come", "could", "did", "different", "do", "does", "done", "end", "every",
    "final", "find", "fix", "for", "format", "formatting", "from", "get",
    "give", "go", "going", "good", "got", "had", "has", "have", "here", "how",
    "into", "issue", "issues", "just", "know", "last", "later", "launch",
    "like", "line", "live", "location", "long", "look", "lot", "love",
    "make", "many", "may", "meeting", "memory", "metric", "miss", "missed",
    "missing", "more", "move", "moves", "moving", "not", "notes", "now",
    "one", "only", "open", "our", "out", "over", "owner", "pause", "plan",
    "point", "points", "preview", "problem", "project", "prompt", "pull",
    "put", "read", "really", "review", "right", "risk", "rollout", "say",
    "see", "send", "section", "sections", "should", "show", "slow", "speak",
    "start", "stay", "take", "tech", "tell",
    "text", "than", "that", "the", "then", "there", "thing", "things", "this",
    "time", "to", "use", "utterance", "very", "want", "was", "we", "word",
    "words", "work", "what", "when", "where", "which", "while", "will", "with",
    "would", "wrong", "you",
    # Common q-words — kept here (not in any per-term negative list) so the
    # near-miss matcher rejects them uniformly regardless of target term.
    "queen", "query", "quick", "quiet", "quite", "quote", "quoth",
}


def _phonetic_candidate_term_allowed(term: str) -> bool:
    token = (term or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]{2,31}", token):
        return False
    if token.casefold() in _COMMON_PHONETIC_REPAIR_WORDS:
        return False
    return True


def _candidate_term_phonetic_confusion(term: str, observed: str, soundex: str) -> bool:
    obs = (observed or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]{1,31}", obs):
        return False
    obs_folded = obs.casefold()
    if obs_folded in _COMMON_PHONETIC_REPAIR_WORDS:
        return False
    term_folded = (term or "").casefold()
    if obs_folded == term_folded:
        return False
    if obs_folded[:1] != term_folded[:1]:
        return False
    if abs(len(obs_folded) - len(term_folded)) > 2:
        return False
    return _soundex(obs_folded) == soundex


def _contains_word(text: str, term: str) -> bool:
    return _count_word_occurrences(text, term) > 0


def _count_word_occurrences(text: str, term: str) -> int:
    token = (term or "").strip()
    if not token:
        return 0
    return len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text or "", flags=re.IGNORECASE))


def _case_like(observed: str, replacement: str) -> str:
    if observed.isupper():
        return replacement.upper()
    if observed[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _soundex(value: str) -> str:
    letters = re.findall(r"[A-Za-z]", value or "")
    if not letters:
        return ""
    first = letters[0].upper()
    codes = {
        **dict.fromkeys("BFPVbfpv", "1"),
        **dict.fromkeys("CGJKQSXZcgjkqsxz", "2"),
        **dict.fromkeys("DTdt", "3"),
        **dict.fromkeys("Ll", "4"),
        **dict.fromkeys("MNmn", "5"),
        **dict.fromkeys("Rr", "6"),
    }
    out: list[str] = []
    previous = codes.get(letters[0], "")
    for ch in letters[1:]:
        code = codes.get(ch, "")
        if code and code != previous:
            out.append(code)
        previous = code
    return (first + "".join(out) + "000")[:4]


def _proper_noun_terms(text: str) -> list[str]:
    out: list[str] = []
    for match in re.finditer(r"(?<!\w)(?:[A-Z][A-Za-z]{2,}|[A-Za-z]+[A-Z][A-Za-z]*)(?!\w)", text or ""):
        token = match.group(0)
        if _proper_noun_candidate(token):
            out.append(token)
    return out


_LOW_SIGNAL_PROPER_NOUNS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "he", "here", "hey",
    "how", "i", "if", "in", "is", "it", "its", "me", "my", "no",
    "not", "now", "of", "on", "or", "our", "she", "that", "the",
    "then", "there", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "who", "why", "with", "yes", "you",
    "also", "app", "applications", "bunch", "clear", "clearly", "code",
    "create", "design", "every", "formatting", "get", "go", "just",
    "like", "missing", "pull", "request", "see", "similarly", "things",
    "which", "would",
}


def _proper_noun_candidate(value: str) -> bool:
    token = (value or "").strip()
    if len(token) < 3 or len(token) > 48:
        return False
    if token.casefold() in _LOW_SIGNAL_PROPER_NOUNS:
        return False
    if _looks_like_glued_pronoun_i(token):
        return False
    return any(ch.isalpha() for ch in token)


def _looks_like_glued_pronoun_i(value: str) -> bool:
    return bool(re.match(r"^[a-z]{2,}I(?:m|d|ll|ve|re)?$", (value or "").strip()))


def _near_proper_noun_spelling(reference: str, observed: str) -> bool:
    a = (reference or "").casefold()
    b = (observed or "").casefold()
    if not a or not b or a == b:
        return False
    if a[:1] != b[:1] or abs(len(a) - len(b)) > 3:
        return False
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio() >= 0.76


_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ALPHA_SEQUENCE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{2,12}|(?:[A-Z]\.){1,11}[A-Z]?\.?)(?![A-Za-z0-9])"
)


def _repair_alphabet_sequence_runs(text: str) -> TranscriptEditResult:
    current = text or ""
    matches = [
        {
            "start": m.start(),
            "end": m.end(),
            "text": m.group(0),
            "letters": re.sub(r"[^A-Z]", "", m.group(0)),
        }
        for m in _ALPHA_SEQUENCE_TOKEN_RE.finditer(current)
    ]
    matches = [m for m in matches if 2 <= len(str(m["letters"])) <= 12]
    if not matches:
        return TranscriptEditResult(text=current, changed=False)

    groups: list[list[dict[str, Any]]] = []
    group: list[dict[str, Any]] = []
    for item in matches:
        if not group:
            group = [item]
            continue
        between = current[int(group[-1]["end"]): int(item["start"])]
        if re.fullmatch(r"[\s,;:.-]+", between or ""):
            group.append(item)
        else:
            groups.append(group)
            group = [item]
    if group:
        groups.append(group)

    replacements: list[tuple[int, int, str, dict[str, Any]]] = []
    for group in groups:
        observed = "".join(str(item["letters"]) for item in group)
        if len(observed) < 8 or len(observed) > len(_ALPHABET):
            continue
        best_expected = ""
        best_mismatches = len(observed) + 1
        for start in range(0, len(_ALPHABET) - len(observed) + 1):
            expected = _ALPHABET[start: start + len(observed)]
            mismatches = sum(1 for a, b in zip(observed, expected) if a != b)
            if mismatches < best_mismatches:
                best_expected = expected
                best_mismatches = mismatches
        if not best_expected or best_mismatches > 2:
            continue
        if (len(observed) - best_mismatches) / max(1, len(observed)) < 0.85:
            continue

        offset = 0
        for item in group:
            letters = str(item["letters"])
            expected_part = best_expected[offset: offset + len(letters)]
            offset += len(letters)
            if letters == expected_part:
                continue
            replacement = " ".join(expected_part)
            replacements.append((
                int(item["start"]),
                int(item["end"]),
                replacement,
                {
                    "from": item["text"],
                    "to": replacement,
                    "expected_compact": expected_part,
                    "source": "alphabet_sequence",
                },
            ))

    if not replacements:
        return TranscriptEditResult(text=current, changed=False)

    out = current
    edits: list[dict[str, Any]] = []
    for start, end, replacement, edit in sorted(replacements, key=lambda row: row[0], reverse=True):
        out = out[:start] + replacement + out[end:]
        edits.append(edit)
    out = re.sub(r"\s+", " ", out).strip()
    return TranscriptEditResult(text=out, changed=out != re.sub(r"\s+", " ", current).strip(), edits=tuple(reversed(edits)))


# Single source of truth for spoken edit markers lives in
# juno_core_v3.dictation.self_corrections (shared with the deterministic
# retake application pass).
_MID_UTTERANCE_EDIT_MARKER_RE = MID_UTTERANCE_EDIT_MARKER_RE


def _apply_mid_utterance_edits(text: str) -> TranscriptEditResult:
    current = re.sub(r"\s+", " ", (text or "").strip())
    if not current:
        return TranscriptEditResult(text="", changed=False)
    edits: list[dict[str, Any]] = []
    for _ in range(8):
        match = _MID_UTTERANCE_EDIT_MARKER_RE.search(current)
        if match is None:
            break
        left = current[: match.start()].rstrip()
        right = current[match.end() :].lstrip(" ,.;:!?-")
        if _looks_like_literal_scratch_that_usage(left, right):
            break
        kept_left, removed = _apply_scratch_that_to_left(left, right)
        next_text = _join_transcript_parts(kept_left, right)
        edits.append(
            {
                "marker": match.group(0).strip(),
                "removed": removed,
                "following_words": len(_word_spans_for_edit(right)),
            }
        )
        if next_text == current:
            break
        current = next_text
    inline_result = _apply_inline_not_corrections(current)
    if inline_result.changed:
        current = inline_result.text
        edits.extend(inline_result.edits)
    current = re.sub(r"\s+", " ", current).strip()
    return TranscriptEditResult(
        text=current,
        changed=current != re.sub(r"\s+", " ", (text or "").strip()),
        edits=tuple(edits),
    )


_INLINE_NOT_CORRECTION_BLOCKLIST = {
    "can", "could", "did", "does", "had", "has", "have", "may", "might",
    "must", "shall", "should", "was", "were", "will", "would",
}
_INLINE_NOT_CORRECTION_STOPWORDS = {
    "and", "are", "but", "for", "from", "not", "now", "our", "that", "the",
    "then", "there", "this", "with", "you",
}


def _apply_inline_not_corrections(text: str) -> TranscriptEditResult:
    current = re.sub(r"\s+", " ", (text or "").strip())
    if not current:
        return TranscriptEditResult(text="", changed=False)
    edits: list[dict[str, Any]] = []

    def repl(match: re.Match[str]) -> str:
        correct = match.group("correct")
        wrong = match.group("wrong")
        if correct.strip().casefold() == wrong.strip().casefold():
            edits.append({"marker": "not", "kept": correct, "removed": wrong, "reason": "redundant_same_word"})
            return correct
        if not _inline_not_correction_allowed(correct, wrong):
            return match.group(0)
        edits.append({"marker": "not", "kept": correct, "removed": wrong})
        return correct

    out = re.sub(
        r"(?<![A-Za-z0-9])(?P<correct>[A-Za-z][A-Za-z'-]{2,31})\s+not\s+(?P<wrong>[A-Za-z][A-Za-z'-]{2,31})(?![A-Za-z0-9])",
        repl,
        current,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s+", " ", out).strip()
    return TranscriptEditResult(text=out, changed=out != current, edits=tuple(edits))


def _inline_not_correction_allowed(correct: str, wrong: str) -> bool:
    left = (correct or "").strip().casefold()
    right = (wrong or "").strip().casefold()
    if not left or not right or left == right:
        return False
    if left in _INLINE_NOT_CORRECTION_BLOCKLIST or right in _INLINE_NOT_CORRECTION_BLOCKLIST:
        return False
    if left in _INLINE_NOT_CORRECTION_STOPWORDS or right in _INLINE_NOT_CORRECTION_STOPWORDS:
        return False
    if abs(len(left) - len(right)) > 2:
        return False
    ratio = difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()
    return ratio >= 0.50 or _soundex(left) == _soundex(right)


def _collapse_adjacent_duplicate_phrases(text: str) -> TranscriptEditResult:
    """Remove exact adjacent phrase repeats introduced by final adjudication."""
    current = text or ""
    edits: list[dict[str, Any]] = []
    for _ in range(8):
        spans = _word_spans_for_edit(current)
        removal: tuple[int, int, str] | None = None
        for width in range(12, 3, -1):
            if len(spans) < width * 2:
                continue
            for idx in range(0, len(spans) - (width * 2) + 1):
                left = [span[0] for span in spans[idx : idx + width]]
                right = [span[0] for span in spans[idx + width : idx + (width * 2)]]
                if left != right or not _meaningful_duplicate_phrase(left):
                    continue
                start = spans[idx + width][1]
                end = _extend_duplicate_phrase_removal(current, spans[idx + (width * 2) - 1][2])
                removal = (start, end, " ".join(left))
                break
            if removal is not None:
                break
        if removal is None:
            break
        start, end, phrase = removal
        current = _join_transcript_parts(current[:start].rstrip(), current[end:].lstrip())
        edits.append({"removed": phrase, "words": len(phrase.split())})
    original = re.sub(r"\s+", " ", (text or "").strip())
    return TranscriptEditResult(
        text=current,
        changed=re.sub(r"\s+", " ", current.strip()) != original,
        edits=tuple(edits),
    )


def _meaningful_duplicate_phrase(tokens: list[str]) -> bool:
    return any(len(token) >= 3 and token not in {"and", "then", "the", "this", "that"} for token in tokens)


def _extend_duplicate_phrase_removal(text: str, end: int) -> int:
    idx = end
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx < len(text) and text[idx] in ".!?":
        idx += 1
        while idx < len(text) and text[idx].isspace():
            idx += 1
    return idx


def _apply_scratch_that_to_left(left: str, right: str) -> tuple[str, str]:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return "", ""
    left_spans = _word_spans_for_edit(left)
    right_spans = _word_spans_for_edit(right)
    if not left_spans:
        return "", left
    if not right_spans:
        cut = _previous_edit_unit_start(left, left_spans)
        return left[:cut].rstrip(), left[cut:].strip()

    left_norm = [span[0] for span in left_spans]
    right_norm = [span[0] for span in right_spans]
    prefix_start = _right_prefix_start_in_left(left_norm, right_norm)
    if prefix_start is not None:
        cut = left_spans[prefix_start][1]
        return left[:cut].rstrip(), left[cut:].strip()

    last_left = left_norm[-1]
    early_right = right_norm[:4]
    if last_left in early_right or len(right_norm) == 1 or right_norm[0] in {"no", "not", "actually", "rather", "instead"}:
        cut = left_spans[-1][1]
        return left[:cut].rstrip(), left[cut:].strip()
    ordinal_cut = _ordinal_clause_wrong_phrase_start(left, left_spans)
    if ordinal_cut is not None:
        return left[:ordinal_cut].rstrip(), left[ordinal_cut:].strip()

    cut = _previous_edit_unit_start(left, left_spans)
    return left[:cut].rstrip(), left[cut:].strip()


def _looks_like_literal_scratch_that_usage(left: str, right: str) -> bool:
    left_spans = _word_spans_for_edit(left)
    right_spans = _word_spans_for_edit(right)
    if not left_spans or not right_spans:
        return False
    left_last = left_spans[-1][0]
    right_norm = [span[0] for span in right_spans]
    if (
        len(right_norm) == 1
        and left_last in {"to", "of", "on", "in", "at", "for", "with"}
        and right_norm[0] not in {"no", "not", "actually", "rather", "instead"}
    ):
        return True
    return False


def _right_prefix_start_in_left(left_norm: list[str], right_norm: list[str]) -> int | None:
    max_prefix = min(10, len(left_norm), len(right_norm))
    for count in range(max_prefix, 1, -1):
        target = right_norm[:count]
        for start in range(0, len(left_norm) - count + 1):
            if left_norm[start : start + count] == target:
                return start
    return None


def _previous_edit_unit_start(left: str, spans: list[tuple[str, int, int]]) -> int:
    boundary = max(left.rfind("."), left.rfind("!"), left.rfind("?"), left.rfind(";"), left.rfind(":"))
    if boundary >= 0:
        return min(len(left), boundary + 1)
    if len(spans) <= 12:
        return 0
    return spans[-6][1]


_SPOKEN_LIST_CUE_RE = re.compile(
    r"\b(?:"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"firstly|secondly|thirdly|fourthly|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"1st|2nd|3rd|[4-9]th|10th"
    r")\b",
    re.IGNORECASE,
)


def _ordinal_clause_wrong_phrase_start(left: str, spans: list[tuple[str, int, int]]) -> int | None:
    unit_start = _previous_edit_unit_start(left, spans)
    unit = left[unit_start:].strip()
    matches = list(_SPOKEN_LIST_CUE_RE.finditer(unit))
    if not matches:
        return None
    cue_start = unit_start + matches[-1].start()
    comma = left.rfind(",", cue_start)
    if comma >= cue_start:
        return comma + 1
    return spans[-1][1]


def _word_spans_for_edit(value: str) -> list[tuple[str, int, int]]:
    return [
        ("".join(ch.casefold() for ch in match.group(0) if ch.isalnum()), match.start(), match.end())
        for match in re.finditer(r"\S+", value or "")
        if "".join(ch.casefold() for ch in match.group(0) if ch.isalnum())
    ]


def _join_transcript_parts(left: str, right: str) -> str:
    left = re.sub(r"\s+", " ", (left or "").strip())
    right = re.sub(r"\s+", " ", (right or "").strip())
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"


def _compact_for_adjudication_skip(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _normalization_applied_blocks_final_skip(applied: list[dict[str, Any]]) -> bool:
    non_blocking_rules = {"mid_utterance_edit"}
    for item in applied:
        if not isinstance(item, dict):
            return True
        rule = str(item.get("rule") or "").strip()
        if rule not in non_blocking_rules:
            return True
    return False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _fallback_adjudicated_text(
    *,
    raw_text: str,
    normalized_text: str,
    memory_candidate_text: str,
) -> str:
    return (memory_candidate_text or normalized_text or raw_text or "").strip()


_UNCONFIRMED_STOCK_TAIL_PHRASES = (
    "thank you for watching",
    "thanks for watching",
    "thank you so much",
    "thank you very much",
    "thank you",
    "thanks",
)


def _strip_unconfirmed_stock_tail(
    text: str,
    *,
    transcript_hint: Any,
    audio_duration_ms: float | None,
) -> tuple[str, str | None]:
    """Strip a stock ASR tail only when live preview did not corroborate it."""

    raw = (text or "").strip()
    hint = str(transcript_hint or "").strip()
    if not raw or not hint:
        return raw, None
    try:
        duration = float(audio_duration_ms or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration < 20_000.0:
        return raw, None
    if len(re.findall(r"[A-Za-z0-9]+", raw)) < 10:
        return raw, None

    hint_tail = " ".join(re.findall(r"[A-Za-z0-9]+", hint.casefold())[-40:])
    for phrase in _UNCONFIRMED_STOCK_TAIL_PHRASES:
        phrase_tokens = " ".join(re.findall(r"[A-Za-z0-9]+", phrase.casefold()))
        if phrase_tokens and phrase_tokens in hint_tail:
            continue
        pattern = r"(?:[\s,.;:!?-]+|^)" + r"[\W_]+".join(
            re.escape(token) for token in phrase_tokens.split()
        ) + r"[\s,.;:!?-]*$"
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match is None:
            continue
        cleaned = raw[: match.start()].rstrip(" \t\r\n,.;:!?-")
        if len(re.findall(r"[A-Za-z0-9]+", cleaned)) < 6:
            return raw, None
        return cleaned, phrase
    return raw, None


def _usable_transcript_hint_fallback(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not any(ch.isalnum() for ch in text):
        return ""
    placeholder = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if placeholder in {
        "listening",
        "still listening",
        "transcribing",
        "refining",
        "thinking",
        "speech recognition status unknown",
    }:
        return ""
    if placeholder.startswith("on device live captions unavailable"):
        return ""
    if placeholder.startswith("speech recognition is off"):
        return ""
    if looks_like_hallucination(text):
        return ""
    return text


def _bounded_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def _text_debug_payload(value: Any, *, max_preview_chars: int = 240) -> dict[str, Any]:
    text = str(value or "")
    return {
        "preview": text[:max_preview_chars],
        "chars": len(text),
    }


def _context_debug_payload(context: TypedContextBundle) -> dict[str, Any]:
    return {
        "app_name": context.app_name,
        "app_category": context.app_category,
        "app_bundle_id": (context.metadata or {}).get("app_bundle_id"),
        "window_title": context.window_title,
        "selected_text": _text_debug_payload(context.selected_text),
        "focused_text_before": _text_debug_payload(context.focused_text_before),
        "focused_text_after": _text_debug_payload(context.focused_text_after),
        "field_text_excerpt": _text_debug_payload(context.field_text_excerpt),
        "candidate_entities": list(context.candidate_entities),
        "focused_file_path": context.focused_file_path,
        "symbol_under_cursor": context.symbol_under_cursor,
    }


def _memory_snapshot_counts(snapshot: MemorySnapshot) -> dict[str, int]:
    return {
        "lexicon": len(snapshot.lexicon),
        "replacements": len(snapshot.replacements),
        "corrections": len(snapshot.corrections),
        "session_entities": len(snapshot.session_entities),
    }


def _looks_structurally_rewritten(literal: str, committed: str) -> bool:
    a = (literal or "").strip()
    b = (committed or "").strip()
    if not a or not b:
        return False
    if "\n" in b and "\n" not in a:
        return True
    structural_markers = ("- ", "* ", "1. ", "2. ", "Subject:", "Dear ", "Hi ", "Summary:")
    if any(marker in b for marker in structural_markers) and not any(marker in a for marker in structural_markers):
        return True
    aw = max(1, len(a.split()))
    bw = max(1, len(b.split()))
    ratio = bw / aw
    return ratio < 0.55 or ratio > 1.85


def _should_learn_from_oneshot_record(
    writer_outcome: WriterOutcome | None,
    literal_text: str,
    final_text: str,
) -> bool:
    """Return whether raw/literal -> committed is safe correction evidence."""
    if writer_outcome is None:
        return not _looks_structurally_rewritten(literal_text, final_text)
    if writer_outcome.action != WriterActionKind.PASS_THROUGH_COMMIT:
        return False
    if writer_outcome.learn_from_commit is False:
        return False
    return not _looks_structurally_rewritten(literal_text, final_text)

def _default_utterance_id_factory() -> str:
    import uuid

    return f"oneshot_{uuid.uuid4().hex[:12]}"


def _anchor_from_context(context: TypedContextBundle) -> ClientSelection | None:
    """Best-effort anchor selection for the writer.

    On the one-shot path we don't have a live workbench buffer the way
    the streaming session does. If the context snapshot carries a
    non-empty selection, synthesize a zero-based anchor over
    ``focused_text_before`` so the writer's selected-text transform
    branch can still fire when the user said "make it polite".
    """
    if not context.selected_text:
        return None
    start = len(context.focused_text_before or "")
    end = start + len(context.selected_text)
    return ClientSelection(start=start, end=end)


def _writer_to_surface_text(
    outcome: WriterOutcome | None, *, fallback: str
) -> tuple[str, str | None, bool, bool]:
    """Translate a :class:`WriterOutcome` into surface-level text.

    Returns ``(surface_text, action_name, deterministic_used, memory_updated)``.
    """
    if outcome is None:
        return fallback, None, False, False
    action = outcome.action
    if action == WriterActionKind.PASS_THROUGH_COMMIT:
        return outcome.output_text or fallback, action.value, outcome.deterministic_used, outcome.memory_updated
    if action in (WriterActionKind.DIRECT_COMMIT, WriterActionKind.TRANSFORM_COMMIT):
        return outcome.output_text or "", action.value, outcome.deterministic_used, outcome.memory_updated
    # NOOP / STATE_MUTATION / MODE_SWITCH / MEMORY_MUTATION -> surface pastes nothing
    return "", action.value, outcome.deterministic_used, outcome.memory_updated


_REDACTION_SENTINEL_RE = re.compile(r"<(?:url|email|phone|address|selection|redacted|private)>", re.I)


def _unsafe_writer_surface_reason(
    writer_text: str,
    *,
    fallback_text: str,
    raw_text: str,
    writer_outcome: WriterOutcome | None,
) -> str | None:
    """Return why writer output must not be pasted, or None when safe.

    The writer may legitimately transform selected text, so this guard is only
    for cases where a long dictated transcript was collapsed into context
    sentinels or a tiny unrelated fragment before final paste.
    """

    surface = (writer_text or "").strip()
    fallback = (fallback_text or "").strip()
    raw = (raw_text or "").strip()
    if not surface or not fallback:
        return None

    if _REDACTION_SENTINEL_RE.search(surface) and not _REDACTION_SENTINEL_RE.search(raw):
        return "redaction_sentinel_output"

    fallback_words = _content_word_list(fallback)
    surface_words = _content_word_list(surface)
    if len(fallback_words) < 24:
        return None

    action = None if writer_outcome is None else writer_outcome.action
    if action not in {
        WriterActionKind.DIRECT_COMMIT,
        WriterActionKind.TRANSFORM_COMMIT,
        WriterActionKind.PASS_THROUGH_COMMIT,
    }:
        return None

    if len(surface_words) <= max(4, int(len(fallback_words) * 0.22)):
        return "long_dictation_collapsed"

    overlap = len(set(surface_words) & set(fallback_words))
    if len(surface_words) <= int(len(fallback_words) * 0.45) and overlap < max(3, int(len(surface_words) * 0.35)):
        return "low_overlap_writer_output"
    return None


def _preserve_writer_context_terms(
    writer_text: str,
    *,
    fallback_text: str,
    compiled_context: CompiledContext,
) -> tuple[str, list[dict[str, str]]]:
    surface = str(writer_text or "")
    fallback = str(fallback_text or "")
    if not surface.strip() or not fallback.strip():
        return surface, []
    repairs: list[dict[str, str]] = []
    out = surface
    for term in _writer_preservation_terms(compiled_context, fallback):
        if term_present_in_text(term, out):
            continue
        out, term_repairs = _repair_writer_term_near_miss(out, term)
        repairs.extend(term_repairs)
    return out, repairs


def _writer_preservation_terms(compiled_context: CompiledContext, fallback_text: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for term in list(getattr(compiled_context, "terms", ()) or [])[:64]:
        source = str(getattr(term, "source", "") or "")
        protected = bool(getattr(term, "protected", False))
        if source not in {"memory", "replacement", "correction", "screen", "selection", "session", "file", "symbol"} and not protected:
            continue
        value = str(getattr(term, "canonical", None) or getattr(term, "text", "") or "").strip()
        if not value or not term_present_in_text(value, fallback_text):
            continue
        if common_english_single_word(value) and not (protected or _single_token_has_identifier_shape(value)):
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return tuple(out)


def _repair_writer_term_near_miss(text: str, term: str) -> tuple[str, list[dict[str, str]]]:
    target = str(term or "").strip()
    if not target or " " in target:
        return text, []
    repairs: list[dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        observed = match.group(0)
        if observed.casefold() == target.casefold():
            return observed
        if not _writer_term_near_miss(target, observed):
            return observed
        repairs.append({"from": observed, "to": target, "source": "writer_context_term_preservation"})
        return target

    out = re.sub(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9'-]{2,31}(?![A-Za-z0-9])", repl, text)
    return out, repairs


def _writer_term_near_miss(target: str, observed: str) -> bool:
    lhs = str(target or "").strip()
    rhs = str(observed or "").strip()
    if not lhs or not rhs or lhs.casefold() == rhs.casefold():
        return False
    if lhs[:1].casefold() != rhs[:1].casefold() or abs(len(lhs) - len(rhs)) > 2:
        return False
    return difflib.SequenceMatcher(a=lhs.casefold(), b=rhs.casefold(), autojunk=False).ratio() >= 0.78


def _content_word_list(text: str) -> list[str]:
    return [
        tok.casefold()
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9']*", text or "")
        if len(tok) > 1
    ]


def _compute_oneshot_paste_kind(
    writer_outcome: WriterOutcome | None,
    *,
    fallback_text: str,
) -> tuple[str, str | None]:
    """Return ``(paste_kind, noop_reason)`` for broker JSON / macOS shell."""
    if writer_outcome is None:
        t = (fallback_text or "").strip()
        return ("insert", None) if t else ("none", "no_writer")

    act = writer_outcome.action
    if act == WriterActionKind.NOOP:
        reason = writer_outcome.metadata.get("reason")
        return "none", str(reason) if reason else "noop"

    if act in (
        WriterActionKind.MODE_SWITCH,
        WriterActionKind.STATE_MUTATION,
        WriterActionKind.MEMORY_MUTATION,
    ):
        out = (writer_outcome.output_text or "").strip()
        if not out:
            return "none", act.value

    if act == WriterActionKind.TRANSFORM_COMMIT:
        return "replace", None

    if act == WriterActionKind.DIRECT_COMMIT:
        cm = writer_outcome.commit_mode
        if cm == CommitMode.REPLACE_SELECTION:
            return "replace", None
        return "insert", None

    if act == WriterActionKind.PASS_THROUGH_COMMIT:
        t = (writer_outcome.output_text or fallback_text or "").strip()
        return ("insert", None) if t else ("none", "empty_transcript")

    return "insert", None
