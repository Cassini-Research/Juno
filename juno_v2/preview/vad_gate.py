"""Silero VAD gate for the preview-service streaming audio buffer.

This is the **first-layer** defense against Whisper silence hallucinations
(arXiv:2501.11378 measured 40.3% hallucination rate on pure non-speech;
SileroVAD pre-filtering takes WER from 104.8% down to 8.0%). The premise is
straightforward: Whisper hallucinates YouTube-corpus phrases ("Thank you",
"Thanks for watching") on silence padded to 30 s, so we just never let pure
silence reach Whisper's audio buffer in the first place.

Architecture choice — VADIterator over per-chunk threshold:

The previous version of this file called `model(tensor, 16000)` on each 32 ms
frame and admitted the chunk if any single frame scored ≥ 0.5. That was
wrong: Silero's per-frame scores are noisy at chunk boundaries (start/end of
word), so real speech got intermittently rejected. The correct API for
streaming is Silero's own ``VADIterator`` class, which tracks ``min_silence
_duration_ms`` (how long silence must persist before declaring speech ended)
and ``speech_pad_ms`` (boundary padding so the first/last syllable of a word
is never clipped). See:

  - silero-vad README → ``VADIterator`` class (snakers4/silero-vad)
  - faster-whisper ``vad.py`` → ``VadOptions`` (SYSTRAN/faster-whisper)
  - whisper_streaming ``VACOnlineASRProcessor`` (ufal/whisper_streaming)

Parameter choices — we err on the side of "let too much audio through"
rather than "clip real speech", because once a real-speech chunk is
dropped the user can never recover it; a few extra silence chunks reaching
Whisper just cost a small decode and are themselves caught by Layers 2-3
(no_speech_prob filter + BoH blocklist).

  - threshold = 0.5
        Silero canonical default. faster-whisper, whisper_streaming, and
        the silero README all use this.
  - sampling_rate = 16000
        Whisper's required input rate; the Juno shell already resamples
        to 16 kHz before sending PCM to the broker.
  - min_silence_duration_ms = 700
        Mid-utterance pauses (inter-word, breath) are typically 100-500 ms.
        We pick 700 ms so a brief pause does not declare "end of speech";
        this matches the conservative end of the streaming-VAD literature.
        (Silero default is 100 ms — too aggressive for our streaming case;
        faster-whisper uses 2000 ms — too conservative for live preview.)
  - speech_pad_ms = 400
        Pad the speech window by 400 ms on each side. Preserves the first
        and last syllable of every utterance under any VAD timing slack.
        Matches faster-whisper's default.

Behavior contract:
  - The first ~1 second of every utterance is admitted unconditionally
    (``warmup_seconds``). This gives VADIterator enough audio to make a
    confident first decision and guarantees the user's opening word is
    never clipped by a cold VAD.
  - After warmup: a chunk is admitted iff the gate's latched
    ``is_speaking`` state was True at any point during the chunk, OR we
    are still within ``speech_pad_ms`` after the most recent end-of-speech
    event.
  - VAD failure (torch / silero_vad not installed, model load error) falls
    back to admitting everything — better to over-admit than crash.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# Silero v5 requires exactly 512-sample frames at 16 kHz (or 256 at 8 kHz).
# Frames smaller than this are rejected by the model.
_SILERO_FRAME_SAMPLES_16K = 512
_SILERO_FRAME_SAMPLES_8K = 256


@dataclass(slots=True)
class VadGate:
    """Per-utterance VAD gate using Silero's ``VADIterator``.

    Stateful across chunks within one utterance. Call ``reset()`` between
    utterances. ``should_admit(chunk)`` returns ``(admit, reason)`` and
    advances the internal speech/silence state machine.
    """

    sample_rate_hz: int = 16_000
    silero_threshold: float = 0.5
    min_silence_duration_ms: int = 700
    speech_pad_ms: int = 400
    warmup_seconds: float = 1.0
    silero_enabled: bool = True
    # Energy fallback threshold for when Silero is unavailable. Liberal so
    # real speech is never rejected; the Layers 2-3 segment filters catch
    # silence Whisper might hallucinate over.
    energy_fallback_rms: float = 0.003

    # Mutable state below — do not touch from callers.
    _model: object | None = None
    _vad_iter: object | None = None
    _torch: object | None = None
    _loaded: bool = False
    _utterance_started_at: float = 0.0
    _frame_residue: np.ndarray | None = None
    _is_speaking_latch: bool = False
    _last_speech_audio_s: float = -1.0
    _absolute_samples_seen: int = 0

    # ------------------------------------------------------------------ lifecycle

    def warm(self) -> None:
        """Lazy-load Silero. Safe to call repeatedly. Never raises."""
        if self._loaded:
            return
        self._loaded = True
        if not self.silero_enabled:
            self._model = None
            self._vad_iter = None
            self._torch = None
            return
        try:
            import torch  # type: ignore

            from silero_vad import VADIterator, load_silero_vad  # type: ignore

            self._torch = torch
            self._model = load_silero_vad()
            self._vad_iter = VADIterator(
                model=self._model,
                threshold=self.silero_threshold,
                sampling_rate=self.sample_rate_hz,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
            )
        except Exception:
            # Silero load failed → energy fallback path.
            self._torch = None
            self._model = None
            self._vad_iter = None

    def reset(self) -> None:
        """Call between utterances. Clears all per-utterance state."""
        self._utterance_started_at = 0.0
        self._frame_residue = None
        self._is_speaking_latch = False
        self._last_speech_audio_s = -1.0
        self._absolute_samples_seen = 0
        # Re-init VADIterator so its internal state resets too.
        if self._vad_iter is not None and self._model is not None:
            try:
                from silero_vad import VADIterator  # type: ignore

                self._vad_iter = VADIterator(
                    model=self._model,
                    threshold=self.silero_threshold,
                    sampling_rate=self.sample_rate_hz,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    speech_pad_ms=self.speech_pad_ms,
                )
            except Exception:
                self._vad_iter = None

    def has_observed_speech(self) -> bool:
        """True if Silero ever reported a speech_start during this utterance."""
        return self._last_speech_audio_s >= 0.0 or self._is_speaking_latch

    # ------------------------------------------------------------------ gate

    def should_admit(self, chunk: np.ndarray) -> tuple[bool, str]:
        """Decide whether this audio chunk should be added to Whisper's buffer.

        Returns ``(admit, reason)``. The reason string is for telemetry.
        """
        if chunk is None or len(chunk) == 0:
            return False, "empty_chunk"

        self.warm()
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return False, "empty_chunk"

        # Track absolute audio position for warmup window logic. Use audio time,
        # not wall-clock time: replay tests, backpressure, or a slow decoder can
        # make wall time unrelated to how much of the utterance has actually
        # reached the gate.
        now = time.monotonic()
        if self._utterance_started_at == 0.0:
            self._utterance_started_at = now
        self._absolute_samples_seen += chunk.size

        # Warmup: admit everything for the first ``warmup_seconds`` so VAD
        # has audio to settle and we never clip the user's opening word.
        audio_seen_seconds = self._absolute_samples_seen / float(self.sample_rate_hz)
        in_warmup = audio_seen_seconds <= self.warmup_seconds
        if in_warmup:
            # Still feed VADIterator so it builds state during warmup, and
            # honor any start/end events it emits. The previous code ignored
            # warmup events, so the first post-warmup chunk could look like
            # silence even though speech had already started.
            self._apply_vad_events(self._advance_vad(chunk), audio_seen_seconds)
            return True, "warmup"

        # Silero unavailable → energy fallback.
        if self._vad_iter is None or self._model is None or self._torch is None:
            rms = float(np.sqrt(np.mean(chunk * chunk)))
            if rms >= self.energy_fallback_rms:
                self._is_speaking_latch = True
                self._last_speech_audio_s = audio_seen_seconds
                return True, "energy_fallback_voice"
            # Even on apparent silence, if we've recently observed speech
            # admit a grace window so we don't clip mid-utterance.
            grace_s = (self.speech_pad_ms + self.min_silence_duration_ms) / 1000.0
            if (
                self._last_speech_audio_s >= 0.0
                and audio_seen_seconds - self._last_speech_audio_s < grace_s
            ):
                return True, "energy_fallback_grace"
            return False, "energy_fallback_silence"

        # Silero path — feed audio frame-by-frame and watch start/end events.
        self._apply_vad_events(self._advance_vad(chunk), audio_seen_seconds)

        if self._is_speaking_latch:
            return True, "vad_speech_active"

        # Post-speech grace: even after Silero says 'end', keep admitting
        # for speech_pad_ms so the final phoneme of the last word reaches
        # Whisper. (VADIterator already includes its own internal padding
        # before firing 'end', but adding an extra grace here is cheap
        # belt-and-braces against word-final clipping.)
        grace_s = self.speech_pad_ms / 1000.0
        if (
            self._last_speech_audio_s >= 0.0
            and audio_seen_seconds - self._last_speech_audio_s < grace_s
        ):
            return True, "vad_post_speech_grace"

        return False, "vad_silence"

    # ------------------------------------------------------------------ internals

    def _apply_vad_events(self, events: list[dict], audio_seen_seconds: float) -> None:
        for ev in events:
            if "start" in ev:
                self._is_speaking_latch = True
                self._last_speech_audio_s = audio_seen_seconds
            elif "end" in ev:
                self._is_speaking_latch = False
                self._last_speech_audio_s = audio_seen_seconds

    def _advance_vad(self, chunk: np.ndarray) -> list[dict]:
        """Feed ``chunk`` to VADIterator in 512-sample frames, return events.

        Silero v5 requires exactly 512-sample frames at 16 kHz. We carry the
        residue across calls so frames are never split / replayed.
        """
        if self._vad_iter is None or self._torch is None:
            return []
        frame_size = (
            _SILERO_FRAME_SAMPLES_16K
            if self.sample_rate_hz == 16_000
            else _SILERO_FRAME_SAMPLES_8K
            if self.sample_rate_hz == 8_000
            else None
        )
        if frame_size is None:
            return []
        # Concatenate prior residue with new chunk.
        if self._frame_residue is not None and self._frame_residue.size > 0:
            audio = np.concatenate([self._frame_residue, chunk])
        else:
            audio = chunk
        events: list[dict] = []
        num_full_frames = audio.size // frame_size
        for i in range(num_full_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            tensor = self._torch.from_numpy(frame).float()
            try:
                ev = self._vad_iter(tensor)
            except Exception:
                # Silero internal error — bail out for this chunk; do not
                # change latch state. Caller will treat as "no event".
                self._frame_residue = None
                return events
            if isinstance(ev, dict):
                events.append(ev)
        # Save residue for next call.
        used = num_full_frames * frame_size
        self._frame_residue = audio[used:].copy() if used < audio.size else None
        return events
