"""Leading/trailing silence trimming for the **final** (one-shot) decode lane.

Why this exists
---------------
The preview lane never lets pure silence reach Whisper: every chunk passes
:class:`juno_v2.preview.vad_gate.VadGate` first. The final lane had no
equivalent — ``OneShotDictationPipeline`` handed the whole recorded buffer
to the backend and only rejected it wholesale when
:func:`~juno_v2.audio.diagnostics.analyze_wav_bytes` said ``low_signal``.
An utterance that *starts* with real speech and then sits on room tone for
minutes is not ``low_signal``, so all of that room tone reached Whisper —
which is exactly the input Whisper hallucinates on (a 336 s recording with
speech ending at ~90 s produced a token loop plus ~120 words of fabricated
advertisement copy).

What it does
------------
Cut silence off the **edges** of the buffer, and only the edges:

* Speech is located with the same per-frame RMS test
  :func:`~juno_v2.audio.diagnostics.analyze_audio_signal` already uses for
  ``non_silent_ms`` (20 ms frames, RMS > 0.006). Reusing that analysis
  keeps one definition of "non-silent frame" in the final lane and adds no
  new dependency — Silero is a streaming-shaped API (stateful
  ``VADIterator`` over sequential chunks) and is the preview lane's tool;
  a one-shot buffer needs a single offline pass, not a state machine.
* ``edge_padding_ms`` of silence is deliberately **kept** at each end, so a
  soft word onset or a trailing consonant can never be clipped and Whisper
  still gets the leading/trailing context it decodes better with.
* An edge is only cut when the removable span exceeds ``min_trim_ms``.
  Normal end-of-sentence pauses and the second or two between "stop
  talking" and "release the hotkey" are untouched; this fires only on the
  pathological case the guard exists for.
* Internal silence is **never** touched. Splitting an utterance on internal
  pauses changes Whisper's context window and its punctuation/casing
  decisions; that is a separate, larger change.

If no speech frame is found at all the buffer is returned untouched — the
``low_signal`` reject upstream owns that case, and handing the backend an
empty buffer would be strictly worse than handing it silence.

The caller is responsible for keeping user-facing duration and segment
timestamps in the *original* recording's timebase; :class:`SilenceTrimResult`
carries the offsets needed to do that.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

from juno_v2.audio.diagnostics import (
    DEFAULT_FRAME_MS,
    DEFAULT_FRAME_RMS_THRESHOLD,
    decode_wav_bytes,
    frame_rms,
)

#: Silence kept on each side of the detected speech span. Generous on
#: purpose: clipping a real syllable is unrecoverable, while 1.5 s of extra
#: room tone costs a few milliseconds of decode and is well inside what
#: Whisper tolerates without hallucinating.
DEFAULT_EDGE_PADDING_MS = 1_500.0

#: An edge is only cut when at least this much silence can be removed
#: *after* honouring the padding. Below this we leave the audio alone:
#: short pauses are normal speech behavior, and every byte we do not touch
#: is a byte we cannot get wrong.
DEFAULT_MIN_TRIM_MS = 5_000.0


@dataclass(frozen=True, slots=True)
class SilenceTrimResult:
    """Outcome of :func:`trim_wav_edge_silence`.

    ``wav_bytes`` is what should be sent to the ASR backend — the trimmed
    buffer when ``trimmed`` is True, otherwise the caller's original blob
    unchanged. ``leading_trimmed_ms`` is the offset that maps a timestamp
    in the trimmed buffer back onto the original recording.
    """

    wav_bytes: bytes
    trimmed: bool
    reason: str
    original_duration_ms: float
    trimmed_duration_ms: float
    leading_trimmed_ms: float
    trailing_trimmed_ms: float

    @property
    def total_trimmed_ms(self) -> float:
        return self.leading_trimmed_ms + self.trailing_trimmed_ms

    def to_dict(self) -> dict[str, Any]:
        """Trace/metadata payload. Never carries the audio itself."""
        return {
            "trimmed": self.trimmed,
            "reason": self.reason,
            "original_duration_ms": self.original_duration_ms,
            "trimmed_duration_ms": self.trimmed_duration_ms,
            "leading_trimmed_ms": self.leading_trimmed_ms,
            "trailing_trimmed_ms": self.trailing_trimmed_ms,
            "total_trimmed_ms": self.total_trimmed_ms,
        }


def _untouched(wav_bytes: bytes, reason: str, duration_ms: float) -> SilenceTrimResult:
    return SilenceTrimResult(
        wav_bytes=wav_bytes,
        trimmed=False,
        reason=reason,
        original_duration_ms=duration_ms,
        trimmed_duration_ms=duration_ms,
        leading_trimmed_ms=0.0,
        trailing_trimmed_ms=0.0,
    )


def trim_wav_edge_silence(
    wav_bytes: bytes,
    *,
    edge_padding_ms: float = DEFAULT_EDGE_PADDING_MS,
    min_trim_ms: float = DEFAULT_MIN_TRIM_MS,
    frame_ms: int = DEFAULT_FRAME_MS,
    frame_rms_threshold: float = DEFAULT_FRAME_RMS_THRESHOLD,
) -> SilenceTrimResult:
    """Trim long leading/trailing silence from a WAV blob.

    Returns the original blob untouched (``trimmed=False``) whenever the
    input is not decodable WAV, carries no detectable speech, or has less
    than ``min_trim_ms`` of removable silence at both edges.

    The returned WAV keeps the input's sample rate, channel count, and
    sample width: the cut is a slice of the raw PCM payload, not a
    re-encode, so the audio the backend sees is bit-identical to the
    corresponding span of the original recording.
    """

    decoded = decode_wav_bytes(wav_bytes)
    if decoded is None:
        return _untouched(wav_bytes, "not_wav", 0.0)

    sample_rate = decoded.sample_rate_hz
    total_samples = int(decoded.samples.size)
    if total_samples == 0 or sample_rate <= 0:
        return _untouched(wav_bytes, "empty_audio", 0.0)

    duration_ms = (total_samples / float(sample_rate)) * 1000.0

    per_frame_rms, frame_len = frame_rms(
        decoded.samples, sample_rate_hz=sample_rate, frame_ms=frame_ms
    )
    speech_frames = np.flatnonzero(per_frame_rms > frame_rms_threshold)
    if speech_frames.size == 0:
        # No frame clears the speech floor. This is the ``low_signal``
        # family, which the pipeline rejects upstream; trimming here would
        # hand the backend an empty buffer instead of silence.
        return _untouched(wav_bytes, "no_speech_detected", duration_ms)

    pad_samples = max(0, int(round(edge_padding_ms * sample_rate / 1000.0)))
    speech_start = int(speech_frames[0]) * frame_len
    speech_end = min(total_samples, (int(speech_frames[-1]) + 1) * frame_len)

    start = max(0, speech_start - pad_samples)
    end = min(total_samples, speech_end + pad_samples)

    leading_trimmed_ms = (start / float(sample_rate)) * 1000.0
    if leading_trimmed_ms <= min_trim_ms:
        start = 0
        leading_trimmed_ms = 0.0

    trailing_trimmed_ms = ((total_samples - end) / float(sample_rate)) * 1000.0
    if trailing_trimmed_ms <= min_trim_ms:
        end = total_samples
        trailing_trimmed_ms = 0.0

    if start == 0 and end == total_samples:
        return _untouched(wav_bytes, "no_long_edge_silence", duration_ms)

    bpf = decoded.bytes_per_frame
    sliced = decoded.raw_frames[start * bpf : end * bpf]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(max(1, decoded.channels))
        out.setsampwidth(max(1, decoded.sample_width))
        out.setframerate(sample_rate)
        out.writeframes(sliced)

    trimmed_duration_ms = ((end - start) / float(sample_rate)) * 1000.0
    return SilenceTrimResult(
        wav_bytes=buffer.getvalue(),
        trimmed=True,
        reason="trimmed",
        original_duration_ms=duration_ms,
        trimmed_duration_ms=trimmed_duration_ms,
        leading_trimmed_ms=leading_trimmed_ms,
        trailing_trimmed_ms=trailing_trimmed_ms,
    )


__all__ = [
    "DEFAULT_EDGE_PADDING_MS",
    "DEFAULT_MIN_TRIM_MS",
    "SilenceTrimResult",
    "trim_wav_edge_silence",
]
