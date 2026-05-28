from __future__ import annotations

from dataclasses import dataclass
import io
import wave
from typing import TypedDict

import numpy as np
import numpy.typing as npt


class AudioDiagnosticsPayload(TypedDict):
    duration_ms: float
    rms: float
    peak: float
    non_silent_ms: float
    clipped_ratio: float
    zero_ratio: float
    low_signal: bool
    speech_likely: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class AudioDiagnostics:
    duration_ms: float
    rms: float
    peak: float
    non_silent_ms: float
    clipped_ratio: float
    zero_ratio: float
    low_signal: bool
    speech_likely: bool
    reason: str | None = None

    def to_dict(self) -> AudioDiagnosticsPayload:
        return {
            "duration_ms": self.duration_ms,
            "rms": self.rms,
            "peak": self.peak,
            "non_silent_ms": self.non_silent_ms,
            "clipped_ratio": self.clipped_ratio,
            "zero_ratio": self.zero_ratio,
            "low_signal": self.low_signal,
            "speech_likely": self.speech_likely,
            "reason": self.reason,
        }


def analyze_audio_signal(
    audio: npt.ArrayLike,
    *,
    sample_rate_hz: int = 16_000,
    frame_ms: int = 20,
    frame_rms_threshold: float = 0.006,
) -> AudioDiagnostics:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sample_rate_hz <= 0:
        return AudioDiagnostics(
            duration_ms=0.0,
            rms=0.0,
            peak=0.0,
            non_silent_ms=0.0,
            clipped_ratio=0.0,
            zero_ratio=1.0,
            low_signal=True,
            speech_likely=False,
            reason="empty_audio",
        )

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -1.0, 1.0)
    abs_arr = np.abs(arr)
    duration_ms = (float(arr.size) / float(sample_rate_hz)) * 1000.0
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    peak = float(np.max(abs_arr)) if arr.size else 0.0
    zero_ratio = float(np.mean(abs_arr <= 1.0 / 32768.0)) if arr.size else 1.0
    clipped_ratio = float(np.mean(abs_arr >= 0.999)) if arr.size else 0.0

    frame_len = max(1, int(sample_rate_hz * max(1, frame_ms) / 1000.0))
    usable = arr[: (arr.size // frame_len) * frame_len]
    if usable.size:
        frames = usable.reshape(-1, frame_len)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        non_silent_frames = int(np.sum(frame_rms > frame_rms_threshold))
        non_silent_ms = float(non_silent_frames * frame_ms)
    else:
        non_silent_ms = duration_ms if rms > frame_rms_threshold else 0.0

    reason: str | None = None
    low_signal = False
    if duration_ms < 40:
        low_signal = True
        reason = "too_short"
    elif zero_ratio >= 0.995 and rms <= 0.0002:
        low_signal = True
        reason = "mostly_zero"
    elif peak <= 0.006 and rms <= 0.0032 and non_silent_ms < 220:
        # This is the class that produced "Yeah", "Thank you", and "Ps Ps"
        # in real logs: long buffers with tiny noise peaks but no sustained
        # speech frames. Quiet speech normally has larger peaks or sustained
        # frames even when average RMS is low.
        low_signal = True
        reason = "below_speech_floor"

    speech_likely = not low_signal and (
        non_silent_ms >= 160
        or peak >= 0.018
        or (duration_ms >= 700 and rms >= 0.0045)
    )
    return AudioDiagnostics(
        duration_ms=duration_ms,
        rms=rms,
        peak=peak,
        non_silent_ms=non_silent_ms,
        clipped_ratio=clipped_ratio,
        zero_ratio=zero_ratio,
        low_signal=low_signal,
        speech_likely=speech_likely,
        reason=reason,
    )


def analyze_wav_bytes(wav_bytes: bytes) -> AudioDiagnostics | None:
    """Return diagnostics for a WAV blob, or None if the blob is not WAV.

    Tests and a few old call sites pass sentinel bytes into fake transcribers.
    Those should continue through the fake path; only decodable audio is
    eligible for the no-paste low-signal gate.
    """

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = int(wf.getnchannels())
            width = int(wf.getsampwidth())
            sample_rate = int(wf.getframerate())
            frames = int(wf.getnframes())
            raw = wf.readframes(frames)
    except Exception:
        return None

    if not raw or channels <= 0 or sample_rate <= 0:
        return analyze_audio_signal([], sample_rate_hz=max(sample_rate, 1))
    if width == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return None
    if channels > 1 and arr.size >= channels:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return analyze_audio_signal(arr, sample_rate_hz=sample_rate)
