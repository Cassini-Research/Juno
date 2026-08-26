from __future__ import annotations

from dataclasses import dataclass
import io
import wave
from typing import TypedDict

import numpy as np
import numpy.typing as npt


#: Frame size used for the per-frame RMS "is this frame speech?" test.
#: Shared by :func:`analyze_audio_signal` and the final-lane silence
#: trimmer so both measure the signal on the same grid.
DEFAULT_FRAME_MS = 20
#: Per-frame RMS above which a frame counts toward ``non_silent_ms``.
#: This is a *diagnostics* threshold: ``non_silent_ms`` only ever makes
#: ``low_signal`` less likely (see ``below_speech_floor``), so erring high
#: here is safe. It is NOT safe as a "may I delete this audio?" test —
#: the silence trimmer deliberately uses its own, lower floor.
DEFAULT_FRAME_RMS_THRESHOLD = 0.006
#: Whole-buffer RMS at or above which a long buffer is called
#: ``speech_likely``. Any audio-dropping decision elsewhere must use a
#: threshold at or below this, or it would discard audio this module is
#: willing to call speech.
SPEECH_LIKELY_RMS_FLOOR = 0.0045


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


def frame_rms(
    audio: npt.ArrayLike,
    *,
    sample_rate_hz: int,
    frame_ms: int = DEFAULT_FRAME_MS,
) -> tuple[npt.NDArray[np.float32], int]:
    """Return ``(per_frame_rms, frame_len_samples)`` for *audio*.

    Only whole frames are measured; a trailing partial frame is dropped
    (it is at most ``frame_ms`` of audio). Callers that need to map a
    frame index back to a sample index multiply by ``frame_len_samples``.
    """

    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_len = max(1, int(sample_rate_hz * max(1, frame_ms) / 1000.0))
    usable = arr[: (arr.size // frame_len) * frame_len]
    if usable.size == 0:
        return np.zeros(0, dtype=np.float32), frame_len
    frames = usable.reshape(-1, frame_len)
    return np.sqrt(np.mean(np.square(frames), axis=1)), frame_len


def analyze_audio_signal(
    audio: npt.ArrayLike,
    *,
    sample_rate_hz: int = 16_000,
    frame_ms: int = DEFAULT_FRAME_MS,
    frame_rms_threshold: float = DEFAULT_FRAME_RMS_THRESHOLD,
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

    per_frame_rms, _frame_len = frame_rms(
        arr, sample_rate_hz=sample_rate_hz, frame_ms=frame_ms
    )
    if per_frame_rms.size:
        non_silent_frames = int(np.sum(per_frame_rms > frame_rms_threshold))
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
        or (duration_ms >= 700 and rms >= SPEECH_LIKELY_RMS_FLOOR)
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


@dataclass(frozen=True, slots=True)
class DecodedWav:
    """A decoded PCM WAV blob, kept in both float and raw-byte form.

    ``samples`` is the mono float32 signal every analysis path works on.
    ``raw_frames`` is the untouched interleaved PCM payload plus the
    container parameters needed to re-emit a bit-identical WAV slice —
    the final-lane silence trimmer cuts audio by slicing these bytes so
    trimming never re-quantizes or down-mixes the audio handed to ASR.

    ``samples`` is empty (and ``raw_frames`` is ``b""``) for a WAV whose
    header parses but that carries no usable audio.
    """

    samples: npt.NDArray[np.float32]
    sample_rate_hz: int
    channels: int
    sample_width: int
    raw_frames: bytes

    @property
    def bytes_per_frame(self) -> int:
        return max(1, self.channels * self.sample_width)


def decode_wav_bytes(wav_bytes: bytes) -> DecodedWav | None:
    """Decode a PCM WAV blob to mono float32, or None if it is not usable WAV.

    Returns ``None`` for anything the ``wave`` module cannot open and for
    sample widths we have no conversion for. Tests and a few old call sites
    pass sentinel bytes into fake transcribers; ``None`` is how callers
    recognize "this is not really audio" and leave it alone.
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
        return DecodedWav(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate_hz=sample_rate,
            channels=channels,
            sample_width=width,
            raw_frames=b"",
        )
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
    return DecodedWav(
        samples=arr,
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width=width,
        raw_frames=raw,
    )


def analyze_wav_bytes(wav_bytes: bytes) -> AudioDiagnostics | None:
    """Return diagnostics for a WAV blob, or None if the blob is not WAV.

    Tests and a few old call sites pass sentinel bytes into fake transcribers.
    Those should continue through the fake path; only decodable audio is
    eligible for the no-paste low-signal gate.
    """

    decoded = decode_wav_bytes(wav_bytes)
    if decoded is None:
        return None
    if decoded.samples.size == 0:
        return analyze_audio_signal([], sample_rate_hz=max(decoded.sample_rate_hz, 1))
    return analyze_audio_signal(decoded.samples, sample_rate_hz=decoded.sample_rate_hz)
