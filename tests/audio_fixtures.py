"""Shared audio fixture helpers for the speech-pipeline test suite.

Not a test file. Generates and caches WAV fixtures (macOS ``say`` TTS plus
numpy-synthesized signals) in a process-lifetime temp directory, and provides
helpers to load WAVs into float32 arrays and slice them into 20 ms
:class:`juno_v2.contracts.audio.AudioFrame` objects via
:class:`juno_v2.audio.clock.AudioFrameClock`.
"""

from __future__ import annotations

import atexit
import hashlib
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from juno_v2.audio.clock import AudioFrameClock
from juno_v2.contracts.audio import AudioFrame

SAMPLE_RATE_HZ = 16_000
FRAME_MS = 20

_CACHE_DIR: Path | None = None


def fixture_cache_dir() -> Path:
    """Session-lifetime cache directory for generated WAV fixtures."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = Path(tempfile.mkdtemp(prefix="juno-audio-fixtures-"))
        atexit.register(shutil.rmtree, _CACHE_DIR, ignore_errors=True)
    return _CACHE_DIR


# ------------------------------------------------------------------ #
# TTS via macOS `say`
# ------------------------------------------------------------------ #

def say_available() -> bool:
    return shutil.which("say") is not None


def require_say() -> None:
    """Skip the calling test when macOS ``say`` is unavailable."""
    if not say_available():
        pytest.skip("macOS `say` binary not available")


def tts_wav_path(text: str, voice: str = "Samantha") -> Path:
    """Generate (once) and return a 16 kHz mono PCM16 WAV of ``text``."""
    require_say()
    key = hashlib.sha1(f"{voice}::{text}".encode("utf-8")).hexdigest()[:16]
    path = fixture_cache_dir() / f"tts_{key}.wav"
    if not path.exists():
        subprocess.run(
            [
                "say",
                "-v",
                voice,
                "-o",
                str(path),
                f"--data-format=LEI16@{SAMPLE_RATE_HZ}",
                text,
            ],
            check=True,
            capture_output=True,
        )
    return path


def tts_wav_bytes(text: str, voice: str = "Samantha") -> bytes:
    return tts_wav_path(text, voice=voice).read_bytes()


def tts_speech_array(text: str, voice: str = "Samantha") -> np.ndarray:
    samples, sample_rate = load_wav_float32(tts_wav_path(text, voice=voice))
    assert sample_rate == SAMPLE_RATE_HZ, f"say produced unexpected rate {sample_rate}"
    return samples


# ------------------------------------------------------------------ #
# numpy-generated signals (float32 mono in [-1, 1])
# ------------------------------------------------------------------ #

def silence(duration_s: float, sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    return np.zeros(int(round(duration_s * sample_rate_hz)), dtype=np.float32)


def sine_tone(
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    amplitude: float = 0.3,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    t = np.arange(int(round(duration_s * sample_rate_hz)), dtype=np.float32)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t / sample_rate_hz)).astype(
        np.float32
    )


def white_noise(
    duration_s: float = 1.0,
    amplitude: float = 0.05,
    seed: int = 1234,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sample_rate_hz))
    return (rng.uniform(-1.0, 1.0, size=n) * amplitude).astype(np.float32)


def speech_then_silence(
    speech: np.ndarray,
    silence_s: float,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Composite: a speech/tone segment followed by pure silence."""
    return np.concatenate(
        [speech.astype(np.float32), silence(silence_s, sample_rate_hz)]
    ).astype(np.float32)


# ------------------------------------------------------------------ #
# WAV I/O
# ------------------------------------------------------------------ #

def write_wav(
    path: Path, samples: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ
) -> Path:
    """Write mono float32 [-1, 1] samples as a PCM16 WAV file."""
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm16.tobytes())
    return path


def cached_wav(
    name: str, samples: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ
) -> Path:
    """Write ``samples`` to a named WAV in the fixture cache (once)."""
    path = fixture_cache_dir() / f"{name}.wav"
    if not path.exists():
        write_wav(path, samples, sample_rate_hz)
    return path


def load_wav_float32(path: Path | str) -> tuple[np.ndarray, int]:
    """Load a PCM16 WAV file as (mono float32 array in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    assert width == 2, f"expected PCM16 WAV, got sample width {width}"
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr.astype(np.float32), sample_rate


# ------------------------------------------------------------------ #
# AudioFrame slicing
# ------------------------------------------------------------------ #

def slice_into_frames(
    samples: np.ndarray,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    frame_ms: int = FRAME_MS,
    source: str = "test_fixture",
) -> list[AudioFrame]:
    """Slice mono float32 audio into AudioFrame objects (incomplete tail dropped)."""
    clock = AudioFrameClock(
        sample_rate_hz=sample_rate_hz, frame_ms=frame_ms, source=source
    )
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    n_frames = arr.shape[0] // clock.frame_samples
    return [
        clock.make_frame(
            i, arr[i * clock.frame_samples : (i + 1) * clock.frame_samples]
        )
        for i in range(n_frames)
    ]
