from __future__ import annotations

import io
import wave

import numpy as np


def encode_wav_bytes(audio: np.ndarray, sample_rate_hz: int) -> bytes:
    """Encode mono float audio in [-1.0, 1.0] as 16-bit PCM WAV bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm16.tobytes())
    return buffer.getvalue()
