"""Unit tests for the low-level audio primitives.

Covers juno_v2.audio.ring_buffer.AudioRingBuffer, juno_v2.audio.clock
.AudioFrameClock, juno_v2.asr.wav.encode_wav_bytes, and the WAV decoding /
stub transcription pieces of juno_core_v3.dictation.transcriber.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from juno_core_v3.dictation.transcriber import StubTranscriber, _decode_wav_to_float32
from juno_v2.asr.wav import encode_wav_bytes
from juno_v2.audio.clock import AudioFrameClock
from juno_v2.audio.ring_buffer import AudioRingBuffer

from tests.audio_fixtures import (
    SAMPLE_RATE_HZ,
    sine_tone,
    slice_into_frames,
)


# ------------------------------------------------------------------ #
# AudioRingBuffer
# ------------------------------------------------------------------ #

def _frames(count: int) -> list:
    return slice_into_frames(sine_tone(duration_s=count * 0.02, amplitude=0.1))


def test_ring_buffer_append_and_len() -> None:
    buf = AudioRingBuffer(max_frames=10)
    frames = _frames(4)
    for frame in frames:
        buf.append(frame)
    assert len(buf) == 4
    assert [f.index for f in buf.snapshot()] == [0, 1, 2, 3]


def test_ring_buffer_fifo_overflow_drops_oldest() -> None:
    buf = AudioRingBuffer(max_frames=3)
    buf.extend(_frames(5))
    assert len(buf) == 3
    assert [f.index for f in buf.snapshot()] == [2, 3, 4]


def test_ring_buffer_duration_ms() -> None:
    buf = AudioRingBuffer(max_frames=100)
    assert buf.duration_ms() == 0.0
    buf.extend(_frames(5))  # frames span 0..100 ms
    assert buf.duration_ms() == pytest.approx(100.0)


def test_ring_buffer_duration_after_overflow() -> None:
    buf = AudioRingBuffer(max_frames=2)
    buf.extend(_frames(5))  # keeps frames 3 and 4: 60..100 ms
    assert buf.duration_ms() == pytest.approx(40.0)


def test_ring_buffer_clear() -> None:
    buf = AudioRingBuffer(max_frames=4)
    buf.extend(_frames(4))
    buf.clear()
    assert len(buf) == 0
    assert buf.snapshot() == []
    assert buf.duration_ms() == 0.0


def test_ring_buffer_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError):
        AudioRingBuffer(max_frames=0)
    with pytest.raises(ValueError):
        AudioRingBuffer(max_frames=-3)


# ------------------------------------------------------------------ #
# AudioFrameClock
# ------------------------------------------------------------------ #

def test_clock_frame_samples_default_16k_20ms() -> None:
    clock = AudioFrameClock()
    assert clock.frame_samples == 320


@pytest.mark.parametrize(
    ("sample_rate_hz", "frame_ms", "expected"),
    [(16_000, 20, 320), (8_000, 20, 160), (16_000, 30, 480), (48_000, 10, 480)],
)
def test_clock_frame_samples_calc(sample_rate_hz: int, frame_ms: int, expected: int) -> None:
    clock = AudioFrameClock(sample_rate_hz=sample_rate_hz, frame_ms=frame_ms)
    assert clock.frame_samples == expected


def test_clock_make_frame_timestamps() -> None:
    clock = AudioFrameClock(sample_rate_hz=16_000, frame_ms=20, source="mic")
    samples = np.zeros(320, dtype=np.float32)
    frame = clock.make_frame(5, samples)
    assert frame.index == 5
    assert frame.start_sample == 1600
    assert frame.end_sample == 1920
    assert frame.start_ms == pytest.approx(100.0)
    assert frame.end_ms == pytest.approx(120.0)
    assert frame.duration_ms == pytest.approx(20.0)
    assert frame.sample_count == 320
    assert frame.source == "mic"
    assert frame.sample_rate_hz == 16_000


def test_clock_make_frame_rejects_wrong_sample_count() -> None:
    clock = AudioFrameClock()
    with pytest.raises(ValueError, match="expected 320 samples"):
        clock.make_frame(0, np.zeros(100, dtype=np.float32))


def test_clock_validation_errors() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        AudioFrameClock(sample_rate_hz=0)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        AudioFrameClock(sample_rate_hz=-16_000)
    with pytest.raises(ValueError, match="frame_ms"):
        AudioFrameClock(frame_ms=0)
    # Positive inputs that still truncate to zero samples per frame.
    with pytest.raises(ValueError, match="zero samples"):
        AudioFrameClock(sample_rate_hz=10, frame_ms=20)


# ------------------------------------------------------------------ #
# encode_wav_bytes
# ------------------------------------------------------------------ #

def test_encode_wav_bytes_round_trip() -> None:
    audio = sine_tone(freq_hz=440.0, duration_s=0.25, amplitude=0.5)
    blob = encode_wav_bytes(audio, SAMPLE_RATE_HZ)

    with wave.open(io.BytesIO(blob), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE_HZ
        assert wf.getnframes() == audio.shape[0]
        raw = wf.readframes(wf.getnframes())

    decoded = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    assert np.max(np.abs(decoded - audio)) < (1.5 / 32767.0)


def test_encode_wav_bytes_clips_out_of_range_input() -> None:
    audio = np.array([2.0, -2.0, 0.0, 1.0, -1.0], dtype=np.float32)
    blob = encode_wav_bytes(audio, SAMPLE_RATE_HZ)
    with wave.open(io.BytesIO(blob), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert pcm[0] == 32767
    assert pcm[1] == -32767
    assert pcm[2] == 0
    assert pcm[3] == 32767
    assert pcm[4] == -32767


# ------------------------------------------------------------------ #
# _decode_wav_to_float32
# ------------------------------------------------------------------ #

def _make_wav_bytes(
    pcm16: np.ndarray, sample_rate_hz: int, channels: int = 1, sampwidth: int = 2
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def test_decode_mono_16k_passthrough() -> None:
    audio = sine_tone(duration_s=0.5, amplitude=0.4)
    blob = encode_wav_bytes(audio, 16_000)
    decoded, sr = _decode_wav_to_float32(blob)
    assert sr == 16_000
    assert decoded.dtype == np.float32
    assert decoded.shape[0] == audio.shape[0]
    assert np.max(np.abs(decoded - audio)) < (2.0 / 32768.0)


def test_decode_stereo_downmix() -> None:
    mono = sine_tone(duration_s=0.2, amplitude=0.3)
    pcm = (mono * 32767.0).astype(np.int16)
    interleaved = np.empty(pcm.shape[0] * 2, dtype=np.int16)
    interleaved[0::2] = pcm  # left
    interleaved[1::2] = pcm  # right (identical -> downmix == mono)
    blob = _make_wav_bytes(interleaved, 16_000, channels=2)
    decoded, sr = _decode_wav_to_float32(blob)
    assert sr == 16_000
    assert decoded.shape[0] == mono.shape[0]
    assert np.max(np.abs(decoded - mono)) < (2.0 / 32768.0)


@pytest.mark.parametrize("source_rate", [8_000, 32_000])
def test_decode_resamples_to_16k(source_rate: int) -> None:
    duration_s = 0.5
    audio = sine_tone(
        freq_hz=200.0, duration_s=duration_s, amplitude=0.4, sample_rate_hz=source_rate
    )
    blob = encode_wav_bytes(audio, source_rate)
    decoded, sr = _decode_wav_to_float32(blob)
    assert sr == 16_000
    expected_len = int(round(audio.shape[0] * (16_000 / source_rate)))
    assert decoded.shape[0] == expected_len
    # Duration preserved within a millisecond.
    assert decoded.shape[0] / 16_000 == pytest.approx(duration_s, abs=0.001)
    # Energy roughly preserved by linear interpolation of a low-freq tone.
    assert float(np.sqrt(np.mean(np.square(decoded)))) == pytest.approx(
        0.4 / np.sqrt(2.0), rel=0.05
    )


def test_decode_rejects_unsupported_sample_width() -> None:
    pcm8 = (np.zeros(160) + 128).astype(np.uint8)
    blob = _make_wav_bytes(pcm8, 16_000, channels=1, sampwidth=1)
    with pytest.raises(ValueError, match="sample width"):
        _decode_wav_to_float32(blob)


# ------------------------------------------------------------------ #
# StubTranscriber
# ------------------------------------------------------------------ #

def test_stub_transcriber_returns_canned_transcript_and_duration() -> None:
    stub = StubTranscriber(canned_transcript="hello from the stub")
    audio = sine_tone(duration_s=1.0, amplitude=0.3)
    blob = encode_wav_bytes(audio, 16_000)
    result = stub.transcribe_wav(blob, language="en")
    assert result.transcript == "hello from the stub"
    assert result.language == "en"
    assert result.backend_name == "stub"
    assert result.audio_duration_ms == pytest.approx(1000.0)
    assert result.decode_ms == 0.0
    assert result.model_path == ""


def test_stub_transcriber_duration_tracks_audio_length() -> None:
    stub = StubTranscriber(canned_transcript="x")
    blob = encode_wav_bytes(sine_tone(duration_s=0.35, amplitude=0.2), 16_000)
    result = stub.transcribe_wav(blob)
    assert result.audio_duration_ms == pytest.approx(350.0)
    assert result.language is None


def test_stub_transcriber_ignores_decode_hints() -> None:
    stub = StubTranscriber(canned_transcript="hinted")
    blob = encode_wav_bytes(sine_tone(duration_s=0.1), 16_000)
    result = stub.transcribe_wav(
        blob, initial_prompt="some prompt", bias_phrases=["alpha", "beta"]
    )
    assert result.transcript == "hinted"
