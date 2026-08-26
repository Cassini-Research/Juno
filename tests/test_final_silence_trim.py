"""Tests for the final-lane edge-silence trim (GitHub issue #81).

Every fixture here is synthesized with numpy — no recorded audio is stored
in the repo. "Speech" is tone-modulated broadband noise with a syllabic
envelope: loud enough to clear the shared per-frame RMS speech floor, which
is all the trimmer's detector cares about.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from juno_core_v3.dictation.pipeline import (
    OneShotDictationPipeline,
    _restore_pretrim_audio_timebase,
)
from juno_core_v3.dictation.transcriber import TranscribeResult
from juno_v2.asr.wav import encode_wav_bytes
from juno_v2.audio.silence_trim import (
    DEFAULT_EDGE_PADDING_MS,
    DEFAULT_MIN_TRIM_MS,
    trim_wav_edge_silence,
)
from juno_v2.contracts.final import FinalSegment
from juno_v2.observability.tracing import TraceRecorder

SR = 16_000


# ------------------------------------------------------------------ #
# Synthetic fixtures
# ------------------------------------------------------------------ #

def speech_like(duration_s: float, *, seed: int = 81, amplitude: float = 0.35) -> np.ndarray:
    """Tone-modulated broadband noise standing in for speech."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * SR))
    t = np.arange(n, dtype=np.float32) / SR
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t))
    carrier = np.sin(2.0 * np.pi * 180.0 * t).astype(np.float32)
    noise = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    return (amplitude * envelope * (0.6 * carrier + 0.4 * noise)).astype(np.float32)


def room_tone(duration_s: float, *, seed: int = 7, amplitude: float = 0.0008) -> np.ndarray:
    """Near-silence: audible room noise well under the speech floor."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * SR))
    return (rng.uniform(-1.0, 1.0, size=n) * amplitude).astype(np.float32)


def wav_of(*parts: np.ndarray) -> bytes:
    return encode_wav_bytes(np.concatenate(parts).astype(np.float32), SR)


def wav_sample_count(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes()


def wav_params(wav_bytes: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate()


def ms_to_samples(ms: float) -> int:
    return int(round(ms * SR / 1000.0))


# ------------------------------------------------------------------ #
# trim_wav_edge_silence
# ------------------------------------------------------------------ #

def test_long_trailing_silence_is_trimmed_to_the_padding() -> None:
    blob = wav_of(speech_like(2.0), room_tone(30.0))
    trim = trim_wav_edge_silence(blob)

    assert trim.trimmed is True
    assert trim.reason == "trimmed"
    assert trim.leading_trimmed_ms == 0.0
    assert trim.original_duration_ms == pytest.approx(32_000.0, abs=1.0)
    # 2 s of speech + the 1.5 s padding we deliberately keep.
    assert trim.trimmed_duration_ms == pytest.approx(3_500.0, abs=60.0)
    assert trim.trailing_trimmed_ms == pytest.approx(28_500.0, abs=60.0)
    assert wav_sample_count(trim.wav_bytes) == pytest.approx(
        ms_to_samples(3_500.0), abs=ms_to_samples(60.0)
    )
    assert wav_sample_count(blob) == ms_to_samples(32_000.0)


def test_long_leading_silence_is_trimmed_to_the_padding() -> None:
    blob = wav_of(room_tone(20.0), speech_like(2.0))
    trim = trim_wav_edge_silence(blob)

    assert trim.trimmed is True
    assert trim.trailing_trimmed_ms == 0.0
    assert trim.leading_trimmed_ms == pytest.approx(18_500.0, abs=60.0)
    assert trim.trimmed_duration_ms == pytest.approx(3_500.0, abs=60.0)


def test_both_edges_trimmed_together() -> None:
    blob = wav_of(room_tone(15.0), speech_like(3.0), room_tone(40.0))
    trim = trim_wav_edge_silence(blob)

    assert trim.trimmed is True
    assert trim.leading_trimmed_ms == pytest.approx(13_500.0, abs=60.0)
    assert trim.trailing_trimmed_ms == pytest.approx(38_500.0, abs=60.0)
    assert trim.trimmed_duration_ms == pytest.approx(6_000.0, abs=120.0)
    assert trim.original_duration_ms == pytest.approx(58_000.0, abs=1.0)


def test_short_trailing_pause_is_left_alone() -> None:
    # 4 s of trailing silence: above the padding but below the 5 s floor,
    # so a normal end-of-utterance pause is never touched.
    blob = wav_of(speech_like(2.0), room_tone(4.0))
    trim = trim_wav_edge_silence(blob)

    assert trim.trimmed is False
    assert trim.reason == "no_long_edge_silence"
    assert trim.wav_bytes == blob
    assert trim.trimmed_duration_ms == trim.original_duration_ms


def test_silence_just_under_and_over_the_floor() -> None:
    # The removable span is (silence - padding). Straddle DEFAULT_MIN_TRIM_MS.
    padding_s = DEFAULT_EDGE_PADDING_MS / 1000.0
    floor_s = DEFAULT_MIN_TRIM_MS / 1000.0

    under = wav_of(speech_like(2.0), room_tone(padding_s + floor_s - 1.0))
    assert trim_wav_edge_silence(under).trimmed is False

    over = wav_of(speech_like(2.0), room_tone(padding_s + floor_s + 1.0))
    assert trim_wav_edge_silence(over).trimmed is True


def test_internal_silence_is_never_trimmed() -> None:
    # A 30 s pause between two utterances stays exactly where it is: this
    # change only ever cuts the edges.
    blob = wav_of(speech_like(2.0), room_tone(30.0), speech_like(2.0))
    trim = trim_wav_edge_silence(blob)

    assert trim.trimmed is False
    assert trim.reason == "no_long_edge_silence"
    assert wav_sample_count(trim.wav_bytes) == wav_sample_count(blob)


def test_internal_silence_survives_an_edge_trim() -> None:
    blob = wav_of(room_tone(20.0), speech_like(2.0), room_tone(30.0), speech_like(2.0), room_tone(20.0))
    trim = trim_wav_edge_silence(blob)

    assert trim.trimmed is True
    # Speech span (2 + 30 + 2 = 34 s) plus 1.5 s padding on each side. The
    # internal 30 s pause is fully inside the retained span.
    assert trim.trimmed_duration_ms == pytest.approx(37_000.0, abs=120.0)


def test_all_silence_is_returned_untouched() -> None:
    # The pipeline's low_signal reject owns this case; trimming would hand
    # the backend an empty buffer.
    blob = wav_of(room_tone(30.0))
    trim = trim_wav_edge_silence(blob)

    assert trim.trimmed is False
    assert trim.reason == "no_speech_detected"
    assert trim.wav_bytes == blob


def test_non_wav_bytes_are_returned_untouched() -> None:
    trim = trim_wav_edge_silence(b"definitely not a wav file")
    assert trim.trimmed is False
    assert trim.reason == "not_wav"
    assert trim.wav_bytes == b"definitely not a wav file"

    empty = trim_wav_edge_silence(b"")
    assert empty.trimmed is False
    assert empty.wav_bytes == b""


def test_trimmed_wav_keeps_container_format_and_bit_exact_samples() -> None:
    blob = wav_of(speech_like(2.0), room_tone(30.0))
    trim = trim_wav_edge_silence(blob)

    assert wav_params(trim.wav_bytes) == wav_params(blob)

    with wave.open(io.BytesIO(blob), "rb") as wf:
        original = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    with wave.open(io.BytesIO(trim.wav_bytes), "rb") as wf:
        trimmed = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")

    start = ms_to_samples(trim.leading_trimmed_ms)
    assert np.array_equal(trimmed, original[start : start + trimmed.size])


def test_stereo_wav_is_trimmed_and_stays_stereo() -> None:
    mono = np.concatenate([speech_like(2.0), room_tone(30.0)]).astype(np.float32)
    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    interleaved = np.repeat(pcm16, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(interleaved.tobytes())
    blob = buf.getvalue()

    trim = trim_wav_edge_silence(blob)
    assert trim.trimmed is True
    assert wav_params(trim.wav_bytes) == (2, 2, SR)
    assert wav_sample_count(trim.wav_bytes) == pytest.approx(
        ms_to_samples(3_500.0), abs=ms_to_samples(60.0)
    )


def test_thresholds_are_configurable() -> None:
    blob = wav_of(speech_like(2.0), room_tone(4.0))
    trim = trim_wav_edge_silence(blob, edge_padding_ms=200.0, min_trim_ms=1_000.0)

    assert trim.trimmed is True
    assert trim.trimmed_duration_ms == pytest.approx(2_200.0, abs=60.0)


def test_to_dict_carries_no_audio() -> None:
    payload = trim_wav_edge_silence(wav_of(speech_like(2.0), room_tone(30.0))).to_dict()
    assert payload["trimmed"] is True
    assert payload["total_trimmed_ms"] == pytest.approx(28_500.0, abs=60.0)
    assert "wav_bytes" not in payload


# ------------------------------------------------------------------ #
# OneShotDictationPipeline integration
# ------------------------------------------------------------------ #

@dataclass
class RecordingTranscriber:
    """Stands in for the real backend and records the bytes it was handed.

    The MLX model is not available in CI, so we assert on the byte length /
    sample count reaching the backend rather than on a decode.
    """

    canned_transcript: str = "hello world this is a dictation test"
    backend_name: str = "recording"
    segments: tuple[Any, ...] = ()
    seen: list[bytes] = field(default_factory=list)

    def transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        language_policy: str | None = None,
        initial_prompt: str | None = None,
        bias_phrases: list[str] | None = None,
    ) -> TranscribeResult:
        del language_policy, initial_prompt, bias_phrases
        self.seen.append(wav_bytes)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            duration_ms = (wf.getnframes() / float(wf.getframerate())) * 1000.0
        return TranscribeResult(
            transcript=self.canned_transcript,
            language=language,
            backend_name=self.backend_name,
            audio_duration_ms=duration_ms,
            decode_ms=0.0,
            segments=self.segments,
        )


def _pipeline(tmp_path, transcriber) -> OneShotDictationPipeline:
    return OneShotDictationPipeline(
        transcriber=transcriber,
        recorder=TraceRecorder(session_id="trim-test", log_dir=tmp_path),
    )


def _trace_names(pipeline: OneShotDictationPipeline) -> list[str]:
    return [event["name"] for event in pipeline.recorder.recent_events()]


def test_pipeline_hands_the_backend_the_trimmed_buffer(tmp_path) -> None:
    blob = wav_of(speech_like(2.0), room_tone(30.0))
    transcriber = RecordingTranscriber()
    pipeline = _pipeline(tmp_path, transcriber)

    result = pipeline.run(blob, save_history=False, save_audio=False)

    assert result.ok is True
    assert len(transcriber.seen) == 1
    sent = transcriber.seen[0]
    assert wav_sample_count(blob) == ms_to_samples(32_000.0)
    assert wav_sample_count(sent) == pytest.approx(
        ms_to_samples(3_500.0), abs=ms_to_samples(60.0)
    )
    assert len(sent) < len(blob)
    assert "oneshot_final_audio_silence_trimmed" in _trace_names(pipeline)


def test_pipeline_reports_the_original_recording_duration(tmp_path) -> None:
    blob = wav_of(speech_like(2.0), room_tone(30.0))
    result = _pipeline(tmp_path, RecordingTranscriber()).run(
        blob, save_history=False, save_audio=False
    )

    # The backend only saw 3.5 s; every metric we report describes the 32 s
    # the user actually recorded.
    assert result.audio_duration_ms == pytest.approx(32_000.0, abs=1.0)
    trim_meta = result.metadata.get("audio_silence_trim")
    assert isinstance(trim_meta, dict)
    assert trim_meta["trimmed"] is True
    assert trim_meta["trailing_trimmed_ms"] == pytest.approx(28_500.0, abs=60.0)


def test_restore_pretrim_timebase_shifts_segments_and_duration() -> None:
    trim = trim_wav_edge_silence(wav_of(room_tone(20.0), speech_like(2.0)))
    assert trim.trimmed is True

    decoded = TranscribeResult(
        transcript="hello world",
        language="en",
        backend_name="recording",
        audio_duration_ms=trim.trimmed_duration_ms,
        decode_ms=12.0,
        segments=(FinalSegment(start_ms=1_500.0, end_ms=3_400.0, text="hello world"),),
    )
    restored = _restore_pretrim_audio_timebase(decoded, trim)

    # 18.5 s was cut off the front, so a segment at 1.5 s in the trimmed
    # buffer sits at 20.0 s in the original recording.
    assert restored.audio_duration_ms == pytest.approx(22_000.0, abs=1.0)
    (segment,) = restored.segments
    assert segment.start_ms == pytest.approx(20_000.0, abs=60.0)
    assert segment.end_ms == pytest.approx(21_900.0, abs=60.0)
    assert segment.text == "hello world"
    # The decoded object is not mutated in place.
    assert decoded.segments[0].start_ms == 1_500.0


def test_restore_pretrim_timebase_tolerates_segments_without_timestamps() -> None:
    trim = trim_wav_edge_silence(wav_of(room_tone(20.0), speech_like(2.0)))
    decoded = TranscribeResult(
        transcript="hello world",
        language="en",
        backend_name="recording",
        audio_duration_ms=trim.trimmed_duration_ms,
        decode_ms=0.0,
        segments=("not a segment", {"text": "dict segment", "start_ms": 100.0, "end_ms": 200.0}),
    )
    restored = _restore_pretrim_audio_timebase(decoded, trim)

    assert restored.segments[0] == "not a segment"
    assert restored.segments[1]["start_ms"] == pytest.approx(18_600.0, abs=60.0)
    assert restored.segments[1]["end_ms"] == pytest.approx(18_700.0, abs=60.0)


def test_pipeline_leaves_short_recordings_untouched(tmp_path) -> None:
    blob = wav_of(speech_like(2.0), room_tone(2.0))
    transcriber = RecordingTranscriber()
    pipeline = _pipeline(tmp_path, transcriber)

    result = pipeline.run(blob, save_history=False, save_audio=False)

    assert result.ok is True
    assert transcriber.seen == [blob]
    assert "oneshot_final_audio_silence_trimmed" not in _trace_names(pipeline)
    assert result.metadata.get("audio_silence_trim") is None
    assert result.audio_duration_ms == pytest.approx(4_000.0, abs=1.0)


def test_low_signal_reject_still_fires_and_never_reaches_the_backend(tmp_path) -> None:
    # Pure silence must still be rejected wholesale, exactly as before —
    # the trimmer must not turn a low-signal buffer into a "trimmed" one.
    blob = encode_wav_bytes(np.zeros(int(0.8 * SR), dtype=np.float32), SR)
    transcriber = RecordingTranscriber()
    pipeline = _pipeline(tmp_path, transcriber)

    result = pipeline.run(blob, save_history=False, save_audio=False)

    assert result.ok is False
    assert result.error_code == "low_signal_audio"
    assert result.noop_reason == "low_signal_audio"
    assert transcriber.seen == []
    assert result.audio_duration_ms == pytest.approx(800.0, abs=1.0)
    names = _trace_names(pipeline)
    assert "oneshot_low_signal_audio_rejected" in names
    assert "oneshot_final_audio_silence_trimmed" not in names


def test_long_room_tone_recording_still_rejected_as_low_signal(tmp_path) -> None:
    # A 60 s buffer of nothing but room tone: the trimmer finds no speech
    # and leaves it alone, and the low-signal gate rejects it as before.
    blob = wav_of(room_tone(60.0))
    transcriber = RecordingTranscriber()
    pipeline = _pipeline(tmp_path, transcriber)

    result = pipeline.run(blob, save_history=False, save_audio=False)

    assert result.ok is False
    assert result.error_code == "low_signal_audio"
    assert transcriber.seen == []
