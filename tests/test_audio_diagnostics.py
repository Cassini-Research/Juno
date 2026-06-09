"""Tests for juno_v2.audio.diagnostics and the one-shot pipeline low-signal gate."""

from __future__ import annotations

import numpy as np
import pytest

from juno_core_v3.dictation.pipeline import OneShotDictationPipeline
from juno_core_v3.dictation.transcriber import StubTranscriber
from juno_v2.asr.wav import encode_wav_bytes
from juno_v2.audio.diagnostics import analyze_audio_signal, analyze_wav_bytes
from juno_v2.observability.tracing import TraceRecorder

from tests.audio_fixtures import (
    require_say,
    silence,
    sine_tone,
    tts_wav_bytes,
    white_noise,
)


# ------------------------------------------------------------------ #
# analyze_audio_signal
# ------------------------------------------------------------------ #

def test_empty_audio_is_low_signal() -> None:
    diag = analyze_audio_signal([])
    assert diag.low_signal is True
    assert diag.speech_likely is False
    assert diag.reason == "empty_audio"
    assert diag.duration_ms == 0.0
    assert diag.zero_ratio == 1.0


def test_pure_silence_is_low_signal_mostly_zero() -> None:
    diag = analyze_audio_signal(silence(0.6))
    assert diag.low_signal is True
    assert diag.reason == "mostly_zero"
    assert diag.speech_likely is False
    assert diag.rms == pytest.approx(0.0)
    assert diag.peak == pytest.approx(0.0)
    assert diag.zero_ratio == pytest.approx(1.0)
    assert diag.non_silent_ms == 0.0
    assert diag.duration_ms == pytest.approx(600.0)


def test_short_burst_is_too_short() -> None:
    # 20 ms < the 40 ms floor — even a loud burst is rejected as too_short.
    diag = analyze_audio_signal(sine_tone(duration_s=0.02, amplitude=0.5))
    assert diag.low_signal is True
    assert diag.reason == "too_short"
    assert diag.speech_likely is False


def test_loud_tone_is_speech_likely() -> None:
    diag = analyze_audio_signal(sine_tone(duration_s=1.0, amplitude=0.3))
    assert diag.low_signal is False
    assert diag.reason is None
    assert diag.speech_likely is True
    assert diag.non_silent_ms >= 160.0
    assert diag.peak == pytest.approx(0.3, rel=0.05)
    assert diag.clipped_ratio == 0.0


def test_tiny_noise_floor_is_below_speech_floor() -> None:
    # Long buffer of sub-threshold noise: the class that produced phantom
    # "Yeah" / "Thank you" transcripts in real logs.
    diag = analyze_audio_signal(white_noise(duration_s=2.0, amplitude=0.004))
    assert diag.low_signal is True
    assert diag.reason == "below_speech_floor"
    assert diag.speech_likely is False


def test_clipped_ratio_reports_full_scale_samples() -> None:
    arr = np.ones(16_000, dtype=np.float32)
    diag = analyze_audio_signal(arr)
    assert diag.clipped_ratio == pytest.approx(1.0)
    assert diag.peak == pytest.approx(1.0)


def test_to_dict_round_trip_keys() -> None:
    diag = analyze_audio_signal(sine_tone(duration_s=0.5, amplitude=0.2))
    payload = diag.to_dict()
    assert payload["low_signal"] is False
    assert payload["speech_likely"] is True
    assert set(payload) == {
        "duration_ms",
        "rms",
        "peak",
        "non_silent_ms",
        "clipped_ratio",
        "zero_ratio",
        "low_signal",
        "speech_likely",
        "reason",
    }


# ------------------------------------------------------------------ #
# analyze_wav_bytes
# ------------------------------------------------------------------ #

def test_analyze_wav_bytes_non_wav_returns_none() -> None:
    assert analyze_wav_bytes(b"definitely not a wav file") is None
    assert analyze_wav_bytes(b"") is None


def test_analyze_wav_bytes_silence() -> None:
    blob = encode_wav_bytes(silence(0.5), 16_000)
    diag = analyze_wav_bytes(blob)
    assert diag is not None
    assert diag.low_signal is True
    assert diag.reason == "mostly_zero"


def test_analyze_wav_bytes_loud_tone() -> None:
    blob = encode_wav_bytes(sine_tone(duration_s=1.0, amplitude=0.3), 16_000)
    diag = analyze_wav_bytes(blob)
    assert diag is not None
    assert diag.low_signal is False
    assert diag.speech_likely is True
    assert diag.duration_ms == pytest.approx(1000.0, abs=1.0)


def test_analyze_wav_bytes_real_tts_speech() -> None:
    require_say()
    diag = analyze_wav_bytes(tts_wav_bytes("hello world this is a dictation test"))
    assert diag is not None
    assert diag.low_signal is False
    assert diag.speech_likely is True
    assert diag.non_silent_ms >= 160.0
    assert diag.duration_ms > 500.0


# ------------------------------------------------------------------ #
# OneShotDictationPipeline low-signal gate
# ------------------------------------------------------------------ #

def _pipeline(tmp_path) -> OneShotDictationPipeline:
    return OneShotDictationPipeline(
        transcriber=StubTranscriber(canned_transcript="should never be pasted"),
        recorder=TraceRecorder(session_id="diag-test", log_dir=tmp_path),
    )


def test_pipeline_rejects_empty_audio(tmp_path) -> None:
    result = _pipeline(tmp_path).run(b"", save_history=False, save_audio=False)
    assert result.ok is False
    assert result.error_code == "empty_audio"
    assert result.error == "empty_audio"
    assert result.transcript == ""
    assert result.paste_kind == "none"
    assert result.noop_reason == "empty_audio"


def test_pipeline_rejects_low_signal_audio(tmp_path) -> None:
    blob = encode_wav_bytes(silence(0.8), 16_000)
    result = _pipeline(tmp_path).run(blob, save_history=False, save_audio=False)
    assert result.ok is False
    assert result.error_code == "low_signal_audio"
    assert result.transcript == ""
    assert result.paste_kind == "none"
    assert result.noop_reason == "low_signal_audio"
    diag = result.metadata.get("audio_diagnostics")
    assert isinstance(diag, dict)
    assert diag["low_signal"] is True
    assert diag["reason"] == "mostly_zero"
    assert result.audio_duration_ms == pytest.approx(800.0, abs=1.0)


def test_pipeline_passes_speech_like_audio_through_gate(tmp_path) -> None:
    # A loud, long tone clears the low-signal gate and reaches the stub
    # transcriber; the canned transcript flows through to the result.
    blob = encode_wav_bytes(sine_tone(duration_s=1.0, amplitude=0.3), 16_000)
    result = _pipeline(tmp_path).run(blob, save_history=False, save_audio=False)
    assert result.ok is True
    assert result.error_code is None
    assert "should never be pasted" in result.transcript
