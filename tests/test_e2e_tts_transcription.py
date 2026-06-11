"""End-to-end: macOS `say` TTS audio through the real faster-whisper tiny.en
model, both via the FinalBackendTranscriber wrapper and through the full
OneShotDictationPipeline.

Marked slow: loads a real ASR model (cached at ~/.cache/huggingface).
"""

from __future__ import annotations

import functools
import re

import pytest

from juno_core_v3.dictation.pipeline import OneShotDictationPipeline
from juno_core_v3.dictation.transcriber import FinalBackendTranscriber
from juno_v2.final.config import FinalAsrConfig
from juno_v2.observability.tracing import TraceRecorder

from tests.audio_fixtures import say_available, tts_wav_bytes

TTS_SENTENCE = "hello world this is a dictation test"
KEY_WORDS = ("hello", "world", "dictation", "test")
TINY_EN_REPO = "Systran/faster-whisper-tiny.en"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not say_available(), reason="macOS `say` binary not available"),
]


def _tiny_en_model_path() -> str:
    """Resolve the cached tiny.en snapshot without touching the network."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(TINY_EN_REPO, local_files_only=True)
    except Exception:
        pytest.skip(f"{TINY_EN_REPO} not present in the local HF cache")


@functools.lru_cache(maxsize=1)
def _real_transcriber() -> FinalBackendTranscriber:
    from juno_v2.final.backends.faster_whisper import FasterWhisperFinalBackend

    config = FinalAsrConfig(
        model_path=_tiny_en_model_path(),
        backend_name="faster_whisper",
        language="en",
        device="cpu",
        compute_type="default",
    )
    return FinalBackendTranscriber(backend=FasterWhisperFinalBackend(config), language="en")


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def test_tts_wav_transcribes_to_expected_words() -> None:
    wav_bytes = tts_wav_bytes(TTS_SENTENCE)
    result = _real_transcriber().transcribe_wav(wav_bytes, language="en")

    assert result.backend_name == "faster_whisper"
    assert result.transcript.strip(), "expected a non-empty transcript"
    words = _normalized_words(result.transcript)
    missing = [w for w in KEY_WORDS if w not in words]
    assert not missing, (
        f"transcript {result.transcript!r} missing key words {missing}"
    )
    # ~0.8 s of TTS audio; allow generous bounds for voice pacing.
    assert 400.0 < result.audio_duration_ms < 5000.0
    assert result.decode_ms > 0.0
    assert result.model_path  # provenance is stamped


def test_tts_wav_through_oneshot_pipeline_with_real_transcriber(tmp_path) -> None:
    pipeline = OneShotDictationPipeline(
        transcriber=_real_transcriber(),
        recorder=TraceRecorder(session_id="e2e-tts", log_dir=tmp_path),
        writer_enabled=False,
    )
    result = pipeline.run(
        tts_wav_bytes(TTS_SENTENCE),
        language="en",
        save_history=False,
        save_audio=False,
    )

    assert result.ok is True, f"pipeline failed: {result.error_code} {result.error}"
    assert result.error_code is None
    words = _normalized_words(result.transcript)
    missing = [w for w in KEY_WORDS if w not in words]
    assert not missing, (
        f"pipeline transcript {result.transcript!r} missing key words {missing}"
    )
    assert result.backend_name == "faster_whisper"
    assert result.raw_transcript.strip()
    assert result.paste_kind == "insert"
