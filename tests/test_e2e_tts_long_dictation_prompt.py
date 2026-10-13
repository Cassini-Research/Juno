"""End-to-end: a ~49 s macOS `say` dictation through the real MLX Whisper final
backend keeps its dictionary prompt past the first 30 s window (issue #118).

Marked slow: loads the production final model (cached at ~/.cache/huggingface).
Skipped when `say`, mlx-whisper or the cached model is not available.
"""

from __future__ import annotations

import functools
import re

import pytest

from juno_core_v3.dictation.transcriber import FinalBackendTranscriber
from juno_v2.final.config import FinalAsrConfig

from tests.audio_fixtures import SAMPLE_RATE_HZ, say_available, tts_speech_array, tts_wav_bytes

MLX_TURBO_REPO = "mlx-community/whisper-large-v3-turbo"

# Invented product names: without the prompt Whisper writes each one as the
# ordinary words it sounds like ("Brightwave", "synchro", ...).
INVENTED_TERMS = ("Brytewave", "Synkro", "Lumenyx", "Flowmatiq")
PROMPT = "App: TestApp | Category: terminal | Prefer exact forms: " + ", ".join(INVENTED_TERMS)

# ~34 s of ordinary sentences, so the invented names below are all spoken
# after Whisper's first 30 s window has ended.
OPENING_SENTENCES = (
    "This is a long dictation that checks the vocabulary prompt.",
    "I am going to talk for a while about nothing in particular.",
    "The garden behind the house needs water every morning in summer.",
    "Yesterday we walked along the river and counted nine small boats.",
    "My sister keeps a notebook where she writes down every recipe she tries.",
    "The library closes early on Fridays, so return the books before lunch.",
    "After the meeting, send the summary to everyone who could not attend.",
    "The bus to the station leaves every twenty minutes from the corner.",
)
CLOSING_SENTENCES = (
    "Now open Brytewave and export the weekly report.",
    "Then start a Synkro job for the new customer.",
    "Ask the Lumenyx team to restart the service.",
    "Finally, send the Flowmatiq invoice and close Brytewave.",
)
# `say` renders the embedded command as 700 ms of silence between sentences.
SENTENCE_BREAK = " [[slnc 700]] "
TTS_TEXT = SENTENCE_BREAK.join(OPENING_SENTENCES + CLOSING_SENTENCES)
# One distinctive word per sentence, in spoken order.
LANDMARK_WORDS = (
    "vocabulary", "particular", "garden", "boats", "notebook", "library",
    "summary", "station", "weekly", "customer", "service", "invoice",
)
FIRST_WINDOW_S = 30.0

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not say_available(), reason="macOS `say` binary not available"),
]


@functools.lru_cache(maxsize=1)
def _mlx_transcriber() -> FinalBackendTranscriber:
    pytest.importorskip("mlx_whisper", reason="mlx-whisper is only installed on Apple Silicon")
    from huggingface_hub import snapshot_download

    from juno_v2.final.backends.mlx_whisper import MlxWhisperFinalBackend

    try:
        model_path = snapshot_download(MLX_TURBO_REPO, local_files_only=True)
    except Exception:
        pytest.skip(f"{MLX_TURBO_REPO} not present in the local HF cache")
    config = FinalAsrConfig(
        model_path=model_path,
        backend_name="mlx_whisper",
        language="en",
        hf_repo_id=MLX_TURBO_REPO,
    )
    return FinalBackendTranscriber(backend=MlxWhisperFinalBackend(config), language="en")


def test_invented_terms_are_honoured_after_the_first_whisper_window(monkeypatch) -> None:
    transcriber = _mlx_transcriber()
    import mlx_whisper.decoding as mlx_decoding

    # Observe the prompt each Whisper window is actually decoded with.
    prompt_tokens_per_window: list[int] = []
    original_init = mlx_decoding.DecodingTask.__init__

    def recording_init(self, model, options):
        prompt = options.prompt
        prompt_tokens_per_window.append(len(prompt) if prompt and not isinstance(prompt, str) else int(bool(prompt)))
        return original_init(self, model, options)

    monkeypatch.setattr(mlx_decoding.DecodingTask, "__init__", recording_init)

    result = transcriber.transcribe_wav(
        tts_wav_bytes(TTS_TEXT),
        language="en",
        language_policy="fixed",
        initial_prompt=PROMPT,
        bias_phrases=list(INVENTED_TERMS),
    )

    assert result.backend_name == "mlx_whisper"
    # The invented names are only spoken once the opening is over, and the
    # opening alone outlasts Whisper's first window.
    opening_s = tts_speech_array(SENTENCE_BREAK.join(OPENING_SENTENCES)).shape[0] / SAMPLE_RATE_HZ
    assert opening_s > FIRST_WINDOW_S, f"opening is only {opening_s:.1f} s; the fixture needs more than one window"
    honoured = [term for term in INVENTED_TERMS if term in result.transcript]
    # All four on the voice this was written against; one is left as slack
    # for TTS voice revisions. Before the fix none survived the first window.
    assert len(honoured) >= 3, (
        f"dictionary terms spelled exactly (all spoken after {opening_s:.1f} s): {honoured}; "
        f"transcript: {result.transcript!r}"
    )
    prompted_windows = [count for count in prompt_tokens_per_window if count > 0]
    assert len(prompted_windows) >= 2, (
        f"prompt tokens per decoded Whisper window: {prompt_tokens_per_window}"
    )


def test_long_tts_dictation_is_transcribed_once_and_in_order() -> None:
    result = _mlx_transcriber().transcribe_wav(
        tts_wav_bytes(TTS_TEXT),
        language="en",
        language_policy="fixed",
        initial_prompt=PROMPT,
        bias_phrases=list(INVENTED_TERMS),
    )

    spoken = re.findall(r"[a-z']+", result.transcript.lower())
    positions = []
    for word in LANDMARK_WORDS:
        assert spoken.count(word) == 1, f"{word!r} appears {spoken.count(word)} times in {result.transcript!r}"
        positions.append(spoken.index(word))
    assert positions == sorted(positions), f"sentences out of order in {result.transcript!r}"
    # Segments stay on the recording's timebase across chunk boundaries, and
    # the transcript is still exactly its segments joined.
    starts = [segment.start_ms for segment in result.segments]
    assert starts == sorted(starts)
    assert all(0.0 <= start < result.audio_duration_ms for start in starts)
    assert " ".join(segment.text for segment in result.segments) == result.transcript
