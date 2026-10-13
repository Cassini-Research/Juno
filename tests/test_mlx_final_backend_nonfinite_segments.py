"""Final-lane handling of mlx_whisper segments with a non-finite confidence.

mlx_whisper can force token id 0 ("!") with a NaN log-probability when its
timestamp rules mask every logit of a decode step. The affected 30 s decode
window then reports ``avg_logprob = NaN`` on all of its segments: the forced
"!" segments and the real speech that follows them.

Every transcript and number below is a synthetic fixture written for these
tests. The ``mlx_whisper`` module is replaced by an object whose
``transcribe`` returns a canned result shaped like the real one (segment texts
tile ``text`` with no separator and carry their own leading space), so the
real ``MlxWhisperFinalBackend.decode`` post-processing runs unmodified.
"""

from __future__ import annotations

import io
import math
import tempfile
import threading
import wave

import numpy as np
import pytest

from juno_core_v3.dictation.pipeline import OneShotDictationPipeline, OneShotDictationResult
from juno_core_v3.dictation.transcriber import FinalBackendTranscriber
from juno_v2.contracts.final import FinalDecodeRequest, FinalDecodeResult
from juno_v2.final.backends.mlx_whisper import MlxWhisperFinalBackend
from juno_v2.final.config import FinalAsrConfig
from juno_v2.transcript.adjudicator import TranscriptAdjudicatorConfig


NAN = float("nan")
SPEECH = "Move the quarterly planning notes into the shared folder before lunch."
SPEECH_TWO = "Then send the summary to the review channel."
SHORT_SPEECH = "Archive this thread."


class _FakeMlxWhisperModule:
    """Stands in for the imported ``mlx_whisper`` module; decode only calls ``transcribe``."""

    def __init__(self, result: dict[str, object]) -> None:
        self._result = result

    def transcribe(self, audio: object, **kwargs: object) -> dict[str, object]:
        del audio, kwargs
        return self._result


class _CannedMlxBackend(MlxWhisperFinalBackend):
    """The real backend with the model load replaced by a canned ``transcribe``."""

    def __init__(self, transcribe_result: dict[str, object]) -> None:
        super().__init__(FinalAsrConfig(model_path="synthetic-mlx-model", backend_name="mlx_whisper"))
        self._transcribe_result = transcribe_result
        self._model_ref = "synthetic-mlx-model"

    def warm(self) -> None:
        self._mlx_whisper = _FakeMlxWhisperModule(self._transcribe_result)
        self._model_thread_id = threading.get_ident()

    def unload(self) -> None:
        self._mlx_whisper = None
        self._model_thread_id = None


def _seg(
    text: str,
    start: float,
    end: float,
    *,
    avg_logprob: float | None = -0.25,
    no_speech_prob: float | None = 0.125,
    compression_ratio: float | None = 1.5,
) -> dict[str, object]:
    return {
        "text": text,
        "start": start,
        "end": end,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
    }


def _transcribe_result(*segments: dict[str, object], text: str | None = None) -> dict[str, object]:
    joined = "".join(str(segment["text"]) for segment in segments)
    return {"text": joined if text is None else text, "segments": list(segments), "language": "en"}


def _decode(transcribe_result: dict[str, object], *, seconds: float = 8.0) -> FinalDecodeResult:
    request = FinalDecodeRequest(
        utterance_id="utt-synthetic-backend",
        audio=np.zeros(int(16_000 * seconds), dtype=np.float32),
        sample_rate_hz=16_000,
        start_ms=0.0,
        end_ms=seconds * 1000.0,
    )
    return _CannedMlxBackend(transcribe_result).decode(request)


class _Recorder:
    def __init__(self) -> None:
        self.log_dir = tempfile.mkdtemp(prefix="juno-nonfinite-segments-test-")
        self.event_names: list[str] = []

    def record(self, *args: object, **kwargs: object) -> None:
        del kwargs
        if len(args) >= 2 and isinstance(args[1], str):
            self.event_names.append(args[1])


def _tone_wav_bytes(seconds: float) -> bytes:
    sample_rate = 16_000
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    pcm = (9000 * np.sin(2 * np.pi * 440 * t)).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def _run_oneshot(
    transcribe_result: dict[str, object], *, seconds: float = 8.0
) -> tuple[OneShotDictationResult, _Recorder]:
    """The production one-shot path: pipeline -> FinalBackendTranscriber -> mlx backend."""
    recorder = _Recorder()
    pipeline = OneShotDictationPipeline(
        transcriber=FinalBackendTranscriber(backend=_CannedMlxBackend(transcribe_result), language="en"),
        recorder=recorder,
        writer_enabled=False,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
    )
    result = pipeline.run(
        _tone_wav_bytes(seconds),
        utterance_id="utt-synthetic-oneshot",
        save_history=False,
        save_audio=False,
        app_bundle_id="com.apple.TextEdit",
    )
    return result, recorder


# --------------------------------------------------------------------------- #
# Dropping the forced-token segments
# --------------------------------------------------------------------------- #


def test_leading_forced_token_segments_are_dropped_from_text_and_segments() -> None:
    result = _decode(
        _transcribe_result(
            _seg("!", 0.0, 1.0, avg_logprob=NAN),
            _seg("!", 1.0, 2.0, avg_logprob=NAN),
            _seg(" " + SPEECH, 2.0, 8.0, avg_logprob=NAN),
        )
    )

    assert result.text == SPEECH
    assert [segment.text for segment in result.segments] == [SPEECH]
    assert result.metadata["dropped_artifact_segments"] == 2


@pytest.mark.parametrize(
    ("segments", "expected_text"),
    [
        pytest.param(
            (
                _seg("!", 0.0, 1.0, avg_logprob=NAN),
                _seg(" " + SPEECH, 1.0, 6.0, avg_logprob=NAN),
                _seg(" " + SPEECH_TWO, 6.0, 9.0, avg_logprob=NAN),
            ),
            f"{SPEECH} {SPEECH_TWO}",
            id="head",
        ),
        pytest.param(
            (
                _seg(" " + SPEECH, 0.0, 5.0, avg_logprob=NAN),
                _seg("!", 5.0, 6.0, avg_logprob=NAN),
                _seg(" " + SPEECH_TWO, 6.0, 9.0, avg_logprob=NAN),
            ),
            f"{SPEECH} {SPEECH_TWO}",
            id="middle",
        ),
        pytest.param(
            (
                _seg(" " + SPEECH, 0.0, 5.0, avg_logprob=NAN),
                _seg(" " + SPEECH_TWO, 5.0, 8.0, avg_logprob=NAN),
                _seg("!", 8.0, 9.0, avg_logprob=NAN),
                _seg("!", 9.0, 10.0, avg_logprob=NAN),
            ),
            f"{SPEECH} {SPEECH_TWO}",
            id="tail",
        ),
        pytest.param(
            # mlx_whisper clears instantaneous segments to an empty text; they
            # must not disturb the re-assembled transcript.
            (
                _seg(" " + SPEECH, 0.0, 5.0, avg_logprob=NAN),
                _seg("", 5.0, 5.0, avg_logprob=NAN),
                _seg("!", 5.0, 6.0, avg_logprob=NAN),
                _seg(" " + SPEECH_TWO, 6.0, 9.0, avg_logprob=NAN),
            ),
            f"{SPEECH} {SPEECH_TWO}",
            id="next-to-cleared-segment",
        ),
    ],
)
def test_forced_token_segments_are_dropped_wherever_they_sit(
    segments: tuple[dict[str, object], ...], expected_text: str
) -> None:
    result = _decode(_transcribe_result(*segments), seconds=10.0)

    assert result.text == expected_text
    assert [segment.text for segment in result.segments] == [SPEECH, SPEECH_TWO]


@pytest.mark.parametrize("artifact_text", ["!", "!!", "...", "?!", "…", "♪"])
def test_any_segment_without_alphanumeric_content_and_nonfinite_confidence_is_dropped(
    artifact_text: str,
) -> None:
    result = _decode(
        _transcribe_result(
            _seg(artifact_text, 0.0, 1.0, avg_logprob=NAN),
            _seg(" " + SPEECH, 1.0, 6.0, avg_logprob=NAN),
        )
    )

    assert result.text == SPEECH
    assert [segment.text for segment in result.segments] == [SPEECH]


def test_utterance_that_is_only_forced_tokens_decodes_to_empty_text() -> None:
    result = _decode(
        _transcribe_result(
            _seg("!", 0.0, 1.0, avg_logprob=NAN),
            _seg("!", 1.0, 2.0, avg_logprob=NAN),
        )
    )

    assert result.text == ""
    assert result.segments == []
    assert result.metadata["avg_logprob"] is None
    assert result.metadata["no_speech_prob"] is None
    assert result.metadata["compression_ratio"] is None


# --------------------------------------------------------------------------- #
# What must NOT be dropped
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("non_finite", [NAN, float("inf"), float("-inf")])
def test_alphanumeric_segment_with_nonfinite_confidence_keeps_its_text(non_finite: float) -> None:
    result = _decode(_transcribe_result(_seg(" " + SPEECH, 0.0, 6.0, avg_logprob=non_finite)))

    assert result.text == SPEECH
    assert [segment.text for segment in result.segments] == [SPEECH]
    # The words are real; only their confidence is unknown.
    assert result.segments[0].avg_logprob is None
    assert result.metadata["avg_logprob"] is None


@pytest.mark.parametrize("script_text", ["你好", "नमस्ते", "42"])
def test_non_latin_and_numeric_segments_count_as_alphanumeric(script_text: str) -> None:
    result = _decode(_transcribe_result(_seg(" " + script_text, 0.0, 2.0, avg_logprob=NAN)))

    assert result.text == script_text
    assert [segment.text for segment in result.segments] == [script_text]


@pytest.mark.parametrize("punctuation", ["?", "...", "!"])
def test_punctuation_only_segment_with_finite_confidence_is_kept(punctuation: str) -> None:
    result = _decode(
        _transcribe_result(
            _seg(" Is the staging build green", 0.0, 3.0, avg_logprob=-0.25),
            _seg(punctuation, 3.0, 3.5, avg_logprob=-0.5),
        )
    )

    assert result.text == f"Is the staging build green{punctuation}"
    assert [segment.text for segment in result.segments] == ["Is the staging build green", punctuation]
    assert result.segments[1].avg_logprob == -0.5


def test_punctuation_only_segment_with_missing_confidence_is_kept() -> None:
    # A backend result that reports no confidence gives no evidence of the
    # masked-logits step, so "missing" is not treated like "non-finite".
    result = _decode(
        _transcribe_result(
            _seg(" Is the staging build green", 0.0, 3.0, avg_logprob=None),
            _seg("?", 3.0, 3.5, avg_logprob=None),
        )
    )

    assert result.text == "Is the staging build green?"
    assert [segment.text for segment in result.segments] == ["Is the staging build green", "?"]


# --------------------------------------------------------------------------- #
# Utterance-level averages
# --------------------------------------------------------------------------- #


def test_utterance_confidence_averages_only_the_finite_segments() -> None:
    # First decode window is poisoned (forced token + real speech, all NaN);
    # the second window reports finite confidence.
    result = _decode(
        _transcribe_result(
            _seg("!", 0.0, 1.0, avg_logprob=NAN),
            _seg(" " + SPEECH, 1.0, 5.0, avg_logprob=NAN),
            _seg(" " + SPEECH_TWO, 30.0, 34.0, avg_logprob=-0.5),
            _seg(" Thanks for the quick turnaround.", 34.0, 36.0, avg_logprob=-0.25),
        ),
        seconds=36.0,
    )

    avg_logprob = result.metadata["avg_logprob"]
    assert avg_logprob is not None and math.isfinite(avg_logprob)
    # Duration-weighted over the two finite segments only: 4 s and 2 s.
    assert avg_logprob == pytest.approx((-0.5 * 4.0 + -0.25 * 2.0) / 6.0)
    # The other two signals are finite on every kept segment (10 s in total)
    # and exclude the dropped forced-token segment.
    assert result.metadata["no_speech_prob"] == pytest.approx(0.125)
    assert result.metadata["compression_ratio"] == pytest.approx(1.5)
    assert [segment.avg_logprob for segment in result.segments] == [None, -0.5, -0.25]


@pytest.mark.parametrize("signal", ["avg_logprob", "no_speech_prob", "compression_ratio"])
@pytest.mark.parametrize("non_finite", [NAN, float("inf"), float("-inf")])
def test_nonfinite_segment_value_never_enters_its_utterance_average(signal: str, non_finite: float) -> None:
    finite = {"avg_logprob": -0.5, "no_speech_prob": 0.25, "compression_ratio": 2.0}
    result = _decode(
        _transcribe_result(
            _seg(" " + SPEECH, 0.0, 5.0, **{signal: non_finite}),
            _seg(" " + SPEECH_TWO, 5.0, 8.0, **finite),
        )
    )

    assert result.text == f"{SPEECH} {SPEECH_TWO}"
    # Only the second segment contributes to the affected signal.
    assert result.metadata[signal] == pytest.approx(finite[signal])
    assert getattr(result.segments[0], signal) is None
    assert getattr(result.segments[1], signal) == finite[signal]
    for name in ("avg_logprob", "no_speech_prob", "compression_ratio"):
        assert math.isfinite(result.metadata[name])


def test_output_is_unchanged_when_every_segment_is_finite() -> None:
    # ``text`` deliberately differs from the joined segment texts (doubled
    # space, newline, outer padding): with nothing dropped it must be passed
    # through as ``text.strip()``, never re-assembled from the segments.
    raw_text = f"  {SPEECH}  \n{SPEECH_TWO}  "
    segments = (
        _seg(" " + SPEECH, 0.0, 4.7, avg_logprob=-0.31, no_speech_prob=0.02, compression_ratio=1.37),
        _seg(" " + SPEECH_TWO, 4.7, 7.9, avg_logprob=-0.58, no_speech_prob=0.11, compression_ratio=1.12),
        _seg("?", 7.9, 7.9, avg_logprob=-0.9, no_speech_prob=0.4, compression_ratio=0.8),
    )

    result = _decode(_transcribe_result(*segments, text=raw_text))

    assert result.text == raw_text.strip()
    assert [segment.text for segment in result.segments] == [SPEECH, SPEECH_TWO, "?"]
    # Reference: one shared duration weight per segment (floored at 50 ms),
    # accumulated in segment order — the formula used before non-finite
    # values were filtered. Equality is exact, not approximate.
    weighted = {"avg_logprob": 0.0, "no_speech_prob": 0.0, "compression_ratio": 0.0}
    total_weight = 0.0
    for segment in segments:
        weight = max(float(segment["end"]) - float(segment["start"]), 0.05)
        for name in weighted:
            weighted[name] += float(segment[name]) * weight
        total_weight += weight
    for name, numerator in weighted.items():
        assert result.metadata[name] == numerator / total_weight
    assert [segment.avg_logprob for segment in result.segments] == [-0.31, -0.58, -0.9]
    assert [segment.no_speech_prob for segment in result.segments] == [0.02, 0.11, 0.4]
    assert [segment.compression_ratio for segment in result.segments] == [1.37, 1.12, 0.8]


# --------------------------------------------------------------------------- #
# Production one-shot path: pipeline -> FinalBackendTranscriber -> backend
# --------------------------------------------------------------------------- #


def test_oneshot_pipeline_commits_speech_without_the_leading_artifact() -> None:
    result, recorder = _run_oneshot(
        _transcribe_result(
            _seg("!", 0.0, 1.0, avg_logprob=NAN),
            _seg("!", 1.0, 2.0, avg_logprob=NAN),
            _seg(" " + SPEECH, 2.0, 8.0, avg_logprob=NAN),
        )
    )

    assert result.ok
    assert result.error_code is None
    assert result.raw_transcript == SPEECH
    assert result.transcript == SPEECH
    assert result.paste_kind == "insert"
    assert "oneshot_hallucination_rejected" not in recorder.event_names
    assert "oneshot_silence_hallucination_rejected" not in recorder.event_names


def test_oneshot_pipeline_keeps_a_short_utterance_from_a_poisoned_window() -> None:
    # A NaN utterance confidence used to satisfy the low-yield garbage guard
    # (NaN is not > the floor), so a dictation of four words or fewer that
    # followed the artefact was rejected outright instead of pasted.
    result, recorder = _run_oneshot(
        _transcribe_result(
            _seg("!", 0.0, 1.0, avg_logprob=NAN),
            _seg(" " + SHORT_SPEECH, 3.0, 4.5, avg_logprob=NAN),
        ),
        seconds=6.0,
    )

    assert result.ok
    assert result.error_code is None
    assert result.transcript == SHORT_SPEECH
    assert result.paste_kind == "insert"
    assert "oneshot_hallucination_rejected" not in recorder.event_names


def test_oneshot_pipeline_still_pastes_nothing_for_an_artifact_only_recording() -> None:
    result, _recorder = _run_oneshot(
        _transcribe_result(
            _seg("!", 0.0, 1.0, avg_logprob=NAN),
            _seg("!", 1.0, 2.0, avg_logprob=NAN),
        ),
        seconds=6.0,
    )

    assert result.transcript == ""
    assert result.paste_kind == "none"
