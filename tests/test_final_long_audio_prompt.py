"""Final decodes longer than one Whisper window keep the bias prompt (issue #118).

Whisper's ``transcribe()`` applies ``initial_prompt`` to its first 30 s window
only (with ``condition_on_previous_text=False``), so a long dictation lost its
dictionary after the first half minute. These tests drive the real
``MlxWhisperFinalBackend`` against :class:`FakeWhisper`, a deterministic
stand-in for the ``mlx_whisper`` module that reproduces exactly that
windowing rule on synthetic audio, so no model is loaded.

The synthetic recordings are made of "words": bursts whose level encodes the
word's index, separated by digital silence. The fake spells a word in its
dictionary form (``W7``) when the window it was decoded in had the prompt and
in its plain form (``w7``) when it did not, so a transcript shows which parts
of the recording the prompt reached.
"""

from __future__ import annotations

import contextlib
import threading

import numpy as np
import pytest

from juno_v2.contracts.final import FinalDecodeRequest
from juno_v2.final.backends import mlx_whisper as mlx_backend_module
from juno_v2.final.backends.mlx_whisper import MlxWhisperFinalBackend
from juno_v2.final.config import FinalAsrConfig
from juno_v2.memory.hallucination import strip_trailing_silence_hallucination

SR = 16_000
WINDOW = 30 * SR
HOP = 160
PROMPT = "App: Notes | Category: editor | Prefer exact forms: Zentrix, Lumaro"
MODEL_REF = "stub/whisper"

_BASE_LEVEL = 0.2
_LEVEL_STEP = 0.002


# ------------------------------------------------------------------ #
# Synthetic recordings
# ------------------------------------------------------------------ #

def at(seconds: float) -> int:
    """Sample index of a time in seconds."""
    return int(round(seconds * SR))


def recording(*parts: tuple[str, float]) -> np.ndarray:
    """``("word", seconds)`` bursts and ``("gap", seconds)`` of digital silence."""
    chunks: list[np.ndarray] = []
    index = 0
    for kind, seconds in parts:
        count = at(seconds)
        if kind == "word":
            level = _BASE_LEVEL + _LEVEL_STEP * index
            signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
            chunks.append((level * signs).astype(np.float32))
            index += 1
        else:
            chunks.append(np.zeros(count, dtype=np.float32))
    return np.concatenate(chunks)


def sentences(count: int, *, words: int = 5, word_s: float = 0.5, gap_s: float = 0.1, pause_s: float = 0.6):
    """Parts for *count* sentences of short-gapped words, a pause after each."""
    parts: list[tuple[str, float]] = []
    for _ in range(count):
        for position in range(words):
            parts.append(("word", word_s))
            if position < words - 1:
                parts.append(("gap", gap_s))
        parts.append(("gap", pause_s))
    return parts


def word_runs(audio: np.ndarray) -> list[tuple[int, int, int]]:
    """``(word_index, start_sample, end_sample)`` for every burst in *audio*."""
    sounding = np.abs(audio) > 0.0
    edges = np.diff(np.concatenate(([0], sounding.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1).tolist()
    ends = np.flatnonzero(edges == -1).tolist()
    runs = []
    for start, end in zip(starts, ends):
        level = float(np.abs(audio[start:end]).max())
        runs.append((int(round((level - _BASE_LEVEL) / _LEVEL_STEP)), start, end))
    return runs


# ------------------------------------------------------------------ #
# A stand-in for the ``mlx_whisper`` module
# ------------------------------------------------------------------ #

class FakeWhisper:
    """Windows audio the way Whisper's ``transcribe()`` does.

    * Decodes 30 s windows; only the first window of a call gets
      ``initial_prompt`` (``condition_on_previous_text`` is off).
    * A window ends at its last complete word when a word runs past it, and
      the next window starts there (Whisper's "seek to the last timestamp").
      ``first_window_limit_s`` / ``later_window_limit_s`` make a window give
      up that early even though more audio was in reach.
    * A burst touching either end of the audio it was handed is a word the
      caller cut in two; it is transcribed with a ``~`` marker.
    """

    def __init__(
        self,
        *,
        first_window_limit_s: float | None = None,
        later_window_limit_s: float | None = None,
        derail_prompted_calls: tuple[int, ...] = (),
        hallucinate_after_silence_s: float | None = None,
        detected_language: str = "en",
    ) -> None:
        self.first_window_limit_s = first_window_limit_s
        self.later_window_limit_s = later_window_limit_s
        self.derail_prompted_calls = set(derail_prompted_calls)
        self.hallucinate_after_silence_s = hallucinate_after_silence_s
        self.detected_language = detected_language
        self.calls: list[dict] = []
        self._prompted_calls = 0

    def transcribe(self, audio, **kwargs):
        audio = np.asarray(audio)
        total = int(audio.shape[0])
        self.calls.append({"audio": audio.copy(), "kwargs": dict(kwargs)})
        prompt = kwargs.get("initial_prompt")
        language = kwargs.get("language") or self.detected_language
        if prompt:
            prompted_call = self._prompted_calls
            self._prompted_calls += 1
            if prompted_call in self.derail_prompted_calls:
                return self._prompt_echo_loop(total, kwargs, language)

        runs = word_runs(audio)
        segments: list[dict] = []
        seek = 0
        window_index = 0
        while seek < total:
            window_end = min(seek + WINDOW, total)
            limit = window_end
            window_limit_s = self.first_window_limit_s if window_index == 0 else self.later_window_limit_s
            if window_limit_s is not None:
                limit = min(limit, seek + at(window_limit_s))
            full_window = window_end - seek == WINDOW

            def fits(run: tuple[int, int, int]) -> bool:
                # Sound reaching the edge of a full window is a word running past it.
                return run[2] < window_end or (run[2] == window_end and not full_window)

            starting = [run for run in runs if seek <= run[1] < window_end]
            heard = [run for run in starting if run[2] <= limit and fits(run)]
            if not heard and starting and fits(starting[0]):
                heard = starting[:1]  # a window always gets through at least one word
            for index, start, end in heard:
                form = "W" if (prompt and window_index == 0) else "w"
                split = "~" if (start == 0 or end == total) else ""
                segments.append(self._segment(seek, start, end, f" {form}{index}{split}"))
            if len(heard) == len(starting):
                seek = window_end
            elif heard:
                seek = heard[-1][2]  # stop at the last complete word; the next window starts there
            else:
                # One word fills the window past its edge: Whisper's hard cut.
                index, start, _ = starting[0]
                segments.append(self._segment(seek, start, window_end, f" w{index}~"))
                seek = window_end
            window_index += 1

        if self.hallucinate_after_silence_s is not None and runs:
            trailing_s = (total - runs[-1][2]) / SR
            if trailing_s >= self.hallucinate_after_silence_s:
                tail = self._segment(segments[-1]["seek"] * HOP, runs[-1][2], total + at(0.8), " Thank you.")
                tail["avg_logprob"] = -0.3
                segments.append(tail)
        for number, segment in enumerate(segments):
            segment["id"] = number
        return {"text": "".join(s["text"] for s in segments), "segments": segments, "language": language}

    @staticmethod
    def _segment(seek: int, start: int, end: int, text: str) -> dict:
        return {
            "seek": seek // HOP,
            "start": start / SR,
            "end": end / SR,
            "text": text,
            "temperature": 0.0,
            "avg_logprob": -0.2,
            "compression_ratio": 1.5,
            "no_speech_prob": 0.0,
        }

    @staticmethod
    def _prompt_echo_loop(total: int, kwargs: dict, language: str) -> dict:
        """What Whisper returns when the prompt derails it: prompt words on repeat."""
        text = " Zentrix," * 90
        temperature = kwargs.get("temperature", 1.0)
        segment = {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": min(total, WINDOW) / SR,
            "text": text,
            "temperature": temperature if isinstance(temperature, float) else 1.0,
            "avg_logprob": -0.18,
            "compression_ratio": 20.0,
            "no_speech_prob": 0.0,
        }
        return {"text": text, "segments": [segment], "language": language}


def backend_with(fake: FakeWhisper, **config) -> MlxWhisperFinalBackend:
    backend = MlxWhisperFinalBackend(
        FinalAsrConfig(model_path="unused", backend_name="mlx_whisper", hf_repo_id=MODEL_REF, **config)
    )
    # Already "warm" on this thread: decode() must not import the real module.
    backend._mlx_whisper = fake
    backend._model_ref = MODEL_REF
    backend._model_thread_id = threading.get_ident()
    return backend


def request(audio: np.ndarray, *, start_ms: float = 0.0, prompt: str | None = PROMPT, **overrides) -> FinalDecodeRequest:
    fields = dict(
        utterance_id="synthetic-utterance",
        audio=audio,
        sample_rate_hz=SR,
        start_ms=start_ms,
        end_ms=start_ms + audio.shape[0] / SR * 1000.0,
        language="en",
        language_policy="fixed",
        initial_prompt=prompt,
    )
    fields.update(overrides)
    return FinalDecodeRequest(**fields)


def dictionary_forms(audio: np.ndarray) -> str:
    return " ".join(f"W{index}" for index, _, _ in word_runs(audio))


LONG_PARTS = [("gap", 0.3), *sentences(21), ("gap", 0.5)]  # 74.3 s, a 0.6 s pause every 3.5 s


# ------------------------------------------------------------------ #
# The defect: the prompt must reach the whole utterance
# ------------------------------------------------------------------ #

def test_prompt_reaches_every_part_of_a_long_dictation() -> None:
    audio = recording(*LONG_PARTS)
    assert audio.shape[0] > 2 * WINDOW
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio))

    # Every word in its dictionary form, each exactly once and in order.
    assert result.text == dictionary_forms(audio)
    assert [segment.text for segment in result.segments] == dictionary_forms(audio).split()
    assert len(fake.calls) > 1
    assert all(call["kwargs"]["initial_prompt"] == PROMPT for call in fake.calls)


def test_prompt_survives_whisper_ending_a_window_early() -> None:
    """Whisper may stop short of the chunk it was given and decode the rest in
    a second window of its own, which gets no prompt. That remainder must be
    decoded again with the prompt, not kept."""
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper(first_window_limit_s=20.0)

    result = backend_with(fake).decode(request(audio))

    assert result.text == dictionary_forms(audio)


def test_hand_over_starts_where_whispers_first_window_stopped() -> None:
    """Whisper may need several further windows for the remainder; the next
    chunk starts at the end of the first one, not at a later one."""
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper(first_window_limit_s=20.0, later_window_limit_s=3.0)

    result = backend_with(fake).decode(request(audio))

    assert result.text == dictionary_forms(audio)
    # The last word the first window got through ends at 19.5 s; word levels
    # are unique, so the second call's opening second pins where it starts.
    np.testing.assert_array_equal(fake.calls[1]["audio"][: at(1.0)], audio[at(19.5) : at(20.5)])


def test_prompt_reaches_speech_that_has_no_pause_to_cut_at() -> None:
    """45 s of unbroken speech: no cut is chosen there; Whisper's own stop is."""
    unbroken = []
    for _ in range(75):
        unbroken += [("word", 0.5), ("gap", 0.1)]
    audio = recording(("gap", 0.5), *sentences(3), *unbroken, ("gap", 0.5), *sentences(4), ("gap", 0.5))
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio))

    assert result.text == dictionary_forms(audio)
    # One chunk was a full window with a word running over its edge; the
    # chunk after it starts at the last complete word, not at that edge.
    full_windows = [i for i, call in enumerate(fake.calls) if call["audio"].shape[0] == WINDOW]
    assert len(full_windows) == 1
    assert fake.calls[full_windows[0]]["audio"][-1] != 0.0
    assert not fake.calls[full_windows[0] + 1]["audio"][: at(0.1)].any()


# ------------------------------------------------------------------ #
# Single-window audio: the existing call, unchanged
# ------------------------------------------------------------------ #

EXPECTED_SINGLE_CALL_KWARGS = {
    "path_or_hf_repo": MODEL_REF,
    "language": "en",
    "initial_prompt": PROMPT,
    "fp16": True,
    "word_timestamps": False,
    "condition_on_previous_text": False,
}


@pytest.mark.parametrize("sentence_count", [1, 4, 8])
def test_audio_within_one_window_is_one_unchanged_transcribe_call(sentence_count: int) -> None:
    audio = recording(("gap", 0.3), *sentences(sentence_count), ("gap", 0.5))
    assert audio.shape[0] <= WINDOW
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio, start_ms=250.0))

    assert len(fake.calls) == 1
    assert fake.calls[0]["kwargs"] == EXPECTED_SINGLE_CALL_KWARGS
    np.testing.assert_array_equal(fake.calls[0]["audio"], audio)
    assert result.text == dictionary_forms(audio)
    runs = word_runs(audio)
    assert [(s.start_ms, s.end_ms) for s in result.segments] == [
        (250.0 + start / SR * 1000.0, 250.0 + end / SR * 1000.0) for _, start, end in runs
    ]


def test_exactly_one_window_of_audio_is_still_a_single_call() -> None:
    parts = [("gap", 0.3), *sentences(8)]
    head = recording(*parts)
    audio = np.concatenate([head, np.zeros(WINDOW - head.shape[0], dtype=np.float32)])
    assert audio.shape[0] == WINDOW
    fake = FakeWhisper()

    backend_with(fake).decode(request(audio))

    assert len(fake.calls) == 1
    assert fake.calls[0]["kwargs"] == EXPECTED_SINGLE_CALL_KWARGS
    assert fake.calls[0]["audio"].shape[0] == WINDOW


def test_condition_on_previous_text_keeps_the_single_call() -> None:
    """Opting into cross-window context means one call owns the whole audio."""
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper()

    backend_with(fake, condition_on_previous_text=True).decode(request(audio))

    assert len(fake.calls) == 1
    assert fake.calls[0]["audio"].shape[0] == audio.shape[0]
    assert fake.calls[0]["kwargs"] == {**EXPECTED_SINGLE_CALL_KWARGS, "condition_on_previous_text": True}


def test_recording_without_any_pause_keeps_the_single_call() -> None:
    """Nothing to cut at: the previous behaviour, exactly."""
    audio = recording(("word", 40.0))
    fake = FakeWhisper()

    backend_with(fake).decode(request(audio))

    assert len(fake.calls) == 1
    assert fake.calls[0]["kwargs"] == EXPECTED_SINGLE_CALL_KWARGS
    assert fake.calls[0]["audio"].shape[0] == audio.shape[0]


# ------------------------------------------------------------------ #
# Chunk boundaries
# ------------------------------------------------------------------ #

def test_long_audio_is_cut_only_inside_pauses() -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio))

    assert "~" not in result.text  # no word was split by a chunk boundary
    edge = at(0.1)
    lengths = [call["audio"].shape[0] for call in fake.calls]
    assert sum(lengths) == audio.shape[0]  # chunks cover the recording once, no overlap
    assert all(length <= WINDOW for length in lengths)
    for call in fake.calls[:-1]:
        assert not call["audio"][-edge:].any()  # at least 100 ms of pause before each cut
    for call in fake.calls[1:]:
        assert not call["audio"][:edge].any()  # and after it


def test_a_sentence_pause_is_preferred_to_later_short_pauses(chunking) -> None:
    # Pauses: 0.4 s at 16.0 s, 1.2 s at 22.0 s, 0.4 s at 27.0 s. Only the
    # 1.2 s one is long enough to be the end of a sentence.
    audio = recording(
        ("word", 16.0), ("gap", 0.4), ("word", 5.6), ("gap", 1.2), ("word", 3.8), ("gap", 0.4), ("word", 12.6),
        ("gap", 0.5),
    )
    pauses = chunking.find_pauses(audio)
    assert pauses == [(at(16.0), at(16.4)), (at(22.0), at(23.2)), (at(27.0), at(27.4))]

    end, planned = chunking.plan_chunk_end(pauses, 0, audio.shape[0])

    assert planned is True
    assert end == at(22.6)  # the middle of the 1.2 s pause


def test_without_a_sentence_pause_the_longest_short_pause_is_cut(chunking) -> None:
    # Pauses: 0.44 s at 17.0 s and 0.36 s at 24.0 s, plus silence up front
    # so the noise floor is well defined. Neither ends a sentence.
    audio = recording(
        ("gap", 2.0), ("word", 15.0), ("gap", 0.44), ("word", 6.56), ("gap", 0.36), ("word", 14.64), ("gap", 0.5),
    )
    pauses = chunking.find_pauses(audio)
    assert pauses == [(0, at(2.0)), (at(17.0), at(17.44)), (at(24.0), at(24.36))]

    assert chunking.plan_chunk_end(pauses, 0, audio.shape[0]) == (at(17.22), True)


def test_the_latest_sentence_length_pause_in_reach_is_the_one_cut(chunking) -> None:
    audio = recording(*LONG_PARTS)
    pauses = chunking.find_pauses(audio)

    end, planned = chunking.plan_chunk_end(pauses, 0, audio.shape[0])

    # 0.6 s pauses end every 3.5 s; the last one reaching into the first
    # 28 s spans 27.7-28.3 s and is cut at its middle.
    assert (end, planned) == (at(28.0), True)


def test_a_later_sentence_pause_wins_over_an_earlier_longer_one(chunking) -> None:
    # Pauses: 1.2 s at 18.0 s, 0.6 s at 25.0 s. Both end a sentence; the
    # later one makes the longer chunk.
    audio = recording(("word", 18.0), ("gap", 1.2), ("word", 5.8), ("gap", 0.6), ("word", 14.4), ("gap", 0.5))
    pauses = chunking.find_pauses(audio)
    assert pauses == [(at(18.0), at(19.2)), (at(25.0), at(25.6))]

    assert chunking.plan_chunk_end(pauses, 0, audio.shape[0]) == (at(25.3), True)


def test_a_late_cut_never_leaves_a_final_chunk_of_a_few_words(chunking) -> None:
    # 32 s with sentence pauses at 20.0 s and 27.4 s: cutting at the later
    # one would leave a 4.3 s final chunk, so the earlier one is cut.
    audio = recording(("word", 20.0), ("gap", 0.6), ("word", 6.8), ("gap", 0.6), ("word", 3.5), ("gap", 0.5))
    assert audio.shape[0] == at(32.0)
    pauses = chunking.find_pauses(audio)
    assert pauses == [(at(20.0), at(20.6)), (at(27.4), at(28.0))]

    assert chunking.plan_chunk_end(pauses, 0, audio.shape[0]) == (at(20.3), True)


def test_a_pause_straddling_the_search_edge_is_cut_inside_the_allowed_stretch(chunking) -> None:
    # One 3 s pause from 27.5 s to 30.5 s; cuts are allowed up to 28 s.
    audio = recording(("word", 27.5), ("gap", 3.0), ("word", 9.0), ("gap", 0.5))
    pauses = chunking.find_pauses(audio)

    end, planned = chunking.plan_chunk_end(pauses, 0, audio.shape[0])

    assert (end, planned) == (at(28.0), True)


def test_an_earlier_pause_is_cut_when_none_lies_later_in_the_window(chunking) -> None:
    # The only pause is at 5-8 s: too early to be the first choice, but a
    # short chunk ended at a real pause beats a window ended at its edge.
    audio = recording(("word", 5.0), ("gap", 3.0), ("word", 40.0), ("gap", 0.5))
    pauses = chunking.find_pauses(audio)
    assert pauses == [(at(5.0), at(8.0))]

    assert chunking.plan_chunk_end(pauses, 0, audio.shape[0]) == (at(6.5), True)


def test_no_pause_in_reach_plans_a_full_window_and_no_cut(chunking) -> None:
    # The only pause is over before the shortest chunk allowed (5 s) ends.
    audio = recording(("word", 1.0), ("gap", 3.0), ("word", 40.0), ("gap", 0.5))
    pauses = chunking.find_pauses(audio)
    assert pauses == [(at(1.0), at(4.0))]

    assert chunking.plan_chunk_end(pauses, 0, audio.shape[0]) == (WINDOW, False)


def test_the_remainder_that_fits_one_window_is_the_last_chunk(chunking) -> None:
    audio = recording(*LONG_PARTS)
    pauses = chunking.find_pauses(audio)
    start = audio.shape[0] - WINDOW

    assert chunking.plan_chunk_end(pauses, start, audio.shape[0]) == (audio.shape[0], True)


def _speech_with_room_tone(tone_rms: float) -> np.ndarray:
    """Speech at -14 dB with room tone of the given RMS in the gaps: a 0.2 s
    gap at 4 s, a 0.8 s gap at 8.2 s, and 3 s of tone at the end."""
    rng = np.random.default_rng(118)
    speech = lambda seconds: (0.2 * np.sign(rng.standard_normal(at(seconds)))).astype(np.float32)  # noqa: E731
    tone = lambda seconds: (tone_rms * rng.standard_normal(at(seconds))).astype(np.float32)  # noqa: E731
    return np.concatenate([speech(4.0), tone(0.2), speech(4.0), tone(0.8), speech(4.0), tone(3.0)])


@pytest.mark.parametrize("tone_rms", [0.002, 0.012])  # room tone 40 dB and 24 dB under the speech
def test_pauses_are_found_relative_to_the_recording_noise_floor(tone_rms: float, chunking) -> None:
    """Room tone in the gaps is still a pause; a gap shorter than 300 ms and
    the silence that runs to the end of the recording are not."""
    pauses = chunking.find_pauses(_speech_with_room_tone(tone_rms))

    assert len(pauses) == 1
    start, end = pauses[0]
    assert abs(start - at(8.2)) <= 320 and abs(end - at(9.0)) <= 320


def test_no_pause_is_reported_when_speech_is_too_close_to_the_noise_floor(chunking) -> None:
    """Room tone only 14 dB under the speech: level cannot tell a pause from a
    quiet word, so nothing is offered to cut at."""
    assert chunking.find_pauses(_speech_with_room_tone(0.04)) == []


def test_no_pause_is_reported_when_level_cannot_tell_speech_from_silence(chunking) -> None:
    rng = np.random.default_rng(118)
    flat = (0.05 * rng.standard_normal(40 * SR)).astype(np.float32)

    assert chunking.find_pauses(flat) == []


def test_noisy_recording_keeps_the_single_call() -> None:
    """Sentences with pauses, but room tone only ~12 dB under the speech."""
    rng = np.random.default_rng(118)
    audio = recording(*LONG_PARTS)
    audio = np.where(audio == 0.0, 0.05 * rng.standard_normal(audio.shape[0]), audio).astype(np.float32)
    fake = FakeWhisper()

    backend_with(fake).decode(request(audio))

    assert len(fake.calls) == 1
    assert fake.calls[0]["kwargs"] == EXPECTED_SINGLE_CALL_KWARGS
    assert fake.calls[0]["audio"].shape[0] == audio.shape[0]


# ------------------------------------------------------------------ #
# Merged result: timestamps, language, confidence, logging, guard
# ------------------------------------------------------------------ #

def test_segment_times_of_a_long_dictation_are_on_the_request_timebase() -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio, start_ms=5_000.0))

    runs = word_runs(audio)
    assert len(result.segments) == len(runs)
    for segment, (_, start, end) in zip(result.segments, runs):
        assert segment.start_ms == pytest.approx(5_000.0 + start / SR * 1000.0, abs=1e-6)
        assert segment.end_ms == pytest.approx(5_000.0 + end / SR * 1000.0, abs=1e-6)
    starts = [segment.start_ms for segment in result.segments]
    assert starts == sorted(starts)
    assert result.start_ms == 5_000.0
    assert result.audio_duration_ms == pytest.approx(audio.shape[0] / SR * 1000.0)


def test_merged_result_keeps_whisper_segment_fields_consistent(chunking) -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper()

    merged = chunking.transcribe_with_prompt_per_chunk(fake.transcribe, audio, initial_prompt=PROMPT, language="en")

    runs = word_runs(audio)
    assert merged["text"] == "".join(f" W{index}" for index, _, _ in runs)
    # Whisper's own invariant, which consumers rely on to re-assemble text.
    assert merged["text"] == "".join(segment["text"] for segment in merged["segments"])
    assert [segment["id"] for segment in merged["segments"]] == list(range(len(runs)))
    chunk_starts = np.cumsum([0] + [call["audio"].shape[0] for call in fake.calls[:-1]]).tolist()
    for segment, (_, start, _) in zip(merged["segments"], runs):
        own_chunk = max(offset for offset in chunk_starts if offset <= start)
        assert segment["seek"] == own_chunk // HOP  # frames from the start of the recording
        assert segment["compression_ratio"] == 1.5 and segment["avg_logprob"] == -0.2
    assert merged["language"] == "en"


def test_single_window_result_is_returned_untouched(chunking) -> None:
    audio = recording(("gap", 0.3), *sentences(4), ("gap", 0.5))
    sentinel = {"text": " as is", "segments": [], "language": "en"}
    seen: list[tuple[np.ndarray, dict]] = []

    def transcribe(samples, **kwargs):
        seen.append((samples, kwargs))
        return sentinel

    result = chunking.transcribe_with_prompt_per_chunk(transcribe, audio, initial_prompt=PROMPT, fp16=True)

    assert result is sentinel
    assert len(seen) == 1 and seen[0][0] is audio
    assert seen[0][1] == {"initial_prompt": PROMPT, "fp16": True}


def test_auto_detected_language_is_detected_once_and_pinned() -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper(detected_language="hi")

    result = backend_with(fake).decode(request(audio, language=None, language_policy="auto"))

    assert fake.calls[0]["kwargs"]["language"] is None  # detected on the first chunk
    assert len(fake.calls) > 1
    assert all(call["kwargs"]["language"] == "hi" for call in fake.calls[1:])
    assert result.language == "hi"


def test_confidence_signals_cover_all_chunks() -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio))

    assert all(segment.avg_logprob == -0.2 for segment in result.segments)
    assert all(segment.no_speech_prob == 0.0 for segment in result.segments)
    assert all(segment.compression_ratio == 1.5 for segment in result.segments)
    assert result.metadata["avg_logprob"] == pytest.approx(-0.2)
    assert result.metadata["no_speech_prob"] == pytest.approx(0.0)
    assert result.metadata["compression_ratio"] == pytest.approx(1.5)


def test_trailing_silence_guard_still_reads_merged_segment_timing() -> None:
    """A stock phrase hallucinated on the silent tail of the last chunk runs
    past the end of the recording; the guard recognises it by that timing,
    which only holds if merged segments are on the recording's timebase."""
    audio = recording(("gap", 0.3), *sentences(21), ("gap", 3.0))
    fake = FakeWhisper(hallucinate_after_silence_s=2.5)

    result = backend_with(fake).decode(request(audio))

    assert result.text == dictionary_forms(audio) + " Thank you."
    tail = result.segments[-1]
    assert tail.end_ms == pytest.approx(result.audio_duration_ms + 800.0)
    cleaned = strip_trailing_silence_hallucination(
        result.text, segments=result.segments, audio_duration_ms=result.audio_duration_ms
    )
    assert cleaned == dictionary_forms(audio)


def test_a_chunked_decode_logs_one_start_one_done_and_its_chunk_plan(capsys) -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio))

    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("[ASR_FINAL]")]
    assert [line.split()[1] for line in lines] == ["decode_start", "decode_chunks", "decode_done"]
    assert f"audio_ms={audio.shape[0] / SR * 1000.0:.0f}" in lines[0]
    assert "chunks=3 prompted=3 calls=3 ends_ms=28000,56000,74300" in lines[1]
    assert f"segs={len(result.segments)}" in lines[2] and f"chars={len(result.text)}" in lines[2]
    assert "W0" not in lines[1]  # the chunk line carries no transcript text


def test_a_single_window_decode_logs_exactly_what_it_logged_before(capsys) -> None:
    audio = recording(("gap", 0.3), *sentences(4), ("gap", 0.5))
    fake = FakeWhisper()

    backend_with(fake).decode(request(audio))

    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("[ASR_FINAL]")]
    assert [line.split()[1] for line in lines] == ["decode_start", "decode_done"]


def test_the_chunk_log_shows_a_withheld_prompt(capsys) -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper(derail_prompted_calls=(1,))

    backend_with(fake).decode(request(audio))

    chunk_lines = [line for line in capsys.readouterr().err.splitlines() if "decode_chunks" in line]
    assert len(chunk_lines) == 1
    assert "chunks=3 prompted=1 calls=4" in chunk_lines[0]


def test_the_mlx_guard_is_held_once_around_all_chunks(monkeypatch) -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper()
    events: list[str] = []

    @contextlib.contextmanager
    def recording_guard():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    original_transcribe = fake.transcribe

    def transcribe(samples, **kwargs):
        events.append("transcribe")
        return original_transcribe(samples, **kwargs)

    fake.transcribe = transcribe  # type: ignore[method-assign]
    monkeypatch.setattr(mlx_backend_module, "mlx_decode_guard", recording_guard)

    backend_with(fake).decode(request(audio))

    assert events[0] == "enter" and events[-1] == "exit"
    assert events.count("enter") == 1 and events.count("exit") == 1
    assert events.count("transcribe") == len(fake.calls) > 1


# ------------------------------------------------------------------ #
# When the prompt is withheld
# ------------------------------------------------------------------ #

def test_a_prompt_that_derails_a_chunk_is_dropped_for_the_rest_of_the_recording() -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper(derail_prompted_calls=(1,))  # the second chunk loops on prompt words

    result = backend_with(fake).decode(request(audio))

    assert "Zentrix" not in result.text  # the loop never reaches the transcript
    spoken = [index for index, _, _ in word_runs(audio)]
    assert [int(word[1:]) for word in result.text.split()] == spoken  # nothing lost, nothing repeated
    first_chunk_words = len(word_runs(fake.calls[0]["audio"]))
    assert result.text.split()[:first_chunk_words] == [f"W{i}" for i in spoken[:first_chunk_words]]
    assert result.text.split()[first_chunk_words:] == [f"w{i}" for i in spoken[first_chunk_words:]]
    prompts = [call["kwargs"]["initial_prompt"] for call in fake.calls]
    assert prompts[:2] == [PROMPT, PROMPT]
    assert all(prompt is None for prompt in prompts[2:])
    assert len(prompts) == 4  # three chunks plus the one wasted prompted attempt
    # The retry decodes the very same audio the derailed attempt was given.
    np.testing.assert_array_equal(fake.calls[1]["audio"], fake.calls[2]["audio"])


def test_prompted_chunks_are_decoded_greedily_and_retries_use_whisper_defaults() -> None:
    """A prompted attempt must not enter Whisper's temperature fallback (each
    re-sample of a derailed prompt costs seconds); the unprompted retry is a
    plain call, as every later window was before."""
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper(derail_prompted_calls=(1,))

    backend_with(fake).decode(request(audio))

    for call in fake.calls:
        if call["kwargs"]["initial_prompt"]:
            assert call["kwargs"]["temperature"] == 0.0
        else:
            assert "temperature" not in call["kwargs"]


def test_chunks_with_next_to_no_sound_are_decoded_without_the_prompt() -> None:
    """A minute of silence inside a dictation, broken by one half-second word.
    Whisper given the prompt and (nearly) nothing to transcribe repeats the
    prompt, so those stretches must be decoded without it."""
    audio = recording(
        ("gap", 0.3), *sentences(6), ("gap", 30.0), ("word", 0.5), ("gap", 30.0), *sentences(6), ("gap", 0.5)
    )
    lone_word = 30  # the word between the two silences
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio))

    spoken = [index for index, _, _ in word_runs(audio)]
    assert result.text.split() == [f"w{i}" if i == lone_word else f"W{i}" for i in spoken]
    sound_s = [np.count_nonzero(call["audio"]) / SR for call in fake.calls]
    assert 0.0 in sound_s and 0.5 in sound_s  # a silent chunk and the lone-word chunk
    for call, seconds in zip(fake.calls, sound_s):
        assert call["kwargs"]["initial_prompt"] == (PROMPT if seconds >= 2.0 else None)


def test_long_audio_without_a_prompt_is_chunked_with_the_callers_options() -> None:
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper()

    result = backend_with(fake).decode(request(audio, prompt=None))

    assert result.text == dictionary_forms(audio).lower()
    assert len(fake.calls) > 1
    assert all(call["kwargs"] == {**EXPECTED_SINGLE_CALL_KWARGS, "initial_prompt": None} for call in fake.calls)


def test_decode_count_is_bounded_when_whisper_keeps_stopping_early(chunking) -> None:
    """A model that only ever gets through half a second per window must not
    turn one utterance into hundreds of decodes."""
    audio = recording(*LONG_PARTS)
    fake = FakeWhisper(first_window_limit_s=0.6)

    merged = chunking.transcribe_with_prompt_per_chunk(fake.transcribe, audio, initial_prompt=PROMPT, language="en")

    spoken = [index for index, _, _ in word_runs(audio)]
    assert [int(word[1:]) for word in merged["text"].split()] == spoken
    assert merged["text"] == "".join(segment["text"] for segment in merged["segments"])
    assert len(fake.calls) <= 2 * 5 + 2 + 1  # twice the 15 s chunk count, plus slack, plus the closing call
    assert merged["calls"] == len(fake.calls)


def _random_parts(rng: np.random.Generator) -> list[tuple[str, float]]:
    """A recording of 35-150 s mixing sentences with pauses, unbroken speech
    with no pause, long silences and lone words (all on a 20 ms grid)."""

    def seconds(low: float, high: float) -> float:
        return float(rng.integers(int(low * 50), int(high * 50) + 1)) / 50.0

    parts: list[tuple[str, float]] = [("gap", seconds(0.1, 1.0))]
    target = seconds(35.0, 150.0)
    while sum(duration for _, duration in parts) < target:
        kind = rng.choice(["sentence", "sentence", "sentence", "unbroken", "silence", "lone word"])
        if kind == "sentence":
            for _ in range(int(rng.integers(2, 9))):
                parts += [("word", seconds(0.2, 0.8)), ("gap", seconds(0.04, 0.16))]
            parts.append(("gap", seconds(0.3, 2.0)))
        elif kind == "unbroken":
            for _ in range(int(rng.integers(25, 80))):
                parts += [("word", seconds(0.2, 0.8)), ("gap", seconds(0.04, 0.16))]
            parts.append(("gap", seconds(0.3, 1.0)))
        elif kind == "silence":
            parts.append(("gap", seconds(5.0, 50.0)))
        else:
            parts += [("word", seconds(0.2, 0.6)), ("gap", seconds(3.0, 20.0))]
    parts += [("word", 0.4), ("gap", seconds(0.3, 4.0))]
    return parts


@pytest.mark.parametrize("seed", range(60))
def test_randomized_recordings_are_transcribed_completely_and_in_place(seed: int, chunking) -> None:
    """Whatever the layout and however Whisper behaves, every word comes out
    exactly once, whole, in order, at its own position in the recording."""
    rng = np.random.default_rng(seed)
    audio = recording(*_random_parts(rng))
    fake = FakeWhisper(
        first_window_limit_s=[None, None, 12.0, 20.0, 25.0][int(rng.integers(5))],
        later_window_limit_s=[None, None, 4.0][int(rng.integers(3))],
        derail_prompted_calls=[(), (), (0,), (1,), (2,)][int(rng.integers(5))],
    )
    prompt = [PROMPT, PROMPT, None][int(rng.integers(3))]

    merged = chunking.transcribe_with_prompt_per_chunk(
        fake.transcribe, audio, initial_prompt=prompt, language="en", condition_on_previous_text=False
    )

    runs = word_runs(audio)
    assert "Zentrix" not in merged["text"]
    # Once each and in order: a word cut in two by a chunk boundary would be
    # transcribed twice, a skipped stretch would leave a gap. (The fake's "~"
    # only says a word touched the edge of the audio it was handed, which a
    # full window ending exactly between two words also produces.)
    assert [int(word[1:].rstrip("~")) for word in merged["text"].split()] == [index for index, _, _ in runs]
    for chunk in merged.get("chunks", [])[1:]:
        boundary = round(chunk["start"] * SR)
        assert not any(start < boundary < end for _, start, end in runs), "a chunk starts inside a word"
    assert merged["text"] == "".join(segment["text"] for segment in merged["segments"])
    for segment, (_, start, end) in zip(merged["segments"], runs):
        assert segment["start"] == pytest.approx(start / SR, abs=1e-9)
        assert segment["end"] == pytest.approx(end / SR, abs=1e-9)
    if prompt is None:
        assert merged["text"] == merged["text"].lower()
    assert len(fake.calls) <= 2 * -(-audio.shape[0] // (15 * SR)) + 3


@pytest.fixture()
def chunking():
    from juno_v2.final import pause_chunking

    return pause_chunking
