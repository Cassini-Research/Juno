"""Decode final-lane audio longer than one Whisper window with the prompt on every window.

Why this exists
---------------
Whisper decodes 30 s of audio at a time. Given a longer recording, a
Whisper ``transcribe()`` walks it window by window, and with
``condition_on_previous_text=False`` (Juno's default, see
:class:`~juno_v2.final.config.FinalAsrConfig`) it resets the decoder prompt
after *every* window. The reset drops ``initial_prompt`` together with the
previous window's text, so the dictionary/bias prompt only steers the first
~30 s of a dictation; the rest is decoded as if the user had no dictionary.
The window seams are also wherever Whisper's loop happens to land — a hard
cut at exactly 30.0 s when a window decodes as a single segment — which
occasionally duplicates or drops a word.

``condition_on_previous_text=True`` is not a fix: the growing transcript
pushes the prompt out of the context anyway and it re-opens the repetition
loops the default exists to avoid. mlx-whisper 0.4.3 has no option to carry
the initial prompt across windows.

What it does
------------
:func:`transcribe_with_prompt_per_chunk` hands ``transcribe()`` one window's
worth of audio at a time, each call with the caller's prompt, and merges the
per-chunk results into one result of the usual shape with segment
timestamps on the original recording's timebase.

Audio that fits one window is passed straight through: one call, the
caller's exact arguments, the backend's own result object returned as is.

Where a chunk ends
------------------
Never in the middle of sound. A chunk ends at one of:

* the end of the recording;
* a *pause* — at least :data:`MIN_PAUSE_S` of audio at the recording's own
  noise floor (:func:`find_pauses`). The planner looks between
  :data:`MIN_CHUNK_S` and :data:`MAX_CHUNK_S` after the chunk start and
  takes the latest pause there that lasts :data:`SENTENCE_PAUSE_S` or more
  (a chunk as long as a window allows, ended where a sentence most likely
  ends); failing that, the longest shorter pause; failing that, a pause
  earlier in the chunk, by the same preference. It cuts in the middle of
  the pause, never closer than :data:`PAUSE_EDGE_GUARD_S` to the sound on
  either side, and never makes a chunk — the final one included — shorter
  than :data:`SHORTEST_CHUNK_S`. Silence running to the end of the
  recording is not a candidate, so trailing silence never becomes a chunk
  of its own;
* where Whisper itself stops. When the whole stretch up to
  :data:`MAX_CHUNK_S` holds no pause (unbroken speech, or noise hiding the
  pauses) the chunk is one full window and no cut is chosen here at all:
  Whisper closes its last complete segment inside the window and the next
  chunk starts at that timestamp — the rule Whisper's own loop has always
  applied, so that stretch is windowed as it was before, only now with the
  prompt. (A full window that Whisper decodes as one single segment still
  ends at its 30 s edge, as it always did.) The same hand-over is used when
  Whisper stops short of a pause-aligned end, because otherwise it would
  decode the remainder itself without the prompt.

A recording in which no pause can be found at all — including any whose
speech is less than 20 dB above its noise floor, where level alone cannot
tell a pause from quiet speech — gives the planner nothing to hold on to; it
is decoded by the previous single call, unchanged.

When the prompt is withheld
---------------------------
A long list-shaped prompt can tip Whisper into repeating prompt words
instead of transcribing — on a minority of mid-recording chunks, and on
most chunks that hold little or no speech. Whisper's remedy, re-sampling at
rising temperature with the prompt still attached, rarely recovers and
costs seconds per attempt. So:

* a chunk with less than :data:`MIN_PROMPTED_SOUND_S` of audio above the
  pause level is decoded without the prompt — there is next to nothing in
  it to bias, and it is the audio the prompt derails most often;
* a prompted chunk is decoded greedily (Whisper's first attempt); if that
  attempt fails Whisper's own acceptance test (:func:`_decode_failed`), the
  chunk and the rest of the recording are decoded without the prompt, with
  Whisper's normal temperature fallback.

Either way the audio is then decoded exactly as every window after the
first was decoded before this module existed, and at most one prompted
attempt per recording is wasted.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

#: Whisper's fixed input rate. The backends hand ``transcribe()`` a bare
#: sample array, which Whisper always reads at this rate, so every duration
#: in this module is counted in these samples and stays consistent with the
#: segment timestamps Whisper reports.
WHISPER_SAMPLE_RATE_HZ = 16_000

#: One Whisper decode window (30 s). Audio up to this length is decoded by
#: a single ``transcribe()`` call and is left alone.
WHISPER_WINDOW_SAMPLES = 30 * WHISPER_SAMPLE_RATE_HZ

#: Whisper's mel hop (10 ms): the unit of a segment's ``seek``. Cuts land on
#: this grid, so a chunk's offset is a whole number of Whisper frames.
WHISPER_HOP_SAMPLES = 160

#: Earliest cut after a chunk start that the planner looks for first: with
#: a pause between here and :data:`MAX_CHUNK_S` a recording needs about as
#: many chunks as Whisper's own loop needed windows.
MIN_CHUNK_S = 15.0

#: Latest pause cut after a chunk start. Kept two seconds short of the
#: window so Whisper sees where the chunk ends and closes its last segment
#: instead of treating the speech as running off the window edge.
MAX_CHUNK_S = 28.0

#: A pause this long is taken for the end of a sentence or a thought, and
#: the latest such pause in reach ends the chunk. Cutting as late as
#: possible keeps the number of chunks — and the decode time — at what
#: Whisper's own windowing needed.
SENTENCE_PAUSE_S = 0.50

#: No cut makes a chunk shorter than this. It keeps a late cut from leaving
#: a few trailing words to be decoded on their own, and it is how early the
#: planner will cut when the only pauses come before :data:`MIN_CHUNK_S` —
#: a short chunk ended at a real pause is better than a full window ended
#: wherever its 30 s run out.
SHORTEST_CHUNK_S = 5.0

#: Shortest stretch of noise-floor-level audio treated as a pause. Gaps
#: between words in fluent speech and stop-consonant closures inside a word
#: stay well under this; clause and sentence breaks are longer.
MIN_PAUSE_S = 0.30

#: Quiet audio kept on each side of a cut, so a soft word onset or a
#: trailing consonant that dips under the pause threshold is never split.
PAUSE_EDGE_GUARD_S = 0.10

#: Least audio above the pause level a chunk must hold to be decoded with
#: the prompt. Measured on real recordings: 20 s of room tone holding 0.5 s
#: of speech fails a greedy prompted decode more often than not, 2 s rarely.
MIN_PROMPTED_SOUND_S = 2.0

# Level analysis: mean power of 20 ms frames, in dB.
_LEVEL_FRAME_SAMPLES = 320
_LEVEL_FLOOR_POWER = 1e-12

# The recording's own noise floor and speech level are read off its level
# distribution, so the pause threshold follows the microphone and the room
# instead of assuming an absolute level.
_NOISE_FLOOR_PERCENTILE = 5.0
_SPEECH_LEVEL_PERCENTILE = 95.0

#: A frame is pause-level when it sits within this margin of the noise
#: floor. The margin shrinks to a quarter of the recording's dynamic range
#: when that range is small, so quiet speech over a high noise floor is not
#: mistaken for a pause.
_MAX_PAUSE_MARGIN_DB = 6.0
_PAUSE_MARGIN_RANGE_FRACTION = 0.25

#: Below this spread between speech level and noise floor, frame energy
#: cannot tell a pause from quiet speech reliably, and no frame is called
#: pause-level (so the recording keeps the previous single call). On a real
#: dictation with 16 dB of spread, cuts landed beside quiet words that
#: Whisper then dropped for want of the speech before them; the recordings
#: the cuts were validated on all had 22 dB or more.
_MIN_DYNAMIC_RANGE_DB = 20.0

# Whisper's defaults for accepting a decode (``transcribe()`` keyword
# arguments of the same names override them).
_COMPRESSION_RATIO_THRESHOLD = 2.4
_LOGPROB_THRESHOLD = -1.0
_NO_SPEECH_THRESHOLD = 0.6


def _pause_level_frames(audio: np.ndarray) -> np.ndarray:
    """Boolean mask over 20 ms frames: True where the level is at the noise floor."""
    frame_count = audio.shape[0] // _LEVEL_FRAME_SAMPLES
    if frame_count == 0:
        return np.zeros(0, dtype=bool)
    frames = audio[: frame_count * _LEVEL_FRAME_SAMPLES].reshape(frame_count, _LEVEL_FRAME_SAMPLES)
    power = np.mean(np.square(frames, dtype=np.float64), axis=1)
    levels_db = 10.0 * np.log10(np.maximum(power, _LEVEL_FLOOR_POWER))
    noise_floor_db, speech_level_db = np.percentile(
        levels_db, [_NOISE_FLOOR_PERCENTILE, _SPEECH_LEVEL_PERCENTILE]
    )
    dynamic_range_db = float(speech_level_db - noise_floor_db)
    if dynamic_range_db < _MIN_DYNAMIC_RANGE_DB:
        return np.zeros(frame_count, dtype=bool)
    margin_db = min(_MAX_PAUSE_MARGIN_DB, dynamic_range_db * _PAUSE_MARGIN_RANGE_FRACTION)
    return levels_db <= (noise_floor_db + margin_db)


def _pauses_from(pause_level: np.ndarray) -> list[tuple[int, int]]:
    frame_count = int(pause_level.shape[0])
    # +1 where a pause-level run starts, -1 one past where it ends.
    edges = np.diff(np.concatenate(([0], pause_level.astype(np.int8), [0])))
    run_starts = np.flatnonzero(edges == 1).tolist()
    run_ends = np.flatnonzero(edges == -1).tolist()
    min_pause_frames = int(round(MIN_PAUSE_S * WHISPER_SAMPLE_RATE_HZ / _LEVEL_FRAME_SAMPLES))
    return [
        (start_frame * _LEVEL_FRAME_SAMPLES, end_frame * _LEVEL_FRAME_SAMPLES)
        for start_frame, end_frame in zip(run_starts, run_ends)
        # A run reaching the last frame is trailing silence, not a pause.
        if end_frame < frame_count and end_frame - start_frame >= min_pause_frames
    ]


def find_pauses(audio: np.ndarray) -> list[tuple[int, int]]:
    """Return the pauses in *audio* as ``(start_sample, end_sample)`` spans.

    A pause is a run of at least :data:`MIN_PAUSE_S` of 20 ms frames whose
    level is at the recording's noise floor: no louder than the 5th
    percentile of its frame levels plus a small margin. Spans are in order
    and do not overlap. Quiet that reaches the end of the audio is trailing
    silence, not a pause between two stretches of sound, and is not
    returned. Nothing is returned when the recording's speech level and
    noise floor are less than 20 dB apart, too close to tell a pause from
    quiet speech.
    """
    return _pauses_from(_pause_level_frames(audio))


def _cut_inside(pause: tuple[int, int], earliest: int, latest: int) -> int | None:
    """Sample to cut at inside *pause* within ``[earliest, latest]``, or None.

    The cut is the middle of the pause, moved into the allowed stretch when
    the pause straddles its edge, and always :data:`PAUSE_EDGE_GUARD_S`
    clear of both ends of the pause.
    """
    guard = int(PAUSE_EDGE_GUARD_S * WHISPER_SAMPLE_RATE_HZ)
    lowest = max(pause[0] + guard, earliest)
    highest = min(pause[1] - guard, latest)
    if lowest > highest:
        return None
    return min(max((pause[0] + pause[1]) // 2, lowest), highest)


def _best_cut(pauses: list[tuple[int, int]], earliest: int, latest: int) -> int | None:
    """The cut to make within ``[earliest, latest]``: inside the latest pause of
    sentence length, else inside the longest shorter pause (the latest on a tie)."""
    sentence_pause = int(SENTENCE_PAUSE_S * WHISPER_SAMPLE_RATE_HZ)
    at_sentence_pause: int | None = None
    at_longest_pause: int | None = None
    longest = -1
    for pause in pauses:  # in order, so a later pause replaces an earlier one
        cut = _cut_inside(pause, earliest, latest)
        if cut is None:
            continue
        length = pause[1] - pause[0]
        if length >= sentence_pause:
            at_sentence_pause = cut
        if length >= longest:
            longest = length
            at_longest_pause = cut
    return at_sentence_pause if at_sentence_pause is not None else at_longest_pause


def plan_chunk_end(pauses: list[tuple[int, int]], start: int, total: int) -> tuple[int, bool]:
    """Where the chunk starting at sample *start* ends: ``(end_sample, planned)``.

    ``planned`` is True when the end is the end of the recording or the
    middle of a pause. It is False when no pause was in reach: the chunk is
    then one full Whisper window, and where it really ends is left to
    Whisper (module docstring, "Where a chunk ends").
    """
    if total - start <= WHISPER_WINDOW_SAMPLES:
        return total, True
    shortest = int(SHORTEST_CHUNK_S * WHISPER_SAMPLE_RATE_HZ)
    preferred_from = start + int(MIN_CHUNK_S * WHISPER_SAMPLE_RATE_HZ)
    latest = min(start + int(MAX_CHUNK_S * WHISPER_SAMPLE_RATE_HZ), total - shortest)
    for earliest, until in ((preferred_from, latest), (start + shortest, preferred_from - 1)):
        cut = _best_cut(pauses, earliest, until)
        if cut is not None:
            return cut, True
    return start + WHISPER_WINDOW_SAMPLES, False


def _decode_failed(segments: list[Mapping[str, Any]], options: Mapping[str, Any]) -> bool:
    """Whisper's own test for a decode it would not accept.

    Mirrors the fallback condition in Whisper's ``transcribe()``: the text
    is too repetitive (compression ratio) or too improbable (average log
    probability), unless Whisper judged the window to be silence.
    """
    compression_limit = options.get("compression_ratio_threshold", _COMPRESSION_RATIO_THRESHOLD)
    logprob_limit = options.get("logprob_threshold", _LOGPROB_THRESHOLD)
    no_speech_limit = options.get("no_speech_threshold", _NO_SPEECH_THRESHOLD)
    for segment in segments:
        compression_ratio = segment.get("compression_ratio")
        avg_logprob = segment.get("avg_logprob")
        no_speech_prob = segment.get("no_speech_prob")
        failed = False
        if compression_limit is not None and compression_ratio is not None:
            failed = failed or compression_ratio > compression_limit
        if logprob_limit is not None and avg_logprob is not None:
            failed = failed or avg_logprob < logprob_limit
        if no_speech_limit is not None and no_speech_prob is not None and no_speech_prob > no_speech_limit:
            failed = False
        if failed:
            return True
    return False


def _seek(segment: Mapping[str, Any]) -> int:
    return int(segment.get("seek", 0) or 0)


def transcribe_with_prompt_per_chunk(
    transcribe: Callable[..., Mapping[str, Any]],
    audio: np.ndarray,
    **options: Any,
) -> Mapping[str, Any]:
    """Run a Whisper ``transcribe(audio, **options)`` so the prompt reaches all of *audio*.

    *transcribe* is a Whisper-style callable returning a mapping with
    ``text``, ``segments`` (dicts with ``start``/``end`` in seconds, ``seek``
    in mel frames, and the decode statistics of their window) and
    ``language``. *audio* is mono float32 at :data:`WHISPER_SAMPLE_RATE_HZ`.

    One ``transcribe(audio, **options)`` call, result returned untouched,
    when *audio* fits one Whisper window, when the caller opted into
    ``condition_on_previous_text`` (separate calls cannot provide its
    cross-window context), or when *audio* has no pause to cut at.

    Otherwise *audio* is decoded one window-sized chunk at a time as the
    module docstring describes, and the merged result carries:

    * ``text`` — the chunks' text concatenated exactly as Whisper
      concatenates its own windows (no separator added or removed);
    * ``segments`` — every kept segment, ``start``/``end``/``seek`` shifted
      by its chunk's position in the recording, ``id`` renumbered;
    * ``language`` — the language of the first chunk. When the language is
      auto-detected it is detected on that chunk and pinned for the rest,
      so the utterance is decoded in one language as a single call would;
    * ``chunks`` and ``calls`` — not Whisper fields: one
      ``{"start", "end", "prompted"}`` entry (seconds) per chunk kept, and
      the number of ``transcribe()`` calls made, for the caller's logging.
    """
    total = int(audio.shape[0])
    if total <= WHISPER_WINDOW_SAMPLES or options.get("condition_on_previous_text"):
        return transcribe(audio, **options)
    pause_level = _pause_level_frames(audio)
    pauses = _pauses_from(pause_level)
    if not pauses:
        return transcribe(audio, **options)

    carry_prompt = bool(options.get("initial_prompt"))
    language = options.get("language")
    detected_language: Any = None
    language_known = False
    texts: list[str] = []
    segments: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    calls = 0
    # Every chunk normally costs one call; a withheld prompt or a hand-over
    # at Whisper's own stop costs one more. Past twice the usual chunk count
    # the recording is not behaving like speech with pauses, and the
    # remainder is left to a single plain call rather than an open-ended loop.
    calls_left = 2 * math.ceil(total / (MIN_CHUNK_S * WHISPER_SAMPLE_RATE_HZ)) + 2

    def run(samples: np.ndarray, **overrides: Any) -> Mapping[str, Any]:
        nonlocal calls, calls_left, language, detected_language, language_known
        calls += 1
        calls_left -= 1
        result = transcribe(samples, **{**options, "language": language, **overrides})
        if not language_known:
            language_known = True
            detected_language = result.get("language")
            if language is None:
                language = detected_language
        return result

    def run_without_prompt(samples: np.ndarray) -> Mapping[str, Any]:
        return run(samples, initial_prompt=None) if options.get("initial_prompt") else run(samples)

    def keep(result_segments: list[Mapping[str, Any]], text: str, start: int, end: int, prompted: bool) -> None:
        texts.append(text)
        offset_s = start / WHISPER_SAMPLE_RATE_HZ
        chunks.append({"start": offset_s, "end": end / WHISPER_SAMPLE_RATE_HZ, "prompted": prompted})
        offset_frames = start // WHISPER_HOP_SAMPLES
        for segment in result_segments:
            merged = dict(segment)
            merged["id"] = len(segments)
            merged["start"] = offset_s + float(segment.get("start", 0.0) or 0.0)
            merged["end"] = offset_s + float(segment.get("end", 0.0) or 0.0)
            if "seek" in segment:
                merged["seek"] = offset_frames + _seek(segment)
            segments.append(merged)

    start = 0
    while start < total:
        if calls_left <= 1:
            rest = run(audio[start:]) if carry_prompt else run_without_prompt(audio[start:])
            keep(list(rest.get("segments") or []), str(rest.get("text") or ""), start, total, carry_prompt)
            break
        end, planned = plan_chunk_end(pauses, start, total)
        chunk = audio[start:end]
        chunk_levels = pause_level[start // _LEVEL_FRAME_SAMPLES : -(-end // _LEVEL_FRAME_SAMPLES)]
        sound_samples = int(np.count_nonzero(~chunk_levels)) * _LEVEL_FRAME_SAMPLES

        result: Mapping[str, Any] | None = None
        prompted = carry_prompt and sound_samples >= MIN_PROMPTED_SOUND_S * WHISPER_SAMPLE_RATE_HZ
        if prompted:
            result = run(chunk, temperature=0.0)
            first_window = [s for s in (result.get("segments") or []) if _seek(s) == 0]
            if _decode_failed(first_window, options):
                carry_prompt = False
                prompted = False
                result = None
        if result is None:
            result = run_without_prompt(chunk)

        chunk_segments = list(result.get("segments") or [])
        later_seeks = [_seek(s) for s in chunk_segments if _seek(s) > 0]
        if later_seeks and (prompted or not planned):
            # Whisper stopped before the end of this chunk and decoded the
            # rest in further windows of its own (without the prompt, and up
            # to an end that is not a pause when ``planned`` is False). Keep
            # its first window; the next chunk starts where that one stopped.
            first_window = [s for s in chunk_segments if _seek(s) == 0]
            stop = start + min(later_seeks) * WHISPER_HOP_SAMPLES
            keep(first_window, "".join(str(s.get("text") or "") for s in first_window), start, stop, prompted)
            start = stop
        else:
            keep(chunk_segments, str(result.get("text") or ""), start, end, prompted)
            start = end
    return {
        "text": "".join(texts),
        "segments": segments,
        "language": detected_language,
        "chunks": chunks,
        "calls": calls,
    }


__all__ = [
    "MAX_CHUNK_S",
    "MIN_CHUNK_S",
    "MIN_PAUSE_S",
    "MIN_PROMPTED_SOUND_S",
    "PAUSE_EDGE_GUARD_S",
    "SENTENCE_PAUSE_S",
    "SHORTEST_CHUNK_S",
    "WHISPER_HOP_SAMPLES",
    "WHISPER_SAMPLE_RATE_HZ",
    "WHISPER_WINDOW_SAMPLES",
    "find_pauses",
    "plan_chunk_end",
    "transcribe_with_prompt_per_chunk",
]
