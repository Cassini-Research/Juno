"""Tail-repetition detector and collapser for ASR output.

Whisper-family models (including ``mlx_whisper-large-v3-turbo``) fail
in a very recognizable way on uncertain audio: the autoregressive
decoder locks into a phrase and emits it 2-5 times in a row, sometimes
truncated at the end because the decoding loop hit a max-tokens
budget. Examples from live preview/final ASR sessions:

    "Some can come to me. Some can come to me. Some"
    "Some can be saved. Some can be saved. Some can be saved. Som"
    "vids can be saved? Some vids can be saved? Some vids can be "
    "can you pick up? Some can you pick up? Some can you pick up?"

These are not caught by the confidence-gated hallucination guard in
``memory.store._looks_like_hallucination`` because mlx_whisper's
``avg_logprob`` stays around -0.6 to -0.8 on these outputs — above
the -1.0 skip threshold. And they are not caught by the word-ratio
check because the Phase 10 fix deliberately let legitimate repeated
speech ("hello hello hello hello hello") pass through.

The signal that DOES work: **the total word count is far higher
than the audio duration can physically support** under any realistic
speech rate. 150-200 words per minute is the high end for natural
speech (2.5-3.3 words per second). Hallucinated repetitions frequently
clock in at 6-10+ words per second of audio.

This module provides two things:

1. ``detect_tail_repetition`` — a pure algorithm that scans the word
   sequence and returns the (period, copies, removed) tuple if the
   tail is periodic. Does not look at audio duration.
2. ``collapse_tail_repetition`` — applies the detection plus an
   audio-duration-gated "is this suspicious?" predicate, and returns
   the cleaned text plus a diagnostic dict. Conservative by default:
   only collapses when BOTH the word sequence is periodic AND the
   words-per-second ratio suggests the model is running away.

Intentionally small, intentionally pure. No runtime deps beyond the
standard library. Lives in ``memory/`` alongside the hallucination
guard because both are "post-hoc quality filters on ASR text".
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

#: Maximum physically plausible words-per-second for natural English
#: speech. Above this, the text is likely a decoder runaway. 200 wpm
#: (fast speech) is ~3.3 wps; we gate at 5 wps to stay well above the
#: ceiling for legitimate dictation while still catching the 6-10+
#: wps hallucination cluster observed in the Phase 11 live session.
_MAX_NATURAL_WORDS_PER_SECOND = 5.0

#: Minimum number of consecutive copies of a phrase before the
#: detector treats it as a repetition. 2 is the absolute minimum;
#: below that we can't distinguish repetition from coincidence.
_MIN_REPETITION_COPIES = 2

#: Minimum period length (in words) to consider. A 1-word period
#: catches legitimate intentional repetition ("hello hello hello")
#: that the confidence gate is designed to let through. Starting at
#: 2 keeps us focused on phrase-level hallucinations.
_MIN_PERIOD_WORDS = 2

#: Single-word repetition is usually legitimate when the whole utterance is
#: just emphasis ("yes yes yes"). It becomes a decoder-runaway signal when a
#: normal sentence is followed by a long same-token suffix.
_MIN_SINGLE_TOKEN_SUFFIX_REPEATS = 8
_MIN_PREFIX_WORDS_BEFORE_SINGLE_TOKEN_SUFFIX = 5
_ABSURD_SINGLE_TOKEN_SUFFIX_REPEATS = 24

# Noisy tail loops do not always stay perfectly periodic. A real example:
# "... help me do it all this thing? of satisfaction because of satisfaction
# because ... satisfaction satisfaction". The suffix is too degraded for the
# strict phrase-period detector and not contiguous enough for the single-token
# suffix detector, but it is still a long low-entropy tail after a complete
# sentence.
_MIN_LOW_ENTROPY_SUFFIX_WORDS = 18
_MAX_LOW_ENTROPY_SUFFIX_DISTINCT_TOKENS = 4
_MIN_LOW_ENTROPY_DOMINANT_REPEATS = 6
_MIN_LOW_ENTROPY_REPEATED_TOKEN_COUNT = 2
_MIN_LOW_ENTROPY_TOKEN_REPEATS = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RepetitionCollapse:
    """Diagnostic record describing what the collapser did (or didn't).

    Always returned from ``collapse_tail_repetition`` so callers can
    log the decision trace-side without re-running the detector."""

    collapsed: bool
    reason: str
    period_words: int | None = None
    copies: int | None = None
    removed_words: int = 0
    words_per_second: float | None = None
    repeated_token: str | None = None

    def to_dict(self) -> dict:
        return {
            "collapsed": self.collapsed,
            "reason": self.reason,
            "period_words": self.period_words,
            "copies": self.copies,
            "removed_words": self.removed_words,
            "words_per_second": self.words_per_second,
            "repeated_token": self.repeated_token,
        }


def detect_tail_repetition(words: list[str]) -> tuple[int, int] | None:
    """Return ``(period, copies)`` if the tail of *words* is a
    periodic repetition of at least ``_MIN_REPETITION_COPIES`` copies,
    else ``None``.

    The detection works by treating the first ``period`` words as a
    candidate repetition unit and checking whether the rest of the
    sequence consists of complete copies of that unit plus (at most)
    a partial trailing copy. The partial trailing copy is allowed
    because whisper frequently truncates mid-phrase when it hits the
    max-tokens budget — "Some can be saved. Some can be saved. Som"
    is a legitimate match for a period of 4 words ("Some can be saved.")
    with 3 full copies plus "Som" as a prefix-truncated fourth.

    Prefers the SHORTEST period that explains the most copies, which
    catches the common "phrase X 3-5 times" pattern without also
    matching longer coincidental periods.
    """
    n = len(words)
    if n < _MIN_PERIOD_WORDS * _MIN_REPETITION_COPIES:
        return None

    best: tuple[int, int] | None = None
    max_period = n  # we allow period up to n because the full text may be a single unit repeated by a partial
    for period in range(_MIN_PERIOD_WORDS, max_period):
        unit = words[:period]
        copies = 1
        i = period
        matched_cleanly = True
        while i < n:
            # Try to match a full copy.
            if i + period <= n:
                if words[i : i + period] == unit:
                    copies += 1
                    i += period
                    continue
                # Not a full match — check if the next token differs.
                matched_cleanly = False
                break
            # Remaining tokens are shorter than a full period. Check
            # if they are a prefix of the unit, with the LAST word
            # allowed to be a character-level string prefix (handles
            # the "Som" → "Some" truncation whisper does on max-tokens).
            remaining = words[i:]
            if _is_prefix_of_unit(remaining, unit):
                break
            matched_cleanly = False
            break

        if not matched_cleanly:
            continue
        if copies < _MIN_REPETITION_COPIES:
            continue
        # Reject units that are themselves a single word repeated —
        # those are really "hello hello hello" cases that the Phase 10
        # confidence gate handles separately. This module focuses on
        # *phrase* repetition, so we require the unit to contain at
        # least 2 distinct tokens.
        if len(set(unit)) < 2:
            continue
        # Prefer shorter periods (more compression) when tied on copies.
        if best is None or copies > best[1] or (copies == best[1] and period < best[0]):
            best = (period, copies)
    return best


def _is_prefix_of_unit(remaining: list[str], unit: list[str]) -> bool:
    """Return True if *remaining* (a short tail) is a prefix of *unit*,
    allowing the last word of *remaining* to be a string prefix of the
    corresponding word in *unit*.

    Examples::

        _is_prefix_of_unit(["Some", "can", "be"],  ["Some", "can", "be", "saved."]) -> True
        _is_prefix_of_unit(["Som"],               ["Some", "can", "be", "saved."]) -> True  (string prefix)
        _is_prefix_of_unit(["cat"],               ["Some", "can", "be", "saved."]) -> False
        _is_prefix_of_unit([],                    ["anything"])                    -> True
    """
    if len(remaining) > len(unit):
        return False
    if not remaining:
        return True
    for i in range(len(remaining) - 1):
        if remaining[i] != unit[i]:
            return False
    last_idx = len(remaining) - 1
    return unit[last_idx].startswith(remaining[last_idx])


def collapse_tail_repetition(
    text: str,
    *,
    audio_duration_ms: float | None,
    max_words_per_second: float = _MAX_NATURAL_WORDS_PER_SECOND,
) -> tuple[str, RepetitionCollapse]:
    """Return ``(cleaned_text, diagnostic)``.

    The collapse fires only when BOTH:

    1. ``detect_tail_repetition`` finds a periodic repetition in the
       word sequence, AND
    2. The total word count per audio second is higher than
       ``max_words_per_second`` — the physically-implausible signal
       that disambiguates legitimate user-spoken repetition from
       decoder runaway.

    If either condition fails, the original text is returned unchanged
    with a diagnostic describing why. Callers should log the
    diagnostic regardless so operators can see near-misses in the
    trace JSONL.

    When ``audio_duration_ms`` is None or zero (unknown or malformed
    metadata), the words-per-second gate is skipped — in that case we
    collapse on the raw repetition pattern alone. That matches the
    behavior the caller would get from post-hoc trace analysis where
    timing is unavailable.
    """
    if not text or not text.strip():
        return text, RepetitionCollapse(collapsed=False, reason="empty_text")

    words = text.split()
    if len(words) < _MIN_PERIOD_WORDS * _MIN_REPETITION_COPIES:
        return text, RepetitionCollapse(collapsed=False, reason="too_short")

    detection = detect_tail_repetition(words)
    if detection is None:
        single_suffix = detect_single_token_tail_repetition(words)
        if single_suffix is None:
            low_entropy_suffix = _detect_low_entropy_repetition_tail(words)
            if low_entropy_suffix is None:
                return text, RepetitionCollapse(collapsed=False, reason="no_repetition_found")
            suffix_start, token, repeats = low_entropy_suffix
            wps = _words_per_second(words, audio_duration_ms)
            kept_words = words[:suffix_start]
            return _rejoin_words(text, kept_words), RepetitionCollapse(
                collapsed=True,
                reason="low_entropy_repetition_tail_collapsed",
                period_words=None,
                copies=repeats,
                removed_words=len(words) - suffix_start,
                words_per_second=wps,
                repeated_token=token,
            )
        token, repeats, suffix_start = single_suffix
        wps = _words_per_second(words, audio_duration_ms)
        if (
            wps is not None
            and wps < max_words_per_second
            and repeats < _ABSURD_SINGLE_TOKEN_SUFFIX_REPEATS
        ):
            return text, RepetitionCollapse(
                collapsed=False,
                reason="within_natural_speech_rate",
                period_words=1,
                copies=repeats,
                removed_words=0,
                words_per_second=wps,
                repeated_token=token,
            )
        kept_words = words[:suffix_start]
        if not kept_words:
            kept_words = words[:1]
            removed = max(0, len(words) - 1)
        else:
            removed = len(words) - len(kept_words)
        return _rejoin_words(text, kept_words), RepetitionCollapse(
            collapsed=True,
            reason="tail_single_token_repetition_collapsed",
            period_words=1,
            copies=repeats,
            removed_words=removed,
            words_per_second=wps,
            repeated_token=token,
        )

    period, copies = detection

    # Words-per-second gate. Real English dictation rarely exceeds
    # 3.5 wps; we gate at max_words_per_second (5.0 by default) to
    # give legitimate fast speech a safety margin.
    wps = _words_per_second(words, audio_duration_ms)
    if wps is not None and wps < max_words_per_second:
        return text, RepetitionCollapse(
            collapsed=False,
            reason="within_natural_speech_rate",
            period_words=period,
            copies=copies,
            removed_words=0,
            words_per_second=wps,
        )

    removed = len(words) - period
    unit = words[:period]
    collapsed_text = _rejoin_words(text, unit)
    return collapsed_text, RepetitionCollapse(
        collapsed=True,
        reason="tail_repetition_collapsed",
        period_words=period,
        copies=copies,
        removed_words=removed,
        words_per_second=wps,
    )


def _rejoin_words(original: str, unit: list[str]) -> str:
    """Rejoin the preserved period back into a single string.

    Uses a simple space-separator. Preserves the original's leading
    whitespace (if any) so the collapsed text still matches the
    surrounding buffer's indentation. If the unit does not end with
    punctuation, inherits the last-character punctuation from the
    original (which whisper usually emits on complete phrases)."""
    leading_ws_match = re.match(r"^\s*", original)
    prefix = leading_ws_match.group(0) if leading_ws_match else ""
    joined = " ".join(unit)
    return prefix + joined


def detect_single_token_tail_repetition(words: list[str]) -> tuple[str, int, int] | None:
    """Return ``(token, repeats, start_index)`` for a repeated-token suffix.

    This deliberately only catches a long suffix after substantive prior
    content. That preserves intentional short utterances like
    ``"yes yes yes yes yes yes yes"`` while stripping final-ASR failures such
    as a correct sentence followed by ``"Cent"`` hundreds of times.
    """
    if len(words) < (
        _MIN_PREFIX_WORDS_BEFORE_SINGLE_TOKEN_SUFFIX + _MIN_SINGLE_TOKEN_SUFFIX_REPEATS
    ):
        return None
    token = _repetition_token(words[-1])
    if not token:
        return None
    start = len(words) - 1
    while start > 0 and _repetition_token(words[start - 1]) == token:
        start -= 1
    repeats = len(words) - start
    if repeats < _MIN_SINGLE_TOKEN_SUFFIX_REPEATS:
        return None
    if start < _MIN_PREFIX_WORDS_BEFORE_SINGLE_TOKEN_SUFFIX:
        return None
    return token, repeats, start


def _detect_low_entropy_repetition_tail(words: list[str]) -> tuple[int, str, int] | None:
    """Return ``(start_index, dominant_token, repeats)`` for noisy tail loops.

    This intentionally requires a sentence boundary before the suffix. The
    strict detectors above handle whole-utterance loops and contiguous token
    suffixes; this fallback is for a complete sentence followed by a degraded
    ASR loop, so the boundary keeps it away from ordinary in-sentence wording.
    """
    n = len(words)
    if n < _MIN_PREFIX_WORDS_BEFORE_SINGLE_TOKEN_SUFFIX + _MIN_LOW_ENTROPY_SUFFIX_WORDS:
        return None

    normalized = [_repetition_token(word) for word in words]
    for start in range(_MIN_PREFIX_WORDS_BEFORE_SINGLE_TOKEN_SUFFIX, n - _MIN_LOW_ENTROPY_SUFFIX_WORDS + 1):
        if not _ends_sentence_boundary(words[start - 1]):
            continue
        suffix = [token for token in normalized[start:] if token]
        if len(suffix) < _MIN_LOW_ENTROPY_SUFFIX_WORDS:
            continue
        counts = Counter(suffix)
        if len(counts) > _MAX_LOW_ENTROPY_SUFFIX_DISTINCT_TOKENS:
            continue
        dominant_token, dominant_count = counts.most_common(1)[0]
        if dominant_count < _MIN_LOW_ENTROPY_DOMINANT_REPEATS:
            continue
        repeated_tokens = [
            token
            for token, count in counts.items()
            if count >= _MIN_LOW_ENTROPY_TOKEN_REPEATS and len(token) >= 3
        ]
        if len(repeated_tokens) < _MIN_LOW_ENTROPY_REPEATED_TOKEN_COUNT:
            continue
        if counts.get(suffix[0], 0) < _MIN_LOW_ENTROPY_TOKEN_REPEATS:
            continue
        return start, dominant_token, dominant_count
    return None


def _ends_sentence_boundary(value: str) -> bool:
    return bool(re.search(r"[.!?:;][\"')\]]*$", value or ""))


def _repetition_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


def _words_per_second(words: list[str], audio_duration_ms: float | None) -> float | None:
    if audio_duration_ms and audio_duration_ms > 0:
        return len(words) / (audio_duration_ms / 1000.0)
    return None
