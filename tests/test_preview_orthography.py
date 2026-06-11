from __future__ import annotations

import pytest

from juno_v2.preview.hallucination_blocklist import (
    is_whisper_silence_hallucination,
    phrase_count,
    strip_trailing_boh,
    trail_strip_phrase_count,
)
from juno_v2.preview.orthography import normalize_preview_orthography


# ---------------------------------------------------------------------- #
# normalize_preview_orthography
# ---------------------------------------------------------------------- #


def test_committed_gets_full_sentence_orthography() -> None:
    committed, tail, meta = normalize_preview_orthography("hello world.", "this is the tail")
    assert committed == "Hello world."
    # Committed ends a sentence, so the tail starts one and is capitalized.
    assert tail == "This is the tail"
    assert meta["preview_orthography_applied"] == 2
    assert meta["preview_orthography_committed_changed"] is True
    assert meta["preview_orthography_tail_changed"] is True


def test_tail_not_capitalized_mid_sentence() -> None:
    committed, tail, _ = normalize_preview_orthography("hello world", "and the tail continues")
    assert committed == "Hello world"
    # No sentence boundary in committed → tail keeps a lowercase start.
    assert tail == "and the tail continues"


def test_tail_capitalized_when_committed_empty() -> None:
    committed, tail, meta = normalize_preview_orthography("", "i went home in january")
    assert committed == ""
    # Empty committed = beginning of utterance → capitalize tail start,
    # standalone "i" → "I", month casing applied.
    assert tail == "I went home in January"
    assert meta["preview_orthography_committed_changed"] is False
    assert meta["preview_orthography_tail_changed"] is True


def test_unexpected_internal_capitals_lowered_in_tail() -> None:
    _, tail, _ = normalize_preview_orthography("hello world", "and The tail Continues")
    # "The" is a common word wrongly capitalized mid-sentence → lowered;
    # "Continues" is not in the common-word list → kept (could be a name).
    assert tail == "and the tail Continues"


def test_acronyms_preserved() -> None:
    committed, _, _ = normalize_preview_orthography("send it to NASA today", "")
    assert committed == "Send it to NASA today"


def test_tail_internal_sentence_boundary_capitalized() -> None:
    _, tail, _ = normalize_preview_orthography("first part", "more. then more")
    # Tail start stays lowercase (mid-sentence) but a hard boundary inside
    # the tail still starts a new sentence.
    assert tail == "more. Then more"


def test_empty_inputs() -> None:
    committed, tail, meta = normalize_preview_orthography("", "")
    assert committed == ""
    assert tail == ""
    assert meta == {
        "preview_orthography_applied": 0,
        "preview_orthography_committed_changed": False,
        "preview_orthography_tail_changed": False,
    }


def test_none_like_inputs_are_safe() -> None:
    committed, tail, meta = normalize_preview_orthography(None, None)
    assert committed == ""
    assert tail == ""
    assert meta["preview_orthography_applied"] == 0


def test_idempotent() -> None:
    committed, tail, _ = normalize_preview_orthography("hello there.", "more text here")
    committed2, tail2, meta2 = normalize_preview_orthography(committed, tail)
    assert committed2 == committed
    assert tail2 == tail
    assert meta2["preview_orthography_applied"] == 0


# ---------------------------------------------------------------------- #
# is_whisper_silence_hallucination
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Thank you.",
        "thank you",
        "Thanks for watching!",
        "I'm sorry",  # apostrophe collapses to "im sorry"
        "Subtitles by the Amara.org community",
        "so",
        "MEOW",
        "♪",
        "...",
        "",
        "   ",
    ],
)
def test_is_hallucination_true(text: str) -> None:
    assert is_whisper_silence_hallucination(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Strict full-segment match: phrases inside real speech survive.
        "Thank you for joining the meeting",
        "I said thank you to her",
        "send the report",
        "meow is what the cat said",
    ],
)
def test_is_hallucination_false_for_real_speech(text: str) -> None:
    assert is_whisper_silence_hallucination(text) is False


def test_is_hallucination_none_is_false() -> None:
    assert is_whisper_silence_hallucination(None) is False


def test_phrase_counts_are_positive() -> None:
    assert phrase_count() > 30
    assert trail_strip_phrase_count() > 10


# ---------------------------------------------------------------------- #
# strip_trailing_boh
# ---------------------------------------------------------------------- #


def test_strip_trailing_boh_basic() -> None:
    cleaned, removed = strip_trailing_boh("Here is the plan. Thanks for watching")
    assert cleaned == "Here is the plan."
    assert removed == "Thanks for watching"


def test_strip_trailing_boh_strips_with_punctuation() -> None:
    cleaned, removed = strip_trailing_boh("Send the report. Thank you.")
    assert cleaned == "Send the report."
    assert removed == "Thank you."


def test_strip_trailing_boh_cascading() -> None:
    cleaned, removed = strip_trailing_boh("Send the report thanks for watching thank you")
    assert cleaned == "Send the report"
    # Both phrases removed, innermost last.
    assert removed == "thank you | thanks for watching"


def test_strip_trailing_boh_no_match() -> None:
    cleaned, removed = strip_trailing_boh("All good here")
    assert cleaned == "All good here"
    assert removed is None


def test_strip_trailing_boh_single_word_fillers_kept() -> None:
    # "so"/"you"/"the" are intentionally NOT Tier-A — they legitimately
    # end real sentences.
    cleaned, removed = strip_trailing_boh("It ended so")
    assert cleaned == "It ended so"
    assert removed is None


def test_strip_trailing_boh_empty() -> None:
    assert strip_trailing_boh("") == ("", None)


def test_strip_trailing_boh_mid_text_phrase_untouched() -> None:
    text = "I wanted to say thank you before the demo started"
    cleaned, removed = strip_trailing_boh(text)
    assert cleaned == text
    assert removed is None


def test_strip_trailing_boh_idempotent() -> None:
    # Documented contract: strip(strip(x)) == strip(x).
    once, _ = strip_trailing_boh("Wrap up the deck. See you in the next video")
    twice, removed = strip_trailing_boh(once)
    assert twice == once
    assert removed is None
