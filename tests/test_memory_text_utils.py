"""Unit tests for the pure text utilities in juno_v2/memory.

Covers term_policy, fold, hallucination, repetition, correction_diff,
and ai_dictionary.
"""

from __future__ import annotations

import re

import pytest

from juno_v2.memory.ai_dictionary import AI_GLOSSARY, is_ai_glossary_term
from juno_v2.memory.correction_diff import diff_pasted_segment
from juno_v2.memory.fold import fold_key, fold_match_pattern
from juno_v2.memory.hallucination import looks_like_hallucination
from juno_v2.memory.repetition import (
    collapse_tail_repetition,
    detect_tail_repetition,
)
from juno_v2.memory.term_policy import (
    MIN_LEARNED_TERM_CHARS,
    learned_term_allowed,
    meaningful_char_count,
)
from juno_v2.memory.stores.corrections import MAX_CORRECTION_TEXT_CHARS


# --------------------------------------------------------------------- #
# term_policy
# --------------------------------------------------------------------- #


def test_meaningful_char_count_empty_and_none() -> None:
    assert meaningful_char_count(None) == 0
    assert meaningful_char_count("") == 0
    assert meaningful_char_count("   ") == 0
    assert meaningful_char_count("!!!---") == 0


def test_meaningful_char_count_counts_only_alnum() -> None:
    assert meaningful_char_count("abc") == 3
    assert meaningful_char_count("a-b c.d") == 4
    assert meaningful_char_count("a1b2") == 4


def test_meaningful_char_count_unicode_letters() -> None:
    # Policy intentionally counts Unicode letters, not just latin.
    assert meaningful_char_count("café") == 4
    assert meaningful_char_count("日本語") == 3
    assert meaningful_char_count("naïve") == 5


def test_meaningful_char_count_nfkc_normalization() -> None:
    # Fullwidth latin compatibility forms normalize to plain letters.
    assert meaningful_char_count("ＡＢＣ") == 3


def test_learned_term_allowed_default_floor() -> None:
    assert MIN_LEARNED_TERM_CHARS == 3
    assert not learned_term_allowed("ab")
    assert learned_term_allowed("abc")
    assert not learned_term_allowed("a-b")  # 2 meaningful chars
    assert not learned_term_allowed("")
    assert not learned_term_allowed(None)


def test_learned_term_allowed_custom_min_chars() -> None:
    assert learned_term_allowed("ab", min_chars=2)
    # Floor never drops below 1 even when asked for 0 / negative.
    assert not learned_term_allowed("", min_chars=0)
    assert learned_term_allowed("a", min_chars=0)
    assert learned_term_allowed("a", min_chars=-5)


def test_learned_term_allowed_unicode_names() -> None:
    # Non-English names must not be rejected.
    assert learned_term_allowed("José")
    assert learned_term_allowed("日本語")
    assert not learned_term_allowed("中文")  # only 2 chars


# --------------------------------------------------------------------- #
# fold_key
# --------------------------------------------------------------------- #


def test_fold_key_empty_inputs() -> None:
    assert fold_key(None) == ""
    assert fold_key("") == ""
    assert fold_key("   ") == ""
    assert fold_key("!!! --- ...") == ""


def test_fold_key_collapses_case_punctuation_whitespace() -> None:
    assert fold_key("Sign-Off") == "signoff"
    assert fold_key("sign off") == "signoff"
    assert fold_key("sign off") == "signoff"  # NBSP
    assert fold_key("SIGN  OFF") == "signoff"
    assert fold_key("Q.B.R") == "qbr"
    assert fold_key("Q B R") == "qbr"
    assert fold_key("my-email") == "myemail"


def test_fold_key_strips_diacritics() -> None:
    assert fold_key("café") == "cafe"
    assert fold_key("naïve") == "naive"
    # NFC and NFD forms of the same word fold identically.
    assert fold_key("café") == "cafe"


def test_fold_key_keeps_digits() -> None:
    assert fold_key("Llama 3.1") == "llama31"


def test_fold_key_non_latin_scripts_fold_to_empty() -> None:
    # Documented behaviour: CJK / Cyrillic survive only via the
    # fold_match_pattern fallback, never via the latin key.
    assert fold_key("日本語") == ""
    assert fold_key("привет") == ""


def test_fold_key_fullwidth_compatibility_forms() -> None:
    # NFKD decomposition maps fullwidth latin into ASCII.
    assert fold_key("ＱＢＲ") == "qbr"


# --------------------------------------------------------------------- #
# fold_match_pattern
# --------------------------------------------------------------------- #


def test_fold_match_pattern_refuses_empty_and_punctuation() -> None:
    assert fold_match_pattern(None) is None
    assert fold_match_pattern("") is None
    assert fold_match_pattern("!!!") is None
    assert fold_match_pattern("  - .. ") is None


def test_fold_match_pattern_matches_whitespace_punct_variants() -> None:
    pattern = fold_match_pattern("sign off")
    assert pattern is not None
    rx = re.compile(pattern, re.IGNORECASE)
    assert rx.search("signoff")
    assert rx.search("Sign-Off")
    assert rx.search("please sign  off now")
    assert rx.search("sign\toff")
    # Word boundaries on the alphanumeric side: no partial-word hits.
    assert not rx.search("signoffer")
    assert not rx.search("resignoff")


def test_fold_match_pattern_acronym_with_separators() -> None:
    pattern = fold_match_pattern("QBR")
    assert pattern is not None
    rx = re.compile(pattern, re.IGNORECASE)
    assert rx.search("Q.B.R")
    assert rx.search("Q B R")
    assert rx.search("qbr")
    # Short triggers must not fire inside longer words.
    assert not rx.search("acquire bar")


def test_fold_match_pattern_non_latin_falls_back_to_exact() -> None:
    pattern = fold_match_pattern("日本語")
    assert pattern is not None
    rx = re.compile(pattern)
    assert rx.search("日本語")
    assert rx.search("the 日本語 course")
    assert not rx.search("english only")


# --------------------------------------------------------------------- #
# looks_like_hallucination
# --------------------------------------------------------------------- #


def test_hallucination_short_or_empty_text_is_not_flagged() -> None:
    assert not looks_like_hallucination("")
    assert not looks_like_hallucination("ab")


def test_hallucination_low_alnum_ratio() -> None:
    assert looks_like_hallucination("... !!! --- ???")


def test_hallucination_all_short_token_noise() -> None:
    # Case (1): "A.D. A.D. A.D." — dominant single-letter tokens.
    assert looks_like_hallucination("A.D. A.D. A.D.")


def test_hallucination_dominant_substantive_token() -> None:
    # Case (2): one real word repeated.
    assert looks_like_hallucination("thu thu thu thu thu")


def test_hallucination_confidence_floor_spares_intentional_repetition() -> None:
    text = "hello hello hello hello hello"
    assert looks_like_hallucination(text, confidence=None)
    assert looks_like_hallucination(text, confidence=-2.0)
    # Confident speech above the -1.0 floor is legitimate emphasis.
    assert not looks_like_hallucination(text, confidence=-0.5)


def test_hallucination_cjk_glyph_loop() -> None:
    assert looks_like_hallucination("あああ")


def test_hallucination_single_character_dominance() -> None:
    assert looks_like_hallucination("aaaaaaaaaaaa b")


def test_hallucination_substring_loop_without_whitespace() -> None:
    # Case (4): no spaces, so \b\w+\b sees a single giant token.
    assert looks_like_hallucination("ansaansaansaansaansa")


def test_hallucination_substring_loop_spared_when_confident() -> None:
    assert not looks_like_hallucination(
        "ansaansaansaansaansa", confidence=-0.5
    )


def test_hallucination_low_distinct_ratio_long_text() -> None:
    # Case (5): 60 copies of one word — fires even with high confidence
    # because the n-gram loop structure corroborates.
    text = " ".join(["cardiac"] * 60)
    assert looks_like_hallucination(text)
    assert looks_like_hallucination(text, confidence=-0.2)


def test_hallucination_phrase_ngram_loop_long_text() -> None:
    text = " ".join(["make sure that we"] * 50)
    assert looks_like_hallucination(text, confidence=-0.2)


def test_hallucination_mixed_script_within_token() -> None:
    # Case (6): Latin + Cyrillic fused in one word.
    assert looks_like_hallucination("the report Clициц arrived")
    # Confident decode is spared.
    assert not looks_like_hallucination(
        "the report Clициц arrived", confidence=-0.5
    )


def test_hallucination_legitimate_code_mixing_with_space_is_fine() -> None:
    assert not looks_like_hallucination("Hello мир and welcome everyone")


def test_hallucination_numeric_marker_loop() -> None:
    assert looks_like_hallucination("Step 1. 2. 3. 4. 5. 6. done")


def test_hallucination_legitimate_short_numbered_list_survives() -> None:
    assert not looks_like_hallucination(
        "1. apples 2. bananas 3. cherries"
    )


def test_hallucination_version_string_is_legitimate() -> None:
    # Spared shape (a) from the docstring.
    assert not looks_like_hallucination("Version 3.0.0.0 released")


def test_hallucination_normal_sentence_is_legitimate() -> None:
    assert not looks_like_hallucination(
        "The quick brown fox jumps over the lazy dog while we record audio"
    )


# --------------------------------------------------------------------- #
# detect_tail_repetition
# --------------------------------------------------------------------- #


def test_detect_tail_repetition_phrase_with_truncated_tail() -> None:
    words = "Some can be saved. Some can be saved. Som".split()
    assert detect_tail_repetition(words) == (4, 2)


def test_detect_tail_repetition_three_full_copies() -> None:
    words = ("Some can come to me. " * 3).split()
    assert detect_tail_repetition(words) == (5, 3)


def test_detect_tail_repetition_rejects_single_word_units() -> None:
    # "hello hello hello" is handled by the confidence gate, not here.
    assert detect_tail_repetition(["hello"] * 6) is None


def test_detect_tail_repetition_none_on_normal_text() -> None:
    assert detect_tail_repetition("the quick brown fox jumps".split()) is None


def test_detect_tail_repetition_none_on_short_input() -> None:
    assert detect_tail_repetition(["a", "b", "c"]) is None
    assert detect_tail_repetition([]) is None


# --------------------------------------------------------------------- #
# collapse_tail_repetition
# --------------------------------------------------------------------- #


def test_collapse_empty_text() -> None:
    cleaned, diag = collapse_tail_repetition("", audio_duration_ms=1000)
    assert cleaned == ""
    assert not diag.collapsed
    assert diag.reason == "empty_text"


def test_collapse_too_short() -> None:
    cleaned, diag = collapse_tail_repetition("just one", audio_duration_ms=1000)
    assert cleaned == "just one"
    assert diag.reason == "too_short"


def test_collapse_fires_on_implausible_speech_rate() -> None:
    text = "Some can be saved. Some can be saved. Some can be saved."
    # 12 words in 1.5s = 8 wps — way past the 5 wps ceiling.
    cleaned, diag = collapse_tail_repetition(text, audio_duration_ms=1500)
    assert cleaned == "Some can be saved."
    assert diag.collapsed
    assert diag.reason == "tail_repetition_collapsed"
    assert diag.period_words == 4
    assert diag.copies == 3
    assert diag.removed_words == 8


def test_collapse_respects_natural_speech_rate() -> None:
    text = "Some can be saved. Some can be saved. Some can be saved."
    # 12 words in 12s = 1 wps — physically plausible, leave it alone.
    cleaned, diag = collapse_tail_repetition(text, audio_duration_ms=12000)
    assert cleaned == text
    assert not diag.collapsed
    assert diag.reason == "within_natural_speech_rate"
    assert diag.period_words == 4
    assert diag.copies == 3


def test_collapse_without_duration_uses_pattern_alone() -> None:
    text = "Some can come to me. Some can come to me. Some"
    cleaned, diag = collapse_tail_repetition(text, audio_duration_ms=None)
    assert cleaned == "Some can come to me."
    assert diag.collapsed
    assert diag.words_per_second is None


def test_collapse_no_repetition_found() -> None:
    text = "this is a perfectly normal sentence with no loops"
    cleaned, diag = collapse_tail_repetition(text, audio_duration_ms=None)
    assert cleaned == text
    assert diag.reason == "no_repetition_found"


def test_collapse_single_token_suffix_after_real_sentence() -> None:
    text = "Please send the final report now " + " ".join(["Cent"] * 30)
    cleaned, diag = collapse_tail_repetition(text, audio_duration_ms=None)
    assert cleaned == "Please send the final report now"
    assert diag.collapsed
    assert diag.reason == "tail_single_token_repetition_collapsed"
    assert diag.repeated_token == "cent"
    assert diag.copies == 30
    assert diag.removed_words == 30


def test_collapse_spares_intentional_short_emphasis() -> None:
    # All-same-token utterance without a substantive prefix: not ours.
    text = "yes yes yes yes yes yes yes"
    cleaned, diag = collapse_tail_repetition(text, audio_duration_ms=None)
    assert cleaned == text
    assert not diag.collapsed


def test_collapse_preserves_leading_whitespace() -> None:
    text = "  Some can be saved. Some can be saved. Some can be saved."
    cleaned, diag = collapse_tail_repetition(text, audio_duration_ms=None)
    assert diag.collapsed
    assert cleaned == "  Some can be saved."


def test_collapse_diagnostic_to_dict_roundtrip() -> None:
    _, diag = collapse_tail_repetition(
        "Some can be saved. Some can be saved. Some can be saved.",
        audio_duration_ms=1000,
    )
    d = diag.to_dict()
    assert d["collapsed"] is True
    assert d["period_words"] == 4
    assert d["copies"] == 3


def test_collapse_is_idempotent() -> None:
    text = "Some can be saved. Some can be saved. Some can be saved."
    once, _ = collapse_tail_repetition(text, audio_duration_ms=None)
    twice, diag = collapse_tail_repetition(once, audio_duration_ms=None)
    assert twice == once
    assert not diag.collapsed


# --------------------------------------------------------------------- #
# diff_pasted_segment
# --------------------------------------------------------------------- #


def test_diff_returns_none_on_empty_inputs() -> None:
    assert diff_pasted_segment(expected="", observed="anything") is None
    assert diff_pasted_segment(expected="anything", observed="") is None
    assert diff_pasted_segment(expected="", observed="") is None


def test_diff_returns_none_when_nothing_changed() -> None:
    assert diff_pasted_segment(expected="hello world", observed="hello world") is None
    # Strip-equal counts as unchanged too.
    assert diff_pasted_segment(expected="  hello world ", observed="hello world") is None


def test_diff_returns_none_when_paste_survived_verbatim() -> None:
    # The user edited some OTHER part of the field; not a correction.
    assert (
        diff_pasted_segment(
            expected="hello world",
            observed="Dear team, hello world. Regards, Sam",
        )
        is None
    )


def test_diff_returns_none_when_user_retyped_from_scratch() -> None:
    assert (
        diff_pasted_segment(
            expected="alpha beta gamma",
            observed="the quick brown fox jumped over everything",
        )
        is None
    )


def test_diff_detects_one_word_fix() -> None:
    result = diff_pasted_segment(
        expected="meet Jhon tomorrow afternoon",
        observed="meet John tomorrow afternoon",
    )
    assert result is not None
    pasted, corrected = result
    assert pasted == "meet Jhon tomorrow afternoon"
    assert corrected == "meet John tomorrow afternoon"


def test_diff_windows_edit_inside_long_field() -> None:
    prefix = "Dear team, thanks for joining the call earlier today. "
    suffix = " Please review before Friday and reply with comments."
    result = diff_pasted_segment(
        expected="the budjet numbers look good",
        observed=prefix + "the budget numbers look good" + suffix,
    )
    assert result is not None
    pasted, corrected = result
    assert pasted == "the budjet numbers look good"
    assert "budget" in corrected
    # The window stays local to the edit; whole-field pollution is the
    # exact bug this helper exists to fix.
    assert "Dear team" not in corrected
    assert "before Friday" not in corrected
    assert len(corrected) <= MAX_CORRECTION_TEXT_CHARS


def test_diff_whitespace_normalized_locate() -> None:
    # Paste survived modulo whitespace; the surrounding edit is captured.
    result = diff_pasted_segment(
        expected="ship the  release notes",
        observed="ship the release notes today",
    )
    assert result is not None
    pasted, corrected = result
    assert pasted == "ship the  release notes"
    assert "today" in corrected


def test_diff_caps_output_lengths() -> None:
    long_pasted = "alpha bravo charlie delta echo " * 8  # ~248 chars
    long_pasted = long_pasted.strip()
    # One character changed near the front so the verbatim find fails.
    observed = "alphA" + long_pasted[5:]
    result = diff_pasted_segment(expected=long_pasted, observed=observed)
    assert result is not None
    pasted, corrected = result
    assert len(pasted) <= MAX_CORRECTION_TEXT_CHARS
    assert len(corrected) <= MAX_CORRECTION_TEXT_CHARS


def test_diff_too_short_paste_cannot_fuzzy_match() -> None:
    assert diff_pasted_segment(expected="hi", observed="ho") is None


# --------------------------------------------------------------------- #
# ai_dictionary
# --------------------------------------------------------------------- #


def test_glossary_matches_case_insensitively() -> None:
    assert is_ai_glossary_term("Qwen")
    assert is_ai_glossary_term("qwen")
    assert is_ai_glossary_term("QWEN")
    assert is_ai_glossary_term("gguf")
    assert is_ai_glossary_term("whisper")


def test_glossary_strips_surrounding_whitespace() -> None:
    assert is_ai_glossary_term("  Llama  ")


def test_glossary_rejects_non_members() -> None:
    assert not is_ai_glossary_term("")
    assert not is_ai_glossary_term("GPT")  # deliberately excluded
    assert not is_ai_glossary_term("banana")
    assert not is_ai_glossary_term("Qwenx")  # no substring matching


def test_glossary_entries_respect_max_length() -> None:
    for entry in AI_GLOSSARY:
        assert len(entry) <= 24


def test_glossary_entries_respect_min_length() -> None:
    for entry in AI_GLOSSARY:
        assert len(entry) >= 3, entry


# --------------------------------------------------------------------- #
# looks_like_low_yield_garbage
# --------------------------------------------------------------------- #


def test_low_yield_garbage_catches_lamb_ampersand() -> None:
    from juno_v2.memory.hallucination import looks_like_low_yield_garbage

    # Production 2026-06-11: 12.5 s of mostly-silent audio decoded to
    # "Lamb &" at avg_logprob -2.11 and pasted.
    assert looks_like_low_yield_garbage(
        "Lamb &", confidence=-2.11, audio_duration_ms=12485.0
    )


def test_low_yield_garbage_spares_confident_short_utterances() -> None:
    from juno_v2.memory.hallucination import looks_like_low_yield_garbage

    # A real short utterance in a long buffer decodes confidently.
    assert not looks_like_low_yield_garbage(
        "I mean.", confidence=-0.4, audio_duration_ms=12485.0
    )
    # Short audio legitimately yields few words at any confidence.
    assert not looks_like_low_yield_garbage(
        "Yes.", confidence=-1.8, audio_duration_ms=1500.0
    )
    # Long low-confidence decodes with real word yield are left to the
    # repetition/loop heuristics, not this gate.
    assert not looks_like_low_yield_garbage(
        "this is a longer sentence with many words in it",
        confidence=-1.8,
        audio_duration_ms=12485.0,
    )
    # Unknown confidence: never fire.
    assert not looks_like_low_yield_garbage(
        "Lamb &", confidence=None, audio_duration_ms=12485.0
    )
