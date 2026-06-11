from __future__ import annotations

import pytest

from juno_v2.language.normalize import LanguageAwareNormalizer, summarize_scripts


@pytest.fixture()
def normalizer() -> LanguageAwareNormalizer:
    return LanguageAwareNormalizer()


# ---------------------------------------------------------------------- #
# normalize_transcript
# ---------------------------------------------------------------------- #


def test_normalize_transcript_collapses_whitespace_and_punct_spacing(
    normalizer: LanguageAwareNormalizer,
) -> None:
    result = normalizer.normalize_transcript(
        "  hello   world ,  test  ",
        requested_language="en",
        observed_language="en",
        policy_name="default",
        scope="final",
    )
    assert result.raw_text == "  hello   world ,  test  "
    assert result.normalized_text == "hello world, test"
    assert [change.source for change in result.applied] == [
        "whitespace_collapse",
        "space_before_punctuation",
    ]


def test_normalize_transcript_metadata(normalizer: LanguageAwareNormalizer) -> None:
    result = normalizer.normalize_transcript(
        "hello",
        requested_language="en",
        observed_language="fr",
        policy_name="strict",
        scope="preview",
    )
    meta = result.metadata
    assert meta["scope"] == "preview"
    assert meta["requested_language"] == "en"
    assert meta["observed_language"] == "fr"
    assert meta["policy_name"] == "strict"
    assert meta["script_summary"]["latin"] == 5


def test_normalize_transcript_empty_and_none_safe(normalizer: LanguageAwareNormalizer) -> None:
    result = normalizer.normalize_transcript(
        "",
        requested_language=None,
        observed_language=None,
        policy_name=None,
        scope="final",
    )
    assert result.raw_text == ""
    assert result.normalized_text == ""
    assert result.applied == []


def test_normalize_transcript_clean_text_is_untouched(normalizer: LanguageAwareNormalizer) -> None:
    result = normalizer.normalize_transcript(
        "Hello, world.",
        requested_language="en",
        observed_language="en",
        policy_name=None,
        scope="final",
    )
    assert result.normalized_text == "Hello, world."
    assert result.applied == []


def test_normalize_transcript_does_not_lowercase(normalizer: LanguageAwareNormalizer) -> None:
    result = normalizer.normalize_transcript(
        "Hello World",
        requested_language="en",
        observed_language=None,
        policy_name=None,
        scope="final",
    )
    assert result.normalized_text == "Hello World"


def test_normalize_transcript_multi_dash_becomes_em_dash(
    normalizer: LanguageAwareNormalizer,
) -> None:
    result = normalizer.normalize_transcript(
        "a -- b --- c",
        requested_language="en",
        observed_language=None,
        policy_name=None,
        scope="final",
    )
    assert result.normalized_text == "a — b — c"
    assert [change.source for change in result.applied] == ["dash_normalization"]


def test_normalize_transcript_hindi_danda_spacing(normalizer: LanguageAwareNormalizer) -> None:
    result = normalizer.normalize_transcript(
        "नमस्ते ।दुनिया",
        requested_language="hi",
        observed_language=None,
        policy_name=None,
        scope="final",
    )
    assert result.normalized_text == "नमस्ते। दुनिया"
    assert "hindi_danda_spacing" in [change.source for change in result.applied]


def test_normalize_transcript_arabic_comma_spacing(normalizer: LanguageAwareNormalizer) -> None:
    result = normalizer.normalize_transcript(
        "مرحبا ،عالم",
        requested_language="ar",
        observed_language=None,
        policy_name=None,
        scope="final",
    )
    assert result.normalized_text == "مرحبا، عالم"
    assert "arabic_comma_spacing" in [change.source for change in result.applied]


def test_normalize_transcript_cjk_punctuation_hugging(normalizer: LanguageAwareNormalizer) -> None:
    result = normalizer.normalize_transcript(
        "你好 。 世界",
        requested_language="zh",
        observed_language=None,
        policy_name=None,
        scope="final",
    )
    # CJK full-width punctuation gets no surrounding spaces.
    assert result.normalized_text == "你好。世界"
    assert "cjk_punctuation_spacing" in [change.source for change in result.applied]


def test_normalize_transcript_observed_language_used_when_requested_missing(
    normalizer: LanguageAwareNormalizer,
) -> None:
    result = normalizer.normalize_transcript(
        "नमस्ते ।दुनिया",
        requested_language=None,
        observed_language="hi-IN",
        policy_name=None,
        scope="final",
    )
    assert result.normalized_text == "नमस्ते। दुनिया"


def test_normalize_transcript_changes_record_before_after(
    normalizer: LanguageAwareNormalizer,
) -> None:
    result = normalizer.normalize_transcript(
        "a  b",
        requested_language="en",
        observed_language=None,
        policy_name=None,
        scope="final",
    )
    assert len(result.applied) == 1
    change = result.applied[0]
    assert change.kind == "language_normalization"
    assert change.before == "a  b"
    assert change.after == "a b"


# ---------------------------------------------------------------------- #
# normalize_for_eval
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("  Hello   World !  ", "en", "hello world!"),
        ("Hello World", None, "hello world"),
        ("Hello , world", "en", "hello, world"),
        ("", "en", ""),
        ("ALL CAPS", "de", "all caps"),
    ],
)
def test_normalize_for_eval_lowercases_latin(text: str, language: str | None, expected: str) -> None:
    assert LanguageAwareNormalizer().normalize_for_eval(text, language=language) == expected


@pytest.mark.parametrize("language", ["zh", "zh-CN", "ja", "ja_JP", "ko", "th"])
def test_normalize_for_eval_cjk_thai_not_lowercased(language: str) -> None:
    # Mixed-script eval text keeps Latin casing intact for CJK/Thai languages.
    out = LanguageAwareNormalizer().normalize_for_eval("你好 World", language=language)
    assert out == "你好 World"


def test_normalize_for_eval_none_text() -> None:
    assert LanguageAwareNormalizer().normalize_for_eval(None) == ""


# ---------------------------------------------------------------------- #
# summarize_scripts
# ---------------------------------------------------------------------- #


def test_summarize_scripts_counts_letters_only() -> None:
    summary = summarize_scripts("abc 123 !?")
    d = summary.to_dict()
    assert d["latin"] == 3
    assert d["total_letters"] == 3


def test_summarize_scripts_mixed_scripts() -> None:
    d = summarize_scripts("hello नमस्ते 你好 こんにちは 안녕 مرحبا привет").to_dict()
    assert d["latin"] == 5
    assert d["devanagari"] == 4
    assert d["han"] == 2
    assert d["kana"] == 5
    assert d["hangul"] == 2
    assert d["arabic"] == 5
    assert d["cyrillic"] == 6
    assert d["code_switch_detected"] is True


def test_summarize_scripts_empty() -> None:
    d = summarize_scripts("").to_dict()
    assert d["total_letters"] == 0
