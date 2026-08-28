from __future__ import annotations

import pytest

from juno_v2.itn.format_policy import ITNFormatPolicy
from juno_v2.itn.rules import (
    apply_code_identifiers,
    apply_currency,
    apply_dates,
    apply_email_url,
    apply_numeric,
    apply_spoken_punctuation,
    apply_terminal_ops,
    apply_times,
)


# ---------------------------------------------------------------------- #
# apply_numeric
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("twenty five", "25"),
        ("ninety nine problems", "99 problems"),
        ("seventeen", "17"),
        ("forty two is the answer", "42 is the answer"),
        ("Twenty Five", "25"),  # case-insensitive
    ],
)
def test_apply_numeric_converts_words(text: str, expected: str) -> None:
    out, applied = apply_numeric(text)
    assert out == expected
    assert applied == ["numeric_words_to_digits"]


@pytest.mark.parametrize(
    "text",
    [
        # Prose-aware numerals: standalone small numbers stay words unless
        # numeric context is present ("chapter one" converts; "one apple"
        # does not). See test_itn_spoken_punctuation.py for the full
        # context-sensitivity matrix.
        "I have one apple",
        "zero",
    ],
)
def test_apply_numeric_keeps_small_numbers_in_prose(text: str) -> None:
    out, applied = apply_numeric(text)
    assert out == text
    assert applied == []


@pytest.mark.parametrize("text", ["", "no numbers here", "onety stays", "thousand"])
def test_apply_numeric_no_op(text: str) -> None:
    out, applied = apply_numeric(text)
    assert out == text
    assert applied == []


def test_apply_numeric_idempotent() -> None:
    once, _ = apply_numeric("twenty five and sixty")
    twice, applied = apply_numeric(once)
    assert twice == once
    assert applied == []


# ---------------------------------------------------------------------- #
# apply_currency
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("five dollars", "$5"),
        ("five dollars and twenty cents", "$5.20"),
        ("three dollars and five cents", "$3.05"),  # zero-padded minor units
        ("one hundred and fifty euros", "€150"),
        ("ten pounds", "£10"),
        ("twenty swiss francs", "CHF 20"),
        ("five hundred yen", "¥500"),
        ("two thousand dollars", "$2000"),
        ("one million dollars", "$1000000"),
    ],
)
def test_apply_currency_conversions(text: str, expected: str) -> None:
    out, applied = apply_currency(text)
    assert out == expected
    assert applied == ["currency"]


def test_apply_currency_inline_context() -> None:
    out, applied = apply_currency("it costs five dollars and twenty cents today")
    assert out == "it costs $5.20 today"
    assert applied == ["currency"]


def test_apply_currency_comma_decimal_policy() -> None:
    fmt = ITNFormatPolicy(currency_decimal="comma")
    out, _ = apply_currency("five dollars and twenty cents", fmt)
    assert out == "$5,20"


@pytest.mark.parametrize("text", ["", "dollars to donuts", "no money here"])
def test_apply_currency_no_op(text: str) -> None:
    out, applied = apply_currency(text)
    assert out == text
    assert applied == []


# ---------------------------------------------------------------------- #
# apply_dates
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("april nineteenth", "Apr 19"),
        ("march 5", "Mar 5"),
        ("june first twenty twenty", "Jun 1, 2020"),
        ("may twelfth nineteen ninety", "May 12, 1990"),
        ("December twenty-fifth", "Dec 25"),
        ("meet on april nineteenth okay", "meet on Apr 19 okay"),
    ],
)
def test_apply_dates_default_policy(text: str, expected: str) -> None:
    out, applied = apply_dates(text)
    assert out == expected
    assert applied == ["dates"]


@pytest.mark.parametrize(
    ("style", "with_year", "without_year"),
    [
        ("iso", "1990-03-03", "03-03"),
        ("dmy_slash", "03/03/1990", "03/03"),
        ("mdy_slash", "03/03/1990", "03/03"),
        ("dmy_long", "3 March 1990", "3 March"),
    ],
)
def test_apply_dates_format_policies(style: str, with_year: str, without_year: str) -> None:
    fmt = ITNFormatPolicy(date_style=style)
    assert apply_dates("march third nineteen ninety", fmt)[0] == with_year
    assert apply_dates("march third", fmt)[0] == without_year


@pytest.mark.parametrize("text", ["", "not a date", "may be later"])
def test_apply_dates_no_op(text: str) -> None:
    out, applied = apply_dates(text)
    assert out == text
    assert applied == []


def test_apply_dates_three_word_year() -> None:
    out, _ = apply_dates("april nineteenth twenty twenty six")
    assert out == "Apr 19, 2026"


# ---------------------------------------------------------------------- #
# apply_times
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("three thirty pm", "3:30 PM"),
        ("nine am", "9:00 AM"),
        ("twelve pm", "12:00 PM"),
        ("eleven fifteen am", "11:15 AM"),
        ("14 hundred", "14:00"),
        ("9 hundred", "09:00"),
        ("see you at three thirty pm sharp", "see you at 3:30 PM sharp"),
    ],
)
def test_apply_times_default_policy(text: str, expected: str) -> None:
    out, applied = apply_times(text)
    assert out == expected
    assert applied == ["times"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("three thirty pm", "15:30"),
        ("twelve am", "00:00"),   # midnight
        ("twelve pm", "12:00"),   # noon
        ("eleven fifteen am", "11:15"),
    ],
)
def test_apply_times_24h_policy(text: str, expected: str) -> None:
    fmt = ITNFormatPolicy(clock="24h")
    out, _ = apply_times(text, fmt)
    assert out == expected


@pytest.mark.parametrize("text", ["", "no time here", "hundred"])
def test_apply_times_no_op(text: str) -> None:
    out, applied = apply_times(text)
    assert out == text
    assert applied == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Word-form military time needs explicit time context ("at" prefix
        # or "hours" suffix); a bare "fourteen hundred" is usually a
        # quantity ("fourteen hundred people") and must not convert.
        ("at fourteen hundred", "at 14:00"),
        ("At fourteen hundred", "At 14:00"),
        ("fourteen hundred hours", "14:00"),
        ("at twenty one hundred", "at 21:00"),
        ("fourteen hundred", "fourteen hundred"),
        ("fourteen hundred people", "fourteen hundred people"),
        ("at twenty five hundred", "at twenty five hundred"),
    ],
)
def test_apply_times_spoken_military_time(text: str, expected: str) -> None:
    out, _ = apply_times(text)
    assert out == expected


def test_apply_times_dotted_meridiem() -> None:
    out, _ = apply_times("three thirty p.m.")
    assert out == "3:30 PM"


# ---------------------------------------------------------------------- #
# apply_email_url
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("support at example dot com", "support@example.com"),
        ("email me at john at company dot org", "email me at john@company.org"),
        ("www dot example dot com", "www.example.com"),
        ("https colon slash slash example", "https://example"),
        ("ftp colon double slash mirror", "ftp://mirror"),
    ],
)
def test_apply_email_url_conversions(text: str, expected: str) -> None:
    out, applied = apply_email_url(text)
    assert out == expected
    assert applied == ["email_url"]


@pytest.mark.parametrize("text", ["", "we met at noon", "dot dot dot"])
def test_apply_email_url_no_op(text: str) -> None:
    out, applied = apply_email_url(text)
    assert out == text
    assert applied == []


# ---------------------------------------------------------------------- #
# apply_code_identifiers
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("main dot ts", "main.ts"),
        ("index underscore test", "index_test"),
        ("feature dash branch", "feature-branch"),
        ("dot slash build.sh", "./build.sh"),
        ("open main dot py now", "open main.py now"),
    ],
)
def test_apply_code_identifiers(text: str, expected: str) -> None:
    out, applied = apply_code_identifiers(text)
    assert out == expected
    assert applied == ["code_identifiers"]


@pytest.mark.parametrize("text", ["", "plain words only", "dot", "underscore"])
def test_apply_code_identifiers_no_op(text: str) -> None:
    out, applied = apply_code_identifiers(text)
    assert out == text
    assert applied == []


# ---------------------------------------------------------------------- #
# apply_terminal_ops
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pipe", "|"),
        ("ls pipe grep foo", "ls | grep foo"),
        ("greater than", ">"),
        ("less than", "<"),
        ("ampersand", "&"),
        ("echo done greater than out.txt", "echo done > out.txt"),
    ],
)
def test_apply_terminal_ops(text: str, expected: str) -> None:
    out, applied = apply_terminal_ops(text)
    assert out == expected
    assert applied == ["terminal_ops"]


@pytest.mark.parametrize("text", ["", "nothing shelly here"])
def test_apply_terminal_ops_no_op(text: str) -> None:
    out, applied = apply_terminal_ops(text)
    assert out == text
    assert applied == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("double ampersand", "&&"),
        ("double pipe", "||"),
        ("make build double ampersand make test", "make build && make test"),
    ],
)
def test_apply_terminal_ops_double_operators(text: str, expected: str) -> None:
    out, _ = apply_terminal_ops(text)
    assert out == expected


@pytest.mark.parametrize(
    "text",
    [
        # A determiner in front makes the operator word a noun, exactly as
        # it does for spoken punctuation ("put a comma here").
        "put a pipe here",
        "the pipe is rusty",
        "an ampersand goes at the end",
        "use the ampersand character",
        "the greater than sign",
        # The guard applies to the double forms too, and blocking "double
        # pipe" must not let the bare "pipe" rule fire on its tail.
        "a double pipe means or",
        "the double ampersand operator",
    ],
)
def test_apply_terminal_ops_literal_mentions_preserved(text: str) -> None:
    out, applied = apply_terminal_ops(text)
    assert out == text
    assert applied == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Bare cues (no determiner) still convert.
        ("ls pipe grep foo", "ls | grep foo"),
        ("ls pipe grep foo double ampersand echo ok", "ls | grep foo && echo ok"),
        ("echo hi greater than out.txt", "echo hi > out.txt"),
    ],
)
def test_apply_terminal_ops_bare_cues_still_convert(text: str, expected: str) -> None:
    out, applied = apply_terminal_ops(text)
    assert out == expected
    assert applied == ["terminal_ops"]


# ---------------------------------------------------------------------- #
# apply_spoken_punctuation
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello comma world", "hello, world"),
        ("done period new line goodbye", "done.\ngoodbye"),
        ("end of sentence full stop next", "end of sentence. next"),
        ("is this right question mark", "is this right?"),
        ("wait exclamation point", "wait!"),
        ("wait exclamation mark", "wait!"),
        ("first item semicolon second item", "first item; second item"),
        ("note colon remember this", "note: remember this"),
        ("one em dash two", "one — two"),
        ("intro new paragraph body", "intro\n\nbody"),
    ],
)
def test_apply_spoken_punctuation(text: str, expected: str) -> None:
    out, applied = apply_spoken_punctuation(text)
    assert out == expected
    assert applied == ["spoken_punctuation"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # De-dup against an existing adjacent glyph (docstring example).
        ("hello, comma world", "hello, world"),
        # De-dup repeated spoken tokens.
        ("hello comma comma world", "hello, world"),
    ],
)
def test_apply_spoken_punctuation_dedup(text: str, expected: str) -> None:
    out, _ = apply_spoken_punctuation(text)
    assert out == expected


@pytest.mark.parametrize(
    "text",
    [
        "type the word comma",
        "the word comma is punctuation",
        "",
    ],
)
def test_apply_spoken_punctuation_literal_mentions_untouched(text: str) -> None:
    out, applied = apply_spoken_punctuation(text)
    assert out == text
    assert applied == []


def test_apply_spoken_punctuation_idempotent() -> None:
    once, _ = apply_spoken_punctuation("hello comma world period done")
    twice, applied = apply_spoken_punctuation(once)
    assert twice == once
    assert applied == []


def test_apply_spoken_punctuation_open_paren_attaches() -> None:
    out, _ = apply_spoken_punctuation("open paren hello close paren")
    assert out == "(hello)"


def test_apply_spoken_punctuation_open_paren_keeps_leading_space() -> None:
    out, _ = apply_spoken_punctuation("see open paren note close paren here")
    assert out == "see (note) here"


def test_apply_spoken_punctuation_hyphen_tight() -> None:
    out, _ = apply_spoken_punctuation("well hyphen known")
    assert out == "well-known"
