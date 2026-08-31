from __future__ import annotations

import pytest

from juno_v2.itn.engine import ITNEngine, ITNProfile
from juno_v2.itn.format_policy import ITNFormatPolicy, resolve_itn_format_policy


# ---------------------------------------------------------------------- #
# ITNEngine.profile_for_category
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("messaging", ITNProfile.PROSE),
        ("email", ITNProfile.FULL),
        ("docs", ITNProfile.FULL),
        ("forms", ITNProfile.PROSE),
        ("code", ITNProfile.CODE),
        ("terminal", ITNProfile.TERMINAL),
        ("unknown", ITNProfile.PROSE),
        ("", ITNProfile.PROSE),
        (None, ITNProfile.PROSE),
        ("TERMINAL", ITNProfile.TERMINAL),  # case-insensitive
        ("bogus-category", ITNProfile.PROSE),  # unmapped falls back to prose
    ],
)
def test_profile_for_category(category: str | None, expected: ITNProfile) -> None:
    assert ITNEngine().profile_for_category(category) is expected


# ---------------------------------------------------------------------- #
# ITNEngine.run
# ---------------------------------------------------------------------- #


def test_run_full_profile_combines_rules() -> None:
    engine = ITNEngine()
    result = engine.run(
        "five dollars on april nineteenth at three thirty pm comma okay",
        profile=ITNProfile.FULL,
    )
    assert result.text == "$5 on Apr 19 at 3:30 PM, okay"
    assert result.original_text == "five dollars on april nineteenth at three thirty pm comma okay"
    assert result.profile == "full"
    assert result.changed is True
    assert result.rules_applied == ["currency", "dates", "times", "spoken_punctuation"]


def test_run_none_profile_passes_through() -> None:
    result = ITNEngine().run("twenty five dollars", profile=ITNProfile.NONE)
    assert result.text == "twenty five dollars"
    assert result.changed is False
    assert result.rules_applied == []
    assert result.profile == "none"


def test_run_accepts_profile_string() -> None:
    result = ITNEngine().run("twenty five", profile="none")
    assert result.text == "twenty five"
    assert result.profile == "none"


def test_run_invalid_profile_string_falls_back_to_prose() -> None:
    result = ITNEngine().run("twenty five", profile="not-a-profile")
    assert result.profile == "prose"
    assert result.text == "25"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Issue #97 — currency runs before the numeric rule, so a currency
        # match that started inside "twenty-five" left a fragment for the
        # numeric rule to convert separately ("twenty-five dollars" -> "20-$5").
        ("twenty-five dollars", "$25"),
        ("twenty-five dollars and thirty-two cents", "$25.32"),
        ("twenty-five percent", "25 percent"),
        ("it costs twenty-five dollars today", "it costs $25 today"),
        # Unchanged: space-separated forms and non-numeric compounds.
        ("twenty five dollars", "$25"),
        ("a multi-million dollar deal", "a multi-million dollar deal"),
        ("well-known state-of-the-art work", "well-known state-of-the-art work"),
    ],
)
def test_run_hyphenated_numbers_normalise_as_whole_values(text: str, expected: str) -> None:
    assert ITNEngine().run(text, profile=ITNProfile.PROSE).text == expected


def test_run_empty_text_short_circuits() -> None:
    result = ITNEngine().run("", profile=ITNProfile.FULL)
    assert result.text == ""
    assert result.original_text == ""
    assert result.changed is False
    assert result.rules_applied == []
    # Empty text still records the requested profile and format snapshot.
    assert result.profile == "full"
    assert result.format_snapshot == ITNFormatPolicy.default().to_summary_dict()


def test_run_code_profile_only_collapses_identifiers() -> None:
    result = ITNEngine().run("main dot ts pipe grep", profile=ITNProfile.CODE)
    # CODE never runs terminal ops — "pipe" stays a word.
    assert result.text == "main.ts pipe grep"
    assert result.rules_applied == ["code_identifiers"]


def test_run_terminal_profile_runs_ops_then_identifiers() -> None:
    result = ITNEngine().run("cat main dot ts pipe grep foo", profile=ITNProfile.TERMINAL)
    assert result.text == "cat main.ts | grep foo"
    assert sorted(result.rules_applied) == ["code_identifiers", "terminal_ops"]


def test_run_prose_profile_does_not_touch_email_forms() -> None:
    result = ITNEngine().run("support at example dot com", profile=ITNProfile.PROSE)
    assert result.text == "support at example dot com"
    assert result.changed is False


def test_run_email_url_profile_converts_email() -> None:
    result = ITNEngine().run("support at example dot com", profile=ITNProfile.EMAIL_URL)
    assert result.text == "support@example.com"
    assert "email_url" in result.rules_applied


def test_run_unchanged_text_sets_changed_false() -> None:
    result = ITNEngine().run("plain words", profile=ITNProfile.FULL)
    assert result.text == "plain words"
    assert result.changed is False
    assert result.rules_applied == []


def test_run_format_snapshot_reflects_policy() -> None:
    fmt = ITNFormatPolicy(date_style="iso", clock="24h")
    result = ITNEngine().run("hello", profile=ITNProfile.PROSE, format_policy=fmt)
    assert result.format_snapshot == {
        "date_style": "iso",
        "clock": "24h",
        "currency_decimal": "period",
    }


def test_run_format_policy_changes_rendering() -> None:
    fmt = ITNFormatPolicy(date_style="iso", clock="24h", currency_decimal="comma")
    result = ITNEngine().run(
        "five dollars and ten cents on march third nineteen ninety at three thirty pm",
        profile=ITNProfile.PROSE,
        format_policy=fmt,
    )
    assert result.text == "$5,10 on 1990-03-03 at 15:30"


def test_itn_result_to_dict_round_trip() -> None:
    result = ITNEngine().run("twenty five", profile=ITNProfile.PROSE)
    d = result.to_dict()
    assert d["text"] == "25"
    assert d["original_text"] == "twenty five"
    assert d["profile"] == "prose"
    assert d["rules_applied"] == ["numeric_words_to_digits"]
    assert d["changed"] is True
    assert d["format"] == ITNFormatPolicy.default().to_summary_dict()
    # to_dict copies — mutating the dict must not touch the result.
    d["rules_applied"].append("bogus")
    assert result.rules_applied == ["numeric_words_to_digits"]


# ---------------------------------------------------------------------- #
# ITNFormatPolicy.default / from_mapping
# ---------------------------------------------------------------------- #


def test_default_policy_is_legacy_us() -> None:
    pol = ITNFormatPolicy.default()
    assert pol.date_style == "us_medium"
    assert pol.clock == "12h"
    assert pol.currency_decimal == "period"


@pytest.mark.parametrize("data", [None, {}])
def test_from_mapping_empty_returns_default(data: dict | None) -> None:
    assert ITNFormatPolicy.from_mapping(data) == ITNFormatPolicy.default()


def test_from_mapping_normalizes_case_and_whitespace() -> None:
    pol = ITNFormatPolicy.from_mapping(
        {"date_style": " ISO ", "clock": "24H", "currency_decimal": "COMMA"}
    )
    assert pol == ITNFormatPolicy(date_style="iso", clock="24h", currency_decimal="comma")


def test_from_mapping_ignores_unknown_values_and_keys() -> None:
    pol = ITNFormatPolicy.from_mapping({"date_style": "weird", "clock": 7, "bogus": 1})
    assert pol == ITNFormatPolicy.default()


def test_from_mapping_partial_keys() -> None:
    pol = ITNFormatPolicy.from_mapping({"clock": "24h"})
    assert pol == ITNFormatPolicy(date_style="us_medium", clock="24h", currency_decimal="period")


# ---------------------------------------------------------------------- #
# ITNFormatPolicy.from_locale_identifier
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("locale_id", "date_style", "clock", "currency_decimal"),
    [
        ("en_US", "us_medium", "12h", "period"),
        ("en_GB", "dmy_long", "12h", "period"),
        ("EN-gb", "dmy_long", "12h", "period"),  # case / separator insensitive
        ("en-AU", "dmy_long", "12h", "period"),
        ("en_IN", "dmy_long", "12h", "period"),
        ("de-DE", "dmy_slash", "24h", "comma"),
        ("fr_FR", "dmy_slash", "24h", "comma"),
        ("pt", "dmy_slash", "24h", "comma"),  # language-only EU tag
        ("ja_JP", "iso", "24h", "period"),
        ("ja", "iso", "24h", "period"),
        ("xx_YY", "us_medium", "12h", "period"),  # unknown → legacy default
        (None, "us_medium", "12h", "period"),
        ("", "us_medium", "12h", "period"),
        ("   ", "us_medium", "12h", "period"),
    ],
)
def test_from_locale_identifier(
    locale_id: str | None, date_style: str, clock: str, currency_decimal: str
) -> None:
    pol = ITNFormatPolicy.from_locale_identifier(locale_id)
    assert pol.date_style == date_style
    assert pol.clock == clock
    assert pol.currency_decimal == currency_decimal


# ---------------------------------------------------------------------- #
# resolve_itn_format_policy
# ---------------------------------------------------------------------- #


def test_resolve_none_context_returns_default() -> None:
    assert resolve_itn_format_policy(None) == ITNFormatPolicy.default()


def test_resolve_dict_without_metadata_returns_default() -> None:
    assert resolve_itn_format_policy({}) == ITNFormatPolicy.default()
    assert resolve_itn_format_policy({"metadata": {}}) == ITNFormatPolicy.default()


def test_resolve_explicit_itn_format_wins_over_locale() -> None:
    pol = resolve_itn_format_policy(
        {"metadata": {"itn_format": {"date_style": "iso"}, "locale_identifier": "de_DE"}}
    )
    assert pol == ITNFormatPolicy(date_style="iso", clock="12h", currency_decimal="period")


def test_resolve_empty_itn_format_falls_back_to_locale() -> None:
    pol = resolve_itn_format_policy(
        {"metadata": {"itn_format": {}, "locale_identifier": "de_DE"}}
    )
    assert pol == ITNFormatPolicy(date_style="dmy_slash", clock="24h", currency_decimal="comma")


def test_resolve_locale_identifier_only() -> None:
    pol = resolve_itn_format_policy({"metadata": {"locale_identifier": "de_DE"}})
    assert pol == ITNFormatPolicy(date_style="dmy_slash", clock="24h", currency_decimal="comma")


def test_resolve_object_with_metadata_attribute() -> None:
    class Ctx:
        metadata = {"locale_identifier": "en_GB"}

    pol = resolve_itn_format_policy(Ctx())
    assert pol == ITNFormatPolicy(date_style="dmy_long", clock="12h", currency_decimal="period")


def test_resolve_blank_locale_identifier_returns_default() -> None:
    pol = resolve_itn_format_policy({"metadata": {"locale_identifier": "   "}})
    assert pol == ITNFormatPolicy.default()
