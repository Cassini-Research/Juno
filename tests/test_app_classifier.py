"""Unit tests for juno_v2/context/app_classifier.py."""

from __future__ import annotations

import pytest

from juno_v2.context.app_classifier import (
    classify_app_category,
    iter_known_categories,
)


# --------------------------------------------------------------------- #
# Bundle-id matching (strongest signal)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("bundle_id", "expected"),
    [
        ("com.apple.terminal", "terminal"),
        ("com.googlecode.iterm2", "terminal"),
        ("com.microsoft.vscode", "code"),
        ("com.jetbrains.pycharm", "code"),  # prefix match
        ("com.apple.dt.xcode", "code"),
        ("com.tinyspeck.slackmacgap", "messaging"),
        ("com.apple.messages", "messaging"),
        ("com.apple.mail", "email"),
        ("com.microsoft.outlook", "email"),
        ("ai.grain.desktop", "meeting"),
        ("com.apple.notes", "docs"),
        ("md.obsidian", "docs"),
        ("org.gnu.emacs", "code"),
    ],
)
def test_bundle_id_rules(bundle_id: str, expected: str) -> None:
    assert classify_app_category(None, None, app_bundle_id=bundle_id) == expected


def test_bundle_id_is_case_insensitive_and_stripped() -> None:
    assert (
        classify_app_category(None, None, app_bundle_id="  COM.APPLE.TERMINAL  ")
        == "terminal"
    )


def test_bundle_id_wins_over_app_name_and_title() -> None:
    # Bundle id says email even though the name/title scream messaging.
    assert (
        classify_app_category(
            "Slack",
            "discord chat",
            app_bundle_id="com.apple.mail",
        )
        == "email"
    )


def test_unknown_bundle_id_falls_through_to_app_name() -> None:
    assert (
        classify_app_category(
            "Slack", None, app_bundle_id="com.example.mystery"
        )
        == "messaging"
    )


# --------------------------------------------------------------------- #
# App-name matching
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("app_name", "expected"),
    [
        ("Visual Studio Code", "code"),
        ("Xcode", "code"),
        ("IntelliJ IDEA", "code"),
        ("Zed", "code"),
        ("iTerm2", "terminal"),
        ("Kitty", "terminal"),
        ("Slack", "messaging"),
        ("WhatsApp", "messaging"),
        ("Telegram", "messaging"),
        ("Mail", "email"),
        ("Airmail", "email"),
        ("Thunderbird", "email"),
        ("Notes", "docs"),
        ("Obsidian", "docs"),
        ("Microsoft Word", "docs"),
        ("Grain", "meeting"),
        ("Safari", "unknown"),
        ("Google Chrome", "unknown"),
    ],
)
def test_app_name_rules(app_name: str, expected: str) -> None:
    assert classify_app_category(app_name, None) == expected


def test_app_name_matching_is_case_insensitive_substring() -> None:
    assert classify_app_category("SLACK — Acme workspace", None) == "messaging"
    assert classify_app_category("  visual studio code  ", None) == "code"


def test_app_name_wins_over_window_title() -> None:
    # A known app name short-circuits the title fallback.
    assert classify_app_category("Slack", "Inbox - Gmail") == "messaging"


# --------------------------------------------------------------------- #
# App-name matching is whole-word, not arbitrary substring
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "app_name",
    [
        # Contains "terminal" but is not a shell. Classifying it as one
        # switches the ITN profile to TERMINAL, makes context handling
        # ``no_touch`` and arms the terminal writer guards.
        "TerminalX",
        "TerminalXpress",
        # Same shape for other categories.
        "Mailtrap",
        "Wordle",
        "Sparkle",
    ],
)
def test_app_name_substring_does_not_match(app_name: str) -> None:
    assert classify_app_category(app_name, None) == "unknown"


@pytest.mark.parametrize(
    ("app_name", "expected"),
    [
        ("Terminal", "terminal"),
        ("Terminal.app", "terminal"),
        ("Apple Terminal", "terminal"),
        # Trailing digits stay part of the same word.
        ("iTerm2", "terminal"),
        ("Warp", "terminal"),
        ("Alacritty", "terminal"),
        ("Hyper", "terminal"),
        ("Microsoft Word", "docs"),
        ("Mail", "email"),
        ("Airmail", "email"),
    ],
)
def test_app_name_whole_word_still_matches(app_name: str, expected: str) -> None:
    assert classify_app_category(app_name, None) == expected


def test_bundle_id_still_decides_for_substring_names() -> None:
    # Bundle-id rules are checked first and are unaffected by the
    # whole-word name matching.
    assert (
        classify_app_category("TerminalX", None, app_bundle_id="com.apple.terminal")
        == "terminal"
    )
    assert (
        classify_app_category(
            "TerminalX", None, app_bundle_id="com.example.notaterminal"
        )
        == "unknown"
    )


def test_unmatched_name_falls_through_to_window_title() -> None:
    # "TerminalX" no longer short-circuits, so the title fallback runs.
    assert classify_app_category("TerminalX", "Inbox (4) - Gmail") == "email"


# --------------------------------------------------------------------- #
# Window-title fallback (browser-hosted surfaces)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("window_title", "expected"),
    [
        ("Inbox (4) - bob@gmail.com - Gmail", "email"),
        ("Q3 plan - Google Docs", "docs"),
        ("Cassini-Research/Juno: voice dictation - github.com", "code"),
        ("Acme | general | Slack", "messaging"),
        ("Customer survey form - forms.gle", "forms"),
        ("Sprint board - Jira", "forms"),
        ("Some random page title", "unknown"),
    ],
)
def test_window_title_fallback(window_title: str, expected: str) -> None:
    assert classify_app_category("Google Chrome", window_title) == expected


def test_window_title_alone_classifies() -> None:
    assert classify_app_category(None, "notion.so workspace") == "docs"


# --------------------------------------------------------------------- #
# Unknown / degenerate inputs
# --------------------------------------------------------------------- #


def test_all_empty_inputs_return_unknown() -> None:
    assert classify_app_category(None, None) == "unknown"
    assert classify_app_category("", "") == "unknown"
    assert classify_app_category("   ", "   ", app_bundle_id="   ") == "unknown"


def test_unrecognized_everything_returns_unknown() -> None:
    assert (
        classify_app_category(
            "Mystery App",
            "untitled window",
            app_bundle_id="com.example.mystery",
        )
        == "unknown"
    )


def test_unicode_inputs_never_raise() -> None:
    assert classify_app_category("日本語アプリ", "メモ – 無題") == "unknown"


# --------------------------------------------------------------------- #
# iter_known_categories
# --------------------------------------------------------------------- #


def test_iter_known_categories_closed_set() -> None:
    categories = list(iter_known_categories())
    assert set(categories) == {
        "messaging",
        "email",
        "docs",
        "code",
        "terminal",
        "forms",
        "meeting",
        "unknown",
    }
    assert len(categories) == len(set(categories))


def test_classifier_only_returns_known_categories() -> None:
    known = set(iter_known_categories())
    samples = [
        (None, None, None),
        ("Slack", None, None),
        (None, "gmail", None),
        (None, None, "com.apple.terminal"),
        ("Grain", None, None),
        ("Mystery", "Mystery", "mystery"),
    ]
    for name, title, bundle in samples:
        assert classify_app_category(name, title, app_bundle_id=bundle) in known
