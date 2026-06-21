"""Unit tests for juno_v2/context/redaction.py and
juno_v2/context/frozen_merge.py."""

from __future__ import annotations

import pytest

from juno_v2.context.frozen_merge import merge_frozen_capability_into_bundle
from juno_v2.context.redaction import ContextRedactor
from juno_v2.contracts.context import TypedContextBundle


# --------------------------------------------------------------------- #
# ContextRedactor.redact
# --------------------------------------------------------------------- #


def test_redact_empty_and_none_safe() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact("")
    assert text == ""
    assert summary.to_dict() == {
        "emails": 0,
        "urls": 0,
        "digit_sequences": 0,
        "secrets": 0,
    }
    text, summary = redactor.redact(None)  # type: ignore[arg-type]
    assert text == ""


def test_redact_emails() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact("write to bob@example.com or a.b+c@sub.example.co.uk")
    assert text == "write to <email> or <email>"
    assert summary.emails == 2
    assert summary.urls == 0


def test_redact_urls() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact(
        "see https://example.com/path?q=1 and HTTP://CAPS.example.org plus www.foo.dev"
    )
    assert text == "see <url> and <url> plus <url>"
    assert summary.urls == 3


def test_redact_secrets() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact("the password: hunter2 was leaked")
    assert text == "the <secret> was leaked"
    assert summary.secrets == 1

    text, summary = redactor.redact("OTP 994321 and pin=0000")
    assert "<secret>" in text
    assert summary.secrets == 2
    # The digits were consumed by the secret rule, not double counted.
    assert summary.digit_sequences == 0


def test_redact_digit_runs() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact("card 4111111111111111 expires soon")
    assert text == "card <digits> expires soon"
    assert summary.digit_sequences == 1


def test_redact_short_digit_runs_survive() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact("room 123 on floor 9")
    assert text == "room 123 on floor 9"
    assert summary.digit_sequences == 0


def test_redact_email_with_digits_counts_once_as_email() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact("mail a1234567@example.com")
    assert text == "mail <email>"
    assert summary.emails == 1
    assert summary.digit_sequences == 0


def test_redact_secret_keywords_require_word_boundary() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact("please ping me when the secretary arrives")
    assert text == "please ping me when the secretary arrives"
    assert summary.secrets == 0


def test_redact_summary_accumulates_all_categories() -> None:
    redactor = ContextRedactor()
    text, summary = redactor.redact(
        "email bob@example.com, visit https://x.io, secret: abc, code 123456"
    )
    assert summary.emails == 1
    assert summary.urls == 1
    assert summary.secrets == 1
    assert summary.digit_sequences == 1
    assert "<email>" in text
    assert "<url>" in text
    assert "<secret>" in text
    assert "<digits>" in text


def test_redact_plain_text_untouched() -> None:
    redactor = ContextRedactor()
    original = "Just a normal sentence about café plans."
    text, summary = redactor.redact(original)
    assert text == original
    assert summary.to_dict() == {
        "emails": 0,
        "urls": 0,
        "digit_sequences": 0,
        "secrets": 0,
    }


# --------------------------------------------------------------------- #
# merge_frozen_capability_into_bundle
# --------------------------------------------------------------------- #


def test_merge_none_returns_false_and_leaves_bundle_alone() -> None:
    ctx = TypedContextBundle(selected_text="keep me")
    assert merge_frozen_capability_into_bundle(ctx, None) is False
    assert ctx.selected_text == "keep me"


def test_merge_empty_dict_applies_nothing() -> None:
    ctx = TypedContextBundle()
    assert merge_frozen_capability_into_bundle(ctx, {}) is False
    assert ctx.app_category is None


def test_merge_text_fields_win_and_are_redacted() -> None:
    ctx = TypedContextBundle(selected_text="stale server-side value")
    frozen = {
        "selected_text": "contact bob@example.com now",
        "focused_text_before": "see https://example.com/page",
        "focused_text_after": "pin: 9911 done",
        "clipboard_text": "plain clipboard",
    }
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    assert ctx.selected_text == "contact <email> now"
    assert ctx.focused_text_before == "see <url>"
    assert ctx.focused_text_after == "<secret> done"
    assert ctx.clipboard_text == "plain clipboard"


def test_merge_explicit_empty_field_overrides() -> None:
    # Key present with None value still counts as "frozen sent this field".
    ctx = TypedContextBundle(selected_text="stale")
    assert merge_frozen_capability_into_bundle(ctx, {"selected_text": None}) is True
    assert ctx.selected_text == ""


def test_merge_surrounding_text_aliases() -> None:
    ctx = TypedContextBundle()
    frozen = {
        "surrounding_text_before": "before text",
        "surrounding_text_after": "after text",
    }
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    assert ctx.focused_text_before == "before text"
    assert ctx.focused_text_after == "after text"


def test_merge_clips_to_max_field_chars() -> None:
    ctx = TypedContextBundle()
    frozen = {"selected_text": "a" * 1000}
    assert merge_frozen_capability_into_bundle(ctx, frozen, max_field_chars=240)
    assert len(ctx.selected_text) == 240


def test_merge_app_identity_and_classification() -> None:
    ctx = TypedContextBundle()
    frozen = {
        "frontmost_app_bundle_id": "com.tinyspeck.slackmacgap",
        "frontmost_app_name": "Slack",
        "window_title": "general — Acme",
    }
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    assert ctx.app_name == "Slack"
    assert ctx.window_title == "general — Acme"
    assert ctx.metadata["app_bundle_id"] == "com.tinyspeck.slackmacgap"
    assert ctx.app_category == "messaging"


def test_merge_app_name_alias_keys() -> None:
    ctx = TypedContextBundle()
    frozen = {"app_name": "Mail", "app_bundle_id": "com.apple.mail"}
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    assert ctx.app_name == "Mail"
    assert ctx.metadata["app_bundle_id"] == "com.apple.mail"
    assert ctx.app_category == "email"


def test_merge_secure_flag_and_locale_and_document_path() -> None:
    ctx = TypedContextBundle()
    frozen = {
        "focused_is_secure": True,
        "locale_identifier": "en_GB",
        "focused_document_path": "/Users/sam/notes.md",
    }
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    assert ctx.metadata["focused_secure"] is True
    assert ctx.metadata["locale_identifier"] == "en_GB"
    assert ctx.focused_file_path == "/Users/sam/notes.md"


def test_merge_candidate_entities_dedup_casefold() -> None:
    ctx = TypedContextBundle(candidate_entities=["Alpha"])
    frozen = {"candidate_entities": ["alpha", "Beta", "", "Beta", "  Gamma  "]}
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    assert ctx.candidate_entities == ["Alpha", "Beta", "Gamma"]
    # Harvested candidates are NOT explicit entities: the shell sends
    # user-named terms under their own "explicit_candidate_entities" key
    # and only those reach the explicit repair gates.
    assert "explicit_candidate_entities" not in ctx.metadata


def test_merge_explicit_candidate_entities_tracked_separately() -> None:
    ctx = TypedContextBundle()
    frozen = {
        "candidate_entities": ["Beta"],
        "explicit_candidate_entities": ["alpha", "", "Alpha", "  Gamma  "],
    }
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    assert ctx.candidate_entities == ["Beta"]
    assert ctx.metadata["explicit_candidate_entities"] == ["alpha", "Gamma"]


def test_merge_candidate_entities_capped() -> None:
    ctx = TypedContextBundle()
    frozen = {
        "candidate_entities": [
            f"Entity{chr(97 + (i // 26))}{chr(97 + (i % 26))}" for i in range(60)
        ]
    }
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    # Only the first 24 raw entries are even considered.
    assert len(ctx.candidate_entities) == 24


def test_merge_no_classification_without_app_identity() -> None:
    ctx = TypedContextBundle()
    frozen = {"selected_text": "hello"}
    assert merge_frozen_capability_into_bundle(ctx, frozen) is True
    assert ctx.app_category is None
