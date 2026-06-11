from __future__ import annotations

import json

import pytest

from juno_core_v3.context.clipboard_ring import ClipboardEntry, ClipboardRingBuffer
from juno_core_v3.context.suppression_config import SuppressionConfig
from juno_core_v3.contracts.context_packet import (
    ContextFieldKey,
    ContextPacket,
    ContextPacketBudgets,
    FieldProvenance,
)

# ---- ClipboardRingBuffer ----------------------------------------------


def test_clipboard_recent_returns_newest_first() -> None:
    ring = ClipboardRingBuffer()
    ring.push("alpha", ts_unix_ms=1)
    ring.push("beta", ts_unix_ms=2)
    ring.push("gamma", ts_unix_ms=3)
    recent = ring.recent(limit=2)
    assert [e.text for e in recent] == ["gamma", "beta"]
    assert [e.ts_unix_ms for e in recent] == [3, 2]
    assert [e.text for e in ring.recent(limit=10)] == ["gamma", "beta", "alpha"]


def test_clipboard_dedups_consecutive_identical_pushes() -> None:
    ring = ClipboardRingBuffer()
    ring.push("same", ts_unix_ms=1)
    ring.push("same", ts_unix_ms=2)
    assert len(ring.recent(limit=10)) == 1
    # Non-consecutive duplicates are kept.
    ring.push("other", ts_unix_ms=3)
    ring.push("same", ts_unix_ms=4)
    assert [e.text for e in ring.recent(limit=10)] == ["same", "other", "same"]


@pytest.mark.parametrize(
    "secret",
    [
        "card 4111111111111111 exp 12/29",  # Visa 16-digit
        "mc 5500000000000004",  # Mastercard
        "ssn is 123-45-6789",
        "password: hunter2",
        "PASSWD : topsecret",
        "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----",
    ],
)
def test_clipboard_redacts_sensitive_content(secret: str) -> None:
    ring = ClipboardRingBuffer()
    ring.push(secret, ts_unix_ms=1)
    (entry,) = ring.recent(limit=10)
    assert entry.redacted is True
    assert entry.text == f"[REDACTED {len(secret)} chars]"
    assert secret not in entry.text


def test_clipboard_plain_text_not_redacted() -> None:
    ring = ClipboardRingBuffer()
    ring.push("just a normal sentence", ts_unix_ms=1)
    (entry,) = ring.recent(limit=10)
    assert entry.redacted is False
    assert entry.text == "just a normal sentence"


def test_clipboard_redacted_marker_passthrough() -> None:
    # Pushing a pre-redacted marker keeps it as-is, flagged redacted.
    ring = ClipboardRingBuffer()
    ring.push("[REDACTED 42 chars]", ts_unix_ms=1)
    (entry,) = ring.recent(limit=10)
    assert entry.redacted is True
    assert entry.text == "[REDACTED 42 chars]"


def test_clipboard_evicts_at_max_items() -> None:
    ring = ClipboardRingBuffer(max_items=3, max_chars=4000)
    for i in range(5):
        ring.push(f"item{i}", ts_unix_ms=i)
    items = ring.recent(limit=10)
    assert [e.text for e in items] == ["item4", "item3", "item2"]


def test_clipboard_evicts_at_max_chars() -> None:
    ring = ClipboardRingBuffer(max_items=20, max_chars=10)
    ring.push("aaaaaa", ts_unix_ms=1)  # 6 chars
    ring.push("bbbbbb", ts_unix_ms=2)  # total 12 > 10 -> evict oldest
    items = ring.recent(limit=10)
    assert [e.text for e in items] == ["bbbbbb"]


def test_clipboard_clear_resets_state_and_dedup() -> None:
    ring = ClipboardRingBuffer()
    ring.push("text", ts_unix_ms=1)
    ring.clear()
    assert ring.recent(limit=10) == []
    # After clear the same text can be pushed again (dedup state reset).
    ring.push("text", ts_unix_ms=2)
    assert [e.text for e in ring.recent(limit=10)] == ["text"]


def test_clipboard_entry_is_frozen() -> None:
    entry = ClipboardEntry(text="x", ts_unix_ms=1)
    with pytest.raises(Exception):
        entry.text = "y"  # type: ignore[misc]


# ---- SuppressionConfig -------------------------------------------------


def test_suppression_config_default_is_empty() -> None:
    cfg = SuppressionConfig.default()
    assert cfg.blocklist_bundle_ids == frozenset()
    assert cfg.warnlist_bundle_ids == frozenset()
    assert cfg.blocklist_window_title_patterns == ()


def test_suppression_config_load_valid_file(tmp_path) -> None:
    path = tmp_path / "suppression.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "blocklist_bundle_ids": ["com.Apple.Keychain", "com.1password.app"],
                "warnlist_bundle_ids": ["Com.Bank.App"],
                "blocklist_window_title_patterns": ["(?i)password", "secret"],
            }
        ),
        encoding="utf-8",
    )
    cfg = SuppressionConfig.load(path)
    # Bundle ids are lower-cased on load.
    assert cfg.blocklist_bundle_ids == frozenset({"com.apple.keychain", "com.1password.app"})
    assert cfg.warnlist_bundle_ids == frozenset({"com.bank.app"})
    assert len(cfg.blocklist_window_title_patterns) == 2
    assert cfg.blocklist_window_title_patterns[0].search("Enter PASSWORD here")
    assert cfg.blocklist_window_title_patterns[1].search("top secret window")


def test_suppression_config_load_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        SuppressionConfig.load(tmp_path / "nope.json")


def test_suppression_config_load_unsupported_version(tmp_path) -> None:
    path = tmp_path / "suppression.json"
    path.write_text(json.dumps({"version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="suppression_config_unsupported_version: 2"):
        SuppressionConfig.load(path)
    path.write_text(json.dumps({"blocklist_bundle_ids": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="suppression_config_unsupported_version: None"):
        SuppressionConfig.load(path)


def test_suppression_config_load_malformed_json(tmp_path) -> None:
    path = tmp_path / "suppression.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="suppression_config_malformed_json"):
        SuppressionConfig.load(path)
    path.write_text(json.dumps(["a", "list"]), encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        SuppressionConfig.load(path)


def test_suppression_config_load_bad_regex(tmp_path) -> None:
    path = tmp_path / "suppression.json"
    path.write_text(
        json.dumps({"version": 1, "blocklist_window_title_patterns": ["(unclosed"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="suppression_config_bad_regex"):
        SuppressionConfig.load(path)


def test_suppression_config_load_non_list_fields_rejected(tmp_path) -> None:
    path = tmp_path / "suppression.json"
    path.write_text(
        json.dumps({"version": 1, "blocklist_bundle_ids": "com.x"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="blocklist_bundle_ids must be a list"):
        SuppressionConfig.load(path)


# ---- ContextPacket.enforce_budgets --------------------------------------


def test_enforce_budgets_no_truncation_when_within_limits() -> None:
    pkt = ContextPacket(
        selected_text="short",
        clipboard_text="clip",
        focused_text_before="before",
        focused_text_after="after",
        field_text_excerpt="excerpt",
        provenance={ContextFieldKey.SELECTED_TEXT.value: FieldProvenance.TEST_FIXTURE},
    )
    out = pkt.enforce_budgets()
    assert out.selected_text == "short"
    assert out.truncation_applied == {}
    assert "budget_exceeded" not in out.metadata


def test_enforce_budgets_truncates_selected_and_clipboard_with_flags() -> None:
    budgets = ContextPacketBudgets(
        max_total_chars=1000,
        max_selected_chars=5,
        max_clipboard_chars=4,
        max_around_chars=100,
        max_field_excerpt_chars=3,
    )
    pkt = ContextPacket(
        selected_text="0123456789",
        clipboard_text="abcdefgh",
        field_text_excerpt="xyzw",
    )
    out = pkt.enforce_budgets(budgets)
    assert out.selected_text == "01234"
    assert out.clipboard_text == "abcd"
    assert out.field_text_excerpt == "xyz"
    assert out.truncation_applied[ContextFieldKey.SELECTED_TEXT.value] is True
    assert out.truncation_applied[ContextFieldKey.CLIPBOARD_TEXT.value] is True
    assert out.truncation_applied[ContextFieldKey.FIELD_TEXT_EXCERPT.value] is True
    # The original packet is untouched.
    assert pkt.selected_text == "0123456789"
    assert pkt.truncation_applied == {}


def test_enforce_budgets_around_chars_split_between_before_and_after() -> None:
    budgets = ContextPacketBudgets(max_around_chars=10)
    pkt = ContextPacket(
        focused_text_before="B" * 20,
        focused_text_after="A" * 4,
    )
    out = pkt.enforce_budgets(budgets)
    # Each side gets max_around_chars // 2.
    assert out.focused_text_before == "B" * 5
    assert out.focused_text_after == "A" * 4  # within half budget: untouched
    assert out.truncation_applied[ContextFieldKey.FOCUSED_TEXT_BEFORE.value] is True
    assert ContextFieldKey.FOCUSED_TEXT_AFTER.value not in out.truncation_applied


def test_enforce_budgets_preserves_provenance_and_metadata() -> None:
    prov = {
        ContextFieldKey.SELECTED_TEXT.value: FieldProvenance.DESKTOP_PROVIDER,
        ContextFieldKey.CLIPBOARD_TEXT.value: FieldProvenance.WORKBENCH_SYNC,
    }
    pkt = ContextPacket(
        selected_text="x" * 50,
        provenance=dict(prov),
        metadata={"origin": "test"},
        candidate_entities=["Juno"],
        app_name="Notes",
        window_title="My Note",
    )
    out = pkt.enforce_budgets(ContextPacketBudgets(max_selected_chars=10))
    assert out.provenance == prov
    assert out.metadata["origin"] == "test"
    assert out.candidate_entities == ["Juno"]
    assert out.app_name == "Notes"
    assert out.window_title == "My Note"


def test_enforce_budgets_total_overflow_clears_clipboard_and_trims_selected() -> None:
    budgets = ContextPacketBudgets(
        max_total_chars=40,
        max_selected_chars=100,
        max_clipboard_chars=100,
        max_around_chars=100,
        max_field_excerpt_chars=100,
    )
    pkt = ContextPacket(selected_text="s" * 30, clipboard_text="c" * 30)
    out = pkt.enforce_budgets(budgets)
    assert out.metadata["budget_exceeded"] is True
    assert out.metadata["budget_total_chars"] == 60
    assert out.clipboard_text == ""
    # Selected shrinks to max_total_chars // 4.
    assert out.selected_text == "s" * 10
