from __future__ import annotations

from juno_v2.transcript.contracts import TranscriptPatchOp
from juno_v2.transcript.patching import (
    diff_to_patch_ops,
    live_patch_is_safe,
    stable_prefix_chars,
    visible_text_hash,
)


def _op(
    op: str = "replace",
    start: int = 0,
    end: int = 0,
    text: str = "",
    reason: str = "asr_correction",
    confidence: float = 0.9,
) -> TranscriptPatchOp:
    return TranscriptPatchOp(
        op=op,  # type: ignore[arg-type]
        start_char=start,
        end_char=end,
        text=text,
        reason=reason,  # type: ignore[arg-type]
        confidence=confidence,
    )


def _apply_ops(base: str, ops: list[TranscriptPatchOp]) -> str:
    out = base
    for op in sorted(ops, key=lambda o: o.start_char, reverse=True):
        out = out[: op.start_char] + op.text + out[op.end_char :]
    return out


# ---------------------------------------------------------------------------
# visible_text_hash
# ---------------------------------------------------------------------------


def test_visible_text_hash_stable_under_whitespace_variation() -> None:
    base = visible_text_hash("hello world")
    assert visible_text_hash("hello   world") == base
    assert visible_text_hash("  hello world  ") == base
    assert visible_text_hash("hello\nworld") == base
    assert visible_text_hash("hello\t world\n") == base


def test_visible_text_hash_differs_for_different_text() -> None:
    assert visible_text_hash("hello world") != visible_text_hash("hello worlds")


def test_visible_text_hash_shape_and_empty_input() -> None:
    h = visible_text_hash("anything")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
    # Empty and whitespace-only normalize to the same digest.
    assert visible_text_hash("") == visible_text_hash("   \n\t ")


# ---------------------------------------------------------------------------
# stable_prefix_chars
# ---------------------------------------------------------------------------


def test_stable_prefix_chars_empty_and_blank() -> None:
    assert stable_prefix_chars("") == 0
    assert stable_prefix_chars("   \n  ") == 0


def test_stable_prefix_chars_short_text_is_fully_unstable() -> None:
    assert stable_prefix_chars("one two three") == 0
    # Exactly the unstable word count is still fully unstable.
    assert stable_prefix_chars("one two three four five") == 0


def test_stable_prefix_chars_cuts_before_last_n_words() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    # 6 words; the last 5 are unstable, so the cut is at the start of "beta".
    assert stable_prefix_chars(text) == 6


def test_stable_prefix_chars_respects_custom_unstable_count() -> None:
    text = "alpha beta gamma"
    # last_n_words_unstable=1 -> cut at start of "gamma".
    assert stable_prefix_chars(text, last_n_words_unstable=1) == text.index("gamma")


def test_stable_prefix_chars_prefers_nearby_sentence_boundary() -> None:
    text = "First part ends here. tail one two three four five"
    # Cut would land at "one"; the "." 7 chars earlier wins -> boundary + 1.
    assert stable_prefix_chars(text) == text.index(".") + 1


def test_stable_prefix_chars_ignores_distant_sentence_boundary() -> None:
    tail = "one two three four five"
    prefix = "End. " + ("filler " * 15)  # > 80 chars between "." and the cut
    text = prefix + tail
    assert stable_prefix_chars(text) == len(prefix)


# ---------------------------------------------------------------------------
# diff_to_patch_ops
# ---------------------------------------------------------------------------


def test_diff_identical_texts_yield_no_ops() -> None:
    assert diff_to_patch_ops(
        "same text here",
        "same text here",
        stable_prefix_chars=14,
        reason="asr_correction",
        confidence=0.9,
    ) == []


def test_diff_replace_op_offsets_and_payload() -> None:
    ops = diff_to_patch_ops(
        "the cat sat",
        "the cat sit",
        stable_prefix_chars=11,
        reason="asr_correction",
        confidence=0.9,
    )
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "replace"
    assert (op.start_char, op.end_char) == (9, 10)
    assert op.text == "i"
    assert op.source_text == "a"
    assert op.reason == "asr_correction"


def test_diff_insert_op_has_empty_base_span() -> None:
    ops = diff_to_patch_ops(
        "hello world",
        "hello brave world",
        stable_prefix_chars=100,
        reason="user_replacement",
        confidence=0.8,
    )
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "insert"
    assert op.start_char == op.end_char == 6
    assert op.text == "brave "
    assert op.source_text is None
    assert op.reason == "user_replacement"


def test_diff_delete_op_carries_source_text() -> None:
    ops = diff_to_patch_ops(
        "hello brave world",
        "hello world",
        stable_prefix_chars=100,
        reason="spacing",
        confidence=0.5,
    )
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "delete"
    assert (op.start_char, op.end_char) == (6, 12)
    assert op.text == ""
    assert op.source_text == "brave "


def test_diff_ops_round_trip_to_corrected_prefix() -> None:
    base = "she sold sea shells by the sea shore"
    corrected = "she sells seashells by the seashore"
    ops = diff_to_patch_ops(
        base, corrected, stable_prefix_chars=len(base), reason="asr_correction", confidence=0.9
    )
    assert ops
    assert _apply_ops(base, ops) == corrected


def test_diff_changes_past_stable_prefix_are_ignored() -> None:
    # Only the first 5 chars ("hello") are compared; the tails differ but no
    # ops are emitted for them.
    ops = diff_to_patch_ops(
        "hello world",
        "hello there friend",
        stable_prefix_chars=5,
        reason="asr_correction",
        confidence=0.9,
    )
    assert ops == []


def test_diff_unknown_reason_coerced_and_confidence_clamped() -> None:
    high = diff_to_patch_ops(
        "ab", "ax", stable_prefix_chars=2, reason="not_a_real_reason", confidence=5.0
    )
    assert high[0].reason == "asr_correction"
    assert high[0].confidence == 1.0
    low = diff_to_patch_ops("ab", "ax", stable_prefix_chars=2, reason="itn", confidence=-3.0)
    assert low[0].reason == "itn"
    assert low[0].confidence == 0.0


# ---------------------------------------------------------------------------
# live_patch_is_safe
# ---------------------------------------------------------------------------


def test_live_patch_empty_ops_is_safe() -> None:
    assert live_patch_is_safe("anything", [], stable_prefix_chars=8) == (True, "empty")


def test_live_patch_small_edit_in_stable_prefix_ok() -> None:
    base = "the quick brown fox jumps over the lazy dog and keeps running"
    ops = [_op("replace", 4, 9, "rapid")]
    ok, reason = live_patch_is_safe(base, ops, stable_prefix_chars=40)
    assert ok and reason == "ok"


def test_live_patch_rejects_op_past_stable_prefix() -> None:
    base = "the quick brown fox jumps over the lazy dog"
    ops = [_op("replace", 35, 39, "calm")]
    ok, reason = live_patch_is_safe(base, ops, stable_prefix_chars=10)
    assert not ok and reason == "touches_unstable_tail"


def test_live_patch_rejects_op_ending_past_stable_prefix() -> None:
    base = "the quick brown fox jumps over the lazy dog"
    ops = [_op("delete", 8, 20, "")]
    ok, reason = live_patch_is_safe(base, ops, stable_prefix_chars=10)
    assert not ok and reason == "touches_unstable_tail"


def test_live_patch_rejects_too_many_ops() -> None:
    base = "x" * 200
    ops = [_op("replace", i, i + 1, "y") for i in range(9)]
    ok, reason = live_patch_is_safe(base, ops, stable_prefix_chars=150)
    assert not ok and reason == "too_many_ops"


def test_live_patch_rejects_invalid_ranges() -> None:
    base = "x" * 100
    ok, reason = live_patch_is_safe(base, [_op("replace", -1, 2, "y")], stable_prefix_chars=50)
    assert not ok and reason == "invalid_range"
    ok, reason = live_patch_is_safe(base, [_op("replace", 10, 5, "y")], stable_prefix_chars=50)
    assert not ok and reason == "invalid_range"


def test_live_patch_rejects_oversized_delete() -> None:
    base = "x" * 300
    ops = [_op("delete", 0, 81, "")]
    ok, reason = live_patch_is_safe(base, ops, stable_prefix_chars=250)
    assert not ok and reason == "delete_too_large"


def test_live_patch_rejects_oversized_insert() -> None:
    base = "x" * 300
    ops = [_op("insert", 0, 0, "y" * 81)]
    ok, reason = live_patch_is_safe(base, ops, stable_prefix_chars=250)
    assert not ok and reason == "insert_too_large"


def test_live_patch_rejects_aggregate_change_over_budget() -> None:
    base = "x" * 120
    # stable=100 -> budget is 30% = 30 changed chars; two 20-char replaces = 40.
    ops = [_op("replace", 0, 20, "y" * 20), _op("replace", 30, 50, "y" * 20)]
    ok, reason = live_patch_is_safe(base, ops, stable_prefix_chars=100)
    assert not ok and reason == "changes_too_much"


def test_live_patch_small_stable_prefix_gets_16_char_floor() -> None:
    base = "x" * 60
    # stable=30 (<40) -> budget max(16, 9) = 16 chars.
    ok, reason = live_patch_is_safe(base, [_op("replace", 0, 16, "y" * 16)], stable_prefix_chars=30)
    assert ok and reason == "ok"
    ok, reason = live_patch_is_safe(base, [_op("replace", 0, 17, "y" * 17)], stable_prefix_chars=30)
    assert not ok and reason == "changes_too_much"
