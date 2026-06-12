"""Deterministic retake application regressions.

Production 2026-06-10: self-correction cues were detected
(oneshot_self_correction_cues_detected fired with char offsets) but never
applied on the pass-through path, so the paste shipped literal
"… after the first. Scratch that after the last install. …".

The deterministic pass applies only unambiguous retakes; ambiguous or
literal uses of the marker words must be preserved verbatim.
"""
from __future__ import annotations

import pytest

from juno_core_v3.dictation.self_corrections import apply_unambiguous_retakes


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        # The shipped production utterance (U6). First marker is a real
        # retake; the second is commentary about the feature and must stay.
        (
            "look into all the utterances that I made after the first. "
            "Scratch that after the last install. Scratch that wasn't caught "
            "properly. so check everything",
            "look into all the utterances that I made after the last install. "
            "Scratch that wasn't caught properly. so check everything",
        ),
        # Canonical time fix.
        ("set an alarm for 3pm scratch that 4pm", "set an alarm for 4pm"),
        # Bare numeric replacement with clause comma.
        ("remind me at 6, no actually 7, to call mom", "remind me at 7, to call mom"),
        # Short same-opener retake.
        ("send it to Bob, no wait, to Alice", "send it to Alice"),
        # Titled value replacement.
        ("the title is Q3 Plan scratch that Q4 Plan", "the title is Q4 Plan"),
    ],
)
def test_unambiguous_retakes_are_applied(spoken: str, expected: str) -> None:
    out, applied = apply_unambiguous_retakes(spoken)
    assert out == expected
    assert applied, "expected at least one applied retake record"


@pytest.mark.parametrize(
    "literal",
    [
        # "delete that" / "remove that" / "scratch that" as content — the
        # old unwired _apply_mid_utterance_edits destroyed all of these.
        "I told him to delete that file from the repo",
        "the scratch that feature is broken",
        "we should remove that dependency before launch",
        # Marker with an after-phrase that does not re-speak anything.
        "Scratch that wasn't caught properly in the last build",
    ],
)
def test_ambiguous_or_literal_markers_are_preserved(literal: str) -> None:
    out, applied = apply_unambiguous_retakes(literal)
    assert out == literal
    assert applied == []


# --------------------------------------------------------------------------- #
# Same-slot retakes + ASR cue variants (2026-06-11 production matrix)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        # Weekday → weekday shares no characters but is the most common
        # spoken correction shape (production case 22 shipped "Friday Monday").
        (
            "I will send the update on Friday scratch that Monday with full details",
            "I will send the update on Monday with full details",
        ),
        # Date → date with article symmetry.
        ("meet on the 5th scratch that the 12th of June", "meet on the 12th of June"),
        # ASR renders "scratch that" as "scratched at" between two clocks
        # (production case 23 rejected the whole action).
        ("remind me at 3pm scratched at 4.15pm to call Sam", "remind me at 4.15pm to call Sam"),
    ],
)
def test_same_slot_and_scratched_at_retakes(spoken: str, expected: str) -> None:
    out, applied = apply_unambiguous_retakes(spoken)
    assert out == expected
    assert applied


@pytest.mark.parametrize(
    "literal",
    [
        "the paint scratched at the edge of the door",
        "he scratched at his beard while thinking",
    ],
)
def test_scratched_at_outside_temporal_context_stays_literal(literal: str) -> None:
    out, applied = apply_unambiguous_retakes(literal)
    assert out == literal
    assert applied == []


def test_multi_token_slot_retake_does_not_duplicate_tokens() -> None:
    # Regression (review F31): the slot picker preferred the SHORTEST
    # symmetric span, pairing "5th" with "June" (both slot "date") and
    # pasting "June June 12th". The longest symmetric pair must win.
    out, applied = apply_unambiguous_retakes("Move it to June 5th, no wait June 12th")
    assert out == "Move it to June 12th"
    assert len(applied) == 1

    out, applied = apply_unambiguous_retakes(
        "Move the meeting to June 5th scratch that June 12th please"
    )
    assert out == "Move the meeting to June 12th please"
    assert len(applied) == 1
