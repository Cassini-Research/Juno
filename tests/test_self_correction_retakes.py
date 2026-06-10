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
