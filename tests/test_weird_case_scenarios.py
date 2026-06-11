"""Weird-case scenario suite — the use-case space Juno must hold.

Each case here encodes a verified production contract for the awkward
realities of dictation: announced list counts that don't match what was
spoken, many actions in one breath, wake words quoted mid-sentence,
filler retention, and app-profile routing. Deterministic layers only —
model-quality behaviors are exercised by the replay harness (see
docs/JUNO_TEST_DESIGN.md).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from juno_core_v3.dictation.pipeline import leading_wake_status
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.itn.engine import ITNEngine, ITNProfile
from juno_v2.turn_plan import actions_from_turn_plan, render_turn_plan, validate_turn_plan
from juno_v2.turn_plan.planner import fallback_structural_turn_plan
from juno_v2.writer.deterministic import strip_fillers


# --------------------------------------------------------------------------- #
# Structured dictation: announced count vs spoken count
# --------------------------------------------------------------------------- #


def test_announced_four_points_but_three_spoken_renders_three() -> None:
    # User claims four points and speaks three — render exactly what was
    # spoken, never invent the missing item.
    src = (
        "note down four points first ship the build second test the HUD "
        "third update the docs"
    )
    plan = fallback_structural_turn_plan(src)
    assert plan is not None
    render = plan["render_plan"]
    assert render["render_kind"] == "numbered_list"
    assert render["claimed_item_count"] == 4
    assert render["spoken_item_count"] == 3
    result = render_turn_plan(
        plan, context=TypedContextBundle(app_name="Notes", app_category="docs"), memory_store=None
    )
    assert result.rendered
    assert result.text == "1. ship the build\n2. test the HUD\n3. update the docs"


def test_announced_ten_points_with_three_spoken_renders_three() -> None:
    src = "note down ten points first alpha second beta third gamma"
    plan = fallback_structural_turn_plan(src)
    assert plan is not None
    result = render_turn_plan(
        plan, context=TypedContextBundle(app_name="Notes", app_category="docs"), memory_store=None
    )
    assert result.rendered
    assert result.text.splitlines() == ["1. alpha", "2. beta", "3. gamma"]


def test_unannounced_ordinals_are_not_structurally_forced() -> None:
    # Without an explicit structure request the deterministic fallback stays
    # out — short utterances reach the model planner for this judgment.
    assert fallback_structural_turn_plan(
        "first ship the build second test the HUD third update the docs"
    ) is None


# --------------------------------------------------------------------------- #
# Compound action batches (10–20 actions in one utterance)
# --------------------------------------------------------------------------- #


def test_fourteen_compound_actions_coerce_without_loss() -> None:
    bodies = [f"checkpoint {w}" for w in (
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
        "eta", "theta", "iota", "kappa", "lambda", "mu",
    )]
    source = (
        "take a note "
        + " and ".join(bodies)
        + " and remind me tomorrow at 9.15am to send the brief"
        + " and set an alarm for 4.30pm to publish the changelog"
    )
    actions = [
        {
            "kind": "note",
            "operation": "create",
            "body": body,
            "evidence_span": body,
            "schedule": {"kind": "none"},
            "missing_fields": [],
        }
        for body in bodies
    ]
    actions.append(
        {
            "kind": "reminder",
            "operation": "create",
            "body": "send the brief",
            "evidence_span": "remind me tomorrow at 9.15am to send the brief",
            "schedule": {"kind": "instant", "source_span": "tomorrow at 9.15am"},
            "missing_fields": [],
        }
    )
    actions.append(
        {
            "kind": "alarm",
            "operation": "create",
            "body": "publish the changelog",
            "evidence_span": "set an alarm for 4.30pm to publish the changelog",
            "schedule": {"kind": "instant", "source_span": "4.30pm"},
            "missing_fields": [],
        }
    )
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": actions,
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }
    validation = validate_turn_plan(plan, source_text=source, context=TypedContextBundle(app_name="Notes", app_category="docs"))
    assert validation.ok, validation.errors
    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 10, 12, 0))
    assert parsed.actions is not None
    assert len(parsed.actions) == 14
    kinds = [a.kind.value for a in parsed.actions]
    assert kinds.count("note") == 12
    assert kinds.count("reminder") == 1
    assert kinds.count("alarm") == 1
    assert parsed.actions[12].when is not None
    assert parsed.actions[13].when is not None


# --------------------------------------------------------------------------- #
# Wake-word placement
# --------------------------------------------------------------------------- #


def test_wake_word_quoted_mid_sentence_stays_dictation() -> None:
    text = "I was telling hey juno to do things and it broke"
    assert leading_wake_status(text, text).verified is False


def test_leading_wake_word_is_verified() -> None:
    text = "Hey Juno, remind me to stretch"
    assert leading_wake_status(text, text).verified is True


# --------------------------------------------------------------------------- #
# Filler policy — conservative by default
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        # The parenthetical ", you know," shape is unambiguous filler.
        ("I think, you know, we should ship it", "I think, we should ship it"),
        # Content usages and bare hesitations are preserved — stripping is
        # app/mode policy, not a default transcription behavior.
        ("you know the answer already", "you know the answer already"),
        ("um so the plan is simple", "um so the plan is simple"),
        ("the um plan is simple", "the um plan is simple"),
    ],
)
def test_filler_stripping_is_conservative(spoken: str, expected: str) -> None:
    assert strip_fillers(spoken) == expected


# --------------------------------------------------------------------------- #
# App-category → ITN profile routing (terminal exactness entry point)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("category", "profile"),
    [
        ("terminal", ITNProfile.TERMINAL),
        ("messaging", ITNProfile.PROSE),
        ("unknown", ITNProfile.PROSE),
        ("", ITNProfile.PROSE),
    ],
)
def test_itn_profile_routing(category: str, profile: ITNProfile) -> None:
    assert ITNEngine().profile_for_category(category) == profile
