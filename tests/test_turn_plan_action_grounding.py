"""Evidence-span grounding for turn-plan action bodies (issue #77).

The planner sees prior-utterance session context, and for a short follow-up it
would write an action ``body`` summarising what the user said *earlier* — text
the user never spoke this turn. The plan validator could not see that: it only
checked ``body`` when no ``evidence_span`` was supplied, and every per-action
problem was a warning, so such a plan validated ``ok`` and its body went on to
be created as a real Apple Note.

These cases are synthetic. Bodies are labelled by whether the user could have
spoken them: ``grounded`` bodies reuse the utterance's content words (however
re-worded), ``ungrounded`` bodies are written from context that is not in the
current utterance.
"""

from __future__ import annotations

from typing import Any

import pytest

from juno_v2.turn_plan import validate_turn_plan
from juno_v2.turn_plan.validators import (
    _ACTION_BODY_MIN_CONTENT_TOKENS,
    _ACTION_BODY_MIN_GROUNDED_RATIO,
    action_body_grounding_ratio,
)

# The issue's own repro: a six-word follow-up utterance.
CHART_SOURCE = "And add a chart as well."

# ...and the paragraph the planner fabricated from the preceding turns.
PRIOR_TURN_PARAGRAPH = (
    "The Q3 revenue review deck needs a full rewrite of the northeast pipeline "
    "section, including the churn analysis Priya sent on Monday and the renewal "
    "forecast that finance disputed. Pull the Salesforce export before the "
    "leadership sync so the regional breakdown lines up with the board summary."
)

NOTE_SOURCE = (
    "Take a note that the quarterly revenue numbers came in higher than we "
    "forecast for the northeast region and the churn rate dropped again."
)
REMINDER_SOURCE = "Remind me to pick up the dry cleaning on Thursday after the standup."


def _plan(
    *,
    source: str,
    body: str,
    evidence: str | None = None,
    kind: str = "note",
    schedule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "kind": kind,
        "operation": "create",
        "body": body,
        "schedule": schedule or {"kind": "none"},
    }
    if evidence is not None:
        action["evidence_span"] = evidence
    return {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source, "corrections": [], "literal_spans": []},
        "target": {"kind": "cursor", "confidence": 0.6},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {
            "operation": "none",
            "instruction": "",
            "transformed_text": None,
            "requires_second_pass": False,
        },
        "actions": [action],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }


def test_fabricated_body_fails_even_with_a_grounded_evidence_span() -> None:
    """The exact #77 shape: real span from this turn, body from the last one."""
    plan = _plan(source=CHART_SOURCE, body=PRIOR_TURN_PARAGRAPH, evidence="add a chart")

    validation = validate_turn_plan(plan, source_text=CHART_SOURCE)

    assert not validation.ok
    assert "action_0_body_ungrounded" in validation.errors


def test_fabricated_body_without_evidence_span_is_plan_fatal_not_a_warning() -> None:
    """Warning-only was the bug: the writer only abandons the lane on an error."""
    plan = _plan(source=CHART_SOURCE, body=PRIOR_TURN_PARAGRAPH)

    validation = validate_turn_plan(plan, source_text=CHART_SOURCE)

    assert not validation.ok
    assert "action_0_body_ungrounded" in validation.errors
    # The pre-existing literal-containment check still reports as a warning.
    assert "action_0_body_not_grounded" in validation.warnings


def test_only_the_ungrounded_action_is_plan_fatal() -> None:
    """Other per-action defects stay warnings so coercion can still salvage them."""
    plan = _plan(source=NOTE_SOURCE, body="quarterly revenue numbers", evidence="revenue numbers")
    plan["actions"][0]["operation"] = "sing"

    validation = validate_turn_plan(plan, source_text=NOTE_SOURCE)

    assert validation.ok
    assert "action_0_invalid_operation" in validation.warnings


GROUNDED_BODIES: list[tuple[str, str, str]] = [
    ("literal span", CHART_SOURCE, "add a chart"),
    ("paraphrase", CHART_SOURCE, "Prepare a chart"),
    ("paraphrase, capitalised", CHART_SOURCE, "Add a chart as well"),
    ("literal long span", NOTE_SOURCE, "the quarterly revenue numbers came in higher than we forecast for the northeast region"),
    (
        "reworded",
        NOTE_SOURCE,
        "Quarterly revenue numbers came in higher than forecast for the northeast region; churn rate dropped again",
    ),
    ("compressed", NOTE_SOURCE, "Northeast region: revenue above forecast, churn dropping"),
    ("reordered", REMINDER_SOURCE, "Pick up dry cleaning after the standup on Thursday"),
    ("imperative rewrite", "Note that the deploy pipeline is flaky again", "Deploy pipeline is flaky again"),
    (
        "inflected",
        "Remind me to email Priya about the renewal forecasts tomorrow",
        "Emailing Priya about the renewal forecast",
    ),
    ("bare title", "Set an alarm for 6:30 in the morning.", "Alarm"),
]

UNGROUNDED_BODIES: list[tuple[str, str, str]] = [
    ("prior-turn paragraph", CHART_SOURCE, PRIOR_TURN_PARAGRAPH),
    ("prior-turn sentence", CHART_SOURCE, "Pull the Salesforce export before the leadership sync"),
    ("prior-turn summary", REMINDER_SOURCE, "Summarise the churn analysis Priya sent on Monday"),
    ("unrelated task", "Set an alarm for 6:30 in the morning.", "Draft the board summary and send it to finance"),
    ("session recap", "Add a chart", "Meeting notes: attendees discussed hiring plans and budget"),
    (
        "invented scope",
        CHART_SOURCE,
        "Prepare a chart summarising the northeast renewal forecast",
    ),
]


@pytest.mark.parametrize(
    ("source", "body"),
    [(source, body) for _, source, body in GROUNDED_BODIES],
    ids=[label for label, _, _ in GROUNDED_BODIES],
)
def test_grounded_bodies_still_validate(source: str, body: str) -> None:
    validation = validate_turn_plan(_plan(source=source, body=body), source_text=source)

    assert validation.ok, validation.errors
    assert not [error for error in validation.errors if "body_ungrounded" in error]


@pytest.mark.parametrize(
    ("source", "body"),
    [(source, body) for _, source, body in UNGROUNDED_BODIES],
    ids=[label for label, _, _ in UNGROUNDED_BODIES],
)
def test_ungrounded_bodies_fail(source: str, body: str) -> None:
    validation = validate_turn_plan(
        _plan(source=source, body=body, evidence=None), source_text=source
    )

    assert not validation.ok
    assert "action_0_body_ungrounded" in validation.errors


def test_grounding_threshold_sits_in_an_empty_band() -> None:
    """The threshold is calibrated, not guessed: nothing in the corpus is near it."""
    judged = [
        (label, action_body_grounding_ratio(body, source))
        for label, source, body in GROUNDED_BODIES + UNGROUNDED_BODIES
        if len(_content_tokens(body)) >= _ACTION_BODY_MIN_CONTENT_TOKENS
    ]
    assert judged
    for label, ratio in judged:
        assert abs(ratio - _ACTION_BODY_MIN_GROUNDED_RATIO) > 0.15, f"{label} sits on the threshold"


def test_short_bodies_are_exempt_from_the_ratio() -> None:
    """Below the content-word floor there is nothing to measure, so no failure."""
    source = "Wake me up at 6:30 in the morning."

    # "Alarm" appears nowhere in the utterance, yet the plan still validates:
    # a one-word body carries too little signal to call fabricated.
    assert action_body_grounding_ratio("Alarm", source) == 0.0
    assert validate_turn_plan(
        _plan(
            source=source,
            body="Alarm",
            kind="alarm",
            schedule={"kind": "instant", "source_span": "6:30 in the morning"},
        ),
        source_text=source,
    ).ok


def test_grounding_ratio_ignores_stopwords_and_repetition() -> None:
    source = "Take a note about the renewal forecast"
    # Function words are free; only "renewal"/"forecast" carry grounding.
    assert action_body_grounding_ratio("The renewal forecast is in the forecast", source) == 1.0
    assert action_body_grounding_ratio("The churn analysis is in the deck", source) == 0.0


def _content_tokens(text: str) -> list[str]:
    from juno_v2.turn_plan.validators import _content_tokens as impl

    return impl(text)
