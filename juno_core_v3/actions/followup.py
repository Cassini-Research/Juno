"""Narrow followup grammar for correcting the most recent Juno action."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from juno_core_v3.actions.contracts import (
    Action,
    ActionKind,
    ActionOperation,
    ParsedTime,
    Schedule,
    SeriesRule,
)
from juno_core_v3.actions.timeparse import parse_when

_WEEKDAYS = ("MO", "TU", "WE", "TH", "FR")


@dataclass(frozen=True, slots=True)
class FollowupIntent:
    kind: str  # "correct" | "confirm" | "cancel"
    field: str | None = None
    new_value: str | None = None


def parse_followup(text: str) -> FollowupIntent | None:
    """Parse only the tight correction window grammar.

    This intentionally does not classify general utterances. Callers should
    invoke it only after a recent action that asked for confirmation.
    """

    cleaned = _clean(text)
    if not cleaned:
        return None
    if cleaned in {"yes", "yeah", "yep", "correct", "looks good", "that's right", "that is right"}:
        return FollowupIntent(kind="confirm")
    if cleaned in {"cancel", "cancel that", "no cancel", "never mind", "nevermind", "no nevermind"}:
        return FollowupIntent(kind="cancel")
    if cleaned in {"undo that", "delete that"}:
        return FollowupIntent(kind="cancel")

    recurrence = _parse_recurrence(cleaned)
    if recurrence is not None:
        return FollowupIntent(kind="correct", field="recurrence", new_value=recurrence)

    time_value = _parse_time_correction(cleaned)
    if time_value is not None:
        return FollowupIntent(kind="correct", field="time", new_value=time_value)

    return None


def followup_action_for_row(
    intent: FollowupIntent,
    previous: dict[str, Any],
    *,
    now: datetime,
) -> Action | None:
    """Build a concrete operation against the previous actions-index row."""

    kind = _action_kind(previous.get("sink_kind"))
    if kind is None:
        return None
    juno_id = str(previous.get("juno_id") or "").strip()
    if not juno_id:
        return None
    sink_id = str(previous.get("sink_id") or "").strip() or None
    body = str(previous.get("body_normalized") or kind.value).strip() or kind.value
    target = {
        "ref_kind": "by_id",
        "id": juno_id,
        "resolved_via": "by_id",
        "confidence": 1.0,
    }

    if intent.kind == "cancel":
        return Action(
            kind=kind,
            body=body,
            raw_span="followup cancel",
            operation=ActionOperation.DELETE,
            target=target,
            juno_id=juno_id,
            sink_id=sink_id,
        )
    if intent.kind != "correct":
        return None
    if intent.field == "time" and intent.new_value:
        parsed = parse_when(intent.new_value, now=now)
        if parsed is None:
            return None
        return Action(
            kind=kind,
            body=body,
            raw_span=f"followup time {intent.new_value}",
            when=parsed,
            schedule=Schedule(kind="instant", instant=parsed),
            operation=ActionOperation.UPDATE,
            target=target,
            juno_id=juno_id,
            sink_id=sink_id,
        )
    if intent.field == "recurrence" and intent.new_value:
        anchor = str(previous.get("due_iso") or "").strip() or _aware_iso(now)
        rule = _recurrence_rule(intent.new_value, anchor)
        if rule is None:
            return None
        return Action(
            kind=kind,
            body=body,
            raw_span=f"followup recurrence {intent.new_value}",
            schedule=Schedule(kind="series", series=rule),
            operation=ActionOperation.UPDATE,
            target=target,
            juno_id=juno_id,
            sink_id=sink_id,
        )
    return None


def _parse_recurrence(cleaned: str) -> str | None:
    if cleaned in {"make it daily", "actually daily", "daily"}:
        return "daily"
    if cleaned in {"make it every day", "actually every day", "every day"}:
        return "daily"
    if cleaned in {"make it every weekday", "actually every weekday", "every weekday", "weekdays"}:
        return "weekday"
    return None


def _parse_time_correction(cleaned: str) -> str | None:
    match = re.search(r"\b(?:no\s+)?(?:i\s+)?meant\s+(.+?)(?:\s+not\s+.+)?$", cleaned)
    if match:
        return match.group(1).strip(" ,.")
    match = re.search(r"\b(?:change|move|set)\s+(?:it|that)?\s*(?:to|for)\s+(.+)$", cleaned)
    if match:
        return match.group(1).strip(" ,.")
    return None


def _recurrence_rule(value: str, anchor: str) -> SeriesRule | None:
    if value == "daily":
        return SeriesRule(freq="DAILY", interval=1, first_occurrence_iso=anchor)
    if value == "weekday":
        return SeriesRule(freq="WEEKLY", interval=1, by_day=_WEEKDAYS, first_occurrence_iso=anchor)
    return None


def _action_kind(value: Any) -> ActionKind | None:
    try:
        return ActionKind(str(value).strip().lower())
    except ValueError:
        return None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .!?")


def _aware_iso(now: datetime) -> str:
    if now.tzinfo is None:
        return ParsedTime(iso=now.isoformat(), source="default").iso
    return now.isoformat()


__all__ = ["FollowupIntent", "parse_followup", "followup_action_for_row"]
