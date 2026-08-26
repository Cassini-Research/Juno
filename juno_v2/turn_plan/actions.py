from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from juno_core_v3.actions.contracts import Action, ActionKind, ActionOperation
from juno_core_v3.actions.timeparse import parse_when
from juno_v2.turn_plan.validators import canonical_action_kind, span_present


@dataclass(slots=True)
class TurnPlanActionsResult:
    actions: list[Action] | None = None
    rejected_reason: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    # Per-action coercion failures that did NOT reject the batch. A batch
    # with at least one valid action ships the valid ones and records the
    # rest here (production 2026-06-11: one unparseable alarm schedule threw
    # away five well-formed sibling actions).
    skipped_reasons: list[str] = field(default_factory=list)


def actions_from_turn_plan(
    plan: dict[str, Any] | None,
    *,
    source_text: str,
    now: datetime | None = None,
    max_actions: int = 25,
) -> TurnPlanActionsResult:
    if not isinstance(plan, dict):
        return TurnPlanActionsResult(rejected_reason="missing_plan")
    raw_actions = plan.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        return TurnPlanActionsResult(actions=None)
    safety = plan.get("safety") if isinstance(plan.get("safety"), dict) else {}
    execute_policy = str(safety.get("execute_policy") or "execute").strip()
    if execute_policy == "no_execute":
        return TurnPlanActionsResult(actions=None, rejected_reason="execute_policy_no_execute")

    out: list[Action] = []
    missing: list[str] = []
    skipped: list[str] = []
    for idx, raw in enumerate(raw_actions[:max_actions]):
        action, reason, missing_for_action = _coerce_action(
            raw,
            idx=idx,
            source_text=source_text,
            now=now,
            force_confirmation=execute_policy == "confirm",
        )
        if missing_for_action:
            missing.extend(missing_for_action)
        if action is not None:
            out.append(action)
            continue
        if reason and "unsupported_operation" in reason:
            # Operations on existing actions (complete / update / delete /
            # snooze) belong to the extractor lane; the pipeline routes the
            # whole utterance there off this batch-level rejection, so it
            # must stay all-or-nothing.
            return TurnPlanActionsResult(actions=None, rejected_reason=reason, missing_fields=missing)
        if reason:
            skipped.append(reason)
    if out:
        return TurnPlanActionsResult(actions=out, missing_fields=missing, skipped_reasons=skipped)
    if skipped:
        return TurnPlanActionsResult(
            actions=None,
            rejected_reason=skipped[0],
            missing_fields=missing,
            skipped_reasons=skipped,
        )
    return TurnPlanActionsResult(actions=None, rejected_reason="no_valid_actions", missing_fields=missing)


def _coerce_action(
    raw: Any,
    *,
    idx: int,
    source_text: str,
    now: datetime | None,
    force_confirmation: bool,
) -> tuple[Action | None, str | None, list[str]]:
    missing_fields: list[str] = []
    if not isinstance(raw, dict):
        return None, f"action_{idx}_not_object", missing_fields
    # Coerce here as well as in the planner's normalizer: a plan the
    # validator accepted must never be dropped for its kind spelling
    # (issue #76 - "create_note" plans were "ok" and un-shippable at once).
    canonical_kind = canonical_action_kind(raw.get("kind"))
    if canonical_kind is None:
        return None, f"action_{idx}_invalid_kind", missing_fields
    kind = ActionKind(canonical_kind)
    try:
        operation = ActionOperation(str(raw.get("operation") or "create"))
    except ValueError:
        return None, f"action_{idx}_invalid_operation", missing_fields
    if operation is not ActionOperation.CREATE:
        return None, f"action_{idx}_unsupported_operation:{operation.value}", missing_fields

    schedule = raw.get("schedule") if isinstance(raw.get("schedule"), dict) else {}
    schedule_kind = str(schedule.get("kind") or "none").strip()
    schedule_span = str(schedule.get("source_span") or raw.get("when_text") or "").strip()
    schedule_time = str(schedule.get("time") or "").strip()
    evidence = str(raw.get("evidence_span") or raw.get("raw_span") or "").strip()
    body = str(raw.get("body") or "").strip()
    if evidence and not span_present(evidence, source_text):
        return None, f"action_{idx}_evidence_not_grounded", missing_fields
    if not evidence and body and not span_present(body, source_text):
        # Without grounded evidence the body itself must be a source span,
        # or a hallucinated/rewritten body could ship into Notes/Reminders.
        # Mirrors the plan validator's per-action rule, enforced here so a
        # single ungrounded action skips instead of failing the plan.
        return None, f"action_{idx}_body_not_grounded", missing_fields
    body = _strip_action_invocation_prefix(kind, body, source_text=source_text)
    body = _latest_self_correction_tail(body)
    evidence = _latest_self_correction_tail(evidence)
    if kind in {ActionKind.REMINDER, ActionKind.ALARM}:
        body = _strip_subsequent_native_action_clause(body)
    if kind in {ActionKind.REMINDER, ActionKind.ALARM} and schedule_span:
        schedule_span = _resolve_corrected_schedule_span(
            schedule_span=schedule_span,
            evidence=evidence,
            body=body,
            source_text=source_text,
            now=now,
        )
        if (
            (not span_present(schedule_span, source_text) or not _schedule_span_looks_temporal(schedule_span))
            and schedule_time
            and span_present(schedule_time, source_text)
            and _schedule_span_looks_temporal(schedule_time)
        ):
            schedule_span = schedule_time
        elif not _schedule_span_looks_temporal(schedule_span):
            schedule_span = ""
        body = _strip_schedule_from_action_body(
            body,
            schedule_span=schedule_span,
            source_text=source_text,
        )
        body = _strip_leading_action_body_connector(body)
    if not evidence:
        evidence = body if body and span_present(body, source_text) else source_text
    if kind in {ActionKind.NOTE, ActionKind.REMINDER} and not body:
        missing_fields.append("body")
        return None, f"action_{idx}_missing_body", missing_fields
    if kind is ActionKind.ALARM and not body:
        body = "Alarm"

    missing_declared = raw.get("missing_fields")
    if isinstance(missing_declared, list):
        missing_fields.extend(str(item) for item in missing_declared if str(item).strip())

    when = None
    if kind in {ActionKind.REMINDER, ActionKind.ALARM}:
        if schedule_kind in {"instant", "vague", "series"} or schedule_span:
            if schedule_span and not span_present(schedule_span, source_text):
                return None, f"action_{idx}_schedule_not_grounded", missing_fields
            when = parse_when(schedule_span, now=now) if schedule_span else None
            if when is None:
                inferred_span = _infer_grounded_schedule_span(
                    source_text=source_text,
                    schedule_span=schedule_span,
                    evidence=evidence,
                    body=body,
                    now=now,
                )
                if inferred_span:
                    schedule_span = inferred_span
                    when = parse_when(schedule_span, now=now)
                    body = _strip_schedule_from_action_body(
                        body,
                        schedule_span=schedule_span,
                        source_text=source_text,
                    )
                    body = _strip_leading_action_body_connector(body)
            if when is None and (schedule_kind == "instant" or kind is ActionKind.ALARM):
                # An alarm without a parsed time can NEVER be created — the
                # shell's alarm sink hard-fails with "An alarm needs a time."
                # Shipping it timeless just converts a per-action skip into a
                # user-visible red error chip (production 2026-06-11: a
                # five-action utterance dispatched two timeless alarms whose
                # planner schedule kind was "vague"). Untimed reminders stay
                # legal — Reminders.app supports them.
                missing_fields.append("schedule")
                return None, f"action_{idx}_time_parse_failed", missing_fields
        elif kind is ActionKind.ALARM:
            missing_fields.append("schedule")
            return None, f"action_{idx}_missing_schedule", missing_fields

    if kind is ActionKind.ALARM and (not body or _action_body_is_schedule_only(body, schedule_span)):
        body = "Alarm"

    return (
        Action(
            kind=kind,
            body=body,
            raw_span=evidence,
            when=when,
            operation=operation,
            needs_confirmation=bool(
                force_confirmation
                or raw.get("needs_confirmation")
                or (when is not None and when.needs_confirmation)
            ),
        ),
        None,
        missing_fields,
    )


def _strip_action_invocation_prefix(kind: ActionKind, body: str, *, source_text: str) -> str:
    value = str(body or "").strip()
    if not value:
        return ""
    patterns: tuple[re.Pattern[str], ...]
    if kind is ActionKind.NOTE:
        patterns = (
            re.compile(
                r"^(?:juno\s+)?(?:take|create|make|add|save|write)\s+(?:a\s+)?note"
                r"(?:\s+(?:that|with|called|titled))?\s*[:.\\-]?\s+(?P<body>.+)$",
                re.IGNORECASE,
            ),
            re.compile(r"^(?:juno\s+)?note\s+down\s*[:.\\-]?\s+(?P<body>.+)$", re.IGNORECASE),
        )
    elif kind is ActionKind.REMINDER:
        patterns = (
            re.compile(
                r"^(?:juno\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder)"
                r"(?:\s+to)?\s*[:.\\-]?\s+(?P<body>.+)$",
                re.IGNORECASE,
            ),
        )
    elif kind is ActionKind.ALARM:
        patterns = (
            re.compile(
                r"^(?:juno\s+)?(?:set|create|add)\s+(?:an?\s+)?alarm"
                r"(?:\s+(?:for|at))?\s*[:.\\-]?\s+(?P<body>.+)$",
                re.IGNORECASE,
            ),
        )
    else:
        return value

    for pattern in patterns:
        match = pattern.match(value)
        if not match:
            continue
        candidate = match.group("body").strip()
        if candidate and span_present(candidate, source_text):
            return candidate
    return value


_CLOCK_TOKEN_RE = r"\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)"
_SCHEDULE_CANDIDATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b{_CLOCK_TOKEN_RE}\s+(?:day\s+after\s+tomorrow|tomorrow|tonight|today)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:day\s+after\s+tomorrow|tomorrow|tonight|today)(?:\s+(?:at\s+)?{_CLOCK_TOKEN_RE})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bin\s+\d+\s+(?:minute|min|hour|hr|day|week|month|year)s?\b", re.IGNORECASE),
    re.compile(r"\bin\s+(?:half|a\s+quarter(?:\s+of\s+an)?|an?)\s+(?:hour|hr)\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:this|next|coming|upcoming|on)\s+"
        rf"(?:mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)(?:day)?"
        rf"(?:\s+(?:at\s+)?{_CLOCK_TOKEN_RE})?"
        rf"(?:\s+(?:in\s+the\s+)?(?:morning|afternoon|evening|night))?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        rf"(?:\s+(?:at\s+)?{_CLOCK_TOKEN_RE})?"
        rf"(?:\s+(?:in\s+the\s+)?(?:morning|afternoon|evening|night))?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:on\s+(?:the\s+)?)?(?:"
        rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?"
        rf"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        rf"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|"
        rf"nov(?:ember)?|dec(?:ember)?)"
        rf"|"
        rf"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        rf"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|"
        rf"nov(?:ember)?|dec(?:ember)?)\s+\d{{1,2}}(?:st|nd|rd|th)?"
        rf")(?:,?\s+\d{{4}})?(?:\s+(?:at\s+)?{_CLOCK_TOKEN_RE})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"),
    re.compile(
        rf"\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?(?:\s+(?:at\s+)?{_CLOCK_TOKEN_RE})?\b",
        re.IGNORECASE,
    ),
    re.compile(rf"\b{_CLOCK_TOKEN_RE}\b", re.IGNORECASE),
)


def _strip_schedule_from_action_body(body: str, *, schedule_span: str, source_text: str = "") -> str:
    value = re.sub(r"\s+", " ", str(body or "").strip(" ,.;:-"))
    if not value:
        return ""
    span = re.sub(r"\s+", " ", str(schedule_span or "").strip(" ,.;:-"))
    if span:
        boundary_cleaned = _strip_schedule_write_boundary(
            value,
            schedule_span=span,
            source_text=source_text,
        )
        if boundary_cleaned != value:
            return boundary_cleaned
        escaped = re.escape(span)
        cleaned = re.sub(rf"^{escaped}\b\s*(?:to|for|about|that)?\s*", "", value, flags=re.IGNORECASE)
        if cleaned != value:
            return cleaned.strip(" ,.;:-")
        cleaned = re.sub(rf"\s+(?:at|by|for)?\s*{escaped}\b\s*$", "", value, flags=re.IGNORECASE)
        if cleaned != value:
            return cleaned.strip(" ,.;:-")
    cleaned = re.sub(
        rf"\s+(?:at|by|for)\s+{_CLOCK_TOKEN_RE}(?:\s+[A-Za-z])?\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    if cleaned != value:
        return cleaned.strip(" ,.;:-")
    cleaned = re.sub(
        rf"\s+{_CLOCK_TOKEN_RE}(?:\s+[A-Za-z])?\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ,.;:-")


def _strip_subsequent_native_action_clause(body: str) -> str:
    cleaned = re.sub(
        r"\s+(?:and|then|plus|next|,|;)\s+"
        r"(?:remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder|create\s+(?:a\s+)?reminder|"
        r"set\s+(?:an?\s+)?alarm|create\s+(?:an?\s+)?alarm|add\s+(?:an?\s+)?alarm|"
        r"take\s+(?:a\s+)?note|make\s+(?:a\s+)?note|create\s+(?:a\s+)?note|save\s+(?:a\s+)?note)\b.*$",
        "",
        str(body or "").strip(),
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    return _strip_trailing_action_connector(cleaned)


def _strip_leading_action_body_connector(body: str) -> str:
    return _strip_trailing_action_connector(
        re.sub(r"^(?:to|for|about|that)\b[\s:,-]*", "", str(body or "").strip(), flags=re.IGNORECASE)
    )


def _strip_trailing_action_connector(body: str) -> str:
    return re.sub(r"\b(?:and|then|plus|next)$", "", str(body or "").strip(), flags=re.IGNORECASE).strip(" ,.;:-")


# Deterministic native-action signal. The model planner may only create
# actions when the spoken text contains an explicit action verb/noun — a
# wake word alone is not consent to reroute dictation into Notes/Reminders.
# (Production 2026-06-10: "Hey Juno, I did not reinstall…" became a Note and
# the dictation was paste-suppressed.) Deliberately permissive on derived
# forms (reminder/reminders/alarms): false positives only re-enable the
# planner's own judgment; false negatives silently demote real commands.
_NATIVE_ACTION_SIGNAL_RE = re.compile(
    r"\bremind\w*"
    r"|\b(?:take|make|create|save|add|write|jot)\s+(?:down\s+)?(?:a\s+|another\s+|quick\s+|new\s+)?notes?\b"
    r"|\bnotes?\s+(?:that|this|down|to\s+self|titled|called)\b"
    r"|\balarm\w*"
    r"|\bwake\s+me\b"
    r"|\btimer\b",
    re.IGNORECASE,
)


def native_action_signal_present(text: str) -> bool:
    return bool(_NATIVE_ACTION_SIGNAL_RE.search(str(text or "")))


# Correction cues must be unambiguous before they may discard body text.
# Bare "actually" and bare "make it/that" are ordinary prose ("how we
# actually think", "remind me to make it scalable") — they only count as
# corrections when comma-adjacent ("at 3, actually 4") or followed by a
# schedule-like replacement ("make that 4.15pm"). Bare "no wait" must look
# like a retake ("no wait, 4pm"), not content ("there's no wait time").
_SELF_CORRECTION_CUE_RE = re.compile(
    r"\b(?:no\s+no\s+scratch\s+that|scratch\s+that|no,?\s+actually)\b"
    r"|\bno,?\s+wait\b(?=\s*[,.!?;:]|\s+(?:make|set|change)\b|\s+(?:at\s+)?\d)"
    r"|(?:(?<=,\s)|(?<=;\s))actually\b"
    r"|\bactually\b(?=\s*[,;])"
    r"|\bmake\s+(?:that|it)\b(?=\s+(?:about\s+|around\s+)?(?:\d|noon\b|midnight\b|tomorrow\b|today\b|tonight\b|next\b))",
    re.IGNORECASE,
)


def _latest_self_correction_tail(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip(" ,;:-"))
    if not value:
        return ""
    matches = list(_SELF_CORRECTION_CUE_RE.finditer(value))
    if not matches:
        return value
    tail = value[matches[-1].end() :].strip(" ,;:-")
    return tail or ""


def _action_body_is_schedule_only(body: str, schedule_span: str) -> bool:
    body_key = _term_key(body)
    if not body_key:
        return True
    if schedule_span and body_key == _term_key(schedule_span):
        return True
    return bool(re.fullmatch(_CLOCK_TOKEN_RE, str(body or "").strip(), flags=re.IGNORECASE))


def _schedule_span_looks_temporal(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(
        re.search(rf"\b{_CLOCK_TOKEN_RE}\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:today|tomorrow|tonight|morning|afternoon|evening|night)\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:next|this|coming|upcoming)\s+(?:week|month|year|mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)(?:day)?\b", text, flags=re.IGNORECASE)
        or re.search(r"\bin\s+(?:\d+|a|an|half)\s+(?:minute|min|hour|hr|day|week|month|year)s?\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)(?:day)?\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", text, flags=re.IGNORECASE)
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text)
    )


def _strip_schedule_write_boundary(body: str, *, schedule_span: str, source_text: str) -> str:
    value = re.sub(r"\s+", " ", str(body or "").strip(" ,.;:-"))
    span = re.sub(r"\s+", " ", str(schedule_span or "").strip(" ,.;:-"))
    if not value or not span:
        return value
    match = re.match(
        rf"^(?P<head>.+?)\s+(?:at|by|for)?\s*{re.escape(span)}\b(?P<tail>.*)$",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return value
    tail = re.sub(r"\s+", " ", match.group("tail") or "").strip(" ,.;:-")
    if not tail:
        return match.group("head").strip(" ,.;:-")
    if _is_source_write_boundary_tail(tail, schedule_span=span, source_text=source_text):
        return match.group("head").strip(" ,.;:-")
    return value


def _is_source_write_boundary_tail(tail: str, *, schedule_span: str, source_text: str) -> bool:
    tail_key = _term_key(tail)
    if not tail_key:
        return False
    source = str(source_text or "")
    if not source:
        return False
    for match in re.finditer(rf"\b{re.escape(schedule_span)}\b", source, flags=re.IGNORECASE):
        suffix = re.sub(r"\s+", " ", source[match.end() :]).strip(" ,.;:-")
        if not suffix:
            continue
        if not re.match(
            r"^(?:and|then|,)\s+(?:write|type|insert|dictate|paste)\b.*$",
            suffix,
            flags=re.IGNORECASE,
        ):
            continue
        suffix_key = _term_key(suffix)
        if suffix_key.startswith(tail_key) or tail_key.startswith(suffix_key):
            return True
    return False


def _term_key(value: str) -> str:
    value = str(value or "").casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _infer_grounded_schedule_span(
    *,
    source_text: str,
    schedule_span: str,
    evidence: str,
    body: str,
    now: datetime | None,
) -> str:
    for text in (schedule_span, evidence, body):
        value = str(text or "").strip()
        if not value:
            continue
        for candidate in _schedule_candidates(value):
            if not span_present(candidate, source_text):
                continue
            if parse_when(candidate, now=now) is not None:
                return candidate
    return ""


def _resolve_corrected_schedule_span(
    *,
    schedule_span: str,
    evidence: str,
    body: str,
    source_text: str,
    now: datetime | None,
) -> str:
    for text in (schedule_span, evidence, body):
        value = str(text or "").strip()
        if not value or not _SELF_CORRECTION_CUE_RE.search(value):
            continue
        tail = _latest_self_correction_tail(value)
        for candidate in _schedule_candidates(tail):
            if span_present(candidate, source_text) and parse_when(candidate, now=now) is not None:
                return candidate
    return schedule_span


def _schedule_candidates(text: str) -> list[str]:
    out: list[str] = []
    for pattern in _SCHEDULE_CANDIDATE_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(0).strip().rstrip(".,;:!?")
            if candidate and candidate not in out:
                out.append(candidate)
    return out
