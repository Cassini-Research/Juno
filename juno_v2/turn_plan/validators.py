from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.memory.entity_policy import common_english_single_word


@dataclass(slots=True)
class PlanValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_UTTERANCE_KINDS = {
    "dictation",
    "format_dictation",
    "transform",
    "actions",
    "mixed",
    "command",
    "memory_mutation",
    "no_op",
    "ambiguous",
}
ACTION_KINDS = ("note", "reminder", "alarm")

_ACTION_KIND_ALIASES = {
    # Note synonyms.
    "notes": "note",
    "memo": "note",
    "note_create": "note",
    "note_action": "note",
    "create_note": "note",
    "new_note": "note",
    "add_note": "note",
    "make_note": "note",
    "save_note": "note",
    "take_note": "note",
    "write_note": "note",
    "apple_note": "note",
    "apple_notes": "note",
    # Reminder synonyms. Juno has no separate task/todo sink; both are reminders.
    "reminders": "reminder",
    "reminder_create": "reminder",
    "reminder_action": "reminder",
    "create_reminder": "reminder",
    "new_reminder": "reminder",
    "add_reminder": "reminder",
    "set_reminder": "reminder",
    "make_reminder": "reminder",
    "apple_reminder": "reminder",
    "apple_reminders": "reminder",
    "task": "reminder",
    "tasks": "reminder",
    "todo": "reminder",
    "todos": "reminder",
    "to_do": "reminder",
    "to_dos": "reminder",
    # Alarm synonyms.
    "alarms": "alarm",
    "alarm_create": "alarm",
    "alarm_action": "alarm",
    "create_alarm": "alarm",
    "new_alarm": "alarm",
    "add_alarm": "alarm",
    "set_alarm": "alarm",
    "make_alarm": "alarm",
    "wake_up_alarm": "alarm",
}


def canonical_action_kind(value: Any) -> str | None:
    """Canonical ``ActionKind`` value for a model-emitted action kind.

    The planner prompt lists three kinds, but the model routinely emits
    verb-prefixed, plural, camelCase, or spaced variants ("create_note",
    "Reminders", "createAlarm"). Those used to survive validation as a
    warning and were then dropped by ``actions_from_turn_plan`` after the
    full decode had been paid for; every such plan was "ok" and
    un-shippable at once. Coerce what is unambiguous here and let callers
    reject the rest outright so the extractor fallback runs immediately.
    """
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or ""))
    key = re.sub(r"[^a-z0-9]+", "_", raw.casefold()).strip("_")
    if not key:
        return None
    if key in ACTION_KINDS:
        return key
    return _ACTION_KIND_ALIASES.get(key)


_RENDER_KINDS = {
    "plain",
    "paragraphs",
    "message",
    "email",
    "note",
    "bulleted_list",
    "numbered_list",
    "checklist",
    "table",
    "ai_prompt",
    "code",
    "terminal",
    "none",
}
def validate_turn_plan(
    plan: dict[str, Any] | None,
    *,
    source_text: str,
    context: TypedContextBundle | None = None,
    max_actions: int = 25,
) -> PlanValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(plan, dict):
        return PlanValidation(ok=False, errors=["plan_not_object"])
    if str(plan.get("schema_version") or "") != "turn_plan_v1":
        errors.append("invalid_schema_version")
    kind = str(plan.get("utterance_kind") or "").strip()
    if kind not in _UTTERANCE_KINDS:
        errors.append("invalid_utterance_kind")
    corrected = plan.get("corrected_transcript")
    corrected_text = _corrected_text(plan)
    if not isinstance(corrected, dict):
        errors.append("missing_corrected_transcript")
    elif not corrected_text and kind not in {"actions", "no_op", "ambiguous"}:
        errors.append("empty_corrected_text")

    render = plan.get("render_plan") if isinstance(plan.get("render_plan"), dict) else {}
    render_kind = str(render.get("render_kind") or "plain").strip()
    if render_kind not in _RENDER_KINDS:
        errors.append("invalid_render_kind")
    units = render.get("content_units") if isinstance(render, dict) else None
    if units is not None and not isinstance(units, list):
        errors.append("content_units_not_array")
    if isinstance(units, list):
        for idx, unit in enumerate(units[:80]):
            if not isinstance(unit, dict):
                errors.append(f"content_unit_{idx}_not_object")
                continue
            text = str(unit.get("text") or "").strip()
            if not text:
                warnings.append(f"content_unit_{idx}_empty")
            source_span = str(unit.get("source_span") or "").strip()
            grounded = span_present(source_span, source_text) or span_present(text, source_text)
            if source_span and not span_present(source_span, source_text) and text and not span_present(text, source_text):
                errors.append(f"content_unit_{idx}_ungrounded")
            elif _render_requires_grounding(kind, render_kind) and text and not grounded:
                errors.append(f"content_unit_{idx}_ungrounded")

    if bool(render.get("markdown_allowed") is False):
        for unit in units if isinstance(units, list) else []:
            text = str(unit.get("text") or "") if isinstance(unit, dict) else ""
            if _markdown_artifact_present(text):
                errors.append("markdown_in_non_markdown_plan")
                break

    actions = plan.get("actions")
    if actions is None:
        actions = []
    if not isinstance(actions, list):
        errors.append("actions_not_array")
    elif len(actions) > max_actions:
        errors.append("too_many_actions")
    elif actions:
        for idx, action in enumerate(actions):
            # Per-action problems are warnings, not plan-fatal errors:
            # actions_from_turn_plan re-checks every one of these and skips
            # the offending action while shipping valid siblings. Failing
            # the whole plan here forced a slow model repair pass over a
            # plan that coercion could already salvage (production
            # 2026-06-11: one ungrounded note body rejected a five-action
            # utterance after the repair decode returned garbage).
            # An unrecognized action kind is the exception: nothing
            # downstream can build an Action from it, so warning here only
            # bought an "ok" plan that was discarded later.
            action_errors, action_warnings = _validate_action_dict(action, idx=idx, source_text=source_text)
            errors.extend(action_errors)
            warnings.extend(action_warnings)

    transform = plan.get("transform") if isinstance(plan.get("transform"), dict) else {}
    if transform:
        transformed = transform.get("transformed_text")
        if transformed is not None and not isinstance(transformed, str):
            errors.append("transformed_text_not_string")
        target = plan.get("target") if isinstance(plan.get("target"), dict) else {}
        if str(transform.get("operation") or "none") != "none" and str(target.get("kind") or "none") == "none":
            warnings.append("transform_without_target")

    memory_candidates = plan.get("memory_candidates") or []
    if isinstance(memory_candidates, list):
        for idx, item in enumerate(memory_candidates[:40]):
            if not isinstance(item, dict):
                continue
            surface = str(item.get("surface") or item.get("canonical") or "").strip()
            if _memory_candidate_is_common_phrase(surface):
                warnings.append(f"memory_candidate_{idx}_common_word")

    if context is not None and bool((context.metadata or {}).get("focused_is_secure")):
        if memory_candidates:
            errors.append("memory_candidates_in_secure_field")

    if kind in {"dictation", "format_dictation", "mixed"}:
        new_numbers = _new_numbers_without_evidence(source_text, _numeric_validation_text(plan, fallback=corrected_text))
        if new_numbers:
            errors.append("new_number_without_evidence:" + ",".join(sorted(new_numbers)[:4]))

    safety = plan.get("safety") if isinstance(plan.get("safety"), dict) else {}
    commit_policy = str(safety.get("commit_policy") or "commit").strip()
    execute_policy = str(safety.get("execute_policy") or "no_execute").strip()
    if commit_policy not in {"commit", "no_commit", "confirm"}:
        errors.append("invalid_commit_policy")
    if execute_policy not in {"execute", "no_execute", "confirm"}:
        errors.append("invalid_execute_policy")
    if actions and execute_policy == "no_execute":
        errors.append("actions_with_no_execute_policy")

    return PlanValidation(ok=not errors, errors=errors, warnings=warnings)


def _memory_candidate_is_common_phrase(value: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", value or "")
    return bool(tokens) and all(common_english_single_word(token) for token in tokens)


def _validate_action_dict(action: Any, *, idx: int, source_text: str) -> tuple[list[str], list[str]]:
    """Return ``(plan_fatal_errors, per_action_warnings)`` for one action."""
    warnings: list[str] = []
    if not isinstance(action, dict):
        return [], [f"action_{idx}_not_object"]
    kind = canonical_action_kind(action.get("kind"))
    if kind is None:
        return [f"action_{idx}_invalid_kind"], warnings
    operation = str(action.get("operation") or "create").strip()
    if operation not in {"create", "update", "delete", "complete", "query", "append_to", "remove_from"}:
        warnings.append(f"action_{idx}_invalid_operation")
    evidence = str(action.get("evidence_span") or "").strip()
    body = str(action.get("body") or "").strip()
    if evidence and not span_present(evidence, source_text):
        warnings.append(f"action_{idx}_evidence_not_grounded")
    if not evidence and body and not span_present(body, source_text):
        warnings.append(f"action_{idx}_body_not_grounded")
    if operation == "create" and kind in {"note", "reminder"} and not body:
        warnings.append(f"action_{idx}_missing_body")
    schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
    schedule_kind = str(schedule.get("kind") or "none").strip()
    if kind in {"reminder", "alarm"} and operation == "create":
        source_span = str(schedule.get("source_span") or "").strip()
        if schedule_kind not in {"none", ""} and source_span and not span_present(source_span, source_text):
            warnings.append(f"action_{idx}_schedule_not_grounded")
        if kind == "alarm" and schedule_kind in {"none", ""}:
            warnings.append(f"action_{idx}_alarm_missing_schedule")
    return [], warnings


def span_present(span: Any, source_text: str) -> bool:
    """Token-boundary containment of ``span`` in ``source_text``.

    Containment must align to word boundaries: the model re-types spans, and a
    character-level substring check grounds misaligned copies — production
    shipped an Apple Note whose body began "e titled, …" because
    "e titled what …" is a raw substring of "… note titled what …".
    """
    needle = _span_key(span)
    haystack = _span_key(source_text)
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


def _span_key(text: Any) -> str:
    s = str(text or "").casefold().strip()
    s = s.strip(" \t\r\n\"'`.,!?;:()[]{}<>")
    # Meridiems must compare equal across ASR/model spellings: Whisper emits
    # "11 p.m." while the planner re-types spans as "11 PM", and bare
    # punctuation stripping below would key them as "p m" vs "pm"
    # (production 2026-06-11: every grounded time in an alarm batch failed).
    # The dot requirement keeps ordinary "…a m…" word sequences untouched.
    s = re.sub(r"\b([ap])\s*\.\s*m\s*\.?(?=$|[\s\W])", r"\1m", s)
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def _corrected_text(plan: dict[str, Any]) -> str:
    corrected = plan.get("corrected_transcript")
    if isinstance(corrected, dict):
        return str(corrected.get("text") or "").strip()
    return ""


def _numeric_validation_text(plan: dict[str, Any], *, fallback: str) -> str:
    render = plan.get("render_plan") if isinstance(plan.get("render_plan"), dict) else {}
    units = render.get("content_units") if isinstance(render, dict) else None
    if isinstance(units, list):
        parts = [
            str(unit.get("text") or "").strip()
            for unit in units
            if isinstance(unit, dict) and str(unit.get("text") or "").strip()
        ]
        if parts:
            return " ".join(parts)
    transform = plan.get("transform") if isinstance(plan.get("transform"), dict) else {}
    transformed = transform.get("transformed_text")
    if isinstance(transformed, str) and transformed.strip():
        return transformed.strip()
    return fallback


def _markdown_artifact_present(text: str) -> bool:
    return bool(re.search(r"```|^\s{0,3}#{1,6}\s+\S|\*\*[^*]+\*\*", text or "", flags=re.MULTILINE))


def _render_requires_grounding(utterance_kind: str, render_kind: str) -> bool:
    if utterance_kind not in {"dictation", "format_dictation", "mixed"}:
        return False
    return render_kind not in {"none"}


def _numeric_tokens(text: str) -> set[str]:
    value = text or ""
    tokens: set[str] = set()
    clock_re = re.compile(
        r"(?<![A-Za-z0-9])(?P<hour>\d{1,2})(?:(?::|\.)0{2})?\s*(?P<ampm>a\.?\s*m\.?|p\.?\s*m\.?|am|pm)(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    clock_spans: list[tuple[int, int]] = []
    for match in clock_re.finditer(value):
        suffix = re.sub(r"[^apm]", "", match.group("ampm").casefold())
        if suffix.startswith("a"):
            suffix = "am"
        elif suffix.startswith("p"):
            suffix = "pm"
        hour = int(match.group("hour"))
        tokens.add(f"{hour}{suffix}")
        tokens.add(f"{hour}:00{suffix}")
        tokens.add(f"{hour}:00")
        clock_spans.append(match.span())

    def in_clock_span(start: int, end: int) -> bool:
        return any(start >= s and end <= e for s, e in clock_spans)

    for match in re.finditer(r"(?<![A-Za-z0-9])\d+(?:[.,:/-]\d+)*(?![A-Za-z0-9])", value):
        if in_clock_span(*match.span()):
            continue
        token = match.group(0).replace(",", "")
        tokens.add(token)
    return tokens


def _new_numbers_without_evidence(source: str, output: str) -> set[str]:
    src = _numeric_tokens(source)
    out = _numeric_tokens(output)
    return out - src
