"""LLM-backed action extractor.

Tier-2 fallback for the deterministic regex grammar. When
:func:`juno_core_v3.actions.grammar.parse_actions` returns ``None`` for a
transcript that *does* start with the Juno wake phrase, we hand the
post-wake text to a registered LLM callable. The model returns a strict
JSON envelope (``intent`` + ``actions``); this module clamps, validates,
and normalizes that envelope into canonical :class:`Action` objects.

Design constraints:

- The core never imports an LLM SDK. The shell/runtime registers an
  implementation at startup via :func:`set_llm_extractor`, mirroring
  ``set_llm_fallback`` in :mod:`juno_core_v3.actions.timeparse`.
- The extractor never raises. Any exception, schema violation, or
  malformed envelope returns ``None`` so callers fall back cleanly.
- Time strings returned by the model go through the existing
  :func:`parse_when` pipeline so the ``ParsedTime`` contract (and the
  resolver's roll-forward / default-fill rules) stays the single source
  of truth for time normalization.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from juno_core_v3.actions.contracts import (
    Action,
    ActionKind,
    ActionOperation,
    ParsedTime,
    Schedule,
    SeriesRule,
    VagueSchedule,
)
from juno_core_v3.actions.timeparse import parse_when
from juno_core_v3.actions.vague_time import resolve_vague

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# An extractor takes the post-wake text plus an optional ``now`` and returns
# the raw model envelope (already JSON-decoded) or ``None`` when it cannot
# produce a usable response. The envelope shape is documented in
# :func:`extract_actions_with_llm`. Returning ``None`` (as opposed to a
# ``non_action`` envelope) is reserved for transport / model errors.
LlmActionExtractor = Callable[[str, datetime | None], dict[str, Any] | None]
TargetResolver = Callable[[dict[str, Any]], Any | None]

_extractor: LlmActionExtractor | None = None


@dataclass(frozen=True, slots=True)
class LlmActionExtractionResult:
    """Detailed outcome from a registered action extractor call."""

    intent: str | None
    actions: list[Action] | None = None
    rejected_reason: str | None = None
    error: str | None = None
    allow_regex_fallback: bool = False


def set_llm_extractor(fn: LlmActionExtractor | None) -> None:
    """Register (or clear) the LLM action-extraction callable.

    Wiring is owned by the shell/runtime layer. Tests inject fakes by
    calling this with a stub and resetting to ``None`` in teardown.
    """

    global _extractor
    _extractor = fn


def get_llm_extractor() -> LlmActionExtractor | None:
    """Return the currently registered extractor, or ``None``."""

    return _extractor


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Closed enum of action kinds the extractor is allowed to emit. Anything
# else is clamped away by the validator.
_ALLOWED_KINDS: frozenset[str] = frozenset(k.value for k in ActionKind)

# Hard cap on actions per utterance. Prevents a runaway model from emitting
# dozens of bogus reminders for one spoken sentence.
_MAX_ACTIONS_PER_UTTERANCE = 25

# Hard cap on body length. Reminder/note bodies are short by nature; longer
# strings almost always mean the model echoed the whole transcript.
_MAX_BODY_CHARS = 500

_EXECUTE_INTENT = "execute_action"
_NON_EXECUTE_INTENTS = {
    "dictation",
    "meta_about_juno",
    "question",
    "example_or_quote",
    "ambiguous",
}
_LEGACY_INTENT_ALIASES = {
    "action_request": _EXECUTE_INTENT,
    "non_action": "dictation",
    "non_action_dictation": "dictation",
    "about_juno_or_meta": "meta_about_juno",
}
_MIN_DECISION_CONFIDENCE = 0.72
_MIN_ACTION_CONFIDENCE = 0.70


def select_actions_schema_version(text: str, *, use_v3: bool | None = None) -> str:
    """Choose the action extraction schema for a post-wake utterance.

    v2 already supports multiple create-only actions. Keep ordinary chains
    such as "add two alarms ... and remind me ..." on v2 so enabling v3 does
    not expose the model to operation/container choices it does not need.
    """

    if use_v3 is None:
        use_v3 = _env_bool("JUNO_ACTIONS_SCHEMA_V3", False)
    if not use_v3:
        return "actions_intent_v2"

    lowered = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    if not lowered:
        return "actions_intent_v2"

    if _env_bool("JUNO_ACTIONS_COMPOUND", False) and _looks_like_compound_action_request(lowered):
        return "actions_intent_v3"

    operation_patterns = (
        r"\b(?:change|move|complete|mark|snooze|cancel|delete|remove|undo|stop|read|show|find)\b",
        r"\blist (?:my|all|the|reminders?|alarms?|notes?)\b",
        r"\bdid i set\b",
        r"\bdo i have\b",
        r"\bwhat reminders\b",
    )
    if any(re.search(pattern, lowered) for pattern in operation_patterns):
        return "actions_intent_v3"

    schedule_patterns = (
        r"\bevery\b",
        r"\bdaily\b",
        r"\bweekly\b",
        r"\bmonthly\b",
        r"\bfor the next\b",
        r"\blater\b",
        r"\bsoon\b",
        r"\bafter lunch\b",
        r"\bthis evening\b",
        r"\btonight\b",
        r"\bweekend\b",
    )
    if any(re.search(pattern, lowered) for pattern in schedule_patterns):
        return "actions_intent_v3"

    container_patterns = (
        r"\b(?:create|make) (?:a |an |the )?[\w -]*\blist\b",
        r"\b(?:add|put|append) .+\bto (?:my |the )?[\w -]*(?:\blist\b|\btodos?\b|\bto-dos?\b|\bgroceries\b|\bshopping\b|\bpacking\b)",
        r"\b(?:show|read|list|find) .*(?:\blist\b|\btodos?\b|\bto-dos?\b|\bgroceries\b|\bshopping\b|\bpacking\b)",
        r"\bremove .+\bfrom (?:my |the )?[\w -]*(?:\blist\b|\btodos?\b|\bto-dos?\b|\bgroceries\b|\bshopping\b|\bpacking\b)",
    )
    if any(re.search(pattern, lowered) for pattern in container_patterns):
        return "actions_intent_v3"

    return "actions_intent_v2"


def _looks_like_compound_action_request(lowered: str) -> bool:
    if re.search(r"\bthen\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+to\b", lowered):
        return True
    if re.search(r"\bset an alarm\b.+\band\s+a\s+reminder\b", lowered):
        return True
    if re.search(r"\bcreate (?:a |an |the )?.*?\blist\b.+\band\s+remind me\b", lowered):
        return True
    if re.search(r"\bremind me at\b.+\band\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b.+\bto\b", lowered):
        return True
    return False


def expected_envelope_schema(*, version: str = "actions_intent_v2") -> dict[str, Any]:
    """Return the JSON schema description the extractor implementation
    should hand to the model. Centralized here so the prompt/schema and
    the validator can never drift.

    ``version`` selects between v2 (today's stable schema) and v3 (the
    Phase 1+ schema with Schedule + Operation + Target + Container).
    The v3 schema is a strict superset: every v2 envelope is a valid
    v3 envelope.
    """

    if version == "actions_intent_v2":
        return _v2_schema()
    if version == "actions_intent_v3":
        return _v3_schema()
    raise ValueError(f"unknown schema version: {version}")


def _v2_schema() -> dict[str, Any]:
    return {
        "$id": "actions_intent_v2",
        "type": "object",
        "required": [
            "intent",
            "should_execute",
            "confidence",
            "decision_evidence_span",
            "actions",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["actions_intent_v2"],
            },
            "intent": {
                "type": "string",
                "enum": [
                    _EXECUTE_INTENT,
                    "dictation",
                    "meta_about_juno",
                    "question",
                    "example_or_quote",
                    "ambiguous",
                ],
            },
            "should_execute": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "decision_evidence_span": {"type": ["string", "null"]},
            "rejected_reason": {"type": ["string", "null"]},
            "actions": {
                "type": "array",
                "maxItems": _MAX_ACTIONS_PER_UTTERANCE,
                "items": {
                    "type": "object",
                    "required": ["kind", "body", "evidence_span"],
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": sorted(_ALLOWED_KINDS),
                        },
                        "body": {"type": "string", "minLength": 1},
                        "evidence_span": {"type": "string", "minLength": 1},
                        "title": {"type": ["string", "null"]},
                        "notes": {"type": ["string", "null"]},
                        "when_text": {"type": ["string", "null"]},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "ambiguity_reason": {"type": ["string", "null"]},
                        "needs_confirmation": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def _v3_schema() -> dict[str, Any]:
    """Phase 1+ schema. Strict superset of v2.

    Adds:
    - per-action ``operation`` (default ``create``)
    - per-action ``schedule`` (instant | series | vague) replacing the
      free-text ``when_text`` for richer cases; ``when_text`` stays
      accepted for back-compat with prompts that haven't switched.
    - per-action ``target`` for non-create operations (Phase 2 only
      honors targets; Phase 1 accepts but rejects non-create at validate
      time).
    - per-action ``container`` for list / folder targeting (Phase 4).
    """

    series_schema = {
        "type": "object",
        "required": ["freq"],
        "properties": {
            "freq": {"type": "string", "enum": ["DAILY", "WEEKLY", "MONTHLY", "YEARLY"]},
            "interval": {"type": "integer", "minimum": 1},
            "by_day": {
                "type": "array",
                "items": {"type": "string", "enum": ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]},
            },
            "by_month_day": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 31}},
            "by_month": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 12}},
            "count": {"type": ["integer", "null"], "minimum": 1},
            "until_iso": {"type": ["string", "null"]},
            "first_occurrence_iso": {"type": ["string", "null"]},
            "tz": {"type": ["string", "null"]},
            "exclude_dates_iso": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }
    instant_schema = {
        "type": "object",
        "required": ["iso"],
        "properties": {
            "iso": {"type": "string"},
            "tz": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }
    vague_schema = {
        "type": "object",
        "required": ["bucket"],
        "properties": {
            "bucket": {
                "type": "string",
                "enum": [
                    "later",
                    "soon",
                    "tonight",
                    "morning",
                    "afternoon",
                    "evening",
                    "weekend",
                    "next_week",
                ],
            },
            "default_iso": {"type": ["string", "null"]},
            "tz": {"type": ["string", "null"]},
            "needs_confirmation": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    schedule_schema = {
        "type": "object",
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string", "enum": ["instant", "series", "vague"]},
            "instant": {"oneOf": [{"type": "null"}, instant_schema]},
            "series": {"oneOf": [{"type": "null"}, series_schema]},
            "vague": {"oneOf": [{"type": "null"}, vague_schema]},
        },
        "additionalProperties": False,
    }
    target_schema = {
        "type": "object",
        "required": ["ref_kind"],
        "properties": {
            "ref_kind": {
                "type": "string",
                "enum": ["by_id", "by_description", "by_pronoun", "by_query"],
            },
            "id": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},
            "pronoun": {
                "type": ["string", "null"],
                "enum": [None, "this", "that", "it", "the_last_one"],
            },
            "filter": {"type": ["object", "null"]},
        },
        "additionalProperties": False,
    }
    container_schema = {
        "type": "object",
        "properties": {
            "list_name": {"type": ["string", "null"]},
            "folder_name": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }

    return {
        "$id": "actions_intent_v3",
        "type": "object",
        "required": [
            "intent",
            "should_execute",
            "confidence",
            "decision_evidence_span",
            "actions",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["actions_intent_v2", "actions_intent_v3"],
            },
            "intent": {
                "type": "string",
                "enum": [
                    _EXECUTE_INTENT,
                    "dictation",
                    "meta_about_juno",
                    "question",
                    "example_or_quote",
                    "ambiguous",
                ],
            },
            "should_execute": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "decision_evidence_span": {"type": ["string", "null"]},
            "rejected_reason": {"type": ["string", "null"]},
            "actions": {
                "type": "array",
                "maxItems": _MAX_ACTIONS_PER_UTTERANCE,
                "items": {
                    "type": "object",
                    "required": ["kind", "evidence_span"],
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "create",
                                "update",
                                "complete",
                                "snooze",
                                "delete",
                                "query",
                                "append_to",
                                "remove_from",
                            ],
                        },
                        "kind": {"type": "string", "enum": sorted(_ALLOWED_KINDS)},
                        "body": {"type": "string"},
                        "title": {"type": ["string", "null"]},
                        "notes": {"type": ["string", "null"]},
                        "evidence_span": {"type": "string", "minLength": 1},
                        "when_text": {"type": ["string", "null"]},
                        "schedule": {"oneOf": [{"type": "null"}, schedule_schema]},
                        "target": {"oneOf": [{"type": "null"}, target_schema]},
                        "container": {"oneOf": [{"type": "null"}, container_schema]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "ambiguity_reason": {"type": ["string", "null"]},
                        "needs_confirmation": {"type": "boolean"},
                        "snooze_offset_seconds": {"type": ["integer", "null"]},
                        "relative_offset_seconds": {"type": ["integer", "null"]},
                        "link_id": {"type": ["string", "null"]},
                        "links_to": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _intent_from_envelope(envelope: Any) -> str | None:
    if not isinstance(envelope, dict):
        return None
    intent = envelope.get("intent")
    if not isinstance(intent, str):
        return None
    return intent.strip().lower()


def _canonical_intent(envelope: Any) -> str | None:
    intent = _intent_from_envelope(envelope)
    if intent is None:
        return None
    return _LEGACY_INTENT_ALIASES.get(intent, intent)


def _coerce_confidence(value: Any, *, default: float = 0.0) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = default
    return max(0.0, min(1.0, conf))


_SPAN_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)


def _span_key(text: Any) -> str:
    s = str(text or "").casefold().strip()
    s = s.strip(" \t\r\n\"'`.,!?;:()[]{}<>")
    s = _SPAN_PUNCT_RE.sub(" ", s)
    return " ".join(s.split())


def _span_present(span: Any, source_text: str) -> bool:
    needle = _span_key(span)
    haystack = _span_key(source_text)
    return bool(needle) and needle in haystack


_SOURCE_CLAUSE_RE = re.compile(r"[^.!?;]+[.!?;]?", re.UNICODE)
_EXPLICIT_REMINDER_RE = re.compile(
    r"\b(?:remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder|create\s+(?:a\s+)?reminder)\b",
    re.IGNORECASE,
)
_EXPLICIT_ALARM_RE = re.compile(
    r"\b(?:alarm|wake\s+me(?:\s+up)?)\b",
    re.IGNORECASE,
)
_ACTION_TOKEN_STOPWORDS = {
    "a", "an", "and", "as", "at", "for", "from", "in", "into", "is", "it",
    "me", "my", "of", "on", "one", "please", "set", "that", "the", "then",
    "this", "to", "with", "you",
    "add", "alarm", "alarms", "create", "make", "note", "notes", "remind",
    "reminder", "reminders", "wake",
    "today", "tomorrow", "tonight", "later", "soon", "weekend", "morning",
    "afternoon", "evening", "first", "second", "third",
}


def _source_clause_containing(source_text: str, terms: list[str]) -> str | None:
    cleaned_terms = [term for term in terms if _span_key(term)]
    if not cleaned_terms:
        return None
    for match in _SOURCE_CLAUSE_RE.finditer(str(source_text or "")):
        clause = match.group(0).strip()
        if clause and all(_span_present(term, clause) for term in cleaned_terms):
            return clause
    return None


def _resolve_action_evidence_span(
    raw: dict[str, Any],
    *,
    kind: ActionKind,
    body: str,
    source_text: str,
) -> str | None:
    evidence = raw.get("evidence_span")
    if _span_present(evidence, source_text):
        return str(evidence).strip()

    # Small local models occasionally copy a digit wrong in evidence_span
    # while still extracting the body and when_text correctly. Keep the
    # evidence gate strict by requiring both independently grounded spans.
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    if kind not in (ActionKind.REMINDER, ActionKind.ALARM):
        return None

    when_text = raw.get("when_text")
    if not isinstance(when_text, str) or not when_text.strip():
        return None
    if not body or not _span_present(body, source_text):
        return None
    if not _span_present(when_text, source_text):
        return None

    return _source_clause_containing(source_text, [when_text, body]) or evidence.strip()


def _kind_with_explicit_span_override(kind: ActionKind, raw_span: str) -> ActionKind:
    """Prefer the action kind explicitly spoken in this action span.

    The model can carry context from an earlier declaration like "add two
    alarms" into a later standalone "please remind me..." clause. The span
    itself is stronger evidence for sink routing than the surrounding list.
    """

    if (
        kind is ActionKind.ALARM
        and _EXPLICIT_REMINDER_RE.search(raw_span)
        and not _EXPLICIT_ALARM_RE.search(raw_span)
    ):
        return ActionKind.REMINDER
    return kind


def _is_execute_envelope(
    envelope: dict[str, Any],
    source_text: str,
) -> tuple[bool, str | None]:
    intent = _canonical_intent(envelope)
    if intent != _EXECUTE_INTENT:
        return False, "not_execute_intent"
    if envelope.get("should_execute") is not True:
        return False, "should_execute_false"
    confidence = _coerce_confidence(envelope.get("confidence"))
    if confidence < _MIN_DECISION_CONFIDENCE:
        return False, "low_decision_confidence"
    evidence = envelope.get("decision_evidence_span")
    if not _span_present(evidence, source_text):
        return False, "missing_decision_evidence"
    raw_actions = envelope.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        return False, "empty_actions"
    return True, None


def _coerce_action_dict(
    raw: Any,
    *,
    source_text: str,
    top_confidence: float,
    now: datetime | None,
    target_resolver: TargetResolver | None = None,
) -> Action | None:
    """Validate one action dict and return a canonical :class:`Action`.

    Returns ``None`` if the dict is malformed, has an unknown kind, or
    has an empty body. Time strings that fail :func:`parse_when` are
    preserved as ``None`` rather than rejected — the action is still
    executable as a note, and the validator's job is to keep things
    safe, not to second-guess the user's wording.

    Accepts both v2 and v3 action shapes. v3 actions may carry
    ``operation`` and ``schedule``; v2 actions never do. Phase 1 only
    *emits* schedules of ``kind == "series"`` for non-create operations
    we are not yet ready to dispatch — those are clamped to
    ``ActionOperation.CREATE`` and rejected at the pipeline boundary
    when the operations flag is off (Phase 2 lifts that).
    """

    if not isinstance(raw, dict):
        return None

    kind_str = raw.get("kind")
    if not isinstance(kind_str, str) or kind_str.lower() not in _ALLOWED_KINDS:
        return None
    kind = ActionKind(kind_str.lower())

    body_raw = raw.get("body")
    body = body_raw.strip() if isinstance(body_raw, str) else ""
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS].rstrip()

    confidence = _coerce_confidence(raw.get("confidence"), default=top_confidence)
    if confidence < _MIN_ACTION_CONFIDENCE:
        return None

    raw_span = _resolve_action_evidence_span(
        raw,
        kind=kind,
        body=body,
        source_text=source_text,
    )
    if raw_span is None:
        return None
    kind = _kind_with_explicit_span_override(kind, raw_span)

    # Operation: v3 only. Default CREATE for v2.
    op_raw = raw.get("operation")
    operation = ActionOperation.CREATE
    if isinstance(op_raw, str):
        try:
            operation = ActionOperation(op_raw.strip().lower())
        except ValueError:
            return None
    body = _ground_or_repair_action_body(
        kind=kind,
        operation=operation,
        body=body,
        raw_span=raw_span,
        when_text=raw.get("when_text"),
    )
    if body is None:
        return None
    snooze_offset_seconds = _int_or_none(raw.get("snooze_offset_seconds"))
    relative_offset_seconds = _int_or_none(raw.get("relative_offset_seconds"))

    target: dict[str, Any] | None = None
    container: dict[str, Any] | None = None
    juno_id: str | None = None
    sink_id: str | None = None
    link_id = _clean_link(raw.get("link_id"))
    links_to = _clean_link(raw.get("links_to"))
    if (link_id or links_to) and not _env_bool("JUNO_ACTIONS_COMPOUND", False):
        return None
    raw_container = raw.get("container")
    if isinstance(raw_container, dict):
        container = _coerce_container_dict(raw_container)
        if container is not None:
            container = _ground_or_repair_container(
                container=container,
                raw_span=raw_span,
                kind=kind,
                operation=operation,
            )
        if container is not None and not _env_bool("JUNO_ACTIONS_CONTAINERS", False):
            return None
        if container is not None:
            if kind is ActionKind.REMINDER and not container.get("list_name"):
                return None
            if kind is ActionKind.NOTE and not container.get("folder_name"):
                return None
            if kind is ActionKind.ALARM:
                return None

    # Operations other than CREATE need a resolved target reference.
    # The feature flag preserves Phase 1 behavior until Phase 2 is
    # explicitly enabled by the shell/runtime.
    if operation is not ActionOperation.CREATE:
        if not _env_bool("JUNO_ACTIONS_OPERATIONS", False):
            return None
        if operation in (
            ActionOperation.APPEND_TO,
            ActionOperation.REMOVE_FROM,
        ) or (operation is ActionOperation.QUERY and container is not None):
            if not _env_bool("JUNO_ACTIONS_CONTAINERS", False) or container is None:
                return None
            if operation in (ActionOperation.APPEND_TO, ActionOperation.REMOVE_FROM) and kind is not ActionKind.REMINDER:
                return None
            raw_target = raw.get("target")
            if isinstance(raw_target, dict):
                target = dict(raw_target)
        else:
            raw_target = raw.get("target")
            if not isinstance(raw_target, dict):
                return None
            target = dict(raw_target)
            if _is_relative_move_snooze(
                operation=operation,
                raw=raw,
                source_text=source_text,
                target=target,
            ):
                operation = ActionOperation.UPDATE
                if relative_offset_seconds is None:
                    relative_offset_seconds = snooze_offset_seconds
                snooze_offset_seconds = None
            if target_resolver is None:
                return None
            resolved = target_resolver(target)
            if resolved is None:
                return None
            candidates = list(getattr(resolved, "candidates", []) or [])
            confidence = float(getattr(resolved, "confidence", 0.0) or 0.0)
            # For mutating single-target operations, ambiguous low-confidence
            # matches must fall through to "no action" until the chooser ships.
            if operation is not ActionOperation.QUERY and len(candidates) > 1 and confidence < 0.7:
                return None
            juno_id = str(getattr(resolved, "juno_id", "") or "").strip() or None
            top_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
            sink_id = str(top_candidate.get("sink_id") or "").strip() or None
            if not juno_id:
                return None
            target["resolved_via"] = str(getattr(resolved, "resolved_via", "") or target.get("ref_kind") or "")
            target["confidence"] = confidence
    if operation is ActionOperation.CREATE and container is not None:
        if not _env_bool("JUNO_ACTIONS_CONTAINERS", False):
            return None

    # CREATE operations must have a non-empty body.
    if operation is ActionOperation.CREATE and not body and container is None:
        return None
    if not body and target is not None:
        body = str(target.get("description") or target.get("ref_kind") or operation.value).strip()

    # Schedule resolution: v3 can emit both a structured schedule and
    # the spoken ``when_text``. For one-shot instant schedules, trust
    # our parser's resolution of the spoken phrase over the model's ISO:
    # the prompt may be relative ("tomorrow 10pm"), while a small local
    # model can hallucinate a stale absolute date.
    when: ParsedTime | None = None
    raw_schedule = raw.get("schedule")
    raw_schedule_kind = _raw_schedule_kind(raw_schedule)
    if raw_schedule_kind == "vague" and not _env_bool("JUNO_ACTIONS_VAGUE_TIME", False):
        return None
    schedule = _coerce_schedule_dict(raw.get("schedule"), kind=kind, now=now)
    when_text = raw.get("when_text")
    when_from_text: ParsedTime | None = None
    has_when_text = isinstance(when_text, str) and bool(when_text.strip())
    if kind in (ActionKind.REMINDER, ActionKind.ALARM) and has_when_text:
        try:
            when_from_text = parse_when(str(when_text).strip(), now=now)
        except Exception:  # noqa: BLE001 - never break the pipeline
            logger.exception("parse_when raised on LLM-supplied clause")
            when_from_text = None
        if when_from_text is not None and bool(raw.get("needs_confirmation")):
            when_from_text = ParsedTime(
                iso=when_from_text.iso,
                confidence=when_from_text.confidence,
                source=when_from_text.source,
                inferred=when_from_text.inferred,
                inference_note=when_from_text.inference_note,
                needs_confirmation=True,
            )
        if schedule is not None and raw_schedule_kind == "instant" and when_from_text is not None:
            try:
                schedule = Schedule(kind="instant", instant=when_from_text)
            except ValueError:
                schedule = None

    if schedule is None and kind in (ActionKind.REMINDER, ActionKind.ALARM):
        if has_when_text:
            when = when_from_text
            if when is None and kind == ActionKind.REMINDER and _span_key(str(when_text)) not in _span_key(body):
                return None
        elif kind == ActionKind.ALARM:
            return None

        if kind == ActionKind.ALARM and when is None:
            return None

    # Fill ``when`` from schedule.instant for back-compat with v2 sinks.
    if when is None and schedule is not None and schedule.instant is not None:
        when = schedule.instant
    if kind in (ActionKind.REMINDER, ActionKind.ALARM) and when is not None and has_when_text:
        body = _strip_leading_when_text_from_body(body, str(when_text))
    needs_confirmation = bool(raw.get("needs_confirmation"))
    if schedule is not None and schedule.vague is not None:
        needs_confirmation = True
        when = ParsedTime(
            iso=schedule.vague.default_iso,
            confidence=0.7,
            source="default",
            inferred=True,
            inference_note=f"vague time: {schedule.vague.bucket}",
            needs_confirmation=True,
        )

    # Reminder created via schedule still needs *some* time when the
    # schedule is series/vague; instant covers itself. Treat series and
    # vague as legitimate and let the resolver hand the Swift sink the
    # right shape downstream.

    return Action(
        kind=kind,
        body=body,
        raw_span=raw_span,
        when=when,
        schedule=schedule,
        operation=operation,
        target=target,
        container=container,
        juno_id=juno_id,
        sink_id=sink_id,
        snooze_offset_seconds=snooze_offset_seconds,
        relative_offset_seconds=relative_offset_seconds,
        link_id=link_id,
        links_to=links_to,
        needs_confirmation=needs_confirmation,
    )


def _strip_leading_when_text_from_body(body: str, when_text: str) -> str:
    cleaned = body.strip()
    when_words = _span_key(when_text).split()
    if not cleaned or not when_words:
        return cleaned
    pattern = r"^\s*" + r"[^\w]+".join(re.escape(word) for word in when_words) + r"\b"
    match = re.match(pattern, cleaned, flags=re.IGNORECASE)
    if match is None:
        return cleaned
    rest = cleaned[match.end():].strip(" \t\r\n\"'`.,!?;:()[]{}<>-")
    rest = re.sub(r"^(?:to|for|about|when)\b[\s:,-]*", "", rest, flags=re.IGNORECASE).strip()
    return rest or cleaned


def _ground_or_repair_action_body(
    *,
    kind: ActionKind,
    operation: ActionOperation,
    body: str,
    raw_span: str,
    when_text: Any,
) -> str | None:
    if kind is ActionKind.ALARM and (not body or body.casefold() == "alarm"):
        return body or "Alarm"
    needs_grounding = (
        operation is ActionOperation.CREATE and kind in (ActionKind.REMINDER, ActionKind.NOTE)
    ) or (
        operation in (ActionOperation.APPEND_TO, ActionOperation.REMOVE_FROM)
    )
    if not needs_grounding:
        return body
    if body and _body_grounded_in_span(body, raw_span):
        return body
    derived = _derive_body_from_action_span(
        raw_span=raw_span,
        kind=kind,
        operation=operation,
        when_text=str(when_text or ""),
    )
    if derived and _body_grounded_in_span(derived, raw_span):
        return derived
    if not body and operation in (ActionOperation.QUERY, ActionOperation.DELETE, ActionOperation.COMPLETE):
        return body
    return None


def _body_grounded_in_span(body: str, raw_span: str) -> bool:
    if not body:
        return False
    if _span_present(body, raw_span):
        return True
    body_tokens = _meaningful_action_tokens(body)
    if not body_tokens:
        return False
    span_tokens = _meaningful_action_tokens(raw_span)
    if not span_tokens:
        return False
    matched = sum(1 for token in body_tokens if _token_or_variant_present(token, span_tokens))
    return matched >= max(1, int(len(body_tokens) * 0.67))


def _meaningful_action_tokens(text: str) -> list[str]:
    tokens = _span_key(text).split()
    return [token for token in tokens if len(token) >= 3 and token not in _ACTION_TOKEN_STOPWORDS]


def _token_or_variant_present(token: str, haystack: list[str]) -> bool:
    if token in haystack:
        return True
    variants = {token}
    if token.endswith("ing") and len(token) > 5:
        variants.add(token[:-3])
    if token.endswith("ed") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 4:
        variants.add(token[:-1])
    return any(variant in haystack for variant in variants)


def _derive_body_from_action_span(
    *,
    raw_span: str,
    kind: ActionKind,
    operation: ActionOperation,
    when_text: str,
) -> str:
    parsed_container = _parse_container_clause(raw_span)
    if parsed_container is not None and operation in (ActionOperation.APPEND_TO, ActionOperation.REMOVE_FROM):
        return parsed_container[1] or ""

    text = re.sub(r"\s+", " ", str(raw_span or "")).strip(" \t\r\n\"'`.,!?;:()[]{}<>")
    if not text:
        return ""
    text = re.sub(r"^(?:and|then|also|please|okay|ok)\b[\s,.;:!?-]*", "", text, flags=re.IGNORECASE).strip()
    if kind is ActionKind.REMINDER:
        text = re.sub(
            r"^(?:i\s+want\s+you\s+to\s+|please\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder|create\s+(?:a\s+)?reminder)\b",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n,.;:!?-")
    elif kind is ActionKind.NOTE:
        text = re.sub(
            r"^(?:i\s+want\s+you\s+to\s+|please\s+)?(?:take\s+(?:a\s+)?note|add\s+(?:a\s+)?note|create\s+(?:a\s+)?note|make\s+(?:a\s+)?note|note\s+that)\b",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n,.;:!?-")
    if when_text.strip():
        text = _remove_loose_phrase(text, when_text)
    if kind is ActionKind.REMINDER:
        to_match = re.search(r"\bto\s+(?P<body>.+)$", text, flags=re.IGNORECASE)
        if to_match is not None:
            text = to_match.group("body")
    text = re.sub(r"^(?:to|about|for|at|on|in|when|that)\b[\s:,-]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:at|on|in|for|when)\s*$", "", text, flags=re.IGNORECASE)
    return text.strip(" \t\r\n\"'`.,!?;:()[]{}<>-")


def _remove_loose_phrase(text: str, phrase: str) -> str:
    words = _span_key(phrase).split()
    if not words:
        return text
    pattern = r"(?<![A-Za-z0-9])" + r"[\W_]+".join(re.escape(word) for word in words) + r"(?![A-Za-z0-9])"
    return re.sub(pattern, " ", text, flags=re.IGNORECASE).strip()


def _ground_or_repair_container(
    *,
    container: dict[str, Any],
    raw_span: str,
    kind: ActionKind,
    operation: ActionOperation,
) -> dict[str, Any] | None:
    if kind is not ActionKind.REMINDER:
        return container
    list_name = str(container.get("list_name") or "").strip()
    if list_name and _span_present(list_name, raw_span):
        return container
    if not list_name and operation is ActionOperation.CREATE:
        return container
    parsed = _parse_container_clause(raw_span)
    if parsed is None:
        return None if operation in (ActionOperation.APPEND_TO, ActionOperation.REMOVE_FROM) else container
    parsed_operation, _body, parsed_list = parsed
    if operation in (ActionOperation.APPEND_TO, ActionOperation.REMOVE_FROM) and parsed_operation != operation:
        return None
    repaired = dict(container)
    repaired["list_name"] = parsed_list
    return repaired


def _parse_container_clause(text: str) -> tuple[ActionOperation, str | None, str] | None:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip().casefold()).strip(" .!?")
    match = re.search(r"\badd (?P<body>.+?) to (?:my |the )?(?P<list>.+?)(?: list)?$", cleaned)
    if match:
        return (ActionOperation.APPEND_TO, match.group("body").strip(), _title_name(match.group("list")))
    match = re.search(r"\bremove (?P<body>.+?) from (?:my |the )?(?P<list>.+?)(?: list)?$", cleaned)
    if match:
        return (ActionOperation.REMOVE_FROM, match.group("body").strip(), _title_name(match.group("list")))
    return None


def _title_name(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    value = re.sub(r"\s+list$", "", value, flags=re.IGNORECASE).strip()
    return " ".join(part.capitalize() for part in value.split())


def _is_relative_move_snooze(
    *,
    operation: ActionOperation,
    raw: dict[str, Any],
    source_text: str,
    target: dict[str, Any],
) -> bool:
    """Normalize model drift for named relative moves.

    The Phase 2 contract distinguishes "snooze that" from "push/move the
    named reminder by X". Qwen sometimes emits ``snooze`` for both. Keep
    true snoozes intact and only normalize descriptive move language.
    """

    if operation is not ActionOperation.SNOOZE:
        return False
    if str(target.get("ref_kind") or "").strip().lower() != "by_description":
        return False
    pieces = [
        raw.get("evidence_span"),
        target.get("description"),
        source_text,
    ]
    text = " ".join(p for p in pieces if isinstance(p, str)).casefold()
    if "snooze" in text:
        return False
    move_terms = ("push ", "move ", "shift ", "delay ", "bump ")
    return any(term in f" {text} " for term in move_terms)


def _coerce_container_dict(raw: dict[str, Any]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    list_name = raw.get("list_name") or raw.get("listName")
    folder_name = raw.get("folder_name") or raw.get("folderName")
    if isinstance(list_name, str) and list_name.strip():
        out["list_name"] = list_name.strip()
    if isinstance(folder_name, str) and folder_name.strip():
        out["folder_name"] = folder_name.strip()
    return out or None


def _clean_link(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    return cleaned or None


def _coerce_schedule_dict(
    raw: Any,
    *,
    kind: ActionKind,
    now: datetime | None,
) -> Schedule | None:
    """Validate one ``schedule`` block. Returns None for invalid shapes.

    Notes never carry a schedule. For reminder/alarm, we accept any of
    the three variants. Ill-formed schedules return None so the caller
    can fall back to the legacy ``when_text`` path.
    """
    if raw is None or not isinstance(raw, dict):
        return None
    if kind == ActionKind.NOTE:
        return None
    sched_kind = raw.get("kind")
    if not isinstance(sched_kind, str):
        return None
    sched_kind = sched_kind.strip().lower()
    if sched_kind == "instant":
        instant_data = raw.get("instant")
        if not isinstance(instant_data, dict):
            return None
        iso = instant_data.get("iso")
        if not isinstance(iso, str) or not iso.strip():
            return None
        try:
            pt = parse_when(iso.strip(), now=now)
        except Exception:  # noqa: BLE001
            pt = None
        if pt is None:
            # Accept the model's literal ISO string when parser declines —
            # the resolver layer downstream rounds-trips through the same
            # ``ParsedTime`` shape.
            pt = ParsedTime(iso=iso.strip(), source="llm", confidence=0.85)
        try:
            return Schedule(kind="instant", instant=pt)
        except ValueError:
            return None
    if sched_kind == "series":
        series_data = raw.get("series")
        if not isinstance(series_data, dict):
            return None
        try:
            rule = SeriesRule.from_dict(series_data)
        except (KeyError, ValueError, TypeError):
            return None
        try:
            return Schedule(kind="series", series=rule)
        except ValueError:
            return None
    if sched_kind == "vague":
        vague_data = raw.get("vague")
        if not isinstance(vague_data, dict):
            return None
        if not _env_bool("JUNO_ACTIONS_VAGUE_TIME", False):
            return None
        bucket = str(vague_data.get("bucket") or "").strip().lower()
        default_iso = vague_data.get("default_iso")
        if not isinstance(default_iso, str) or not default_iso.strip():
            resolver_now = now or datetime.now().astimezone()
            tz = str(vague_data.get("tz") or _tz_name(resolver_now) or "").strip()
            try:
                default_iso = resolve_vague(bucket, now=resolver_now, tz=tz).isoformat()
            except ValueError:
                return None
        vague_payload = dict(vague_data)
        vague_payload["bucket"] = bucket
        vague_payload["default_iso"] = default_iso
        vague_payload["needs_confirmation"] = True
        try:
            vague = VagueSchedule.from_dict(vague_payload)
        except (KeyError, ValueError, TypeError):
            return None
        try:
            return Schedule(kind="vague", vague=vague)
        except ValueError:
            return None
    return None


def _raw_schedule_kind(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if not isinstance(kind, str):
        return None
    return kind.strip().lower() or None


def _validate_action_links(actions: list[Action]) -> bool:
    """Phase 6 link integrity: links only point to prior declared actions."""

    seen: set[str] = set()
    for action in actions:
        if action.link_id:
            if action.link_id in seen:
                return False
            seen.add(action.link_id)
        if action.links_to and action.links_to not in seen:
            return False
    return True


def _tz_name(value: datetime) -> str:
    tzinfo = value.tzinfo
    key = getattr(tzinfo, "key", None)
    if isinstance(key, str):
        return key
    if tzinfo is None:
        return ""
    return str(tzinfo)


def validate_envelope(
    envelope: Any,
    *,
    raw_span_fallback: str,
    now: datetime | None = None,
    target_resolver: TargetResolver | None = None,
) -> list[Action] | None:
    """Validate a model envelope into canonical actions.

    Returns a non-empty list of :class:`Action` only when the model proved
    executable intent with evidence spans and confidence. Non-execute
    envelopes and rejected execute envelopes return ``None``.
    """

    if not isinstance(envelope, dict):
        return None

    source_text = raw_span_fallback
    ok, _reason = _is_execute_envelope(envelope, source_text)
    if not ok:
        return None

    raw_actions = envelope.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        return None

    top_confidence = _coerce_confidence(envelope.get("confidence"))
    out: list[Action] = []
    for raw in raw_actions[:_MAX_ACTIONS_PER_UTTERANCE]:
        action = _coerce_action_dict(
            raw,
            source_text=source_text,
            top_confidence=top_confidence,
            now=now,
            target_resolver=target_resolver,
        )
        if action is not None:
            out.append(action)

    if out and not _validate_action_links(out):
        return None

    return out or None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_actions_with_llm_result(
    post_wake_text: str,
    *,
    now: datetime | None = None,
    target_resolver: TargetResolver | None = None,
) -> LlmActionExtractionResult:
    """Run the registered LLM extractor on post-wake text.

    Returns a detailed result so the pipeline can distinguish an explicit
    model ``non_action`` decision from a provider/schema failure. Never raises.

    The expected model envelope is::

        {
          "intent": "execute_action" | "dictation" | "meta_about_juno" |
                    "question" | "example_or_quote" | "ambiguous",
          "should_execute": bool,
          "confidence": 0.0-1.0,
          "decision_evidence_span": "exact spoken command span",
          "actions": [...]
        }
    """

    if _extractor is None:
        return LlmActionExtractionResult(
            intent=None,
            rejected_reason="no_extractor",
            allow_regex_fallback=True,
        )

    cleaned = (post_wake_text or "").strip()
    if not cleaned:
        return LlmActionExtractionResult(intent=None, rejected_reason="empty_input")

    try:
        envelope = _extractor(cleaned, now)
    except Exception as exc:  # noqa: BLE001 - protect dictation
        logger.exception("LLM action extractor raised for text=%r", cleaned)
        return LlmActionExtractionResult(
            intent=None,
            rejected_reason="extractor_error",
            error=str(exc),
            allow_regex_fallback=True,
        )

    if envelope is None:
        return LlmActionExtractionResult(
            intent=None,
            rejected_reason="no_envelope",
            allow_regex_fallback=True,
        )

    intent = _canonical_intent(envelope)
    if intent in _NON_EXECUTE_INTENTS:
        return LlmActionExtractionResult(intent=intent, rejected_reason="non_action")
    if intent != _EXECUTE_INTENT:
        return LlmActionExtractionResult(intent=intent, rejected_reason="invalid_intent")

    actions = validate_envelope(
        envelope,
        raw_span_fallback=cleaned,
        now=now,
        target_resolver=target_resolver,
    )
    if not actions:
        return LlmActionExtractionResult(
            intent=_EXECUTE_INTENT,
            rejected_reason="validator_rejected",
        )
    return LlmActionExtractionResult(intent=_EXECUTE_INTENT, actions=actions)


def extract_actions_with_llm(
    post_wake_text: str,
    *,
    now: datetime | None = None,
    target_resolver: TargetResolver | None = None,
) -> list[Action] | None:
    """Compatibility wrapper returning only validated actions."""

    result = extract_actions_with_llm_result(
        post_wake_text,
        now=now,
        target_resolver=target_resolver,
    )
    if not result.actions:
        return None
    return result.actions


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "LlmActionExtractor",
    "LlmActionExtractionResult",
    "TargetResolver",
    "set_llm_extractor",
    "get_llm_extractor",
    "extract_actions_with_llm",
    "extract_actions_with_llm_result",
    "expected_envelope_schema",
    "select_actions_schema_version",
    "validate_envelope",
]
