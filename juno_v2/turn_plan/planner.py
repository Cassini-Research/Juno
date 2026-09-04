from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.modes import ModePolicy, ModeSelection
from juno_v2.contracts.writer import WriterMode, WriterTransformRequest
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.turn_plan.validators import span_present

if TYPE_CHECKING:
    from juno_v2.writer.backends.base import WriterBackend

_RENDER_KIND_ALIASES = {
    "paragraph": "paragraphs",
    "paragraph_list": "paragraphs",
    "bullet": "bulleted_list",
    "bullets": "bulleted_list",
    "bullet_list": "bulleted_list",
    "numbered": "numbered_list",
    "ordered_list": "numbered_list",
    "todo": "checklist",
    "todos": "checklist",
    "tasks": "checklist",
    "prompt": "ai_prompt",
}

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}

_LIST_MARKER_RE = re.compile(
    r"\b(?P<marker>"
    r"number\s+(?:"
    + "|".join(sorted((*_NUMBER_WORDS.keys(),), key=len, reverse=True))
    + r"|\d{1,2})|"
    + "|".join(sorted((*_ORDINAL_WORDS.keys(), *_NUMBER_WORDS.keys()), key=len, reverse=True))
    + r"|\d{1,2}(?:st|nd|rd|th)?)\b[\s).,:;-]*",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class TurnPlanPacket:
    utterance_id: str
    final_text: str
    raw_text: str
    context: TypedContextBundle
    mode_policy: ModePolicy | None = None
    mode_selection: ModeSelection | None = None
    memory_store: JsonMemoryStore | None = None
    memory_snapshot: MemorySnapshot | None = None
    memory_packet: dict[str, Any] | None = None
    language_hint: str | None = None
    partial_text: str | None = None
    writer_tone_addon: str | None = None
    wake_verified: bool = False
    now_iso: str | None = None
    allowed_action_kinds: tuple[str, ...] = ("note", "reminder", "alarm")
    max_actions: int = 25
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        snippets = _served_snippet_payload(
            store_snippets=_snippet_payload(self.memory_store, self.context.app_category),
            memory_packet=self.memory_packet,
        )
        replacements = _replacement_payload(self.memory_snapshot)
        return {
            "task": "turn_planning_v1",
            "utterance_id": self.utterance_id,
            "asr": {
                "final_text": self.final_text,
                "raw_text": self.raw_text,
                "language_hint": self.language_hint,
                "partial_text": self.partial_text or "",
            },
            "context": {
                "app_name": self.context.app_name,
                "app_bundle_id": (self.context.metadata or {}).get("app_bundle_id"),
                "app_category": self.context.app_category,
                "window_title": self.context.window_title,
                "selected_text": _bound(self.context.selected_text, 4000),
                "focused_text_before": _bound(self.context.focused_text_before, 1600, tail=True),
                "focused_text_after": _bound(self.context.focused_text_after, 800),
                "field_text_excerpt": _bound(self.context.field_text_excerpt, 2400),
                "candidate_entities": list(self.context.candidate_entities or [])[:48],
                "focused_file_path": self.context.focused_file_path,
                "symbol_under_cursor": self.context.symbol_under_cursor,
                "recent_clipboard": list(self.context.recent_clipboard or [])[:5],
                "secure_field": bool((self.context.metadata or {}).get("focused_is_secure")),
                "target_capabilities": {
                    "markdown_allowed": _markdown_allowed(self.context),
                    "rich_text_allowed": bool((self.context.metadata or {}).get("rich_text_allowed", False)),
                    "app_category": self.context.app_category or "unknown",
                },
            },
            "mode": None if self.mode_policy is None else self.mode_policy.to_dict(),
            "mode_selection": None if self.mode_selection is None else self.mode_selection.to_dict(),
            "writer_tone_addon": self.writer_tone_addon,
            "memory": {
                "packet": self.memory_packet or {},
                "snapshot_counts": _snapshot_counts(self.memory_snapshot),
                "snippets": snippets,
                "replacements": replacements,
            },
            "actions": _action_surface_payload(
                wake_verified=self.wake_verified,
                allowed_action_kinds=self.allowed_action_kinds,
                max_actions=self.max_actions,
                now_iso=self.now_iso,
                permission_state=self.metadata.get("action_permission_state") or "unknown",
            ),
            "output_schema": (
                _turn_plan_schema_hint()
                if self.wake_verified
                else _turn_plan_schema_hint_without_actions()
            ),
        }


@dataclass(slots=True)
class TurnPlanResult:
    plan: dict[str, Any] | None
    status: str
    backend_name: str | None = None
    decode_ms: float = 0.0
    raw_output: str = ""
    errors: list[str] = field(default_factory=list)
    repair_attempted: bool = False
    repair_status: str | None = None
    initial_status: str | None = None
    initial_errors: list[str] = field(default_factory=list)
    validation_errors_before_repair: list[str] = field(default_factory=list)
    validation_warnings_before_repair: list[str] = field(default_factory=list)
    normalization_notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.status == "ok"




def _env_flag(name: str, default: bool) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class TurnPlanner:
    def __init__(self, backend: "WriterBackend") -> None:
        self.backend = backend

    def plan(self, packet: TurnPlanPacket) -> TurnPlanResult:
        payload = packet.to_payload()
        req = WriterTransformRequest(
            utterance_id=packet.utterance_id,
            instruction=(
                "Return exactly one JSON object matching turn_planning_v1. "
                "Plan meaning only; do not execute actions."
            ),
            source_text=packet.final_text,
            mode=_writer_mode(packet),
            context_payload={
                "task": "turn_planning_v1",
                "schema_version": "turn_planning_v1",
                "payload": payload,
            },
            metadata={
                "kind": "turn_planning_v1",
                "max_tokens": _planner_max_tokens(packet.final_text, packet.context.selected_text),
                # Static system prompt is KV-cached like the dictation
                # editor's — action turns paid 15s of cold prefill without it.
                "cache_prefix": _env_flag("JUNO_V2_ACTION_PROMPT_CACHE", True),
            },
        )
        try:
            result = self.backend.rewrite(req)
        except Exception as exc:  # noqa: BLE001
            return TurnPlanResult(plan=None, status="backend_error", errors=[f"{type(exc).__name__}: {exc}"])

        planned = _result_from_backend(result)
        _normalize_result(planned, packet.final_text)
        return planned

    def repair(
        self,
        packet: TurnPlanPacket,
        prior: TurnPlanResult,
        *,
        validation_errors: list[str] | None = None,
        validation_warnings: list[str] | None = None,
    ) -> TurnPlanResult:
        payload = {
            "task": "turn_repair_v1",
            "utterance_id": packet.utterance_id,
            "source_text": packet.final_text,
            "raw_text": packet.raw_text,
            "context": {
                "app_name": packet.context.app_name,
                "app_category": packet.context.app_category,
                "selected_text": _bound(packet.context.selected_text, 1200),
                "markdown_allowed": _markdown_allowed(packet.context),
            },
            "allowed_values": _turn_plan_schema_hint()["allowed_values"],
            "invalid": {
                "status": prior.status,
                "errors": list(prior.errors),
                "validation_errors": list(validation_errors or []),
                "validation_warnings": list(validation_warnings or []),
                "raw_output": _bound(prior.raw_output, 4000),
                "plan": prior.plan if isinstance(prior.plan, dict) else None,
            },
        }
        req = WriterTransformRequest(
            utterance_id=packet.utterance_id,
            instruction=(
                "Repair this failed turn_plan_v1 into one strict JSON object. "
                "Only fix JSON/schema/validation failures; preserve grounded user meaning."
            ),
            source_text=packet.final_text,
            mode=_writer_mode(packet),
            context_payload={
                "task": "turn_repair_v1",
                "schema_version": "turn_repair_v1",
                "payload": payload,
            },
            metadata={
                "kind": "turn_repair_v1",
                "max_tokens": _planner_max_tokens(packet.final_text, packet.context.selected_text),
            },
        )
        try:
            result = self.backend.rewrite(req)
        except Exception as exc:  # noqa: BLE001
            fallback = _fallback_structural_turn_plan(packet.final_text)
            if fallback is not None:
                fallback_result = TurnPlanResult(
                    plan=fallback,
                    status="ok",
                    repair_attempted=True,
                    repair_status="fallback",
                    initial_status=prior.status,
                    initial_errors=list(prior.errors),
                    validation_errors_before_repair=list(validation_errors or []),
                    validation_warnings_before_repair=list(validation_warnings or []),
                    normalization_notes=["structural_render_fallback_from_source"],
                )
                _normalize_result(fallback_result, packet.final_text)
                return fallback_result
            return TurnPlanResult(
                plan=None,
                status="repair_backend_error",
                errors=[f"{type(exc).__name__}: {exc}"],
                repair_attempted=True,
                repair_status="backend_error",
                initial_status=prior.status,
                initial_errors=list(prior.errors),
                validation_errors_before_repair=list(validation_errors or []),
                validation_warnings_before_repair=list(validation_warnings or []),
            )

        repaired = _result_from_backend(result)
        _normalize_result(repaired, packet.final_text)
        if not repaired.ok:
            fallback = _fallback_structural_turn_plan(packet.final_text)
            if fallback is not None:
                fallback_result = TurnPlanResult(
                    plan=fallback,
                    status="ok",
                    backend_name=getattr(result, "backend_name", None),
                    decode_ms=float(getattr(result, "decode_ms", 0.0) or 0.0),
                    raw_output=str(getattr(result, "text", "") or ""),
                    repair_attempted=True,
                    repair_status="fallback",
                    initial_status=prior.status,
                    initial_errors=list(prior.errors),
                    validation_errors_before_repair=list(validation_errors or []),
                    validation_warnings_before_repair=list(validation_warnings or []),
                    normalization_notes=[
                        *repaired.normalization_notes,
                        "structural_render_fallback_from_source",
                    ],
                )
                _normalize_result(fallback_result, packet.final_text)
                return fallback_result
        if not repaired.ok and isinstance(prior.plan, dict):
            # The repair decode produced nothing usable (observed: the model
            # echoed the turn_repair_v1 request back instead of emitting a
            # plan). The prior plan at least parsed — return it so per-action
            # validation and coercion can salvage what is grounded, rather
            # than failing the whole turn (production 2026-06-11).
            return TurnPlanResult(
                plan=prior.plan,
                status="ok",
                backend_name=repaired.backend_name or prior.backend_name,
                decode_ms=prior.decode_ms + repaired.decode_ms,
                raw_output=repaired.raw_output,
                errors=[],
                repair_attempted=True,
                repair_status=f"unusable_repair_kept_initial:{repaired.status}",
                initial_status=prior.status,
                initial_errors=list(prior.errors),
                validation_errors_before_repair=list(validation_errors or []),
                validation_warnings_before_repair=list(validation_warnings or []),
                normalization_notes=[
                    *repaired.normalization_notes,
                    "repair_unusable_initial_plan_restored",
                ],
            )
        repaired.repair_attempted = True
        repaired.repair_status = "ok" if repaired.ok else repaired.status
        repaired.initial_status = prior.status
        repaired.initial_errors = list(prior.errors)
        repaired.validation_errors_before_repair = list(validation_errors or [])
        repaired.validation_warnings_before_repair = list(validation_warnings or [])
        return repaired


def _result_from_backend(result: WriterTransformResult) -> TurnPlanResult:
    raw = str(getattr(result, "text", "") or "")
    obj, parse_notes = _json_object_with_notes(raw)
    if obj is None:
        return TurnPlanResult(
            plan=None,
            status="invalid_json",
            backend_name=getattr(result, "backend_name", None),
            decode_ms=float(getattr(result, "decode_ms", 0.0) or 0.0),
            raw_output=raw,
            errors=["invalid_json"],
        )
    if not _looks_like_turn_plan_object(obj):
        return TurnPlanResult(
            plan=None,
            status="invalid_plan_object",
            backend_name=getattr(result, "backend_name", None),
            decode_ms=float(getattr(result, "decode_ms", 0.0) or 0.0),
            raw_output=raw,
            errors=["invalid_turn_plan_object"],
            normalization_notes=parse_notes,
        )
    obj.setdefault("schema_version", "turn_plan_v1")
    return TurnPlanResult(
        plan=obj,
        status="ok",
        backend_name=getattr(result, "backend_name", None),
        decode_ms=float(getattr(result, "decode_ms", 0.0) or 0.0),
        raw_output=raw,
        normalization_notes=parse_notes,
    )


def _looks_like_turn_plan_object(obj: dict[str, Any]) -> bool:
    if str(obj.get("schema_version") or "").strip() == "turn_plan_v1":
        return True
    if str(obj.get("task") or "").strip() in {"turn_planning_v1", "turn_repair_v1"}:
        return False
    required_shape_keys = {
        "utterance_kind",
        "corrected_transcript",
        "target",
        "render_plan",
        "actions",
        "safety",
    }
    return len(required_shape_keys.intersection(obj.keys())) >= 3


def _fallback_structural_turn_plan(source_text: str) -> dict[str, Any] | None:
    text = str(source_text or "").strip()
    if not text:
        return None
    instruction = _structural_list_instruction(text)
    if instruction is None:
        return None
    tail = text[instruction.end() :].strip(" .,:;-")
    items = _spoken_list_items(tail)
    if not items:
        return None
    render_kind = _structural_render_kind(instruction.group(0))
    claimed_count = _claimed_list_count(instruction.group(0)) or _claimed_list_count(text)
    spoken_count = len(items)
    units = [
        {
            "kind": "item",
            "text": item,
            "source_span": item,
            "order": idx,
        }
        for idx, item in enumerate(items, start=1)
    ]
    uncertainties: list[dict[str, Any]] = []
    if claimed_count is not None and claimed_count != spoken_count:
        uncertainties.append({
            "type": "claimed_count_mismatch",
            "claimed": claimed_count,
            "spoken": spoken_count,
        })
    return {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "format_dictation",
        "corrected_transcript": {
            "text": text,
            "corrections": [],
            "literal_spans": [],
        },
        "target": {"kind": "cursor", "confidence": 0.7},
        "render_plan": {
            "render_kind": render_kind,
            "markdown_allowed": False,
            "claimed_item_count": claimed_count or spoken_count,
            "spoken_item_count": spoken_count,
            "content_units": units,
        },
        "transform": {
            "operation": "none",
            "instruction": "",
            "transformed_text": None,
            "requires_second_pass": False,
        },
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": uncertainties,
    }


def fallback_structural_turn_plan(source_text: str) -> dict[str, Any] | None:
    return _fallback_structural_turn_plan(source_text)


def structural_instruction_present(source_text: str) -> bool:
    """True when the utterance explicitly asks for list/checklist structure.

    Used by the writer service to decide whether a non-wake utterance is
    worth a model turn-plan decode after the deterministic structural
    renderer failed to itemize it.
    """
    return _structural_list_instruction(str(source_text or "")) is not None


def _structural_list_instruction(text: str) -> re.Match[str] | None:
    explicit = re.search(
        r"\b(?:note\s+down|write\s+down|list|make|create|capture|give\s+me|put\s+down)\b"
        r".{0,120}?"
        r"\b(?:points?|items?|steps?|bullets?|bullet\s+points?|checklist|numbered\s+list|list)\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit is not None:
        return explicit
    return re.search(
        r"\b(?:focus\s+on|focused\s+on|cover|talk\s+about|discuss|handle|"
        r"prioriti[sz]e|need\s+to\s+focus\s+on|we\s+need\s+to\s+focus\s+on)\b"
        r".{0,120}?"
        r"\b(?:\d{1,2}|"
        + "|".join(sorted(_NUMBER_WORDS.keys(), key=len, reverse=True))
        + r")\s+(?:things?|points?|items?|steps?|reasons?|priorities|topics?|"
        r"goals?|tasks?|takeaways?|focus\s+areas?)\b",
        text,
        flags=re.IGNORECASE,
    )


def _structural_render_kind(instruction_text: str) -> str:
    key = _term_key(instruction_text)
    if "checklist" in key:
        return "checklist"
    if "bullet" in key and "numbered" not in key:
        return "bulleted_list"
    return "numbered_list"


def _claimed_list_count(text: str) -> int | None:
    match = re.search(
        r"\b(?P<count>\d{1,2}|"
        + "|".join(sorted(_NUMBER_WORDS.keys(), key=len, reverse=True))
        + r")\s+(?:things?|points?|items?|steps?|bullets?|bullet\s+points?|"
        r"reasons?|priorities|topics?|goals?|tasks?|takeaways?|focus\s+areas?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    raw = match.group("count").casefold()
    if raw.isdigit():
        value = int(raw)
    else:
        value = _NUMBER_WORDS.get(raw, 0)
    return value if 0 < value <= 50 else None


def _spoken_list_items(tail: str) -> list[str]:
    text = str(tail or "").strip()
    if not text:
        return []
    accepted: list[tuple[int, re.Match[str]]] = []
    for match in _LIST_MARKER_RE.finditer(text):
        order = _list_marker_order(match.group("marker"))
        if order is None:
            continue
        if not accepted:
            if order != 1:
                continue
        elif order != accepted[-1][0] + 1:
            continue
        accepted.append((order, match))
    if not accepted:
        return []
    # Lazy import: the writer package's __init__ imports the service, which
    # imports this module back. By item-split time everything is initialized.
    from juno_v2.writer.deterministic import clean_spoken_list_item

    items: list[str] = []
    for idx, (_order, marker) in enumerate(accepted):
        start = marker.end()
        end = accepted[idx + 1][1].start() if idx + 1 < len(accepted) else len(text)
        item = clean_spoken_list_item(text[start:end])
        if item:
            items.append(item)
    return items


def _list_marker_order(marker: str) -> int | None:
    value = str(marker or "").casefold().strip()
    if value.startswith("number "):
        value = value.removeprefix("number ").strip()
    if value in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[value]
    if value in _NUMBER_WORDS:
        return _NUMBER_WORDS[value]
    match = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?$", value)
    if match is None:
        return None
    order = int(match.group(1))
    return order if 1 <= order <= 50 else None


def normalize_turn_plan(plan: dict[str, Any], *, source_text: str) -> tuple[dict[str, Any], list[str]]:
    """Coerce useful Qwen output into the strict turn_plan_v1 contract.

    This is a schema boundary, not an utterance handler: the model frequently
    returns semantically useful fields with small shape errors such as
    ``corrected_transcript`` as a string, char-span dictionaries instead of
    source-span strings, or stray native-action objects attached to ordinary
    dictation render plans. The product layer should validate after these
    contract repairs instead of throwing away the whole plan and falling into a
    less capable parser.
    """

    out = deepcopy(plan)
    notes: list[str] = []
    out["schema_version"] = "turn_plan_v1"

    corrected = out.get("corrected_transcript")
    if isinstance(corrected, str):
        out["corrected_transcript"] = {
            "text": corrected.strip(),
            "corrections": [],
            "literal_spans": [],
        }
        notes.append("corrected_transcript_string_to_object")
    elif not isinstance(corrected, dict):
        fallback = str(source_text or "").strip()
        out["corrected_transcript"] = {
            "text": fallback,
            "corrections": [],
            "literal_spans": [],
        }
        notes.append("corrected_transcript_defaulted")

    target = out.get("target")
    if not isinstance(target, dict):
        out["target"] = {"kind": "cursor", "confidence": 0.5}
        notes.append("target_defaulted")

    render = out.get("render_plan")
    if not isinstance(render, dict):
        render = {"render_kind": "plain", "markdown_allowed": False, "content_units": []}
        out["render_plan"] = render
        notes.append("render_plan_defaulted")
    raw_render_kind = str(render.get("render_kind") or "plain").strip()
    render_kind_alias = _RENDER_KIND_ALIASES.get(raw_render_kind.casefold())
    if render_kind_alias:
        render["render_kind"] = render_kind_alias
        notes.append("render_kind_alias_normalized")
    units = render.get("content_units")
    if not isinstance(units, list):
        render["content_units"] = []
        if units is not None:
            notes.append("content_units_defaulted")
    else:
        for unit in units:
            if not isinstance(unit, dict):
                continue
            if _coerce_span_field(unit, "source_span", source_text):
                notes.append("content_unit_source_span_object_to_text")

    transform = out.get("transform")
    if not isinstance(transform, dict):
        out["transform"] = {
            "operation": "none",
            "instruction": "",
            "transformed_text": None,
            "requires_second_pass": False,
        }
        notes.append("transform_defaulted")

    snippets = out.get("snippets")
    if not isinstance(snippets, list):
        out["snippets"] = []
        if snippets is not None:
            notes.append("snippets_defaulted")

    memory_candidates = out.get("memory_candidates")
    if not isinstance(memory_candidates, list):
        out["memory_candidates"] = []
        if memory_candidates is not None:
            notes.append("memory_candidates_defaulted")

    actions = out.get("actions")
    if not isinstance(actions, list):
        actions = []
        out["actions"] = actions
        notes.append("actions_defaulted")
    else:
        normalized_actions: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                notes.append("action_non_object_dropped")
                continue
            action = deepcopy(action)
            if _coerce_span_field(action, "evidence_span", source_text):
                notes.append("action_evidence_span_object_to_text")
            kind = str(action.get("kind") or "").strip()
            body = str(action.get("body") or "").strip()
            evidence = str(action.get("evidence_span") or "").strip()
            if evidence and not span_present(evidence, source_text):
                snapped = _snap_span_to_source(evidence, source_text)
                if snapped:
                    action["evidence_span"] = snapped
                    evidence = snapped
                    notes.append("action_evidence_span_snapped_to_source")
                elif body and span_present(body, source_text):
                    action["evidence_span"] = body
                    notes.append("action_evidence_span_replaced_with_body")
                else:
                    action.pop("evidence_span", None)
                    notes.append("action_ungrounded_evidence_removed")
            elif evidence and body and span_present(body, source_text) and not _span_contains(evidence, body):
                action["evidence_span"] = body
                notes.append("action_evidence_span_replaced_with_body")
            schedule = action.get("schedule")
            if isinstance(schedule, dict):
                if _coerce_span_field(schedule, "source_span", source_text):
                    notes.append("action_schedule_span_object_to_text")
                schedule_span = str(schedule.get("source_span") or "").strip()
                schedule_time = str(schedule.get("time") or "").strip()
                if (
                    schedule_span
                    and not _schedule_span_looks_temporal(schedule_span)
                    and schedule_time
                    and span_present(schedule_time, source_text)
                    and _schedule_span_looks_temporal(schedule_time)
                ):
                    schedule["source_span"] = schedule_time
                    schedule_span = schedule_time
                    notes.append("action_schedule_span_replaced_with_time")
                elif schedule_span and not _schedule_span_looks_temporal(schedule_span):
                    temporal_candidate = _first_temporal_candidate(
                        " ".join(
                            part
                            for part in (
                                body,
                                str(action.get("evidence_span") or action.get("raw_span") or ""),
                                source_text,
                            )
                            if part
                        )
                    )
                    if temporal_candidate and span_present(temporal_candidate, source_text):
                        schedule["source_span"] = temporal_candidate
                        schedule_span = temporal_candidate
                        notes.append("action_schedule_span_replaced_with_temporal_candidate")
                if schedule_span and not span_present(schedule_span, source_text):
                    schedule.pop("source_span", None)
                    notes.append("action_ungrounded_schedule_span_removed")
                elif schedule_span and not _schedule_span_looks_temporal(schedule_span):
                    schedule.pop("source_span", None)
                    notes.append("action_non_temporal_schedule_span_removed")
                elif kind in {"reminder", "alarm"}:
                    if schedule_span and body and _span_contains(body, schedule_span):
                        cleaned_body = _remove_surface_phrase(body, schedule_span)
                        cleaned_body = _strip_trailing_schedule_clause(cleaned_body)
                        if cleaned_body:
                            action["body"] = cleaned_body
                            notes.append("action_schedule_removed_from_body")
                    elif body:
                        cleaned_body = _strip_trailing_schedule_clause(body)
                        if cleaned_body and cleaned_body != body:
                            action["body"] = cleaned_body
                            notes.append("action_schedule_removed_from_body")
            repaired_body = _repair_action_body_from_evidence(action, source_text=source_text)
            if repaired_body:
                action["body"] = repaired_body
                notes.append("action_body_repaired_from_evidence")
            if kind == "note":
                sliced = _note_body_source_slice(source_text)
                current_body = str(action.get("body") or "")
                # The model re-types note bodies and truncates long ones
                # (production: 600-char note shipped as 79 chars). When the
                # deterministic source slice is substantially longer, the
                # spoken words win over the model's copy.
                if sliced and len(sliced) > max(40, int(len(current_body) * 1.4)):
                    action["body"] = sliced
                    notes.append("note_body_sliced_from_source")
            if _clear_satisfied_action_missing_fields(action):
                notes.append("action_missing_fields_cleared")
            normalized_actions.append(action)
        # Compound batches: the spoken anchors define which words belong to
        # which action. Re-ground every body / schedule / evidence span
        # inside its own segment, or rebuild outright on misalignment.
        if normalized_actions:
            segments = segment_native_actions(source_text)
            if len(segments) >= 2:
                normalized_actions, seg_notes = _realign_actions_to_segments(
                    normalized_actions, segments, source_text=source_text
                )
                notes.extend(seg_notes)
            normalized_actions, ordinal_notes = _expand_missing_ordinal_actions(
                normalized_actions, source_text=source_text
            )
            notes.extend(ordinal_notes)
        out["actions"] = normalized_actions
        actions = normalized_actions

    if actions and _repair_render_to_write_clause(render, actions=actions, source_text=source_text):
        notes.append("write_clause_render_repaired_from_source")

    if actions and _lift_write_clause_note_action_to_render(actions, render=render, source_text=source_text):
        out["actions"] = actions
        notes.append("write_clause_note_lifted_to_render")

    pre_resolution_kind = str(out.get("utterance_kind") or "").strip()
    allow_render_action_resolution = pre_resolution_kind in {"command", "actions", "mixed"} or (
        pre_resolution_kind == "dictation" and _native_action_requested(source_text)
    )
    if allow_render_action_resolution and actions and _render_has_content(render):
        if _trim_render_to_write_clause(render, source_text=source_text):
            notes.append("mixed_render_trimmed_to_write_clause")
        filtered_actions, dropped = _drop_render_only_note_actions(
            actions,
            render=render,
            source_text=source_text,
        )
        if dropped:
            out["actions"] = filtered_actions
            actions = filtered_actions
            notes.append("render_only_note_actions_dropped")
        if actions and _render_duplicates_actions(render, actions=actions):
            render["render_kind"] = "none"
            render["content_units"] = []
            notes.append("action_duplicate_render_collapsed")

    if actions and _collapse_action_only_render(render, actions=actions, source_text=source_text):
        render["render_kind"] = "none"
        render["content_units"] = []
        notes.append("action_command_render_collapsed")

    kind = str(out.get("utterance_kind") or "").strip()
    render_kind = str(render.get("render_kind") or "plain").strip()
    render_has_content = _render_has_content(render)
    if kind == "command":
        out["utterance_kind"] = "actions" if actions and not render_has_content else ("mixed" if actions else "dictation")
        notes.append("command_kind_mapped")
    elif not kind:
        out["utterance_kind"] = "actions" if actions and not render_has_content else "dictation"
        notes.append("utterance_kind_defaulted")
    elif kind == "mixed" and actions and not render_has_content:
        out["utterance_kind"] = "actions"
        notes.append("mixed_action_only_mapped_to_actions")
    elif kind == "mixed" and not actions:
        out["utterance_kind"] = "format_dictation" if render_kind in {"bulleted_list", "numbered_list", "checklist", "table"} else "dictation"
        notes.append("mixed_without_actions_mapped")
    elif kind == "actions" and actions and render_has_content:
        out["utterance_kind"] = "mixed"
        notes.append("actions_with_render_mapped_to_mixed")
    elif kind == "dictation" and actions and not render_has_content and _native_action_requested(source_text):
        out["utterance_kind"] = "actions"
        notes.append(
            "dictation_native_note_mapped_to_actions"
            if _native_note_requested(source_text)
            else "dictation_native_action_mapped_to_actions"
        )
    elif kind == "dictation" and render_kind in {"bulleted_list", "numbered_list", "checklist", "table"}:
        out["utterance_kind"] = "format_dictation"
        notes.append("dictation_structural_render_mapped")

    kind = str(out.get("utterance_kind") or "").strip()
    if kind in {"dictation", "format_dictation"} and actions:
        out["actions"] = []
        actions = []
        notes.append("dictation_actions_dropped")

    safety = out.get("safety")
    if not isinstance(safety, dict):
        safety = {}
        out["safety"] = safety
        notes.append("safety_defaulted")
    if "commit_policy" not in safety:
        safety["commit_policy"] = "commit" if kind not in {"actions", "no_op"} else "no_commit"
        notes.append("commit_policy_defaulted")
    if "execute_policy" not in safety:
        safety["execute_policy"] = "execute" if out.get("actions") else "no_execute"
        notes.append("execute_policy_defaulted")
    elif not out.get("actions") and str(safety.get("execute_policy") or "").strip() in {"execute", "confirm"}:
        safety["execute_policy"] = "no_execute"
        notes.append("execute_policy_cleared_without_actions")

    uncertainties = out.get("uncertainties")
    if not isinstance(uncertainties, list):
        out["uncertainties"] = []
        if uncertainties is not None:
            notes.append("uncertainties_defaulted")

    return out, _dedupe(notes)


def _normalize_result(result: TurnPlanResult, source_text: str) -> None:
    if not isinstance(result.plan, dict):
        return
    normalized, notes = normalize_turn_plan(result.plan, source_text=source_text)
    result.plan = normalized
    result.normalization_notes.extend(notes)


def _coerce_span_field(obj: dict[str, Any], key: str, source_text: str) -> bool:
    raw = obj.get(key)
    if isinstance(raw, dict):
        span = _span_from_offsets(raw, source_text)
        if span:
            obj[key] = span
        else:
            obj.pop(key, None)
        return True
    if isinstance(raw, (list, tuple)) and len(raw) >= 2 and not isinstance(raw[0], str):
        span = _span_from_offsets({"start": raw[0], "end": raw[1]}, source_text)
        if span:
            obj[key] = span
        else:
            obj.pop(key, None)
        return True
    return False


def _span_from_offsets(raw: dict[str, Any], source_text: str) -> str:
    try:
        start = int(raw.get("start"))
        end = int(raw.get("end"))
    except (TypeError, ValueError):
        return ""
    text = source_text or ""
    if start < 0 or end <= start or end > len(text):
        return ""
    return text[start:end].strip()


def _render_has_content(render: dict[str, Any]) -> bool:
    units = render.get("content_units")
    if not isinstance(units, list):
        return False
    return any(isinstance(unit, dict) and str(unit.get("text") or "").strip() for unit in units)


_ACTION_ANCHOR_RE = re.compile(
    r"""
    (?:\b(?:and|then|also)\s+)*
    (?P<verb>
        (?:take|create|make|add|save|write)\s+(?:a\s+|another\s+|new\s+)?note\b
      | remind\s+me\b
      | set\s+(?:a\s+)?reminder\b
      | (?:add|create)\s+a?\s*reminder\b
      | (?:set|add|create)\s+(?:an?\s+)?alarm\b
      | set\s+a\s+timer\b
      | wake\s+me\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def segment_native_actions(source_text: str) -> list[dict[str, Any]]:
    """Deterministic segmentation of a compound native-action utterance.

    Each segment runs from one native action verb to the next anchor.
    When two or more anchors exist, the segments — not the model's re-typed
    spans — are the authority for which words belong to which action
    (production 2026-06-11: six-action batch shipped sibling-bled bodies
    and a reminder that borrowed another sibling's time).
    """
    text = str(source_text or "")
    matches = list(_ACTION_ANCHOR_RE.finditer(text))
    segments: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        verb = re.sub(r"\s+", " ", m.group("verb").lower())
        if "note" in verb:
            kind = "note"
        elif "alarm" in verb or "timer" in verb or "wake" in verb:
            kind = "alarm"
        else:
            kind = "reminder"
        start = m.start("verb")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg_text = text[start:end].strip(" ,.;")
        seg_text = re.sub(r"\s+(?:and|then|also)$", "", seg_text, flags=re.IGNORECASE).strip(" ,.;")
        segments.append({"kind": kind, "text": seg_text})
    return segments


_SEGMENT_CLOCK = r"\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)"
_SEGMENT_DAY = (
    r"(?:(?:next|this|coming)\s+)?"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|today|tomorrow|tonight|day\s+after\s+tomorrow)"
)
# Word boundaries throughout ("within 5 minutes" must not read as
# "in 5 minutes"), and BOTH orders of clock and day: speech says
# "9pm today" as often as "today at 9pm" (production 2026-06-12: the
# day-first-only pattern reduced "Darpan 9pm today" to "today", which
# parse_when default-filled to 9 AM and rolled to the wrong morning).
_SEGMENT_TEMPORAL_RE = re.compile(
    rf"""
    \b{_SEGMENT_DAY}(?:\s+(?:at\s+)?{_SEGMENT_CLOCK})?\b
    | \b(?:at\s+)?{_SEGMENT_CLOCK}\s+{_SEGMENT_DAY}\b
    | \bin\s+(?:\d+|a|an|half)\s+(?:minutes?|mins?|hours?|hrs?)\b
    | \b(?:at\s+)?{_SEGMENT_CLOCK}\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SEGMENT_CLOCK_RE = re.compile(rf"\b{_SEGMENT_CLOCK}\b", re.IGNORECASE)
_SEGMENT_DAY_RE = re.compile(rf"\b{_SEGMENT_DAY}\b", re.IGNORECASE)


def _segment_temporal_clause(seg_text: str) -> str:
    """Best temporal clause inside one action segment.

    Prefer candidates carrying a clock over bare day words (a clause with
    "9pm" beats a longer bare "tomorrow"), then day+clock together, then
    longer text.
    """
    best = ""
    best_score = (-1, -1, -1)
    for m in _SEGMENT_TEMPORAL_RE.finditer(seg_text or ""):
        candidate = m.group(0).strip()
        has_clock = 1 if _SEGMENT_CLOCK_RE.search(candidate) else 0
        has_day = 1 if _SEGMENT_DAY_RE.search(candidate) else 0
        score = (has_clock, has_clock + has_day, len(candidate))
        if score > best_score:
            best_score = score
            best = candidate
    return re.sub(r"^at\s+", "", best, flags=re.IGNORECASE)


_COMPOUND_DECLARATION_RE = re.compile(
    r"\b(?:set\s+up|set|create|add|make)\s+"
    r"(?:two|three|four|five|six|2|3|4|5|6)\s+"
    r"(?P<kind>alarms|reminders)\b",
    re.IGNORECASE,
)
_ORDINAL_ANCHOR_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|1st|2nd|3rd|4th|5th|6th)\b[,.\s]+",
    re.IGNORECASE,
)


def _expand_missing_ordinal_actions(
    actions: list[dict[str, Any]],
    *,
    source_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Synthesize ordinal-clause actions the model failed to emit.

    "Set up two alarms. First, to call my wife at 10 pm today. Second, …"
    carries no action verb inside the ordinal clauses, and the 4B planner
    regularly under-emits or kind-confuses them (production 2026-06-12,
    utterance macshell-26819FF2: two spoken alarms produced zero alarm
    actions). When the source declares a compound alarm/reminder batch AND
    ordinal clauses with their own grounded temporal spans exist AND no
    existing action already claims a clause's temporal span, add a
    deterministic action derived from the clause. Existing model actions
    are never modified or removed, and every synthesized action still
    passes the per-action coercion guards (grounding, timeless-alarm skip)
    downstream.
    """
    notes: list[str] = []
    text = str(source_text or "")
    declaration = _COMPOUND_DECLARATION_RE.search(text)
    if declaration is None:
        return actions, notes
    kind = "alarm" if declaration.group("kind").lower().startswith("alarm") else "reminder"

    anchors = list(_ORDINAL_ANCHOR_RE.finditer(text, declaration.end()))
    if len(anchors) < 2:
        return actions, notes

    claimed_spans = {
        _norm_span(str((a.get("schedule") or {}).get("source_span") or ""))
        for a in actions
        if isinstance(a, dict)
    }
    claimed_spans.discard("")

    expanded = list(actions)
    added = 0
    for i, anchor in enumerate(anchors[:6]):
        clause_end = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
        clause = text[anchor.end():clause_end]
        # A clause ends at its own sentence; trailing sentences belong to
        # other intents ("…at 9 pm today. Please add a note…").
        clause = re.split(r"[.!?]", clause, maxsplit=1)[0].strip(" ,;")
        if not clause:
            continue
        temporal = _segment_temporal_clause(clause)
        if not temporal or not _SEGMENT_CLOCK_RE.search(temporal):
            # Only synthesize when the clause carries an explicit clock —
            # bare day words default-fill and create wrong-time actions.
            continue
        if _norm_span(temporal) in claimed_spans:
            continue
        action: dict[str, Any] = {
            "kind": kind,
            "operation": "create",
            "body": "",
            "evidence_span": clause,
            "schedule": {"kind": "instant", "source_span": temporal},
            "missing_fields": [],
        }
        body = _action_body_candidate_from_text(kind, clause, action, source_text=source_text)
        body = re.sub(r"^to\s+", "", str(body or clause).strip(), flags=re.IGNORECASE)
        body = re.sub(
            rf"\s*(?:at\s+)?{re.escape(temporal)}\s*$", "", body, flags=re.IGNORECASE
        ).strip(" ,;")
        action["body"] = body or clause
        claimed_spans.add(_norm_span(temporal))
        expanded.append(action)
        added += 1
    if added:
        notes.append("actions_expanded_from_ordinals")
    return expanded, notes


def _norm_span(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _realign_actions_to_segments(
    actions: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    source_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-segment grounding for compound batches.

    Positionally aligned actions get their body / schedule span / evidence
    re-grounded inside their own segment. On count or kind mismatch the
    model batch is kept untouched: the verb scanner only sees clauses with
    explicit action verbs, so a mismatch usually means the SCANNER is the
    one missing intents, not the model. Rebuilding from segments here
    deleted four correctly-timed alarms whose ordinal clauses ("First, to
    call…") carry no verb anchor (production 2026-06-12, utterance
    macshell-12A45392: model emitted 6 actions, scanner saw 2, rebuild
    shipped only a note and a wrong-time reminder). The model batch has
    already passed plan validation and still faces per-action coercion
    guards (grounding, timeless-alarm skip) downstream.
    """
    notes: list[str] = []
    aligned = (
        len(actions) == len(segments)
        and all(
            str(a.get("kind") or "") == s["kind"] for a, s in zip(actions, segments)
        )
    )
    if not aligned:
        notes.append("segment_realign_skipped_misaligned")
        return actions, notes

    for action, seg in zip(actions, segments):
        seg_text = seg["text"]
        kind = str(action.get("kind") or "")
        schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else None
        if kind in {"reminder", "alarm"}:
            seg_temporal = _segment_temporal_clause(seg_text)
            span = str((schedule or {}).get("source_span") or "").strip()
            if span and not span_present(span, seg_text):
                # A schedule span outside this segment is a sibling's time.
                if seg_temporal:
                    schedule["source_span"] = seg_temporal
                    notes.append("action_schedule_regrounded_to_segment")
                else:
                    schedule.pop("source_span", None)
                    notes.append("action_sibling_schedule_dropped")
            elif not span and seg_temporal:
                if schedule is None:
                    schedule = {"kind": "instant"}
                    action["schedule"] = schedule
                schedule["source_span"] = seg_temporal
                notes.append("action_schedule_filled_from_segment")
            # The segment is authoritative for reminder/alarm bodies too —
            # deterministic derivation strips the verb + schedule clause,
            # so bodies can never carry "Friday at 2pm to …" or a sibling.
            candidate = _action_body_candidate_from_text(kind, seg_text, action, source_text=source_text)
            if candidate and candidate != str(action.get("body") or "").strip():
                action["body"] = candidate
                notes.append("action_body_derived_from_segment")
        else:
            body = str(action.get("body") or "").strip()
            if not body or not span_present(body, seg_text):
                candidate = _action_body_candidate_from_text(kind, seg_text, action, source_text=source_text)
                if candidate:
                    action["body"] = candidate
                    notes.append("action_body_regrounded_to_segment")
        evidence = str(action.get("evidence_span") or "").strip()
        if not evidence or not span_present(evidence, seg_text):
            action["evidence_span"] = seg_text
            notes.append("action_evidence_set_to_segment")
    return actions, notes


def _snap_span_to_source(span: str, source_text: str, *, max_trim_per_side: int = 2) -> str:
    """Recover a grounded span from a model-retyped one by trimming edge tokens.

    The model re-types spans and commonly truncates or pads an edge token
    ("…and w" for "…and write", "e titled …" for "note titled …"). Rather
    than discarding the whole span — which loses schedule text downstream —
    trim up to ``max_trim_per_side`` tokens from either edge and keep the
    first token-boundary-grounded remainder. End trims are preferred because
    tail truncation is the common failure.
    """
    tokens = str(span or "").split()
    if len(tokens) < 3:
        return ""
    trims = sorted(
        (
            (start, end)
            for start in range(max_trim_per_side + 1)
            for end in range(max_trim_per_side + 1)
            if start + end > 0
        ),
        key=lambda pair: (pair[0] + pair[1], pair[0]),
    )
    for start, end in trims:
        remaining = tokens[start : len(tokens) - end if end else None]
        if len(remaining) < 2:
            continue
        candidate = " ".join(remaining)
        if span_present(candidate, source_text):
            return candidate
    return ""


def _span_contains(container: str, content: str) -> bool:
    haystack = _term_key(container)
    needle = _term_key(content)
    if not needle:
        return False
    # Token-boundary containment — character-level substring grounds
    # mid-word fragments of model-retyped spans (see span_present).
    return f" {needle} " in f" {haystack} "


def _trim_render_to_write_clause(render: dict[str, Any], *, source_text: str) -> bool:
    tail = _write_clause_tail(source_text)
    if not tail:
        return False
    render_text = _render_text(render)
    if not render_text:
        return False
    tail_key = _term_key(tail)
    render_key = _term_key(render_text)
    if not tail_key or tail_key == render_key:
        return False
    if tail_key not in render_key:
        return False
    units = render.get("content_units")
    if not isinstance(units, list):
        return False
    render["content_units"] = [
        {
            "kind": "paragraph",
            "text": tail.strip(" ."),
            "source_span": tail.strip(" ."),
            "order": 1,
        }
    ]
    if str(render.get("render_kind") or "").strip() in {"", "none", "note", "paragraphs"}:
        render["render_kind"] = "plain"
    return True


def _write_clause_tail(source_text: str) -> str:
    text = str(source_text or "").strip()
    if not text:
        return ""
    matches = list(
        re.finditer(
            r"\b(?:and\s+)?(?:write|type|insert|dictate|paste)\s+(?P<body>.+)$",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        return ""
    body = matches[-1].group("body").strip()
    body = re.sub(r"\s+", " ", body).strip(" .,:;")
    return body


def _drop_render_only_note_actions(
    actions: list[dict[str, Any]],
    *,
    render: dict[str, Any],
    source_text: str,
) -> tuple[list[dict[str, Any]], int]:
    render_key = _term_key(_render_text(render))
    if not render_key:
        return actions, 0
    native_note_requested = _native_note_requested(source_text)
    write_tail = _write_clause_tail(source_text)
    out: list[dict[str, Any]] = []
    dropped = 0
    for action in actions:
        kind = str(action.get("kind") or "").strip()
        body = str(action.get("body") or "").strip()
        if (
            kind == "note"
            and (
                (write_tail and not native_note_requested)
                or (
                    body
                    and _term_key(body) in render_key
                    and (write_tail or not native_note_requested)
                )
            )
        ):
            dropped += 1
            continue
        out.append(action)
    return out, dropped


def _lift_write_clause_note_action_to_render(
    actions: list[dict[str, Any]],
    *,
    render: dict[str, Any],
    source_text: str,
) -> bool:
    write_tail = _write_clause_tail(source_text)
    if (
        not write_tail
        or _native_note_requested(source_text)
        or not _native_non_note_action_requested(source_text)
        or _render_has_content(render)
    ):
        return False
    write_key = _term_key(write_tail)
    if not write_key:
        return False
    kept: list[dict[str, Any]] = []
    dropped = 0
    for action in actions:
        if not isinstance(action, dict):
            kept.append(action)
            continue
        kind = str(action.get("kind") or "").strip()
        body_key = _term_key(str(action.get("body") or ""))
        evidence_key = _term_key(str(action.get("evidence_span") or action.get("raw_span") or ""))
        if kind == "note" and (body_key == write_key or evidence_key == write_key):
            dropped += 1
            continue
        kept.append(action)
    if not dropped:
        return False
    actions[:] = kept
    if not _render_has_content(render):
        tail = write_tail.strip(" .")
        render["render_kind"] = "plain"
        render["content_units"] = [
            {
                "kind": "paragraph",
                "text": tail,
                "source_span": tail,
                "order": 1,
            }
        ]
    return True


def _repair_render_to_write_clause(
    render: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
    source_text: str,
) -> bool:
    write_tail = _write_clause_tail(source_text)
    if (
        not write_tail
        or _native_note_requested(source_text)
        or not _native_non_note_action_requested(source_text)
        or not _render_has_content(render)
    ):
        return False
    if not any(str(action.get("kind") or "").strip() in {"reminder", "alarm"} for action in actions):
        return False
    render_text = _render_text(render)
    if not render_text:
        return False
    if span_present(render_text, source_text):
        return False
    tail = write_tail.strip(" .")
    if not tail or not span_present(tail, source_text):
        return False
    render["render_kind"] = "plain"
    render["content_units"] = [
        {
            "kind": "paragraph",
            "text": tail,
            "source_span": tail,
            "order": 1,
        }
    ]
    return True


def _collapse_action_only_render(
    render: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
    source_text: str,
) -> bool:
    if _write_clause_tail(source_text) or not _native_action_requested(source_text):
        return False
    if not _render_has_content(render):
        return False
    if _render_duplicates_actions(render, actions=actions):
        return True
    if _render_is_action_command_surface(render):
        return True
    render_text = _render_text(render)
    if render_text and span_present(render_text, source_text) and _action_command_residual(render_text, actions=actions) == "":
        return True
    return bool(render_text and span_present(render_text, source_text) and _action_command_residual(source_text, actions=actions) == "")


def _render_is_action_command_surface(render: dict[str, Any]) -> bool:
    units = render.get("content_units")
    if not isinstance(units, list):
        return False
    texts = [
        str(unit.get("text") or "").strip()
        for unit in units
        if isinstance(unit, dict) and str(unit.get("text") or "").strip()
    ]
    if not texts:
        return False
    return all(_native_action_requested(text) and not _write_clause_tail(text) for text in texts)


def _render_duplicates_actions(render: dict[str, Any], *, actions: list[dict[str, Any]]) -> bool:
    residual = _term_key(_render_text(render))
    if not residual:
        return False
    original = residual
    for action in actions:
        if not isinstance(action, dict):
            continue
        for value in (
            action.get("body"),
            action.get("evidence_span"),
            action.get("raw_span"),
        ):
            residual = _remove_key_phrase(residual, str(value or ""))
        schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
        residual = _remove_key_phrase(residual, str(schedule.get("source_span") or ""))
    residual = " ".join(word for word in residual.split() if word not in _ACTION_RENDER_STOPWORDS)
    if not residual:
        return True
    # Rendering only the native-action command itself is still a duplicate,
    # even when the model included a small grammatical bridge around it.
    return original != residual and all(word in _ACTION_RENDER_STOPWORDS for word in residual.split())


def _action_command_residual(render_text: str, *, actions: list[dict[str, Any]]) -> str:
    residual = _term_key(render_text)
    for action in actions:
        if not isinstance(action, dict):
            continue
        for value in (
            action.get("body"),
            action.get("evidence_span"),
            action.get("raw_span"),
        ):
            residual = _remove_key_phrase(residual, str(value or ""))
        schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
        residual = _remove_key_phrase(residual, str(schedule.get("source_span") or ""))
    residual = _strip_residual_clock_tokens(residual)
    return " ".join(word for word in residual.split() if word not in _ACTION_RENDER_STOPWORDS)


def _strip_residual_clock_tokens(text_key: str) -> str:
    value = str(text_key or "")
    value = re.sub(r"\b\d{1,2}(?:\d{2})?(?:am|pm)?\b", " ", value)
    value = re.sub(r"\b(?:am|pm|a m|p m)\b", " ", value)
    return " ".join(value.split())


_ACTION_RENDER_STOPWORDS = {
    "a",
    "alarm",
    "am",
    "an",
    "and",
    "at",
    "create",
    "for",
    "juno",
    "make",
    "me",
    "note",
    "pm",
    "remind",
    "reminder",
    "save",
    "set",
    "take",
    "the",
    "to",
}


_NOTE_INSTRUCTION_HEAD_RE = re.compile(
    r"^(?:hey\s+)?(?:juno[,.\s]+)?(?:please\s+)?(?:take|create|make|add|save|write)\s+"
    r"(?:a\s+|another\s+|new\s+)?note(?:\s+(?:that|with|called|titled))?\s*[:.,\-]?\s*",
    re.IGNORECASE,
)
_NEXT_ACTION_CLAUSE_RE = re.compile(
    r"[,.;]?\s*\b(?:and\s+|then\s+|also\s+|and\s+then\s+|and\s+also\s+)?"
    r"(?:set\s+an?\s+alarm|remind\s+me|set\s+a\s+reminder|add\s+a?\s*reminder|"
    r"create\s+a?\s*reminder|set\s+a\s+timer|wake\s+me)\b",
    re.IGNORECASE,
)


def _note_body_source_slice(source_text: str) -> str:
    """Note body straight from the spoken words: instruction end → next
    action clause (or end of utterance). The model never re-types content."""
    text = str(source_text or "")
    head = _NOTE_INSTRUCTION_HEAD_RE.match(text)
    if head is None:
        return ""
    start = head.end()
    tail = text[start:]
    nxt = _NEXT_ACTION_CLAUSE_RE.search(tail)
    body = tail[: nxt.start()] if nxt else tail
    return body.strip(" ,.;:-")


def _native_note_requested(source_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:take|make|create|save|add)\s+(?:a\s+)?note\b|\bnote\s+to\s+self\b",
            source_text or "",
            flags=re.IGNORECASE,
        )
    )


def _native_action_requested(source_text: str) -> bool:
    text = source_text or ""
    if _native_note_requested(text):
        return True
    return _native_non_note_action_requested(text)


def _native_non_note_action_requested(source_text: str) -> bool:
    text = source_text or ""
    return bool(
        re.search(
            r"\b(?:remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder|"
            r"create\s+(?:a\s+)?reminder|set\s+(?:an?\s+)?alarm|"
            r"create\s+(?:an?\s+)?alarm|add\s+(?:an?\s+)?alarm|alarm\s+for)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _repair_action_body_from_evidence(action: dict[str, Any], *, source_text: str) -> str:
    kind = str(action.get("kind") or "").strip()
    if kind not in {"note", "reminder", "alarm"}:
        return ""
    body = str(action.get("body") or "").strip()
    evidence = str(action.get("evidence_span") or action.get("raw_span") or "").strip()
    candidate_sources: list[str] = []
    if kind in {"reminder", "alarm"} and _source_starts_with_action_kind(source_text, kind):
        candidate_sources.append(source_text)
    if evidence and span_present(evidence, source_text):
        candidate_sources.append(evidence)
    if kind in {"reminder", "alarm"}:
        schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
        segment = _native_action_segment_for_kind(
            source_text,
            kind,
            anchor=str(schedule.get("source_span") or action.get("when_text") or body or ""),
        )
        if segment:
            candidate_sources.append(segment)
    if not candidate_sources:
        return ""
    body_grounded = bool(body and span_present(body, source_text))
    if kind == "note" and body_grounded and any(
        _native_non_note_action_requested(candidate_source) for candidate_source in candidate_sources
    ):
        return ""
    body_key = _term_key(body)
    body_token_count = len(body_key.split())
    for candidate_source in candidate_sources:
        candidate = _action_body_candidate_from_text(kind, candidate_source, action, source_text=source_text)
        if not candidate or not span_present(candidate, source_text):
            continue
        candidate_key = _term_key(candidate)
        if body_grounded:
            if candidate_key == body_key:
                continue
            if body_key and body_key in candidate_key:
                if len(candidate_key.split()) <= body_token_count:
                    continue
            elif not (
                kind in {"reminder", "alarm"}
                and _action_body_looks_schedule_only(action, body=body)
            ) and not _action_body_boundary_reduction(action, body=body, candidate=candidate):
                continue
        return candidate
    return ""


def _action_body_candidate_from_text(kind: str, text: str, action: dict[str, Any], *, source_text: str) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return ""
    if kind in {"reminder", "alarm"}:
        candidate = _strip_action_write_tail(candidate)
    if kind == "note":
        # The instruction tail may carry ASR-attached punctuation
        # ("take a note titled, What is …"); consume the connector word and
        # its punctuation so the body never starts with "titled," / "called:".
        candidate = re.sub(
            r"^(?:juno\s+)?(?:take|create|make|add|save|write)\s+(?:a\s+)?note"
            r"(?:\s+(?:that|with|called|titled))?\s*[:.,\-]?\s*(?=\S)",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
    elif kind == "reminder":
        candidate = re.sub(
            r"^(?:juno\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder|"
            r"create\s+(?:a\s+)?reminder)(?:\s+to)?\s*[:.\-]?\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
    elif kind == "alarm":
        candidate = re.sub(
            r"^(?:juno\s+)?(?:set|create|add)\s+(?:an?\s+)?alarm(?:\s+(?:for|at))?\s*[:.\-]?\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
    if kind in {"reminder", "alarm"}:
        candidate = _strip_subsequent_native_action_clause(candidate)
    schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
    schedule_span = str(schedule.get("source_span") or action.get("when_text") or "").strip()
    if schedule_span and kind in {"reminder", "alarm"}:
        candidate = _strip_schedule_write_boundary(candidate, schedule_span=schedule_span, source_text=source_text)
        candidate = _remove_surface_phrase(candidate, schedule_span)
    if kind in {"reminder", "alarm"}:
        candidate = re.sub(r"^(?:to|for|about|that)\b[\s:,-]*", "", candidate, flags=re.IGNORECASE)
    candidate = _strip_trailing_schedule_clause(candidate)
    candidate = _strip_trailing_action_connector(candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")
    return "" if _term_key(candidate) in {"", "alarm", "reminder"} else candidate


def _strip_action_write_tail(text: str) -> str:
    return re.sub(
        r"\s+\band\s+(?:write|type|insert|dictate|paste)\b.+$",
        "",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")


def _strip_subsequent_native_action_clause(text: str) -> str:
    cleaned = re.sub(
        r"\s+(?:and|then|plus|next|,|;)\s+"
        r"(?:remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder|create\s+(?:a\s+)?reminder|"
        r"set\s+(?:an?\s+)?alarm|create\s+(?:an?\s+)?alarm|add\s+(?:an?\s+)?alarm|"
        r"take\s+(?:a\s+)?note|make\s+(?:a\s+)?note|create\s+(?:a\s+)?note|save\s+(?:a\s+)?note)\b.*$",
        "",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    return _strip_trailing_action_connector(cleaned)


def _strip_trailing_action_connector(text: str) -> str:
    return re.sub(r"\b(?:and|then|plus|next)$", "", str(text or "").strip(), flags=re.IGNORECASE).strip(" ,.;:-")


def _strip_schedule_write_boundary(text: str, *, schedule_span: str, source_text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip(" ,.;:-"))
    span = re.sub(r"\s+", " ", str(schedule_span or "").strip(" ,.;:-"))
    if not value or not span:
        return value
    pattern = re.compile(
        rf"^(?P<head>.+?)\s+(?:at|by|for)?\s*{re.escape(span)}\b(?P<tail>.*)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(value)
    if not match:
        return value
    tail = re.sub(r"\s+", " ", match.group("tail") or "").strip(" ,.;:-")
    if not tail:
        return match.group("head").strip(" ,.;:-")
    if _is_source_write_boundary_tail(tail, schedule_span=schedule_span, source_text=source_text):
        return match.group("head").strip(" ,.;:-")
    return value


def _is_source_write_boundary_tail(tail: str, *, schedule_span: str, source_text: str) -> bool:
    tail_key = _term_key(tail)
    if not tail_key:
        return False
    span = str(schedule_span or "").strip()
    source = str(source_text or "")
    if not span or not source:
        return False
    matches = list(re.finditer(rf"\b{re.escape(span)}\b", source, flags=re.IGNORECASE))
    for match in matches:
        suffix = re.sub(r"\s+", " ", source[match.end() :]).strip(" ,.;:-")
        if not suffix:
            continue
        boundary = re.match(
            r"^(?:and|then|,)\s+(?:write|type|insert|dictate|paste)\b.*$",
            suffix,
            flags=re.IGNORECASE,
        )
        if not boundary:
            continue
        suffix_key = _term_key(suffix)
        if suffix_key.startswith(tail_key) or tail_key.startswith(suffix_key):
            return True
    return False


def _action_body_boundary_reduction(action: dict[str, Any], *, body: str, candidate: str) -> bool:
    body_key = _term_key(body)
    candidate_key = _term_key(candidate)
    if not body_key or not candidate_key or candidate_key not in body_key:
        return False
    schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
    schedule_span = str(schedule.get("source_span") or action.get("when_text") or "").strip()
    schedule_key = _term_key(schedule_span)
    if schedule_key and schedule_key in body_key and schedule_key not in candidate_key:
        return True
    return False


def _action_body_looks_schedule_only(action: dict[str, Any], *, body: str) -> bool:
    body_key = _term_key(body)
    if not body_key:
        return True
    schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
    schedule_span = str(schedule.get("source_span") or action.get("when_text") or "").strip()
    if schedule_span and body_key == _term_key(schedule_span):
        return True
    return bool(re.fullmatch(r"\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)", str(body or "").strip(), flags=re.IGNORECASE))


def _clear_satisfied_action_missing_fields(action: dict[str, Any]) -> bool:
    missing = action.get("missing_fields")
    if not isinstance(missing, list):
        return False
    kind = str(action.get("kind") or "").strip()
    body = str(action.get("body") or "").strip()
    schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
    schedule_span = str(schedule.get("source_span") or schedule.get("time") or "").strip()
    schedule_kind = str(schedule.get("kind") or "none").strip()
    satisfied: set[str] = set()
    if body:
        satisfied.add("body")
    if kind == "reminder":
        satisfied.update({"schedule", "time", "trigger_time"})
    elif kind == "alarm" and schedule_kind not in {"", "none"} and _schedule_span_looks_temporal(schedule_span):
        satisfied.update({"schedule", "time", "trigger_time"})
    if not satisfied:
        return False
    kept = [item for item in missing if str(item) not in satisfied]
    if kept == missing:
        return False
    action["missing_fields"] = kept
    return True


def _schedule_span_looks_temporal(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(
        re.search(r"\b\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:today|tomorrow|tonight|morning|afternoon|evening|night)\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:next|this|coming|upcoming)\s+(?:week|month|year|mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)(?:day)?\b", text, flags=re.IGNORECASE)
        or re.search(r"\bin\s+(?:\d+|a|an|half)\s+(?:minute|min|hour|hr|day|week|month|year)s?\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)(?:day)?\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", text, flags=re.IGNORECASE)
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text)
    )


def _first_temporal_candidate(value: str) -> str:
    text = str(value or "")
    patterns = (
        r"\b(?:at\s+)?\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)\b",
        r"\b(?:today|tomorrow|tonight)(?:\s+(?:at\s+)?\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm))?\b",
        r"\b(?:next|this|coming|upcoming)\s+(?:week|month|year|mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)(?:day)?(?:\s+(?:at\s+)?\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm))?\b",
        r"\bin\s+(?:\d+|a|an|half)\s+(?:minute|min|hour|hr|day|week|month|year)s?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""


def _source_starts_with_action_kind(source_text: str, kind: str) -> bool:
    text = str(source_text or "").strip()
    if kind == "reminder":
        return bool(
            re.search(
                r"^(?:juno\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder|"
                r"create\s+(?:a\s+)?reminder)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    if kind == "alarm":
        return bool(
            re.search(
                r"^(?:juno\s+)?(?:set|create|add)\s+(?:an?\s+)?alarm\b|^(?:juno\s+)?alarm\s+for\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    return False


def _native_action_segment_for_kind(source_text: str, kind: str, *, anchor: str) -> str:
    source = str(source_text or "").strip()
    if not source:
        return ""
    anchor_key = _term_key(anchor)
    hits: list[tuple[int, int, str]] = []
    pattern = re.compile(
        r"\b(?:"
        r"remind\s+me|set\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder|create\s+(?:a\s+)?reminder|"
        r"set\s+(?:an?\s+)?alarm|create\s+(?:an?\s+)?alarm|add\s+(?:an?\s+)?alarm|alarm\s+(?:for|at)|"
        r"take\s+(?:a\s+)?note|make\s+(?:a\s+)?note|create\s+(?:a\s+)?note|save\s+(?:a\s+)?note"
        r")\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(source):
        text = match.group(0)
        matched_kind = "reminder"
        if re.search(r"\balarm\b", text, flags=re.IGNORECASE):
            matched_kind = "alarm"
        elif re.search(r"\bnote\b", text, flags=re.IGNORECASE):
            matched_kind = "note"
        hits.append((match.start(), match.end(), matched_kind))
    for idx, (start, _end, matched_kind) in enumerate(hits):
        if matched_kind != kind:
            continue
        end = hits[idx + 1][0] if idx + 1 < len(hits) else len(source)
        segment = source[start:end].strip(" ,.;:-")
        if not segment:
            continue
        if anchor_key and anchor_key not in _term_key(segment):
            continue
        return segment
    return ""


def _render_text(render: dict[str, Any]) -> str:
    units = render.get("content_units")
    if not isinstance(units, list):
        return ""
    parts: list[str] = []
    for unit in units:
        if isinstance(unit, dict):
            text = str(unit.get("text") or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _term_key(text: str) -> str:
    value = str(text or "").casefold().strip()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _remove_key_phrase(value_key: str, phrase: str) -> str:
    phrase_key = _term_key(phrase)
    if not phrase_key:
        return value_key
    pattern = re.compile(rf"(?:^|\s){re.escape(phrase_key)}(?:\s|$)")
    prior = None
    out = value_key
    while out and out != prior:
        prior = out
        out = pattern.sub(" ", out)
        out = " ".join(out.split())
    return out


def _remove_surface_phrase(text: str, phrase: str) -> str:
    value = str(text or "").strip()
    target = str(phrase or "").strip()
    if not value or not target:
        return value
    pattern = re.compile(rf"\b{re.escape(target)}\b", flags=re.IGNORECASE)
    out = pattern.sub(" ", value)
    out = re.sub(r"\s+", " ", out).strip(" ,.;:-")
    out = re.sub(r"\s+(?:at|on|by|for)$", "", out, flags=re.IGNORECASE).strip(" ,.;:-")
    if out != value:
        return out
    key_target = _term_key(target)
    words = value.split()
    kept: list[str] = []
    idx = 0
    target_parts = key_target.split()
    while idx < len(words):
        window = _term_key(" ".join(words[idx : idx + len(target_parts)]))
        if target_parts and window == key_target:
            idx += len(target_parts)
            continue
        kept.append(words[idx])
        idx += 1
    out = re.sub(r"\s+", " ", " ".join(kept)).strip(" ,.;:-")
    return re.sub(r"\s+(?:at|on|by|for)$", "", out, flags=re.IGNORECASE).strip(" ,.;:-")


def _strip_trailing_schedule_clause(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    out = re.sub(
        r"\s+(?:at|by|for)\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|am|pm)?$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\s+on\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)$",
        "",
        out,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", out).strip(" ,.;:-")


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _writer_mode(packet: TurnPlanPacket) -> WriterMode:
    name = ""
    if packet.mode_selection is not None:
        name = packet.mode_selection.effective_mode or ""
    elif packet.mode_policy is not None:
        name = packet.mode_policy.base_mode or packet.mode_policy.mode_name
    try:
        return WriterMode(name)
    except ValueError:
        return WriterMode.DEFAULT_SURFACE


def _planner_max_tokens(final_text: str, selected_text: str) -> int:
    words = len((final_text or "").split()) + min(800, len((selected_text or "").split()))
    if words > 260:
        return 3072
    if words > 120:
        return 2048
    return 1536


def _json_object(raw: str) -> dict[str, Any] | None:
    obj, _notes = _json_object_with_notes(raw)
    return obj


def _json_object_with_notes(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
    parsed = _parse_json_object(text)
    if parsed is not None:
        return parsed, []
    repaired = _repair_misnested_render_plan_fields(text)
    if repaired != text:
        parsed = _parse_json_object(repaired)
        if parsed is not None:
            return parsed, ["json_repaired_misnested_render_plan_fields"]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None, []
    candidate = match.group(0)
    parsed = _parse_json_object(candidate)
    if parsed is not None:
        return parsed, []
    repaired = _repair_misnested_render_plan_fields(candidate)
    if repaired != candidate:
        parsed = _parse_json_object(repaired)
        if parsed is not None:
            return parsed, ["json_repaired_misnested_render_plan_fields"]
    return None, []


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


_RENDER_PLAN_ARRAY_LEVEL_KEYS = frozenset({
    "claimed_item_count",
    "spoken_item_count",
})


def _repair_misnested_render_plan_fields(text: str) -> str:
    """Recover a near-valid turn_plan where render-level keys landed in an array.

    Qwen sometimes emits the right semantic schema but misses the closing
    bracket for ``render_plan.content_units`` before render-level metadata
    such as ``claimed_item_count``. This repair is schema-level: it only fires
    while scanning a ``content_units`` array and only when a render-plan field
    key appears at the array's top level.
    """

    value = str(text or "")
    key = '"content_units"'
    idx = 0
    edits: list[int] = []
    while True:
        key_pos = value.find(key, idx)
        if key_pos < 0:
            break
        arr_start = value.find("[", key_pos + len(key))
        if arr_start < 0:
            break
        misplaced = _find_misnested_render_plan_field(value, arr_start + 1)
        if misplaced is not None:
            edits.append(misplaced)
            idx = misplaced + 1
        else:
            idx = arr_start + 1
    if not edits:
        return value
    out = value
    for pos in reversed(edits):
        out = f"{out[:pos]}]{out[pos:]}"
    return out


def _find_misnested_render_plan_field(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch in "[{":
            depth += 1
            i += 1
            continue
        if ch == "]":
            if depth == 0:
                return None
            depth -= 1
            i += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
            i += 1
            continue
        if ch == "," and depth == 0:
            field = _field_name_after_comma(text, i + 1)
            if field in _RENDER_PLAN_ARRAY_LEVEL_KEYS:
                return i
        i += 1
    return None


def _field_name_after_comma(text: str, start: int) -> str | None:
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != '"':
        return None
    i += 1
    chars: list[str] = []
    escape = False
    while i < len(text):
        ch = text[i]
        if escape:
            chars.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            return "".join(chars) if i < len(text) and text[i] == ":" else None
        else:
            chars.append(ch)
        i += 1
    return None


def _bound(text: str | None, limit: int, *, tail: bool = False) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[-limit:] if tail else value[:limit]


def _markdown_allowed(context: TypedContextBundle) -> bool:
    cat = (context.app_category or "").strip().lower()
    if cat in {"code", "terminal"}:
        return False
    meta = context.metadata or {}
    if "markdown_allowed" in meta:
        return bool(meta.get("markdown_allowed"))
    return cat in {"docs", "notes", "unknown"}


def _snapshot_counts(snapshot: MemorySnapshot | None) -> dict[str, int] | None:
    if snapshot is None:
        return None
    return {
        "lexicon": len(snapshot.lexicon),
        "replacements": len(snapshot.replacements),
        "corrections": len(snapshot.corrections),
        "session_entities": len(snapshot.session_entities),
    }


def _replacement_payload(snapshot: MemorySnapshot | None) -> list[dict[str, str]]:
    if snapshot is None:
        return []
    out: list[dict[str, str]] = []
    for item in list(snapshot.replacements or [])[:32]:
        trigger = str(getattr(item, "trigger", "") or "").strip()
        replacement = str(getattr(item, "replacement", "") or "").strip()
        if trigger and replacement:
            out.append({"trigger": trigger, "replacement": replacement})
    return out


def _snippet_payload(memory_store: JsonMemoryStore | None, app_category: str | None) -> list[dict[str, Any]]:
    snippets = getattr(memory_store, "snippets", None) if memory_store is not None else None
    list_fn = getattr(snippets, "list", None)
    if not callable(list_fn):
        return []
    scope = (app_category or "global").strip().lower() or "global"
    out: list[dict[str, Any]] = []
    try:
        entries = list(list_fn())
    except Exception:
        return []
    for item in entries[:80]:
        trigger = str(getattr(item, "trigger", "") or "").strip()
        body = str(getattr(item, "body", "") or "")
        snip_scope = str(getattr(item, "scope", "global") or "global").strip().lower()
        if not trigger or not body:
            continue
        if snip_scope not in {scope, "global", "messaging", "email", "docs"}:
            continue
        out.append({
            "trigger": trigger,
            "scope": snip_scope,
            "body_preview": body[:500],
            "body_chars": len(body),
            "case_sensitive": bool(getattr(item, "case_sensitive", False)),
        })
    return out


def _served_snippet_payload(
    *,
    store_snippets: list[dict[str, Any]],
    memory_packet: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        trigger = str(item.get("trigger") or "").strip()
        if not trigger:
            return
        scope = str(item.get("scope") or "global").strip().lower() or "global"
        key = f"{scope}:{trigger.casefold()}"
        if key in seen:
            return
        seen.add(key)
        try:
            body_chars = int(item.get("body_chars") or len(str(item.get("body") or item.get("body_preview") or "")))
        except (TypeError, ValueError):
            body_chars = len(str(item.get("body") or item.get("body_preview") or ""))
        out.append({
            "trigger": trigger,
            "scope": scope,
            "body_preview": str(item.get("body_preview") or item.get("body") or "")[:500],
            "body_chars": body_chars,
            "case_sensitive": bool(item.get("case_sensitive", False)),
        })

    for item in store_snippets:
        if isinstance(item, dict):
            add(item)
    packet_snippets = memory_packet.get("snippets") if isinstance(memory_packet, dict) else None
    if isinstance(packet_snippets, list):
        for item in packet_snippets:
            if isinstance(item, dict):
                add(item)
    return out[:16]


def _turn_plan_schema_hint() -> dict[str, Any]:
    return {
        "schema_version": "turn_plan_v1",
        "required_top_level_keys": [
            "schema_version",
            "utterance_kind",
            "corrected_transcript",
            "target",
            "render_plan",
            "transform",
            "actions",
            "snippets",
            "memory_candidates",
            "safety",
            "uncertainties",
        ],
        "allowed_values": {
            "utterance_kind": [
                "dictation",
                "format_dictation",
                "transform",
                "actions",
                "mixed",
                "command",
                "memory_mutation",
                "no_op",
                "ambiguous",
            ],
            "target.kind": ["cursor", "selection", "recent_commit", "explicit_span", "none"],
            "render_plan.render_kind": [
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
            ],
            "render_plan.content_units.kind": ["paragraph", "heading", "item", "code", "row"],
            "transform.operation": [
                "none",
                "rewrite",
                "shorten",
                "expand",
                "summarize",
                "translate",
                "tone",
                "format",
                "fix_grammar",
                "extract",
            ],
            "actions.kind": ["note", "reminder", "alarm"],
            "actions.operation": ["create"],
            "actions.schedule.kind": ["none", "instant", "vague", "series"],
            "safety.commit_policy": ["commit", "no_commit", "confirm"],
            "safety.execute_policy": ["execute", "no_execute", "confirm"],
        },
        "corrected_transcript": {
            "text": "string",
            "corrections": "array",
            "literal_spans": "array",
        },
        "target": {"kind": "one allowed target.kind value", "confidence": "0..1"},
        "render_plan": {
            "render_kind": "one allowed render_plan.render_kind value",
            "markdown_allowed": "boolean",
            "content_units": "array of {kind,text,source_span,order}",
            "claimed_item_count": "number|null",
            "spoken_item_count": "number|null",
        },
        "transform": {
            "operation": "one allowed transform.operation value",
            "instruction": "string",
            "transformed_text": "string|null",
            "requires_second_pass": "boolean",
        },
        "actions": "array of planned native actions with kind, operation, body, evidence_span, schedule, missing_fields",
        "snippets": "array of snippet operations",
        "memory_candidates": "array",
        "safety": {
            "commit_policy": "one allowed safety.commit_policy value",
            "execute_policy": "one allowed safety.execute_policy value",
        },
        "uncertainties": "array",
        "do_not_copy": [
            "allowed_values",
            "required_top_level_keys",
            "one allowed ... value text",
        ],
    }


def _action_surface_payload(
    *,
    wake_verified: bool,
    allowed_action_kinds: tuple[str, ...],
    max_actions: int,
    now_iso: str | None,
    permission_state: str,
) -> dict[str, Any]:
    """The native-action surface offered to the planner for one turn.

    Native actions are dispatched in exactly one place - the pipeline's
    ``if wake_status.verified and action_source_text:`` branch. When the wake
    gate reported no wake word there is no dispatch site, and
    ``WriterService._turn_plan_outcome`` discards every ``actions`` plan
    unconditionally in favour of text delivery
    (``turn_plan_actions_fell_back_to_text``). Advertising up to 25 actions
    across 3 kinds on those turns bought decode time for a classification the
    code guarantees it will throw away (issue #107).

    So on a non-wake turn the offer is withdrawn: no kinds, no operations, a
    cap of zero. Only the *offer* changes. The packet fields are untouched,
    the planner still runs, and every other planning capability (structural
    rendering, transforms, corrected transcript, memory candidates, snippets,
    commit policy) is unaffected.
    """
    if not wake_verified:
        return {
            "wake_verified": False,
            "allowed_action_kinds": [],
            "supported_operations": {},
            "max_actions": 0,
            "now_iso": now_iso,
            "permission_state": permission_state,
            "destructive_operations_require_confirmation": True,
            "reason": "no_wake_word_this_turn",
        }
    return {
        "wake_verified": True,
        "allowed_action_kinds": list(allowed_action_kinds),
        "supported_operations": {str(kind): ["create"] for kind in allowed_action_kinds},
        "max_actions": max_actions,
        "now_iso": now_iso,
        "permission_state": permission_state,
        "destructive_operations_require_confirmation": True,
    }


def _turn_plan_schema_hint_without_actions() -> dict[str, Any]:
    """Schema hint for a turn on which no native action can be dispatched.

    Derived from ``_turn_plan_schema_hint()`` instead of duplicating it so the
    two stay in lock-step: only the action-bearing entries are narrowed.
    ``mixed`` deliberately stays allowed - a non-wake ``mixed`` plan still
    renders and commits its text (``WriterService`` suppresses the paste for
    ``mixed`` only when ``wake_verified`` is true), so removing it would drop a
    currently-deliverable outcome.
    """
    hint = _turn_plan_schema_hint()
    allowed = dict(hint.get("allowed_values") or {})
    allowed["utterance_kind"] = [
        value for value in (allowed.get("utterance_kind") or []) if value != "actions"
    ]
    allowed["actions.kind"] = []
    hint["allowed_values"] = allowed
    hint["actions"] = "must be [] on this turn; no native action can be dispatched"
    hint["actions_not_permitted"] = (
        "actions.wake_verified is false: 'actions' is not an allowed utterance_kind for this "
        "turn and the actions array must be empty. Classify the turn as dictation, "
        "format_dictation, transform, memory_mutation, no_op, or ambiguous instead."
    )
    return hint
