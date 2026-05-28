"""Pipeline integration hook for Actions.

Detection order:

1. Strict wake gate: only leading "Juno" or "Hey Juno" is eligible.
2. LLM extractor: intent classification plus structured action extraction.
3. Deterministic grammar fallback: used only when no extractor is registered
   or the extractor/provider fails without making a non-action decision.

Failure mode: this helper *never* raises. A broken parser, schema
mismatch, or extractor crash must not break dictation. All exceptions
are swallowed and reported via dedicated trace events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from juno_core_v3.actions.contracts import Action
from juno_core_v3.actions.grammar import parse_actions, strip_wake
from juno_core_v3.actions.llm_extractor import (
    _env_bool,
    extract_actions_with_llm_result,
    get_llm_extractor,
    select_actions_schema_version,
)


class _Recorder(Protocol):  # pragma: no cover - structural type only
    def record(self, kind: Any, event: str, payload: dict[str, Any]) -> None: ...


def _record(
    recorder: _Recorder | None,
    trace_kind: Any,
    event: str,
    payload: dict[str, Any],
) -> None:
    if recorder is None:
        return
    try:
        recorder.record(trace_kind, event, payload)
    except Exception:  # pragma: no cover - never break the pipeline
        pass


def detect_actions_for_pipeline(
    *,
    utterance_id: str,
    normalized_text: str,
    recorder: _Recorder | None,
    trace_kind: Any,
    now: datetime | None = None,
    wake_verified: bool = False,
    raw_wake_text: str | None = None,
    context_packet: Any | None = None,
) -> list[Action] | None:
    """Run the tiered parser and emit a structured trace event.

    Returns the list of detected actions or ``None`` (no actions). The
    return value is consumed by the dictation pipeline to suppress the
    paste for action utterances and to ship the action payload to the
    macOS shell for execution.
    """

    emit_vnext_trace = wake_verified or raw_wake_text is not None or context_packet is not None
    post_wake = (normalized_text or "").strip() if wake_verified else strip_wake(normalized_text)
    if post_wake is None:
        _record(
            recorder,
            trace_kind,
            "actions_detect",
            {
                "utterance_id": utterance_id,
                "detected": False,
                "source": "wake_gate",
                "wake_detected": False,
            },
        )
        return None

    if not post_wake.strip():
        _record(
            recorder,
            trace_kind,
            "actions_detect",
            {
                "utterance_id": utterance_id,
                "detected": False,
                "source": "wake_gate",
                "wake_detected": True,
                "intent_gate_result": "empty_post_wake",
            },
        )
        return None

    followup_actions = _followup_actions_for_pipeline(post_wake, recorder, trace_kind, now=now)
    if followup_actions:
        _record(
            recorder,
            trace_kind,
            "actions_detect",
            {
                "utterance_id": utterance_id,
                "detected": True,
                "source": "followup",
                "wake_detected": True,
                "count": len(followup_actions),
                "kinds": [a.kind.value for a in followup_actions],
                "actions": [a.to_dict() for a in followup_actions],
            },
        )
        return followup_actions

    extractor = get_llm_extractor()
    if extractor is not None:
        target_resolver = _target_resolver_for_pipeline(recorder, now=now)
        if emit_vnext_trace:
            _record(
                recorder,
                trace_kind,
                "action_extraction_started",
                {
                    "utterance_id": utterance_id,
                        "schema_version": _action_schema_version_for_trace(post_wake),
                    "corrected_text_hash": _short_hash(post_wake),
                    "raw_wake_text": raw_wake_text,
                    "context_terms": _context_term_count(context_packet),
                },
            )
        llm_result = extract_actions_with_llm_result(
            post_wake,
            now=now,
            target_resolver=target_resolver,
        )
        if llm_result.actions:
            if emit_vnext_trace:
                _record(
                    recorder,
                    trace_kind,
                    "action_extraction_valid",
                    {
                        "utterance_id": utterance_id,
                        "action_count": len(llm_result.actions),
                        "action_types": [a.kind.value for a in llm_result.actions],
                    },
                )
            _record(
                recorder,
                trace_kind,
                "actions_detect",
                {
                    "utterance_id": utterance_id,
                    "detected": True,
                    "source": "llm",
                    "wake_detected": True,
                    "intent_gate_result": llm_result.intent,
                    "count": len(llm_result.actions),
                    "kinds": [a.kind.value for a in llm_result.actions],
                    "actions": [a.to_dict() for a in llm_result.actions],
                },
            )
            return llm_result.actions

        if not llm_result.allow_regex_fallback:
            if emit_vnext_trace:
                _record(
                    recorder,
                    trace_kind,
                    "action_extraction_rejected",
                    {
                        "utterance_id": utterance_id,
                        "reason": llm_result.rejected_reason,
                        "confidence": None,
                    },
                )
            _record(
                recorder,
                trace_kind,
                "actions_detect",
                {
                    "utterance_id": utterance_id,
                    "detected": False,
                    "source": "llm",
                    "wake_detected": True,
                    "llm_available": True,
                    "intent_gate_result": llm_result.intent,
                    "rejected_reason": llm_result.rejected_reason,
                },
            )
            return None

        _record(
            recorder,
            trace_kind,
            "actions_llm_fallback",
            {
                "utterance_id": utterance_id,
                "wake_detected": True,
                "llm_available": True,
                "intent_gate_result": llm_result.intent,
                "rejected_reason": llm_result.rejected_reason,
                "error": llm_result.error,
            },
        )

    # Fallback parser remains available for offline/provider-failure paths.
    try:
        fallback_text = f"Juno {post_wake}" if wake_verified else normalized_text
        actions = parse_actions(fallback_text, now=now)
    except Exception as exc:  # noqa: BLE001 — never break the pipeline
        _record(
            recorder,
            trace_kind,
            "actions_parser_error",
            {
                "utterance_id": utterance_id,
                "error": str(exc),
                "stage": "regex_fallback",
            },
        )
        actions = None

    if actions:
        if emit_vnext_trace:
            _record(
                recorder,
                trace_kind,
                "action_extraction_valid",
                {
                    "utterance_id": utterance_id,
                    "action_count": len(actions),
                    "action_types": [a.kind.value for a in actions],
                },
            )
        _record(
            recorder,
            trace_kind,
            "actions_detect",
            {
                "utterance_id": utterance_id,
                "detected": True,
                "source": "regex_fallback",
                "wake_detected": True,
                "llm_available": extractor is not None,
                "count": len(actions),
                "kinds": [a.kind.value for a in actions],
                "actions": [a.to_dict() for a in actions],
            },
        )
        return actions

    if emit_vnext_trace:
        _record(
            recorder,
            trace_kind,
            "action_extraction_rejected",
            {
                "utterance_id": utterance_id,
                "reason": "no_valid_actions",
                "confidence": None,
            },
        )
    _record(
        recorder,
        trace_kind,
        "actions_detect",
        {
            "utterance_id": utterance_id,
            "detected": False,
            "source": "regex_fallback",
            "wake_detected": True,
            "llm_available": extractor is not None,
        },
    )
    return None


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _context_term_count(packet: Any | None) -> int:
    if packet is None:
        return 0
    terms = getattr(packet, "context_terms", None)
    try:
        return len(terms or [])
    except TypeError:
        return 0


def _action_schema_version_for_trace(text: str) -> str:
    return select_actions_schema_version(text, use_v3=_env_bool("JUNO_ACTIONS_SCHEMA_V3", False))


def _target_resolver_for_pipeline(
    recorder: _Recorder | None,
    *,
    now: datetime | None,
) -> Any | None:
    if not _env_bool("JUNO_ACTIONS_OPERATIONS", False):
        return None
    log_dir = getattr(recorder, "log_dir", None)
    session_id = str(getattr(recorder, "session_id", "") or "")
    if log_dir is None or not session_id:
        return None
    try:
        from juno_core_v3.actions.reference_resolver import resolve_target
        from juno_v2.observability.actions_index import get_actions_index

        index = get_actions_index(Path(log_dir))
    except Exception:
        return None
    resolver_now = now or datetime.now(timezone.utc)

    def _resolve(target: dict[str, Any]) -> Any | None:
        return resolve_target(
            target,
            index=index,
            session_id=session_id,
            now=resolver_now,
        )

    return _resolve


def _followup_actions_for_pipeline(
    post_wake: str,
    recorder: _Recorder | None,
    trace_kind: Any,
    *,
    now: datetime | None,
) -> list[Action] | None:
    if not _env_bool("JUNO_ACTIONS_FOLLOWUP", False):
        return None
    log_dir = getattr(recorder, "log_dir", None)
    session_id = str(getattr(recorder, "session_id", "") or "")
    if log_dir is None or not session_id:
        return None
    try:
        from juno_core_v3.actions.followup import followup_action_for_row, parse_followup
        from juno_v2.observability.actions_index import get_actions_index

        intent = parse_followup(post_wake)
        if intent is None or intent.kind == "confirm":
            return None
        index = get_actions_index(Path(log_dir))
        row = index.last_touched(kind="reminder")
    except Exception:
        return None
    if not row:
        return None
    if str(row.get("last_seen_session") or "") != session_id:
        return None
    if str(row.get("status") or "") != "active":
        return None
    followup_now = now or datetime.now(timezone.utc)
    try:
        last_modified = int(row.get("last_modified_at") or 0)
    except (TypeError, ValueError):
        return None
    now_ms = int(followup_now.timestamp() * 1000)
    if last_modified < now_ms - 30_000:
        return None
    action = followup_action_for_row(intent, row, now=followup_now)
    if action is None:
        return None
    _record(
        recorder,
        trace_kind,
        "action_followup_matched",
        {
            "kind": intent.kind,
            "field": intent.field,
            "juno_id": row.get("juno_id"),
        },
    )
    return [action]

