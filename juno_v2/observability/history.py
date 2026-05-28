"""Utterance history reader (P2 item 13).

The workbench already records everything that happens during a
dictation as a JSONL trace. The Mac shell and ops tools need a much
smaller view: *"give me the last N utterances with enough signal to
debug a bad transcription"*. Dumping raw trace events is too noisy
(hundreds of VAD frames / partial updates per utterance); the
existing :mod:`session_reader` is designed for human consumption,
not a JSON API.

This module sits between the two: it walks the trace file once,
groups events by ``utterance_id``, and produces one compact record
per utterance with the five fields the API contract asks for:

* ``mode`` — "oneshot" or "streaming"; inferred from event names.
* ``model_path`` — resolved from
  ``oneshot_transcribed`` / ``final_decode_completed``.
* ``context`` — ``{app_name, window_title, app_category,
  selection_chars}``; sourced from ``utterance_context_planned`` or
  ``oneshot_plan_built``.
* ``provenance`` — ``{preview_backend, final_backend,
  writer_backend, engine_mode}``; session-scoped.
* ``failure_reason`` — first of ``capability_blocked``,
  ``transcribe_unavailable``, ``transcribe_error``,
  ``commit.conflict_reason``, ``editable_sync_failed``, or
  ``None`` when the utterance committed cleanly.

The reader is I/O-bound and holds no state, so it's safe to call on
every ``GET /api/broker/history`` hit. For large trace files we
stream line-by-line instead of loading everything; the ``limit``
argument caps the number of utterances returned (newest first).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# Events we look at. Matching on exact names keeps the reader fast
# and keeps the contract explicit — if engine events get renamed,
# the test for this module fails loudly instead of silently dropping
# a field.
_SESSION_EVENT_NAMES = frozenset({
    "dictation_session_started",
})

_CONTEXT_EVENT_NAMES = frozenset({
    "utterance_context_planned",
    "oneshot_plan_built",
})

_TRANSCRIBE_EVENT_NAMES = frozenset({
    "oneshot_transcribed",
    "final_decode_completed",
})

_FAILURE_EVENT_NAMES = frozenset({
    "oneshot_capability_blocked",
    "oneshot_transcribe_unavailable",
    "oneshot_transcribe_error",
    "editable_sync_failed",
})

_COMMIT_EVENT_NAMES = frozenset({
    "commit_completed",
})

_BROKER_EVENT_NAMES = frozenset({
    "broker_session_started",
    "broker_transform",
    "oneshot_itn_applied",
    "streaming_itn_applied",
    "oneshot_audio_retained",
    "insertion_committed",
})


@dataclass(slots=True)
class UtteranceHistoryEntry:
    utterance_id: str
    ts_unix_ms: int = 0
    mode: str = "unknown"  # "oneshot" | "streaming" | "unknown"
    model_path: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    recovery: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    transcript: str | None = None
    raw_transcript: str | None = None
    session_class: str | None = None  # "insert" | "transform"
    transform_type: str | None = None  # "deterministic" | "model_backed" | "degraded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "ts_unix_ms": self.ts_unix_ms,
            "mode": self.mode,
            "model_path": self.model_path,
            "context": dict(self.context),
            "provenance": dict(self.provenance),
            "recovery": dict(self.recovery),
            "failure_reason": self.failure_reason,
            "transcript": self.transcript,
            "raw_transcript": self.raw_transcript,
            "session_class": self.session_class,
            "transform_type": self.transform_type,
        }


def read_history(
    log_path: Path | str,
    *,
    limit: int = 50,
    replay_available_fn: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` utterances from a trace file.

    Results are ordered newest-first. Returns an empty list if the
    file doesn't exist yet (which is the common "fresh install"
    case); callers should treat that as "no history", not an error.
    """
    p = Path(log_path)
    if not p.exists():
        return []

    session_provenance: dict[str, Any] = {}
    utterances: dict[str, UtteranceHistoryEntry] = {}
    order: list[str] = []  # insertion order of first-seen utterance_id

    with p.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = event.get("name") or ""
            payload = event.get("payload") or {}
            ts = int(event.get("ts_unix_ms") or 0)

            # Session-scoped provenance (overwrite on re-start so the
            # latest session's backends win).
            if name in _SESSION_EVENT_NAMES:
                session_provenance = {
                    "engine_mode": payload.get("engine_mode"),
                    "preview_backend": payload.get("preview_backend"),
                    "final_backend": payload.get("final_backend"),
                    "writer_backend": payload.get("writer_backend"),
                    "writer_model_path": payload.get("writer_model_path"),
                }
                continue

            uid = payload.get("utterance_id")
            if not isinstance(uid, str) or not uid:
                continue

            entry = utterances.get(uid)
            if entry is None:
                entry = UtteranceHistoryEntry(
                    utterance_id=uid,
                    ts_unix_ms=ts,
                    provenance=dict(session_provenance),
                )
                utterances[uid] = entry
                order.append(uid)

            # Always keep the earliest ts (first sign of utterance)
            if ts and (entry.ts_unix_ms == 0 or ts < entry.ts_unix_ms):
                entry.ts_unix_ms = ts

            if name in _CONTEXT_EVENT_NAMES:
                _apply_context(entry, name, payload)
                continue
            if name in _TRANSCRIBE_EVENT_NAMES:
                _apply_transcribe(entry, name, payload)
                continue
            if name in _COMMIT_EVENT_NAMES:
                _apply_commit(entry, payload)
                continue
            if name in _FAILURE_EVENT_NAMES:
                _apply_failure(entry, name, payload)
                continue
            if name in _BROKER_EVENT_NAMES:
                _apply_broker(entry, name, payload)
                continue

    # Newest first. We use the first-seen ts per utterance so
    # a flurry of late events on an old utterance doesn't jump it
    # back to the top.
    ordered = sorted(
        utterances.values(),
        key=lambda e: (e.ts_unix_ms, order.index(e.utterance_id)),
        reverse=True,
    )
    results = [e.to_dict() for e in ordered[: max(0, int(limit))]]
    if replay_available_fn is not None:
        for entry in results:
            uid = str(entry.get("utterance_id") or "")
            if not uid:
                continue
            available = bool(replay_available_fn(uid))
            recovery = dict(entry.get("recovery") or {})
            recovery["replay_available"] = available
            recovery["rerun_available"] = available
            entry["recovery"] = recovery
    return results


# ------------------------------------------------------------------ #
# Per-event appliers
# ------------------------------------------------------------------ #


def _apply_context(entry: UtteranceHistoryEntry, name: str, payload: dict) -> None:
    if name == "oneshot_plan_built":
        # One-shot pipeline emits context fields flat on the payload.
        if entry.mode == "unknown":
            entry.mode = "oneshot"
        entry.context.setdefault("app_name", payload.get("app_name"))
        entry.context.setdefault("window_title", payload.get("window_title"))
        entry.context.setdefault("language", payload.get("language"))
        entry.context.setdefault(
            "bias_phrase_count", payload.get("bias_phrase_count")
        )
        return
    # streaming ``utterance_context_planned``: context is nested.
    entry.mode = "streaming"
    ctx = payload.get("context") or {}
    entry.context.setdefault("app_name", ctx.get("app_name"))
    entry.context.setdefault("window_title", ctx.get("window_title"))
    entry.context.setdefault("app_category", ctx.get("app_category"))
    entry.context.setdefault("selection_chars", len(ctx.get("selected_text") or ""))
    entry.context.setdefault("initial_prompt", payload.get("initial_prompt"))


def _apply_transcribe(entry: UtteranceHistoryEntry, name: str, payload: dict) -> None:
    model_path = payload.get("model_path")
    if model_path:
        entry.model_path = str(model_path)
    backend = payload.get("backend") or payload.get("backend_name")
    if backend:
        # Record the backend that actually produced the transcript.
        # For streaming this is the final_asr backend; for oneshot
        # it's the lane's transcriber. Storing both under
        # ``transcriber_backend`` avoids confusion with the
        # session-scoped preview/final backends.
        entry.provenance["transcriber_backend"] = backend
    if name == "oneshot_transcribed":
        if entry.mode == "unknown":
            entry.mode = "oneshot"
        # The one-shot path doesn't fire a separate commit event; the
        # normalized transcript surfaces via the HTTP response. We
        # capture what we can from the trace so the API is still
        # useful for oneshot utterances: at least the chars counts.
        entry.context.setdefault("raw_chars", payload.get("raw_chars"))
        entry.context.setdefault("normalized_chars", payload.get("normalized_chars"))
        if "writer_action" in payload:
            entry.provenance["writer_action"] = payload["writer_action"]
        if "writer_deterministic" in payload:
            entry.provenance["writer_deterministic"] = payload["writer_deterministic"]
    elif name == "final_decode_completed":
        # Streaming final transcript may not be the committed text
        # (writer may rewrite, normalizer may tweak). Store the
        # normalized form we have access to; commit event will
        # override with the canonical committed_text.
        norm = payload.get("normalization") or {}
        raw_text = norm.get("raw_text") or payload.get("text")
        normalized = norm.get("normalized_text") or payload.get("text")
        if raw_text and entry.raw_transcript is None:
            entry.raw_transcript = str(raw_text)
        if normalized and entry.transcript is None:
            entry.transcript = str(normalized)


def _apply_commit(entry: UtteranceHistoryEntry, payload: dict) -> None:
    committed = payload.get("committed")
    committed_text = payload.get("committed_text")
    if committed and committed_text is not None:
        entry.transcript = str(committed_text)
    conflict = payload.get("conflict_reason")
    if conflict and entry.failure_reason is None and not committed:
        entry.failure_reason = f"commit_conflict:{conflict}"


def _apply_broker(entry: UtteranceHistoryEntry, name: str, payload: dict) -> None:
    if name == "broker_session_started":
        session = payload.get("kind") or (payload.get("broker_session") or {}).get("kind")
        if session and entry.session_class is None:
            entry.session_class = str(session)
    elif name == "broker_transform":
        result = payload
        deterministic = result.get("deterministic")
        degraded = result.get("degraded")
        if degraded:
            entry.transform_type = "degraded"
        elif deterministic:
            entry.transform_type = "deterministic"
        else:
            entry.transform_type = "model_backed"
    elif name in {"oneshot_itn_applied", "streaming_itn_applied"}:
        entry.provenance["itn_profile"] = payload.get("profile")
        entry.provenance["itn_rules"] = payload.get("rules_applied")
    elif name == "oneshot_audio_retained":
        entry.recovery["replay_available"] = bool(payload.get("replay_available"))
        entry.recovery["rerun_available"] = bool(payload.get("rerun_available", payload.get("replay_available")))
        entry.recovery["storage"] = payload.get("storage")
        entry.recovery["retention_limit"] = payload.get("retention_limit")
    elif name == "insertion_committed":
        if payload.get("ok") and payload.get("transcript"):
            entry.transcript = str(payload["transcript"])


def _apply_failure(entry: UtteranceHistoryEntry, name: str, payload: dict) -> None:
    # Don't overwrite an existing reason; the *first* failure is the
    # most informative (a transcribe_error cascading into an
    # editable_sync_failed would lie if we kept the latter).
    if entry.failure_reason is not None:
        return
    if name == "oneshot_capability_blocked":
        entry.failure_reason = f"capability_blocked:{payload.get('reason') or ''}".rstrip(":")
    elif name == "oneshot_transcribe_unavailable":
        entry.failure_reason = f"transcribe_unavailable:{payload.get('code') or ''}".rstrip(":")
    elif name == "oneshot_transcribe_error":
        entry.failure_reason = "transcribe_error"
    elif name == "editable_sync_failed":
        entry.failure_reason = "editable_sync_failed"


__all__ = ["read_history", "UtteranceHistoryEntry"]
