"""SQLite mirror of actions Juno has touched.

The product history table stores per-utterance payloads for the History UI.
This index is action-centric: one row per sink operation, keyed by the Juno
UUID that later update/delete/query flows resolve against.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS actions_index (
  juno_id TEXT PRIMARY KEY,
  utterance_id TEXT NOT NULL,
  sink_kind TEXT NOT NULL,
  sink_id TEXT,
  series_id TEXT,
  body_normalized TEXT NOT NULL,
  due_iso TEXT,
  schedule_kind TEXT,
  recurrence_freq TEXT,
  recurrence_count INTEGER,
  recurrence_until_iso TEXT,
  list_name TEXT,
  app_bundle_id TEXT,
  created_at INTEGER NOT NULL,
  last_modified_at INTEGER NOT NULL,
  last_seen_session TEXT,
  deleted_at INTEGER,
  status TEXT
);
CREATE INDEX IF NOT EXISTS idx_actions_index_created ON actions_index(created_at);
CREATE INDEX IF NOT EXISTS idx_actions_index_body_norm ON actions_index(body_normalized);
CREATE INDEX IF NOT EXISTS idx_actions_index_due_iso ON actions_index(due_iso);
"""

_locks: dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    if key not in _locks:
        _locks[key] = threading.Lock()
    return _locks[key]


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_body(body: str) -> str:
    lowered = (body or "").lower()
    stripped = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip()


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _schedule_fields(schedule: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schedule, dict):
        return {
            "due_iso": None,
            "schedule_kind": None,
            "recurrence_freq": None,
            "recurrence_count": None,
            "recurrence_until_iso": None,
        }
    kind = _clean_str(schedule.get("kind"))
    due_iso: str | None = None
    recurrence_freq: str | None = None
    recurrence_count: int | None = None
    recurrence_until_iso: str | None = None
    if kind == "instant":
        instant = schedule.get("instant") if isinstance(schedule.get("instant"), dict) else {}
        due_iso = _clean_str(instant.get("iso")) or _clean_str(schedule.get("iso"))
    elif kind == "series":
        series = schedule.get("series") if isinstance(schedule.get("series"), dict) else {}
        due_iso = _clean_str(series.get("first_occurrence_iso"))
        recurrence_freq = _clean_str(series.get("freq"))
        recurrence_count = _int_or_none(series.get("count"))
        recurrence_until_iso = _clean_str(series.get("until_iso"))
    elif kind == "vague":
        vague = schedule.get("vague") if isinstance(schedule.get("vague"), dict) else {}
        due_iso = _clean_str(vague.get("default_iso"))
    return {
        "due_iso": due_iso,
        "schedule_kind": kind,
        "recurrence_freq": recurrence_freq,
        "recurrence_count": recurrence_count,
        "recurrence_until_iso": recurrence_until_iso,
    }


class ActionsIndex:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.db_path = (self.log_dir / "actions_index.sqlite").resolve()
        self.init_schema()

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                con.executescript(_SCHEMA)
                con.commit()
            finally:
                con.close()

    def upsert(
        self,
        *,
        juno_id: str,
        utterance_id: str,
        sink_kind: str,
        sink_id: str | None,
        body: str,
        schedule: dict[str, Any] | None,
        app_bundle_id: str | None,
        status: str = "active",
        series_id: str | None = None,
        list_name: str | None = None,
        last_seen_session: str | None = None,
    ) -> None:
        jid = _clean_str(juno_id)
        uid = _clean_str(utterance_id)
        kind = _clean_str(sink_kind)
        if not jid or not uid or not kind:
            return
        body_norm = _normalize_body(body)
        fields = _schedule_fields(schedule)
        effective_status = _clean_str(status) or "active"
        now_ms = int(time.time() * 1000)
        effective_series_id = _clean_str(series_id)
        if not effective_series_id and fields["schedule_kind"] == "series":
            effective_series_id = jid
        deleted_at = now_ms if effective_status == "deleted" else None
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                con.execute(
                    """
                    INSERT INTO actions_index (
                        juno_id, utterance_id, sink_kind, sink_id, series_id,
                        body_normalized, due_iso, schedule_kind, recurrence_freq,
                        recurrence_count, recurrence_until_iso, list_name,
                        app_bundle_id, created_at, last_modified_at,
                        last_seen_session, deleted_at, status
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(juno_id) DO UPDATE SET
                        utterance_id = excluded.utterance_id,
                        sink_kind = excluded.sink_kind,
                        sink_id = excluded.sink_id,
                        series_id = excluded.series_id,
                        body_normalized = excluded.body_normalized,
                        due_iso = excluded.due_iso,
                        schedule_kind = excluded.schedule_kind,
                        recurrence_freq = excluded.recurrence_freq,
                        recurrence_count = excluded.recurrence_count,
                        recurrence_until_iso = excluded.recurrence_until_iso,
                        list_name = excluded.list_name,
                        app_bundle_id = excluded.app_bundle_id,
                        last_modified_at = excluded.last_modified_at,
                        last_seen_session = excluded.last_seen_session,
                        deleted_at = CASE
                            WHEN excluded.status = 'deleted' THEN COALESCE(actions_index.deleted_at, excluded.deleted_at)
                            WHEN excluded.status != 'deleted' THEN NULL
                            ELSE actions_index.deleted_at
                        END,
                        status = excluded.status
                    """,
                    (
                        jid,
                        uid,
                        kind,
                        _clean_str(sink_id),
                        effective_series_id,
                        body_norm,
                        fields["due_iso"],
                        fields["schedule_kind"],
                        fields["recurrence_freq"],
                        fields["recurrence_count"],
                        fields["recurrence_until_iso"],
                        _clean_str(list_name),
                        _clean_str(app_bundle_id),
                        now_ms,
                        now_ms,
                        _clean_str(last_seen_session),
                        deleted_at,
                        effective_status,
                    ),
                )
                con.commit()
            finally:
                con.close()

    def mark_completed(self, juno_id: str) -> bool:
        return self._mark(juno_id, status="completed", deleted=False)

    def mark_deleted(self, juno_id: str) -> bool:
        return self._mark(juno_id, status="deleted", deleted=True)

    def _mark(self, juno_id: str, *, status: str, deleted: bool) -> bool:
        jid = _clean_str(juno_id)
        if not jid:
            return False
        now_ms = int(time.time() * 1000)
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.execute(
                    """
                    UPDATE actions_index
                    SET status = ?,
                        last_modified_at = ?,
                        deleted_at = CASE WHEN ? THEN COALESCE(deleted_at, ?) ELSE deleted_at END
                    WHERE juno_id = ?
                    """,
                    (status, now_ms, 1 if deleted else 0, now_ms, jid),
                )
                con.commit()
                return (cur.rowcount or 0) > 0
            finally:
                con.close()

    def get(self, juno_id: str) -> dict[str, Any] | None:
        jid = _clean_str(juno_id)
        if not jid:
            return None
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            try:
                row = con.execute(
                    "SELECT * FROM actions_index WHERE juno_id = ?",
                    (jid,),
                ).fetchone()
            finally:
                con.close()
        return dict(row) if row is not None else None

    def find(
        self,
        *,
        body_substr: str | None = None,
        date_range: tuple[str, str] | None = None,
        list_name: str | None = None,
        kind: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        clauses = ["deleted_at IS NULL"]
        params: list[Any] = []
        body = _normalize_body(body_substr or "")
        if body:
            clauses.append("body_normalized LIKE ?")
            params.append(f"%{body}%")
        if date_range is not None:
            start, end = date_range
            clauses.append("due_iso >= ? AND due_iso <= ?")
            params.extend([start, end])
        if list_name:
            clauses.append("list_name = ?")
            params.append(list_name)
        if kind:
            clauses.append("sink_kind = ?")
            params.append(kind)
        try:
            n = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            n = 25
        params.append(n)
        sql = (
            "SELECT * FROM actions_index WHERE "
            + " AND ".join(clauses)
            + " ORDER BY last_modified_at DESC, created_at DESC LIMIT ?"
        )
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            try:
                rows = con.execute(sql, tuple(params)).fetchall()
            finally:
                con.close()
        return [dict(row) for row in rows]

    def last_touched(self, *, kind: str | None = None) -> dict[str, Any] | None:
        clauses = ["deleted_at IS NULL"]
        params: list[Any] = []
        if kind:
            clauses.append("sink_kind = ?")
            params.append(kind)
        sql = (
            "SELECT * FROM actions_index WHERE "
            + " AND ".join(clauses)
            + " ORDER BY last_modified_at DESC, created_at DESC LIMIT 1"
        )
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            try:
                row = con.execute(sql, tuple(params)).fetchone()
            finally:
                con.close()
        return dict(row) if row is not None else None

    def prune_deleted_older_than(self, *, days: int = 7) -> int:
        try:
            keep_days = max(0, int(days))
        except (TypeError, ValueError):
            keep_days = 7
        cutoff = int((time.time() - (keep_days * 86400)) * 1000)
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.execute(
                    "DELETE FROM actions_index WHERE deleted_at IS NOT NULL AND deleted_at < ?",
                    (cutoff,),
                )
                con.commit()
                return int(cur.rowcount or 0)
            finally:
                con.close()


_STORES: dict[str, ActionsIndex] = {}


def get_actions_index(workbench_log_dir: Path) -> ActionsIndex:
    key = str((Path(workbench_log_dir) / "actions_index.sqlite").resolve())
    st = _STORES.get(key)
    if st is None:
        st = ActionsIndex(workbench_log_dir)
        _STORES[key] = st
    return st


__all__ = ["ActionsIndex", "get_actions_index"]
