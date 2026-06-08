"""SQLite-backed product utterance history (repair doc P9).

Rows are keyed by ``utterance_id`` and updated as broker-on-pause refines
the same session. Optional one-time import from legacy ``history.jsonl``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
    utterance_id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    app_bundle_id TEXT,
    app_name TEXT,
    window_title TEXT,
    mode TEXT,
    surface_id TEXT,
    transcript TEXT,
    raw_transcript TEXT,
    committed_text TEXT,
    corrected_text TEXT,
    paste_status TEXT,
    paste_kind TEXT,
    words INTEGER,
    processing_ms REAL,
    final_transcription_ms REAL,
    language TEXT,
    language_mode TEXT,
    environment_profile TEXT,
    context_summary_json TEXT,
    context_used_json TEXT,
    audio_path TEXT,
    audio_expires_at INTEGER,
    replay_available INTEGER DEFAULT 0,
    correction_count INTEGER DEFAULT 0,
    failure_reason TEXT,
    deleted_at INTEGER,
    actions_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_utterances_created ON utterances(created_at);
CREATE INDEX IF NOT EXISTS idx_utterances_app ON utterances(app_bundle_id);
"""

# Idempotent column adds for existing databases. Tuple of
# (column_name, ALTER TABLE statement). Each entry is applied only when
# the column is missing — safe to run on every startup.
_SCHEMA_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("actions_json", "ALTER TABLE utterances ADD COLUMN actions_json TEXT"),
    ("final_transcription_ms", "ALTER TABLE utterances ADD COLUMN final_transcription_ms REAL"),
)


def _existing_columns(con: sqlite3.Connection) -> set[str]:
    cur = con.execute("PRAGMA table_info(utterances)")
    return {str(row[1]) for row in cur.fetchall()}

_locks: dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    if key not in _locks:
        _locks[key] = threading.Lock()
    return _locks[key]


def _actions_payload_has_execution_status(raw: str | None) -> bool:
    """Return True only for shell-posted action result payloads.

    Pipeline records store parsed intents without a ``status`` field.
    Shell execution results include ``status`` plus sink metadata. We only
    preserve an existing payload across later pipeline upserts after the
    shell has posted results; otherwise broker-on-pause refinement must still
    be able to improve the parsed action list.
    """

    if not raw:
        return False
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(decoded, list):
        return False
    return any(isinstance(item, dict) and item.get("status") for item in decoded)


class ProductHistoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                con.executescript(_SCHEMA)
                # Idempotent column adds for databases that predate a column.
                # Wrapped per-column so a partial migration (interrupted) is
                # always safe to retry on the next boot.
                cols = _existing_columns(con)
                for col, ddl in _SCHEMA_MIGRATIONS:
                    if col not in cols:
                        try:
                            con.execute(ddl)
                        except sqlite3.OperationalError:
                            # Race with another process that already added it.
                            pass
                con.commit()
            finally:
                con.close()

    def maybe_migrate_jsonl(self, jsonl_path: Path) -> None:
        marker = self.db_path.with_suffix(self.db_path.suffix + ".migrated_jsonl")
        if marker.exists() or not jsonl_path.is_file():
            return
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                try:
                    self.upsert_from_pipeline_record(obj)
                except Exception:
                    continue
        try:
            marker.write_text(str(int(time.time())), encoding="utf-8")
        except OSError:
            pass

    def upsert_from_pipeline_record(self, record: dict[str, Any]) -> None:
        uid = str(record.get("utterance_id") or "").strip()
        if not uid:
            return
        now_ms = int(record.get("ts_unix_ms") or int(time.time() * 1000))
        ctx = record.get("context") if isinstance(record.get("context"), dict) else {}
        app_bundle_id = str(ctx.get("app_bundle_id") or record.get("app_bundle_id") or "").strip() or None
        app_name = str(ctx.get("app_name") or record.get("app_name") or "").strip() or None
        window_title = str(ctx.get("window_title") or record.get("window_title") or "").strip() or None
        summary = {
            "selected_text": bool(ctx.get("selection_chars")),
            "focused_text": bool(ctx.get("focused_text")),
            "clipboard": bool(ctx.get("clipboard")),
            "window_title": bool(window_title),
            "app": bool(app_name or app_bundle_id),
        }
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                row = con.execute(
                    "SELECT created_at, correction_count, actions_json FROM utterances WHERE utterance_id = ?",
                    (uid,),
                ).fetchone()
                created = int(row[0]) if row else now_ms
                prev_corr = int(row[1]) if row and row[1] is not None else 0
                previous_actions_json = str(row[2]) if row and row[2] else None
                corr = int(record.get("correction_count") or prev_corr or 0)
                actions_value = record.get("actions")
                actions_json = (
                    json.dumps(actions_value, ensure_ascii=False)
                    if actions_value
                    else None
                )
                if _actions_payload_has_execution_status(previous_actions_json):
                    # The macOS shell has already posted execution outcomes.
                    # Keep those across later broker-on-pause refinements so
                    # History does not regress from "saved" to "Saving...".
                    stored_actions_json = previous_actions_json
                else:
                    # No execution result yet: allow pipeline refinements to
                    # replace the parsed intent shape until the shell posts
                    # statuses.
                    stored_actions_json = actions_json or previous_actions_json
                con.execute(
                    """
                    INSERT INTO utterances (
                        utterance_id, created_at, updated_at,
                        app_bundle_id, app_name, window_title,
                        mode, surface_id, transcript, raw_transcript,
                        committed_text, corrected_text, paste_status, paste_kind,
                        words, processing_ms, final_transcription_ms, language, language_mode, environment_profile,
                        context_summary_json, context_used_json,
                        audio_path, audio_expires_at, replay_available, correction_count,
                        failure_reason, deleted_at, actions_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(utterance_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        app_bundle_id = COALESCE(excluded.app_bundle_id, utterances.app_bundle_id),
                        app_name = COALESCE(excluded.app_name, utterances.app_name),
                        window_title = COALESCE(excluded.window_title, utterances.window_title),
                        mode = COALESCE(excluded.mode, utterances.mode),
                        surface_id = COALESCE(excluded.surface_id, utterances.surface_id),
                        transcript = excluded.transcript,
                        raw_transcript = excluded.raw_transcript,
                        words = excluded.words,
                        processing_ms = excluded.processing_ms,
                        final_transcription_ms = excluded.final_transcription_ms,
                        context_summary_json = excluded.context_summary_json,
                        context_used_json = excluded.context_used_json,
                        replay_available = MAX(excluded.replay_available, utterances.replay_available),
                        correction_count = MAX(excluded.correction_count, utterances.correction_count),
                        failure_reason = COALESCE(excluded.failure_reason, utterances.failure_reason),
                        actions_json = excluded.actions_json
                    """,
                    (
                        uid,
                        created,
                        now_ms,
                        app_bundle_id,
                        app_name,
                        window_title,
                        str(record.get("mode") or "") or None,
                        str(record.get("surface_id") or "") or None,
                        str(record.get("transcript") or "") or None,
                        str(record.get("raw_transcript") or "") or None,
                        str(record.get("committed_text") or "") or None,
                        str(record.get("corrected_text") or "") or None,
                        str(record.get("paste_status") or "") or None,
                        str(record.get("paste_kind") or "") or None,
                        int(record.get("words") or 0) or 0,
                        float(record.get("processing_ms") or 0) or 0.0,
                        float(record.get("final_transcription_ms") or record.get("decode_ms") or 0) or 0.0,
                        str(record.get("language") or "") or None,
                        str(record.get("language_mode") or "") or None,
                        str(record.get("environment_profile") or "") or None,
                        json.dumps(summary, ensure_ascii=False),
                        json.dumps(ctx, ensure_ascii=False),
                        str(record.get("audio_path") or "") or None,
                        int(record["audio_expires_at"]) if record.get("audio_expires_at") else None,
                        1 if record.get("replay_available") else 0,
                        corr,
                        str(record.get("failure_reason") or "") or None,
                        None,
                        stored_actions_json,
                    ),
                )
                con.commit()
            finally:
                con.close()

    def list_entries(
        self,
        *,
        limit: int = 50,
        before_updated_at_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Newest-first page of utterance rows.

        ``before_updated_at_ms`` (optional) returns only rows older than
        the given updated_at, enabling cursor-based pagination from the
        UI without holding any per-session state in the broker.
        """
        try:
            n = max(0, int(limit))
        except (TypeError, ValueError):
            n = 50
        if n <= 0:
            return []
        cursor_ms: int | None = None
        if before_updated_at_ms is not None:
            try:
                cursor_ms = int(before_updated_at_ms)
                if cursor_ms <= 0:
                    cursor_ms = None
            except (TypeError, ValueError):
                cursor_ms = None
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                if cursor_ms is None:
                    cur = con.execute(
                        """
                        SELECT utterance_id, created_at, updated_at, app_bundle_id, app_name, window_title,
                               mode, surface_id, transcript, raw_transcript, committed_text, corrected_text,
                               paste_status, paste_kind, words, processing_ms, final_transcription_ms, language, language_mode,
                               environment_profile, context_summary_json, context_used_json,
                               audio_path, audio_expires_at, replay_available, correction_count, failure_reason,
                               actions_json
                        FROM utterances
                        WHERE deleted_at IS NULL
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (n,),
                    )
                else:
                    cur = con.execute(
                        """
                        SELECT utterance_id, created_at, updated_at, app_bundle_id, app_name, window_title,
                               mode, surface_id, transcript, raw_transcript, committed_text, corrected_text,
                               paste_status, paste_kind, words, processing_ms, final_transcription_ms, language, language_mode,
                               environment_profile, context_summary_json, context_used_json,
                               audio_path, audio_expires_at, replay_available, correction_count, failure_reason,
                               actions_json
                        FROM utterances
                        WHERE deleted_at IS NULL AND updated_at < ?
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (cursor_ms, n),
                    )
                rows = cur.fetchall()
            finally:
                con.close()
        out: list[dict[str, Any]] = []
        # Column index reference (must match the SELECT above):
        # 0:utterance_id 1:created_at 2:updated_at 3:app_bundle_id
        # 4:app_name 5:window_title 6:mode 7:surface_id 8:transcript
        # 9:raw_transcript 10:committed_text 11:corrected_text
        # 12:paste_status 13:paste_kind 14:words 15:processing_ms
        # 16:final_transcription_ms 17:language 18:language_mode
        # 19:environment_profile 20:context_summary_json 21:context_used_json
        # 22:audio_path 23:audio_expires_at 24:replay_available
        # 25:correction_count 26:failure_reason 27:actions_json
        for r in rows:
            ctx_used: dict[str, Any] = {}
            try:
                ctx_used = json.loads(r[21] or "{}")
            except (json.JSONDecodeError, ValueError):
                ctx_used = {}
            actions_payload: list[dict[str, Any]] | None = None
            raw_actions = r[27] if len(r) > 27 else None
            if raw_actions:
                try:
                    decoded = json.loads(raw_actions)
                    if isinstance(decoded, list):
                        actions_payload = decoded
                except (json.JSONDecodeError, ValueError):
                    actions_payload = None
            entry: dict[str, Any] = {
                "utterance_id": r[0],
                "ts_unix_ms": int(r[2]),
                "created_at_ms": int(r[1]),
                "updated_at_ms": int(r[2]),
                "transcript": r[8] or "",
                "raw_transcript": r[9] or "",
                "committed_text": r[10],
                "corrected_text": r[11],
                "mode": r[6] or "",
                "surface_id": r[7],
                "final_backend": "",
                "model_path": "",
                "context": {
                    "app_name": r[4],
                    "app_bundle_id": r[3],
                    "window_title": r[5],
                    "app_category": None,
                    **ctx_used,
                },
                "failure_reason": r[26],
                "session_class": "insert",
                "processing_ms": r[15],
                "final_transcription_ms": r[16],
                "words": r[14],
                "replay_available": bool(r[24]),
                "paste_kind": r[13],
                "paste_status": r[12],
                "correction_count": r[25],
                "audio_path": r[22],
                "audio_expires_at": r[23],
            }
            if actions_payload is not None:
                entry["actions"] = actions_payload
            out.append(entry)
        return out

    def update_insertion_commit(
        self,
        *,
        utterance_id: str,
        committed_text: str | None,
        ok: bool,
        paste_kind: str | None,
        failure_reason: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Persist shell paste outcome on the product history row.

        The broker trace is append-only, but the History UI reads this SQLite
        table. Paste failures must therefore be visible here even when the
        original pipeline row already exists or the shell races ahead of it.
        """
        uid = (utterance_id or "").strip()
        if not uid:
            return False
        now_ms = int(time.time() * 1000)
        ctx = context if isinstance(context, dict) else {}
        app_bundle_id = str(ctx.get("app_bundle_id") or "").strip() or None
        app_name = str(ctx.get("app_name") or "").strip() or None
        window_title = str(ctx.get("window_title") or "").strip() or None
        summary = {
            "selected_text": bool(ctx.get("selection_chars")),
            "focused_text": bool(ctx.get("focused_text")),
            "clipboard": bool(ctx.get("clipboard")),
            "window_title": bool(window_title),
            "app": bool(app_name or app_bundle_id),
        }
        committed = str(committed_text or "").strip() or None
        status = "pasted" if ok else "failed"
        kind = str(paste_kind or "").strip() or None
        failure = None if ok else (str(failure_reason or "paste_failed").strip() or "paste_failed")
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                row = con.execute(
                    "SELECT created_at FROM utterances WHERE utterance_id = ?",
                    (uid,),
                ).fetchone()
                created = int(row[0]) if row else now_ms
                con.execute(
                    """
                    INSERT INTO utterances (
                        utterance_id, created_at, updated_at,
                        app_bundle_id, app_name, window_title,
                        transcript, committed_text, paste_status, paste_kind,
                        context_summary_json, context_used_json,
                        failure_reason, deleted_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(utterance_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        app_bundle_id = COALESCE(excluded.app_bundle_id, utterances.app_bundle_id),
                        app_name = COALESCE(excluded.app_name, utterances.app_name),
                        window_title = COALESCE(excluded.window_title, utterances.window_title),
                        transcript = COALESCE(utterances.transcript, excluded.transcript),
                        committed_text = excluded.committed_text,
                        paste_status = excluded.paste_status,
                        paste_kind = excluded.paste_kind,
                        context_summary_json = COALESCE(excluded.context_summary_json, utterances.context_summary_json),
                        context_used_json = COALESCE(excluded.context_used_json, utterances.context_used_json),
                        failure_reason = COALESCE(excluded.failure_reason, utterances.failure_reason)
                    """,
                    (
                        uid,
                        created,
                        now_ms,
                        app_bundle_id,
                        app_name,
                        window_title,
                        committed,
                        committed,
                        status,
                        kind,
                        json.dumps(summary, ensure_ascii=False),
                        json.dumps(ctx, ensure_ascii=False),
                        failure,
                        None,
                    ),
                )
                con.commit()
                return True
            finally:
                con.close()

    def patch_paste_destination(
        self,
        utterance_id: str,
        *,
        app_bundle_id: str | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
    ) -> bool:
        """Update app columns from paste-time metadata without touching transcript text.

        Used when the macOS shell posts ``/api/broker/insertion/committed`` with
        optional ``paste_app_*`` fields so Product History reflects where Cmd+V landed.
        ``None`` for a field means leave the existing column value unchanged.
        """
        uid = (utterance_id or "").strip()
        if not uid:
            return False
        now_ms = int(time.time() * 1000)
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.execute(
                    """
                    UPDATE utterances SET
                        updated_at = ?,
                        app_bundle_id = COALESCE(?, app_bundle_id),
                        app_name = COALESCE(?, app_name),
                        window_title = COALESCE(?, window_title)
                    WHERE utterance_id = ?
                    """,
                    (now_ms, app_bundle_id, app_name, window_title, uid),
                )
                con.commit()
                return cur.rowcount > 0
            finally:
                con.close()

    def update_actions(self, utterance_id: str, actions: list[dict[str, Any]] | None) -> bool:
        """Overwrite ``actions_json`` for the row, creating a stub if missing.

        Called by the macOS shell after it has dispatched parsed actions to
        their sinks (Reminders / Notes / Calendar) and produced execution
        results (sink IDs, deep-link URLs, status, errors).

        **Race tolerance:** the shell can post results before the broker
        has finished writing the history row (rare but possible on slow
        SQLite locks or if the shell is talking to a freshly-restarted
        broker that received a queued action post). Instead of returning
        ``not_found`` and dropping the result on the floor — which is what
        leaves rows stuck on "pending" forever — we INSERT a stub row
        keyed by ``utterance_id`` carrying just the actions and timestamps.
        The next pipeline upsert fills in transcript / context / etc. via
        ``ON CONFLICT DO UPDATE``, and the COALESCE rule on ``actions_json``
        keeps the result payload alive across that update.
        """

        uid = (utterance_id or "").strip()
        if not uid:
            return False
        payload = (
            json.dumps(actions, ensure_ascii=False) if actions else None
        )
        now_ms = int(time.time() * 1000)
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                con.execute(
                    """
                    INSERT INTO utterances (
                        utterance_id, created_at, updated_at, actions_json
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(utterance_id) DO UPDATE SET
                        actions_json = excluded.actions_json,
                        updated_at = excluded.updated_at
                    """,
                    (uid, now_ms, now_ms, payload),
                )
                con.commit()
                return True
            finally:
                con.close()

    def get_entry_text(self, utterance_id: str) -> dict[str, Any] | None:
        """Return ``{transcript, committed_text, raw_transcript}`` for *uid*.

        Used by the History "Insert again" recovery action: the shell calls
        the broker to fetch the saved transcript for a row whose original
        paste failed, then re-pastes via the existing capability path.
        Returns ``None`` if the row is missing or soft-deleted.
        """
        uid = (utterance_id or "").strip()
        if not uid:
            return None
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                row = con.execute(
                    """
                    SELECT transcript, committed_text, raw_transcript,
                           app_bundle_id, app_name, window_title
                    FROM utterances
                    WHERE utterance_id = ? AND deleted_at IS NULL
                    """,
                    (uid,),
                ).fetchone()
            finally:
                con.close()
        if row is None:
            return None
        return {
            "transcript": row[0] or "",
            "committed_text": row[1] or "",
            "raw_transcript": row[2] or "",
            "app_bundle_id": row[3] or "",
            "app_name": row[4] or "",
            "window_title": row[5] or "",
        }

    def delete_utterance(self, utterance_id: str) -> bool:
        uid = (utterance_id or "").strip()
        if not uid:
            return False
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.execute("DELETE FROM utterances WHERE utterance_id = ?", (uid,))
                con.commit()
                return cur.rowcount > 0
            finally:
                con.close()

    def clear_all(self) -> None:
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                con.execute("DELETE FROM utterances")
                con.commit()
            finally:
                con.close()

    def prune_by_age_ms(self, *, keep_days: int) -> dict[str, Any]:
        if keep_days <= 0:
            self.clear_all()
            return {"ok": True, "kept": 0, "removed": 0}
        cutoff = int((time.time() - (keep_days * 86400)) * 1000)
        with _lock_for(self.db_path):
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.execute("DELETE FROM utterances WHERE updated_at < ?", (cutoff,))
                removed = cur.rowcount or 0
                con.commit()
                cur2 = con.execute("SELECT COUNT(*) FROM utterances")
                kept = int(cur2.fetchone()[0])
            finally:
                con.close()
        return {"ok": True, "kept": kept, "removed": removed}

    def stats(self) -> tuple[int, int]:
        with _lock_for(self.db_path):
            if not self.db_path.exists():
                return 0, 0
            try:
                sz = int(self.db_path.stat().st_size)
            except OSError:
                sz = 0
            con = sqlite3.connect(self.db_path)
            try:
                cur = con.execute("SELECT COUNT(*) FROM utterances WHERE deleted_at IS NULL")
                n = int(cur.fetchone()[0])
            finally:
                con.close()
        return n, sz


_STORES: dict[str, ProductHistoryStore] = {}


def get_product_history_store(workbench_log_dir: Path) -> ProductHistoryStore:
    from juno_v2.runtime.paths import product_history_db_path

    db = product_history_db_path(workbench_log_dir=workbench_log_dir)
    key = str(db.resolve())
    st = _STORES.get(key)
    if st is None:
        st = ProductHistoryStore(db)
        st.init_schema()
        st.maybe_migrate_jsonl(Path(workbench_log_dir) / "history.jsonl")
        _STORES[key] = st
    return st


def increment_correction_count(workbench_log_dir: Path, utterance_id: str) -> None:
    uid = (utterance_id or "").strip()
    if not uid:
        return
    st = get_product_history_store(workbench_log_dir)
    with _lock_for(st.db_path):
        con = sqlite3.connect(st.db_path)
        try:
            con.execute(
                "UPDATE utterances SET correction_count = correction_count + 1, updated_at = ? WHERE utterance_id = ?",
                (int(time.time() * 1000), uid),
            )
            con.commit()
        finally:
            con.close()


__all__ = [
    "ProductHistoryStore",
    "get_product_history_store",
    "increment_correction_count",
]
