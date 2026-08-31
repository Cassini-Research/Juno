from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from juno_v2.observability.reliability_provenance import test_provenance_fields


@dataclass(frozen=True, slots=True)
class HistoryPaths:
    log_dir: Path

    @property
    def history_path(self) -> Path:
        return self.log_dir / "history.jsonl"


def _safe_json_dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def append_history_record(log_dir: Path | str, record: dict[str, Any]) -> None:
    """Persist utterance history (SQLite product store; optional JSONL mirror)."""
    paths = HistoryPaths(Path(log_dir))
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("ts_unix_ms", int(time.time() * 1000))
    try:
        from juno_v2.observability.product_history import get_product_history_store

        get_product_history_store(Path(log_dir)).upsert_from_pipeline_record(payload)
    except Exception:
        # Never block dictation on history persistence.
        pass
    if os.environ.get("JUNO_DUAL_WRITE_HISTORY_JSONL", "").strip().lower() in {"1", "true", "yes"}:
        line = _safe_json_dumps(payload) + "\n"
        with paths.history_path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def read_persistent_history(
    log_dir: Path | str,
    *,
    limit: int = 50,
    before_updated_at_ms: int | None = None,
    test_run_id: str | None = None,
    test_case_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read newest ``limit`` records (SQLite first; legacy JSONL fallback).

    ``before_updated_at_ms`` (optional) returns rows older than the cursor;
    used by the History UI to load a next page without holding broker state.
    Pagination only works on the SQLite path; the JSONL fallback ignores
    the cursor (legacy installs only).
    """
    try:
        n = max(0, int(limit))
    except (TypeError, ValueError):
        n = 50
    if n <= 0:
        return []
    test_provenance = test_provenance_fields(
        test_run_id=test_run_id,
        test_case_id=test_case_id,
    )
    if test_run_id is not None and "test_run_id" not in test_provenance:
        return []
    if test_case_id is not None and "test_case_id" not in test_provenance:
        return []
    try:
        from juno_v2.observability.product_history import get_product_history_store

        rows = get_product_history_store(Path(log_dir)).list_entries(
            limit=n,
            before_updated_at_ms=before_updated_at_ms,
            test_run_id=test_provenance.get("test_run_id"),
            test_case_id=test_provenance.get("test_case_id"),
        )
        if rows:
            return rows
        # When using a cursor, an empty page is a legitimate end-of-list,
        # not a "fall through to legacy JSONL" signal.
        if before_updated_at_ms is not None:
            return []
    except Exception:
        pass
    paths = HistoryPaths(Path(log_dir))
    if not paths.history_path.exists():
        return []
    lines = paths.history_path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if any(obj.get(key) != value for key, value in test_provenance.items()):
                continue
            out.append(obj)
        if len(out) >= n:
            break
    return out


def resolve_history_utterance_id(log_dir: Path | str, *, utterance_id: str) -> str | None:
    """Resolve a shell-visible utterance id to the stored canonical id.

    Prefers exact matches, then accepts unique prefix matches so the broker delete
    path can stay tolerant if a shell row is holding an abbreviated or stale copy.
    """
    uid = (utterance_id or "").strip()
    if not uid:
        return None
    try:
        from juno_v2.observability.product_history import get_product_history_store

        store = get_product_history_store(Path(log_dir))
        db_path = store.db_path
        lock = None
        try:
            from juno_v2.observability.product_history import _lock_for  # type: ignore

            lock = _lock_for(db_path)
        except Exception:
            lock = None
        if lock is not None:
            lock.acquire()
        try:
            import sqlite3

            con = sqlite3.connect(db_path)
            try:
                row = con.execute(
                    "SELECT utterance_id FROM utterances WHERE deleted_at IS NULL AND utterance_id = ?",
                    (uid,),
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
                rows = con.execute(
                    "SELECT utterance_id FROM utterances WHERE deleted_at IS NULL AND utterance_id LIKE ? LIMIT 2",
                    (f"{uid}%",),
                ).fetchall()
                if len(rows) == 1 and rows[0][0]:
                    return str(rows[0][0])
            finally:
                con.close()
        finally:
            if lock is not None:
                lock.release()
    except Exception:
        pass

    paths = HistoryPaths(Path(log_dir))
    if not paths.history_path.exists():
        return None
    exact: str | None = None
    prefix_matches: list[str] = []
    for line in paths.history_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        candidate = str(obj.get("utterance_id") or "").strip()
        if not candidate:
            continue
        if candidate == uid:
            exact = candidate
            break
        if candidate.startswith(uid):
            prefix_matches.append(candidate)
            if len(prefix_matches) > 1:
                break
    if exact:
        return exact
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def delete_history_entry(log_dir: Path | str, *, utterance_id: str) -> bool:
    """Remove utterance from SQLite product history and legacy JSONL."""
    uid = resolve_history_utterance_id(log_dir, utterance_id=utterance_id) or (utterance_id or "").strip()
    if not uid:
        return False
    changed = False
    try:
        from juno_v2.observability.product_history import get_product_history_store

        if get_product_history_store(Path(log_dir)).delete_utterance(uid):
            changed = True
    except Exception:
        pass
    paths = HistoryPaths(Path(log_dir))
    if not paths.history_path.exists():
        return changed

    jsonl_changed = False
    keep: list[str] = []
    for line in paths.history_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            keep.append(raw)
            continue
        if isinstance(obj, dict) and str(obj.get("utterance_id") or "") == uid:
            jsonl_changed = True
            continue
        keep.append(raw)

    if jsonl_changed:
        tmp = paths.history_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        tmp.replace(paths.history_path)
        changed = True
    return changed


def compute_storage_stats(
    *,
    audio_dir: Path | None,
    history_path: Path,
) -> dict[str, Any]:
    """Compute storage stats for Settings/UI surfaces."""
    audio_files = 0
    audio_bytes = 0
    oldest_audio_ts: int | None = None
    if audio_dir is not None and audio_dir.exists():
        for p in audio_dir.glob("*.wav"):
            try:
                st = p.stat()
            except OSError:
                continue
            audio_files += 1
            audio_bytes += int(st.st_size)
            ts = int(st.st_mtime * 1000)
            oldest_audio_ts = ts if oldest_audio_ts is None else min(oldest_audio_ts, ts)

    history_entries = 0
    history_bytes = 0
    if history_path.exists():
        try:
            history_bytes = int(history_path.stat().st_size)
        except OSError:
            history_bytes = 0
        try:
            with history_path.open("r", encoding="utf-8") as fh:
                for _ in fh:
                    history_entries += 1
        except OSError:
            history_entries = 0

    return {
        "audio_files": audio_files,
        "audio_bytes": audio_bytes,
        "oldest_audio_ts": oldest_audio_ts,
        "history_entries": history_entries,
        "history_bytes": history_bytes,
    }


def prune_persistent_history_by_days(log_dir: Path | str, *, keep_days: int) -> dict[str, Any]:
    """Prune product SQLite history and legacy ``history.jsonl`` by age (best-effort)."""
    paths = HistoryPaths(Path(log_dir))
    sql_out: dict[str, Any] = {"ok": True, "kept": 0, "removed": 0}
    try:
        from juno_v2.observability.product_history import get_product_history_store

        sql_out = get_product_history_store(Path(log_dir)).prune_by_age_ms(keep_days=int(keep_days))
    except Exception:
        sql_out = {"ok": False, "error": "sqlite_prune_failed"}
    if not paths.history_path.exists():
        return sql_out
    try:
        days = int(keep_days)
    except (TypeError, ValueError):
        return {"ok": False, "error": "keep_days_must_be_int"}
    if days <= 0:
        try:
            paths.history_path.write_text("", encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": str(exc), "sqlite": sql_out}
        return {"ok": True, "sqlite": sql_out, "jsonl": {"ok": True, "kept": 0, "removed": 0}}

    cutoff_ms = int((time.time() - (days * 86400)) * 1000)
    kept_lines: list[str] = []
    kept = 0
    removed = 0
    for line in paths.history_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Keep malformed lines rather than accidentally destroying history.
            kept_lines.append(raw)
            kept += 1
            continue
        if not isinstance(obj, dict):
            kept_lines.append(raw)
            kept += 1
            continue
        ts = obj.get("ts_unix_ms")
        try:
            ts_int = int(ts)
        except (TypeError, ValueError):
            ts_int = 0
        if ts_int and ts_int < cutoff_ms:
            removed += 1
            continue
        kept_lines.append(raw)
        kept += 1

    tmp = paths.history_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    tmp.replace(paths.history_path)
    jsonl_out = {"ok": True, "kept": kept, "removed": removed}
    return {"ok": True, "sqlite": sql_out, "jsonl": jsonl_out}


__all__ = [
    "append_history_record",
    "read_persistent_history",
    "resolve_history_utterance_id",
    "delete_history_entry",
    "compute_storage_stats",
    "prune_persistent_history_by_days",
]
