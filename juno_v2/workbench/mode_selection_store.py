"""Persistence for ``manual_writer_mode`` / ``custom_writer_mode``.

Issue #9 (audit 2026-05-07): the workbench held both fields in memory
only, so a broker restart silently dropped the user's mode choice. This
module stores them in a small JSON file under
``<log_dir>/juno_workbench_data/writer_mode_selection.json`` and is
read once at :class:`juno_v2.workbench.store.WorkbenchStore`
construction time.

Coerce-clear (issue #5 interaction): the broker rejects
``default_surface`` as a manual mode (it is the AUTO fallback policy,
not a clickable mode). Stale installs may have ``default_surface`` in
the persisted file from before #5 landed; on load we drop it (and any
other non-manually-selectable / unknown id) so the bad value does not
re-surface as a pinned manual override.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def default_writer_mode_selection_path(log_dir: Path | str) -> Path:
    return Path(log_dir) / "juno_workbench_data" / "writer_mode_selection.json"


class WriterModeSelectionStore:
    """Tiny JSON store for the user's mode selection.

    Schema::

        {"manual": <str|null>, "custom": <str|null>}

    Writes are atomic via tempfile + ``os.replace``. JSON parse errors
    are tolerated: the store treats a corrupt file as empty so a single
    bad write cannot brick the broker. Write failures propagate to the
    caller so the broker can trace them instead of pretending persistence
    succeeded.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read_raw(self) -> dict[str, Any]:
        with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}

    def _write_raw(self, data: dict[str, Any]) -> None:
        with self._lock:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)

    def load(self) -> tuple[str | None, str | None]:
        """Return ``(manual, custom)`` with #5 coerce-clear applied to manual."""
        raw = self._read_raw()
        manual = raw.get("manual") if isinstance(raw, dict) else None
        custom = raw.get("custom") if isinstance(raw, dict) else None
        manual_coerced = _coerce_manual(manual)
        custom_str = (custom or None) if isinstance(custom, str) and custom.strip() else None
        return manual_coerced, custom_str

    def save(self, *, manual: str | None, custom: str | None) -> None:
        self._write_raw({"manual": manual, "custom": custom})


def _coerce_manual(value: Any) -> str | None:
    """Drop anything that isn't a manually-selectable built-in mode.

    Importing the modes table here keeps the store decoupled from the
    rest of the workbench package at import time (``mode_selection_store``
    has no other workbench-side imports).
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name:
        return None
    from juno_v2.modes.defaults import BUILTIN_MODES

    policy = BUILTIN_MODES.get(name)
    if policy is None or not getattr(policy, "manual_selectable", False):
        return None
    return name
