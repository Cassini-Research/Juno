from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from juno_v2.contracts.modes import CustomModeRecord


def default_modes_data_path(log_dir: Path | str) -> Path:
    """Stable JSON path under the workbench / session log root."""
    return Path(log_dir) / "juno_workbench_data" / "custom_modes.json"


class CustomModeStore:
    """Persist custom modes as JSON (not ad hoc loose dicts)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write_raw({"schema_version": 1, "modes": []})

    def _read_raw(self) -> dict[str, Any]:
        with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"schema_version": 1, "modes": []}

    def _write_raw(self, data: dict[str, Any]) -> None:
        with self._lock:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_modes(self) -> list[CustomModeRecord]:
        raw = self._read_raw()
        out: list[CustomModeRecord] = []
        for item in raw.get("modes", []):
            if isinstance(item, dict):
                out.append(CustomModeRecord.from_dict(item))
        return out

    def get(self, name: str) -> CustomModeRecord | None:
        key = (name or "").strip()
        if not key:
            return None
        for m in self.list_modes():
            if m.name.casefold() == key.casefold():
                return m
        return None

    def upsert(self, record: CustomModeRecord) -> None:
        modes = self.list_modes()
        key = record.name.strip()
        if not key:
            return
        replaced = False
        new_list: list[CustomModeRecord] = []
        for m in modes:
            if m.name.casefold() == key.casefold():
                new_list.append(record)
                replaced = True
            else:
                new_list.append(m)
        if not replaced:
            new_list.append(record)
        self._write_raw({"schema_version": 1, "modes": [m.to_dict() for m in new_list]})

    def delete(self, name: str) -> bool:
        key = (name or "").strip()
        if not key:
            return False
        modes = self.list_modes()
        new_list = [m for m in modes if m.name.casefold() != key.casefold()]
        if len(new_list) == len(modes):
            return False
        self._write_raw({"schema_version": 1, "modes": [m.to_dict() for m in new_list]})
        return True
