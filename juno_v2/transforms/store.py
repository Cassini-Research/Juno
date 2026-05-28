from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from juno_v2.contracts.transforms import CustomTransformRecord


def default_transforms_data_path(log_dir: Path | str) -> Path:
    return Path(log_dir) / "juno_workbench_data" / "custom_transforms.json"


class CustomTransformStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write({"schema_version": 1, "transforms": []})

    def _read(self) -> dict[str, Any]:
        with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"schema_version": 1, "transforms": []}

    def _write(self, data: dict[str, Any]) -> None:
        with self._lock:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_all(self) -> list[CustomTransformRecord]:
        raw = self._read()
        return [CustomTransformRecord.from_dict(x) for x in raw.get("transforms", []) if isinstance(x, dict)]

    def get(self, name: str) -> CustomTransformRecord | None:
        key = (name or "").strip().casefold()
        for t in self.list_all():
            if t.name.casefold() == key:
                return t
        return None

    def upsert(self, record: CustomTransformRecord) -> None:
        items = self.list_all()
        key = record.name.strip()
        out = [t for t in items if t.name.casefold() != key.casefold()]
        out.append(record)
        self._write({"schema_version": 1, "transforms": [t.to_dict() for t in out]})

    def delete(self, name: str) -> bool:
        items = self.list_all()
        key = (name or "").strip().casefold()
        new_items = [t for t in items if t.name.casefold() != key]
        if len(new_items) == len(items):
            return False
        self._write({"schema_version": 1, "transforms": [t.to_dict() for t in new_items]})
        return True
