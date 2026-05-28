from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(slots=True)
class RecoveryEntry:
    """Append-only recovery record (local disk, inspectable)."""

    ts_unix_ms: int
    broker_session_id: str
    kind: str  # committed | staged_fallback | retry_applied | capture_note
    utterance_id: str | None
    text: str
    metadata: dict[str, Any]

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False) + "\n"

    @staticmethod
    def from_json_line(line: str) -> "RecoveryEntry":
        data = json.loads(line)
        return RecoveryEntry(
            ts_unix_ms=int(data["ts_unix_ms"]),
            broker_session_id=str(data["broker_session_id"]),
            kind=str(data["kind"]),
            utterance_id=data.get("utterance_id"),
            text=str(data.get("text") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


class RecoveryLedger:
    """Append-only JSONL ledger under ``recovery_root / broker_session_id / ledger.jsonl``."""

    def __init__(self, *, recovery_root: Path, broker_session_id: str) -> None:
        self.recovery_root = Path(recovery_root)
        self.broker_session_id = broker_session_id
        self.session_dir = self.recovery_root / broker_session_id
        self.path = self.session_dir / "ledger.jsonl"

    def append(self, entry: RecoveryEntry) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.to_json_line())

    def iter_entries(self) -> Iterator[RecoveryEntry]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield RecoveryEntry.from_json_line(line)

    def read_all(self) -> list[RecoveryEntry]:
        return list(self.iter_entries())
