from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from juno_v2.memory.term_policy import learned_term_allowed
from juno_v2.memory.stores._base import JsonFileStore


@dataclass(slots=True)
class ContextEntityObservation:
    token: str
    observation_count: int
    acceptance_count: int


class JunoPersonalizationLearnedStore:
    """Separate durable JSON state for seed-adjacent learning (not seed files).

    Lives under the same ``memory_dir`` as :class:`~juno_v2.memory.store.JsonMemoryStore`.
    """

    FILENAME = "juno_personalization_learned.json"

    def __init__(self, memory_dir: Path | str) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._fs = JsonFileStore(self.memory_dir, self.FILENAME, lock=threading.RLock())
        self._fs.ensure_default({"schema_version": 1, "context_entities": []})

    def _read(self) -> dict[str, Any]:
        data = self._fs.read({})
        if not isinstance(data, dict):
            return {"schema_version": 1, "context_entities": []}
        data.setdefault("schema_version", 1)
        data.setdefault("context_entities", [])
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self._fs.write(data)

    def clear(self) -> None:
        with self._fs.lock:
            self._write({"schema_version": 1, "context_entities": []})

    def increment_observation(self, token: str, *, from_suppressed_context: bool) -> None:
        """Count a surface-context entity observed in a transcript.

        Observations from suppressed contexts are tracked separately and never promote.
        """
        t = (token or "").strip()
        if not learned_term_allowed(t):
            return
        key = t.casefold()
        with self._fs.lock:
            data = self._read()
            rows: list[dict[str, Any]] = list(data.get("context_entities") or [])
            found = False
            for i, row in enumerate(rows):
                if str(row.get("token", "")).casefold() == key:
                    if from_suppressed_context:
                        rows[i] = {
                            **row,
                            "suppressed_observation_count": int(row.get("suppressed_observation_count", 0)) + 1,
                        }
                    else:
                        rows[i] = {**row, "observation_count": int(row.get("observation_count", 0)) + 1}
                    found = True
                    break
            if not found:
                rows.append(
                    {
                        "token": t,
                        "observation_count": 0 if from_suppressed_context else 1,
                        "acceptance_count": 0,
                        "suppressed_observation_count": 1 if from_suppressed_context else 0,
                    }
                )
            data["context_entities"] = rows
            self._write(data)

    def increment_acceptance(self, token: str, *, from_suppressed_context: bool) -> None:
        t = (token or "").strip()
        if not learned_term_allowed(t):
            return
        if from_suppressed_context:
            return
        key = t.casefold()
        with self._fs.lock:
            data = self._read()
            rows = list(data.get("context_entities") or [])
            found = False
            for i, row in enumerate(rows):
                if str(row.get("token", "")).casefold() == key:
                    rows[i] = {**row, "acceptance_count": int(row.get("acceptance_count", 0)) + 1}
                    found = True
                    break
            if not found:
                rows.append({"token": t, "observation_count": 0, "acceptance_count": 1, "suppressed_observation_count": 0})
            data["context_entities"] = rows
            self._write(data)

    def observation_snapshot(self, token: str) -> ContextEntityObservation | None:
        key = (token or "").strip().casefold()
        if not key:
            return None
        data = self._read()
        for row in data.get("context_entities") or []:
            if str(row.get("token", "")).casefold() == key:
                return ContextEntityObservation(
                    token=str(row.get("token", "")),
                    observation_count=int(row.get("observation_count", 0)),
                    acceptance_count=int(row.get("acceptance_count", 0)),
                )
        return None
