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

    @staticmethod
    def _default_data() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "context_entities": [],
            "seed_replacement_overrides": {},
            "disabled_seed_replacement_ids": [],
        }

    def __init__(self, memory_dir: Path | str) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._fs = JsonFileStore(self.memory_dir, self.FILENAME, lock=threading.RLock())
        self._fs.ensure_default(self._default_data())

    def _read(self) -> dict[str, Any]:
        data = self._fs.read({})
        if not isinstance(data, dict):
            return self._default_data()
        data.setdefault("schema_version", 1)
        data.setdefault("context_entities", [])
        data.setdefault("seed_replacement_overrides", {})
        data.setdefault("disabled_seed_replacement_ids", [])
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self._fs.write(data)

    def clear(self) -> None:
        with self._fs.lock:
            self._write(self._default_data())

    def seed_replacement_overrides(self) -> dict[str, dict[str, str]]:
        data = self._read()
        raw = data.get("seed_replacement_overrides")
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, str]] = {}
        for rule_id, value in raw.items():
            if not isinstance(value, dict):
                continue
            trigger = str(value.get("trigger") or "").strip()
            replacement = str(value.get("replacement") or "").strip()
            if trigger and replacement:
                out[str(rule_id)] = {
                    "trigger": trigger,
                    "replacement": replacement,
                }
        return out

    def disabled_seed_replacement_ids(self) -> frozenset[str]:
        data = self._read()
        raw = data.get("disabled_seed_replacement_ids")
        if not isinstance(raw, list):
            return frozenset()
        return frozenset(str(value) for value in raw if str(value).strip())

    def set_seed_replacement_override(
        self,
        rule_id: str,
        *,
        trigger: str,
        replacement: str,
    ) -> None:
        key = (rule_id or "").strip()
        new_trigger = (trigger or "").strip()
        new_replacement = (replacement or "").strip()
        if not key or not new_trigger or not new_replacement:
            raise ValueError("seed replacement id, trigger, and replacement are required")
        with self._fs.lock:
            data = self._read()
            raw_overrides = data.get("seed_replacement_overrides")
            overrides = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
            overrides[key] = {
                "trigger": new_trigger,
                "replacement": new_replacement,
            }
            data["seed_replacement_overrides"] = overrides
            raw_disabled = data.get("disabled_seed_replacement_ids")
            disabled = (
                {str(value) for value in raw_disabled if str(value).strip()}
                if isinstance(raw_disabled, list)
                else set()
            )
            disabled.discard(key)
            data["disabled_seed_replacement_ids"] = sorted(disabled)
            self._write(data)

    def disable_seed_replacement(self, rule_id: str) -> bool:
        key = (rule_id or "").strip()
        if not key:
            return False
        with self._fs.lock:
            data = self._read()
            raw_disabled = data.get("disabled_seed_replacement_ids")
            disabled = (
                {str(value) for value in raw_disabled if str(value).strip()}
                if isinstance(raw_disabled, list)
                else set()
            )
            already_disabled = key in disabled
            disabled.add(key)
            data["disabled_seed_replacement_ids"] = sorted(disabled)
            raw_overrides = data.get("seed_replacement_overrides")
            overrides = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
            overrides.pop(key, None)
            data["seed_replacement_overrides"] = overrides
            self._write(data)
            return not already_disabled

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
