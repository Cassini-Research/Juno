from __future__ import annotations

from juno_v2.contracts.memory import SessionEntity
from juno_v2.memory.entity_policy import session_entity_allowed
from juno_v2.memory.fold import fold_key
from juno_v2.memory.stores._base import JsonFileStore


class EntityStore:
    """Session-bounded named entities (people, project codenames, ...).

    Stored as ``session_entities.json``. These decay implicitly because the
    serving packet ranks by count and recency, and because the broker can
    clear them between sessions.
    """

    FILENAME = "session_entities.json"

    def __init__(self, file_store: JsonFileStore) -> None:
        self._fs = file_store
        self._fs.ensure_default([])

    def list(self) -> list[SessionEntity]:
        return [SessionEntity(**item) for item in self._fs.read([])]

    def raw(self) -> list[dict]:
        return list(self._fs.read([]))

    def upsert_many(self, entities: list[str], *, source: str = "session") -> None:
        clean = [e.strip() for e in entities if e and e.strip() and session_entity_allowed(e)]
        if not clean:
            return
        with self._fs.lock:
            data = self._fs.read([])
            index: dict[str, int] = {}
            for i, item in enumerate(data):
                k = fold_key(str(item.get("value", "")))
                if k:
                    index[k] = i
            for entity in clean:
                key = fold_key(entity)
                if not key:
                    continue
                if key in index:
                    pos = index[key]
                    data[pos] = {
                        **data[pos],
                        "value": entity,
                        "count": int(data[pos].get("count", 1)) + 1,
                        "source": source,
                    }
                else:
                    index[key] = len(data)
                    data.append(
                        SessionEntity(value=entity, count=1, source=source).to_dict()
                    )
            self._fs.write(data)
