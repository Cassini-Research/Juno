from __future__ import annotations

from juno_v2.contracts.memory import ReplacementRule
from juno_v2.memory.fold import fold_key
from juno_v2.memory.stores._base import JsonFileStore


class ReplacementStore:
    """Trigger -> replacement rules applied at the ASR/writer boundary.

    Stored as ``replacements.json``. The editable identity is
    ``(fold_key(trigger), scope, case_sensitive)`` for case-insensitive
    rules and ``(trigger, scope, True)`` for case-sensitive rules, so
    ``"my email"``, ``"my-email"`` and ``"My  Email"`` all resolve to one
    row and editing replacement text updates that row instead of leaving
    parallel near-duplicates.
    """

    FILENAME = "replacements.json"

    def __init__(self, file_store: JsonFileStore) -> None:
        self._fs = file_store
        self._fs.ensure_default([])

    def list(self) -> list[ReplacementRule]:
        return [ReplacementRule(**item) for item in self._fs.read([])]

    def raw(self) -> list[dict]:
        return list(self._fs.read([]))

    @staticmethod
    def _key(trigger: str, scope: str, *, case_sensitive: bool) -> tuple[str, str, bool]:
        return (
            trigger if case_sensitive else fold_key(trigger),
            scope,
            bool(case_sensitive),
        )

    def add(
        self,
        *,
        trigger: str,
        replacement: str,
        scope: str = "global",
        case_sensitive: bool = False,
        source: str = "user",
    ) -> None:
        trigger = (trigger or "").strip()
        if not trigger:
            return
        if not case_sensitive and not fold_key(trigger):
            # All-punctuation trigger — would match every empty key.
            return
        key = self._key(trigger, scope, case_sensitive=case_sensitive)
        with self._fs.lock:
            data = self._fs.read([])
            for idx, item in enumerate(data):
                existing_trigger = str(item.get("trigger", ""))
                existing_cs = bool(item.get("case_sensitive", False))
                existing_key = self._key(
                    existing_trigger,
                    str(item.get("scope", "global")),
                    case_sensitive=existing_cs,
                )
                if existing_key == key:
                    data[idx] = ReplacementRule(
                        trigger=trigger,
                        replacement=replacement,
                        scope=scope,
                        case_sensitive=case_sensitive,
                        source=source,
                    ).to_dict()
                    self._fs.write(data)
                    return
            data.append(
                ReplacementRule(
                    trigger=trigger,
                    replacement=replacement,
                    scope=scope,
                    case_sensitive=case_sensitive,
                    source=source,
                ).to_dict()
            )
            self._fs.write(data)

    def remove(
        self,
        trigger: str,
        *,
        scope: str = "global",
        case_sensitive: bool = False,
    ) -> bool:
        """Delete every rule matching ``(trigger, scope)``. Case-sensitive
        callers match character-for-character; case-insensitive callers
        match the trigger via :func:`fold_key`, so ``remove("Sign Off")``
        clears rows stored as ``"signoff"``, ``"Sign-Off"``, etc.
        """
        needle_raw = trigger or ""
        if not needle_raw:
            return False
        needle_fold = fold_key(needle_raw)
        with self._fs.lock:
            data = self._fs.read([])
            before = len(data)
            kept: list[dict] = []
            for item in data:
                item_trigger = str(item.get("trigger", ""))
                item_scope = str(item.get("scope", "global"))
                item_cs = bool(item.get("case_sensitive", False))
                if item_scope != scope:
                    kept.append(item)
                    continue
                if case_sensitive or item_cs:
                    if item_trigger == needle_raw:
                        continue
                else:
                    if fold_key(item_trigger) == needle_fold and needle_fold:
                        continue
                kept.append(item)
            if len(kept) == before:
                return False
            self._fs.write(kept)
            return True
