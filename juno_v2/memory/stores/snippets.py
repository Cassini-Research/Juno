"""User-defined snippets expanded by the writer.

A snippet is a short trigger phrase ("brb", "sig") that expands to a canned
body of text at insertion time. Unlike :class:`ReplacementStore` rules,
snippets are scoped by app category and optionally by explicit trigger,
and are applied *by the writer* after the ASR finalises — so the user
sees the expansion, not the raw trigger.

Stored as ``snippets.json``.

Match semantics: a stored trigger of ``"signoff"`` matches spoken text
``"sign off"``, ``"Sign-Off"``, ``"signoff"``, etc. The store dedupes by
``fold_key(trigger)``, so ``add("Sign Off", ...)`` over an existing
``"signoff"`` entry updates that single row rather than creating a
parallel one. ``case_sensitive=True`` bypasses fold matching entirely.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from juno_v2.memory.fold import fold_key
from juno_v2.memory.stores._base import JsonFileStore


@dataclass(slots=True)
class Snippet:
    trigger: str
    body: str
    scope: str = "global"  # "global" | "messaging" | "email" | "code" | "terminal" | "docs"
    case_sensitive: bool = False
    source: str = "user"
    description: str = ""
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _key_for(trigger: str, scope: str, *, case_sensitive: bool) -> tuple[str, str]:
    """Build the dedup/lookup key for a snippet row.

    For case-sensitive triggers we keep the exact string (so ``"Foo"`` and
    ``"foo"`` are distinct entries). For case-insensitive triggers — the
    common case — we collapse via :func:`fold_key` so whitespace,
    punctuation, and case variants all resolve to the same row.
    """
    return (trigger if case_sensitive else fold_key(trigger), scope)


class SnippetStore:
    FILENAME = "snippets.json"

    def __init__(self, file_store: JsonFileStore) -> None:
        self._fs = file_store
        self._fs.ensure_default([])

    def list(self) -> list[Snippet]:
        return [Snippet(**item) for item in self._fs.read([])]

    def raw(self) -> list[dict]:
        return list(self._fs.read([]))

    def add(
        self,
        *,
        trigger: str,
        body: str,
        scope: str = "global",
        case_sensitive: bool = False,
        source: str = "user",
        description: str = "",
    ) -> None:
        trigger = (trigger or "").strip()
        body = body or ""
        if not trigger or not body:
            return
        # For case-insensitive triggers the key requires non-empty fold —
        # all-punctuation/empty-fold triggers can never match anything in
        # ASR output, so don't store them.
        if not case_sensitive and not fold_key(trigger):
            return
        with self._fs.lock:
            data = self._fs.read([])
            key = _key_for(trigger, scope, case_sensitive=case_sensitive)
            for idx, item in enumerate(data):
                existing_cs = bool(item.get("case_sensitive", False))
                existing_key = _key_for(
                    str(item.get("trigger", "")),
                    str(item.get("scope", "global")),
                    case_sensitive=existing_cs,
                )
                if existing_key == key:
                    data[idx] = Snippet(
                        trigger=trigger,
                        body=body,
                        scope=scope,
                        case_sensitive=case_sensitive,
                        source=source,
                        description=description or str(item.get("description", "")),
                    ).to_dict()
                    self._fs.write(data)
                    return
            data.append(
                Snippet(
                    trigger=trigger,
                    body=body,
                    scope=scope,
                    case_sensitive=case_sensitive,
                    source=source,
                    description=description,
                ).to_dict()
            )
            self._fs.write(data)

    def remove(self, trigger: str, *, scope: str = "global") -> bool:
        # ``remove`` always treats the trigger as case-insensitive — the
        # UI's Remove button shouldn't have to know whether the row was
        # stored case-sensitive. Exact string fallback covers
        # case-sensitive rows whose case happens to match.
        needle_key = fold_key(trigger)
        with self._fs.lock:
            data = self._fs.read([])
            before = len(data)
            kept: list[dict] = []
            for item in data:
                item_trigger = str(item.get("trigger", ""))
                item_scope = str(item.get("scope", "global"))
                item_cs = bool(item.get("case_sensitive", False))
                same_scope = item_scope == scope
                if item_cs:
                    matches = same_scope and item_trigger == trigger
                else:
                    matches = same_scope and fold_key(item_trigger) == needle_key
                if matches:
                    continue
                kept.append(item)
            if len(kept) == before:
                return False
            self._fs.write(kept)
            return True

    def resolve(self, trigger: str, *, scope: str = "global") -> Snippet | None:
        """Return the best snippet for *trigger*, preferring scoped over global.

        Lookup folds the needle and stored triggers through
        :func:`fold_key` so ``"signoff"`` finds a stored ``"sign off"``
        snippet (and vice versa), and case/punctuation variants all hit
        the same row. ``case_sensitive=True`` entries bypass folding and
        require exact string equality. Scope-specific wins over global
        when both match.
        """
        needle_key = fold_key(trigger)
        raw = self._fs.read([])
        candidate: Snippet | None = None
        for item in raw:
            s = Snippet(**item)
            if s.case_sensitive:
                if s.trigger != trigger:
                    continue
            else:
                if not needle_key:
                    continue
                if fold_key(s.trigger) != needle_key:
                    continue
            if s.scope == scope:
                return s
            if s.scope == "global":
                candidate = s
        return candidate
