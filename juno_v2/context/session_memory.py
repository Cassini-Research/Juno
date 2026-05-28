from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.memory.store import _looks_like_hallucination

_TITLE_OR_ACRONYM_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_./-]{1,}|[A-Z]{2,})\b")
_TECH_RE = re.compile(
    r"(?<!\w)(?:[a-z][a-z0-9]*(?:(?:_[a-z0-9]+)+|(?:-[a-z][a-z0-9]*)+)|[A-Za-z0-9_.-]+\.(?:py|swift|ts|tsx|js|jsx|md|json|yaml|yml|toml|sql|html|css|go|rs|java|kt|sh|zsh))(?!\w)",
    re.ASCII,
)
_STOP = {
    "App",
    "The",
    "This",
    "That",
    "And",
    "For",
    "With",
    "From",
    "Your",
    "You",
    "Juno",
}


@dataclass(slots=True)
class SessionContextMemory:
    """Rolling high-signal screen/session context for writer and ASR hints.

    This is intentionally not a transcript or screen log. It keeps compact
    terms that are likely to help spelling, entity continuity, and local
    references as the user moves across apps: proper nouns, acronyms,
    filenames, identifiers, and focused document symbols.
    """

    max_terms: int = 64
    _items: OrderedDict[str, dict[str, Any]] = field(default_factory=OrderedDict)
    _tick: int = 0

    def observe(self, bundle: TypedContextBundle) -> list[str]:
        self._tick += 1
        added: list[str] = []
        sources: list[tuple[str, str]] = []
        for value in bundle.candidate_entities or []:
            sources.append((str(value), "candidate_entities"))
        for value in (bundle.app_name, bundle.window_title):
            if value:
                sources.extend((term, "surface") for term in _extract_terms(str(value)))
        for value in (
            bundle.focused_file_path,
            bundle.symbol_under_cursor,
        ):
            if value:
                sources.extend((term, "editor") for term in _file_symbol_terms(str(value)))
        for value in (
            bundle.focused_text_before[-1200:],
            bundle.focused_text_after[:800],
            bundle.selected_text[:1200],
        ):
            if value:
                sources.extend((term, "field") for term in _extract_terms(str(value)))

        for term, source in sources:
            clean = _clean_term(term)
            if not clean:
                continue
            key = clean.casefold()
            item = self._items.pop(key, None)
            if item is None:
                item = {"value": clean, "count": 0, "sources": set()}
                added.append(clean)
            item["count"] = int(item.get("count", 0)) + 1
            item["last_seen"] = self._tick
            item["sources"].add(source)
            self._items[key] = item
        while len(self._items) > self.max_terms:
            self._items.popitem(last=False)
        return added

    def snapshot(self, *, limit: int = 24) -> list[str]:
        rows = list(self._items.values())
        rows.sort(key=lambda item: (-int(item.get("last_seen", 0)), -int(item.get("count", 0)), str(item.get("value", "")).casefold()))
        return [str(item["value"]) for item in rows[: max(0, limit)]]


def _extract_terms(text: str) -> list[str]:
    out: list[str] = []
    for regex in (_TITLE_OR_ACRONYM_RE, _TECH_RE):
        for match in regex.finditer(text or ""):
            out.append(match.group(0))
    return out


def _file_symbol_terms(value: str) -> list[str]:
    out = [value]
    base = os.path.basename(value)
    if base and base != value:
        out.append(base)
        stem, _ext = os.path.splitext(base)
        if stem:
            out.append(stem)
    return out


def _clean_term(value: str) -> str:
    v = (value or "").strip(" \t\r\n,.;:!?()[]{}<>\"'`")
    if len(v) < 2 or len(v) > 96:
        return ""
    if v in _STOP:
        return ""
    if _looks_like_hallucination(v):
        return ""
    if not any(ch.isalpha() for ch in v):
        return ""
    return v


__all__ = ["SessionContextMemory"]
