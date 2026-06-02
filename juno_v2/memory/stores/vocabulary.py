from __future__ import annotations

from juno_v2.contracts.memory import LexiconEntry
from juno_v2.memory.fold import fold_key
from juno_v2.memory.stores._base import JsonFileStore
from juno_v2.memory.term_policy import learned_term_allowed


class VocabularyStore:
    """Lexicon / named-entity vocabulary the user explicitly teaches us.

    Stored as ``lexicon.json``. Dedup uses :func:`fold_key` on the
    canonical form and aliases, so ``"Polkafoundation"``,
    ``"polka foundation"``, and ``"Polka-Foundation"`` all resolve to the
    same lexicon row instead of fragmenting into competing entries that
    fight in the bias plan.
    """

    FILENAME = "lexicon.json"

    def __init__(self, file_store: JsonFileStore) -> None:
        self._fs = file_store
        self._fs.ensure_default([])

    def list(self) -> list[LexiconEntry]:
        return [LexiconEntry(**item) for item in self._fs.read([])]

    def raw(self) -> list[dict]:
        return list(self._fs.read([]))

    def upsert(
        self,
        *,
        term: str,
        canonical_form: str | None = None,
        aliases: list[str] | None = None,
        pronunciation_hint: str | None = None,
        boost: float = 1.0,
        source: str = "user",
    ) -> None:
        canonical = (canonical_form or term or "").strip()
        term = (term or "").strip()
        if not learned_term_allowed(term) or not learned_term_allowed(canonical):
            return
        with self._fs.lock:
            data = self._fs.read([])
            key = fold_key(canonical)
            # Aliases dedupe by fold_key against the canonical and each
            # other, so we don't store ``["polka foundation"]`` next to a
            # canonical ``"Polkafoundation"``.
            seen_alias_keys: set[str] = {key} if key else set()
            cleaned_aliases: list[str] = []
            for a in aliases or []:
                if not a:
                    continue
                if not learned_term_allowed(a):
                    continue
                ak = fold_key(a)
                if not ak or ak in seen_alias_keys:
                    continue
                seen_alias_keys.add(ak)
                cleaned_aliases.append(a)
            aliases = cleaned_aliases
            for idx, item in enumerate(data):
                existing_canon = fold_key(
                    str(item.get("canonical_form", item.get("term", "")))
                )
                # Merge ONLY when the canonical fold-keys match. We
                # intentionally do not collapse across canonicals via
                # alias overlap: "CodexTerm with alias 'juno'" must not
                # silently absorb into the protected "Juno" row, and a
                # user adding a new term should not unify with a stale
                # row just because one alias happens to coincide. The
                # within-entry alias dedup above keeps a single row
                # tidy; cross-row dedup stays canonical-only.
                if key and existing_canon == key:
                    merged_aliases = list(
                        dict.fromkeys(list(item.get("aliases", [])) + aliases + [term])
                    )
                    data[idx] = {
                        **item,
                        "term": canonical,
                        "canonical_form": canonical,
                        "aliases": merged_aliases,
                        "pronunciation_hint": pronunciation_hint or item.get("pronunciation_hint"),
                        "boost": max(float(item.get("boost", 1.0)), float(boost)),
                        "source": source,
                    }
                    self._fs.write(data)
                    return
            data.append(
                LexiconEntry(
                    term=term,
                    canonical_form=canonical,
                    aliases=aliases,
                    pronunciation_hint=pronunciation_hint,
                    boost=boost,
                    source=source,
                ).to_dict()
            )
            self._fs.write(data)

    def remove(self, term: str) -> bool:
        """Delete the lexicon entry whose canonical form (or any alias)
        folds to the same key as ``term``. Returns ``True`` when an entry
        was removed.
        """
        needle = fold_key(term)
        if not needle:
            return False
        with self._fs.lock:
            data = self._fs.read([])
            before = len(data)
            kept: list[dict] = []
            for item in data:
                canon = fold_key(str(item.get("canonical_form", item.get("term", ""))))
                aliases = {fold_key(str(a)) for a in (item.get("aliases") or []) if a}
                if canon == needle or needle in aliases:
                    continue
                kept.append(item)
            if len(kept) == before:
                return False
            self._fs.write(kept)
            return True
