"""Top-level personalization facade.

This module used to own the whole memory implementation in one class. The
implementation has since been decomposed into six domain-focused stores
under :mod:`juno_v2.memory.stores` — vocabulary, replacements,
corrections, session entities, snippets, and style cards — and
``JsonMemoryStore`` is now a thin composition on top of them.

We keep the original class name and public API unchanged so every existing
caller (writer, commit controller, eval harness, v2 tests) continues to
work without modification. New code should prefer the typed sub-stores
exposed as attributes: ``memory.snippets``, ``memory.replacements``, etc.
"""

from __future__ import annotations

import threading
from pathlib import Path

from juno_v2.contracts.memory import (
    MemoryServingPacket,
    MemorySnapshot,
)
from juno_v2.memory.hallucination import (
    HALLUCINATION_CONFIDENCE_FLOOR,
    looks_like_hallucination,
)
from juno_v2.memory.entity_policy import session_entity_allowed
from juno_v2.memory.ranking import _low_signal_lexicon_pair
from juno_v2.memory.stores import (
    CorrectionStore,
    EntityStore,
    JsonFileStore,
    ReplacementStore,
    Snippet,
    SnippetStore,
    VocabularyStore,
)
from juno_v2.memory.stores.corrections import is_safe_correction_pair

SCHEMA_VERSION = 1

# Kept as module-level names because existing callers import them directly
# (and tests reference both the constant and the helper).
_HALLUCINATION_CONFIDENCE_FLOOR = HALLUCINATION_CONFIDENCE_FLOOR
_looks_like_hallucination = looks_like_hallucination
_MAX_CORRECTION_TEXT_CHARS = 120  # re-exported so tests that import the name still resolve

# Issue #15: protected vocabulary the engine seeds at boot so headless
# flows (CLI dictation, smoke tests, fresh-boot eval) get the wake-word
# into the bias plan without depending on the macOS Memory page being
# opened. Each entry is (term, canonical_form). Seeding is one-shot per
# memory directory: if the user later removes the row we do not
# resurrect it on subsequent boots.
PROTECTED_VOCABULARY: tuple[tuple[str, str], ...] = (
    ("Juno", "Juno"),
)


class JsonMemoryStore:
    """Composed personalization store.

    Attributes:
        vocabulary: :class:`VocabularyStore`
        replacements: :class:`ReplacementStore`
        corrections: :class:`CorrectionStore`
        entities: :class:`EntityStore`
        snippets: :class:`SnippetStore`

    The original API (``add_lexicon_entry``, ``add_replacement``,
    ``record_correction``, ``upsert_session_entities``, ``snapshot``,
    ``serving_packet``) delegates to these sub-stores.
    """

    def __init__(self, memory_dir: Path | str) -> None:
        self.memory_dir = Path(memory_dir)
        self.lock = threading.RLock()
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        def fs(name: str) -> JsonFileStore:
            return JsonFileStore(self.memory_dir, name, lock=self.lock)

        self.vocabulary = VocabularyStore(fs(VocabularyStore.FILENAME))
        self.replacements = ReplacementStore(fs(ReplacementStore.FILENAME))
        self.corrections = CorrectionStore(fs(CorrectionStore.FILENAME))
        self.entities = EntityStore(fs(EntityStore.FILENAME))
        self.snippets = SnippetStore(fs(SnippetStore.FILENAME))

        self._manifest = fs("manifest.json")
        if not self._manifest.path.exists():
            self._manifest.write({"schema_version": SCHEMA_VERSION})

        self._seed_protected_vocabulary()

    # ---------------------------------------------------------------- #
    # Boot-time protected-vocabulary seeding (Issue #15)
    # ---------------------------------------------------------------- #

    def _seed_protected_vocabulary(self) -> None:
        """Insert each :data:`PROTECTED_VOCABULARY` term into the lexicon
        on first boot of this memory directory.

        Once a term is recorded as seeded in the manifest, a subsequent
        boot does NOT resurrect it — this keeps a user's explicit
        deletion sticky. The check is per-term so adding new protected
        entries in a future release seeds them on next boot without
        clobbering user deletions of older ones.

        For pre-existing memory dirs (manifest already present without
        the ``seeded_protected_vocabulary`` key) we treat the upgrade as
        first-boot-for-this-term: seed only if no row with the canonical
        form exists yet, then mark it seeded.
        """
        from juno_v2.memory.fold import fold_key

        with self.lock:
            manifest = self._manifest.read({"schema_version": SCHEMA_VERSION})
            seeded = set(manifest.get("seeded_protected_vocabulary", []))
            # Seeded keys land in the manifest under fold_key so they
            # survive case/punctuation/whitespace variation in the
            # canonical form (no functional change for the existing
            # "juno" entry; future protected entries get the same
            # robustness for free).
            existing = {
                fold_key(str(item.get("canonical_form", item.get("term", ""))))
                for item in self.vocabulary.raw()
            }
            existing.discard("")
            changed = False
            for term, canonical in PROTECTED_VOCABULARY:
                key = fold_key(canonical)
                if not key or key in seeded:
                    continue
                # Mark seeded regardless so user deletion later is sticky.
                seeded.add(key)
                changed = True
                if key in existing:
                    # Row already present (e.g. UI-side seed ran first,
                    # or pre-upgrade user already had one). Don't double-add.
                    continue
                self.vocabulary.upsert(
                    term=term,
                    canonical_form=canonical,
                    source="builtin",
                )
            if changed:
                manifest["seeded_protected_vocabulary"] = sorted(seeded)
                self._manifest.write(manifest)

    # ---------------------------------------------------------------- #
    # Snapshot / serving packet
    # ---------------------------------------------------------------- #

    def snapshot(self) -> MemorySnapshot:
        with self.lock:
            return MemorySnapshot(
                schema_version=SCHEMA_VERSION,
                lexicon=self.vocabulary.list(),
                replacements=self.replacements.list(),
                corrections=self.corrections.list(),
                session_entities=self.entities.list(),
                snippets=[item.to_dict() for item in self.snippets.list()],
                metadata=self._manifest.read({"schema_version": SCHEMA_VERSION}),
            )

    def serving_packet(
        self,
        *,
        max_lexicon: int = 10,
        max_replacements: int = 6,
        max_corrections: int = 6,
        max_entities: int = 8,
    ) -> MemoryServingPacket:
        snapshot = self.snapshot()
        lexicon = sorted(
            (item for item in snapshot.lexicon if not _low_signal_lexicon_pair(item.term, item.canonical_form)),
            key=lambda item: (-float(item.boost), item.canonical_form.casefold()),
        )
        replacements = sorted(
            snapshot.replacements,
            key=lambda item: (-len(item.trigger), item.trigger.casefold()),
        )
        corrections = sorted(
            (
                item
                for item in snapshot.corrections
                if is_safe_correction_pair(item.observed, item.corrected)
            ),
            key=lambda item: (
                -int(item.count),
                -len(item.corrected),
                item.corrected.casefold(),
            ),
        )
        entities = sorted(
            (item for item in snapshot.session_entities if session_entity_allowed(item.value)),
            key=lambda item: (-int(item.count), item.value.casefold()),
        )

        return MemoryServingPacket(
            lexicon_terms=[item.canonical_form for item in lexicon[:max_lexicon]],
            replacements=[
                {
                    "trigger": item.trigger,
                    "replacement": item.replacement,
                    "scope": item.scope,
                }
                for item in replacements[:max_replacements]
            ],
            corrections=[
                {
                    "observed": item.observed,
                    "corrected": item.corrected,
                    "count": item.count,
                }
                for item in corrections[:max_corrections]
            ],
            session_entities=[item.value for item in entities[:max_entities]],
            snippets=[
                {
                    "trigger": str(item.get("trigger") or ""),
                    "scope": str(item.get("scope") or "global"),
                    "body_preview": str(item.get("body") or "")[:500],
                    "body_chars": len(str(item.get("body") or "")),
                    "case_sensitive": bool(item.get("case_sensitive", False)),
                }
                for item in list(snapshot.snippets or [])[:8]
                if str(item.get("trigger") or "").strip() and str(item.get("body") or "")
            ],
            metadata={
                "lexicon_total": len(snapshot.lexicon),
                "replacement_total": len(snapshot.replacements),
                "correction_total": len(snapshot.corrections),
                "correction_served_total": len(corrections),
                "session_entity_total": len(snapshot.session_entities),
                "snippet_total": len(snapshot.snippets),
                "lexicon_truncated": len(snapshot.lexicon) > max_lexicon,
                "replacement_truncated": len(snapshot.replacements) > max_replacements,
                "correction_truncated": len(corrections) > max_corrections,
                "session_entity_truncated": len(snapshot.session_entities) > max_entities,
            },
        )

    # ---------------------------------------------------------------- #
    # Legacy pass-through API (preserved for existing v2 callers)
    # ---------------------------------------------------------------- #

    def add_lexicon_entry(
        self,
        *,
        term: str,
        canonical_form: str | None = None,
        aliases: list[str] | None = None,
        pronunciation_hint: str | None = None,
        boost: float = 1.0,
        source: str = "user",
    ) -> None:
        self.vocabulary.upsert(
            term=term,
            canonical_form=canonical_form,
            aliases=aliases,
            pronunciation_hint=pronunciation_hint,
            boost=boost,
            source=source,
        )

    def add_replacement(
        self,
        *,
        trigger: str,
        replacement: str,
        scope: str = "global",
        case_sensitive: bool = False,
        source: str = "user",
    ) -> None:
        self.replacements.add(
            trigger=trigger,
            replacement=replacement,
            scope=scope,
            case_sensitive=case_sensitive,
            source=source,
        )

    def record_correction(self, observed: str, corrected: str) -> bool:
        return self.corrections.record(observed, corrected)

    def upsert_session_entities(
        self, entities: list[str], *, source: str = "session"
    ) -> None:
        self.entities.upsert_many(entities, source=source)

    # ---------------------------------------------------------------- #
    # New snippet / style APIs (Phase D additions)
    # ---------------------------------------------------------------- #

    def add_snippet(
        self,
        *,
        trigger: str,
        body: str,
        scope: str = "global",
        case_sensitive: bool = False,
        source: str = "user",
        description: str = "",
    ) -> None:
        self.snippets.add(
            trigger=trigger,
            body=body,
            scope=scope,
            case_sensitive=case_sensitive,
            source=source,
            description=description,
        )

    def resolve_snippet(self, trigger: str, *, scope: str = "global") -> Snippet | None:
        return self.snippets.resolve(trigger, scope=scope)

    def remove_snippet(self, trigger: str, *, scope: str = "global") -> bool:
        return self.snippets.remove(trigger, scope=scope)

    def remove_lexicon_entry(self, term: str) -> bool:
        return self.vocabulary.remove(term)

    def remove_replacement(
        self,
        trigger: str,
        *,
        scope: str = "global",
        case_sensitive: bool = False,
    ) -> bool:
        return self.replacements.remove(
            trigger, scope=scope, case_sensitive=case_sensitive
        )

    def remove_correction(self, observed: str, corrected: str | None = None) -> bool:
        return self.corrections.remove(observed, corrected)


__all__ = [
    "JsonMemoryStore",
    "SCHEMA_VERSION",
    "_MAX_CORRECTION_TEXT_CHARS",
    "_HALLUCINATION_CONFIDENCE_FLOOR",
    "_looks_like_hallucination",
]
