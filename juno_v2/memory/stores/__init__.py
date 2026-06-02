"""Domain-focused personalization stores.

Phase 3 of the personalization roadmap decomposes the original
``JsonMemoryStore`` into a handful of logical stores, each responsible for
one knowledge domain:

- :class:`VocabularyStore`   — lexicon / named entities the user teaches us
- :class:`ReplacementStore`  — trigger -> replacement rules (ASR-level)
- :class:`CorrectionStore`   — observed -> corrected pairs learned post-commit
- :class:`EntityStore`       — session-bounded named entities
- :class:`SnippetStore`      — user snippets expanded by the writer

All five wrap a shared ``memory_dir`` and share a ``threading.RLock``; that
way the top-level ``JsonMemoryStore`` facade can continue to atomically
produce a cross-store snapshot/serving-packet while each domain stays
independently testable.
"""

from juno_v2.memory.stores._base import JsonFileStore
from juno_v2.memory.stores.corrections import CorrectionStore
from juno_v2.memory.stores.entities import EntityStore
from juno_v2.memory.stores.replacements import ReplacementStore
from juno_v2.memory.stores.snippets import Snippet, SnippetStore
from juno_v2.memory.stores.vocabulary import VocabularyStore

__all__ = [
    "CorrectionStore",
    "EntityStore",
    "JsonFileStore",
    "ReplacementStore",
    "Snippet",
    "SnippetStore",
    "VocabularyStore",
]
