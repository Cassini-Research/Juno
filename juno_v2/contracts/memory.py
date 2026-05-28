from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class LexiconEntry:
    term: str
    canonical_form: str
    aliases: List[str] = field(default_factory=list)
    pronunciation_hint: str | None = None
    boost: float = 1.0
    source: str = 'user'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplacementRule:
    trigger: str
    replacement: str
    scope: str = 'global'
    case_sensitive: bool = False
    source: str = 'user'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CorrectionPair:
    observed: str
    corrected: str
    count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionEntity:
    value: str
    count: int = 1
    source: str = 'session'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemorySnapshot:
    schema_version: int
    lexicon: List[LexiconEntry] = field(default_factory=list)
    replacements: List[ReplacementRule] = field(default_factory=list)
    corrections: List[CorrectionPair] = field(default_factory=list)
    session_entities: List[SessionEntity] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'lexicon': [item.to_dict() for item in self.lexicon],
            'replacements': [item.to_dict() for item in self.replacements],
            'corrections': [item.to_dict() for item in self.corrections],
            'session_entities': [item.to_dict() for item in self.session_entities],
            'metadata': dict(self.metadata),
        }


@dataclass(slots=True)
class NormalizationChange:
    kind: str
    source: str
    before: str
    after: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptNormalization:
    raw_text: str
    normalized_text: str
    applied: List[NormalizationChange] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.raw_text != self.normalized_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            'raw_text': self.raw_text,
            'normalized_text': self.normalized_text,
            'changed': self.changed,
            'applied': [item.to_dict() for item in self.applied],
            'metadata': dict(self.metadata),
        }


@dataclass(slots=True)
class MemoryServingPacket:
    lexicon_terms: List[str] = field(default_factory=list)
    replacements: List[Dict[str, Any]] = field(default_factory=list)
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    session_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'lexicon_terms': list(self.lexicon_terms),
            'replacements': list(self.replacements),
            'corrections': list(self.corrections),
            'session_entities': list(self.session_entities),
            'metadata': dict(self.metadata),
        }
