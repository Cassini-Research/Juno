from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from juno_v2.contracts.memory import MemoryServingPacket


@dataclass(slots=True)
class PersonalizationProfile:
    """Bounded retrieval view for writer / transform."""

    vocabulary_candidates: list[str] = field(default_factory=list)
    replacements: list[dict[str, Any]] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    style_cards: list[str] = field(default_factory=list)
    active_mode_preferences: dict[str, Any] = field(default_factory=dict)
    recent_accepted_corrections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'vocabulary_candidates': list(self.vocabulary_candidates),
            'replacements': list(self.replacements),
            'snippets': list(self.snippets),
            'entities': list(self.entities),
            'style_cards': list(self.style_cards),
            'active_mode_preferences': dict(self.active_mode_preferences),
            'recent_accepted_corrections': list(self.recent_accepted_corrections),
            'metadata': dict(self.metadata),
        }


def build_personalization_profile(
    packet: MemoryServingPacket,
    *,
    effective_mode: str | None,
    top_n: int = 8,
) -> PersonalizationProfile:
    return PersonalizationProfile(
        vocabulary_candidates=list(packet.lexicon_terms)[:top_n],
        replacements=list(packet.replacements)[:top_n],
        snippets=[],
        entities=list(packet.session_entities)[:top_n],
        style_cards=[],
        active_mode_preferences={'effective_mode': effective_mode},
        recent_accepted_corrections=list(packet.corrections)[:top_n],
        metadata={'bounded': True},
    )
