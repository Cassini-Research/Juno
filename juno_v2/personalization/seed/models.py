from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SeedTermEntry:
    """One row from a pack JSON or personal_entities."""

    pack_name: str
    canonical: str
    kind: str
    aliases: tuple[str, ...]
    spoken_forms: tuple[str, ...]
    tags: tuple[str, ...]
    locale: str
    scope_defaults: tuple[str, ...]
    source: str
    confidence: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_name": self.pack_name,
            "canonical": self.canonical,
            "kind": self.kind,
            "aliases": list(self.aliases),
            "spoken_forms": list(self.spoken_forms),
            "tags": list(self.tags),
            "locale": self.locale,
            "scope_defaults": list(self.scope_defaults),
            "source": self.source,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass(slots=True)
class SurfaceRoute:
    surface_keywords: tuple[str, ...]
    enable_packs: tuple[str, ...]


@dataclass(slots=True)
class RoutingPolicy:
    version: int
    always_include: tuple[str, ...]
    surface_routes: tuple[SurfaceRoute, ...]
    fallback_enabled_packs: tuple[str, ...]
    notes: str = ""


@dataclass(slots=True)
class PromotionPolicy:
    version: int
    manual_seed_entries_trusted_immediately: bool
    manual_user_added_entries_trusted_immediately: bool
    correction_promote_after: int
    context_observation_min_count: int
    context_acceptance_min_count: int
    screen_derived_candidates_runtime_only_until_accepted: bool
    never_auto_promote_from_suppressed_context: bool
    max_runtime_candidates_per_surface: int
    max_bias_bundle_items_before_clipping: int
    specialized_pack_requires_explicit_route: bool
    notes: str = ""


@dataclass(slots=True)
class SuppressedSurfacePolicy:
    version: int
    never_persist_from_surfaces: tuple[str, ...]
    allow_runtime_only_bias_from_suppressed_surfaces: bool
    notes: str = ""


@dataclass(slots=True)
class JunoSeedManifest:
    version: int
    default_enabled_packs: tuple[str, ...]
    optional_packs: tuple[str, ...]
    promotion_policy: dict[str, Any]
    generated_bundle: str = ""


@dataclass(slots=True)
class JunoPersonalizationSeedLayer:
    """Read-only in-memory view of the shipped seed bundle."""

    bundle_dir: Path
    manifest: JunoSeedManifest
    routing: RoutingPolicy
    promotion: PromotionPolicy
    suppressed_surface: SuppressedSurfacePolicy
    packs: dict[str, tuple[SeedTermEntry, ...]]
    personal_entities: tuple[SeedTermEntry, ...]
    validation_report: dict[str, Any]

    def pack_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.packs.keys()))


@dataclass(slots=True)
class ActivePackSelection:
    """Resolved pack ids for one utterance (``personal_entities`` is not a pack file)."""

    pack_ids: tuple[str, ...]
    specialized_pack_allowed: bool
    matched_route_indices: tuple[int, ...]
    used_fallback: bool


@dataclass(slots=True)
class BiasBundleTerm:
    """One deduped row in the bounded runtime bias bundle."""

    canonical: str
    bias_strings: tuple[str, ...]
    tags: tuple[str, ...]
    pack_name: str
    rank_key: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "bias_strings": list(self.bias_strings),
            "tags": list(self.tags),
            "pack_name": self.pack_name,
        }


@dataclass(slots=True)
class StructuredBiasBundle:
    """Structured + flattened ASR-facing shapes."""

    terms: tuple[BiasBundleTerm, ...]
    flattened_bias_phrases: tuple[str, ...]
    clipped: bool
    clip_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": [t.to_dict() for t in self.terms],
            "flattened_bias_phrases": list(self.flattened_bias_phrases),
            "clipped": self.clipped,
            "clip_reason": self.clip_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SeedBiasAttachment:
    """Optional extension merged into :class:`~juno_v2.memory.bias.RecognitionBiasEngine` plans."""

    extra_bias_phrases: tuple[str, ...]
    structured_bundle: StructuredBiasBundle | None
    canonicalization_tuples: tuple[tuple[str, str, str], ...]
    metadata: dict[str, Any] = field(default_factory=dict)
