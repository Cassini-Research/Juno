from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.memory.term_policy import learned_term_allowed
from juno_v2.personalization.seed.asr_bias_adapter import AsrBiasBackendCapabilities, adapt_structured_bundle_for_asr
from juno_v2.personalization.seed.bundle_builder import (
    build_structured_bias_bundle,
    seed_canonicalization_candidates,
)
from juno_v2.personalization.seed.learned_state import JunoPersonalizationLearnedStore
from juno_v2.personalization.seed.load_bundle import load_seed_bundle
from juno_v2.personalization.seed.models import JunoPersonalizationSeedLayer, SeedBiasAttachment
from juno_v2.personalization.seed.paths import default_seed_bundle_dir
from juno_v2.personalization.seed.promotion import PromotionCoordinator
from juno_v2.personalization.seed.registry import PackRegistry
from juno_v2.personalization.seed.routing import select_active_packs
from juno_v2.personalization.seed.suppressed import is_durable_memory_suppressed

if TYPE_CHECKING:
    from juno_v2.memory.store import JsonMemoryStore

_LOG = logging.getLogger(__name__)


class JunoSeedPersonalizationRuntime:
    """Loads read-only seed data once and serves per-utterance bias + promotion hooks."""

    def __init__(
        self,
        seed: JunoPersonalizationSeedLayer,
        *,
        memory_store: JsonMemoryStore | None,
    ) -> None:
        self._seed = seed
        self._registry = PackRegistry(seed)
        mem_dir = memory_store.memory_dir if memory_store is not None else Path(".juno_v2_memory")
        self._learned = JunoPersonalizationLearnedStore(mem_dir)
        self._promotion = PromotionCoordinator(
            seed=seed,
            memory_store=memory_store,
            learned_store=self._learned,
        )

    @property
    def seed_layer(self) -> JunoPersonalizationSeedLayer:
        return self._seed

    @property
    def promotion(self) -> PromotionCoordinator:
        return self._promotion

    @property
    def learned_store(self) -> JunoPersonalizationLearnedStore:
        return self._learned

    @classmethod
    def try_load(
        cls,
        *,
        bundle_dir: Path | str | None = None,
        memory_store: JsonMemoryStore | None,
    ) -> JunoSeedPersonalizationRuntime | None:
        root = Path(bundle_dir) if bundle_dir is not None else default_seed_bundle_dir()
        if not root.is_dir():
            _LOG.warning("juno_seed_personalization: bundle dir missing %s — seed layer disabled", root)
            return None
        try:
            seed = load_seed_bundle(root)
        except Exception:
            _LOG.exception("juno_seed_personalization: failed to load seed bundle from %s", root)
            return None
        _LOG.info(
            "juno_seed_personalization: loaded bundle v%s from %s packs=%s",
            seed.manifest.version,
            root,
            ",".join(sorted(seed.packs.keys())),
        )
        return cls(seed, memory_store=memory_store)

    def context_plane_suppression_value(self, plan_metadata: dict[str, Any] | None) -> str | None:
        if not plan_metadata:
            return None
        cp = plan_metadata.get("context_plane")
        if not isinstance(cp, dict):
            return None
        v = cp.get("suppression")
        return str(v) if v is not None else None

    def durable_memory_suppressed(
        self,
        context: TypedContextBundle,
        *,
        context_plane_suppression: str | None,
    ) -> bool:
        return is_durable_memory_suppressed(
            self._seed,
            context,
            context_plane_suppression=context_plane_suppression,
        )

    def build_seed_attachment(
        self,
        *,
        snapshot: MemorySnapshot,
        context: TypedContextBundle,
        context_plane_suppression: str | None,
    ) -> SeedBiasAttachment:
        suppressed = self.durable_memory_suppressed(
            context,
            context_plane_suppression=context_plane_suppression,
        )
        selection = select_active_packs(self._seed, context)
        bundle = build_structured_bias_bundle(
            seed=self._seed,
            registry=self._registry,
            selection=selection,
            memory=snapshot,
            context=context,
            promotion=self._seed.promotion,
            context_suppressed=suppressed,
        )
        canon = seed_canonicalization_candidates(bundle)
        meta = {
            "active_pack_ids": list(selection.pack_ids),
            "matched_route_indices": list(selection.matched_route_indices),
            "used_fallback": selection.used_fallback,
            "specialized_pack_allowed": selection.specialized_pack_allowed,
            "durable_memory_suppressed": suppressed,
            "structured_bundle": bundle.to_dict(),
        }
        return SeedBiasAttachment(
            extra_bias_phrases=bundle.flattened_bias_phrases,
            structured_bundle=bundle,
            canonicalization_tuples=canon,
            metadata=meta,
        )

    def observe_transcript_for_context_entities(
        self,
        raw_text: str,
        context: TypedContextBundle,
        *,
        durable_memory_suppressed: bool,
    ) -> None:
        """Increment observation counts when context candidates appear in raw ASR."""
        if not raw_text.strip():
            return
        hay = raw_text.casefold()
        for cand in context.candidate_entities[:48]:
            c = (cand or "").strip()
            if not learned_term_allowed(c):
                continue
            if c.casefold() not in hay:
                continue
            self._learned.increment_observation(c, from_suppressed_context=durable_memory_suppressed)

    def build_asr_delivery(
        self,
        *,
        base_initial_prompt: str | None,
        base_bias_phrases: list[str],
        structured_bundle,
        capabilities: AsrBiasBackendCapabilities | None = None,
    ):
        return adapt_structured_bundle_for_asr(
            base_initial_prompt=base_initial_prompt,
            base_bias_phrases=base_bias_phrases,
            structured=structured_bundle,
            capabilities=capabilities,
        )
