from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

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
        canon = self._effective_seed_canonicalization(
            seed_canonicalization_candidates(bundle)
        )
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

    @staticmethod
    def seed_replacement_rule_id(
        trigger: str,
        replacement: str,
        source: str,
    ) -> str:
        identity = "\0".join(
            (
                (source or "").strip().casefold(),
                (trigger or "").strip().casefold(),
                (replacement or "").strip().casefold(),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        return f"seed-replacement:{digest}"

    @staticmethod
    def _pack_name_from_source(source: str) -> str:
        prefix = "seed_bias:"
        value = (source or "").strip()
        return value[len(prefix):] if value.startswith(prefix) else value

    @staticmethod
    def _pack_display_name(pack_name: str) -> str:
        value = (pack_name or "").strip()
        if value == "personal_entities":
            return "Personal entities"
        if value.startswith("domain_"):
            value = value[len("domain_"):]
        return value.replace("_", " ").strip().title() or "Juno defaults"

    def _all_seed_canonicalization(self) -> tuple[tuple[str, str, str], ...]:
        rows: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        def add_entry(entry: Any, *, pack_name: str) -> None:
            canonical = str(getattr(entry, "canonical", "") or "").strip()
            if not canonical:
                return
            source = f"seed_bias:{pack_name}"
            values = [
                *(getattr(entry, "aliases", ()) or ()),
                *(getattr(entry, "spoken_forms", ()) or ()),
            ]
            for value in values:
                trigger = str(value or "").strip()
                if (
                    not trigger
                    or trigger.casefold() == canonical.casefold()
                    or len(trigger) < 3
                    or len(trigger) > 48
                ):
                    continue
                key = (trigger.casefold(), canonical.casefold(), source)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((trigger, canonical, source))

        for pack_name in sorted(self._seed.packs):
            for entry in self._seed.packs[pack_name]:
                add_entry(entry, pack_name=pack_name)
        for entry in self._seed.personal_entities:
            add_entry(entry, pack_name="personal_entities")
        rows.sort(key=lambda row: (row[0].casefold(), row[1].casefold(), row[2]))
        return tuple(rows)

    def _effective_seed_canonicalization(
        self,
        candidates: tuple[tuple[str, str, str], ...],
    ) -> tuple[tuple[str, str, str], ...]:
        disabled = self._learned.disabled_seed_replacement_ids()
        overrides = self._learned.seed_replacement_overrides()
        out: list[tuple[str, str, str]] = []
        for trigger, replacement, source in candidates:
            rule_id = self.seed_replacement_rule_id(trigger, replacement, source)
            if rule_id in disabled:
                continue
            override = overrides.get(rule_id)
            if override is not None:
                trigger = override["trigger"]
                replacement = override["replacement"]
            out.append((trigger, replacement, source))
        out.sort(key=lambda row: len(row[0]), reverse=True)
        return tuple(out)

    def list_default_replacements(self) -> list[dict[str, Any]]:
        disabled = self._learned.disabled_seed_replacement_ids()
        overrides = self._learned.seed_replacement_overrides()
        rows: list[dict[str, Any]] = []
        for original_trigger, original_replacement, source in self._all_seed_canonicalization():
            rule_id = self.seed_replacement_rule_id(
                original_trigger,
                original_replacement,
                source,
            )
            if rule_id in disabled:
                continue
            override = overrides.get(rule_id)
            trigger = override["trigger"] if override is not None else original_trigger
            replacement = (
                override["replacement"] if override is not None else original_replacement
            )
            pack_name = self._pack_name_from_source(source)
            rows.append(
                {
                    "trigger": trigger,
                    "replacement": replacement,
                    "scope": f"seed:{pack_name}",
                    "scope_label": self._pack_display_name(pack_name),
                    "case_sensitive": False,
                    "source": "builtin_seed_override" if override is not None else "builtin_seed",
                    "seed_rule_id": rule_id,
                    "is_builtin": True,
                    "inactive_in_verbatim": True,
                    "original_trigger": original_trigger,
                    "original_replacement": original_replacement,
                }
            )
        return rows

    def update_default_replacement(
        self,
        rule_id: str,
        *,
        trigger: str,
        replacement: str,
    ) -> bool:
        known_ids = {
            self.seed_replacement_rule_id(candidate, canonical, source)
            for candidate, canonical, source in self._all_seed_canonicalization()
        }
        if rule_id not in known_ids:
            return False
        self._learned.set_seed_replacement_override(
            rule_id,
            trigger=trigger,
            replacement=replacement,
        )
        return True

    def remove_default_replacement(self, rule_id: str) -> bool:
        known_ids = {
            self.seed_replacement_rule_id(candidate, canonical, source)
            for candidate, canonical, source in self._all_seed_canonicalization()
        }
        if rule_id not in known_ids:
            return False
        return self._learned.disable_seed_replacement(rule_id)

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
