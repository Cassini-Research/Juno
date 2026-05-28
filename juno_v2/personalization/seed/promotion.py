from __future__ import annotations

import logging
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.memory.term_policy import learned_term_allowed
from juno_v2.personalization.seed.learned_state import JunoPersonalizationLearnedStore
from juno_v2.personalization.seed.models import JunoPersonalizationSeedLayer, PromotionPolicy

_LOG = logging.getLogger(__name__)


class PromotionCoordinator:
    """Explicit promotion rules from ``promotion_policy.json`` (no hidden heuristics)."""

    def __init__(
        self,
        *,
        seed: JunoPersonalizationSeedLayer | None,
        memory_store: JsonMemoryStore | None,
        learned_store: JunoPersonalizationLearnedStore | None,
    ) -> None:
        self._seed = seed
        self._memory = memory_store
        self._learned = learned_store

    @property
    def enabled(self) -> bool:
        return self._seed is not None and self._memory is not None

    def policy(self) -> PromotionPolicy | None:
        if self._seed is None:
            return None
        return self._seed.promotion

    def maybe_promote_correction_to_lexicon(
        self,
        *,
        observed: str,
        corrected: str,
        durable_memory_suppressed: bool,
    ) -> dict[str, Any]:
        """Rule: correction-learned items promote after ``correction_promote_after`` threshold."""
        if not self.enabled or self._memory is None or self._seed is None:
            return {"promoted": False, "reason": "disabled"}
        pol = self._seed.promotion
        if durable_memory_suppressed:
            if pol.never_auto_promote_from_suppressed_context:
                return {"promoted": False, "reason": "suppressed_surface"}
        after = max(1, int(pol.correction_promote_after))
        for pair in self._memory.corrections.list():
            if (
                pair.observed.strip().casefold() == (observed or "").strip().casefold()
                and pair.corrected.strip().casefold() == (corrected or "").strip().casefold()
            ):
                if int(pair.count) < after:
                    return {"promoted": False, "reason": "below_threshold", "count": int(pair.count)}
                if not learned_term_allowed(pair.corrected):
                    return {"promoted": False, "reason": "term_too_short"}
                self._memory.add_lexicon_entry(
                    term=pair.corrected.strip(),
                    canonical_form=pair.corrected.strip(),
                    aliases=[pair.observed.strip()] if pair.observed.strip() else None,
                    boost=1.0,
                    source="correction_promoted",
                )
                _LOG.info(
                    "juno_promotion: correction_promoted_to_lexicon observed=%r corrected=%r count=%s",
                    pair.observed,
                    pair.corrected,
                    pair.count,
                )
                return {"promoted": True, "reason": "correction_threshold_met", "count": int(pair.count)}
        return {"promoted": False, "reason": "pair_not_found"}

    def maybe_promote_context_entity_to_lexicon(
        self,
        *,
        token: str,
        durable_memory_suppressed: bool,
    ) -> dict[str, Any]:
        """Rule: context-observed items need observation + acceptance thresholds."""
        if not self.enabled or self._memory is None or self._seed is None or self._learned is None:
            return {"promoted": False, "reason": "disabled"}
        pol = self._seed.promotion
        if durable_memory_suppressed:
            if pol.never_auto_promote_from_suppressed_context:
                return {"promoted": False, "reason": "suppressed_surface"}
        if pol.screen_derived_candidates_runtime_only_until_accepted:
            pass  # explicit gate: we require acceptance_count >= min before promote
        if not learned_term_allowed(token):
            return {"promoted": False, "reason": "term_too_short"}
        snap = self._learned.observation_snapshot(token)
        if snap is None:
            return {"promoted": False, "reason": "no_observation_state"}
        need_obs = max(1, int(pol.context_observation_min_count))
        need_acc = max(1, int(pol.context_acceptance_min_count))
        if snap.observation_count < need_obs:
            return {
                "promoted": False,
                "reason": "observation_below_threshold",
                "observation_count": snap.observation_count,
            }
        if snap.acceptance_count < need_acc:
            return {
                "promoted": False,
                "reason": "acceptance_below_threshold",
                "acceptance_count": snap.acceptance_count,
            }
        self._memory.add_lexicon_entry(
            term=token.strip(),
            canonical_form=token.strip(),
            boost=1.0,
            source="context_promoted",
        )
        _LOG.info("juno_promotion: context_entity_promoted_to_lexicon token=%r", token)
        return {"promoted": True, "reason": "context_thresholds_met"}

    def run_initial_promotion(self, memory: JsonMemoryStore) -> dict[str, Any]:
        """Seed user memory with a small set of shipped personalization terms.

        This is intended to run once on first broker start so the Memory UI and
        bias engine have some "alive" vocabulary immediately.
        """
        if self._seed is None:
            return {"ok": False, "reason": "seed_disabled", "promoted": 0}
        if memory is None:
            return {"ok": False, "reason": "memory_missing", "promoted": 0}

        promoted = 0
        seen: set[str] = set()

        def _promote_entry(entry) -> None:
            nonlocal promoted
            canonical = (getattr(entry, "canonical", "") or "").strip()
            if not canonical or not learned_term_allowed(canonical):
                return
            key = canonical.casefold()
            if key in seen:
                return
            seen.add(key)
            aliases = []
            aliases.extend([a for a in (getattr(entry, "aliases", ()) or ()) if isinstance(a, str) and a.strip()])
            aliases.extend([a for a in (getattr(entry, "spoken_forms", ()) or ()) if isinstance(a, str) and a.strip()])
            # Keep alias list bounded and deduped.
            alias_clean: list[str] = []
            alias_seen: set[str] = set()
            for a in aliases:
                v = a.strip()
                if not v or not learned_term_allowed(v):
                    continue
                vk = v.casefold()
                if vk in alias_seen:
                    continue
                alias_seen.add(vk)
                alias_clean.append(v)
                if len(alias_clean) >= 8:
                    break
            memory.add_lexicon_entry(
                term=canonical,
                canonical_form=canonical,
                aliases=alias_clean or None,
                boost=1.0,
                source="seed_promotion",
            )
            promoted += 1

        # 1) Always promote personal_entities (curated, small).
        for entry in (self._seed.personal_entities or ()):
            _promote_entry(entry)

        # 2) Promote a bounded number of terms from default enabled packs.
        # This keeps first-run memory non-empty without dumping the whole bundle.
        max_pack_terms = 200
        for pack_id in (self._seed.manifest.default_enabled_packs or ()):
            terms = self._seed.packs.get(pack_id) or ()
            for entry in terms:
                _promote_entry(entry)
                if promoted >= max_pack_terms:
                    break
            if promoted >= max_pack_terms:
                break

        _LOG.info("juno_promotion: initial_promotion completed promoted=%s", promoted)
        return {"ok": True, "promoted": promoted}
