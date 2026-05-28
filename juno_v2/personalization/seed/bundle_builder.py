from __future__ import annotations

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.memory.bias import (
    is_biasable_runtime_context_candidate,
    is_low_signal_lexicon_pair,
)
from juno_v2.memory.stores.corrections import is_safe_correction_pair
from juno_v2.personalization.seed.models import (
    ActivePackSelection,
    BiasBundleTerm,
    JunoPersonalizationSeedLayer,
    PromotionPolicy,
    StructuredBiasBundle,
)
from juno_v2.personalization.seed.registry import PackRegistry


def _dedupe_key(canonical: str) -> str:
    return " ".join(canonical.split()).casefold()


def build_structured_bias_bundle(
    *,
    seed: JunoPersonalizationSeedLayer,
    registry: PackRegistry,
    selection: ActivePackSelection,
    memory: MemorySnapshot,
    context: TypedContextBundle,
    promotion: PromotionPolicy,
    context_suppressed: bool,
) -> StructuredBiasBundle:
    """Compose bounded bias terms: seed packs + personal entities + memory + runtime."""
    max_items = int(promotion.max_bias_bundle_items_before_clipping)
    max_runtime = int(promotion.max_runtime_candidates_per_surface)

    rows: list[BiasBundleTerm] = []
    seen: set[str] = set()

    def add_row(
        canonical: str,
        bias_strings: list[str],
        *,
        tags: tuple[str, ...],
        pack_name: str,
        rank_key: tuple,
    ) -> None:
        key = _dedupe_key(canonical)
        if key in seen:
            return
        seen.add(key)
        uniq_bias: list[str] = []
        sseen: set[str] = set()
        for s in bias_strings:
            t = s.strip()
            if not t:
                continue
            k = t.casefold()
            if k in sseen:
                continue
            sseen.add(k)
            uniq_bias.append(t)
        rows.append(
            BiasBundleTerm(
                canonical=canonical.strip(),
                bias_strings=tuple(uniq_bias),
                tags=tags,
                pack_name=pack_name,
                rank_key=rank_key,
            )
        )

    # 1) Seed packs + routed packs (registry order)
    for entry in registry.terms_for_active_packs(selection.pack_ids):
        bias: list[str] = [entry.canonical]
        bias.extend(entry.aliases)
        bias.extend(entry.spoken_forms)
        rk = (0, entry.pack_name, entry.canonical.casefold())
        add_row(entry.canonical, bias, tags=entry.tags, pack_name=entry.pack_name, rank_key=rk)

    # 2) Personal entities always (not necessarily in pack_ids)
    for entry in seed.personal_entities:
        bias = [entry.canonical, *entry.aliases, *entry.spoken_forms]
        rk = (0, "personal_entities", entry.canonical.casefold())
        add_row(entry.canonical, bias, tags=entry.tags, pack_name="personal_entities", rank_key=rk)

    # 3) Learned lexicon (user + promoted)
    for lex in memory.lexicon:
        if is_low_signal_lexicon_pair(lex.term, lex.canonical_form):
            continue
        bias = [lex.canonical_form, lex.term, *lex.aliases]
        rk = (1, "memory_lexicon", lex.canonical_form.casefold())
        add_row(lex.canonical_form, bias, tags=(), pack_name="memory_lexicon", rank_key=rk)

    # 4) Corrections (prefer corrected form as canonical bias)
    for cor in memory.corrections:
        if not is_safe_correction_pair(cor.observed, cor.corrected):
            continue
        bias = [cor.corrected, cor.observed]
        rk = (2, "memory_correction", cor.corrected.casefold(), -int(cor.count))
        add_row(cor.corrected, bias, tags=(), pack_name="memory_correction", rank_key=rk)

    # 5) Runtime-only context candidates (skipped when suppressed unless policy allows bias)
    allow_runtime = not context_suppressed or seed.suppressed_surface.allow_runtime_only_bias_from_suppressed_surfaces
    if allow_runtime:
        runtime_added = 0
        for cand in context.candidate_entities:
            if runtime_added >= max_runtime:
                break
            c = (cand or "").strip()
            if not c:
                continue
            if not is_biasable_runtime_context_candidate(c):
                continue
            key = _dedupe_key(c)
            if key in seen:
                continue
            rk = (3, "runtime_context", c.casefold())
            add_row(c, [c], tags=("runtime_only",), pack_name="runtime_context", rank_key=rk)
            runtime_added += 1

    rows.sort(key=lambda r: (r.rank_key, r.canonical.casefold()))

    clipped = len(rows) > max_items
    clip_reason = None
    if clipped:
        clip_reason = f"clipped_to_{max_items}"
        rows = rows[:max_items]

    flat: list[str] = []
    fseen: set[str] = set()
    for r in rows:
        for s in r.bias_strings:
            k = s.casefold()
            if k in fseen:
                continue
            fseen.add(k)
            flat.append(s)

    return StructuredBiasBundle(
        terms=tuple(rows),
        flattened_bias_phrases=tuple(flat),
        clipped=clipped,
        clip_reason=clip_reason,
        metadata={
            "active_pack_ids": list(selection.pack_ids),
            "used_fallback": selection.used_fallback,
            "context_suppressed": context_suppressed,
        },
    )


def seed_canonicalization_candidates(
    bundle: StructuredBiasBundle,
    *,
    min_variant_len: int = 3,
    max_variant_len: int = 48,
) -> tuple[tuple[str, str, str], ...]:
    """Build (variant, canonical, source) tuples for conservative post-ASR normalization."""
    out: list[tuple[str, str, str]] = []
    for term in bundle.terms:
        canon = term.canonical.strip()
        if not canon:
            continue
        for src in term.bias_strings:
            v = src.strip()
            if not v or v.casefold() == canon.casefold():
                continue
            if len(v) < min_variant_len or len(v) > max_variant_len:
                continue
            out.append((v, canon, f"seed_bias:{term.pack_name}"))
    out.sort(key=lambda t: len(t[0]), reverse=True)
    return tuple(out)
