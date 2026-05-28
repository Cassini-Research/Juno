from __future__ import annotations

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.personalization.seed.models import ActivePackSelection, JunoPersonalizationSeedLayer, RoutingPolicy

SPECIALIZED_PACK = "specialized_distributed_protocols"


def _haystack(context: TypedContextBundle) -> str:
    parts: list[str] = []
    if context.app_name:
        parts.append(context.app_name)
    if context.window_title:
        parts.append(context.window_title)
    bundle = context.metadata.get("app_bundle_id")
    if isinstance(bundle, str) and bundle.strip():
        parts.append(bundle)
    return " ".join(parts).casefold()


def _pack_ids_from_always(routing: RoutingPolicy) -> set[str]:
    out: set[str] = set()
    for name in routing.always_include:
        if name == "personal_entities":
            continue
        out.add(name)
    return out


def select_active_packs(
    seed: JunoPersonalizationSeedLayer,
    context: TypedContextBundle,
) -> ActivePackSelection:
    """Apply ``routing_policy.json`` surface keyword routing."""
    routing = seed.routing
    hay = _haystack(context)
    base = _pack_ids_from_always(routing)
    matched: list[int] = []
    routed: set[str] = set()
    if hay.strip():
        for idx, route in enumerate(routing.surface_routes):
            if any(kw in hay for kw in route.surface_keywords):
                matched.append(idx)
                routed.update(route.enable_packs)

    if routed:
        pack_ids = base | routed
        used_fallback = False
    else:
        pack_ids = {p for p in routing.fallback_enabled_packs if p != "personal_entities"}
        used_fallback = True

    # Drop specialized pack unless an explicit route enabled it.
    specialized_allowed = SPECIALIZED_PACK in routed
    if SPECIALIZED_PACK in pack_ids and not specialized_allowed:
        if seed.promotion.specialized_pack_requires_explicit_route:
            pack_ids.discard(SPECIALIZED_PACK)

    # Deterministic order: sorted pack names stable for tests.
    ordered = tuple(sorted(pack_ids))
    return ActivePackSelection(
        pack_ids=ordered,
        specialized_pack_allowed=specialized_allowed,
        matched_route_indices=tuple(matched),
        used_fallback=used_fallback,
    )
