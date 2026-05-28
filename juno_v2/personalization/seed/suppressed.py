from __future__ import annotations

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.personalization.seed.models import JunoPersonalizationSeedLayer


def context_plane_suppressed(*, suppression_metadata: str | None) -> bool:
    """True when ContextPlane marked secure/sensitive suppression."""
    return suppression_metadata not in (None, "", "none")


def seed_bundle_suppressed_surface(seed: JunoPersonalizationSeedLayer, context: TypedContextBundle) -> bool:
    """Keyword match against ``suppressed_surface_policy.json`` (broader than ContextPlane)."""
    hay = f"{context.app_name or ''} {context.window_title or ''}".casefold()
    for token in seed.suppressed_surface.never_persist_from_surfaces:
        if token in hay:
            return True
    return False


def is_durable_memory_suppressed(
    seed: JunoPersonalizationSeedLayer | None,
    context: TypedContextBundle,
    *,
    context_plane_suppression: str | None,
) -> bool:
    """True when no correction / entity / observation must be written to disk."""
    if context_plane_suppressed(suppression_metadata=context_plane_suppression):
        return True
    if seed is not None and seed_bundle_suppressed_surface(seed, context):
        return True
    return False
