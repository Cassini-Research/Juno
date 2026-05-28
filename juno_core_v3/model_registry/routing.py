from __future__ import annotations

from dataclasses import dataclass

from juno_core_v3.model_registry.contracts import ModelPromotionStage, ModelSlot, SurfaceClass
from juno_core_v3.model_registry.registry import ModelPackage, ModelRegistry


@dataclass(slots=True, frozen=True)
class RouteRequest:
    slot: ModelSlot
    language: str | None = None
    requires_streaming: bool = False
    surface: SurfaceClass = SurfaceClass.DESKTOP
    ram_budget_mb: int | None = None


@dataclass(slots=True, frozen=True)
class RouteResult:
    request: RouteRequest
    chosen: ModelPackage | None
    reason: str


class RouteChooser:
    """Rules-first route chooser (Phase 4).

    This chooses a package based on manifest constraints and promotion stage.
    It does not load models; it only selects.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def choose(self, req: RouteRequest) -> RouteResult:
        candidates = [p for p in self.registry.list(slot=req.slot) if self._eligible(p, req)]
        if not candidates:
            return RouteResult(req, None, reason="no_eligible_packages")

        # Prefer promoted > staged > candidate; never choose retired.
        stage_rank = {
            ModelPromotionStage.PROMOTED: 0,
            ModelPromotionStage.STAGED: 1,
            ModelPromotionStage.CANDIDATE: 2,
            ModelPromotionStage.RETIRED: 99,
        }
        candidates.sort(key=lambda p: (stage_rank.get(p.promotion, 50), p.manifest.min_ram_mb, p.package_id))
        return RouteResult(req, candidates[0], reason="ok")

    def _eligible(self, pkg: ModelPackage, req: RouteRequest) -> bool:
        if pkg.promotion == ModelPromotionStage.RETIRED:
            return False
        m = pkg.manifest
        if not m.supports_language(req.language):
            return False
        if not m.allows_surface(req.surface):
            return False
        if not m.meets_streaming(req.requires_streaming):
            return False
        if req.ram_budget_mb is not None and m.min_ram_mb > req.ram_budget_mb:
            return False
        return True

