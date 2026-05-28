"""Registry-driven engine route resolution.

Bridges :class:`juno_core_v3.model_registry.routing.RouteChooser` to the
concrete backend fields used by
:class:`juno_v2.engine.factory.CanonicalEngineBuildSpec`.

Until now, ``build_canonical_engine`` took the backend names and model
paths as bare strings set by the launcher. That worked but bypassed the
capability manifest contract: the registry lists what the product *could*
run, yet the engine was started from unrelated env vars. This module
closes that gap without forcing every caller to migrate at once:

- Packages may carry concrete runtime info in ``ModelPackage.metadata``
  under the keys ``model_path``, ``hf_repo_id``, ``endpoint``.
- :func:`resolve_engine_routes` asks the chooser for one package per
  slot and normalises the answer.
- :func:`apply_routes_to_spec` pushes that answer into a build spec,
  leaving user-provided values intact unless ``override=True``.

Callers that don't want to touch the registry just skip this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from juno_core_v3.model_registry.contracts import ModelSlot, SurfaceClass
from juno_core_v3.model_registry.routing import RouteChooser, RouteRequest, RouteResult


# Sentinel defaults that indicate "caller did not configure this field".
# ``build_canonical_engine`` already accepts many of these defaults, so we
# only overwrite when a field still has its default value.
_DEFAULT_PREVIEW_BACKEND = "faster_whisper"
_DEFAULT_FINAL_BACKEND = "faster_whisper"
_DEFAULT_WRITER_BACKEND: str | None = None


@dataclass(slots=True, frozen=True)
class ResolvedSlot:
    """A concrete, actionable routing answer for one model slot."""

    slot: ModelSlot
    package_id: str
    backend_name: str
    model_path: str | None
    hf_repo_id: str | None
    endpoint: str | None
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot.value,
            "package_id": self.package_id,
            "backend_name": self.backend_name,
            "model_path": self.model_path,
            "hf_repo_id": self.hf_repo_id,
            "endpoint": self.endpoint,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class ResolvedEngineRoutes:
    """Resolved routes for the three slots the v2 factory consumes."""

    preview: ResolvedSlot | None
    final: ResolvedSlot | None
    writer: ResolvedSlot | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "preview": None if self.preview is None else self.preview.to_dict(),
            "final": None if self.final is None else self.final.to_dict(),
            "writer": None if self.writer is None else self.writer.to_dict(),
        }


def _package_runtime_bits(result: RouteResult) -> tuple[str | None, str | None, str | None]:
    """Extract ``(model_path, hf_repo_id, endpoint)`` from a chosen package.

    These come from the package metadata. The registry's contract treats
    them as hints — a package without any of them still routes, the
    factory just falls back to the caller-provided defaults.
    """
    pkg = result.chosen
    if pkg is None:
        return (None, None, None)
    md = pkg.metadata or {}
    return (
        md.get("model_path"),
        md.get("hf_repo_id") or md.get("final_hf_repo_id"),
        md.get("endpoint"),
    )


def resolve_slot(
    chooser: RouteChooser,
    slot: ModelSlot,
    *,
    surface: SurfaceClass = SurfaceClass.DESKTOP,
    language: str | None = None,
    requires_streaming: bool = False,
    ram_budget_mb: int | None = None,
) -> ResolvedSlot | None:
    req = RouteRequest(
        slot=slot,
        language=language,
        requires_streaming=requires_streaming,
        surface=surface,
        ram_budget_mb=ram_budget_mb,
    )
    result = chooser.choose(req)
    if result.chosen is None:
        return None
    path, repo, endpoint = _package_runtime_bits(result)
    pkg = result.chosen
    return ResolvedSlot(
        slot=slot,
        package_id=pkg.package_id,
        backend_name=pkg.manifest.backend.value,
        model_path=path,
        hf_repo_id=repo,
        endpoint=endpoint,
        reason=result.reason,
        metadata=dict(pkg.metadata or {}),
    )


def resolved_slot_is_actionable(slot: ResolvedSlot | None) -> tuple[bool, str | None]:
    """Return whether a resolved route carries enough runtime bits to build truthfully."""
    if slot is None:
        return (False, "missing_route")

    backend = (slot.backend_name or "").strip().lower()
    has_model_path = bool((slot.model_path or "").strip())
    has_endpoint = bool((slot.endpoint or "").strip())
    has_hf_repo = bool((slot.hf_repo_id or "").strip())

    if backend == "faster_whisper":
        return (has_model_path, None if has_model_path else "missing_model_path")
    if backend == "mlx_whisper":
        ok = has_model_path or has_hf_repo
        return (ok, None if ok else "missing_model_ref")
    if backend == "mlx_lm":
        return (has_model_path, None if has_model_path else "missing_model_path")
    if backend in {"local_http_json", "streaming_local_http_json"}:
        return (has_endpoint, None if has_endpoint else "missing_endpoint")
    return (False, f"unsupported_backend:{backend}")


def resolve_engine_routes(
    chooser: RouteChooser,
    *,
    surface: SurfaceClass = SurfaceClass.DESKTOP,
    language: str | None = None,
) -> ResolvedEngineRoutes:
    """Resolve preview/final/writer in one shot.

    Preview ASR requires streaming; final ASR doesn't. The writer slot is
    optional so a missing writer package is not an error.
    """
    return ResolvedEngineRoutes(
        preview=resolve_slot(
            chooser, ModelSlot.PREVIEW_ASR, surface=surface, language=language, requires_streaming=True
        ),
        final=resolve_slot(
            chooser, ModelSlot.FINAL_ASR, surface=surface, language=language
        ),
        writer=resolve_slot(chooser, ModelSlot.WRITER, surface=surface, language=language),
    )


def apply_routes_to_spec(spec, routes: ResolvedEngineRoutes, *, override: bool = False):  # type: ignore[no-untyped-def]
    """Return a copy of *spec* with registry-resolved backend fields filled.

    We accept ``spec`` untyped to avoid a circular import on
    :class:`juno_v2.engine.factory.CanonicalEngineBuildSpec`. Behaviour:

    - Each slot updates only the fields the registry actually knows about.
    - When ``override`` is ``False``, we skip a field whose current value
      differs from the baseline default — the caller has taken a deliberate
      position on it.
    - When ``override`` is ``True``, registry answers win unconditionally.
    """

    def keep_current(current: Any, default: Any) -> bool:
        return current != default and not override

    kwargs: dict[str, Any] = {}

    if routes.preview is not None:
        if not keep_current(spec.preview_backend_name, _DEFAULT_PREVIEW_BACKEND):
            kwargs["preview_backend_name"] = routes.preview.backend_name
        if routes.preview.model_path is not None and (
            not keep_current(str(spec.preview_model_path), "")
            or not str(spec.preview_model_path)
        ):
            kwargs["preview_model_path"] = routes.preview.model_path
        if routes.preview.endpoint is not None and not keep_current(spec.preview_endpoint, None):
            kwargs["preview_endpoint"] = routes.preview.endpoint

    if routes.final is not None:
        if not keep_current(spec.final_backend_name, _DEFAULT_FINAL_BACKEND):
            kwargs["final_backend_name"] = routes.final.backend_name
        if routes.final.model_path is not None and (
            not keep_current(str(spec.final_model_path), "")
            or not str(spec.final_model_path)
        ):
            kwargs["final_model_path"] = routes.final.model_path
        if routes.final.endpoint is not None and not keep_current(spec.final_endpoint, None):
            kwargs["final_endpoint"] = routes.final.endpoint
        if routes.final.hf_repo_id is not None and not keep_current(spec.final_hf_repo_id, None):
            kwargs["final_hf_repo_id"] = routes.final.hf_repo_id

    if routes.writer is not None:
        if not keep_current(spec.writer_backend_name, _DEFAULT_WRITER_BACKEND):
            kwargs["writer_backend_name"] = routes.writer.backend_name
        if routes.writer.model_path is not None and not keep_current(spec.writer_model_path, None):
            kwargs["writer_model_path"] = routes.writer.model_path
        if routes.writer.endpoint is not None and not keep_current(spec.writer_endpoint, None):
            kwargs["writer_endpoint"] = routes.writer.endpoint

    return replace(spec, **kwargs) if kwargs else spec


__all__ = [
    "ResolvedEngineRoutes",
    "ResolvedSlot",
    "apply_routes_to_spec",
    "resolve_engine_routes",
    "resolve_slot",
    "resolved_slot_is_actionable",
]
