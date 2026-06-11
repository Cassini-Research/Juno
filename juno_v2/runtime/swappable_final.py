"""Hot-swappable wrapper around a final-ASR backend.

Implements the :class:`FinalAsrBackend` protocol by delegating to a
mutable inner backend. Existing references — held by ``EngineSession``,
``FinalBackendTranscriber``, lifecycle metadata — stay valid across
swaps because they all point at the same wrapper instance.

The wrapper is the runtime's mutable surface for the model registry. It
takes the responsibility of:

  * mapping a registered :class:`ModelPackage` into a
    :class:`FinalAsrConfig` and instantiating the new backend
  * gating swaps on (a) no in-flight decode and (b) memory-budget fit
  * emitting trace events so observers can correlate transcripts to the
    backend that produced them
  * dropping the prior backend's reference so Python (and MLX) can
    reclaim weights — full release isn't guaranteed without a process
    restart, but freeing the strong reference is the most we can do in
    a single-process runtime.

Out of scope for this module:

  * Preview-lane and writer-lane swaps. Same shape, separate file when
    we need them.
  * Persisting the swapped selection across restarts. Restart resets to
    CLI defaults — see ``run_demo_v2.sh``.
"""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from juno_v2.contracts.final import FinalDecodeRequest, FinalDecodeResult
from juno_v2.contracts.tracing import TraceKind
from juno_v2.final.backends.base import FinalAsrBackend
from juno_v2.final.config import FinalAsrConfig
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.runtime.backends import create_final_backend

from juno_core_v3.model_registry.contracts import ModelSlot, RuntimeBackend
from juno_core_v3.model_registry.registry import ModelPackage, ModelRegistry


@dataclass(slots=True, frozen=True)
class SwapResult:
    """Outcome of a swap attempt — JSON-serializable for the HTTP layer."""

    ok: bool
    package_id: str | None = None
    backend_name: str | None = None
    error_code: str | None = None
    error: str | None = None
    load_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None or k == "ok"}


# Map RuntimeBackend enum (registry) → backend_name string (FinalAsrConfig).
_BACKEND_NAMES: dict[RuntimeBackend, str] = {
    RuntimeBackend.FASTER_WHISPER: "faster_whisper",
    RuntimeBackend.MLX_WHISPER: "mlx_whisper",
    RuntimeBackend.LOCAL_HTTP_JSON: "local_http_json",
}


def _config_from_package(
    package: ModelPackage,
    *,
    template: FinalAsrConfig,
) -> FinalAsrConfig:
    """Build a ``FinalAsrConfig`` for *package*, inheriting unspecified
    fields (device, compute_type, beam_size, language) from *template*
    so a swap doesn't accidentally regress runtime tuning that the
    operator passed on the command line."""
    backend_name = _BACKEND_NAMES.get(package.manifest.backend)
    if backend_name is None:
        raise ValueError(
            f"unsupported registry backend for swap: {package.manifest.backend}"
        )
    md = package.metadata or {}
    model_path = md.get("model_path") or template.model_path
    hf_repo_id = md.get("hf_repo_id") or template.hf_repo_id
    return FinalAsrConfig(
        model_path=Path(str(model_path)) if model_path else template.model_path,
        backend_name=backend_name,
        language=template.language,
        initial_prompt=template.initial_prompt,
        device=template.device,
        compute_type=template.compute_type,
        beam_size=template.beam_size,
        best_of=template.best_of,
        condition_on_previous_text=template.condition_on_previous_text,
        local_http_endpoint=template.local_http_endpoint,
        local_http_timeout_sec=template.local_http_timeout_sec,
        hf_repo_id=hf_repo_id,
    )


@dataclass(slots=True)
class SwappableFinalBackend:
    """Wrapper that delegates to a current inner :class:`FinalAsrBackend`.

    Constructed once at runtime startup with the engine's loaded backend.
    The workbench server holds a reference and exposes ``swap_to`` over
    HTTP so users can compare different ASR models on the same audio.
    """

    inner: FinalAsrBackend
    registry: ModelRegistry
    config_template: FinalAsrConfig
    recorder: TraceRecorder | None = None
    gpu_budget_mb: int | None = None
    initial_package_id: str | None = None
    _busy_count: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _current_package_id: str | None = None

    def __post_init__(self) -> None:
        self._current_package_id = self.initial_package_id

    # FinalAsrBackend protocol -------------------------------------------

    @property
    def backend_name(self) -> str:
        return self.inner.backend_name

    def warm(self) -> None:
        with self._lock:
            inner = self.inner
        inner.warm()

    def decode(self, req: FinalDecodeRequest) -> FinalDecodeResult:
        with self._lock:
            self._busy_count += 1
            inner = self.inner
        try:
            return inner.decode(req)
        finally:
            with self._lock:
                self._busy_count -= 1

    # Swap surface --------------------------------------------------------

    @property
    def current_package_id(self) -> str | None:
        return self._current_package_id

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy_count > 0

    def list_alternatives(self) -> list[ModelPackage]:
        """All registered final-ASR packages, most-recent-first by
        ``promotion`` ordering ('staged' before 'candidate')."""
        return [
            p for p in self.registry._packages.values()
            if p.manifest.slot == ModelSlot.FINAL_ASR
        ]

    def swap_to(self, package_id: str) -> SwapResult:
        """Atomically replace the inner backend with one built from
        *package_id* in the registry. Refuses while a decode is in
        flight (``utterance_in_progress``); if load fails, the previous
        backend is preserved."""
        package = self.registry._packages.get(package_id)
        if package is None or package.manifest.slot != ModelSlot.FINAL_ASR:
            return SwapResult(
                ok=False, package_id=package_id, error_code="unknown_package",
                error=f"no final-ASR package registered as {package_id!r}",
            )

        if self.gpu_budget_mb is not None:
            min_ram = int(package.manifest.min_ram_mb or 0)
            if min_ram > self.gpu_budget_mb:
                return SwapResult(
                    ok=False, package_id=package_id,
                    error_code="exceeds_memory_budget",
                    error=(
                        f"package requires {min_ram} MB; "
                        f"runtime budget is {self.gpu_budget_mb} MB"
                    ),
                )

        with self._lock:
            if self._busy_count > 0:
                return SwapResult(
                    ok=False, package_id=package_id,
                    error_code="utterance_in_progress",
                    error="cannot swap final backend while a decode is in flight",
                )

            self._record(
                "backend_swap_started",
                {
                    "from_package_id": self._current_package_id,
                    "to_package_id": package_id,
                    "from_backend_name": self.inner.backend_name,
                },
            )
            t0 = time.monotonic()
            try:
                config = _config_from_package(package, template=self.config_template)
                new_backend = create_final_backend(config)
                new_backend.warm()
            except Exception as exc:
                self._record(
                    "backend_swap_failed",
                    {
                        "to_package_id": package_id,
                        "error": str(exc),
                    },
                )
                return SwapResult(
                    ok=False, package_id=package_id,
                    error_code="load_failed", error=str(exc),
                )

            old_backend = self.inner
            self.inner = new_backend
            self._current_package_id = package_id
            self.config_template = config

        # Drop the strong reference outside the lock so the rest of the
        # runtime can keep decoding while Python reclaims the old weights.
        del old_backend
        gc.collect()

        load_ms = (time.monotonic() - t0) * 1000.0
        result = SwapResult(
            ok=True, package_id=package_id,
            backend_name=new_backend.backend_name,
            load_ms=load_ms,
            metadata={"languages": list(package.manifest.languages)},
        )
        self._record(
            "backend_swap_completed",
            {
                "package_id": package_id,
                "backend_name": new_backend.backend_name,
                "load_ms": load_ms,
            },
        )
        return result

    # Observability -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly summary for ``GET /api/broker/runtime/backends``."""
        alternatives = []
        for p in self.list_alternatives():
            alternatives.append({
                "package_id": p.package_id,
                "version": p.version,
                "backend": p.manifest.backend.value,
                "languages": list(p.manifest.languages),
                "promotion": p.promotion.value,
                "min_ram_mb": p.manifest.min_ram_mb,
                "wer_p50": p.manifest.wer_p50,
                "latency_ms_p50": p.manifest.latency_ms_p50,
                "is_current": p.package_id == self._current_package_id,
            })
        return {
            "current": {
                "package_id": self._current_package_id,
                "backend_name": self.inner.backend_name,
            },
            "alternatives": alternatives,
            "gpu_budget_mb": self.gpu_budget_mb,
            "busy": self.is_busy(),
        }

    def _record(self, name: str, payload: dict[str, Any]) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.record(TraceKind.SYSTEM, name, payload)
        except Exception:
            pass


__all__ = [
    "SwapResult",
    "SwappableFinalBackend",
]
