from __future__ import annotations

from dataclasses import dataclass

from juno_core_v3.model_registry.contracts import ModelSlot, RuntimeBackend, SurfaceClass


@dataclass(slots=True, frozen=True)
class CapabilityManifest:
    """Model capability manifest.

    This is intentionally **symbolic and inspectable**: it describes constraints and
    expected behavior without binding to one model family.
    """

    slot: ModelSlot
    backend: RuntimeBackend

    # Supported languages/domains are declared at the package level.
    languages: tuple[str, ...] = ("en",)
    domains: tuple[str, ...] = ()

    # Deployment/runtime behavior
    quantizations: tuple[str, ...] = ("fp16",)
    streaming: bool = False
    min_ram_mb: int = 0
    expected_working_set_mb: int = 0
    warm_load_target_s: float = 0.0
    thermal_class: str = "unknown"  # e.g. "desktop", "phone"

    # Surface support constraints
    disallow_surfaces: tuple[SurfaceClass, ...] = ()

    # Observed quality / latency metrics populated from the eval gates
    # (see ``apply_eval_report``). ``None`` means "not measured yet"
    # and is the correct default for a freshly-bundled package; the
    # router must tolerate it rather than assume the model is bad.
    # ``wer_p50`` is a fraction in [0, 1] (word error rate). 
    # ``latency_ms_p50`` is the end-to-end utterance latency in
    # milliseconds measured against the eval corpus.
    wer_p50: float | None = None
    latency_ms_p50: float | None = None

    def supports_language(self, language: str | None) -> bool:
        if language is None:
            return True
        return language in self.languages

    def allows_surface(self, surface: SurfaceClass) -> bool:
        return surface not in self.disallow_surfaces

    def meets_streaming(self, requires_streaming: bool) -> bool:
        return (not requires_streaming) or self.streaming
