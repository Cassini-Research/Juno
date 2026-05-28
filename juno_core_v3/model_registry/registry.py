from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from juno_core_v3.model_registry.contracts import ModelPromotionStage, ModelSlot, PackageSignature
from juno_core_v3.model_registry.manifest import CapabilityManifest
from juno_core_v3.model_registry.signature import canonical_payload, verify_signature


@dataclass(slots=True)
class ModelPackage:
    package_id: str
    version: str
    manifest: CapabilityManifest
    signature: PackageSignature | None = None
    rollback_target: str | None = None
    promotion: ModelPromotionStage = ModelPromotionStage.CANDIDATE
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "package_id": self.package_id,
            "version": self.version,
            "manifest": {
                "slot": self.manifest.slot.value,
                "backend": self.manifest.backend.value,
                "languages": list(self.manifest.languages),
                "domains": list(self.manifest.domains),
                "quantizations": list(self.manifest.quantizations),
                "streaming": self.manifest.streaming,
                "min_ram_mb": self.manifest.min_ram_mb,
                "expected_working_set_mb": self.manifest.expected_working_set_mb,
                "warm_load_target_s": self.manifest.warm_load_target_s,
                "thermal_class": self.manifest.thermal_class,
                "disallow_surfaces": [s.value for s in self.manifest.disallow_surfaces],
                "wer_p50": self.manifest.wer_p50,
                "latency_ms_p50": self.manifest.latency_ms_p50,
            },
            "signature": None if self.signature is None else {"algo": self.signature.algo, "value": self.signature.value},
            "rollback_target": self.rollback_target,
            "promotion": self.promotion.value,
            "metadata": dict(self.metadata),
        }


class ModelRegistry:
    """In-memory registry (Phase 4).

    Later phases can back this with disk state + signed package verification.
    """

    def __init__(
        self,
        *,
        trust_keys: Mapping[str, bytes] | None = None,
    ) -> None:
        self._packages: dict[str, ModelPackage] = {}
        self._trust_keys: Mapping[str, bytes] | None = trust_keys

    def add(self, package: ModelPackage) -> None:
        if package.package_id in self._packages:
            raise ValueError(f"package already registered: {package.package_id}")
        if self._trust_keys is not None:
            payload = canonical_payload(package.to_dict())
            verdict = verify_signature(
                payload=payload,
                signature=package.signature,
                trust_keys=self._trust_keys,
            )
            if not verdict.ok:
                raise ValueError(f"signature_rejected: {verdict.reason}")
        self._packages[package.package_id] = package

    def get(self, package_id: str) -> ModelPackage | None:
        return self._packages.get(package_id)

    def list(self, *, slot: ModelSlot | None = None) -> list[ModelPackage]:
        out = list(self._packages.values())
        if slot is not None:
            out = [p for p in out if p.manifest.slot == slot]
        return sorted(out, key=lambda p: (p.manifest.slot.value, p.package_id))

    def promote(self, package_id: str) -> None:
        pkg = self._require(package_id)
        pkg.promotion = ModelPromotionStage.PROMOTED

    def stage(self, package_id: str) -> None:
        pkg = self._require(package_id)
        pkg.promotion = ModelPromotionStage.STAGED

    def retire(self, package_id: str) -> None:
        pkg = self._require(package_id)
        pkg.promotion = ModelPromotionStage.RETIRED

    def rollback(self, from_package_id: str) -> ModelPackage:
        pkg = self._require(from_package_id)
        if not pkg.rollback_target:
            raise ValueError(f"no rollback_target configured for {from_package_id}")
        target = self._require(pkg.rollback_target)
        return target

    def _require(self, package_id: str) -> ModelPackage:
        pkg = self.get(package_id)
        if pkg is None:
            raise KeyError(package_id)
        return pkg
