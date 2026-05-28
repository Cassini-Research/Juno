from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from juno_core_v3.model_registry.contracts import ModelPromotionStage, ModelSlot, PackageSignature, RuntimeBackend, SurfaceClass
from juno_core_v3.model_registry.manifest import CapabilityManifest
from juno_core_v3.model_registry.registry import ModelPackage, ModelRegistry


def _surface_classes(raw: list[str] | None) -> tuple[SurfaceClass, ...]:
    if not raw:
        return ()
    out: list[SurfaceClass] = []
    for x in raw:
        out.append(SurfaceClass(x))
    return tuple(out)


def manifest_from_dict(d: dict[str, Any]) -> CapabilityManifest:
    return CapabilityManifest(
        slot=ModelSlot(d["slot"]),
        backend=RuntimeBackend(d["backend"]),
        languages=tuple(d.get("languages") or ("en",)),
        domains=tuple(d.get("domains") or ()),
        quantizations=tuple(d.get("quantizations") or ("fp16",)),
        streaming=bool(d.get("streaming", False)),
        min_ram_mb=int(d.get("min_ram_mb", 0)),
        expected_working_set_mb=int(d.get("expected_working_set_mb", 0)),
        warm_load_target_s=float(d.get("warm_load_target_s", 0.0)),
        thermal_class=str(d.get("thermal_class", "unknown")),
        disallow_surfaces=_surface_classes(d.get("disallow_surfaces")),
    )


def package_from_dict(d: dict[str, Any]) -> ModelPackage:
    sig_raw = d.get("signature")
    sig = None
    if isinstance(sig_raw, dict) and "algo" in sig_raw and "value" in sig_raw:
        sig = PackageSignature(algo=str(sig_raw["algo"]), value=str(sig_raw["value"]))
    promo_raw = d.get("promotion", "candidate")
    promotion = ModelPromotionStage(str(promo_raw))
    return ModelPackage(
        package_id=str(d["package_id"]),
        version=str(d.get("version", "0")),
        manifest=manifest_from_dict(d["manifest"]),
        signature=sig,
        rollback_target=d.get("rollback_target"),
        promotion=promotion,
        metadata=dict(d.get("metadata") or {}),
    )


def load_registry_from_json_dir(directory: str | Path, *, require_integrity: bool = False) -> ModelRegistry:
    """Load ``*.json`` package descriptors from a directory into a registry.

    Optional per-package fields:
    - ``artifact_path``: relative path under ``directory`` for a file whose sha256 must match.
    - ``artifact_sha256``: hex digest for that file (integrity gate for on-disk store).
    """
    root = Path(directory)
    reg = ModelRegistry()
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if require_integrity:
            ap = data.get("artifact_path")
            exp = data.get("artifact_sha256")
            if ap and exp:
                artifact = (root / str(ap)).resolve()
                if not str(artifact).startswith(str(root.resolve())):
                    raise ValueError(f"artifact_path escapes store: {ap}")
                digest = sha256_file(artifact)
                if digest.lower() != str(exp).lower():
                    raise ValueError(f"sha256 mismatch for {path.name}: expected {exp}, got {digest}")
        pkg = package_from_dict({k: v for k, v in data.items() if k not in ("artifact_path", "artifact_sha256")})
        reg.add(pkg)
    return reg


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_package_descriptor(path: Path, package: ModelPackage, *, artifact_path: str | None = None) -> None:
    """Serialize a package to JSON (for tooling); optional artifact hashes can be added by callers."""
    man = package.manifest
    manifest_dict: dict[str, Any] = {
        "slot": man.slot.value,
        "backend": man.backend.value,
        "languages": list(man.languages),
        "domains": list(man.domains),
        "quantizations": list(man.quantizations),
        "streaming": man.streaming,
        "min_ram_mb": man.min_ram_mb,
        "expected_working_set_mb": man.expected_working_set_mb,
        "warm_load_target_s": man.warm_load_target_s,
        "thermal_class": man.thermal_class,
        "disallow_surfaces": [s.value for s in man.disallow_surfaces],
    }
    body: dict[str, Any] = {
        "package_id": package.package_id,
        "version": package.version,
        "manifest": manifest_dict,
        "rollback_target": package.rollback_target,
        "promotion": package.promotion.value,
        "metadata": dict(package.metadata),
    }
    if package.signature is not None:
        body["signature"] = {"algo": package.signature.algo, "value": package.signature.value}
    if artifact_path is not None:
        body["artifact_path"] = artifact_path
        p = path.parent / artifact_path
        if p.is_file():
            body["artifact_sha256"] = sha256_file(p)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
