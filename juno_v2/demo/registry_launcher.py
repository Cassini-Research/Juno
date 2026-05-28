from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_LAST_RESOLVED_SLOTS_TABLE: dict[str, str] | None = None


class RegistryConfigError(Exception):
    """Raised for malformed or inconsistent registry launcher configs."""

    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


def load_registry_config(path: Path | str) -> dict[str, Any]:
    """Parse + validate the registry JSON. Raises RegistryConfigError on
    any problem. Does NOT construct a ModelRegistry yet."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryConfigError(f"registry_config_read_error: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryConfigError(
            f"registry_config_malformed_json: {exc.msg} (line {exc.lineno} col {exc.colno})"
        ) from exc
    if not isinstance(data, dict):
        raise RegistryConfigError("registry_config_malformed_json: root must be an object")
    version = data.get("version")
    if version != 1:
        raise RegistryConfigError(f"registry_config_unsupported_version: {version!r}")
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RegistryConfigError("registry_config_packages_invalid: packages must be a non-empty list")
    from juno_core_v3.model_registry.contracts import ModelSlot, RuntimeBackend

    required_top = ("surface",)
    for key in required_top:
        if key not in data:
            raise RegistryConfigError(f"registry_config_missing_field: {key}")

    slots_seen: set[str] = set()
    for i, pkg in enumerate(packages):
        if not isinstance(pkg, dict):
            raise RegistryConfigError(f"registry_config_packages_invalid: entry {i} must be an object")
        for name in ("package_id", "version", "slot", "backend", "model_path"):
            if name not in pkg:
                raise RegistryConfigError(f"registry_config_missing_field: {name}")
        slot_s = pkg["slot"]
        try:
            ModelSlot(slot_s)
        except ValueError:
            raise RegistryConfigError(f"registry_config_unknown_slot: {slot_s!r}") from None
        slots_seen.add(slot_s)
        backend_s = pkg["backend"]
        try:
            RuntimeBackend(backend_s)
        except ValueError:
            raise RegistryConfigError(f"registry_config_unknown_backend: {backend_s!r}") from None

    for required in ("preview_asr", "final_asr"):
        if required not in slots_seen:
            raise RegistryConfigError(f"registry_config_missing_required_slot: {required}")

    return data


def build_registry_from_config(
    config: dict[str, Any],
) -> tuple[object, dict[str, str]]:
    """Build an in-memory ModelRegistry from a parsed config. Returns
    (registry, resolved_slots_table) where resolved_slots_table is a
    dict like {"preview_asr": "preview-faster-whisper-small-en", ...}.
    """
    from juno_core_v3.model_registry.contracts import ModelPromotionStage, ModelSlot, RuntimeBackend
    from juno_core_v3.model_registry.keystore import load_keystore, sign_packages
    from juno_core_v3.model_registry.manifest import CapabilityManifest
    from juno_core_v3.model_registry.registry import ModelPackage, ModelRegistry

    # Resolve trust keys before building the registry. Config entries
    # are unsigned JSON, so when an explicit keystore is active we
    # HMAC-sign each package on the way in.
    try:
        trust_keys = load_keystore()
    except ValueError as exc:
        raise RegistryConfigError(f"keystore_invalid: {exc}") from exc

    registry = ModelRegistry(trust_keys=trust_keys)
    last_by_slot: dict[ModelSlot, ModelPackage] = {}
    packages_raw = config["packages"]
    built: list[ModelPackage] = []
    for pkg in packages_raw:
        slot = ModelSlot(pkg["slot"])
        langs_raw = pkg.get("languages", ("en",))
        if isinstance(langs_raw, list):
            languages = tuple(str(x) for x in langs_raw)
        else:
            languages = ("en",)
        streaming = slot == ModelSlot.PREVIEW_ASR
        manifest = CapabilityManifest(
            slot=slot,
            backend=RuntimeBackend(pkg["backend"]),
            languages=languages,
            streaming=streaming,
        )
        mp = ModelPackage(
            package_id=pkg["package_id"],
            version=pkg["version"],
            manifest=manifest,
            metadata={
                "model_path": pkg["model_path"],
                "endpoint": pkg.get("endpoint"),
                "hf_repo_id": pkg.get("hf_repo_id"),
            },
        )
        built.append(mp)

    if trust_keys:
        sign_packages(built, trust_keys=trust_keys)

    for mp in built:
        slot = mp.manifest.slot
        if slot in last_by_slot:
            prev = last_by_slot[slot]
            _logger.warning(
                "registry_config_duplicate_slot: %s %s",
                prev.package_id,
                mp.package_id,
            )
            prev.promotion = ModelPromotionStage.RETIRED
        registry.add(mp)
        last_by_slot[slot] = mp

    resolved: dict[str, str] = {
        s.value: last_by_slot[s].package_id for s in last_by_slot
    }
    return registry, resolved


def launch_engine_from_config(
    path: Path | str,
    *,
    session_id: str | None = None,
) -> object:
    """Top-level entry point. Loads the config, builds a registry,
    synthesises a minimal CanonicalEngineBuildSpec, and calls
    build_canonical_engine_from_registry. Returns the
    CanonicalEngineArtifacts object."""
    global _LAST_RESOLVED_SLOTS_TABLE
    config = load_registry_config(path)
    registry, resolved = build_registry_from_config(config)
    _LAST_RESOLVED_SLOTS_TABLE = resolved

    from juno_v2.engine.factory import CanonicalEngineBuildSpec, build_canonical_engine_from_registry

    prefix = config.get("session_id_prefix", "demo_registry")
    sid = session_id or f"{prefix}_{uuid.uuid4().hex[:8]}"
    spec = CanonicalEngineBuildSpec(
        engine_mode="live",
        session_id=sid,
        preview_model_path="",
        final_model_path="",
        language=config.get("language"),
    )
    return build_canonical_engine_from_registry(
        spec,
        registry=registry,
        surface=config["surface"],
        override_spec=False,
    )


def serve_from_config(
    path: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> tuple[object, object, object]:
    """Launch registry-backed engine artifacts and serve the workbench HTTP API."""

    artifacts = launch_engine_from_config(path)

    from juno_core_v3.dictation import FinalBackendTranscriber
    from juno_v2.runtime.config import WorkbenchRuntimeConfig
    from juno_v2.workbench.server import WorkbenchApp, start_http_server

    final_backend = getattr(artifacts, "final_backend", None)
    language = getattr(getattr(artifacts, "runner", None), "language", None)
    transcriber = (
        FinalBackendTranscriber(backend=final_backend, language=language)
        if final_backend is not None
        else None
    )
    runner = getattr(artifacts, "runner", None)
    app = WorkbenchApp(
        WorkbenchRuntimeConfig(host=host, port=port),
        session_id=getattr(runner, "session_id", None),
        recorder=getattr(artifacts, "recorder", None),
        store=getattr(artifacts, "store", None),
        commit=getattr(artifacts, "controller", None),
        transcriber=transcriber,
        memory=getattr(artifacts, "memory_store", None),
        context_provider=getattr(runner, "context_provider", None),
        writer_service=getattr(artifacts, "writer_service", None),
        writer_backend=getattr(getattr(artifacts, "writer_service", None), "backend", None),
        language_planner=getattr(runner, "language_planner", None),
        bias_engine=getattr(runner, "bias_engine", None),
        context_plane=getattr(runner, "context_plane", None),
        juno_seed_runtime=getattr(runner, "juno_seed_runtime", None),
    )
    httpd, thread = start_http_server(app)
    return app, httpd, thread


if __name__ == "__main__":
    import argparse as _argparse

    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _root.addHandler(_h)

    parser = _argparse.ArgumentParser(
        description="Launch Juno v2 engine from a ModelRegistry JSON config."
    )
    parser.add_argument("config", help="Path to registry JSON config.")
    args = parser.parse_args()

    logging.info("registry_launcher_start: config_path=%s", args.config)
    launch_engine_from_config(args.config)
    table = _LAST_RESOLVED_SLOTS_TABLE or {}
    for key in sorted(table):
        print(f"{key}: {table[key]}")
