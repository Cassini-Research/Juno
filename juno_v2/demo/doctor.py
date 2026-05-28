from __future__ import annotations

import argparse
import importlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from juno_v2.demo.config import DEFAULT_DEMO_PROFILE, DemoPaths, load_demo_config
from juno_v2.demo.models import is_model_ready
from juno_v2.final.config import FinalAsrConfig
from juno_v2.runtime.platform import detect_platform_name


@dataclass(slots=True)
class DiagnosticResult:
    name: str
    ok: bool
    detail: str
    required: bool = True
    category: str = "required"
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "required": self.required,
            "category": self.category,
            "detail": self.detail,
            "metadata": self.metadata,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Juno environment doctor.")
    parser.add_argument("--profile", default=DEFAULT_DEMO_PROFILE)
    parser.add_argument("--root-dir", default=".juno_v2_demo")
    parser.add_argument("--preview-port", type=int, default=8795)
    parser.add_argument("--workbench-port", type=int, default=8765)
    parser.add_argument("--ci", action="store_true", help="Skip interactive hardware checks and fail on required setup issues.")
    parser.add_argument("--require-audio", action="store_true", help="Require a usable input audio device.")
    parser.add_argument("--skip-audio", action="store_true", help="Skip input audio device probing.")
    parser.add_argument("--json", action="store_true")
    return parser


def run_doctor(
    *,
    paths: DemoPaths,
    profile_name: str,
    preview_port: int,
    workbench_port: int,
    ci: bool = False,
    require_audio: bool = False,
    skip_audio: bool = False,
) -> list[DiagnosticResult]:
    config = load_demo_config(paths=paths, profile_name=profile_name)
    preview_service_required = config.preview_backend == "streaming_local_http_json"
    preview_service_backend = (getattr(config, "preview_service_backend", "") or "").strip().lower()
    preview_uses_mlx = preview_service_backend in {"mlx_whisper", "mlx_whisper_streaming"}
    final_backend = (config.final_backend or "").strip().lower()
    final_uses_mlx = final_backend == "mlx_whisper"
    writer_backend = (config.writer_backend or "").strip().lower() or None
    platform_name = detect_platform_name()
    mac_arm64 = platform_name == "macos" and platform.machine().lower() == "arm64"

    results = [
        DiagnosticResult(
            name="configuration",
            ok=True,
            detail=f"profile={config.profile_name} platform={platform_name} speech={config.speech_profile}",
            metadata={
                "supported_languages": list(config.supported_languages),
                "language_policy": config.language_policy,
                "speech_profile": config.speech_profile,
            },
        ),
        DiagnosticResult(
            name="runtime_plan",
            ok=True,
            detail=f"preview={config.preview_backend} final={config.final_backend} writer={config.writer_backend or 'built_in'}",
            metadata={
                "preview_backend": config.preview_backend,
                "preview_service_backend": preview_service_backend or None,
                "preview_lane_class": _preview_lane_class(config.preview_backend),
                "final_backend": config.final_backend,
                "final_lane_class": _final_lane_class(config.final_backend),
                "writer_backend": config.writer_backend,
                "preview_service_required": preview_service_required,
                "writer_endpoint": config.writer_endpoint,
                "gpu_memory_budget_mb": config.gpu_memory_budget_mb,
                "preview_residency_policy": config.preview_residency_policy,
                "final_residency_policy": config.final_residency_policy,
                "writer_residency_policy": config.writer_residency_policy,
            },
        ),
        DiagnosticResult(
            name="python",
            ok=True,
            detail=f"python={sys.executable}",
            metadata={"version": sys.version.split()[0], "in_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix)},
        ),
        _check_imports(),
        _check_model_path("preview_model", config.preview_model_path, optional=preview_uses_mlx or ci),
        _check_model_path("final_model", config.final_model_path, optional=final_uses_mlx or ci),
        _check_final_lane(
            config.final_backend,
            config.final_endpoint,
            config.final_model_path,
            config.final_repo_id,
            mac_arm64=mac_arm64,
            ci=ci,
        ),
        _check_writer_lane(
            writer_backend,
            config.writer_endpoint,
            config.writer_model_path,
            mac_arm64=mac_arm64,
            ci=ci,
        ),
        _check_port("preview_port", preview_port) if preview_service_required else DiagnosticResult("preview_port", True, "preview service not required"),
        _check_port("workbench_port", workbench_port),
    ]

    if not skip_audio and (require_audio or not ci):
        results.append(_check_audio_input(required=require_audio))
    else:
        results.append(DiagnosticResult("audio_input", True, "skipped for this run", required=False, category="optional"))

    if platform_name == "macos":
        results.append(_check_macos_permissions(required=not ci))
    else:
        results.append(DiagnosticResult("macos_permissions", True, "not required on this platform", required=False, category="platform"))
    return results


def _check_imports() -> DiagnosticResult:
    modules = ["numpy", "sounddevice", "torch", "faster_whisper", "silero_vad", "webrtcvad", "dateparser"]
    missing: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        return DiagnosticResult("python_imports", False, f"missing={','.join(missing)}")
    return DiagnosticResult("python_imports", True, "required Python imports succeeded")


def _check_audio_input(*, required: bool) -> DiagnosticResult:
    category = "required" if required else "optional"
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        return DiagnosticResult("audio_input", not required, f"sounddevice import failed: {exc}", required=required, category=category)
    try:
        devices = sd.query_devices()
        input_devices = [dev for dev in devices if int(dev.get("max_input_channels", 0)) > 0]
    except Exception as exc:
        return DiagnosticResult("audio_input", not required, f"device query failed: {exc}", required=required, category=category)
    return DiagnosticResult(
        "audio_input",
        ok=bool(input_devices) or not required,
        detail=f"input_devices={len(input_devices)}",
        required=required,
        category=category,
    )


def _check_macos_permissions(*, required: bool) -> DiagnosticResult:
    category = "required" if required else "platform"
    try:
        from juno_v2.context.macos_desktop import MacOSDesktopContextProvider

        caps = MacOSDesktopContextProvider().capabilities()
    except Exception as exc:
        return DiagnosticResult("macos_permissions", not required, f"macOS capability probe failed: {exc}", required=required, category=category)
    ok = bool(caps.get("osascript_available", False))
    return DiagnosticResult(
        "macos_permissions",
        ok=ok or not required,
        detail="macOS desktop automation tools detected" if ok else "macOS desktop automation tools unavailable",
        required=required,
        category=category,
        metadata=caps,
    )


def _check_model_path(name: str, model_path: Path, *, optional: bool) -> DiagnosticResult:
    ok = is_model_ready(model_path)
    return DiagnosticResult(
        name,
        ok=ok or optional,
        detail=f"path={model_path}" + ("" if ok else " (not present locally)"),
        required=not optional,
        category="required" if not optional else "optional",
    )


def _final_lane_class(backend: str) -> str:
    if backend == "mlx_whisper":
        return "apple_silicon_native"
    if backend == "local_http_json":
        return "external_local_service"
    return "embedded_whisper"


def _preview_lane_class(backend: str) -> str:
    if backend == "streaming_local_http_json":
        return "local_streaming_service"
    if backend == "local_http_json":
        return "external_local_service"
    return "embedded_whisper"


def _check_final_lane(
    final_backend: str,
    final_endpoint: str | None,
    final_model_path: Path,
    final_repo_id: str,
    *,
    mac_arm64: bool,
    ci: bool,
) -> DiagnosticResult:
    if final_backend == "local_http_json":
        if not final_endpoint:
            return DiagnosticResult("final_lane", False, "final_backend=local_http_json requires final_endpoint")
        return DiagnosticResult("final_lane", True, "external local final service configured", metadata={"final_backend": final_backend, "final_endpoint": final_endpoint})
    if final_backend == "mlx_whisper":
        required = mac_arm64 and not ci
        category = "required" if required else "optional"
        if not mac_arm64:
            return DiagnosticResult("final_lane", True, "mlx_whisper is Mac Apple Silicon only; not required on this platform", required=False, category="platform")
        try:
            importlib.import_module("mlx_whisper")
        except Exception:
            return DiagnosticResult("final_lane", False if required else True, "mlx_whisper is not installed", required=required, category=category)
        mlx_core = _probe_mlx_core_import()
        if not mlx_core.ok:
            return DiagnosticResult("final_lane", False if required else True, mlx_core.detail, required=required, category=category, metadata=mlx_core.metadata)
        from juno_v2.final.backends.mlx_whisper import MlxWhisperFinalBackend

        backend = MlxWhisperFinalBackend(FinalAsrConfig(model_path=final_model_path, backend_name="mlx_whisper", hf_repo_id=final_repo_id))
        model_ref, source_type = backend._resolve_model_ref()
        return DiagnosticResult(
            "final_lane",
            bool(model_ref) or not required,
            f"mlx_whisper ready: source_type={source_type} ref={model_ref}",
            required=required,
            category=category,
            metadata={"final_backend": final_backend, "model_source_type": source_type, "model_ref": model_ref},
        )
    return DiagnosticResult("final_lane", True, f"embedded final backend configured: {final_backend}", metadata={"final_backend": final_backend})


def _probe_mlx_core_import() -> DiagnosticResult:
    proc = subprocess.run([sys.executable, "-c", "import mlx.core as mx; print('mlx.core ok')"], capture_output=True, text=True)
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        return DiagnosticResult("mlx_core_import", False, "mlx.core import probe failed", metadata={"returncode": proc.returncode, "output": out.strip()[:800]})
    return DiagnosticResult("mlx_core_import", True, "mlx.core import probe ok")


def _check_writer_lane(
    writer_backend: str | None,
    writer_endpoint: str | None,
    writer_model_path: str | None,
    *,
    mac_arm64: bool,
    ci: bool,
) -> DiagnosticResult:
    if not writer_backend:
        return DiagnosticResult("writer_lane", True, "built-in writer paths available", metadata={"writer_backend": None})
    if writer_backend == "local_http_json" and not writer_endpoint:
        return DiagnosticResult("writer_lane", False, "writer_backend=local_http_json requires writer_endpoint")
    if writer_backend == "mlx_lm":
        required = mac_arm64 and not ci
        category = "required" if required else "optional"
        if not mac_arm64:
            return DiagnosticResult("writer_lane", True, "mlx_lm is Mac Apple Silicon only; not required on this platform", required=False, category="platform")
        proc = subprocess.run([sys.executable, "-c", "import mlx_lm; print('mlx_lm ok')"], capture_output=True, text=True)
        if proc.returncode != 0:
            out = ((proc.stdout or "") + (proc.stderr or "")).strip()[:800]
            return DiagnosticResult("writer_lane", False if required else True, "mlx_lm import probe failed", required=required, category=category, metadata={"returncode": proc.returncode, "output": out})
        return DiagnosticResult("writer_lane", True, f"mlx_lm writer configured: {writer_model_path}", required=required, category=category)
    return DiagnosticResult("writer_lane", True, f"writer backend configured: {writer_backend}", metadata={"writer_backend": writer_backend})


def _check_port(name: str, port: int) -> DiagnosticResult:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        return DiagnosticResult(name, False, f"port {port} is already in use")
    finally:
        sock.close()
    return DiagnosticResult(name, True, f"port {port} is available")


def _print_human_results(results: list[DiagnosticResult]) -> None:
    for item in results:
        if item.ok:
            prefix = "OK"
        elif item.required:
            prefix = "FAIL"
        else:
            prefix = "WARN"
        scope = "required" if item.required else item.category
        print(f"[{prefix}] {item.name} ({scope}): {item.detail}")


def main() -> int:
    args = build_arg_parser().parse_args()
    paths = DemoPaths(root_dir=Path(args.root_dir))
    results = run_doctor(
        paths=paths,
        profile_name=args.profile,
        preview_port=args.preview_port,
        workbench_port=args.workbench_port,
        ci=args.ci,
        require_audio=args.require_audio,
        skip_audio=args.skip_audio,
    )
    required_failures = [item for item in results if item.required and not item.ok]
    payload = {
        "ok": not required_failures,
        "required_failures": [item.name for item in required_failures],
        "results": [item.to_dict() for item in results],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human_results(results)
        if required_failures:
            print("\nNext steps:")
            for item in required_failures:
                print(f"- Fix {item.name}: {item.detail}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
