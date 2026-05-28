from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

from juno_v2.demo.config import DEFAULT_DEMO_PROFILE, DemoConfig, DemoPaths, load_demo_config
from juno_v2.demo.models import provision_demo_models


def _detect_macos_context_helper() -> str | None:
    """Locate the ``juno-capability`` Swift helper if we're on macOS.

    When present, returns an absolute command string suitable for
    ``--context-helper-command`` so the ``macos_desktop`` context
    provider can harvest frontmost-app identity, focused text, and
    clipboard from the live UI. Without this, the broker falls back
    to a workbench-only context that the Mac shell never populates
    (see ``MacOSDesktopContextProvider`` vs ``WorkbenchContextProvider``
    — only the former reads native AX state).

    Resolution order:

    1. ``JUNO_V2_CONTEXT_HELPER_COMMAND`` explicit override (any OS).
    2. ``shells/macos/.build/<release|debug>/juno-capability`` — the
       artifact ``swift build`` drops next to JunoShell.
    3. ``juno-capability`` on ``PATH`` (user installed it globally).

    Returns ``None`` when nothing is found or we're not on macOS; the
    caller keeps the legacy ``--context-source workbench`` behaviour.
    """
    override = os.environ.get("JUNO_V2_CONTEXT_HELPER_COMMAND")
    if override:
        return override
    if sys.platform != "darwin":
        return None
    repo_root = Path(__file__).resolve().parents[2]
    for variant in ("release", "debug"):
        candidate = repo_root / "shells" / "macos" / ".build" / variant / "juno-capability"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    path_bin = shutil.which("juno-capability")
    if path_bin:
        return path_bin
    return None


@dataclass(slots=True)
class RunningProcess:
    name: str
    proc: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIOWrapper


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Juno local source stack")
    parser.add_argument("--profile", default=DEFAULT_DEMO_PROFILE)
    parser.add_argument("--root-dir", default=".juno_v2_demo")
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--replay-wav", default=None)
    parser.add_argument("--preview-port", type=int, default=8795)
    parser.add_argument("--workbench-port", type=int, default=8765)
    parser.add_argument("--preview-host", default="127.0.0.1")
    parser.add_argument("--workbench-host", default="127.0.0.1")
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--force-model-download", action="store_true")
    parser.add_argument("--speech-profile", default=None)
    parser.add_argument("--preview-backend", default=None)
    parser.add_argument("--final-backend", default=None)
    parser.add_argument("--writer-backend", default=None)
    parser.add_argument("--writer-model-path", default=None)
    parser.add_argument("--writer-residency-policy", default=None)
    parser.add_argument("--writer-idle-unload-ttl-s", type=float, default=None)
    parser.add_argument("--preview-endpoint", default=None)
    parser.add_argument("--final-endpoint", default=None)
    parser.add_argument("--writer-endpoint", default=None)
    return parser


def build_preview_service_command(
    *,
    python_bin: str,
    preview_host: str,
    preview_port: int,
    model_path: Path,
    device: str,
    compute_type: str,
    backend: str = "faster_whisper",
    hf_repo_id: str | None = None,
) -> list[str]:
    command = [
        python_bin,
        "-m",
        "juno_v2.preview.streaming_service",
        "--backend",
        backend,
        "--host",
        preview_host,
        "--port",
        str(preview_port),
        "--model-path",
        str(model_path),
        "--device",
        device,
        "--compute-type",
        compute_type,
    ]
    if hf_repo_id:
        command.extend(["--hf-repo-id", hf_repo_id])
    return command


def build_runtime_service_command(
    *,
    python_bin: str,
    mode: str,
    preview_backend: str,
    final_backend: str,
    writer_backend: str | None,
    writer_model_path: str | None,
    writer_max_tokens: int,
    writer_temperature: float,
    writer_top_p: float,
    preview_endpoint: str | None,
    final_endpoint: str | None,
    writer_endpoint: str | None,
    preview_model_path: Path,
    final_model_path: Path,
    runtime_dir: Path,
    memory_dir: Path,
    workbench_host: str,
    workbench_port: int,
    supported_languages: tuple[str, ...],
    language: str | None,
    language_policy: str,
    preview_device: str,
    final_device: str,
    preview_compute_type: str,
    final_compute_type: str,
    speech_profile: str = "standard",
    gpu_memory_budget_mb: int | None = None,
    preview_gpu_memory_mb: int = 0,
    final_gpu_memory_mb: int = 0,
    writer_gpu_memory_mb: int = 0,
    live_corrector_gpu_memory_mb: int = 0,
    preview_residency_policy: str = 'resident',
    final_residency_policy: str = 'resident',
    writer_residency_policy: str = 'resident',
    live_corrector_residency_policy: str = 'resident',
    writer_idle_unload_ttl_s: float | None = None,
    replay_wav: str | None = None,
    log_dir: Path | None = None,
    summary_json: Path | None = None,
    startup_profile_json: Path | None = None,
    health_json: Path | None = None,
    final_hf_repo_id: str | None = None,
    context_source: str = "workbench",
    context_helper_command: str | None = None,
    live_corrector_backend: str | None = None,
    live_corrector_model_path: str | None = None,
    live_corrector_max_tokens: int = 160,
    live_corrector_temperature: float = 0.0,
    live_corrector_top_p: float = 1.0,
) -> list[str]:
    command = [
        python_bin,
        "-m",
        "juno_v2.runtime.service",
        "--mode",
        mode,
        "--preview-backend",
        preview_backend,
        "--final-backend",
        final_backend,
        "--preview-model-path",
        str(preview_model_path),
        "--final-model-path",
        str(final_model_path),
        "--context-source",
        context_source,
        "--runtime-dir",
        str(runtime_dir),
        "--memory-dir",
        str(memory_dir),
        "--supported-languages",
        ",".join(supported_languages),
        "--language-policy",
        language_policy,
        "--preview-device",
        preview_device,
        "--final-device",
        final_device,
        "--preview-compute-type",
        preview_compute_type,
        "--final-compute-type",
        final_compute_type,
        "--speech-profile",
        speech_profile,
    ]
    if gpu_memory_budget_mb is not None:
        command.extend(['--gpu-memory-budget-mb', str(gpu_memory_budget_mb)])
    command.extend(['--preview-gpu-memory-mb', str(preview_gpu_memory_mb), '--final-gpu-memory-mb', str(final_gpu_memory_mb), '--writer-gpu-memory-mb', str(writer_gpu_memory_mb), '--live-corrector-gpu-memory-mb', str(live_corrector_gpu_memory_mb)])
    command.extend(['--preview-residency-policy', preview_residency_policy, '--final-residency-policy', final_residency_policy, '--writer-residency-policy', writer_residency_policy, '--live-corrector-residency-policy', live_corrector_residency_policy])
    # The source launcher serves the embedded HTTP workbench so callers
    # of juno_v2.demo.launcher can hit
    # http://workbench_host:workbench_port for /healthz, /api/broker/*,
    # etc. Production (Juno.app) goes over UDS via --engine-socket and
    # does not need this; that path is in shells/macos/.
    command.extend([
        '--serve-workbench',
        '--workbench-host', workbench_host,
        '--workbench-port', str(workbench_port),
    ])
    if writer_idle_unload_ttl_s is not None:
        command.extend(["--writer-idle-unload-ttl-s", str(float(writer_idle_unload_ttl_s))])
    if preview_endpoint:
        command.extend(["--preview-endpoint", preview_endpoint])
    if final_endpoint:
        command.extend(["--final-endpoint", final_endpoint])
    if writer_backend:
        command.extend(["--writer-backend", writer_backend])
    if writer_model_path:
        command.extend(["--writer-model-path", writer_model_path])
    command.extend(["--writer-max-tokens", str(writer_max_tokens), "--writer-temperature", str(writer_temperature), "--writer-top-p", str(writer_top_p)])
    if writer_endpoint:
        command.extend(["--writer-endpoint", writer_endpoint])
    if live_corrector_backend:
        command.extend(["--live-corrector-enabled", "--live-corrector-backend", live_corrector_backend])
    if live_corrector_model_path:
        command.extend(["--live-corrector-model-path", live_corrector_model_path])
    command.extend([
        "--live-corrector-max-tokens", str(live_corrector_max_tokens),
        "--live-corrector-temperature", str(live_corrector_temperature),
        "--live-corrector-top-p", str(live_corrector_top_p),
    ])
    if log_dir is not None:
        command.extend(["--log-dir", str(log_dir)])
    if summary_json is not None:
        command.extend(["--summary-json", str(summary_json)])
    if startup_profile_json is not None:
        command.extend(["--startup-profile-json", str(startup_profile_json)])
    if health_json is not None:
        command.extend(["--health-json", str(health_json)])
    if language:
        command.extend(["--language", language])
    if final_hf_repo_id:
        command.extend(["--final-hf-repo-id", final_hf_repo_id])
    if context_helper_command:
        command.extend(["--context-helper-command", context_helper_command])
    if mode == "replay":
        if not replay_wav:
            raise ValueError("replay_wav is required in replay mode")
        command.extend(["--replay-wav", replay_wav])
    return command


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = DemoPaths(root_dir=Path(args.root_dir))
    config = load_demo_config(paths=paths, profile_name=args.profile)
    if config.profile_name != args.profile:
        config = DemoConfig.from_profile(args.profile, paths=paths)
    if args.speech_profile is not None:
        config.speech_profile = args.speech_profile
    if args.preview_backend is not None:
        config.preview_backend = args.preview_backend
    if args.final_backend is not None:
        config.final_backend = args.final_backend
    if args.writer_backend is not None:
        config.writer_backend = args.writer_backend
    if args.writer_model_path is not None:
        config.writer_model_path = args.writer_model_path
    if args.writer_residency_policy is not None:
        config.writer_residency_policy = args.writer_residency_policy
    if args.preview_endpoint is not None:
        config.preview_endpoint = args.preview_endpoint
    if args.final_endpoint is not None:
        config.final_endpoint = args.final_endpoint
    if args.writer_endpoint is not None:
        config.writer_endpoint = args.writer_endpoint
    config = provision_demo_models(config, paths=paths, force=args.force_model_download)
    paths.ensure_dirs()
    python_bin = sys.executable
    preview_port = choose_available_port(args.preview_host, args.preview_port)
    workbench_port = choose_available_port(args.workbench_host, args.workbench_port)
    workbench_url = f"http://{args.workbench_host}:{workbench_port}"

    # Pick the right context plane for the host OS. On macOS we wire
    # the native Accessibility helper so the one-shot pipeline sees
    # the focused app / window / selection in real time. On Linux /
    # replay / CI we keep the workbench context provider which is
    # driven by ``POST /api/sync`` from a test harness.
    context_helper_command = _detect_macos_context_helper()
    context_source = "macos_desktop" if context_helper_command else "workbench"

    preview_process: RunningProcess | None = None
    preview_endpoint = config.preview_endpoint
    if config.preview_backend == "streaming_local_http_json":
        preview_endpoint = preview_endpoint or f"http://{args.preview_host}:{preview_port}"
        preview_process = start_logged_process(
            name="preview_service",
            command=build_preview_service_command(
                python_bin=python_bin,
                preview_host=args.preview_host,
                preview_port=preview_port,
                model_path=config.preview_model_path,
                device=config.preview_device,
                compute_type=config.preview_compute_type,
                backend=config.preview_service_backend,
                hf_repo_id=config.preview_repo_id,
            ),
            log_path=paths.resolved_preview_service_log(),
        )
    elif config.preview_backend == "local_http_json" and not preview_endpoint:
        raise RuntimeError("preview_backend=local_http_json requires a preview endpoint")

    runtime_process: RunningProcess | None = None
    try:
        if preview_process is not None:
            wait_for_health(f"{preview_endpoint}/healthz", preview_process)
        runtime_process = start_logged_process(
            name="runtime_service",
            command=build_runtime_service_command(
                python_bin=python_bin,
                mode=args.mode,
                replay_wav=args.replay_wav,
                preview_backend=config.preview_backend,
                final_backend=config.final_backend,
                writer_backend=config.writer_backend,
                writer_model_path=config.writer_model_path,
                writer_max_tokens=config.writer_max_tokens,
                writer_temperature=config.writer_temperature,
                writer_top_p=config.writer_top_p,
                live_corrector_backend=config.live_corrector_backend,
                live_corrector_model_path=config.live_corrector_model_path,
                live_corrector_max_tokens=config.live_corrector_max_tokens,
                live_corrector_temperature=config.live_corrector_temperature,
                live_corrector_top_p=config.live_corrector_top_p,
                preview_endpoint=preview_endpoint,
                final_endpoint=config.final_endpoint,
                writer_endpoint=config.writer_endpoint,
                preview_model_path=config.preview_model_path,
                final_model_path=config.final_model_path,
                runtime_dir=paths.resolved_runtime_dir(),
                memory_dir=paths.resolved_memory_dir(),
                workbench_host=args.workbench_host,
                workbench_port=workbench_port,
                supported_languages=config.supported_languages,
                language=config.language,
                language_policy=config.language_policy,
                preview_device=config.preview_device,
                final_device=config.final_device,
                preview_compute_type=config.preview_compute_type,
                final_compute_type=config.final_compute_type,
                speech_profile=config.speech_profile,
                gpu_memory_budget_mb=config.gpu_memory_budget_mb,
                preview_gpu_memory_mb=config.preview_gpu_memory_mb,
                final_gpu_memory_mb=config.final_gpu_memory_mb,
                writer_gpu_memory_mb=config.writer_gpu_memory_mb,
                live_corrector_gpu_memory_mb=config.live_corrector_gpu_memory_mb,
                preview_residency_policy=config.preview_residency_policy,
                final_residency_policy=config.final_residency_policy,
                writer_residency_policy=config.writer_residency_policy,
                live_corrector_residency_policy=config.live_corrector_residency_policy,
                writer_idle_unload_ttl_s=args.writer_idle_unload_ttl_s,
                log_dir=paths.resolved_logs_dir() / "service",
                summary_json=paths.resolved_runtime_dir() / "summary.json",
                startup_profile_json=paths.resolved_runtime_dir() / "startup_profile.json",
                health_json=paths.resolved_runtime_dir() / "health.json",
                final_hf_repo_id=config.final_repo_id,
                context_source=context_source,
                context_helper_command=context_helper_command,
            ),
            log_path=paths.resolved_runtime_service_log(),
        )
        if args.mode != "replay":
            # The canonical best_local path warms the MLX final lane at startup (fail-fast).
            # First-run HF downloads can exceed the default health timeout; keep the timeout
            # scoped to this demo launcher path.
            warm_timeout_sec = 600.0 if config.final_backend == "mlx_whisper" else 120.0
            wait_for_health(f"{workbench_url}/healthz", runtime_process, timeout_sec=warm_timeout_sec)
        trace_dir = paths.resolved_logs_dir() / "service"
        payload = {
            "ok": True,
            "mode": args.mode,
            "profile": config.profile_name,
            "profile_class": config.profile_class,
            "target_machine": config.target_machine,
            "notes": list(config.notes),
            "speech_profile": config.speech_profile,
            "preview_backend": config.preview_backend,
            "final_backend": config.final_backend,
            "writer_backend": config.writer_backend,
            "writer_model_path": config.writer_model_path,
            "live_corrector_backend": config.live_corrector_backend,
            "live_corrector_model_path": config.live_corrector_model_path,
            "preview_endpoint": preview_endpoint,
            "final_endpoint": config.final_endpoint,
            "writer_endpoint": config.writer_endpoint,
            "preview_service_started": preview_process is not None,
            "workbench_url": workbench_url,
            "preview_log": None if preview_process is None else str(preview_process.log_path),
            "runtime_log": str(runtime_process.log_path),
            "runtime_dir": str(paths.resolved_runtime_dir()),
            "trace_log_dir": str(trace_dir),
            "session_jsonl_glob": str(trace_dir / "*.jsonl"),
            "context_source": context_source,
            "context_helper_command": context_helper_command,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not args.no_open_browser:
            webbrowser.open(workbench_url)
        if args.mode == "replay":
            # Replay runs are finite; exit cleanly when the runtime finishes.
            code = runtime_process.proc.wait()
            if code != 0:
                raise RuntimeError(f"runtime_service exited with code {code}. See {runtime_process.log_path}")
            return
        while True:
            ensure_running(runtime_process)
            if preview_process is not None:
                ensure_running(preview_process)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        terminate_process(runtime_process)
        terminate_process(preview_process)


def start_logged_process(*, name: str, command: list[str], log_path: Path) -> RunningProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningProcess(name=name, proc=proc, log_path=log_path, log_handle=log_handle)


def terminate_process(process: RunningProcess | None) -> None:
    if process is None:
        return
    if process.proc.poll() is not None:
        process.log_handle.close()
        return
    process.proc.terminate()
    try:
        process.proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.proc.kill()
        process.proc.wait(timeout=5)
    finally:
        process.log_handle.close()


def wait_for_health(url: str, process: RunningProcess, *, timeout_sec: float = 120.0) -> None:
    deadline = time.time() + timeout_sec
    last_error = "unknown"
    while time.time() < deadline:
        ensure_running(process)
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"{process.name} did not become healthy: {last_error}. See {process.log_path}")


def ensure_running(process: RunningProcess) -> None:
    code = process.proc.poll()
    if code is None:
        return
    raise RuntimeError(f"{process.name} exited with code {code}. See {process.log_path}")


def choose_available_port(host: str, preferred_port: int) -> int:
    if is_port_available(host, preferred_port):
        return preferred_port
    for port in range(preferred_port + 1, preferred_port + 50):
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"Could not find a free port near {preferred_port}")


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


if __name__ == "__main__":
    main()
