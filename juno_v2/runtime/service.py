from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from juno_core_v3.context.clipboard_ring import ClipboardRingBuffer
from juno_v2.audio.sources import LiveMicrophoneAudioSource, ReplayWavAudioSource
from juno_v2.context.provider import WorkbenchContextProvider
from juno_v2.engine.factory import CanonicalEngineBuildSpec, build_canonical_engine
from juno_v2.runtime.config import WorkbenchRuntimeConfig
from juno_v2.runtime.deployment import ProductionServiceConfig
from juno_v2.runtime.faults import FaultJournal, FaultPolicy, ServiceFault
from juno_v2.runtime.health import ServiceHealthSnapshot
from juno_v2.runtime.ids import new_session_id
from juno_v2.runtime.profile import StartupProfile
from juno_v2.workbench.server import WorkbenchApp, start_http_server, stop_http_server
from juno_v2.runtime.uds_dispatch import make_uds_server
from juno_v2.runtime.uds_jsonrpc import JsonRpcServer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServiceRunResult:
    success: bool
    attempts: int
    session_id: str
    summary_path: Path | None = None
    startup_profile_path: Path | None = None
    health_path: Path | None = None
    last_fault: dict | None = None

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'attempts': self.attempts,
            'session_id': self.session_id,
            'summary_path': None if self.summary_path is None else str(self.summary_path),
            'startup_profile_path': None if self.startup_profile_path is None else str(self.startup_profile_path),
            'health_path': None if self.health_path is None else str(self.health_path),
            'last_fault': self.last_fault,
        }


class ProductionServiceRunner:
    def __init__(self, config: ProductionServiceConfig) -> None:
        self.config = config
        self.config.validate()
        self.config.paths.ensure_dirs()
        self.fault_policy = FaultPolicy(max_restarts=config.max_restarts)
        self.journal = FaultJournal(self.config.paths.resolved_incidents_dir())
        self._last_fault: ServiceFault | None = None
        # Preview-lane diagnostic state. This covers both non-fatal rebuild
        # failures (the shell can fall back to SFSpeech) and deliberately
        # slow fallback configuration such as in-process faster_whisper
        # preview. Kept as a dict (latest event wins) since this is
        # diagnostic context, not an audited fault chain.
        preview_backend = (config.preview_backend or '').strip().lower()
        self._preview_lane_status: dict | None = (
            {
                'lane': 'preview',
                'state': 'slow_fallback_configured',
                'backend': preview_backend,
                'recommended_backend': 'streaming_local_http_json',
                'reason': 'in_process_faster_whisper_preview_cannot_reliably_beat_final_asr',
            }
            if preview_backend == 'faster_whisper'
            else None
        )

    def _build_spec(self, *, session_id: str) -> CanonicalEngineBuildSpec:
        return CanonicalEngineBuildSpec(
            engine_mode=f'production_{self.config.mode}',
            session_id=session_id,
            preview_model_path=self.config.preview_model_path,
            final_model_path=self.config.final_model_path,
            preview_backend_name=self.config.preview_backend,
            final_backend_name=self.config.final_backend,
            preview_endpoint=self.config.preview_endpoint,
            final_endpoint=self.config.final_endpoint,
            writer_backend_name=self.config.writer_backend,
            writer_endpoint=self.config.writer_endpoint,
            writer_model_path=self.config.writer_model_path,
            writer_max_tokens=self.config.writer_max_tokens,
            writer_temperature=self.config.writer_temperature,
            writer_top_p=self.config.writer_top_p,
            writer_idle_unload_ttl_s=self.config.writer_idle_unload_ttl_s,
            live_corrector_enabled=self.config.live_corrector_enabled,
            live_corrector_backend_name=self.config.live_corrector_backend,
            live_corrector_endpoint=self.config.live_corrector_endpoint,
            live_corrector_model_path=self.config.live_corrector_model_path,
            live_corrector_max_tokens=self.config.live_corrector_max_tokens,
            live_corrector_temperature=self.config.live_corrector_temperature,
            live_corrector_top_p=self.config.live_corrector_top_p,
            live_corrector_residency_policy=self.config.live_corrector_residency_policy,
            language=self.config.language,
            language_policy=self.config.language_policy,
            supported_languages=self.config.supported_languages,
            sample_rate_hz=self.config.sample_rate_hz,
            frame_ms=self.config.frame_ms,
            speech_profile_name=self.config.speech_profile,
            memory_dir=self.config.paths.memory_dir,
            # Align with the embedded workbench's log_dir
            # (paths.log_dir / 'workbench') so the recorder, the
            # OneShot pipeline's `append_history_record(recorder.log_dir,
            # …)` writes, and the broker's `broker_utterance_history`
            # reads all share one root. Pre-fix the recorder wrote
            # history to `<paths.log_dir>/product_history.sqlite` but
            # the broker read from `<paths.log_dir>/workbench/product_
            # history.sqlite` — two distinct empty/populated DBs, so
            # `/api/broker/history` returned `entries: []` even after
            # successful dictations. Audio paths stored in the SQLite
            # are also relative to this root, so they only resolve to
            # the actual WAVs (under workbench/audio/...) when both
            # writer and reader use the workbench-rooted path.
            trace_log_dir=self.config.paths.log_dir / 'workbench',
            app_name=self.config.app_name,
            window_title=self.config.window_title,
            context_source=self.config.context_source,
            context_helper_command=self.config.context_helper_command,
            insertion_target=self.config.insertion_target,
            insertion_helper_command=self.config.insertion_helper_command,
            gpu_memory_budget_mb=self.config.gpu_memory_budget_mb,
            preview_gpu_memory_mb=self.config.preview_gpu_memory_mb,
            final_gpu_memory_mb=self.config.final_gpu_memory_mb,
            writer_gpu_memory_mb=self.config.writer_gpu_memory_mb,
            live_corrector_gpu_memory_mb=self.config.live_corrector_gpu_memory_mb,
            preview_residency_group=self.config.preview_residency_group,
            final_residency_group=self.config.final_residency_group,
            preview_residency_policy=self.config.preview_residency_policy,
            final_residency_policy=self.config.final_residency_policy,
            writer_residency_policy=self.config.writer_residency_policy,
            platform_name=self.config.platform_name,
            preview_device=self.config.preview_device,
            final_device=self.config.final_device,
            preview_compute_type=self.config.preview_compute_type,
            final_compute_type=self.config.final_compute_type,
            final_hf_repo_id=self.config.final_hf_repo_id,
        )

    def _shell_broker_mode(self) -> bool:
        return bool(self.config.engine_socket_path or self.config.serve_workbench)

    def _initial_warm_skip_roles(self, *, preview_worker_warms_backend: bool) -> set[str] | None:
        skip_roles: set[str] = set()
        if preview_worker_warms_backend:
            skip_roles.add('preview_asr')
        if self._shell_broker_mode():
            # The macOS shell respawns an engine that misses the early health
            # window. Keep startup to the components needed to answer health and
            # final ASR, then warm user-facing preview/live-correction/writer
            # roles behind the socket.
            skip_roles.update({'live_corrector', 'writer'})
        return skip_roles or None

    def _start_shell_background_warmup(
        self,
        *,
        workbench_app: object | None,
        lifecycle_manager: object,
        health_path: Path,
        session_id: str,
        startup: StartupProfile,
        workbench_url: str | None,
    ) -> None:
        if workbench_app is None or not self._shell_broker_mode():
            return

        if hasattr(workbench_app, "set_warm_state"):
            try:
                workbench_app.set_warm_state("warming")
            except Exception:  # noqa: BLE001
                logger.exception("workbench_set_warm_state_failed")

        def _background_warm() -> None:
            errors: list[str] = []

            # Preview comes first because this is what makes the HUD start
            # showing words immediately on the user's first utterance. The
            # rebuild future runs on the dedicated decode worker, preserving
            # MLX thread affinity.
            if hasattr(workbench_app, "_ensure_preview_decode_executor"):
                try:
                    workbench_app._ensure_preview_decode_executor()
                    rebuild_future = getattr(workbench_app, "_preview_rebuild_future", None)
                    if rebuild_future is not None:
                        rebuild_future.result(timeout=120)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("preview_decode_worker_background_prewarm_failed")
                    self._preview_lane_status = {
                        'lane': 'preview',
                        'state': 'fallback_active',
                        'fallback': 'sfspeech',
                        'reason': 'preview_decode_worker_background_prewarm_failed',
                        'exception_type': type(exc).__name__,
                        'message': str(exc),
                        'timestamp_unix': time.time(),
                    }
                    errors.append(f"preview_asr: {type(exc).__name__}: {exc}")

            warm_component = getattr(lifecycle_manager, "warm_component", None)
            if callable(warm_component):
                for role in ("live_corrector", "writer"):
                    if self._shell_background_role_is_on_demand(role):
                        logger.info("background_component_warm_skipped_on_demand role=%s", role)
                        continue
                    if not self._shell_background_role_cache_ready(role):
                        logger.info("background_component_warm_skipped_missing_cache role=%s", role)
                        continue
                    try:
                        warm_component(role)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("background_component_warm_failed role=%s", role)
                        errors.append(f"{role}: {type(exc).__name__}: {exc}")

            if hasattr(workbench_app, "set_warm_state"):
                try:
                    if errors:
                        workbench_app.set_warm_state("error", error="; ".join(errors))
                    else:
                        workbench_app.set_warm_state("ready")
                except Exception:  # noqa: BLE001
                    logger.exception("workbench_set_warm_state_finished_failed")

            try:
                lifecycle_snapshot = (
                    lifecycle_manager.snapshot()
                    if hasattr(lifecycle_manager, "snapshot")
                    else {}
                )
                status = "running" if not errors else "running_with_warmup_error"
                self._health_snapshot(
                    status=status,
                    session_id=session_id,
                    lifecycle=lifecycle_snapshot,
                    startup=startup.to_dict(),
                    workbench_url=workbench_url,
                ).write_json(health_path)
            except Exception:  # noqa: BLE001
                logger.exception("background_warm_health_snapshot_failed")

        threading.Thread(
            target=_background_warm,
            name="juno-shell-background-warmup",
            daemon=True,
        ).start()

    def _shell_background_role_is_on_demand(self, role: str) -> bool:
        if role == "writer":
            return self.config.writer_residency_policy == "on_demand"
        if role == "live_corrector":
            return self.config.live_corrector_residency_policy == "on_demand"
        return False

    def _shell_background_role_cache_ready(self, role: str) -> bool:
        if role == "live_corrector":
            backend = str(self.config.live_corrector_backend or "").strip().lower()
            model_path = str(self.config.live_corrector_model_path or "").strip()
            if not self.config.live_corrector_enabled or backend in {"", "none"}:
                return False
        elif role == "writer":
            backend = str(self.config.writer_backend or "").strip().lower()
            model_path = str(self.config.writer_model_path or "").strip()
            if backend in {"", "none"}:
                return False
        else:
            return True

        if backend != "mlx_lm" or not model_path:
            return True
        try:
            from juno_v2.demo.models import is_hf_model_cached

            return bool(is_hf_model_cached(model_path))
        except Exception:  # noqa: BLE001
            # If the cache probe itself fails, do not risk a hidden network
            # download in the startup path. Setup/install can surface the
            # repairable error explicitly.
            logger.exception("background_component_cache_probe_failed role=%s", role)
            return False

    def _start_lifecycle_idle_reaper(
        self,
        *,
        lifecycle_manager: object,
        health_path: Path,
        session_id: str,
        startup: StartupProfile,
        workbench_url: str | None,
    ) -> threading.Event | None:
        if not self._shell_broker_mode():
            return None
        reap_idle = getattr(lifecycle_manager, "reap_idle", None)
        snapshot = getattr(lifecycle_manager, "snapshot", None)
        if not callable(reap_idle) or not callable(snapshot):
            return None

        stop_event = threading.Event()

        def _loop() -> None:
            tick = 0
            while not stop_event.wait(timeout=10.0):
                tick += 1
                try:
                    reaped = list(reap_idle() or [])
                    if reaped:
                        logger.info("lifecycle_idle_reaper_unloaded roles=%s", ",".join(reaped))
                    if reaped or tick % 3 == 0:
                        self._health_snapshot(
                            status='running',
                            session_id=session_id,
                            lifecycle=snapshot(),
                            startup=startup.to_dict(),
                            workbench_url=workbench_url,
                            metadata={
                                'heartbeat_unix': time.time(),
                                'idle_reaper': {
                                    'last_reaped_roles': reaped,
                                },
                            },
                        ).write_json(health_path)
                except Exception:  # noqa: BLE001
                    logger.exception("lifecycle_idle_reaper_failed")

        threading.Thread(target=_loop, name='juno-lifecycle-idle-reaper', daemon=True).start()
        return stop_event

    def _source_iter(self):
        if self.config.mode == 'replay':
            assert self.config.replay_wav is not None
            return ReplayWavAudioSource(self.config.replay_wav).frames()
        return LiveMicrophoneAudioSource(sample_rate_hz=self.config.sample_rate_hz, frame_ms=self.config.frame_ms).frames()

    def _health_snapshot(
        self,
        *,
        status: str,
        session_id: str,
        lifecycle: dict | None = None,
        startup: dict | None = None,
        workbench_url: str | None = None,
        metadata: dict | None = None,
    ) -> ServiceHealthSnapshot:
        snapshot_metadata = dict(metadata or {})
        # Surface preview-lane fallback state in health.json without
        # introducing any user-visible banner (F1: silent fallback, neat
        # diagnostic trail). The SFSpeech fallback continues to serve the
        # user regardless.
        if self._preview_lane_status is not None:
            snapshot_metadata.setdefault('lane_status', {})['preview'] = dict(
                self._preview_lane_status
            )
        return ServiceHealthSnapshot(
            status=status,
            mode=self.config.mode,
            session_id=session_id,
            lifecycle=lifecycle or {},
            startup_profile=startup or {},
            last_fault=None if self._last_fault is None else self._last_fault.to_dict(),
            workbench_url=workbench_url,
            metadata=snapshot_metadata,
        )

    def _run_attempt(self, attempt: int, session_id: str) -> ServiceRunResult:
        startup = StartupProfile()
        summary_path = self.config.paths.resolved_summary_json()
        profile_path = self.config.paths.resolved_startup_profile_json()
        health_path = self.config.paths.resolved_health_json()
        if self.config.mode == 'live':
            from juno_v2.runtime.metal_mlx_preflight import config_requests_mlx_stack, mlx_metal_operational

            if config_requests_mlx_stack(
                preview_backend=self.config.preview_backend,
                final_backend=self.config.final_backend,
                writer_backend=self.config.writer_backend,
                live_corrector_backend=self.config.live_corrector_backend if self.config.live_corrector_enabled else None,
            ):
                ok_mlx, detail = mlx_metal_operational()
                if not ok_mlx:
                    raise RuntimeError(
                        "MLX/Metal preflight failed — live stack needs a working MLX runtime. "
                        f"Detail: {detail}. "
                        "For automated tests use JUNO_SKIP_MLX_PREFLIGHT=1; for operators, "
                        "install Apple Silicon MLX wheels or switch the preview service "
                        "backend explicitly via JUNO_V2_PREVIEW_SERVICE_BACKEND."
                    ) from None
        with startup.stage('build_engine', attempt=attempt):
            artifacts = build_canonical_engine(self._build_spec(session_id=session_id))
        preview_worker_warms_backend = bool(self.config.engine_socket_path) and self.config.preview_backend in {
            'qwen_asr',
        }
        with startup.stage('warm_backends', attempt=attempt):
            artifacts.lifecycle_manager.warm_all(
                skip_roles=self._initial_warm_skip_roles(
                    preview_worker_warms_backend=preview_worker_warms_backend
                )
            )
            lifecycle_snapshot = artifacts.lifecycle_manager.snapshot()
        # Wrap the loaded final-ASR backend with the hot-swap shim so the
        # workbench can switch models at runtime without restarting the
        # service. The wrapper implements the FinalAsrBackend protocol,
        # so existing references on `artifacts.runner` and
        # `artifacts.final_backend` keep working — they just resolve to
        # whatever backend is currently active inside the wrapper.
        from juno_v2.runtime.swappable_final import SwappableFinalBackend
        from juno_core_v3.model_registry.defaults import build_default_registry
        registry = build_default_registry()
        # Best-effort: tag the wrapper with the registry package_id whose
        # backend matches what the engine factory just constructed, so a
        # snapshot can show "you started with X". Falls through to None
        # for backends that don't map cleanly (e.g. local_http_json).
        initial_pkg = None
        for p in registry._packages.values():
            if p.manifest.slot.value != "final_asr":
                continue
            if p.manifest.backend.value == artifacts.final_backend.backend_name:
                initial_pkg = p.package_id
                break
        swap_wrapper = SwappableFinalBackend(
            inner=artifacts.final_backend,
            registry=registry,
            config_template=artifacts.runner.final_config,
            recorder=artifacts.recorder,
            gpu_budget_mb=self.config.gpu_memory_budget_mb,
            initial_package_id=initial_pkg,
        )
        artifacts.final_backend = swap_wrapper
        artifacts.runner.final_backend = swap_wrapper
        # Shared clipboard ring: a single bounded history used by both
        # the streaming session and the workbench one-shot pipeline.
        # Wiring it here (rather than letting each component create its
        # own) means a clipboard entry pushed from the Mac shell
        # insertion_committed handler is immediately visible to the
        # next streaming utterance's context bundle, and vice versa.
        shared_clipboard_ring = ClipboardRingBuffer()
        artifacts.runner.clipboard_ring = shared_clipboard_ring
        workbench_app = None
        if self.config.engine_socket_path or self.config.serve_workbench:
            with startup.stage('start_embedded_workbench', attempt=attempt):
                if getattr(artifacts.runner.context_provider, '__class__', type(None)).__name__ == 'WorkbenchContextProvider':
                    artifacts.runner.context_provider = WorkbenchContextProvider(artifacts.store)
                # Reuse the engine's already-loaded final ASR backend for
                # the one-shot dictation HTTP endpoints. Without this the
                # workbench would fall back to env-var resolution (which
                # would construct a *second* model) or, worse, to the
                # UnavailableTranscriber default.
                from juno_core_v3.dictation import FinalBackendTranscriber
                transcriber = None
                if artifacts.final_backend is not None:
                    transcriber = FinalBackendTranscriber(
                        backend=artifacts.final_backend,
                        language=self.config.language,
                    )
                # Hand the workbench the same context provider, writer
                # service, writer backend, language planner, and bias
                # engine the streaming DictationSessionRunner uses. That
                # is what makes the one-shot HTTP path (Mac shell /
                # compatible clients) go through the full pipeline rather
                # than a bare ASR decode.
                runner = artifacts.runner
                workbench_app = WorkbenchApp(
                    WorkbenchRuntimeConfig(
                        host=self.config.workbench_host,
                        port=self.config.workbench_port,
                        log_dir=self.config.paths.log_dir / 'workbench',
                        # When embedded in the production service, expose the canonical
                        # runtime directory so broker setup/status can read the live
                        # health snapshot instead of the standalone demo profile defaults.
                        runtime_dir=self.config.paths.runtime_dir,
                    ),
                    session_id=session_id,
                    recorder=artifacts.recorder,
                    store=artifacts.store,
                    commit=artifacts.controller,
                    transcriber=transcriber,
                    memory=artifacts.memory_store,
                    context_provider=runner.context_provider,
                    writer_service=artifacts.writer_service,
                    live_corrector_service=getattr(artifacts, "live_corrector_service", None),
                    writer_backend=getattr(artifacts.writer_service, 'backend', None),
                    language_planner=runner.language_planner,
                    bias_engine=runner.bias_engine,
                    clipboard_ring=shared_clipboard_ring,
                    context_plane=runner.context_plane,
                    final_swap=swap_wrapper,
                    juno_seed_runtime=getattr(runner, 'juno_seed_runtime', None),
                )
                from juno_v2.runtime.local_broker_token import (
                    regenerate_local_broker_token,
                )
                # Mark this app instance as the canonical production engine.
                # The Swift shell only attaches when ``runtime_role`` matches
                # ``PRODUCTION_RUNTIME_ROLE`` — without this the shell would
                # treat a standalone ``python -m juno_v2.workbench.server``
                # on the same port as live engine and silently misroute.
                from juno_v2.runtime.shell_engine_contract import (
                    PRODUCTION_RUNTIME_ROLE,
                )

                # The production UDS contract uses a per-engine-spawn
                # local secret, not a long-lived install token. Generate it
                # immediately before the socket comes up so the Swift shell
                # reads the same file that this process verifies.
                if self.config.engine_socket_path:
                    regenerate_local_broker_token()
                workbench_app.runtime_role = PRODUCTION_RUNTIME_ROLE
                workbench_app.deployment_profile = {
                    "preview_backend": self.config.preview_backend,
                    "final_backend": self.config.final_backend,
                    "writer_backend": self.config.writer_backend,
                    "live_corrector_backend": self.config.live_corrector_backend,
                    "writer_residency_policy": self.config.writer_residency_policy,
                    "live_corrector_residency_policy": self.config.live_corrector_residency_policy,
                    "preview_model_path": self.config.preview_model_path,
                    "final_model_path": self.config.final_model_path,
                    "writer_model_path": self.config.writer_model_path,
                    "live_corrector_model_path": self.config.live_corrector_model_path,
                }
                workbench_app.lifecycle_manager = artifacts.lifecycle_manager
                # Hand the workbench a reference to the streaming dictation
                # runner so the broker live-caption setter can mutate
                # ``preview_decode_enabled`` for the next utterance, and apply
                # the persisted setting now in case the user toggled it off
                # before this run started. Final lane / writer / one-shot
                # ingest are unaffected — the gate lives strictly inside the
                # streaming preview decode wrapper.
                #
                # ``getattr`` guards: tests stub WorkbenchApp with a bare
                # ``types.SimpleNamespace`` to skip the real init, so the
                # ``_settings`` / ``dictation_runner`` attributes may not
                # exist there. Production WorkbenchApp always provides them.
                try:
                    workbench_app.dictation_runner = artifacts.runner
                except Exception:  # noqa: BLE001 — never break startup over a stub
                    pass
                _live_caption_enabled = bool(
                    getattr(workbench_app, "_settings", {}).get("live_caption_enabled", False)
                )
                artifacts.runner.preview_decode_enabled = _live_caption_enabled
        active_workbench_url: str | None = None
        http_server = None
        if workbench_app is not None and self.config.serve_workbench:
            with startup.stage('start_http_workbench', attempt=attempt):
                http_server, _http_thread = start_http_server(workbench_app)
                active_workbench_url = (
                    f"http://{self.config.workbench_host}:{self.config.workbench_port}"
                )
        # Production transport: framed JSON-RPC over a Unix domain socket.
        # The Swift shell consumes broker calls only over UDS; the dev
        # workbench HTTP server is available only when explicitly requested.
        # ``--engine-socket`` overrides the default path.
        engine_uds: JsonRpcServer | None = None
        if workbench_app is not None and self.config.engine_socket_path:
            with startup.stage('start_engine_uds', attempt=attempt):
                from pathlib import Path as _Path
                engine_uds = make_uds_server(
                    _Path(self.config.engine_socket_path),
                    workbench_app,
                )
                engine_uds.start()
        idle_reaper_stop: threading.Event | None = None
        try:
            self._health_snapshot(status='starting', session_id=session_id, lifecycle=lifecycle_snapshot, startup=startup.to_dict(), workbench_url=active_workbench_url).write_json(health_path)
            self._start_shell_background_warmup(
                workbench_app=workbench_app,
                lifecycle_manager=artifacts.lifecycle_manager,
                health_path=health_path,
                session_id=session_id,
                startup=startup,
                workbench_url=active_workbench_url,
            )
            idle_reaper_stop = self._start_lifecycle_idle_reaper(
                lifecycle_manager=artifacts.lifecycle_manager,
                health_path=health_path,
                session_id=session_id,
                startup=startup,
                workbench_url=active_workbench_url,
            )
            if self.config.serve_workbench or self.config.engine_socket_path:
                # Shell-driven mode (Juno's macOS deployment, both HTTP
                # and UDS transports). The Swift shell submits per-
                # utterance WAVs via the broker — either over HTTP
                # ``/api/broker/dictation/ingest_wav`` (when --serve-
                # workbench is on) or framed JSON-RPC over the Unix
                # domain socket created above. In both cases the
                # streaming runner.run() path is dead code:
                #
                #   1. It opens a *second* live mic via
                #      LiveMicrophoneAudioSource, competing with the
                #      Swift-side capture for the input device.
                #   2. The streaming preview/final stages run on
                #      StageExecutor worker threads. PR #36's
                #      mlx_decode_guard fix only covers backends that
                #      *enter* the guard — the streaming runner has
                #      additional MLX call sites (warm-time imports,
                #      stream-state allocation, VAD frontend) that fire on
                #      a worker thread that never had
                #      `mx.new_thread_local_stream(...)` set up. The
                #      shell-broker path uses the guard at every decode and
                #      so works correctly.
                #   3. The streaming session's transcripts are not
                #      consumed anywhere — Juno reads only the broker
                #      routes.
                #
                # We keep the workbench app's HTTP and/or UDS thread
                # running and block the main thread on a signal-driven
                # event so launchd / the macOS shell can SIGTERM us
                # cleanly.
                running_profile_payload = startup.to_dict()
                profile_path.write_text(json.dumps(running_profile_payload, ensure_ascii=False, indent=2), encoding='utf-8')
                self._health_snapshot(status='running', session_id=session_id, lifecycle=lifecycle_snapshot, startup=running_profile_payload, workbench_url=active_workbench_url).write_json(health_path)
                with startup.stage('serve_until_signal', attempt=attempt):
                    self._block_until_signal()
                profile_payload = startup.to_dict()
                profile_path.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding='utf-8')
                self._health_snapshot(
                    status='ok',
                    session_id=session_id,
                    lifecycle=lifecycle_snapshot,
                    startup=profile_payload,
                    workbench_url=active_workbench_url,
                    metadata={
                        'mode': 'workbench_only',
                        'startup_profile_json': str(profile_path),
                    },
                ).write_json(health_path)
                return ServiceRunResult(
                    success=True,
                    attempts=attempt,
                    session_id=session_id,
                    summary_path=None,
                    startup_profile_path=profile_path,
                    health_path=health_path,
                )
            with startup.stage('prepare_source', attempt=attempt, mode=self.config.mode):
                frames = self._source_iter()
            self._health_snapshot(status='running', session_id=session_id, lifecycle=lifecycle_snapshot, startup=startup.to_dict(), workbench_url=None).write_json(health_path)
            with startup.stage('run_session', attempt=attempt):
                summary = artifacts.runner.run(frames, allow_interrupt=(self.config.mode == 'live'))
            summary_payload = summary.to_dict()
            summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')
            startup.add_stage('write_summary', 0.0, path=str(summary_path))
            profile_payload = startup.to_dict()
            profile_path.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding='utf-8')
            self._health_snapshot(
                status='ok',
                session_id=session_id,
                lifecycle=summary_payload.get('metadata', {}).get('lifecycle', {}),
                startup=profile_payload,
                workbench_url=None,
                metadata={
                    'runtime_truth': summary_payload.get('metadata', {}).get('runtime_truth', {}),
                    'summary_json': str(summary_path),
                    'startup_profile_json': str(profile_path),
                },
            ).write_json(health_path)
            return ServiceRunResult(
                success=True,
                attempts=attempt,
                session_id=session_id,
                summary_path=summary_path,
                startup_profile_path=profile_path,
                health_path=health_path,
            )
        finally:
            if idle_reaper_stop is not None:
                idle_reaper_stop.set()
            if http_server is not None and workbench_app is not None:
                try:
                    stop_http_server(workbench_app, http_server)
                except Exception:
                    pass
            if engine_uds is not None:
                try:
                    engine_uds.stop()
                except Exception:
                    pass

    def _block_until_signal(self) -> None:
        """Block the main thread until SIGINT or SIGTERM arrives.

        Used by the workbench-only deployment (Juno's macOS shell): broker
        transports own all audio ingress, so the main thread just needs to
        stay alive until launchd / the parent process sends a termination
        signal. Using a ``threading.Event`` rather than
        ``signal.pause()`` lets us wake periodically (a one-second
        timeout) so a hung signal-handler thread can't strand the
        process — the event check is the source of truth for shutdown.
        """
        import signal
        import threading

        stop_event = threading.Event()

        def _handler(_signum, _frame):  # noqa: ANN001
            stop_event.set()

        old_int = signal.signal(signal.SIGINT, _handler)
        old_term = signal.signal(signal.SIGTERM, _handler)
        try:
            while not stop_event.is_set():
                stop_event.wait(timeout=1.0)
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)

    def run(self) -> ServiceRunResult:
        session_id = new_session_id(prefix=f'service_{self.config.mode}')
        attempt = 1
        while True:
            try:
                return self._run_attempt(attempt, session_id)
            except Exception as exc:  # pragma: no cover
                self._last_fault = self.fault_policy.classify(exc, stage='service_run', attempt=attempt)
                self.journal.record(self._last_fault)
                self._health_snapshot(status='error', session_id=session_id, startup={}, lifecycle={}, metadata={'attempt': attempt}).write_json(self.config.paths.resolved_health_json())
                if self.fault_policy.should_retry(self._last_fault):
                    # Exponential backoff (capped at 30s) so a single MLX
                    # hiccup doesn't burn through all max_restarts in
                    # microseconds. Floor at the configured base so the
                    # first retry still respects ``restart_backoff_sec``.
                    backoff = max(
                        float(self.config.restart_backoff_sec),
                        min(float(2 ** attempt), 30.0),
                    )
                    time.sleep(backoff)
                    attempt += 1
                    continue
                return ServiceRunResult(
                    success=False,
                    attempts=attempt,
                    session_id=session_id,
                    health_path=self.config.paths.resolved_health_json(),
                    last_fault=self._last_fault.to_dict(),
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Juno v2 production service')
    parser.add_argument('--mode', choices=['live', 'replay'], default=None)
    parser.add_argument('--preview-model-path', default=None)
    parser.add_argument('--final-model-path', default=None)
    parser.add_argument('--preview-backend', default=None)
    parser.add_argument('--final-backend', default=None)
    parser.add_argument('--preview-endpoint', default=None)
    parser.add_argument('--final-endpoint', default=None)
    parser.add_argument('--writer-backend', default=None)
    parser.add_argument('--writer-endpoint', default=None)
    parser.add_argument('--writer-model-path', default=None)
    parser.add_argument('--writer-max-tokens', type=int, default=None)
    parser.add_argument('--writer-temperature', type=float, default=None)
    parser.add_argument('--writer-top-p', type=float, default=None)
    parser.add_argument('--writer-idle-unload-ttl-s', type=float, default=None)
    parser.add_argument('--live-corrector-enabled', action='store_true', default=None)
    parser.add_argument('--live-corrector-backend', default=None)
    parser.add_argument('--live-corrector-endpoint', default=None)
    parser.add_argument('--live-corrector-model-path', default=None)
    parser.add_argument('--live-corrector-max-tokens', type=int, default=None)
    parser.add_argument('--live-corrector-temperature', type=float, default=None)
    parser.add_argument('--live-corrector-top-p', type=float, default=None)
    parser.add_argument('--live-corrector-residency-policy', default=None)
    parser.add_argument('--replay-wav', default=None)
    parser.add_argument('--language', default=None)
    parser.add_argument('--speech-profile', default=None)
    parser.add_argument('--language-policy', default=None)
    parser.add_argument('--supported-languages', default=None)
    parser.add_argument(
        '--engine-socket',
        default=None,
        help=(
            "Path to the Unix domain socket the engine binds for the "
            "Juno macOS shell (production transport). Default: "
            "<JUNO_APP_SUPPORT_DIR>/runtime/engine.sock or "
            "~/Library/Application Support/<bundle-id>/runtime/engine.sock."
        ),
    )
    parser.add_argument('--runtime-dir', default=None)
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--memory-dir', default=None)
    parser.add_argument('--summary-json', default=None)
    parser.add_argument('--startup-profile-json', default=None)
    parser.add_argument('--health-json', default=None)
    parser.add_argument('--max-restarts', type=int, default=None)
    parser.add_argument('--restart-backoff-sec', type=float, default=None)
    parser.add_argument('--app-name', default=None)
    parser.add_argument('--window-title', default=None)
    parser.add_argument('--context-source', default=None)
    parser.add_argument('--context-helper-command', default=None)
    parser.add_argument('--insertion-target', default=None)
    parser.add_argument('--insertion-helper-command', default=None)
    parser.add_argument('--gpu-memory-budget-mb', type=int, default=None)
    parser.add_argument('--preview-gpu-memory-mb', type=int, default=None)
    parser.add_argument('--final-gpu-memory-mb', type=int, default=None)
    parser.add_argument('--writer-gpu-memory-mb', type=int, default=None)
    parser.add_argument('--live-corrector-gpu-memory-mb', type=int, default=None)
    parser.add_argument('--preview-residency-group', default=None)
    parser.add_argument('--final-residency-group', default=None)
    parser.add_argument('--preview-residency-policy', default=None)
    parser.add_argument('--final-residency-policy', default=None)
    parser.add_argument('--writer-residency-policy', default=None)
    parser.add_argument('--platform-name', default=None)
    parser.add_argument('--preview-device', default=None)
    parser.add_argument('--final-device', default=None)
    parser.add_argument('--preview-compute-type', default=None)
    parser.add_argument('--final-compute-type', default=None)
    parser.add_argument('--final-hf-repo-id', default=None)
    parser.add_argument(
        '--serve-workbench',
        action='store_true',
        default=False,
        help=(
            "Start the embedded HTTP workbench/broker server on "
            "--workbench-host:--workbench-port. Equivalent to setting "
            "JUNO_V2_SERVE_WORKBENCH=1. Required for the dev demo "
            "launcher path; production (UDS) does not need this."
        ),
    )
    parser.add_argument('--workbench-host', default=None)
    parser.add_argument('--workbench-port', type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ProductionServiceConfig.from_env()
    if args.final_hf_repo_id is not None:
        config.final_hf_repo_id = args.final_hf_repo_id
    for name in (
        'mode',
        'preview_model_path',
        'final_model_path',
        'preview_backend',
        'final_backend',
        'preview_endpoint',
        'final_endpoint',
        'writer_backend',
        'writer_endpoint',
        'writer_model_path',
        'live_corrector_backend',
        'live_corrector_endpoint',
        'live_corrector_model_path',
        'live_corrector_residency_policy',
        'language',
        'language_policy',
        'speech_profile',
        'context_source',
        'context_helper_command',
        'insertion_target',
        'insertion_helper_command',
        'preview_residency_group',
        'final_residency_group',
        'preview_residency_policy',
        'final_residency_policy',
        'writer_residency_policy',
        'platform_name',
        'preview_device',
        'final_device',
        'preview_compute_type',
        'final_compute_type',
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.live_corrector_enabled is not None:
        config.live_corrector_enabled = bool(args.live_corrector_enabled)
    if args.writer_idle_unload_ttl_s is not None:
        config.writer_idle_unload_ttl_s = float(args.writer_idle_unload_ttl_s)
    if args.live_corrector_max_tokens is not None:
        config.live_corrector_max_tokens = int(args.live_corrector_max_tokens)
    if args.live_corrector_temperature is not None:
        config.live_corrector_temperature = float(args.live_corrector_temperature)
    if args.live_corrector_top_p is not None:
        config.live_corrector_top_p = float(args.live_corrector_top_p)
    if args.replay_wav is not None:
        config.replay_wav = Path(args.replay_wav)
    if args.supported_languages is not None:
        config.supported_languages = tuple(item.strip() for item in args.supported_languages.split(',') if item.strip())
    if args.engine_socket is not None:
        config.engine_socket_path = args.engine_socket
    if args.runtime_dir is not None:
        config.paths.runtime_dir = Path(args.runtime_dir)
    if args.log_dir is not None:
        config.paths.log_dir = Path(args.log_dir)
    if args.memory_dir is not None:
        config.paths.memory_dir = Path(args.memory_dir)
    if args.summary_json is not None:
        config.paths.summary_json = Path(args.summary_json)
    if args.startup_profile_json is not None:
        config.paths.startup_profile_json = Path(args.startup_profile_json)
    if args.health_json is not None:
        config.paths.health_json = Path(args.health_json)
    if args.max_restarts is not None:
        config.max_restarts = args.max_restarts
    if args.restart_backoff_sec is not None:
        config.restart_backoff_sec = args.restart_backoff_sec
    if args.app_name is not None:
        config.app_name = args.app_name
    if args.window_title is not None:
        config.window_title = args.window_title
    if args.gpu_memory_budget_mb is not None:
        config.gpu_memory_budget_mb = args.gpu_memory_budget_mb
    if args.preview_gpu_memory_mb is not None:
        config.preview_gpu_memory_mb = args.preview_gpu_memory_mb
    if args.final_gpu_memory_mb is not None:
        config.final_gpu_memory_mb = args.final_gpu_memory_mb
    if args.writer_max_tokens is not None:
        config.writer_max_tokens = args.writer_max_tokens
    if args.writer_temperature is not None:
        config.writer_temperature = args.writer_temperature
    if args.writer_top_p is not None:
        config.writer_top_p = args.writer_top_p
    if args.writer_gpu_memory_mb is not None:
        config.writer_gpu_memory_mb = args.writer_gpu_memory_mb
    if args.live_corrector_gpu_memory_mb is not None:
        config.live_corrector_gpu_memory_mb = args.live_corrector_gpu_memory_mb
    if args.serve_workbench:
        config.serve_workbench = True
    if args.workbench_host is not None:
        config.workbench_host = args.workbench_host
    if args.workbench_port is not None:
        config.workbench_port = args.workbench_port

    # If every model the runtime needs is already in the local HF cache,
    # set HF_HUB_OFFLINE=1 so model loads don't burn a 10 s timeout per
    # file revalidating against the hub. Without this, offline boot can
    # stall for 40+ s on a fully-installed app. Deliberate installs
    # temporarily clear the flag (see broker_setup_install).
    from juno_v2.runtime.offline_mode import (
        enable_offline_mode_if_cache_complete,
        required_hf_repo_ids_from_config,
    )
    enable_offline_mode_if_cache_complete(required_hf_repo_ids_from_config(config))

    result = ProductionServiceRunner(config).run()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if not result.success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
