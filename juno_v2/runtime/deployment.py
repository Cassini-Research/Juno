from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from juno_v2.language.catalog import DEFAULT_SUPPORTED_LANGUAGES
from juno_v2.speech.config import DEFAULT_SPEECH_PROFILE, get_speech_front_end_profile
from juno_v2.runtime.platform import detect_platform_name, validate_context_source, validate_insertion_target

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
# Keep the old name for any callers that reference it directly.
TRUE_VALUES = _TRUE_VALUES


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return tuple(item.strip() for item in raw.split(',') if item.strip())


@dataclass(slots=True)
class ServicePathConfig:
    runtime_dir: Path = Path('.juno_v2_runtime')
    log_dir: Path = Path('.juno_v2_logs') / 'service'
    memory_dir: Path = Path('.juno_v2_memory')
    summary_json: Path | None = None
    startup_profile_json: Path | None = None
    health_json: Path | None = None
    incidents_dir: Path | None = None

    def resolved_summary_json(self) -> Path:
        return self.summary_json or (self.runtime_dir / 'summary.json')

    def resolved_startup_profile_json(self) -> Path:
        return self.startup_profile_json or (self.runtime_dir / 'startup_profile.json')

    def resolved_health_json(self) -> Path:
        return self.health_json or (self.runtime_dir / 'health.json')

    def resolved_incidents_dir(self) -> Path:
        return self.incidents_dir or (self.runtime_dir / 'incidents')

    def ensure_dirs(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_incidents_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_summary_json().parent.mkdir(parents=True, exist_ok=True)
        self.resolved_startup_profile_json().parent.mkdir(parents=True, exist_ok=True)
        self.resolved_health_json().parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ProductionServiceConfig:
    mode: str = 'live'
    preview_model_path: str = ''
    final_model_path: str = ''
    preview_backend: str = 'faster_whisper'
    final_backend: str = 'faster_whisper'
    preview_endpoint: str | None = None
    final_endpoint: str | None = None
    writer_backend: str | None = None
    writer_endpoint: str | None = None
    writer_model_path: str | None = None
    writer_max_tokens: int = 512
    writer_temperature: float = 0.0
    writer_top_p: float = 1.0
    writer_idle_unload_ttl_s: float = 300.0
    live_corrector_enabled: bool = False
    live_corrector_backend: str | None = None
    live_corrector_endpoint: str | None = None
    live_corrector_model_path: str | None = None
    live_corrector_max_tokens: int = 160
    live_corrector_temperature: float = 0.0
    live_corrector_top_p: float = 1.0
    live_corrector_residency_policy: str = "resident"
    replay_wav: Path | None = None
    language: str | None = None
    language_policy: str = 'auto_supported'
    supported_languages: tuple[str, ...] = DEFAULT_SUPPORTED_LANGUAGES
    sample_rate_hz: int = 16000
    frame_ms: int = 20
    speech_profile: str = DEFAULT_SPEECH_PROFILE
    serve_workbench: bool = False
    workbench_host: str = '127.0.0.1'
    workbench_port: int = 8765
    # Production-grade revamp Phase 2: framed JSON-RPC over Unix domain
    # socket. When set (the bundled ``run_engine.sh`` always sets it),
    # the service binds a UDS at this path in addition to the optional
    # HTTP workbench server. The Swift shell prefers the UDS surface;
    # HTTP is now developer-tool-only.
    engine_socket_path: str | None = None
    app_name: str | None = None
    window_title: str | None = None
    context_source: str = 'static'
    context_helper_command: str | None = None
    insertion_target: str = 'none'
    insertion_helper_command: str | None = None
    # MLX raises bare RuntimeError on transient device-state failures
    # (Metal queue pressure, model swap-in races between worker threads,
    # GPU stream lifecycle issues). We tolerate up to 3 restarts with
    # exponential backoff so a single transient hiccup doesn't kill the
    # engine; persistent failures still terminate normally. See
    # juno_v2/runtime/faults.py for the matching MLXTransientError
    # discrimination logic.
    max_restarts: int = 3
    restart_backoff_sec: float = 1.0
    gpu_memory_budget_mb: int | None = None
    preview_gpu_memory_mb: int = 0
    final_gpu_memory_mb: int = 0
    writer_gpu_memory_mb: int = 0
    live_corrector_gpu_memory_mb: int = 0
    preview_residency_group: str | None = None
    final_residency_group: str | None = None
    preview_residency_policy: str = "resident"
    final_residency_policy: str = "resident"
    writer_residency_policy: str = "resident"
    platform_name: str | None = None
    preview_device: str = 'auto'
    final_device: str = 'auto'
    preview_compute_type: str = 'default'
    final_compute_type: str = 'default'
    final_hf_repo_id: str | None = None
    paths: ServicePathConfig = field(default_factory=ServicePathConfig)

    @classmethod
    def from_env(cls) -> 'ProductionServiceConfig':
        paths = ServicePathConfig(
            runtime_dir=Path(os.getenv('JUNO_V2_RUNTIME_DIR', '.juno_v2_runtime')),
            log_dir=Path(os.getenv('JUNO_V2_LOG_DIR', '.juno_v2_logs/service')),
            memory_dir=Path(os.getenv('JUNO_V2_MEMORY_DIR', '.juno_v2_memory')),
            summary_json=Path(os.getenv('JUNO_V2_SUMMARY_JSON')) if os.getenv('JUNO_V2_SUMMARY_JSON') else None,
            startup_profile_json=Path(os.getenv('JUNO_V2_STARTUP_PROFILE_JSON')) if os.getenv('JUNO_V2_STARTUP_PROFILE_JSON') else None,
            health_json=Path(os.getenv('JUNO_V2_HEALTH_JSON')) if os.getenv('JUNO_V2_HEALTH_JSON') else None,
            incidents_dir=Path(os.getenv('JUNO_V2_INCIDENTS_DIR')) if os.getenv('JUNO_V2_INCIDENTS_DIR') else None,
        )
        gpu_budget_raw = os.getenv('JUNO_V2_GPU_MEMORY_BUDGET_MB')
        return cls(
            mode=os.getenv('JUNO_V2_MODE', 'live').strip() or 'live',
            preview_model_path=os.getenv('JUNO_V2_PREVIEW_MODEL_PATH', ''),
            final_model_path=os.getenv('JUNO_V2_FINAL_MODEL_PATH', ''),
            preview_backend=os.getenv('JUNO_V2_PREVIEW_BACKEND', 'faster_whisper'),
            final_backend=os.getenv('JUNO_V2_FINAL_BACKEND', 'faster_whisper'),
            preview_endpoint=os.getenv('JUNO_V2_PREVIEW_ENDPOINT') or None,
            final_endpoint=os.getenv('JUNO_V2_FINAL_ENDPOINT') or None,
            writer_backend=os.getenv('JUNO_V2_WRITER_BACKEND') or None,
            writer_endpoint=os.getenv('JUNO_V2_WRITER_ENDPOINT') or None,
            writer_model_path=os.getenv('JUNO_V2_WRITER_MODEL_PATH') or None,
            writer_max_tokens=_env_int('JUNO_V2_WRITER_MAX_TOKENS', 512),
            writer_temperature=float(os.getenv('JUNO_V2_WRITER_TEMPERATURE', '0.0')),
            writer_top_p=float(os.getenv('JUNO_V2_WRITER_TOP_P', '1.0')),
            writer_idle_unload_ttl_s=_env_float('JUNO_V2_WRITER_IDLE_UNLOAD_TTL_S', 300.0),
            live_corrector_enabled=_env_bool('JUNO_V2_LIVE_CORRECTOR_ENABLED', False),
            live_corrector_backend=os.getenv('JUNO_V2_LIVE_CORRECTOR_BACKEND') or None,
            live_corrector_endpoint=os.getenv('JUNO_V2_LIVE_CORRECTOR_ENDPOINT') or None,
            live_corrector_model_path=os.getenv('JUNO_V2_LIVE_CORRECTOR_MODEL_PATH') or None,
            live_corrector_max_tokens=_env_int('JUNO_V2_LIVE_CORRECTOR_MAX_TOKENS', 160),
            live_corrector_temperature=float(os.getenv('JUNO_V2_LIVE_CORRECTOR_TEMPERATURE', '0.0')),
            live_corrector_top_p=float(os.getenv('JUNO_V2_LIVE_CORRECTOR_TOP_P', '1.0')),
            live_corrector_residency_policy=os.getenv('JUNO_V2_LIVE_CORRECTOR_RESIDENCY_POLICY', 'resident'),
            replay_wav=Path(os.getenv('JUNO_V2_REPLAY_WAV')) if os.getenv('JUNO_V2_REPLAY_WAV') else None,
            language=os.getenv('JUNO_V2_LANGUAGE') or None,
            language_policy=os.getenv('JUNO_V2_LANGUAGE_POLICY', 'auto_supported'),
            supported_languages=_env_list('JUNO_V2_SUPPORTED_LANGUAGES', DEFAULT_SUPPORTED_LANGUAGES),
            sample_rate_hz=_env_int('JUNO_V2_SAMPLE_RATE_HZ', 16000),
            frame_ms=_env_int('JUNO_V2_FRAME_MS', 20),
            speech_profile=os.getenv('JUNO_V2_SPEECH_PROFILE', DEFAULT_SPEECH_PROFILE),
            serve_workbench=_env_bool('JUNO_V2_SERVE_WORKBENCH', False),
            workbench_host=os.getenv('JUNO_V2_WORKBENCH_HOST', '127.0.0.1'),
            workbench_port=_env_int('JUNO_V2_WORKBENCH_PORT', 8765),
            engine_socket_path=os.getenv('JUNO_ENGINE_SOCKET') or None,
            app_name=os.getenv('JUNO_V2_APP_NAME') or None,
            window_title=os.getenv('JUNO_V2_WINDOW_TITLE') or None,
            context_source=os.getenv('JUNO_V2_CONTEXT_SOURCE', 'static'),
            context_helper_command=os.getenv('JUNO_V2_CONTEXT_HELPER_COMMAND') or None,
            insertion_target=os.getenv('JUNO_V2_INSERTION_TARGET', 'none'),
            insertion_helper_command=os.getenv('JUNO_V2_INSERTION_HELPER_COMMAND') or None,
            max_restarts=_env_int('JUNO_V2_MAX_RESTARTS', 3),
            restart_backoff_sec=float(os.getenv('JUNO_V2_RESTART_BACKOFF_SEC', '1.0')),
            gpu_memory_budget_mb=(int(gpu_budget_raw) if gpu_budget_raw else None),
            preview_gpu_memory_mb=_env_int('JUNO_V2_PREVIEW_GPU_MEMORY_MB', 0),
            final_gpu_memory_mb=_env_int('JUNO_V2_FINAL_GPU_MEMORY_MB', 0),
            writer_gpu_memory_mb=_env_int('JUNO_V2_WRITER_GPU_MEMORY_MB', 0),
            live_corrector_gpu_memory_mb=_env_int('JUNO_V2_LIVE_CORRECTOR_GPU_MEMORY_MB', 0),
            preview_residency_group=os.getenv('JUNO_V2_PREVIEW_RESIDENCY_GROUP') or None,
            final_residency_group=os.getenv('JUNO_V2_FINAL_RESIDENCY_GROUP') or None,
            preview_residency_policy=os.getenv('JUNO_V2_PREVIEW_RESIDENCY_POLICY', 'resident'),
            final_residency_policy=os.getenv('JUNO_V2_FINAL_RESIDENCY_POLICY', 'resident'),
            writer_residency_policy=os.getenv('JUNO_V2_WRITER_RESIDENCY_POLICY', 'resident'),
            platform_name=os.getenv('JUNO_V2_PLATFORM_NAME') or None,
            preview_device=os.getenv('JUNO_V2_PREVIEW_DEVICE', 'auto'),
            final_device=os.getenv('JUNO_V2_FINAL_DEVICE', 'auto'),
            preview_compute_type=os.getenv('JUNO_V2_PREVIEW_COMPUTE_TYPE', 'default'),
            final_compute_type=os.getenv('JUNO_V2_FINAL_COMPUTE_TYPE', 'default'),
            final_hf_repo_id=os.getenv('JUNO_V2_FINAL_HF_REPO_ID') or None,
            paths=paths,
        )

    def validate(self) -> None:
        if self.mode not in {'live', 'replay'}:
            raise ValueError(f'Unsupported service mode: {self.mode}')
        if self.preview_backend in {'local_http_json', 'streaming_local_http_json'}:
            if not self.preview_endpoint:
                raise ValueError('preview_endpoint is required for local HTTP preview backends')
        elif not self.preview_model_path:
            raise ValueError('preview_model_path is required')
        if self.final_backend == 'local_http_json':
            if not self.final_endpoint:
                raise ValueError('final_endpoint is required for local HTTP final backends')
        elif not self.final_model_path:
            raise ValueError('final_model_path is required')
        if self.writer_backend == 'local_http_json' and not self.writer_endpoint:
            raise ValueError('writer_endpoint is required for writer_backend=local_http_json')
        if self.writer_backend == 'mlx_lm' and not self.writer_model_path:
            raise ValueError('writer_model_path is required for writer_backend=mlx_lm')
        if self.live_corrector_enabled:
            if not self.live_corrector_backend:
                raise ValueError('live_corrector_backend is required when live_corrector_enabled is true')
            if self.live_corrector_backend == 'local_http_json' and not self.live_corrector_endpoint:
                raise ValueError('JUNO_V2_LIVE_CORRECTOR_ENDPOINT is required for live_corrector_backend=local_http_json')
            if self.live_corrector_backend == 'mlx_lm' and not self.live_corrector_model_path:
                raise ValueError('live_corrector_model_path is required for live_corrector_backend=mlx_lm')
        if self.mode == 'replay' and self.replay_wav is None:
            raise ValueError('replay_wav is required in replay mode')
        if self.frame_ms <= 0:
            raise ValueError('frame_ms must be > 0')
        if self.sample_rate_hz <= 0:
            raise ValueError('sample_rate_hz must be > 0')
        get_speech_front_end_profile(self.speech_profile)
        if self.max_restarts < 0:
            raise ValueError('max_restarts must be >= 0')
        platform_name = detect_platform_name(self.platform_name)
        validate_context_source(self.context_source, platform_name=platform_name)
        validate_insertion_target(self.insertion_target, platform_name=platform_name)
        if self.insertion_target == 'macos_active_app' and self.context_source != 'macos_desktop':
            raise ValueError('macos_active_app insertion requires context_source=macos_desktop for truthful anchor sync')
        self.platform_name = platform_name
        if self.gpu_memory_budget_mb is not None and self.gpu_memory_budget_mb <= 0:
            raise ValueError('gpu_memory_budget_mb must be > 0 when set')
        for name, value in (('preview_residency_policy', self.preview_residency_policy), ('final_residency_policy', self.final_residency_policy), ('writer_residency_policy', self.writer_residency_policy), ('live_corrector_residency_policy', self.live_corrector_residency_policy)):
            if value not in {'resident', 'on_demand'}:
                raise ValueError(f'{name} must be resident or on_demand')
