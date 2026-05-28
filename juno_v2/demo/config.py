from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from juno_v2.language.catalog import DEFAULT_SUPPORTED_LANGUAGES
from juno_v2.speech.config import DEFAULT_SPEECH_PROFILE


DEFAULT_DEMO_PROFILE = "best_local"


@dataclass(frozen=True, slots=True)
class DemoProfile:
    name: str
    description: str
    preview_repo_id: str
    final_repo_id: str
    supported_languages: tuple[str, ...]
    language: str | None = None
    language_policy: str = "auto_supported"
    preview_device: str = "cpu"
    final_device: str = "cpu"
    preview_compute_type: str = "int8"
    final_compute_type: str = "int8"
    preview_backend: str = "streaming_local_http_json"
    preview_service_backend: str = "mlx_whisper"
    final_backend: str = "faster_whisper"
    writer_backend: str | None = None
    writer_model_path: str | None = None
    writer_max_tokens: int = 512
    writer_temperature: float = 0.0
    writer_top_p: float = 1.0
    live_corrector_backend: str | None = None
    live_corrector_model_path: str | None = None
    live_corrector_max_tokens: int = 160
    live_corrector_temperature: float = 0.0
    live_corrector_top_p: float = 1.0
    preview_endpoint: str | None = None
    final_endpoint: str | None = None
    writer_endpoint: str | None = None
    speech_profile: str = DEFAULT_SPEECH_PROFILE
    profile_class: str = "standard"
    target_machine: str = "apple_silicon_laptop"
    preview_lane_status: str = "default"
    final_lane_status: str = "default"
    writer_lane_status: str = "default"
    gpu_memory_budget_mb: int | None = None
    preview_gpu_memory_mb: int = 0
    final_gpu_memory_mb: int = 0
    writer_gpu_memory_mb: int = 0
    live_corrector_gpu_memory_mb: int = 0
    preview_residency_policy: str = "resident"
    final_residency_policy: str = "resident"
    writer_residency_policy: str = "resident"
    live_corrector_residency_policy: str = "resident"
    notes: tuple[str, ...] = ()


DEMO_PROFILES: dict[str, DemoProfile] = {
    "best_local": DemoProfile(
        name="best_local",
        description="Best local product path on an Apple Silicon Mac.",
        preview_repo_id="mlx-community/whisper-large-v3-turbo",
        final_repo_id="mlx-community/whisper-large-v3-turbo",
        supported_languages=DEFAULT_SUPPORTED_LANGUAGES,
        language=None,
        language_policy="auto_supported",
        preview_backend="streaming_local_http_json",
        preview_service_backend="mlx_whisper",
        final_backend="mlx_whisper",
        writer_backend="mlx_lm",
        writer_model_path="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        writer_max_tokens=512,
        writer_temperature=0.0,
        writer_top_p=1.0,
        live_corrector_backend=None,
        live_corrector_model_path=None,
        live_corrector_max_tokens=160,
        live_corrector_temperature=0.0,
        live_corrector_top_p=1.0,
        speech_profile="standard",
        preview_lane_status="default",
        final_lane_status="default",
        writer_lane_status="default",
        gpu_memory_budget_mb=12000,
        preview_gpu_memory_mb=4200,
        final_gpu_memory_mb=4200,
        writer_gpu_memory_mb=2600,
        live_corrector_gpu_memory_mb=0,
        preview_residency_policy="resident",
        final_residency_policy="resident",
        writer_residency_policy="resident",
        live_corrector_residency_policy="resident",
        notes=(
            "This is the recommended local source path.",
            "Preview uses the local streaming preview service with MLX Whisper large-v3-turbo checkpoints.",
            "Final uses the same MLX Whisper large-v3-turbo ASR family before writer/action processing.",
            "The writer lane runs locally via MLX LM with a Qwen3 4B instruct model on Apple Silicon.",
            "Use fixed or pair language policies via runtime flags when testing a narrower market path.",
        ),
    ),
}

_PROFILE_ALIASES = {
    'global_balanced': 'best_local',
    'default': 'best_local',
    'english_fast': 'best_local',
}


@dataclass(slots=True)
class DemoPaths:
    root_dir: Path = Path(".juno_v2_demo")
    config_json: Path | None = None
    models_dir: Path | None = None
    logs_dir: Path | None = None
    runtime_dir: Path | None = None
    memory_dir: Path | None = None

    def resolved_config_json(self) -> Path:
        return self.config_json or (self.root_dir / "config.json")

    def resolved_models_dir(self) -> Path:
        return self.models_dir or (self.root_dir / "models")

    def resolved_logs_dir(self) -> Path:
        return self.logs_dir or (self.root_dir / "logs")

    def resolved_runtime_dir(self) -> Path:
        return self.runtime_dir or (self.root_dir / "runtime")

    def resolved_memory_dir(self) -> Path:
        return self.memory_dir or (self.root_dir / "memory")

    def resolved_preview_model_dir(self) -> Path:
        return self.resolved_models_dir() / "preview"

    def resolved_final_model_dir(self) -> Path:
        return self.resolved_models_dir() / "final"

    def resolved_preview_service_log(self) -> Path:
        return self.resolved_logs_dir() / "preview_service.log"

    def resolved_runtime_service_log(self) -> Path:
        return self.resolved_logs_dir() / "runtime_service.log"

    def ensure_dirs(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_models_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_logs_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_runtime_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_memory_dir().mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DemoConfig:
    profile_name: str = DEFAULT_DEMO_PROFILE
    preview_repo_id: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_repo_id
    final_repo_id: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].final_repo_id
    preview_model_path: Path = field(default_factory=lambda: Path(".juno_v2_demo/models/preview"))
    final_model_path: Path = field(default_factory=lambda: Path(".juno_v2_demo/models/final"))
    supported_languages: tuple[str, ...] = tuple(DEMO_PROFILES[DEFAULT_DEMO_PROFILE].supported_languages)
    language: str | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].language
    language_policy: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].language_policy
    preview_device: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_device
    final_device: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].final_device
    preview_compute_type: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_compute_type
    final_compute_type: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].final_compute_type
    preview_backend: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_backend
    preview_service_backend: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_service_backend
    final_backend: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].final_backend
    writer_backend: str | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_backend
    writer_model_path: str | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_model_path
    writer_max_tokens: int = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_max_tokens
    writer_temperature: float = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_temperature
    writer_top_p: float = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_top_p
    live_corrector_backend: str | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].live_corrector_backend
    live_corrector_model_path: str | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].live_corrector_model_path
    live_corrector_max_tokens: int = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].live_corrector_max_tokens
    live_corrector_temperature: float = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].live_corrector_temperature
    live_corrector_top_p: float = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].live_corrector_top_p
    preview_endpoint: str | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_endpoint
    final_endpoint: str | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].final_endpoint
    writer_endpoint: str | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_endpoint
    profile_class: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].profile_class
    target_machine: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].target_machine
    notes: tuple[str, ...] = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].notes
    speech_profile: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].speech_profile
    preview_lane_status: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_lane_status
    final_lane_status: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].final_lane_status
    writer_lane_status: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_lane_status
    gpu_memory_budget_mb: int | None = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].gpu_memory_budget_mb
    preview_gpu_memory_mb: int = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_gpu_memory_mb
    final_gpu_memory_mb: int = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].final_gpu_memory_mb
    writer_gpu_memory_mb: int = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_gpu_memory_mb
    live_corrector_gpu_memory_mb: int = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].live_corrector_gpu_memory_mb
    preview_residency_policy: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].preview_residency_policy
    final_residency_policy: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].final_residency_policy
    writer_residency_policy: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].writer_residency_policy
    live_corrector_residency_policy: str = DEMO_PROFILES[DEFAULT_DEMO_PROFILE].live_corrector_residency_policy

    @classmethod
    def from_profile(cls, profile_name: str, *, paths: DemoPaths) -> "DemoConfig":
        profile = get_demo_profile(profile_name)
        return cls(
            profile_name=profile.name,
            preview_repo_id=profile.preview_repo_id,
            final_repo_id=profile.final_repo_id,
            preview_model_path=paths.resolved_preview_model_dir(),
            final_model_path=paths.resolved_final_model_dir(),
            supported_languages=profile.supported_languages,
            language=profile.language,
            language_policy=profile.language_policy,
            preview_device=profile.preview_device,
            final_device=profile.final_device,
            preview_compute_type=profile.preview_compute_type,
            final_compute_type=profile.final_compute_type,
            preview_backend=profile.preview_backend,
            preview_service_backend=profile.preview_service_backend,
            final_backend=profile.final_backend,
            writer_backend=profile.writer_backend,
            writer_model_path=profile.writer_model_path,
            writer_max_tokens=profile.writer_max_tokens,
            writer_temperature=profile.writer_temperature,
            writer_top_p=profile.writer_top_p,
            live_corrector_backend=profile.live_corrector_backend,
            live_corrector_model_path=profile.live_corrector_model_path,
            live_corrector_max_tokens=profile.live_corrector_max_tokens,
            live_corrector_temperature=profile.live_corrector_temperature,
            live_corrector_top_p=profile.live_corrector_top_p,
            preview_endpoint=profile.preview_endpoint,
            final_endpoint=profile.final_endpoint,
            writer_endpoint=profile.writer_endpoint,
            profile_class=profile.profile_class,
            target_machine=profile.target_machine,
            notes=profile.notes,
            speech_profile=profile.speech_profile,
            preview_lane_status=profile.preview_lane_status,
            final_lane_status=profile.final_lane_status,
            writer_lane_status=profile.writer_lane_status,
            gpu_memory_budget_mb=profile.gpu_memory_budget_mb,
            preview_gpu_memory_mb=profile.preview_gpu_memory_mb,
            final_gpu_memory_mb=profile.final_gpu_memory_mb,
            writer_gpu_memory_mb=profile.writer_gpu_memory_mb,
            live_corrector_gpu_memory_mb=profile.live_corrector_gpu_memory_mb,
            preview_residency_policy=profile.preview_residency_policy,
            final_residency_policy=profile.final_residency_policy,
            writer_residency_policy=profile.writer_residency_policy,
            live_corrector_residency_policy=profile.live_corrector_residency_policy,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "DemoConfig":
        profile_name = _PROFILE_ALIASES.get(str(payload.get("profile_name", DEFAULT_DEMO_PROFILE)), str(payload.get("profile_name", DEFAULT_DEMO_PROFILE)))
        fallback = DEMO_PROFILES.get(profile_name, DEMO_PROFILES[DEFAULT_DEMO_PROFILE])
        return cls(
            profile_name=profile_name,
            preview_repo_id=str(payload.get("preview_repo_id", fallback.preview_repo_id)),
            final_repo_id=str(payload.get("final_repo_id", fallback.final_repo_id)),
            preview_model_path=Path(payload.get("preview_model_path", ".juno_v2_demo/models/preview")),
            final_model_path=Path(payload.get("final_model_path", ".juno_v2_demo/models/final")),
            supported_languages=tuple(payload.get("supported_languages", fallback.supported_languages)),
            language=payload.get("language", fallback.language),
            language_policy=str(payload.get("language_policy", fallback.language_policy)),
            preview_device=str(payload.get("preview_device", fallback.preview_device)),
            final_device=str(payload.get("final_device", fallback.final_device)),
            preview_compute_type=str(payload.get("preview_compute_type", fallback.preview_compute_type)),
            final_compute_type=str(payload.get("final_compute_type", fallback.final_compute_type)),
            preview_backend=str(payload.get("preview_backend", fallback.preview_backend)),
            preview_service_backend=str(payload.get("preview_service_backend", fallback.preview_service_backend)),
            final_backend=str(payload.get("final_backend", fallback.final_backend)),
            writer_backend=payload.get("writer_backend", fallback.writer_backend),
            writer_model_path=payload.get("writer_model_path", fallback.writer_model_path),
            writer_max_tokens=int(payload.get("writer_max_tokens", fallback.writer_max_tokens)),
            writer_temperature=float(payload.get("writer_temperature", fallback.writer_temperature)),
            writer_top_p=float(payload.get("writer_top_p", fallback.writer_top_p)),
            live_corrector_backend=payload.get("live_corrector_backend", fallback.live_corrector_backend),
            live_corrector_model_path=payload.get("live_corrector_model_path", fallback.live_corrector_model_path),
            live_corrector_max_tokens=int(payload.get("live_corrector_max_tokens", fallback.live_corrector_max_tokens)),
            live_corrector_temperature=float(payload.get("live_corrector_temperature", fallback.live_corrector_temperature)),
            live_corrector_top_p=float(payload.get("live_corrector_top_p", fallback.live_corrector_top_p)),
            preview_endpoint=payload.get("preview_endpoint", fallback.preview_endpoint),
            final_endpoint=payload.get("final_endpoint", fallback.final_endpoint),
            writer_endpoint=payload.get("writer_endpoint", fallback.writer_endpoint),
            profile_class=str(payload.get("profile_class", fallback.profile_class)),
            target_machine=str(payload.get("target_machine", fallback.target_machine)),
            notes=tuple(payload.get("notes", fallback.notes)),
            speech_profile=str(payload.get("speech_profile", fallback.speech_profile)),
            preview_lane_status=str(payload.get("preview_lane_status", fallback.preview_lane_status)),
            final_lane_status=str(payload.get("final_lane_status", fallback.final_lane_status)),
            writer_lane_status=str(payload.get("writer_lane_status", fallback.writer_lane_status)),
            gpu_memory_budget_mb=payload.get("gpu_memory_budget_mb", fallback.gpu_memory_budget_mb),
            preview_gpu_memory_mb=int(payload.get("preview_gpu_memory_mb", fallback.preview_gpu_memory_mb)),
            final_gpu_memory_mb=int(payload.get("final_gpu_memory_mb", fallback.final_gpu_memory_mb)),
            writer_gpu_memory_mb=int(payload.get("writer_gpu_memory_mb", fallback.writer_gpu_memory_mb)),
            live_corrector_gpu_memory_mb=int(payload.get("live_corrector_gpu_memory_mb", fallback.live_corrector_gpu_memory_mb)),
            preview_residency_policy=str(payload.get("preview_residency_policy", fallback.preview_residency_policy)),
            final_residency_policy=str(payload.get("final_residency_policy", fallback.final_residency_policy)),
            writer_residency_policy=str(payload.get("writer_residency_policy", fallback.writer_residency_policy)),
            live_corrector_residency_policy=str(payload.get("live_corrector_residency_policy", fallback.live_corrector_residency_policy)),
        )

    def to_dict(self) -> dict:
        return {
            "profile_name": self.profile_name,
            "preview_repo_id": self.preview_repo_id,
            "final_repo_id": self.final_repo_id,
            "preview_model_path": str(self.preview_model_path),
            "final_model_path": str(self.final_model_path),
            "supported_languages": list(self.supported_languages),
            "language": self.language,
            "language_policy": self.language_policy,
            "preview_device": self.preview_device,
            "final_device": self.final_device,
            "preview_compute_type": self.preview_compute_type,
            "final_compute_type": self.final_compute_type,
            "preview_backend": self.preview_backend,
            "preview_service_backend": self.preview_service_backend,
            "final_backend": self.final_backend,
            "writer_backend": self.writer_backend,
            "writer_model_path": self.writer_model_path,
            "writer_max_tokens": self.writer_max_tokens,
            "writer_temperature": self.writer_temperature,
            "writer_top_p": self.writer_top_p,
            "live_corrector_backend": self.live_corrector_backend,
            "live_corrector_model_path": self.live_corrector_model_path,
            "live_corrector_max_tokens": self.live_corrector_max_tokens,
            "live_corrector_temperature": self.live_corrector_temperature,
            "live_corrector_top_p": self.live_corrector_top_p,
            "preview_endpoint": self.preview_endpoint,
            "final_endpoint": self.final_endpoint,
            "writer_endpoint": self.writer_endpoint,
            "profile_class": self.profile_class,
            "target_machine": self.target_machine,
            "notes": list(self.notes),
            "speech_profile": self.speech_profile,
            "preview_lane_status": self.preview_lane_status,
            "final_lane_status": self.final_lane_status,
            "writer_lane_status": self.writer_lane_status,
            "gpu_memory_budget_mb": self.gpu_memory_budget_mb,
            "preview_gpu_memory_mb": self.preview_gpu_memory_mb,
            "final_gpu_memory_mb": self.final_gpu_memory_mb,
            "writer_gpu_memory_mb": self.writer_gpu_memory_mb,
            "live_corrector_gpu_memory_mb": self.live_corrector_gpu_memory_mb,
            "preview_residency_policy": self.preview_residency_policy,
            "final_residency_policy": self.final_residency_policy,
            "writer_residency_policy": self.writer_residency_policy,
            "live_corrector_residency_policy": self.live_corrector_residency_policy,
        }

    def save(self, *, paths: DemoPaths) -> Path:
        paths.ensure_dirs()
        target = paths.resolved_config_json()
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def resolve_profile_name(profile_name: str) -> str:
    return _PROFILE_ALIASES.get(profile_name, profile_name)


def get_demo_profile(profile_name: str) -> DemoProfile:
    resolved = resolve_profile_name(profile_name)
    try:
        return DEMO_PROFILES[resolved]
    except KeyError as exc:
        raise ValueError(f"Unknown Juno profile: {profile_name}") from exc


def list_demo_profiles(*, profile_class: str | None = None) -> list[DemoProfile]:
    profiles = list(DEMO_PROFILES.values())
    if profile_class is not None:
        profiles = [profile for profile in profiles if profile.profile_class == profile_class]
    return profiles


def load_demo_config(*, paths: DemoPaths, profile_name: str | None = None) -> DemoConfig:
    config_path = paths.resolved_config_json()
    if config_path.exists():
        cfg = DemoConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        # Keep canonical profiles truthful even if a stale local config exists from
        # older model stacks. This is narrowly targeted to the best_local path.
        if cfg.profile_name == "best_local":
            profile = DEMO_PROFILES["best_local"]
            if (cfg.final_backend or "").strip().lower() == "mlx_whisper" and cfg.final_repo_id != profile.final_repo_id:
                cfg.final_repo_id = profile.final_repo_id
            if (cfg.preview_backend or "").strip().lower() == "streaming_local_http_json" and cfg.preview_repo_id != profile.preview_repo_id:
                cfg.preview_repo_id = profile.preview_repo_id
            cfg.preview_service_backend = profile.preview_service_backend
            cfg.live_corrector_backend = profile.live_corrector_backend
            cfg.live_corrector_model_path = profile.live_corrector_model_path
            cfg.live_corrector_max_tokens = profile.live_corrector_max_tokens
            cfg.live_corrector_temperature = profile.live_corrector_temperature
            cfg.live_corrector_top_p = profile.live_corrector_top_p
            cfg.live_corrector_gpu_memory_mb = profile.live_corrector_gpu_memory_mb
            cfg.live_corrector_residency_policy = profile.live_corrector_residency_policy
        return cfg
    return DemoConfig.from_profile(resolve_profile_name(profile_name or DEFAULT_DEMO_PROFILE), paths=paths)
