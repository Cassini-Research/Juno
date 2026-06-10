from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from juno_v2.commit.controller import CommitController
from juno_v2.context.linux_desktop import LinuxDesktopContextProvider, LinuxDesktopContextProviderConfig
from juno_v2.context.macos_desktop import MacOSDesktopContextProvider, MacOSDesktopContextProviderConfig
from juno_v2.context.provider import ContextProvider, StaticContextProvider, WorkbenchContextProvider
from juno_v2.engine.session import DictationSessionRunner
from juno_v2.presets.surface_presets import SurfacePresetStore, default_surface_presets_path
from juno_v2.final.config import FinalAsrConfig
from juno_v2.language.catalog import (
    DEFAULT_SUPPORTED_LANGUAGES,
    parse_pair_policy_string,
)
from juno_v2.language.policy import LanguagePlanner, LanguagePlannerConfig
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.personalization.seed.runtime import JunoSeedPersonalizationRuntime
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.preview.config import PreviewAsrConfig
from juno_v2.runtime.backends import create_final_backend, create_preview_backend, create_writer_backend
from juno_v2.runtime.lifecycle import BackendLifecycleManager
from juno_v2.runtime.platform import detect_platform_name, resolve_backend_runtime, validate_context_source, validate_insertion_target
from juno_v2.speech.config import DEFAULT_SPEECH_PROFILE, dual_vad_policy_for_profile, speech_state_config_for_profile
from juno_v2.insertion.macos_desktop import MacOSActiveAppInserter, MacOSActiveAppInserterConfig
from juno_v2.workbench.store import WorkbenchStore
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.service import WriterService
from juno_core_v3.model_registry.contracts import SurfaceClass
from juno_core_v3.model_registry.registry import ModelRegistry
from juno_core_v3.model_registry.resolver import ResolvedEngineRoutes, apply_routes_to_spec, resolve_engine_routes, resolved_slot_is_actionable
from juno_core_v3.model_registry.routing import RouteChooser


@dataclass(slots=True)
class CanonicalEngineArtifacts:
    runner: DictationSessionRunner
    recorder: TraceRecorder
    store: WorkbenchStore
    controller: CommitController
    context_provider: ContextProvider
    memory_store: JsonMemoryStore
    writer_service: WriterService
    lifecycle_manager: BackendLifecycleManager
    live_corrector_service: WriterService | None = None
    # final_backend is exposed so the one-shot dictation HTTP path
    # (POST /api/broker/dictation/ingest_wav and the OpenAI-compatible
    # /v1/audio/transcriptions) can reuse the same loaded ASR model as
    # the live streaming runner. Without this, the workbench falls back
    # to a stub transcriber even when a real model is running right next
    # to it, which is how we silently shipped a broken voice-to-typing
    # flow in the first audit.
    final_backend: object | None = None
    inserter: object | None = None


@dataclass(slots=True)
class CanonicalEngineBuildSpec:
    engine_mode: str
    session_id: str
    preview_model_path: str | Path
    final_model_path: str | Path
    preview_backend_name: str = 'faster_whisper'
    final_backend_name: str = 'faster_whisper'
    preview_endpoint: str | None = None
    final_endpoint: str | None = None
    writer_backend_name: str | None = None
    writer_endpoint: str | None = None
    writer_model_path: str | None = None
    writer_max_tokens: int = 512
    writer_temperature: float = 0.0
    writer_top_p: float = 1.0
    writer_idle_unload_ttl_s: float = 300.0
    live_corrector_enabled: bool = False
    live_corrector_backend_name: str | None = None
    live_corrector_endpoint: str | None = None
    live_corrector_model_path: str | None = None
    live_corrector_max_tokens: int = 160
    live_corrector_temperature: float = 0.0
    live_corrector_top_p: float = 1.0
    live_corrector_residency_policy: str = 'resident'
    language: str | None = None
    language_policy: str = 'auto_supported'
    supported_languages: tuple[str, ...] = DEFAULT_SUPPORTED_LANGUAGES
    sample_rate_hz: int = 16000
    frame_ms: int = 20
    speech_profile_name: str = DEFAULT_SPEECH_PROFILE
    memory_dir: str | Path = '.juno_v2_memory'
    trace_log_dir: str | Path = '.juno_v2_logs/engine'
    audio_save_dir: str | Path | None = None  # None = auto (saves to trace_log_dir/audio/)
    app_name: str | None = None
    window_title: str | None = None
    context_source: str = 'static'
    context_helper_command: str | None = None
    context_provider: ContextProvider | None = None
    insertion_target: str = 'none'
    insertion_helper_command: str | None = None
    gpu_memory_budget_mb: int | None = None
    preview_gpu_memory_mb: int = 0
    final_gpu_memory_mb: int = 0
    writer_gpu_memory_mb: int = 0
    live_corrector_gpu_memory_mb: int = 0
    preview_residency_group: str | None = None
    final_residency_group: str | None = None
    preview_residency_policy: str = 'resident'
    final_residency_policy: str = 'resident'
    writer_residency_policy: str = 'resident'
    platform_name: str | None = None
    preview_device: str = 'auto'
    final_device: str = 'auto'
    preview_compute_type: str = 'default'
    final_compute_type: str = 'default'
    final_hf_repo_id: str | None = None
    # When False the engine session skips per-utterance preview-lane decoding
    # to save CPU/GPU. The preview backend is *still constructed and registered
    # with the lifecycle manager* — the model on disk and the resident
    # streaming-preview service are unaffected; only ``preview_backend.decode``
    # is bypassed at the session level. Read at session start; the broker
    # setter only takes effect on the next dictation session.
    live_caption_enabled: bool = True


def build_canonical_engine(spec: CanonicalEngineBuildSpec) -> CanonicalEngineArtifacts:
    recorder = TraceRecorder(session_id=spec.session_id, log_dir=Path(spec.trace_log_dir))
    store = WorkbenchStore(recorder)
    controller = CommitController(store)
    memory_store = JsonMemoryStore(spec.memory_dir)
    juno_seed_runtime = JunoSeedPersonalizationRuntime.try_load(memory_store=memory_store)

    platform_name = detect_platform_name(spec.platform_name)
    validate_context_source(spec.context_source, platform_name=platform_name)
    validate_insertion_target(spec.insertion_target, platform_name=platform_name)
    preview_runtime = resolve_backend_runtime(
        backend_name=spec.preview_backend_name,
        requested_device=spec.preview_device,
        requested_compute_type=spec.preview_compute_type,
        platform_name=platform_name,
    )
    final_runtime = resolve_backend_runtime(
        backend_name=spec.final_backend_name,
        requested_device=spec.final_device,
        requested_compute_type=spec.final_compute_type,
        platform_name=platform_name,
    )
    preview_config = PreviewAsrConfig(
        model_path=Path(spec.preview_model_path),
        language=spec.language,
        backend_name=spec.preview_backend_name,
        device=preview_runtime.resolved_device,
        compute_type=preview_runtime.resolved_compute_type,
        local_http_endpoint=spec.preview_endpoint,
    )
    final_config = FinalAsrConfig(
        model_path=Path(spec.final_model_path),
        language=spec.language,
        backend_name=spec.final_backend_name,
        device=final_runtime.resolved_device,
        compute_type=final_runtime.resolved_compute_type,
        local_http_endpoint=spec.final_endpoint,
        hf_repo_id=spec.final_hf_repo_id,
    )
    preview_backend = create_preview_backend(preview_config)
    final_backend = create_final_backend(final_config)
    # Optional hook: backends that know how to emit structured warm/unload
    # events can consume the shared TraceRecorder instead of print(). Silently
    # no-op for backends that don't expose set_tracer.
    final_set_tracer = getattr(final_backend, 'set_tracer', None)
    if callable(final_set_tracer):
        final_set_tracer(recorder)

    context_provider = spec.context_provider or _build_context_provider(spec, store)
    inserter = _build_inserter(spec, context_provider)
    controller.inserter = inserter
    writer_config = WriterConfig(
        backend_name=spec.writer_backend_name,
        local_http_endpoint=spec.writer_endpoint,
        model_path=spec.writer_model_path,
        max_tokens=spec.writer_max_tokens,
        temperature=spec.writer_temperature,
        top_p=spec.writer_top_p,
        idle_unload_ttl_s=float(spec.writer_idle_unload_ttl_s),
    )
    writer_service = WriterService(config=writer_config, recorder=recorder, backend=create_writer_backend(writer_config))
    live_corrector_service: WriterService | None = None
    if spec.live_corrector_enabled and spec.live_corrector_backend_name:
        live_corrector_config = WriterConfig(
            backend_name=spec.live_corrector_backend_name,
            local_http_endpoint=spec.live_corrector_endpoint,
            model_path=spec.live_corrector_model_path,
            max_tokens=spec.live_corrector_max_tokens,
            temperature=spec.live_corrector_temperature,
            top_p=spec.live_corrector_top_p,
            residency_policy=spec.live_corrector_residency_policy,
        )
        live_corrector_service = WriterService(
            config=live_corrector_config,
            recorder=recorder,
            backend=create_writer_backend(live_corrector_config),
        )
    # Strictly validate ``pair:<a>,<b>`` at construction time so the product
    # fails fast instead of silently falling back to (en, hi) at runtime when
    # an operator passes a malformed policy string.
    normalized_pair: tuple[str, str] | None = None
    if spec.language_policy.startswith('pair:'):
        normalized_pair = parse_pair_policy_string(
            spec.language_policy,
            supported=tuple(spec.supported_languages) or DEFAULT_SUPPORTED_LANGUAGES,
        )
    language_planner = LanguagePlanner(LanguagePlannerConfig(
        policy_name=spec.language_policy,
        fixed_language=spec.language if spec.language_policy == 'fixed' else None,
        supported_languages=list(spec.supported_languages),
        pair_languages=normalized_pair,
    ))
    lifecycle = BackendLifecycleManager(total_gpu_memory_mb=spec.gpu_memory_budget_mb)
    # Streaming MLX preview backends bind state to the thread that warms
    # them. The workbench server warms them on a dedicated decode worker.
    # Flagging the registration prevents warm_all() (which runs on the main
    # thread) from doing a wasted warm that would just be unloaded and
    # re-warmed on the worker.
    _preview_warm_on_main_thread = True
    lifecycle.register_backend('preview_asr', preview_backend, metadata={
        'configured_backend': preview_config.backend_name,
        'model_path': str(preview_config.model_path),
        'language': preview_config.language,
        'device': preview_config.device,
        'compute_type': preview_config.compute_type,
        'engine_mode': spec.engine_mode,
        'platform_name': platform_name,
        'runtime_resolution': preview_runtime.to_dict(),
    }, gpu_memory_mb=spec.preview_gpu_memory_mb, residency_group=spec.preview_residency_group, residency_policy=spec.preview_residency_policy, warm_on_main_thread=_preview_warm_on_main_thread)
    lifecycle.register_backend('final_asr', final_backend, metadata={
        'configured_backend': final_config.backend_name,
        'model_path': str(final_config.model_path),
        'language': final_config.language,
        'device': final_config.device,
        'compute_type': final_config.compute_type,
        'engine_mode': spec.engine_mode,
        'platform_name': platform_name,
        'runtime_resolution': final_runtime.to_dict(),
    }, gpu_memory_mb=spec.final_gpu_memory_mb, residency_group=spec.final_residency_group, residency_policy=spec.final_residency_policy)
    if writer_service.backend is not None:
        lifecycle.register_backend(
            'writer',
            writer_service.backend,
            metadata={
                'configured_backend': spec.writer_backend_name,
                'model_path': spec.writer_model_path,
                'max_tokens': spec.writer_max_tokens,
                'temperature': spec.writer_temperature,
                'top_p': spec.writer_top_p,
                'engine_mode': spec.engine_mode,
                'platform_name': platform_name,
            },
            gpu_memory_mb=spec.writer_gpu_memory_mb,
            residency_policy=spec.writer_residency_policy,
            # TTL only has an effect under residency_policy='on_demand'. The
            # writer backend defaults to 'resident' now (to keep the Qwen
            # model warm and avoid the 3-5s MLX cold-start on every command),
            # so this TTL is a no-op on the default path. It still applies
            # when callers explicitly opt into on_demand (e.g. memory-pressure
            # builds or benchmarking).
            idle_unload_ttl_s=writer_config.idle_unload_ttl_s,
        )
        writer_service.backend_acquire = lambda: lifecycle.acquire('writer')
        writer_service.backend_release = lambda: lifecycle.release('writer')
    if live_corrector_service is not None and live_corrector_service.backend is not None:
        lifecycle.register_backend(
            'live_corrector',
            live_corrector_service.backend,
            metadata={
                'configured_backend': spec.live_corrector_backend_name,
                'model_path': spec.live_corrector_model_path,
                'max_tokens': spec.live_corrector_max_tokens,
                'temperature': spec.live_corrector_temperature,
                'top_p': spec.live_corrector_top_p,
                'engine_mode': spec.engine_mode,
                'platform_name': platform_name,
            },
            gpu_memory_mb=spec.live_corrector_gpu_memory_mb,
            residency_policy=spec.live_corrector_residency_policy,
        )
        live_corrector_service.backend_acquire = lambda: lifecycle.acquire('live_corrector')
        live_corrector_service.backend_release = lambda: lifecycle.release('live_corrector')
    lifecycle.register_component('writer_service', writer_service.state.mode.value, lambda: None, metadata={
        'backend_enabled': writer_service.backend is not None,
        'engine_mode': spec.engine_mode,
    }, residency_policy='resident')
    if inserter is not None:
        lifecycle.register_component('active_app_inserter', getattr(inserter, 'name', 'active_app_inserter'), inserter.warm, metadata={
            'engine_mode': spec.engine_mode,
            'insertion_target': spec.insertion_target,
            'capabilities': inserter.capabilities(),
        })
    _audio_save_dir: Path | None = (
        Path(spec.audio_save_dir) if spec.audio_save_dir is not None
        else Path(spec.trace_log_dir) / "audio"
    )
    surface_preset_store = SurfacePresetStore(default_surface_presets_path(recorder.log_dir))
    runner = DictationSessionRunner(
        state_config=speech_state_config_for_profile(spec.speech_profile_name, sample_rate_hz=spec.sample_rate_hz, frame_ms=spec.frame_ms),
        preview_config=preview_config,
        final_config=final_config,
        vad_policy=dual_vad_policy_for_profile(spec.speech_profile_name),
        preview_backend=preview_backend,
        final_backend=final_backend,
        recorder=recorder,
        controller=controller,
        context_provider=context_provider,
        memory_store=memory_store,
        surface_preset_store=surface_preset_store,
        writer_service=writer_service,
        language_planner=language_planner,
        lifecycle_manager=lifecycle,
        engine_mode=spec.engine_mode,
        audio_save_dir=_audio_save_dir,
        juno_seed_runtime=juno_seed_runtime,
        preview_decode_enabled=spec.live_caption_enabled,
    )
    return CanonicalEngineArtifacts(
        runner=runner,
        recorder=recorder,
        store=store,
        controller=controller,
        context_provider=context_provider,
        memory_store=memory_store,
        writer_service=writer_service,
        live_corrector_service=live_corrector_service,
        lifecycle_manager=lifecycle,
        final_backend=final_backend,
        inserter=inserter,
    )


def _surface_class_for_registry(surface: str) -> SurfaceClass:
    normalized = (surface or '').strip().lower()
    if not normalized:
        return SurfaceClass.DESKTOP
    if normalized in {member.value for member in SurfaceClass}:
        return SurfaceClass(normalized)
    if normalized in {'iphone_app'}:
        return SurfaceClass.PHONE_CLASS
    if normalized in {'iphone_keyboard'}:
        return SurfaceClass.KEYBOARD_EXTENSION
    return SurfaceClass.DESKTOP


def build_canonical_engine_from_registry(
    spec: CanonicalEngineBuildSpec,
    *,
    registry: object,
    surface: str,
    override_spec: bool = False,
) -> CanonicalEngineArtifacts:
    """Build the canonical engine using registry-selected routes as runtime truth.

    ``override_spec=False`` preserves explicit caller choices already present on
    ``spec`` and only fills empty/default fields from the registry. ``override_spec=True``
    makes the registry authoritative for every resolved slot.
    """
    if not isinstance(registry, ModelRegistry):
        raise TypeError(
            f"build_canonical_engine_from_registry expected ModelRegistry, got {type(registry).__name__}"
        )

    surface_class = _surface_class_for_registry(surface)
    chooser = RouteChooser(registry)
    routes = resolve_engine_routes(
        chooser,
        surface=surface_class,
        language=spec.language,
    )
    preview_ok, preview_reason = resolved_slot_is_actionable(routes.preview)
    final_ok, final_reason = resolved_slot_is_actionable(routes.final)
    if routes.preview is None or not preview_ok:
        raise ValueError(f"registry_unbuildable_slot: preview_asr:{preview_reason or 'missing_route'}")
    if routes.final is None or not final_ok:
        raise ValueError(f"registry_unbuildable_slot: final_asr:{final_reason or 'missing_route'}")

    writer_route = routes.writer
    writer_skip_reason: str | None = None
    if writer_route is not None:
        writer_ok, writer_reason = resolved_slot_is_actionable(writer_route)
        if not writer_ok:
            writer_skip_reason = writer_reason or 'missing_runtime_bits'
            writer_route = None

    runtime_truth_routes = ResolvedEngineRoutes(
        preview=routes.preview,
        final=routes.final,
        writer=writer_route,
    )
    resolved_spec = apply_routes_to_spec(spec, runtime_truth_routes, override=override_spec)

    import logging as _logging
    _logging.getLogger(__name__).info(
        'registry_launch: surface=%s surface_class=%s override_spec=%s preview=%s final=%s writer=%s writer_skip_reason=%s',
        surface,
        surface_class.value,
        override_spec,
        routes.preview.package_id,
        routes.final.package_id,
        writer_route.package_id if writer_route is not None else None,
        writer_skip_reason,
    )

    return build_canonical_engine(resolved_spec)


def build_workbench_context_provider(store: WorkbenchStore) -> WorkbenchContextProvider:
    return WorkbenchContextProvider(store)


def _build_context_provider(spec: CanonicalEngineBuildSpec, store: WorkbenchStore) -> ContextProvider:
    if spec.context_source == 'workbench':
        return WorkbenchContextProvider(store)
    if spec.context_source == 'linux_desktop':
        return LinuxDesktopContextProvider(LinuxDesktopContextProviderConfig(helper_command=spec.context_helper_command))
    if spec.context_source == 'macos_desktop':
        return MacOSDesktopContextProvider(MacOSDesktopContextProviderConfig(helper_command=spec.context_helper_command))
    return StaticContextProvider(app_name=spec.app_name, window_title=spec.window_title)


def _build_inserter(spec: CanonicalEngineBuildSpec, context_provider: ContextProvider) -> object | None:
    if spec.insertion_target == 'macos_active_app':
        state_provider = context_provider if isinstance(context_provider, MacOSDesktopContextProvider) else None
        return MacOSActiveAppInserter(
            config=MacOSActiveAppInserterConfig(helper_command=spec.insertion_helper_command or spec.context_helper_command),
            state_provider=state_provider,
        )
    return None
