from __future__ import annotations

from pathlib import Path

from juno_v2.demo.config import DemoConfig, DemoPaths


def is_model_ready(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())


def is_hf_model_cached(repo_id: str, *, filename: str = "config.json") -> bool:
    """Return True if the HF-cached model exists locally (no network call)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        # Returns a path string when cached, or _CACHED_NO_EXIST / None when not
        return isinstance(result, str)
    except ImportError:
        return False


def is_writer_model_cached(repo_id: str) -> bool:
    return is_hf_model_cached(repo_id)


def provision_model(repo_id: str, target_dir: Path, *, force: bool = False) -> Path:
    if is_model_ready(target_dir) and not force:
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface-hub is required to provision local models") from exc
    snapshot_download(repo_id=repo_id, local_dir=str(target_dir), force_download=force)
    return target_dir


def _safe_repo_dir_name(repo_id: str) -> str:
    return (
        "hf__"
        + str(repo_id or "")
        .strip()
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def provision_hf_model_cache(
    repo_id: str,
    *,
    force: bool = False,
    fallback_dir: Path | None = None,
) -> None:
    repo = str(repo_id or "").strip()
    if not repo:
        return
    if is_hf_model_cached(repo) and not force:
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface-hub is required to provision local models") from exc
    try:
        snapshot_download(repo_id=repo, force_download=force)
    except TypeError as exc:
        # Some tests and older hub shims only expose the local_dir form.
        # Keep production on the HF cache path, but retain compatibility
        # with those shims without changing the runtime model identifier.
        if "local_dir" not in str(exc):
            raise
        if fallback_dir is None:
            raise
        fallback_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=str(fallback_dir), force_download=force)


def provision_demo_models(config: DemoConfig, *, paths: DemoPaths, force: bool = False) -> DemoConfig:
    paths.ensure_dirs()
    preview_service_backend = (getattr(config, "preview_service_backend", "") or "").strip().lower()
    if (config.preview_backend or '').strip().lower() == 'streaming_local_http_json' and preview_service_backend in {'mlx_whisper', 'mlx_whisper_streaming'}:
        provision_hf_model_cache(
            config.preview_repo_id,
            force=force,
            fallback_dir=paths.resolved_models_dir() / _safe_repo_dir_name(config.preview_repo_id),
        )
        config.preview_model_path.mkdir(parents=True, exist_ok=True)
    else:
        config.preview_model_path = provision_model(config.preview_repo_id, config.preview_model_path, force=force)
    # For mlx_whisper, prefer passing the HF repo ID directly to mlx_whisper so it can
    # resolve/download backend-native weights into the HF cache. Snapshot-downloading
    # into the demo-local directory risks pulling the wrong artifact format.
    if (config.final_backend or '').strip().lower() == 'mlx_whisper':
        provision_hf_model_cache(
            config.final_repo_id,
            force=force,
            fallback_dir=paths.resolved_models_dir() / _safe_repo_dir_name(config.final_repo_id),
        )
        config.final_model_path.mkdir(parents=True, exist_ok=True)
    else:
        config.final_model_path = provision_model(config.final_repo_id, config.final_model_path, force=force)
    # For mlx_lm writer, download to HF cache so the model is pre-cached before first dictation.
    # mlx_lm.load() accepts the HF repo ID directly and resolves from cache at runtime.
    if (config.writer_backend or '').strip().lower() == 'mlx_lm' and config.writer_model_path:
        writer_repo = str(config.writer_model_path).strip()
        provision_hf_model_cache(
            writer_repo,
            force=force,
            fallback_dir=paths.resolved_models_dir() / _safe_repo_dir_name(writer_repo),
        )
    if (config.live_corrector_backend or '').strip().lower() == 'mlx_lm' and config.live_corrector_model_path:
        live_repo = str(config.live_corrector_model_path).strip()
        provision_hf_model_cache(
            live_repo,
            force=force,
            fallback_dir=paths.resolved_models_dir() / _safe_repo_dir_name(live_repo),
        )
    config.save(paths=paths)
    return config
