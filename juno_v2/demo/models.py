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


# Weight artifacts any of our model repos can ship (mlx_whisper uses
# weights.npz, mlx_lm uses [sharded] safetensors, faster-whisper uses
# model.bin, gguf covers llama.cpp-style packages).
_WEIGHT_FILE_GLOBS = ("*.safetensors", "*.npz", "*.bin", "*.gguf")


def _snapshot_file_ok(path: Path) -> bool:
    # Snapshot entries are symlinks into blobs/; a dangling link (blob
    # deleted) or zero-byte blob means the snapshot is not usable.
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def is_hf_model_cache_complete(repo_id: str) -> bool:
    """True iff the HF cache holds a *usable* snapshot of ``repo_id`` —
    config plus weights — without any network call.

    ``is_hf_model_cached`` probes only ``config.json``, which is one of
    the first (tiny) files a snapshot download writes. A download killed
    mid-flight leaves config.json cached while the multi-GB weights are
    still missing; treating that as "cached" makes the engine warm from
    — or pin ``HF_HUB_OFFLINE`` to — a broken snapshot, and makes
    provisioning skip the repair it was asked to do. This probe
    additionally requires at least one non-empty weight artifact in the
    snapshot, and when the repo ships a sharded-weights index, every
    shard the index names.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        cfg = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    except ImportError:
        return False
    if not isinstance(cfg, str):
        return False
    snapshot_dir = Path(cfg).parent

    index_path = snapshot_dir / "model.safetensors.index.json"
    if _snapshot_file_ok(index_path):
        try:
            import json
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
            shards = set(weight_map.values())
        except (OSError, ValueError):
            return False
        if not shards:
            return False
        return all(_snapshot_file_ok(snapshot_dir / shard) for shard in shards)

    for pattern in _WEIGHT_FILE_GLOBS:
        if any(_snapshot_file_ok(candidate) for candidate in snapshot_dir.rglob(pattern)):
            return True
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
    # Completeness, not just config.json presence: a previous download
    # killed mid-flight must be finished here, not skipped.
    if is_hf_model_cache_complete(repo) and not force:
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
