"""Fresh-install provisioning regression tests.

A DMG user's first launch starts the engine with an empty HF cache. The
historical failure mode: the fatal startup warm tried to load (and thus
download) models inside a 15s HTTP budget, crash-looped the engine every
~20s, and each restart killed the in-flight download — so setup never
finished and onboarding showed a featureless spinner forever. These tests
pin the three structural defenses:

1. Cache-completeness probes require weights, not just config.json, so an
   interrupted download reads as "needs install" everywhere (provisioning,
   offline-mode pinning, setup-status lanes).
2. In shell broker mode, ASR lanes with incomplete caches are skipped from
   the fatal initial warm and deferred to the post-install warm.
3. The streaming preview warm() polls with a cold-start deadline instead
   of failing the engine on one slow request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from juno_v2.demo.models import is_hf_model_cache_complete
from juno_v2.preview.backends.streaming_local_http_json import (
    StreamingLocalHttpJsonPreviewBackend,
)
from juno_v2.preview.config import PreviewAsrConfig
from juno_v2.runtime.offline_mode import enable_offline_mode_if_cache_complete


REPO_ID = "test-org/test-model"


def _make_cached_snapshot(
    root: Path,
    *,
    weights: bool = True,
    dangling_weights: bool = False,
    shards: dict[str, bool] | None = None,
) -> Path:
    """Build a minimal HF-cache-shaped snapshot dir (symlinks into blobs/)."""
    repo_dir = root / "models--test-org--test-model"
    snapshot = repo_dir / "snapshots" / "abc123"
    blobs = repo_dir / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)

    (blobs / "cfg").write_text("{}", encoding="utf-8")
    (snapshot / "config.json").symlink_to(blobs / "cfg")

    if weights:
        (blobs / "w").write_bytes(b"x" * 64)
        (snapshot / "weights.npz").symlink_to(blobs / "w")
    if dangling_weights:
        # Symlink whose blob was never written (download interrupted).
        (snapshot / "weights.npz").symlink_to(blobs / "missing-blob")
    if shards is not None:
        import json

        weight_map = {f"layer{i}": shard for i, shard in enumerate(shards)}
        (blobs / "idx").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        (snapshot / "model.safetensors.index.json").symlink_to(blobs / "idx")
        for shard, present in shards.items():
            if present:
                (blobs / shard).write_bytes(b"x" * 64)
                (snapshot / shard).symlink_to(blobs / shard)
    return snapshot


def _patch_cache_probe(monkeypatch: pytest.MonkeyPatch, snapshot: Path | None) -> None:
    import huggingface_hub

    def fake_try_to_load_from_cache(*, repo_id: str, filename: str):
        if snapshot is None or repo_id != REPO_ID:
            return None
        candidate = snapshot / filename
        return str(candidate) if candidate.is_file() else None

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", fake_try_to_load_from_cache
    )


class TestCacheCompleteProbe:
    def test_complete_snapshot(self, tmp_path, monkeypatch):
        snapshot = _make_cached_snapshot(tmp_path, weights=True)
        _patch_cache_probe(monkeypatch, snapshot)
        assert is_hf_model_cache_complete(REPO_ID) is True

    def test_config_only_interrupted_download(self, tmp_path, monkeypatch):
        # The original field failure: config.json landed, the multi-GB
        # weights did not. Must read as incomplete.
        snapshot = _make_cached_snapshot(tmp_path, weights=False)
        _patch_cache_probe(monkeypatch, snapshot)
        assert is_hf_model_cache_complete(REPO_ID) is False

    def test_dangling_weights_symlink(self, tmp_path, monkeypatch):
        snapshot = _make_cached_snapshot(tmp_path, weights=False, dangling_weights=True)
        _patch_cache_probe(monkeypatch, snapshot)
        assert is_hf_model_cache_complete(REPO_ID) is False

    def test_not_cached_at_all(self, monkeypatch):
        _patch_cache_probe(monkeypatch, None)
        assert is_hf_model_cache_complete(REPO_ID) is False

    def test_sharded_complete(self, tmp_path, monkeypatch):
        snapshot = _make_cached_snapshot(
            tmp_path,
            weights=False,
            shards={"model-00001-of-00002.safetensors": True,
                    "model-00002-of-00002.safetensors": True},
        )
        _patch_cache_probe(monkeypatch, snapshot)
        assert is_hf_model_cache_complete(REPO_ID) is True

    def test_sharded_missing_shard(self, tmp_path, monkeypatch):
        snapshot = _make_cached_snapshot(
            tmp_path,
            weights=False,
            shards={"model-00001-of-00002.safetensors": True,
                    "model-00002-of-00002.safetensors": False},
        )
        _patch_cache_probe(monkeypatch, snapshot)
        assert is_hf_model_cache_complete(REPO_ID) is False


class TestOfflineModePinning:
    def test_incomplete_cache_never_pins_offline(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        snapshot = _make_cached_snapshot(tmp_path, weights=False)
        _patch_cache_probe(monkeypatch, snapshot)
        assert enable_offline_mode_if_cache_complete([REPO_ID]) is False
        import os

        assert "HF_HUB_OFFLINE" not in os.environ

    def test_complete_cache_pins_offline(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        snapshot = _make_cached_snapshot(tmp_path, weights=True)
        _patch_cache_probe(monkeypatch, snapshot)
        try:
            assert enable_offline_mode_if_cache_complete([REPO_ID]) is True
            import os

            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        finally:
            monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


class TestInitialWarmSkipRoles:
    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from juno_v2.runtime.deployment import ProductionServiceConfig
        from juno_v2.runtime.service import ProductionServiceRunner

        config = ProductionServiceConfig(
            mode="live",
            preview_backend="streaming_local_http_json",
            preview_model_path=REPO_ID,
            preview_endpoint="http://127.0.0.1:1",
            final_backend="mlx_whisper",
            final_model_path=REPO_ID,
            final_hf_repo_id=REPO_ID,
            engine_socket_path=str(tmp_path / "engine.sock"),
        )
        return ProductionServiceRunner(config)

    def test_uncached_asr_lanes_skip_fatal_warm(self, runner, tmp_path, monkeypatch):
        snapshot = _make_cached_snapshot(tmp_path / "cache", weights=False)
        _patch_cache_probe(monkeypatch, snapshot)
        skip = runner._initial_warm_skip_roles(preview_worker_warms_backend=False)
        assert skip is not None
        assert {"preview_asr", "final_asr", "live_corrector", "writer"} <= skip

    def test_cached_asr_lanes_warm_as_before(self, runner, tmp_path, monkeypatch):
        snapshot = _make_cached_snapshot(tmp_path / "cache", weights=True)
        _patch_cache_probe(monkeypatch, snapshot)
        skip = runner._initial_warm_skip_roles(preview_worker_warms_backend=False)
        assert skip == {"live_corrector", "writer"}

    def test_probe_error_defers_rather_than_crashes(self, runner, monkeypatch):
        # If the probe itself blows up, prefer a deferred (lazy) warm over
        # a potentially fatal in-band download.
        import juno_v2.demo.models as demo_models

        def boom(_repo_id):
            raise OSError("probe failed")

        monkeypatch.setattr(demo_models, "is_hf_model_cache_complete", boom)
        skip = runner._initial_warm_skip_roles(preview_worker_warms_backend=False)
        assert {"preview_asr", "final_asr"} <= (skip or set())


class TestProfileRoot:
    def test_packaged_app_uses_support_root(self, tmp_path, monkeypatch):
        # cwd inside a sealed .app bundle with no .juno_v2_demo: scratch
        # must land in Application Support, never next to the binary.
        monkeypatch.delenv("JUNO_DEV_MODE", raising=False)
        monkeypatch.setenv("JUNO_APP_SUPPORT_DIR", str(tmp_path / "support"))
        from juno_v2.runtime.paths import juno_profile_root

        bundle_cwd = tmp_path / "Juno.app" / "Contents" / "Resources" / "engine"
        bundle_cwd.mkdir(parents=True)
        root = juno_profile_root(bundle_cwd)
        assert root == (tmp_path / "support" / "demo").resolve()

    def test_existing_repo_demo_dir_is_preserved(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JUNO_DEV_MODE", raising=False)
        monkeypatch.setenv("JUNO_APP_SUPPORT_DIR", str(tmp_path / "support"))
        from juno_v2.runtime.paths import juno_profile_root

        repo_cwd = tmp_path / "repo"
        (repo_cwd / ".juno_v2_demo").mkdir(parents=True)
        assert juno_profile_root(repo_cwd) == (repo_cwd / ".juno_v2_demo").resolve()

    def test_dev_mode_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JUNO_DEV_MODE", "1")
        monkeypatch.setenv("JUNO_APP_SUPPORT_DIR", str(tmp_path / "support"))
        from juno_v2.runtime.paths import juno_profile_root

        cwd = tmp_path / "anywhere"
        cwd.mkdir()
        assert juno_profile_root(cwd) == (cwd / ".juno_v2_demo").resolve()


class TestStreamingWarmDeadline:
    def _backend(self, *, deadline: float) -> StreamingLocalHttpJsonPreviewBackend:
        return StreamingLocalHttpJsonPreviewBackend(
            PreviewAsrConfig(
                model_path=REPO_ID,
                backend_name="streaming_local_http_json",
                local_http_endpoint="http://127.0.0.1:1",
                local_http_timeout_sec=0.01,
                warm_deadline_sec=deadline,
            )
        )

    def _silence_sleep(self, monkeypatch):
        monkeypatch.setattr(
            "juno_v2.preview.backends.streaming_local_http_json.time.sleep",
            lambda _s: None,
        )

    def test_warm_polls_through_timeouts_and_warming(self, monkeypatch):
        backend = self._backend(deadline=30.0)
        # Request timeout ({}), then alive-but-loading, then ready: the
        # exact shape of a cold service whose first /warm blocks on the
        # model load.
        responses = iter([{}, {"ok": False, "error": None}, {"ok": True}])
        monkeypatch.setattr(backend, "_get_status", lambda path, strict=False: next(responses))
        self._silence_sleep(monkeypatch)
        backend.warm()  # must not raise, must consume all three polls
        with pytest.raises(StopIteration):
            next(responses)

    def test_warm_fails_fast_on_service_startup_error(self, monkeypatch):
        backend = self._backend(deadline=30.0)
        monkeypatch.setattr(
            backend,
            "_get_status",
            lambda path, strict=False: {"ok": False, "error": "RuntimeError: no metal device"},
        )
        self._silence_sleep(monkeypatch)
        with pytest.raises(RuntimeError, match="no metal device"):
            backend.warm()

    def test_warm_raises_after_deadline(self, monkeypatch):
        backend = self._backend(deadline=0.05)
        monkeypatch.setattr(backend, "_get_status", lambda path, strict=False: {})
        self._silence_sleep(monkeypatch)
        with pytest.raises(RuntimeError, match="service unreachable"):
            backend.warm()
