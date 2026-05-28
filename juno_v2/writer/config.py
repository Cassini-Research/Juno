from __future__ import annotations

from dataclasses import dataclass

from juno_v2.contracts.writer import WriterMode


@dataclass(slots=True)
class WriterConfig:
    backend_name: str | None = None
    local_http_endpoint: str | None = None
    local_http_timeout_sec: float = 30.0
    model_path: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    default_mode: WriterMode = WriterMode.DEFAULT_SURFACE
    enable_model_transforms: bool = True
    enable_deterministic_transforms: bool = True
    # When >0 and the backend uses residency_policy='on_demand', the
    # BackendLifecycleManager defers unload by this many seconds after the
    # last release. If a new acquire arrives within the window, the pending
    # unload is cancelled. This trades a little GPU memory for much lower
    # writer-lane TTFT when a user strings commands together back-to-back.
    # 0.0 preserves the unload-on-every-release legacy behavior.
    idle_unload_ttl_s: float = 30.0
    # Residency policy used when the lifecycle manager registers the writer
    # backend via the DictationSessionRunner fallback path. The factory path
    # already passes its own value via spec.writer_residency_policy; this
    # field only governs the in-process fallback registration in
    # engine/session.py. Default 'resident' matches the rest of the stack —
    # the writer (Qwen3-4B via MLX) pays a 3-5s cold-start on every reload,
    # so keeping it resident is the correct default for an interactive
    # dictation product.
    residency_policy: str = 'resident'
