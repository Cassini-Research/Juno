"""Best-effort MLX/Metal probe before loading heavy ASR/writer models.

Avoids some startup hard-crashes by failing fast with a Python exception
instead of a Metal assertion inside unrelated threads. Full model warm-up
is not exercised here — only that ``mlx.core`` can allocate and eval a tiny
array on the default device.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def _env_skip_preflight() -> bool:
    raw = os.environ.get("JUNO_SKIP_MLX_PREFLIGHT", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def mlx_metal_operational() -> tuple[bool, str | None]:
    """Return (ok, error_detail). ``error_detail`` is None when ok is True."""
    if _env_skip_preflight():
        return True, None
    code = (
        "import mlx.core as mx\n"
        "a = mx.array([1.0, 2.0])\n"
        "b = mx.array([0.5, 1.5])\n"
        "c = a + b\n"
        "mx.eval(c)\n"
        "print('ok')\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - hardware/env-specific
        mach = platform.machine()
        return False, f"{type(exc).__name__}: {exc} (machine={mach})"
    if proc.returncode != 0:
        mach = platform.machine()
        detail = (proc.stderr or proc.stdout or "").strip()
        if len(detail) > 800:
            detail = detail[:800] + "..."
        return False, f"mlx probe exited {proc.returncode}: {detail} (machine={mach})"
    return True, None


def config_requests_mlx_stack(
    *,
    preview_backend: str,
    final_backend: str,
    writer_backend: str | None,
    live_corrector_backend: str | None = None,
) -> bool:
    """True when any configured lane commonly uses MLX on Apple Silicon."""
    p = (preview_backend or "").lower()
    f = (final_backend or "").lower()
    w = (writer_backend or "").lower()
    lc = (live_corrector_backend or "").lower()
    mlxish = {
        "qwen_asr",
        "mlx_whisper",
        "mlx_lm",
    }
    return p in mlxish or f in mlxish or w in mlxish or lc in mlxish


__all__ = ["config_requests_mlx_stack", "mlx_metal_operational"]
