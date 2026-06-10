from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelSlot(str, Enum):
    """Stable model slots."""

    TINY_ROUTER = "tiny_router"
    PREVIEW_ASR = "preview_asr"
    FINAL_ASR = "final_asr"
    WRITER = "writer"
    DOMAIN_OVERLAY = "domain_overlay"


class RuntimeBackend(str, Enum):
    """Backend runtime identifier."""

    FASTER_WHISPER = "faster_whisper"
    MLX_WHISPER = "mlx_whisper"
    MLX_LM = "mlx_lm"
    LOCAL_HTTP_JSON = "local_http_json"


# Backends that the runtime factory (juno_v2.runtime.backends) actually supports.
RUNTIME_SUPPORTED_BACKENDS: frozenset[RuntimeBackend] = frozenset({
    RuntimeBackend.FASTER_WHISPER,
    RuntimeBackend.MLX_WHISPER,
    RuntimeBackend.MLX_LM,
    RuntimeBackend.LOCAL_HTTP_JSON,
})


class SurfaceClass(str, Enum):
    """Coarse surface classes for routing (per eval + support policy)."""

    DESKTOP = "desktop"
    PHONE_CLASS = "phone_class"
    KEYBOARD_EXTENSION = "keyboard_extension"


class ModelPromotionStage(str, Enum):
    """Promotion lifecycle for a model package."""

    CANDIDATE = "candidate"
    STAGED = "staged"
    PROMOTED = "promoted"
    RETIRED = "retired"


@dataclass(slots=True, frozen=True)
class PackageSignature:
    """Package signature metadata."""

    algo: str
    value: str
