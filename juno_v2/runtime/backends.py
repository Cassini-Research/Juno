from __future__ import annotations

from juno_v2.final.backends.base import FinalAsrBackend
from juno_v2.final.backends.faster_whisper import FasterWhisperFinalBackend
from juno_v2.final.backends.local_http_json import LocalHttpJsonFinalBackend
from juno_v2.final.backends.mlx_whisper import MlxWhisperFinalBackend
from juno_v2.final.config import FinalAsrConfig
from juno_v2.preview.backends.base import PreviewAsrBackend
from juno_v2.preview.backends.faster_whisper import FasterWhisperPreviewBackend
from juno_v2.preview.backends.local_http_json import LocalHttpJsonPreviewBackend
from juno_v2.preview.backends.streaming_local_http_json import StreamingLocalHttpJsonPreviewBackend
from juno_v2.preview.config import PreviewAsrConfig


def create_preview_backend(config: PreviewAsrConfig) -> PreviewAsrBackend:
    """Construct a preview-lane backend.

    Production path is ``streaming_local_http_json`` — a thin HTTP client to
    the resident preview-service subprocess. The other backends are kept for
    Linux CI and portable test fixtures.
    """
    backend = (config.backend_name or "faster_whisper").lower()
    if backend == "faster_whisper":
        return FasterWhisperPreviewBackend(config)
    if backend == "local_http_json":
        return LocalHttpJsonPreviewBackend(config)
    if backend == "streaming_local_http_json":
        return StreamingLocalHttpJsonPreviewBackend(config)
    raise ValueError(f"Unsupported preview backend: {config.backend_name}")


def create_final_backend(config: FinalAsrConfig) -> FinalAsrBackend:
    backend = (config.backend_name or "faster_whisper").lower()
    if backend == "faster_whisper":
        return FasterWhisperFinalBackend(config)
    if backend == "local_http_json":
        return LocalHttpJsonFinalBackend(config)
    if backend == "mlx_whisper":
        return MlxWhisperFinalBackend(config)
    raise ValueError(f"Unsupported final backend: {config.backend_name}")


from juno_v2.writer.backends.base import WriterBackend
from juno_v2.writer.backends.local_http_json import LocalHttpJsonWriterBackend
from juno_v2.writer.backends.mlx_lm import MlxLmWriterBackend
from juno_v2.writer.config import WriterConfig


def create_writer_backend(config: WriterConfig) -> WriterBackend | None:
    backend = (config.backend_name or "").lower().strip()
    if not backend:
        return None
    if backend == "local_http_json":
        return LocalHttpJsonWriterBackend(config)
    if backend == "mlx_lm":
        return MlxLmWriterBackend(config)
    raise ValueError(f"Unsupported writer backend: {config.backend_name}")
