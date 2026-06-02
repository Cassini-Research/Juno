from __future__ import annotations

import os
import platform
from dataclasses import dataclass


SUPPORTED_CONTEXT_SOURCES = {
    'linux': {'static', 'workbench', 'linux_desktop'},
    'macos': {'static', 'workbench', 'macos_desktop'},
    'windows': {'static', 'workbench'},
    'unknown': {'static', 'workbench'},
}



SUPPORTED_INSERTION_TARGETS = {
    'linux': {'none'},
    'macos': {'none', 'macos_active_app'},
    'windows': {'none'},
    'unknown': {'none'},
}

@dataclass(slots=True)
class ResolvedBackendRuntime:
    backend_name: str
    platform_name: str
    requested_device: str
    resolved_device: str
    requested_compute_type: str
    resolved_compute_type: str

    def to_dict(self) -> dict:
        return {
            'backend_name': self.backend_name,
            'platform_name': self.platform_name,
            'requested_device': self.requested_device,
            'resolved_device': self.resolved_device,
            'requested_compute_type': self.requested_compute_type,
            'resolved_compute_type': self.resolved_compute_type,
        }


def detect_platform_name(explicit: str | None = None) -> str:
    raw = (explicit or '').strip().lower()
    if raw in {'linux', 'macos', 'windows'}:
        return raw
    sysname = platform.system().strip().lower()
    if sysname == 'darwin':
        return 'macos'
    if sysname == 'linux':
        return 'linux'
    if sysname.startswith('win'):
        return 'windows'
    return 'unknown'


def validate_context_source(context_source: str, *, platform_name: str) -> None:
    allowed = SUPPORTED_CONTEXT_SOURCES.get(platform_name, SUPPORTED_CONTEXT_SOURCES['unknown'])
    if context_source not in allowed:
        raise ValueError(
            f'Context source {context_source!r} is not supported on platform {platform_name!r}. '
            f'Allowed sources: {sorted(allowed)}'
        )


def resolve_backend_runtime(
    *,
    backend_name: str,
    requested_device: str | None,
    requested_compute_type: str | None,
    platform_name: str | None = None,
) -> ResolvedBackendRuntime:
    normalized_backend = (backend_name or 'faster_whisper').strip().lower()
    normalized_platform = detect_platform_name(platform_name)
    device = (requested_device or 'auto').strip().lower()
    compute = (requested_compute_type or 'default').strip().lower()

    if normalized_backend in {'local_http_json', 'streaming_local_http_json'}:
        return ResolvedBackendRuntime(
            backend_name=normalized_backend,
            platform_name=normalized_platform,
            requested_device=device,
            resolved_device='external_service',
            requested_compute_type=compute,
            resolved_compute_type='external_service',
        )

    if normalized_backend == 'mlx_whisper':
        # Apple Silicon native lane: runtime resolution is largely informational since the backend
        # ignores faster-whisper-specific device/compute knobs.
        return ResolvedBackendRuntime(
            backend_name=normalized_backend,
            platform_name=normalized_platform,
            requested_device=device,
            resolved_device='mlx',
            requested_compute_type=compute,
            resolved_compute_type='mlx',
        )

    if normalized_backend != 'faster_whisper':
        raise ValueError(f'Unsupported backend runtime resolution for backend: {backend_name}')

    if device == 'auto':
        if normalized_platform == 'linux' and _cuda_hint_available():
            resolved_device = 'cuda'
            resolved_compute = 'float16' if compute == 'default' else compute
        else:
            resolved_device = 'cpu'
            resolved_compute = 'int8' if compute == 'default' else compute
    elif device == 'cuda':
        if normalized_platform != 'linux':
            raise ValueError('CUDA execution is only supported on Linux in Juno v2 runtime resolution')
        if not _cuda_hint_available():
            raise ValueError('CUDA was requested but no CUDA-capable runtime was detected')
        resolved_device = 'cuda'
        resolved_compute = 'float16' if compute == 'default' else compute
    elif device == 'cpu':
        resolved_device = 'cpu'
        resolved_compute = 'int8' if compute == 'default' else compute
    elif device == 'mps':
        raise ValueError(
            'The in-repo faster-whisper baseline backend does not support a truthful MPS path yet. '
            'Use cpu on macOS in 10A, or add a dedicated macOS-native backend in a later phase.'
        )
    else:
        raise ValueError(f'Unsupported device selection: {requested_device}')

    return ResolvedBackendRuntime(
        backend_name=normalized_backend,
        platform_name=normalized_platform,
        requested_device=device,
        resolved_device=resolved_device,
        requested_compute_type=compute,
        resolved_compute_type=resolved_compute,
    )


def _cuda_hint_available() -> bool:
    for name in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'):
        value = os.getenv(name, '').strip()
        if value and value.lower() not in {'none', 'void'}:
            return True
    return False


def validate_insertion_target(insertion_target: str, *, platform_name: str) -> None:
    allowed = SUPPORTED_INSERTION_TARGETS.get(platform_name, SUPPORTED_INSERTION_TARGETS['unknown'])
    if insertion_target not in allowed:
        raise ValueError(
            f'Insertion target {insertion_target!r} is not supported on platform {platform_name!r}. '
            f'Allowed targets: {sorted(allowed)}'
        )
