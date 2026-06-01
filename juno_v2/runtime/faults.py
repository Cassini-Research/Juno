from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


class MLXTransientError(RuntimeError):
    """Wrapper for transient MLX/Metal failures (queue pressure, model
    swap-in races, ``RuntimeError: There is no Stream(gpu, N) in current
    thread`` between worker-thread reuses, etc.).

    MLX raises bare :class:`RuntimeError` for these conditions, with no
    narrower exception type to discriminate on. Adding bare
    :class:`RuntimeError` to :data:`RECOVERABLE_EXCEPTIONS` would mask
    legitimate bugs across the rest of the codebase, so we wrap MLX
    decode/generate calls at the MLX-lock boundary (see
    :func:`juno_v2.runtime.mlx_lock.mlx_decode_guard`) and convert the
    bare :class:`RuntimeError` to this narrower type. The supervisor
    treats it as recoverable: up to ``max_restarts`` retries with
    exponential backoff. Persistent failures still terminate normally.
    """


# Recoverable exceptions trigger the supervisor's restart loop.
#
# MLX-side: MLX raises bare :class:`RuntimeError` on transient device-state
# failures (Metal queue pressure, model swap-in races between worker
# threads, GPU stream lifecycle issues observed in the example user's
# 2026-04-30 incident bundle). Wrapping at the mlx_decode_guard boundary
# converts these to :class:`MLXTransientError` so we tolerate up to
# ``max_restarts`` with backoff; persistent failures (e.g. a misconfigured
# model path) escalate normally once attempts are exhausted.
RECOVERABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
    MLXTransientError,
)


@dataclass(slots=True)
class ServiceFault:
    stage: str
    exception_type: str
    message: str
    attempt: int
    recoverable: bool
    timestamp_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'stage': self.stage,
            'exception_type': self.exception_type,
            'message': self.message,
            'attempt': self.attempt,
            'recoverable': self.recoverable,
            'timestamp_unix': self.timestamp_unix,
        }


@dataclass(slots=True)
class FaultPolicy:
    max_restarts: int = 1

    def classify(self, exc: BaseException, *, stage: str, attempt: int) -> ServiceFault:
        recoverable = isinstance(exc, RECOVERABLE_EXCEPTIONS)
        return ServiceFault(
            stage=stage,
            exception_type=type(exc).__name__,
            message=str(exc),
            attempt=attempt,
            recoverable=recoverable,
        )

    def should_retry(self, fault: ServiceFault) -> bool:
        return fault.recoverable and fault.attempt <= self.max_restarts


class FaultJournal:
    def __init__(self, incidents_dir: Path) -> None:
        self.incidents_dir = incidents_dir
        self.incidents_dir.mkdir(parents=True, exist_ok=True)

    def record(self, fault: ServiceFault) -> Path:
        filename = f"fault-{int(fault.timestamp_unix * 1000)}-attempt{fault.attempt}.json"
        path = self.incidents_dir / filename
        path.write_text(json.dumps(fault.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        return path
