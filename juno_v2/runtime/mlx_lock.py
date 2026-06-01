"""Process-global mutex serializing MLX/Metal command-buffer usage,
plus thread-local stream management so MLX inference works on worker
threads spawned by ``runtime.execution.StageExecutor``.

Two concerns layered together:

1. **Mutex (original).** juno_v2 runs preview and final ASR stages on
   separate worker threads. When both stages are MLX-backed (e.g.
   MLX preview + mlx_whisper final), both threads submit Metal
   command buffers to the shared GPU queue. MLX does not serialize
   command encoders across threads, so Metal raises:

       failed assertion _status < MTLCommandBufferStatusCommitted at
       line 323 in -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]

   The plain ``threading.Lock`` here ensures only one MLX backend
   touches Metal at a time. Non-MLX backends (faster-whisper,
   local_http_json) bypass the lock and keep their existing thread-safe
   behavior.

2. **Thread-local stream (added 2026-04-30 after the example user's
   ``RuntimeError: There is no Stream(gpu, 0) in current thread``
   incident at .juno_v2_runtime/incidents/fault-1777462756708-attempt1.json).**
   MLX 0.31.x requires every thread that submits Metal work to enter a
   stream context first. Worker threads spawned by ``StageExecutor``
   never had one — the original implementation only initialized the
   default stream on the main thread, so the first per-utterance
   request from a worker crashed the runtime. The canonical pattern
   from mlx-lm 0.31.x (``mlx_lm/generate.py:226``):

       generation_stream = mx.new_thread_local_stream(mx.default_device())
       with mx.stream(generation_stream):
           ...  # MLX ops here

   We create a single module-level stream lazily on first use and wrap
   the guard's body in ``mx.stream(...)``. Backends call
   ``mlx_decode_guard()`` exactly as before; they automatically pick up
   the stream context. The writer (``juno_v2/writer/backends/mlx_lm.py``)
   doesn't use this guard because ``mlx_lm.generate`` manages its own
   stream internally.

This is a **concurrency safety fix**, not a throughput optimization.
Serializing MLX decodes means preview and final cannot run in parallel
on Apple Silicon, which is a small latency cost compared to the
alternative of the process crashing. For workloads that need parallel
MLX decoding the right answer is to run the stages in separate
processes (e.g. the streaming_local_http_json preview service pattern),
not separate threads inside the same runtime.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Iterator

_mlx_decode_lock = threading.Lock()

_thread_local = threading.local()
_stream_init_failed: bool = False


def _ensure_mlx_thread_stream() -> Any | None:
    """Return the calling thread's MLX stream, creating it on first use.

    Returns ``None`` (and stays None for the rest of the process) if MLX
    is not importable or stream creation fails. Callers fall back to the
    pre-stream behavior — which kept working on the main thread because
    MLX initializes a default stream there at import.
    """
    global _stream_init_failed
    stream = getattr(_thread_local, "stream", None)
    if stream is not None:
        return stream
    if _stream_init_failed:
        return None
    try:
        import mlx.core as mx  # type: ignore

        stream = mx.new_thread_local_stream(mx.default_device())
        _thread_local.stream = stream
        return stream
    except Exception:
        _stream_init_failed = True
        return None


def reset_mlx_lock_wait_ms() -> None:
    """Reset per-thread accumulated wait time for the global MLX lock."""

    _thread_local.lock_wait_ms_total = 0.0
    _thread_local.lock_wait_ms_last = 0.0


def consume_mlx_lock_wait_ms() -> float:
    """Return and reset accumulated lock wait time for this thread."""

    total = float(getattr(_thread_local, "lock_wait_ms_total", 0.0) or 0.0)
    reset_mlx_lock_wait_ms()
    return total


def last_mlx_lock_wait_ms() -> float:
    """Return the most recent single acquire wait for this thread."""

    return float(getattr(_thread_local, "lock_wait_ms_last", 0.0) or 0.0)


@contextlib.contextmanager
def mlx_decode_guard() -> Iterator[None]:
    """Acquire the process-global MLX decode lock + enter the
    module-level thread-local stream for the duration of the with-block.

    Usage inside an MLX backend (unchanged from before the stream fix —
    the stream context is layered transparently)::

        from juno_v2.runtime.mlx_lock import mlx_decode_guard

        def decode(self, req):
            ...
            with mlx_decode_guard():
                result = self._model.transcribe(path)
            ...
    """
    waiting_since = time.perf_counter_ns()
    _mlx_decode_lock.acquire()
    acquired_at = time.perf_counter_ns()
    wait_ms = (acquired_at - waiting_since) / 1_000_000.0
    _thread_local.lock_wait_ms_last = wait_ms
    _thread_local.lock_wait_ms_total = float(getattr(_thread_local, "lock_wait_ms_total", 0.0) or 0.0) + wait_ms
    try:
        stream = _ensure_mlx_thread_stream()
        try:
            if stream is None:
                yield
            else:
                import mlx.core as mx  # type: ignore

                stream_cm = getattr(mx, "stream", None)
                if stream_cm is None:
                    yield
                else:
                    with stream_cm(stream):
                        yield
        except RuntimeError as exc:
            # Convert bare MLX/Metal :class:`RuntimeError` (queue pressure,
            # missing thread-local stream, model swap-in races) into
            # :class:`MLXTransientError` so the supervisor's recoverable
            # exception tuple can match it without us having to add bare
            # :class:`RuntimeError` to the recoverable set (which would
            # mask legitimate bugs). Already-wrapped errors pass through.
            from juno_v2.runtime.faults import MLXTransientError

            if isinstance(exc, MLXTransientError):
                raise
            raise MLXTransientError(str(exc)) from exc
    finally:
        # Force all outstanding Metal command buffers to commit BEFORE
        # releasing the lock so the next thread that acquires cannot
        # see a dirty command-encoder state. mx.synchronize() is the
        # MLX equivalent of torch.cuda.synchronize().
        try:
            import mlx.core as mx  # type: ignore

            mx.synchronize()
        except Exception:
            pass
        _mlx_decode_lock.release()
