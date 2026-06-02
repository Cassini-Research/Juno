from __future__ import annotations

import heapq
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from juno_v2.runtime.mlx_lock import consume_mlx_lock_wait_ms, reset_mlx_lock_wait_ms

T = TypeVar("T")


class InferenceJobCancelled(RuntimeError):
    """Raised when a queued or stale inference job is intentionally dropped."""


@dataclass(slots=True)
class InferenceJobResult:
    result: Any
    stage: str
    utterance_id: str
    queue_wait_ms: float
    worker_service_ms: float
    mlx_lock_wait_ms: float
    scheduler_enqueued_ns: int
    scheduler_started_ns: int
    scheduler_finished_ns: int


class _Job:
    __slots__ = (
        "priority",
        "sequence",
        "stage",
        "utterance_id",
        "live_generation",
        "fn",
        "future",
        "submitted_ns",
        "cancelled",
    )

    def __init__(
        self,
        *,
        priority: int,
        sequence: int,
        stage: str,
        utterance_id: str,
        live_generation: int,
        fn: Callable[[], Any],
        future: Future[InferenceJobResult],
        submitted_ns: int,
    ) -> None:
        self.priority = priority
        self.sequence = sequence
        self.stage = stage
        self.utterance_id = utterance_id
        self.live_generation = live_generation
        self.fn = fn
        self.future = future
        self.submitted_ns = submitted_ns
        self.cancelled = False

    def __lt__(self, other: "_Job") -> bool:
        return (self.priority, self.sequence) < (other.priority, other.sequence)


class InferenceScheduler:
    """Small priority scheduler for broker-side MLX-bound dictation work.

    It does not attempt unsafe mid-MLX preemption. Instead it cancels pending
    live work, coalesces stale live corrections, and guarantees that final
    stop delivery is the next job dispatched once any already-running job
    leaves the model boundary.
    """

    FINAL_PRIORITY = 0
    PREVIEW_PRIORITY = 10
    LIVE_PRIORITY = 20
    BACKGROUND_PRIORITY = 100

    def __init__(self, *, name: str = "juno-inference") -> None:
        self._name = name
        self._cv = threading.Condition()
        self._queue: list[_Job] = []
        self._sequence = 0
        self._closed = False
        self._live_pending_by_utterance: dict[str, _Job] = {}
        self._live_generation_by_utterance: dict[str, int] = {}
        self._stats = {
            "live_jobs_canceled": 0,
            "live_jobs_stale_dropped": 0,
            "live_jobs_enqueued": 0,
            "final_jobs_enqueued": 0,
        }
        self._worker = threading.Thread(target=self._run, name=name, daemon=True)
        self._worker.start()

    def submit(
        self,
        *,
        stage: str,
        utterance_id: str | None,
        priority: int,
        fn: Callable[[], T],
    ) -> Future[InferenceJobResult]:
        uid = str(utterance_id or "")
        future: Future[InferenceJobResult] = Future()
        submitted_ns = time.perf_counter_ns()
        with self._cv:
            if self._closed:
                future.set_exception(RuntimeError("inference scheduler is closed"))
                return future
            if stage == "final_stop_delivery" and uid:
                self._invalidate_live_locked(uid)
                self._stats["final_jobs_enqueued"] += 1
            live_generation = self._live_generation_by_utterance.get(uid, 0)
            self._sequence += 1
            job = _Job(
                priority=int(priority),
                sequence=self._sequence,
                stage=stage,
                utterance_id=uid,
                live_generation=live_generation,
                fn=fn,
                future=future,
                submitted_ns=submitted_ns,
            )
            if stage == "live_adjudication" and uid:
                old = self._live_pending_by_utterance.get(uid)
                if old is not None and not old.future.done():
                    old.cancelled = True
                    old.future.set_exception(InferenceJobCancelled("coalesced_live_adjudication"))
                    self._stats["live_jobs_canceled"] += 1
                self._live_pending_by_utterance[uid] = job
                self._stats["live_jobs_enqueued"] += 1
            heapq.heappush(self._queue, job)
            self._cv.notify()
        return future

    def stats(self) -> dict[str, int]:
        with self._cv:
            live_pending = sum(1 for job in self._live_pending_by_utterance.values() if not job.future.done())
            queued = sum(1 for job in self._queue if not job.cancelled)
            return {
                **self._stats,
                "live_jobs_in_flight": 1 if getattr(self, "_current_live", False) else 0,
                "live_jobs_pending": live_pending,
                "jobs_queued": queued,
            }

    def shutdown(self, *, wait: bool = True) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if wait:
            self._worker.join(timeout=2.0)

    def _invalidate_live_locked(self, uid: str) -> None:
        self._live_generation_by_utterance[uid] = self._live_generation_by_utterance.get(uid, 0) + 1
        old = self._live_pending_by_utterance.pop(uid, None)
        if old is not None and not old.future.done():
            old.cancelled = True
            old.future.set_exception(InferenceJobCancelled("final_delivery_superseded_live"))
            self._stats["live_jobs_canceled"] += 1

    def _run(self) -> None:
        self._current_live = False
        while True:
            with self._cv:
                while not self._queue and not self._closed:
                    self._cv.wait()
                if self._closed and not self._queue:
                    return
                job = heapq.heappop(self._queue)
                if job.stage == "live_adjudication" and self._live_pending_by_utterance.get(job.utterance_id) is job:
                    self._live_pending_by_utterance.pop(job.utterance_id, None)
            if job.cancelled or job.future.cancelled() or job.future.done():
                continue
            if self._is_stale_live(job):
                self._drop_stale(job, "stale_live_adjudication")
                continue
            started_ns = time.perf_counter_ns()
            try:
                self._current_live = job.stage == "live_adjudication"
                reset_mlx_lock_wait_ms()
                result = job.fn()
                lock_wait_ms = consume_mlx_lock_wait_ms()
                finished_ns = time.perf_counter_ns()
                if self._is_stale_live(job):
                    self._drop_stale(job, "final_delivery_superseded_live")
                    continue
                if not job.future.done():
                    job.future.set_result(
                        InferenceJobResult(
                            result=result,
                            stage=job.stage,
                            utterance_id=job.utterance_id,
                            queue_wait_ms=(started_ns - job.submitted_ns) / 1_000_000.0,
                            worker_service_ms=(finished_ns - started_ns) / 1_000_000.0,
                            mlx_lock_wait_ms=lock_wait_ms,
                            scheduler_enqueued_ns=job.submitted_ns,
                            scheduler_started_ns=started_ns,
                            scheduler_finished_ns=finished_ns,
                        )
                    )
            except BaseException as exc:  # noqa: BLE001
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self._current_live = False

    def _is_stale_live(self, job: _Job) -> bool:
        if job.stage != "live_adjudication" or not job.utterance_id:
            return False
        with self._cv:
            return self._live_generation_by_utterance.get(job.utterance_id, 0) != job.live_generation

    def _drop_stale(self, job: _Job, reason: str) -> None:
        with self._cv:
            self._stats["live_jobs_stale_dropped"] += 1
        if not job.future.done():
            job.future.set_exception(InferenceJobCancelled(reason))
