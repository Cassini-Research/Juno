from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar('T')


@dataclass(slots=True)
class StageResult(Generic[T]):
    result: T
    queue_wait_ms: float
    worker_service_ms: float


@dataclass(slots=True)
class StageTask(Generic[T]):
    stage: str
    utterance_id: str
    submitted_ns: int
    future: Future[StageResult[T]]


class StageExecutor:
    def __init__(self, stage: str, *, max_workers: int = 1) -> None:
        self.stage = stage
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f'juno_v2_{stage}')

    def submit(self, utterance_id: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> StageTask[T]:
        submitted_ns = time.perf_counter_ns()

        def _wrapped() -> StageResult[T]:
            started_ns = time.perf_counter_ns()
            result = fn(*args, **kwargs)
            finished_ns = time.perf_counter_ns()
            return StageResult(
                result=result,
                queue_wait_ms=(started_ns - submitted_ns) / 1_000_000.0,
                worker_service_ms=(finished_ns - started_ns) / 1_000_000.0,
            )

        return StageTask(
            stage=self.stage,
            utterance_id=utterance_id,
            submitted_ns=submitted_ns,
            future=self._executor.submit(_wrapped),
        )

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
