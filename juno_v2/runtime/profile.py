from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean


@dataclass(slots=True)
class StageProfile:
    name: str
    duration_ms: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'duration_ms': self.duration_ms,
            'metadata': dict(self.metadata),
        }


@dataclass(slots=True)
class StartupProfile:
    stages: list[StageProfile] = field(default_factory=list)
    started_at_unix: float = field(default_factory=time.time)

    @contextmanager
    def stage(self, name: str, **metadata):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            self.stages.append(StageProfile(
                name=name,
                duration_ms=(end - start) / 1_000_000.0,
                metadata=metadata,
            ))

    def add_stage(self, name: str, duration_ms: float, **metadata) -> None:
        self.stages.append(StageProfile(name=name, duration_ms=duration_ms, metadata=metadata))

    def to_dict(self) -> dict:
        durations = [stage.duration_ms for stage in self.stages]
        return {
            'started_at_unix': self.started_at_unix,
            'stage_count': len(self.stages),
            'total_duration_ms': sum(durations),
            'average_stage_ms': mean(durations) if durations else None,
            'stages': [stage.to_dict() for stage in self.stages],
        }
