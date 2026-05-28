from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List

from juno_v2.contracts.audio import AudioFrame


@dataclass(slots=True)
class AudioRingBuffer:
    max_frames: int
    _frames: Deque[AudioFrame] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self._frames: Deque[AudioFrame] = deque(maxlen=self.max_frames)

    def append(self, frame: AudioFrame) -> None:
        self._frames.append(frame)

    def extend(self, frames: Iterable[AudioFrame]) -> None:
        for frame in frames:
            self.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    def snapshot(self) -> List[AudioFrame]:
        return list(self._frames)

    def duration_ms(self) -> float:
        if not self._frames:
            return 0.0
        return self._frames[-1].end_ms - self._frames[0].start_ms

    def __len__(self) -> int:
        return len(self._frames)
