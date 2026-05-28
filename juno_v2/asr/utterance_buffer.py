from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from juno_v2.contracts.audio import AudioFrame


@dataclass(slots=True)
class UtteranceAudioBuffer:
    utterance_id: str
    frames: List[AudioFrame] = field(default_factory=list)
    _last_frame_index: int = -1
    _last_partial_decode_end_ms: float = 0.0
    _last_emitted_text: str = ""
    _partial_decode_seq: int = 0

    def seed(self, frames: list[AudioFrame]) -> None:
        for frame in frames:
            self.append(frame)

    def append(self, frame: AudioFrame) -> None:
        if frame.index <= self._last_frame_index:
            return
        self.frames.append(frame)
        self._last_frame_index = frame.index

    @property
    def start_ms(self) -> float:
        return self.frames[0].start_ms if self.frames else 0.0

    @property
    def end_ms(self) -> float:
        return self.frames[-1].end_ms if self.frames else 0.0

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)

    @property
    def partial_decode_seq(self) -> int:
        return self._partial_decode_seq

    @property
    def last_emitted_text(self) -> str:
        return self._last_emitted_text

    def should_decode_partial(self, interval_ms: int, min_audio_ms: int) -> bool:
        if self.duration_ms < min_audio_ms:
            return False
        return (self.end_ms - self._last_partial_decode_end_ms) >= float(interval_ms)

    def mark_partial_decode(self) -> None:
        self._last_partial_decode_end_ms = self.end_ms
        self._partial_decode_seq += 1

    def audio(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([frame.samples for frame in self.frames]).astype(np.float32, copy=False)

    def update_last_emitted_text(self, text: str) -> int:
        prev = self._last_emitted_text
        self._last_emitted_text = text
        return stability_delta_chars(prev, text)


def stability_delta_chars(prev: str, current: str) -> int:
    if prev == current:
        return 0
    limit = min(len(prev), len(current))
    lcp = 0
    while lcp < limit and prev[lcp] == current[lcp]:
        lcp += 1
    return (len(prev) - lcp) + (len(current) - lcp)
