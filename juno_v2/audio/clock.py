from __future__ import annotations

from dataclasses import dataclass, field

from juno_v2.contracts.audio import AudioFrame


@dataclass(slots=True)
class AudioFrameClock:
    sample_rate_hz: int = 16000
    frame_ms: int = 20
    source: str = "unknown"
    frame_samples: int = field(init=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        frame_samples = int(self.sample_rate_hz * self.frame_ms / 1000)
        if frame_samples <= 0:
            raise ValueError("frame_ms produces zero samples")
        self.frame_samples = frame_samples

    def make_frame(self, index: int, samples) -> AudioFrame:
        if len(samples) != self.frame_samples:
            raise ValueError(f"expected {self.frame_samples} samples, got {len(samples)}")
        start_sample = index * self.frame_samples
        end_sample = start_sample + self.frame_samples
        start_ms = (start_sample / self.sample_rate_hz) * 1000.0
        end_ms = (end_sample / self.sample_rate_hz) * 1000.0
        return AudioFrame(
            index=index,
            sample_rate_hz=self.sample_rate_hz,
            start_sample=start_sample,
            end_sample=end_sample,
            start_ms=start_ms,
            end_ms=end_ms,
            samples=samples,
            source=self.source,
        )
