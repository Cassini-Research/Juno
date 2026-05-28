from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

AudioSamples = npt.NDArray[np.float32]


@dataclass(slots=True)
class AudioFrame:
    """Single mono PCM frame normalized to float32 [-1, 1]."""

    index: int
    sample_rate_hz: int
    start_sample: int
    end_sample: int
    start_ms: float
    end_ms: float
    samples: AudioSamples
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_samples:
            data.pop("samples", None)
        else:
            data["samples"] = self.samples.tolist()
        return data
