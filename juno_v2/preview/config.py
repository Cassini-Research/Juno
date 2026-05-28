from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PreviewAsrConfig:
    model_path: Path | str
    backend_name: str = "faster_whisper"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 1
    best_of: int = 1
    condition_on_previous_text: bool = False
    partial_decode_interval_ms: int = 240
    min_decode_audio_ms: int = 300
    retain_final_partial: bool = True
    initial_prompt: str | None = None
    local_http_endpoint: str | None = None
    local_http_timeout_sec: float = 15.0
