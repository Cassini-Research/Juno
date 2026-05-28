from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FinalAsrConfig:
    model_path: str | Path
    backend_name: str = "faster_whisper"
    language: str | None = "en"
    initial_prompt: str | None = None
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 5
    best_of: int = 5
    condition_on_previous_text: bool = False
    local_http_endpoint: str | None = None
    local_http_timeout_sec: float = 30.0
    # HuggingFace repo ID used by backends that download their own weights
    # (e.g. mlx_whisper). When set, the backend falls back to this if the
    # local model_path does not contain valid backend-native weights.
    hf_repo_id: str | None = None
