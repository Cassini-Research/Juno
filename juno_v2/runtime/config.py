from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class WorkbenchRuntimeConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    log_dir: Path = Path(".juno_v2_logs") / "workbench"
    recent_event_limit: int = 200
    debug: bool = False
    runtime_dir: Path | None = None
    logs_dir: Path | None = None
