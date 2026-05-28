from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict


def new_trace_id() -> str:
    """Return a fresh random trace identifier (32-char hex UUID)."""
    return uuid.uuid4().hex


class TraceKind(str, Enum):
    SYSTEM = "system"
    WORKBENCH = "workbench"
    UI = "ui"
    API = "api"
    ASR_PREVIEW = "asr_preview"
    ASR_FINAL = "asr_final"
    COMMIT = "commit"
    CONTEXT = "context"
    MEMORY = "memory"
    WRITER = "writer"
    METRICS = "metrics"


@dataclass(slots=True)
class TraceEvent:
    session_id: str
    trace_id: str
    seq: int
    ts_unix_ms: int
    ts_monotonic_ns: int
    kind: TraceKind
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data
