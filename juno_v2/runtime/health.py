from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ServiceHealthSnapshot:
    status: str
    mode: str
    session_id: str
    lifecycle: dict = field(default_factory=dict)
    startup_profile: dict = field(default_factory=dict)
    last_fault: dict | None = None
    workbench_url: str | None = None
    updated_at_unix: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'status': self.status,
            'mode': self.mode,
            'session_id': self.session_id,
            'lifecycle': dict(self.lifecycle),
            'startup_profile': dict(self.startup_profile),
            'last_fault': None if self.last_fault is None else dict(self.last_fault),
            'workbench_url': self.workbench_url,
            'updated_at_unix': self.updated_at_unix,
            'metadata': dict(self.metadata),
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
