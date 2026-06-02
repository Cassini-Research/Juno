from __future__ import annotations

import time
import uuid

# Re-export from contracts so that runtime.ids remains importable as a
# backward-compatible alias, while the canonical definition lives in the
# contracts layer (which has no upward dependencies).
from juno_v2.contracts.tracing import new_trace_id as new_trace_id  # noqa: F401


def new_session_id(prefix: str = "session") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
