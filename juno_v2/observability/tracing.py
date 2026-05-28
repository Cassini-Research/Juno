from __future__ import annotations

import json
import time
from collections import deque
import threading
from pathlib import Path
from typing import Any, Deque, Dict, List

from juno_v2.contracts.tracing import TraceEvent, TraceKind, new_trace_id


class TraceRecorder:
    """Append-only JSONL trace writer with optional size-based rotation.

    By default rotation is OFF (``max_log_bytes=0``) so unit tests and short
    sessions keep the original single-file layout. When rotation is enabled,
    each append that would push the primary file past ``max_log_bytes``
    triggers a rename-and-retire cycle:

        <session_id>.jsonl            ← active file
        <session_id>.jsonl.1          ← most-recent retired generation
        <session_id>.jsonl.2          ← next-oldest
        ...
        <session_id>.jsonl.<keep>     ← oldest kept generation

    When the number of retired generations exceeds ``max_log_generations``
    the oldest is deleted. Rotation happens lazily — checked on each
    ``record()`` call — so there is no background thread.
    """

    def __init__(
        self,
        session_id: str,
        log_dir: Path,
        recent_limit: int = 200,
        *,
        max_log_bytes: int = 0,
        max_log_generations: int = 3,
    ) -> None:
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{session_id}.jsonl"
        self.max_log_bytes = max(0, int(max_log_bytes))
        self.max_log_generations = max(1, int(max_log_generations))
        self._seq = 0
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=recent_limit)
        self._lock = threading.RLock()

    def record(self, kind: TraceKind, name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = TraceEvent(
                session_id=self.session_id,
                trace_id=new_trace_id(),
                seq=self._seq,
                ts_unix_ms=int(time.time() * 1000),
                ts_monotonic_ns=time.monotonic_ns(),
                kind=kind,
                name=name,
                payload=payload or {},
            )
            data = event.to_dict()
            self._recent.appendleft(data)
            serialized = json.dumps(data, ensure_ascii=False) + "\n"
            self._maybe_rotate(len(serialized.encode("utf-8")))
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(serialized)
            return data

    def recent_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent)

    def _maybe_rotate(self, pending_bytes: int) -> None:
        """Rotate the primary log file if appending the next event would
        push it past ``max_log_bytes``. Caller must hold ``self._lock``."""
        if self.max_log_bytes <= 0:
            return
        if not self.log_path.exists():
            return
        current_size = self.log_path.stat().st_size
        if current_size + pending_bytes <= self.max_log_bytes:
            return
        # Slide existing generations: .N -> .(N+1), oldest evicted.
        for generation in range(self.max_log_generations, 0, -1):
            src = self.log_path.with_suffix(self.log_path.suffix + f".{generation}")
            if not src.exists():
                continue
            if generation >= self.max_log_generations:
                src.unlink()
                continue
            dst = self.log_path.with_suffix(self.log_path.suffix + f".{generation + 1}")
            src.rename(dst)
        # Retire the current active file as generation .1.
        retired = self.log_path.with_suffix(self.log_path.suffix + ".1")
        self.log_path.rename(retired)
