from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from juno_v2.contracts.tracing import TraceKind
from juno_v2.contracts.workbench import CommitMode, FinalCommitRequest
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.workbench.store import WorkbenchStore

from juno_core_v3.recovery.ledger import RecoveryEntry, RecoveryLedger


@dataclass(slots=True)
class RecoverySession:
    """Output / trust / recovery plane (Phase 5).

    Guarantees we implement now:
    - durable append-only ledger of committed (and staged-fallback) text
    - paste-last transcript (read from ledger)
    - retry without re-speaking (re-append last committed text to buffer)
    - local history enumeration (ledger replay)

    Surfaces still perform actual clipboard/paste UX; the broker exposes the canonical text.
    """

    broker_session_id: str
    ledger: RecoveryLedger
    recorder: TraceRecorder

    def __init__(self, *, broker_session_id: str, recovery_root: str | Path, recorder: TraceRecorder) -> None:
        self.broker_session_id = broker_session_id
        self.recorder = recorder
        self.ledger = RecoveryLedger(recovery_root=Path(recovery_root), broker_session_id=broker_session_id)

    def ingest_from_workbench_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Record post-run state so text is not lost if commit failed or only staged."""
        last_text = str(snapshot.get("last_committed_text") or "")
        utt = snapshot.get("last_committed_utterance_id")
        final_cand = str(snapshot.get("final_candidate_text") or "")
        pending = bool(snapshot.get("pending_commit"))
        conflict = snapshot.get("commit_conflict")

        if last_text.strip():
            self._append(
                kind="committed",
                utterance_id=str(utt) if utt else None,
                text=last_text,
                metadata={"source": "workbench_snapshot", "pending_commit": pending, "commit_conflict": conflict},
            )
            return

        if pending and final_cand.strip():
            # Never-lose-text fallback: final exists but did not become last_committed_text.
            self._append(
                kind="staged_fallback",
                utterance_id=str(utt) if utt else None,
                text=final_cand,
                metadata={"source": "workbench_snapshot", "reason": "pending_final_without_commit_record"},
            )

    def paste_last_transcript(self) -> str | None:
        """Return last recoverable text (committed preferred, else last staged fallback)."""
        committed: str | None = None
        staged: str | None = None
        for e in self.ledger.read_all():
            if e.kind == "committed" and e.text.strip():
                committed = e.text
            if e.kind == "staged_fallback" and e.text.strip():
                staged = e.text
        chosen = committed if committed is not None else staged
        self.recorder.record(
            TraceKind.SYSTEM,
            "broker_recovery_paste_last",
            {"has_text": bool(chosen and chosen.strip()), "prefer_committed": committed is not None},
        )
        return chosen

    def retry_last_commit_append(self, store: WorkbenchStore) -> dict[str, Any]:
        """Re-apply the last committed transcript without new ASR (append to buffer)."""
        last_committed: RecoveryEntry | None = None
        for e in self.ledger.read_all():
            if e.kind == "committed" and e.text.strip():
                last_committed = e
        if last_committed is None:
            self.recorder.record(TraceKind.SYSTEM, "broker_recovery_retry_skipped", {"reason": "no_committed_text"})
            raise ValueError("no committed transcript to retry")

        utt = f"retry_{uuid.uuid4().hex[:12]}"
        req = FinalCommitRequest(
            text=last_committed.text,
            commit_mode=CommitMode.APPEND,
            utterance_id=utt,
        )
        snap = store.commit_final(req)
        self._append(
            kind="retry_applied",
            utterance_id=utt,
            text=last_committed.text,
            metadata={"source": "retry_without_respeaking", "from_utterance": last_committed.utterance_id},
        )
        self.recorder.record(
            TraceKind.SYSTEM,
            "broker_recovery_retry_applied",
            {"utterance_id": utt, "text_length": len(last_committed.text)},
        )
        return snap

    def history(self) -> list[dict[str, Any]]:
        return [
            {
                "ts_unix_ms": e.ts_unix_ms,
                "kind": e.kind,
                "utterance_id": e.utterance_id,
                "text_length": len(e.text),
                "text_preview": e.text[:200],
                "metadata": dict(e.metadata),
            }
            for e in self.ledger.read_all()
        ]

    def record_transform(
        self,
        *,
        selected_text: str,
        replacement_text: str,
        hint: str,
        deterministic: bool,
        degraded: bool = False,
        session_id: str | None = None,
    ) -> None:
        """Record a Transform session output to the ledger for history and replay."""
        self._append(
            kind="transform",
            utterance_id=session_id,
            text=replacement_text,
            metadata={
                "selected_text_preview": selected_text[:200],
                "selected_text_len": len(selected_text),
                "hint": hint,
                "deterministic": deterministic,
                "degraded": degraded,
            },
        )

    def _append(self, *, kind: str, utterance_id: str | None, text: str, metadata: dict[str, Any]) -> None:
        entry = RecoveryEntry(
            ts_unix_ms=int(time.time() * 1000),
            broker_session_id=self.broker_session_id,
            kind=kind,
            utterance_id=utterance_id,
            text=text,
            metadata=metadata,
        )
        self.ledger.append(entry)
        self.recorder.record(
            TraceKind.SYSTEM,
            "broker_recovery_ledger_append",
            {"kind": kind, "utterance_id": utterance_id, "text_length": len(text)},
        )
