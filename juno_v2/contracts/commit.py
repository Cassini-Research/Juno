from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict

from juno_v2.contracts.workbench import ClientSelection, CommitMode


class CommitStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMMITTED = "committed"
    CONFLICT = "conflict"
    ABORTED = "aborted"


@dataclass(slots=True)
class CommitAnchor:
    buffer_text: str
    selection: ClientSelection
    revision: int
    commit_mode: CommitMode

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["commit_mode"] = self.commit_mode.value
        return data


@dataclass(slots=True)
class ActiveCommitSession:
    utterance_id: str
    anchor: CommitAnchor
    latest_partial_text: str = ""
    final_text: str = ""
    # Metadata from the FinalTranscript (avg_logprob, compression_ratio,
    # script_summary, ...). The hallucination guard uses avg_logprob to
    # decide whether to apply the word-repetition heuristic: real
    # speech that happens to repeat a word five times ("Hello hello
    # hello hello hello") is high-confidence and should NOT be flagged,
    # whereas whisper hallucinating the same shape on silent audio has
    # a distinctly lower avg_logprob.
    final_metadata: Dict[str, object] = field(default_factory=dict)
    status: CommitStatus = CommitStatus.ACTIVE
    conflict_reason: str | None = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["status"] = self.status.value
        data["anchor"] = self.anchor.to_dict()
        return data


@dataclass(slots=True)
class CommitDecision:
    utterance_id: str
    committed: bool
    conflict_reason: str | None = None
    commit_mode: CommitMode | None = None
    committed_text: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "utterance_id": self.utterance_id,
            "committed": self.committed,
            "conflict_reason": self.conflict_reason,
            "commit_mode": self.commit_mode.value if self.commit_mode else None,
            "committed_text": self.committed_text,
        }
