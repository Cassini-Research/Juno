from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class CommitMode(str, Enum):
    INSERT_AT_CARET = "insert_at_caret"
    REPLACE_SELECTION = "replace_selection"
    APPEND = "append"
    REPLACE_ALL = "replace_all"


@dataclass(slots=True)
class ClientSelection:
    start: int = 0
    end: int = 0

    def normalized(self, text_len: int) -> "ClientSelection":
        start = max(0, min(self.start, text_len))
        end = max(0, min(self.end, text_len))
        if start > end:
            start, end = end, start
        return ClientSelection(start=start, end=end)

    def to_dict(self) -> Dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(slots=True)
class WorkbenchState:
    buffer_text: str = ""
    app_name: str | None = None
    app_bundle_id: str | None = None
    window_title: str | None = None
    clipboard_text: str = ""
    partial_text: str = ""
    final_candidate_text: str = ""
    selection: ClientSelection = field(default_factory=ClientSelection)
    revision: int = 0
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    active_utterance_id: str | None = None
    anchor_revision: int | None = None
    active_commit_mode: str | None = None
    pending_commit: bool = False
    commit_conflict: str | None = None
    last_committed_utterance_id: str | None = None
    last_committed_text: str = ""
    last_committed_start: int | None = None
    last_committed_end: int | None = None
    writer_mode: str = "default_surface"
    manual_writer_mode: str | None = None
    custom_writer_mode: str | None = None
    last_writer_action: str | None = None
    requested_language: str | None = None
    observed_language: str | None = None
    language_policy: str | None = None
    runtime_status: str = "idle"
    runtime_mode: str | None = None
    current_phase: str = "idle"
    phase_detail: Dict[str, Any] = field(default_factory=dict)
    live_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class SyncClientStateRequest:
    buffer_text: str
    selection_start: int = 0
    selection_end: int = 0
    app_name: str | None = None
    app_bundle_id: str | None = None
    window_title: str | None = None
    clipboard_text: str = ""

    def selection(self) -> ClientSelection:
        return ClientSelection(start=self.selection_start, end=self.selection_end)


@dataclass(slots=True)
class PartialCommitRequest:
    text: str


@dataclass(slots=True)
class FinalCandidateRequest:
    text: str


@dataclass(slots=True)
class FinalCommitRequest:
    text: str
    commit_mode: CommitMode = CommitMode.INSERT_AT_CARET
    selection_start: int | None = None
    selection_end: int | None = None
    utterance_id: str | None = None

    def selection_override(self) -> ClientSelection | None:
        if self.selection_start is None or self.selection_end is None:
            return None
        return ClientSelection(start=self.selection_start, end=self.selection_end)


@dataclass(slots=True)
class ResetRequest:
    keep_buffer: bool = False
