from __future__ import annotations

from dataclasses import dataclass

from juno_v2.contracts.workbench import ClientSelection, CommitMode


@dataclass(slots=True)
class InsertionRequest:
    """Commit payload handed from the commit controller to an active-app sink."""

    utterance_id: str
    text: str
    commit_mode: CommitMode
    selection: ClientSelection
    anchor_text: str
