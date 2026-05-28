from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Protocol
from urllib import request

from juno_v2.contracts.final import FinalTranscript
from juno_v2.contracts.workbench import FinalCandidateRequest
from juno_v2.workbench.store import WorkbenchStore


class FinalTranscriptSink(Protocol):
    def emit(self, transcript: FinalTranscript) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class MemoryFinalTranscriptSink:
    transcripts: List[FinalTranscript] = field(default_factory=list)
    cleared_count: int = 0

    def emit(self, transcript: FinalTranscript) -> None:
        self.transcripts.append(transcript)

    def clear(self) -> None:
        self.cleared_count += 1


@dataclass(slots=True)
class WorkbenchFinalCandidateSink:
    store: WorkbenchStore

    def emit(self, transcript: FinalTranscript) -> None:
        self.store.set_final_candidate(FinalCandidateRequest(text=transcript.text))

    def clear(self) -> None:
        self.store.clear_final_candidate()


@dataclass(slots=True)
class RemoteWorkbenchFinalCandidateSink:
    base_url: str
    timeout_sec: float = 3.0

    def _post(self, path: str, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with request.urlopen(req, timeout=self.timeout_sec) as resp:  # noqa: S310
            resp.read()

    def emit(self, transcript: FinalTranscript) -> None:
        self._post("/api/final/candidate", {"text": transcript.text})

    def clear(self) -> None:
        self._post("/api/final/candidate/clear", {})
