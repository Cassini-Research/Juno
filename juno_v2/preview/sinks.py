from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Protocol
from urllib import request

from juno_v2.contracts.preview import PreviewEmission
from juno_v2.contracts.workbench import PartialCommitRequest
from juno_v2.workbench.store import WorkbenchStore


class PreviewSink(Protocol):
    def emit(self, emission: PreviewEmission) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class MemoryPreviewSink:
    emissions: List[PreviewEmission] = field(default_factory=list)
    cleared_count: int = 0

    def emit(self, emission: PreviewEmission) -> None:
        self.emissions.append(emission)

    def clear(self) -> None:
        self.cleared_count += 1


@dataclass(slots=True)
class WorkbenchPreviewSink:
    store: WorkbenchStore
    retain_final_partial: bool = True

    def emit(self, emission: PreviewEmission) -> None:
        text = emission.text if (not emission.is_final or self.retain_final_partial) else ""
        self.store.apply_partial(PartialCommitRequest(text=text))

    def clear(self) -> None:
        self.store.clear_partial()


@dataclass(slots=True)
class RemoteWorkbenchPreviewSink:
    base_url: str
    retain_final_partial: bool = True
    timeout_sec: float = 3.0

    def _post(self, path: str, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with request.urlopen(req, timeout=self.timeout_sec) as resp:  # noqa: S310 - controlled local URL expected
            resp.read()

    def emit(self, emission: PreviewEmission) -> None:
        text = emission.text if (not emission.is_final or self.retain_final_partial) else ""
        self._post("/api/partial", {"text": text})

    def clear(self) -> None:
        self._post("/api/partial/clear", {})
