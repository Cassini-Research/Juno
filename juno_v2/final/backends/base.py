from __future__ import annotations

from typing import Protocol

from juno_v2.contracts.final import FinalDecodeRequest, FinalDecodeResult


class FinalAsrBackend(Protocol):
    backend_name: str

    def warm(self) -> None:
        raise NotImplementedError

    def decode(self, req: FinalDecodeRequest) -> FinalDecodeResult:
        raise NotImplementedError
