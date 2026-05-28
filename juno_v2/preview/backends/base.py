from __future__ import annotations

from typing import Protocol

from juno_v2.contracts.preview import PreviewDecodeRequest, PreviewDecodeResult


class PreviewAsrBackend(Protocol):
    backend_name: str

    def warm(self) -> None:
        raise NotImplementedError

    def decode(self, req: PreviewDecodeRequest) -> PreviewDecodeResult:
        raise NotImplementedError
