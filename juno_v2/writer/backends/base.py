from __future__ import annotations

from abc import ABC, abstractmethod

from juno_v2.contracts.writer import WriterTransformRequest, WriterTransformResult


class WriterBackend(ABC):
    backend_name: str = "unknown"

    @abstractmethod
    def warm(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        raise NotImplementedError
