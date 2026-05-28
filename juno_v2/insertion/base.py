from __future__ import annotations

from typing import Protocol

from juno_v2.contracts.insertion import InsertionRequest


class ActiveAppInserter(Protocol):
    name: str

    def warm(self) -> None: ...
    def capabilities(self) -> dict[str, object]: ...
    def apply_commit(self, req: InsertionRequest) -> None: ...
