from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ScreenFallbackPolicy(str, Enum):
    """Permission-gated, last-resort screen-derived fallback contract (North Star).

    Phase 3 defines the contract but does not implement OCR/capture.
    """

    DISABLED = "disabled"
    PERMISSION_GATED_LAST_RESORT = "permission_gated_last_resort"


@dataclass(slots=True)
class ScreenFallbackRequest:
    """A tightly scoped request, intended to be satisfied by a surface (not the broker core)."""

    reason: str
    surface_id: str
    max_chars: int = 2_000
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ScreenFallbackResult:
    text: str
    allowed: bool
    metadata: dict[str, Any] | None = None


class ScreenFallbackProvider:
    """Interface: surfaces may implement this later (Mac overlay etc)."""

    def capture(self, req: ScreenFallbackRequest) -> ScreenFallbackResult:  # pragma: no cover
        raise NotImplementedError

