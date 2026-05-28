from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid

from juno_core_v3.contracts.resource_hints import HostResourceHints


class SessionKind(str, Enum):
    """Product session classes: Insert, Transform, or Action.

    ``ACTION`` is reserved for utterances that begin with a Juno wake-word and
    contain a recognised action verb (notes / reminders). Routing to ``ACTION``
    happens *after* ASR (the planner cannot see the transcript), so the value
    is set by the dictation pipeline once ``parse_actions`` confirms a hit.
    """

    INSERT = "insert"
    TRANSFORM = "transform"
    ACTION = "action"


@dataclass(slots=True)
class UserIntentSignals:
    """Rules-first inputs for classification (no ML in Phase 1)."""

    has_selected_text: bool = False
    explicit_transform: bool = False
    explicit_insert: bool = False
    surface_id: str | None = None
    host_hints: HostResourceHints | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_selected_text": self.has_selected_text,
            "explicit_transform": self.explicit_transform,
            "explicit_insert": self.explicit_insert,
            "surface_id": self.surface_id,
            "host_hints": None if self.host_hints is None else self.host_hints.to_dict(),
        }


@dataclass(slots=True)
class BrokerSession:
    """Broker-owned session record (minimal Phase 1 state)."""

    session_id: str
    kind: SessionKind
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(
        kind: SessionKind,
        metadata: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> "BrokerSession":
        """Create a broker session. If ``session_id`` is set (e.g. workbench session), use it for trace/recovery alignment."""
        sid = session_id if session_id else f"bro_{uuid.uuid4().hex[:16]}"
        return BrokerSession(session_id=sid, kind=kind, metadata=dict(metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "kind": self.kind.value,
            "metadata": dict(self.metadata),
        }
