from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from juno_core_v3.broker.planner import (
    RouterAwarePlanner,
    RulesFirstPlanner,
)
from juno_core_v3.contracts.session import BrokerSession, SessionKind, UserIntentSignals


# A planner is anything that exposes ``classify(signals) -> SessionKind``.
# Both :class:`RulesFirstPlanner` and :class:`RouterAwarePlanner` satisfy
# this, so the manager can be injected with either without the callers
# caring which flavour is active.
Planner = Union[RulesFirstPlanner, RouterAwarePlanner]


@dataclass(slots=True)
class SessionManager:
    """Owns broker session lifecycle (Phase 1: create + classify only)."""

    planner: Planner = field(default_factory=RulesFirstPlanner)
    _sessions: dict[str, BrokerSession] = field(default_factory=dict)

    def classify_only(self, signals: UserIntentSignals) -> SessionKind:
        """Return session kind without persisting."""
        return self.planner.classify(signals)

    def start_session(self, signals: UserIntentSignals | None = None) -> BrokerSession:
        """Create a new broker session via the configured planner.

        Uses the injected planner (rules-first by default) so that a
        :class:`RouterAwarePlanner` can influence classification when
        signals are ambiguous. Historically this called
        ``classify_session_kind`` directly — that bypassed the router
        and was a latent wiring bug once the router slot landed.
        """
        sig = signals or UserIntentSignals()
        kind = self.planner.classify(sig)
        session = BrokerSession.new(kind, metadata={"signals": sig.to_dict()})
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> BrokerSession | None:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
