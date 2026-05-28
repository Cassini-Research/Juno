from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from juno_core_v3.contracts.session import SessionKind, UserIntentSignals
from juno_core_v3.policy.surface_gate import SurfaceCapabilityGate, SurfaceId, SupportLevel


class PolicyDecisionMetadata(TypedDict, total=False):
    surface_id: str
    static_level: str


class PolicyDecisionPayload(TypedDict):
    ok: bool
    support_level: str
    session_kind: str
    blocked_reason: str | None
    degraded_features: list[str]
    metadata: PolicyDecisionMetadata


@dataclass(slots=True)
class PolicyDecision:
    """Result of broker policy evaluation (surface + session kind + resource hints)."""

    ok: bool
    support_level: SupportLevel
    session_kind: SessionKind
    blocked_reason: str | None = None
    degraded_features: tuple[str, ...] = ()
    metadata: PolicyDecisionMetadata = field(default_factory=dict)

    def to_dict(self) -> PolicyDecisionPayload:
        return {
            "ok": self.ok,
            "support_level": self.support_level.value,
            "session_kind": self.session_kind.value,
            "blocked_reason": self.blocked_reason,
            "degraded_features": list(self.degraded_features),
            "metadata": dict(self.metadata),
        }


class BrokerPolicyEngine:
    """Rules-first policy: composes surface gate with session-kind and resource rules."""

    def __init__(self, gate: SurfaceCapabilityGate | None = None) -> None:
        self._gate = gate or SurfaceCapabilityGate()

    def decide(self, signals: UserIntentSignals, session_kind: SessionKind) -> PolicyDecision:
        sid = signals.surface_id or SurfaceId.UNKNOWN.value
        hints = signals.host_hints
        level = self._gate.effective_level(sid, hints)

        meta: PolicyDecisionMetadata = {"surface_id": sid, "static_level": self._gate.level_for(sid).value}

        if level == SupportLevel.RED:
            return PolicyDecision(
                ok=False,
                support_level=level,
                session_kind=session_kind,
                blocked_reason="surface_or_resources_red",
                metadata=meta,
            )

        degraded: list[str] = []
        if level == SupportLevel.YELLOW:
            degraded.append("reduce_optional_lanes")
            if session_kind == SessionKind.TRANSFORM:
                degraded.append("writer_lane_budget_tight")

        # Terminal / CLI surfaces stay on the supported path, but use safer
        # transform defaults even on GOLD. They often carry shell fragments and
        # commands where free-form rewrites are more dangerous than helpful.
        if sid == SurfaceId.TERMINAL_TEST.value and session_kind == SessionKind.TRANSFORM and level != SupportLevel.RED:
            degraded.append("terminal_safe_transform")

        return PolicyDecision(
            ok=True,
            support_level=level,
            session_kind=session_kind,
            degraded_features=tuple(degraded),
            metadata=meta,
        )
