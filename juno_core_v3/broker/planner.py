from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from juno_core_v3.contracts.session import SessionKind, UserIntentSignals


@dataclass(slots=True)
class RulesFirstPlanner:
    """Deterministic routing only (guardrail: no router model in Phase 1)."""

    def classify(self, signals: UserIntentSignals) -> SessionKind:
        return classify_session_kind(signals)


def classify_session_kind(signals: UserIntentSignals) -> SessionKind:
    """Rules-first session classification (Insert / Transform).

    Precedence:
    1. explicit_transform -> TRANSFORM
    2. explicit_insert -> INSERT
    3. has_selected_text -> TRANSFORM (selected-text workflow)
    4. default -> INSERT
    """
    if signals.explicit_transform:
        return SessionKind.TRANSFORM
    if signals.explicit_insert:
        return SessionKind.INSERT
    if signals.has_selected_text:
        return SessionKind.TRANSFORM
    return SessionKind.INSERT


def has_rule_hit(signals: UserIntentSignals) -> bool:
    """Return True iff the rules can classify ``signals`` without guessing.

    Used by :class:`RouterAwarePlanner` to decide whether to consult the
    optional tiny-router model. When *any* signal fires, the rules are
    authoritative — the router adds no value and would introduce
    non-determinism into a path that is currently fully test-covered.
    """
    return bool(
        signals.explicit_transform
        or signals.explicit_insert
        or signals.has_selected_text
    )


@runtime_checkable
class RouterModel(Protocol):
    """Optional tiny-router contract.

    Implementations receive the user signals plus the rules-first
    fallback and must return either a ``SessionKind`` with enough
    confidence to override silence, or ``None`` to defer back to the
    rules. This signature keeps the rules-first guardrail intact:
    the router cannot *downgrade* a clear rules outcome; it can only
    fill in when the rules would otherwise default to ``INSERT``
    with no positive signal.
    """

    def classify(
        self,
        signals: UserIntentSignals,
        *,
        rules_fallback: SessionKind,
    ) -> SessionKind | None: ...


@dataclass(slots=True)
class RouterAwarePlanner:
    """Rules-first planner that consults an optional tiny-router.

    Behaviour:

    * If the rules have a positive signal (``has_rule_hit``), the rules
      win — the router is **not** consulted. This preserves the Phase 1
      guardrail and keeps deterministic-signal paths free of model
      influence.
    * Otherwise (no signals, default ``INSERT`` path) and a router is
      configured, we ask the router. If it returns a ``SessionKind``
      that is not ``INSERT``, we accept that as the planner outcome.
      Returning ``INSERT`` or ``None`` falls back to the rules.
    * Router exceptions are swallowed and tracked — a flaky router
      must never make the product worse than rules-only.

    ``router_errors`` exposes a small counter so health/metrics can
    promote the router back to CANDIDATE after repeated failures.
    """

    router: RouterModel | None = None
    router_errors: int = field(default=0)

    def classify(self, signals: UserIntentSignals) -> SessionKind:
        rules_outcome = classify_session_kind(signals)
        if self.router is None or has_rule_hit(signals):
            return rules_outcome
        try:
            router_outcome = self.router.classify(
                signals, rules_fallback=rules_outcome
            )
        except Exception:  # noqa: BLE001 — must never break the planner
            self.router_errors += 1
            return rules_outcome
        if router_outcome is None or router_outcome == SessionKind.INSERT:
            return rules_outcome
        return router_outcome
