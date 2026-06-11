from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from juno_core_v3.broker.planner import (
    RouterAwarePlanner,
    RulesFirstPlanner,
    classify_session_kind,
    has_rule_hit,
)
from juno_core_v3.contracts.resource_hints import HostResourceHints
from juno_core_v3.contracts.session import SessionKind, UserIntentSignals
from juno_core_v3.policy.surface_gate import SupportLevel, SurfaceCapabilityGate, SurfaceId

# ---- SurfaceCapabilityGate ----------------------------------------------


def test_level_for_defaults() -> None:
    gate = SurfaceCapabilityGate()
    assert gate.level_for(SurfaceId.WORKBENCH_DEV.value) is SupportLevel.GOLD
    assert gate.level_for(SurfaceId.TERMINAL_TEST.value) is SupportLevel.GOLD
    assert gate.level_for(SurfaceId.MAC_OVERLAY.value) is SupportLevel.GOLD
    assert gate.level_for(SurfaceId.TRANSFORM_SHORTCUT.value) is SupportLevel.GOLD
    assert gate.level_for(SurfaceId.IPHONE_APP.value) is SupportLevel.YELLOW
    assert gate.level_for(SurfaceId.IPHONE_KEYBOARD.value) is SupportLevel.RED
    assert gate.level_for(SurfaceId.UNKNOWN.value) is SupportLevel.RED
    # Unmapped surfaces fall back to RED.
    assert gate.level_for("some_future_surface") is SupportLevel.RED


def test_allows_dictation_red_is_blocked() -> None:
    gate = SurfaceCapabilityGate()
    assert gate.allows_dictation(SurfaceId.MAC_OVERLAY.value) is True
    assert gate.allows_dictation(SurfaceId.IPHONE_APP.value) is True
    assert gate.allows_dictation(SurfaceId.IPHONE_KEYBOARD.value) is False
    assert gate.allows_dictation("unmapped") is False


def test_overrides_replace_default_levels() -> None:
    gate = SurfaceCapabilityGate(
        overrides={
            SurfaceId.IPHONE_KEYBOARD.value: SupportLevel.YELLOW,
            "custom_surface": SupportLevel.GOLD,
        }
    )
    assert gate.level_for(SurfaceId.IPHONE_KEYBOARD.value) is SupportLevel.YELLOW
    assert gate.allows_dictation(SurfaceId.IPHONE_KEYBOARD.value) is True
    assert gate.level_for("custom_surface") is SupportLevel.GOLD
    # Untouched defaults stay intact.
    assert gate.level_for(SurfaceId.MAC_OVERLAY.value) is SupportLevel.GOLD


def test_policy_map_returns_plain_strings() -> None:
    gate = SurfaceCapabilityGate()
    pm = gate.policy_map()
    assert pm[SurfaceId.WORKBENCH_DEV.value] == "gold"
    assert pm[SurfaceId.IPHONE_APP.value] == "yellow"
    assert pm[SurfaceId.UNKNOWN.value] == "red"
    assert all(isinstance(v, str) for v in pm.values())


def test_effective_level_no_hints_returns_base() -> None:
    gate = SurfaceCapabilityGate()
    assert gate.effective_level(SurfaceId.MAC_OVERLAY.value, None) is SupportLevel.GOLD
    nominal = HostResourceHints(memory_pressure="nominal", thermal_pressure="nominal")
    assert gate.effective_level(SurfaceId.MAC_OVERLAY.value, nominal) is SupportLevel.GOLD


@pytest.mark.parametrize("pressure_field", ["memory_pressure", "thermal_pressure"])
def test_effective_level_critical_pressure_degrades_one_step(pressure_field: str) -> None:
    gate = SurfaceCapabilityGate()
    hints = HostResourceHints(**{pressure_field: "critical"})
    assert gate.effective_level(SurfaceId.MAC_OVERLAY.value, hints) is SupportLevel.YELLOW
    assert gate.effective_level(SurfaceId.IPHONE_APP.value, hints) is SupportLevel.RED
    # RED stays RED (no further degradation).
    assert gate.effective_level(SurfaceId.UNKNOWN.value, hints) is SupportLevel.RED


def test_effective_level_memory_warning_plus_battery_low_degrades_gold_only() -> None:
    gate = SurfaceCapabilityGate()
    hints = HostResourceHints(memory_pressure="warning", battery_low=True)
    assert gate.effective_level(SurfaceId.MAC_OVERLAY.value, hints) is SupportLevel.YELLOW
    # YELLOW is not degraded by the warning+battery combination.
    assert gate.effective_level(SurfaceId.IPHONE_APP.value, hints) is SupportLevel.YELLOW


def test_effective_level_warning_without_battery_low_is_noop() -> None:
    gate = SurfaceCapabilityGate()
    assert (
        gate.effective_level(
            SurfaceId.MAC_OVERLAY.value, HostResourceHints(memory_pressure="warning")
        )
        is SupportLevel.GOLD
    )
    # Thermal warning alone (no critical) does not degrade either.
    assert (
        gate.effective_level(
            SurfaceId.MAC_OVERLAY.value,
            HostResourceHints(thermal_pressure="warning", battery_low=True),
        )
        is SupportLevel.GOLD
    )


# ---- classify_session_kind / has_rule_hit --------------------------------


def test_classify_explicit_transform_wins() -> None:
    signals = UserIntentSignals(explicit_transform=True, explicit_insert=True, has_selected_text=True)
    assert classify_session_kind(signals) is SessionKind.TRANSFORM


def test_classify_explicit_insert_beats_selection() -> None:
    signals = UserIntentSignals(explicit_insert=True, has_selected_text=True)
    assert classify_session_kind(signals) is SessionKind.INSERT


def test_classify_selected_text_implies_transform() -> None:
    assert classify_session_kind(UserIntentSignals(has_selected_text=True)) is SessionKind.TRANSFORM


def test_classify_default_is_insert() -> None:
    assert classify_session_kind(UserIntentSignals()) is SessionKind.INSERT


def test_rules_first_planner_delegates_to_classify() -> None:
    planner = RulesFirstPlanner()
    assert planner.classify(UserIntentSignals(explicit_transform=True)) is SessionKind.TRANSFORM
    assert planner.classify(UserIntentSignals()) is SessionKind.INSERT


def test_has_rule_hit() -> None:
    assert has_rule_hit(UserIntentSignals(explicit_transform=True)) is True
    assert has_rule_hit(UserIntentSignals(explicit_insert=True)) is True
    assert has_rule_hit(UserIntentSignals(has_selected_text=True)) is True
    assert has_rule_hit(UserIntentSignals()) is False
    assert has_rule_hit(UserIntentSignals(surface_id="mac_overlay")) is False


# ---- RouterAwarePlanner ----------------------------------------------------


@dataclass
class FakeRouter:
    outcome: SessionKind | None = None
    raises: bool = False
    calls: list[UserIntentSignals] = field(default_factory=list)

    def classify(
        self, signals: UserIntentSignals, *, rules_fallback: SessionKind
    ) -> SessionKind | None:
        self.calls.append(signals)
        if self.raises:
            raise RuntimeError("router exploded")
        return self.outcome


def test_router_aware_planner_without_router_uses_rules() -> None:
    planner = RouterAwarePlanner()
    assert planner.classify(UserIntentSignals()) is SessionKind.INSERT
    assert planner.classify(UserIntentSignals(has_selected_text=True)) is SessionKind.TRANSFORM


def test_rules_win_over_router_when_rule_hit() -> None:
    router = FakeRouter(outcome=SessionKind.INSERT)
    planner = RouterAwarePlanner(router=router)
    result = planner.classify(UserIntentSignals(explicit_transform=True))
    assert result is SessionKind.TRANSFORM
    # The router must not even be consulted on a rule hit.
    assert router.calls == []


def test_router_consulted_only_when_no_rule_hit() -> None:
    router = FakeRouter(outcome=SessionKind.TRANSFORM)
    planner = RouterAwarePlanner(router=router)
    result = planner.classify(UserIntentSignals())
    assert result is SessionKind.TRANSFORM
    assert len(router.calls) == 1


def test_router_returning_none_or_insert_falls_back_to_rules() -> None:
    for outcome in (None, SessionKind.INSERT):
        router = FakeRouter(outcome=outcome)
        planner = RouterAwarePlanner(router=router)
        assert planner.classify(UserIntentSignals()) is SessionKind.INSERT
        assert len(router.calls) == 1


def test_router_exception_falls_back_to_rules_and_counts_error() -> None:
    router = FakeRouter(raises=True)
    planner = RouterAwarePlanner(router=router)
    assert planner.router_errors == 0
    assert planner.classify(UserIntentSignals()) is SessionKind.INSERT
    assert planner.router_errors == 1
    assert planner.classify(UserIntentSignals()) is SessionKind.INSERT
    assert planner.router_errors == 2


def test_router_protocol_runtime_checkable() -> None:
    from juno_core_v3.broker.planner import RouterModel

    assert isinstance(FakeRouter(), RouterModel)
