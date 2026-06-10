from juno_v2.turn_plan.planner import (
    TurnPlanPacket,
    TurnPlanResult,
    TurnPlanner,
    fallback_structural_turn_plan,
)
from juno_v2.turn_plan.renderer import RenderResult, render_turn_plan
from juno_v2.turn_plan.validators import PlanValidation, validate_turn_plan

__all__ = [
    "PlanValidation",
    "RenderResult",
    "TurnPlanPacket",
    "TurnPlanResult",
    "TurnPlanner",
    "actions_from_turn_plan",
    "fallback_structural_turn_plan",
    "render_turn_plan",
    "validate_turn_plan",
]


def __getattr__(name: str):
    if name == "actions_from_turn_plan":
        from juno_v2.turn_plan.actions import actions_from_turn_plan

        return actions_from_turn_plan
    raise AttributeError(name)
