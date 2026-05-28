from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TransformSource = Literal["builtin", "custom"]


@dataclass(slots=True)
class CatalogTransform:
    transform_id: str
    display_name: str
    target_types_supported: tuple[str, ...]
    mode_constraints: tuple[str, ...] = field(default_factory=tuple)
    deterministic_preprocessors: tuple[str, ...] = field(default_factory=tuple)
    model_prompt_template: str = ""
    post_processors: tuple[str, ...] = field(default_factory=tuple)
    fallback_behavior: str = "degrade_to_polish"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in (
            "target_types_supported",
            "mode_constraints",
            "deterministic_preprocessors",
            "post_processors",
        ):
            d[k] = list(d[k])
        return d


@dataclass(slots=True)
class CustomTransformRecord:
    name: str
    instruction: str
    base_transform_id: str | None = None
    mode_constraints: tuple[str, ...] = field(default_factory=tuple)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode_constraints"] = list(self.mode_constraints)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomTransformRecord:
        mc = data.get("mode_constraints") or ()
        return cls(
            name=str(data.get("name", "")),
            instruction=str(data.get("instruction", "")),
            base_transform_id=data.get("base_transform_id"),
            mode_constraints=tuple(str(x) for x in mc) if isinstance(mc, (list, tuple)) else (),
            enabled=bool(data.get("enabled", True)),
        )
