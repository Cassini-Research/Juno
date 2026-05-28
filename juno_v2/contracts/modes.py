from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class ModeSource(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    CUSTOM = "custom"
    PRESET = "preset"


@dataclass(slots=True)
class ModePolicy:
    """Typed runtime policy for a resolved writing mode."""

    mode_name: str
    base_mode: str
    manual_selectable: bool
    writer_behavior: str
    transform_behavior: str
    command_behavior: str
    itn_policy: str
    punctuation_policy: str
    cleanup_policy: str
    snippet_scope_policy: str
    style_scope_policy: str
    allow_auto_transform: bool
    allow_model_insert_rewrite: bool
    allow_inline_commands: bool
    allow_recent_target_commands: bool
    allow_selection_commands: bool
    command_ambiguity_policy: str
    degradation_behavior: str
    prompt_prefix: str
    post_processors: tuple[str, ...] = field(default_factory=tuple)
    transcript_correction_policy: str = "standard"
    final_formatting_policy: str = "minimal"
    live_correction_policy: str = "stable_span_standard"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["post_processors"] = list(self.post_processors)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModePolicy:
        pp = data.get("post_processors") or ()
        if isinstance(pp, list):
            pp = tuple(str(x) for x in pp)
        return cls(
            mode_name=str(data.get("mode_name", "")),
            base_mode=str(data.get("base_mode", "")),
            manual_selectable=bool(data.get("manual_selectable", False)),
            writer_behavior=str(data.get("writer_behavior", "")),
            transform_behavior=str(data.get("transform_behavior", "")),
            command_behavior=str(data.get("command_behavior", "")),
            itn_policy=str(data.get("itn_policy", "")),
            punctuation_policy=str(data.get("punctuation_policy", "")),
            cleanup_policy=str(data.get("cleanup_policy", "")),
            snippet_scope_policy=str(data.get("snippet_scope_policy", "")),
            style_scope_policy=str(data.get("style_scope_policy", "")),
            allow_auto_transform=bool(data.get("allow_auto_transform", False)),
            allow_model_insert_rewrite=bool(data.get("allow_model_insert_rewrite", False)),
            allow_inline_commands=bool(data.get("allow_inline_commands", True)),
            allow_recent_target_commands=bool(data.get("allow_recent_target_commands", True)),
            allow_selection_commands=bool(data.get("allow_selection_commands", True)),
            command_ambiguity_policy=str(data.get("command_ambiguity_policy", "")),
            degradation_behavior=str(data.get("degradation_behavior", "")),
            prompt_prefix=str(data.get("prompt_prefix", "")),
            post_processors=pp if isinstance(pp, tuple) else (),
            transcript_correction_policy=str(data.get("transcript_correction_policy", "standard")),
            final_formatting_policy=str(data.get("final_formatting_policy", "minimal")),
            live_correction_policy=str(data.get("live_correction_policy", "stable_span_standard")),
        )


@dataclass(slots=True)
class ModeSelection:
    """How the effective mode was chosen for this utterance / transform."""

    effective_mode: str
    mode_source: ModeSource
    manual_mode_name: str | None
    custom_mode_name: str | None
    resolved_from_surface: str | None
    surface_preset_id: str | None = None
    surface_bundle_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_mode": self.effective_mode,
            "mode_source": self.mode_source.value,
            "manual_mode_name": self.manual_mode_name,
            "custom_mode_name": self.custom_mode_name,
            "resolved_from_surface": self.resolved_from_surface,
            "surface_preset_id": self.surface_preset_id,
            "surface_bundle_id": self.surface_bundle_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModeSelection:
        src = data.get("mode_source", "auto")
        try:
            mode_source = ModeSource(str(src))
        except ValueError:
            mode_source = ModeSource.AUTO
        return cls(
            effective_mode=str(data.get("effective_mode", "default_surface")),
            mode_source=mode_source,
            manual_mode_name=data.get("manual_mode_name"),
            custom_mode_name=data.get("custom_mode_name"),
            resolved_from_surface=data.get("resolved_from_surface"),
            surface_preset_id=data.get("surface_preset_id"),
            surface_bundle_id=data.get("surface_bundle_id"),
        )


@dataclass(slots=True)
class CustomModeRecord:
    """Persisted user-defined mode overlay."""

    name: str
    base_mode: str
    description: str = ""
    prompt_prefix: str = ""
    itn_override: str | None = None
    cleanup_override: str | None = None
    style_card_name: str | None = None
    snippet_scope: str | None = None
    command_policy: str | None = None
    auto_transform_id: str | None = None
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomModeRecord:
        return cls(
            name=str(data.get("name", "")),
            base_mode=str(data.get("base_mode", "default_surface")),
            description=str(data.get("description", "")),
            prompt_prefix=str(data.get("prompt_prefix", "")),
            itn_override=data.get("itn_override"),
            cleanup_override=data.get("cleanup_override"),
            style_card_name=data.get("style_card_name"),
            snippet_scope=data.get("snippet_scope"),
            command_policy=data.get("command_policy"),
            auto_transform_id=data.get("auto_transform_id"),
            enabled=bool(data.get("enabled", True)),
        )


CommandPolicyLiteral = Literal["strict", "moderate", "wide", "narrow"]
