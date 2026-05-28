from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from juno_v2.contracts.modes import CustomModeRecord, ModePolicy, ModeSelection, ModeSource
from juno_v2.modes.defaults import BUILTIN_MODES


def mode_policy_for(mode_name: str, *, base_only: bool = False) -> ModePolicy:
    """Return policy for *mode_name* (built-in id or legacy alias)."""
    key = (mode_name or "default_surface").strip()
    if key in BUILTIN_MODES:
        p = BUILTIN_MODES[key]
        return replace(p, mode_name=key) if base_only else deepcopy(p)
    # Legacy writer enum strings still in flight
    legacy = {
        "technical_precise": "default_surface",
    }
    mapped = legacy.get(key, "default_surface")
    p = BUILTIN_MODES[mapped]
    return replace(p, mode_name=key, base_mode=mapped)


def resolve_mode_selection(
    *,
    manual_mode_name: str | None,
    custom_mode_name: str | None,
    custom_record: CustomModeRecord | None,
    surface_hint: str | None = None,
) -> tuple[ModeSelection, ModePolicy]:
    """Resolve effective mode and materialized policy for one utterance."""
    if manual_mode_name:
        eff = manual_mode_name.strip()
        pol = mode_policy_for(eff)
        pol = replace(pol, mode_name=eff, base_mode=eff)
        sel = ModeSelection(
            effective_mode=eff,
            mode_source=ModeSource.MANUAL,
            manual_mode_name=eff,
            custom_mode_name=None,
            resolved_from_surface=surface_hint,
        )
        return sel, pol
    if custom_mode_name and custom_record is not None and custom_record.enabled:
        base = (custom_record.base_mode or "default_surface").strip()
        base_pol = mode_policy_for(base)
        pol = apply_custom_overrides(base_pol, custom_record, display_name=custom_mode_name.strip())
        sel = ModeSelection(
            effective_mode=custom_mode_name.strip(),
            mode_source=ModeSource.CUSTOM,
            manual_mode_name=None,
            custom_mode_name=custom_mode_name.strip(),
            resolved_from_surface=surface_hint,
        )
        return sel, pol
    sel = ModeSelection(
        effective_mode="default_surface",
        mode_source=ModeSource.AUTO,
        manual_mode_name=None,
        custom_mode_name=None,
        resolved_from_surface=surface_hint,
    )
    pol = mode_policy_for("default_surface")
    pol = replace(pol, mode_name="default_surface", base_mode="default_surface")
    return sel, pol


def apply_custom_overrides(base: ModePolicy, custom: CustomModeRecord, *, display_name: str) -> ModePolicy:
    """Merge *custom* onto *base* without mutating the built-in template."""
    pol = deepcopy(base)
    pol.mode_name = display_name
    pol.base_mode = custom.base_mode
    if custom.prompt_prefix:
        pol.prompt_prefix = (pol.prompt_prefix + "\n" if pol.prompt_prefix else "") + custom.prompt_prefix.strip()
    if custom.itn_override:
        pol.itn_policy = custom.itn_override
    if custom.cleanup_override:
        pol.cleanup_policy = custom.cleanup_override
    if custom.snippet_scope:
        pol.snippet_scope_policy = f"custom:{custom.snippet_scope}"
    if custom.command_policy:
        pol.command_ambiguity_policy = custom.command_policy
    if custom.auto_transform_id:
        pol.allow_auto_transform = True
    return pol
