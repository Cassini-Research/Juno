from __future__ import annotations

from juno_v2.contracts.modes import CustomModeRecord, ModeSource
from juno_v2.modes.defaults import BUILTIN_MODE_NAMES, BUILTIN_MODES
from juno_v2.modes.policy import (
    apply_custom_overrides,
    mode_policy_for,
    resolve_mode_selection,
)


# ---------------------------------------------------------------------------
# mode_policy_for
# ---------------------------------------------------------------------------


def test_mode_policy_for_every_builtin_id() -> None:
    for name in BUILTIN_MODE_NAMES:
        pol = mode_policy_for(name)
        assert pol.mode_name == name
        assert pol.base_mode == name


def test_mode_policy_for_returns_independent_copies() -> None:
    pol = mode_policy_for("verbatim")
    original_prefix = BUILTIN_MODES["verbatim"].prompt_prefix
    pol.prompt_prefix = "mutated"
    assert BUILTIN_MODES["verbatim"].prompt_prefix == original_prefix
    assert mode_policy_for("verbatim").prompt_prefix == original_prefix


def test_mode_policy_for_legacy_alias_maps_to_default_surface() -> None:
    pol = mode_policy_for("technical_precise")
    assert pol.mode_name == "technical_precise"
    assert pol.base_mode == "default_surface"
    # The behavioral payload comes from the mapped built-in.
    assert pol.writer_behavior == BUILTIN_MODES["default_surface"].writer_behavior
    assert pol.cleanup_policy == BUILTIN_MODES["default_surface"].cleanup_policy


def test_mode_policy_for_unknown_name_falls_back_to_default_surface() -> None:
    pol = mode_policy_for("totally_made_up_mode")
    assert pol.mode_name == "totally_made_up_mode"
    assert pol.base_mode == "default_surface"
    assert pol.writer_behavior == BUILTIN_MODES["default_surface"].writer_behavior


def test_mode_policy_for_empty_name_defaults_to_default_surface() -> None:
    pol = mode_policy_for("")
    assert pol.mode_name == "default_surface"
    assert pol.base_mode == "default_surface"


def test_mode_policy_for_strips_whitespace() -> None:
    pol = mode_policy_for("  verbatim  ")
    assert pol.mode_name == "verbatim"
    assert pol.base_mode == "verbatim"


def test_mode_policy_for_base_only_keeps_mode_name() -> None:
    pol = mode_policy_for("casual_chat", base_only=True)
    assert pol.mode_name == "casual_chat"
    assert pol.base_mode == "casual_chat"


# ---------------------------------------------------------------------------
# resolve_mode_selection
# ---------------------------------------------------------------------------


def _custom_record(**overrides: object) -> CustomModeRecord:
    base: dict[str, object] = dict(
        name="Meeting Notes+",
        base_mode="structured_notes",
        prompt_prefix="Always use my note style.",
        enabled=True,
    )
    base.update(overrides)
    return CustomModeRecord(**base)  # type: ignore[arg-type]


def test_manual_selection_wins_over_custom() -> None:
    sel, pol = resolve_mode_selection(
        manual_mode_name="verbatim",
        custom_mode_name="Meeting Notes+",
        custom_record=_custom_record(),
        surface_hint="notes_app",
    )
    assert sel.mode_source is ModeSource.MANUAL
    assert sel.effective_mode == "verbatim"
    assert sel.manual_mode_name == "verbatim"
    assert sel.custom_mode_name is None
    assert sel.resolved_from_surface == "notes_app"
    assert pol.mode_name == "verbatim"
    assert pol.base_mode == "verbatim"


def test_custom_selection_used_when_no_manual() -> None:
    sel, pol = resolve_mode_selection(
        manual_mode_name=None,
        custom_mode_name="Meeting Notes+",
        custom_record=_custom_record(),
    )
    assert sel.mode_source is ModeSource.CUSTOM
    assert sel.effective_mode == "Meeting Notes+"
    assert sel.custom_mode_name == "Meeting Notes+"
    assert sel.manual_mode_name is None
    assert pol.mode_name == "Meeting Notes+"
    assert pol.base_mode == "structured_notes"
    assert pol.prompt_prefix.endswith("Always use my note style.")


def test_disabled_custom_record_falls_back_to_auto() -> None:
    sel, pol = resolve_mode_selection(
        manual_mode_name=None,
        custom_mode_name="Meeting Notes+",
        custom_record=_custom_record(enabled=False),
    )
    assert sel.mode_source is ModeSource.AUTO
    assert sel.effective_mode == "default_surface"
    assert pol.mode_name == "default_surface"


def test_custom_name_without_record_falls_back_to_auto() -> None:
    sel, _pol = resolve_mode_selection(
        manual_mode_name=None,
        custom_mode_name="Meeting Notes+",
        custom_record=None,
    )
    assert sel.mode_source is ModeSource.AUTO
    assert sel.effective_mode == "default_surface"


def test_no_selection_resolves_to_auto_default_surface() -> None:
    sel, pol = resolve_mode_selection(
        manual_mode_name=None,
        custom_mode_name=None,
        custom_record=None,
        surface_hint="mail_compose",
    )
    assert sel.mode_source is ModeSource.AUTO
    assert sel.effective_mode == "default_surface"
    assert sel.resolved_from_surface == "mail_compose"
    assert pol.mode_name == "default_surface"
    assert pol.base_mode == "default_surface"


def test_manual_selection_strips_whitespace() -> None:
    sel, pol = resolve_mode_selection(
        manual_mode_name=" verbatim ",
        custom_mode_name=None,
        custom_record=None,
    )
    assert sel.effective_mode == "verbatim"
    assert pol.mode_name == "verbatim"


# ---------------------------------------------------------------------------
# apply_custom_overrides
# ---------------------------------------------------------------------------


def test_apply_custom_overrides_merges_all_fields() -> None:
    base = mode_policy_for("verbatim")
    custom = CustomModeRecord(
        name="Loud Verbatim",
        base_mode="verbatim",
        prompt_prefix="  Shout everything.  ",
        itn_override="standard",
        cleanup_override="full",
        snippet_scope="email",
        command_policy="wide",
        auto_transform_id="polish",
    )
    pol = apply_custom_overrides(base, custom, display_name="Loud Verbatim")
    assert pol.mode_name == "Loud Verbatim"
    assert pol.base_mode == "verbatim"
    assert pol.prompt_prefix == base.prompt_prefix + "\nShout everything."
    assert pol.itn_policy == "standard"
    assert pol.cleanup_policy == "full"
    assert pol.snippet_scope_policy == "custom:email"
    assert pol.command_ambiguity_policy == "wide"
    # verbatim disallows auto transforms; an auto_transform_id re-enables them.
    assert base.allow_auto_transform is False
    assert pol.allow_auto_transform is True


def test_apply_custom_overrides_does_not_mutate_base() -> None:
    base = mode_policy_for("verbatim")
    snapshot = base.to_dict()
    custom = CustomModeRecord(
        name="Loud Verbatim",
        base_mode="verbatim",
        prompt_prefix="Shout everything.",
        itn_override="standard",
        cleanup_override="full",
        snippet_scope="email",
        command_policy="wide",
        auto_transform_id="polish",
    )
    apply_custom_overrides(base, custom, display_name="Loud Verbatim")
    assert base.to_dict() == snapshot


def test_apply_custom_overrides_empty_optionals_keep_base_values() -> None:
    base = mode_policy_for("default_surface")
    custom = CustomModeRecord(name="Plain", base_mode="default_surface")
    pol = apply_custom_overrides(base, custom, display_name="Plain")
    assert pol.mode_name == "Plain"
    assert pol.base_mode == "default_surface"
    assert pol.prompt_prefix == base.prompt_prefix
    assert pol.itn_policy == base.itn_policy
    assert pol.cleanup_policy == base.cleanup_policy
    assert pol.snippet_scope_policy == base.snippet_scope_policy
    assert pol.command_ambiguity_policy == base.command_ambiguity_policy
    assert pol.allow_auto_transform == base.allow_auto_transform


def test_apply_custom_overrides_prefix_on_empty_base_has_no_leading_newline() -> None:
    base = mode_policy_for("default_surface")
    assert base.prompt_prefix == ""
    custom = CustomModeRecord(
        name="Prefixed", base_mode="default_surface", prompt_prefix="My style."
    )
    pol = apply_custom_overrides(base, custom, display_name="Prefixed")
    assert pol.prompt_prefix == "My style."
