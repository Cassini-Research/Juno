from __future__ import annotations

import pytest

from juno_v2.presets.surface_presets import SurfacePresetStore
from juno_v2.writer.final_formatter import apply_commit_boundary_rules


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sounds good.", "Sounds good"),
        ("I will check it.", "I will check it"),
        ("I checked it. Looks fine.", "I checked it. Looks fine"),
        ("Are you coming?", "Are you coming?"),
        ("Perfect!", "Perfect!"),
        ("Maybe...", "Maybe..."),
        ("The version is 2.0.", "The version is 2.0"),
        ("Check example.com", "Check example.com"),
    ],
)
def test_casual_messaging_boundary(text: str, expected: str) -> None:
    assert (
        apply_commit_boundary_rules(
            text,
            app_category="messaging",
            mode_name="casual_chat",
            final_formatting_policy="messaging",
        )
        == expected
    )


def test_non_messaging_modes_keep_terminal_periods() -> None:
    assert (
        apply_commit_boundary_rules(
            "Will do.",
            app_category="email",
            mode_name="formal_email",
            final_formatting_policy="email",
        )
        == "Will do."
    )
    assert (
        apply_commit_boundary_rules(
            "Will do.",
            app_category="docs",
            mode_name="structured_notes",
            final_formatting_policy="structured_notes",
        )
        == "Will do."
    )


def test_casual_chat_boundary_does_not_expand_fragments() -> None:
    assert (
        apply_commit_boundary_rules(
            "yeah I think Cassini is a good place to do it and we can help you launch it.",
            app_category="messaging",
            mode_name="casual_chat",
            final_formatting_policy="messaging",
        )
        == "Yeah I think Cassini is a good place to do it and we can help you launch it"
    )


def test_builtin_messaging_presets_share_light_cleanup_guidance(tmp_path) -> None:
    store = SurfacePresetStore(tmp_path / "surface_presets.json")
    by_id = {preset.id: preset for preset in store.list_presets_merged()}
    messaging_ids = [
        "builtin-slack",
        "builtin-telegram",
        "builtin-whatsapp",
        "builtin-messages",
    ]

    assert len({by_id[preset_id].asr_addon for preset_id in messaging_ids}) == 1
    tones = {by_id[preset_id].writer_tone_addon for preset_id in messaging_ids}
    assert len(tones) == 1
    assert "Do not formalize" in next(iter(tones))
