from __future__ import annotations

import pytest

from juno_v2.list_content import protect_list_render


def test_list_render_preserves_mixed_language_prefix() -> None:
    source = (
        "هذا مهم. First protect the opening text. "
        "Second keep the list safe."
    )

    protected = protect_list_render(
        source,
        "- protect the opening text\n- keep the list safe",
    )

    assert protected.text == (
        "هذا مهم.\n"
        "- protect the opening text\n"
        "- keep the list safe"
    )
    assert protected.mode == "substantive_prefix_preserved"


def test_list_render_preserves_declarative_prefix() -> None:
    source = (
        "I create things. First prototypes help users. "
        "Second tests keep them safe."
    )

    protected = protect_list_render(
        source,
        "- prototypes help users\n- tests keep them safe",
    )

    assert protected.text == (
        "I create things.\n"
        "- prototypes help users\n"
        "- tests keep them safe"
    )
    assert protected.mode == "substantive_prefix_preserved"


def test_list_render_falls_back_when_content_looks_like_a_marker() -> None:
    source = "There are two things. First choose option one. Second ship."

    protected = protect_list_render(source, "- choose option\n- ship")

    assert protected.text == source
    assert protected.mode == "complete_transcript_fallback"


@pytest.mark.parametrize("content", ["priority", "task", "focus area"])
def test_list_render_falls_back_for_ambiguous_item_label(content: str) -> None:
    source = (
        "There are two things. First choose the option. "
        f"Second {content} ship."
    )

    protected = protect_list_render(source, "- choose the option\n- ship")

    assert protected.text == source
    assert protected.mode == "complete_transcript_fallback"


def test_list_render_accepts_complete_item_marker_phrase() -> None:
    source = (
        "There are two things. First choose the option. "
        "Second priority is ship."
    )

    protected = protect_list_render(source, "- choose the option\n- ship")

    assert protected.text == "- choose the option\n- ship"
    assert protected.mode == "list_rendered"
