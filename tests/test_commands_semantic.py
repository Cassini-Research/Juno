from __future__ import annotations

from dataclasses import replace

import pytest

from juno_v2.commands.semantic import interpret_semantic_command
from juno_v2.contracts.commands import CommandTargetClass
from juno_v2.contracts.writer import WriterMode
from juno_v2.modes.policy import mode_policy_for


def _interpret(
    text: str,
    *,
    mode_policy=None,
    active_mode: WriterMode = WriterMode.DEFAULT_SURFACE,
    target_class: CommandTargetClass = CommandTargetClass.SELECTED_TEXT,
    target_text: str | None = "some selected text",
):
    return interpret_semantic_command(
        text,
        mode_policy=mode_policy,
        active_mode=active_mode,
        target_class=target_class,
        target_text=target_text,
    )


# ---------------------------------------------------------------------------
# Recognized intents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "intent_name"),
    [
        ("make that shorter", "make_shorter"),
        ("make this more concise", "make_shorter"),
        ("can you make it brief", "make_shorter"),
        ("make that clearer", "make_clearer"),
        ("make this more formal", "make_formal"),
        ("make that professional", "make_formal"),
        ("make this casual", "make_casual"),
        ("make it more friendly", "make_casual"),
        ("fix the grammar", "fix_grammar"),
        ("clean up the spelling", "fix_grammar"),
        ("correct the punctuation", "fix_grammar"),
        ("summarize that", "summarize"),
        ("turn that into bullet points", "bullets"),
        ("make this a numbered list", "numbered"),
        ("expand on that", "expand"),
        ("make this more detailed", "expand"),
        ("simplify that", "simplify"),
    ],
)
def test_recognized_intents(text: str, intent_name: str) -> None:
    intent = _interpret(text)
    assert intent is not None
    assert intent.intent_name == intent_name
    assert intent.requires_confirmation is False
    assert intent.ambiguity_reason is None
    assert intent.rewrite_instruction
    assert 0.0 < intent.target_confidence <= 1.0
    assert intent.target_class == CommandTargetClass.SELECTED_TEXT


def test_intent_matching_is_case_insensitive() -> None:
    intent = _interpret("MAKE THAT SHORTER")
    assert intent is not None and intent.intent_name == "make_shorter"


def test_intent_carries_requested_target_class() -> None:
    intent = _interpret(
        "fix the grammar",
        target_class=CommandTargetClass.RECENT_COMMIT,
    )
    assert intent is not None
    assert intent.target_class == CommandTargetClass.RECENT_COMMIT


# ---------------------------------------------------------------------------
# Unrecognized / empty input
# ---------------------------------------------------------------------------


def test_unrecognized_text_returns_none() -> None:
    assert _interpret("hello world") is None
    assert _interpret("send an email to bob") is None


def test_empty_and_whitespace_text_returns_none() -> None:
    assert _interpret("") is None
    assert _interpret("   \n ") is None


def test_none_target_class_returns_none_even_for_recognized_text() -> None:
    assert _interpret("make that shorter", target_class=CommandTargetClass.NONE) is None


def test_keyword_without_deictic_reference_returns_none() -> None:
    # "shorter" alone, with no that/this/it pointing at a target, is not a command.
    assert _interpret("a shorter version would be nice maybe") is None


# ---------------------------------------------------------------------------
# Mode-policy gating
# ---------------------------------------------------------------------------


def test_none_policy_allows_semantics() -> None:
    intent = _interpret("make that shorter", mode_policy=None)
    assert intent is not None and intent.intent_name == "make_shorter"


def test_selection_commands_blocked_by_policy() -> None:
    policy = replace(mode_policy_for("default_surface"), allow_selection_commands=False)
    intent = _interpret(
        "make that shorter",
        mode_policy=policy,
        target_class=CommandTargetClass.SELECTED_TEXT,
    )
    assert intent is not None
    assert intent.intent_name == "declined_semantic"
    assert intent.requires_confirmation is True
    assert intent.ambiguity_reason == "mode_disallows_model_semantics"
    assert intent.rewrite_instruction == ""
    assert intent.target_confidence == pytest.approx(0.2)


def test_selection_commands_allowed_by_policy() -> None:
    policy = mode_policy_for("default_surface")
    assert policy.allow_selection_commands is True
    intent = _interpret(
        "make that shorter",
        mode_policy=policy,
        target_class=CommandTargetClass.SELECTED_TEXT,
    )
    assert intent is not None and intent.intent_name == "make_shorter"


def test_recent_target_commands_blocked_by_policy() -> None:
    policy = replace(mode_policy_for("default_surface"), allow_recent_target_commands=False)
    intent = _interpret(
        "fix the grammar",
        mode_policy=policy,
        target_class=CommandTargetClass.RECENT_COMMIT,
    )
    assert intent is not None and intent.intent_name == "declined_semantic"


def test_other_targets_gated_by_model_insert_rewrite() -> None:
    # default_surface disallows model insert rewrites, so an active-utterance
    # semantic command is declined under that policy.
    policy = mode_policy_for("default_surface")
    assert policy.allow_model_insert_rewrite is False
    intent = _interpret(
        "make that shorter",
        mode_policy=policy,
        target_class=CommandTargetClass.ACTIVE_UTTERANCE,
    )
    assert intent is not None and intent.intent_name == "declined_semantic"

    permissive = mode_policy_for("explicit_rewrite")
    assert permissive.allow_model_insert_rewrite is True
    intent = _interpret(
        "make that shorter",
        mode_policy=permissive,
        target_class=CommandTargetClass.ACTIVE_UTTERANCE,
    )
    assert intent is not None and intent.intent_name == "make_shorter"


def test_command_mode_bypasses_policy_gates() -> None:
    policy = replace(
        mode_policy_for("default_surface"),
        allow_selection_commands=False,
        allow_recent_target_commands=False,
        allow_model_insert_rewrite=False,
    )
    intent = _interpret(
        "make that shorter",
        mode_policy=policy,
        active_mode=WriterMode.COMMAND_MODE,
        target_class=CommandTargetClass.SELECTED_TEXT,
    )
    assert intent is not None and intent.intent_name == "make_shorter"


def test_blocked_mode_declines_before_template_matching() -> None:
    # The gate is evaluated before template matching, so even unrecognized
    # text yields a declined intent when the mode disallows semantics.
    policy = replace(mode_policy_for("default_surface"), allow_selection_commands=False)
    intent = _interpret("hello world", mode_policy=policy)
    assert intent is not None and intent.intent_name == "declined_semantic"
