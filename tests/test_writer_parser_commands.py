"""Regression tests for the writer intent parser's memory-command grammar.

Bug (2026-06-10): _ADD_REPLACEMENT_RULES used \\b + re.search, so ordinary
prose containing "change X to Y" mid-sentence ("we should change these two
buttons to ...") was consumed as an add_replacement memory command. The
dictated text vanished (writer returned empty output), a junk replacement
rule was written into user memory, and History showed a misleading
"Juno could not insert text" failure. The rules are now anchored to the
start of the utterance and triggers are capped at 6 words.
"""
from __future__ import annotations

import pytest

from juno_v2.writer.parser import WriterIntentParser
from juno_v2.contracts.writer import WriterIntentKind


@pytest.fixture()
def parser() -> WriterIntentParser:
    return WriterIntentParser()


@pytest.mark.parametrize(
    "prose",
    [
        # The exact dictation that triggered the bug.
        "We should change these two buttons to what is Juno and Quickstart "
        "instead of start using Juno and install Juno.",
        "I will replace the old logo with the new one tomorrow",
        "The plan is to change the API to use protobuf instead of JSON",
        "Let's change the meeting to Thursday afternoon",
        "They want us to replace the banner with a carousel",
    ],
)
def test_prose_containing_change_to_is_dictation(parser: WriterIntentParser, prose: str) -> None:
    intent = parser.parse(prose)
    assert intent.kind == WriterIntentKind.DICTATE, (
        f"prose was consumed as {intent.kind}: trigger={getattr(intent, 'trigger', None)!r}"
    )


@pytest.mark.parametrize(
    "command",
    [
        "change Chino to Juno",
        "Juno change Chino to Juno",
        "hey juno change Chino to Juno",
        "please change Chino to Juno",
        "always replace hey chino with hey juno",
        "remember that QBR means quarterly business review",
    ],
)
def test_explicit_replacement_commands_still_fire(parser: WriterIntentParser, command: str) -> None:
    intent = parser.parse(command)
    assert intent.kind == WriterIntentKind.ADD_REPLACEMENT, f"{command!r} -> {intent.kind}"
    assert intent.trigger
    assert intent.replacement


def test_bare_replace_with_goes_to_deterministic_command_path(parser: WriterIntentParser) -> None:
    # Pre-existing precedence: "replace X with Y" is owned by
    # parse_deterministic_command (COMMAND_RESULT), not the memory rule.
    intent = parser.parse("replace my email with paresh@example.com")
    assert intent.kind == WriterIntentKind.COMMAND_RESULT


def test_long_trigger_is_rejected_as_command(parser: WriterIntentParser) -> None:
    # Anchored, but the "trigger" is a prose clause — the 6-word trigger cap
    # keeps sentence-shaped utterances out of memory.
    intent = parser.parse(
        "change the way we talk about onboarding in the docs to something friendlier"
    )
    assert intent.kind != WriterIntentKind.ADD_REPLACEMENT


def test_replacement_command_round_trip_fields(parser: WriterIntentParser) -> None:
    intent = parser.parse("change Chino to Juno")
    assert intent.trigger == "Chino"
    assert intent.replacement == "Juno"


def test_downstream_ai_prompt_is_dictation_without_a_juno_target(
    parser: WriterIntentParser,
) -> None:
    intent = parser.parse(
        "Summarize this whole chat. 5 bullet points. What all was made?",
        selection_present=False,
    )

    assert intent.kind == WriterIntentKind.DICTATE


def test_selection_summary_prompt_remains_a_transform(
    parser: WriterIntentParser,
) -> None:
    intent = parser.parse("Summarize this in five points", selection_present=True)

    assert intent.kind == WriterIntentKind.MODEL_TRANSFORM


@pytest.mark.parametrize(
    "command",
    [
        "Summarize that",
        "Summarize the last paragraph",
        "Expand that with more detail",
    ],
)
def test_explicit_recent_semantic_commands_remain_commands(
    parser: WriterIntentParser,
    command: str,
) -> None:
    intent = parser.parse(command, selection_present=False)

    assert intent.kind != WriterIntentKind.DICTATE


@pytest.mark.parametrize(
    "prose",
    [
        "This paragraph discusses bullet points.",
        "I mentioned bullet points in this paragraph.",
        "Bullet points are the topic of this paragraph.",
    ],
)
def test_selected_prose_that_mentions_bullets_is_not_a_transform(
    parser: WriterIntentParser,
    prose: str,
) -> None:
    intent = parser.parse(prose, selection_present=True)

    assert intent.kind == WriterIntentKind.DICTATE
