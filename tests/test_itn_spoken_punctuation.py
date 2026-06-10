"""Spoken-punctuation ITN regressions.

Production 2026-06-10: "…exist? New paragraph, text…" shipped as
"…exist?\n\n, text…" — the newline cue replacement kept the comma the ASR
had glued onto the spoken cue, so the pasted note began a paragraph with
", text". These tests pin cue-adjacent punctuation consumption and the
literal-mention guard.
"""
from __future__ import annotations

import pytest

from juno_v2.itn.engine import ITNEngine, ITNProfile


@pytest.fixture(scope="module")
def engine() -> ITNEngine:
    return ITNEngine()


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        # ASR attaches a comma AFTER the cue — the new paragraph must not
        # start with ", text" (the shipped production corruption).
        (
            "What is Juno and why does it exist? New paragraph, text is still the main interface.",
            "What is Juno and why does it exist?\n\ntext is still the main interface.",
        ),
        # ASR attaches a comma BEFORE the cue — the paragraph must not end
        # with a dangling comma.
        (
            "between models and people, new paragraph but voice is how we think.",
            "between models and people\n\nbut voice is how we think.",
        ),
        # Spoken terminal mark replaces an ASR comma in front of it.
        ("hold on, period", "hold on."),
        # Plain inline cues keep working.
        ("hello comma world", "hello, world"),
        ("end of thought period next idea", "end of thought. next idea"),
        ("first point new line second point", "first point\nsecond point"),
        # De-dup against an existing glyph.
        ("hello, comma world", "hello, world"),
    ],
)
def test_cue_adjacent_punctuation(engine: ITNEngine, spoken: str, expected: str) -> None:
    assert engine.run(spoken, profile=ITNProfile("prose")).text == expected


@pytest.mark.parametrize(
    "literal",
    [
        # Determiner before the cue means a noun mention, not a command.
        "the new paragraph is short",
        "a comma goes here",
        "that new line looks wrong",
        "every period in this draft is misplaced",
    ],
)
def test_literal_mentions_are_left_alone(engine: ITNEngine, literal: str) -> None:
    assert engine.run(literal, profile=ITNProfile("prose")).text == literal


@pytest.mark.parametrize(
    ("profile", "spoken", "expected"),
    [
        # Terminal exactness: paired spoken quotes + colon inside them.
        (
            "terminal",
            "git commit dash m quote fix colon preserve Qwen turn plan metadata quote",
            'git commit-m "fix: preserve Qwen turn plan metadata"',
        ),
        ("prose", "He said quote we ship Friday unquote to the team.", 'He said "we ship Friday" to the team.'),
        ("prose", "she told me quote the deadline moved unquote yesterday", 'she told me "the deadline moved" yesterday'),
        ("prose", "open quote alpha close quote", '"alpha"'),
        # Terminal operator phrases; longest-first ordering keeps the double
        # forms reachable (was: "double ampersand" → "double &").
        ("terminal", "echo done double ampersand make build", "echo done && make build"),
        ("terminal", "ls pipe grep juno", "ls | grep juno"),
    ],
)
def test_spoken_quotes_and_terminal_ops(engine: ITNEngine, profile: str, spoken: str, expected: str) -> None:
    assert engine.run(spoken, profile=ITNProfile(profile)).text == expected


@pytest.mark.parametrize(
    "literal",
    [
        # Verb / noun usages of bare "quote" stay literal; unpaired cues too.
        "I'll quote the answer in the doc",
        "the quote from the article is long",
        "we quote prices daily for customers",
    ],
)
def test_bare_quote_literal_usages_preserved(engine: ITNEngine, literal: str) -> None:
    assert engine.run(literal, profile=ITNProfile("prose")).text == literal
