from __future__ import annotations

from dataclasses import dataclass

import pytest

from juno_v2.writer.deterministic import (
    AppCategory,
    apply_app_formatting,
    apply_newline_policy,
    expand_snippets,
    normalize_dictation_orthography,
    normalize_explicit_numbered_markers,
    normalize_plain_dictation,
    render_bullets,
    render_list_from_ordinal_sentences,
    render_list_from_ordinals,
    render_lowercase,
    render_numbered,
    render_three_things_agenda,
    render_title_case,
    render_two_bullet_points,
    render_uppercase,
    resolve_backtrack,
    run_pipeline,
    strip_correction_chants,
    strip_fillers,
)
from juno_v2.writer.guards import is_no_touch_surface


# ---------------------------------------------------------------------- #
# Layer 1: casing / structure primitives
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("apples, bananas, cherries", "- apples\n- bananas\n- cherries"),
        ("Buy milk. Walk dog. Sleep.", "- Buy milk\n- Walk dog\n- Sleep"),
        ("alpha; beta; gamma", "- alpha\n- beta\n- gamma"),
        ("line one\nline two", "- line one\n- line two"),
        ("just one thing", "- just one thing"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_render_bullets(text: str, expected: str) -> None:
    assert render_bullets(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("alpha; beta; gamma", "1. alpha\n2. beta\n3. gamma"),
        ("apples, bananas", "1. apples\n2. bananas"),
        ("", ""),
    ],
)
def test_render_numbered(text: str, expected: str) -> None:
    assert render_numbered(text) == expected


def test_casing_primitives() -> None:
    assert render_uppercase("hello") == "HELLO"
    assert render_lowercase("HeLLo") == "hello"
    assert render_title_case("hello world") == "Hello World"
    assert render_uppercase("") == ""
    assert render_lowercase("") == ""
    assert render_title_case("") == ""


# ---------------------------------------------------------------------- #
# Layer 2: normalize_plain_dictation
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  hello   world  ", "hello world"),
        ("hello , world", "hello, world"),
        ("hello,world", "hello, world"),
        ("hello;world", "hello; world"),
        ("done:next", "done: next"),
        ("hello.world", "hello. world"),  # long token after period = sentence boundary
        ("", ""),
    ],
)
def test_normalize_plain_dictation_spacing(text: str, expected: str) -> None:
    assert normalize_plain_dictation(text) == expected


@pytest.mark.parametrize(
    "preserved",
    [
        "1,000",
        "1,234,567",
        "5:30 pm",
        "1:1",
        "16:9",
        "https://x.com",
        "ws://host",
        "no!!!",
        "What?!",
        "wait... what?",
        "auth.ts",
        "file.json",
        "1.5.2",
        "github.com",
    ],
)
def test_normalize_plain_dictation_preserves_documented_tokens(preserved: str) -> None:
    # Each token is explicitly listed in the module as a shape the
    # per-character punctuation rules must not break.
    assert normalize_plain_dictation(preserved) == preserved


def test_normalize_plain_dictation_idempotent() -> None:
    once = normalize_plain_dictation("hello ,world!next  thing")
    assert normalize_plain_dictation(once) == once


# ---------------------------------------------------------------------- #
# normalize_dictation_orthography
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("i think i saw b and c in january. yes", "I think I saw B and C in January. Yes"),
        ("don't worry about b", "Don't worry about B"),
        ("hello. world", "Hello. World"),
        ("", ""),
    ],
)
def test_normalize_dictation_orthography(text: str, expected: str) -> None:
    assert normalize_dictation_orthography(text) == expected


def test_normalize_dictation_orthography_letter_a_untouched() -> None:
    # 'a' is an article and is not in the standalone-letter class.
    assert normalize_dictation_orthography("pick a card") == "Pick a card"


# ---------------------------------------------------------------------- #
# Layer 3: apply_newline_policy
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello new line world", "hello\nworld"),
        ("hello newline world", "hello\nworld"),
        ("hello line break world", "hello\nworld"),
        ("go to new line. next", "\nnext"),
        ("okay go to new line next", "\nnext"),
        ("intro new paragraph body", "intro\n\nbody"),
        ("a new paragraph new paragraph b", "a\n\nb"),  # runs collapse to one break
        ("", ""),
    ],
)
def test_apply_newline_policy(text: str, expected: str) -> None:
    assert apply_newline_policy(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "the new paragraph is short",
        "this new line of products",
        "my new paragraph needs work",
        "every line break in the file",
    ],
)
def test_apply_newline_policy_leaves_legitimate_mention_alone(text: str) -> None:
    assert apply_newline_policy(text) == text


# ---------------------------------------------------------------------- #
# Layer 4: expand_snippets
# ---------------------------------------------------------------------- #


@dataclass
class _Snip:
    trigger: str
    body: str
    scope: str = "global"
    case_sensitive: bool = False


class _ListResolver:
    def __init__(self, snips: list[_Snip]) -> None:
        self._snips = snips

    def list(self) -> list[_Snip]:
        return self._snips

    def resolve(self, trigger: str, *, scope: str = "global") -> _Snip | None:
        return None


class _ResolveOnly:
    """Resolver without ``list`` — exercises the fallback path."""

    def resolve(self, trigger: str, *, scope: str = "global") -> _Snip | None:
        if trigger.lower() == "omw":
            return _Snip("omw", "on my way")
        return None


def test_expand_snippets_basic() -> None:
    resolver = _ListResolver([_Snip("signoff", "Best,\nJuno")])
    assert expand_snippets("please add my signoff", resolver=resolver) == "please add my Best,\nJuno"


def test_expand_snippets_fold_aware_segmentation() -> None:
    # Stored "signoff" fires on spoken "sign off" and "Sign-Off" (docstring).
    resolver = _ListResolver([_Snip("signoff", "Best,\nJuno")])
    assert expand_snippets("please sign off now", resolver=resolver) == "please Best,\nJuno now"
    assert expand_snippets("Sign-Off", resolver=resolver) == "Best,\nJuno"


def test_expand_snippets_word_boundaries() -> None:
    # "brb" must not fire inside "brbrain" (docstring step 3).
    resolver = _ListResolver([_Snip("brb", "be right back")])
    assert expand_snippets("brbrain stays", resolver=resolver) == "brbrain stays"


def test_expand_snippets_case_sensitive_trigger() -> None:
    resolver = _ListResolver([_Snip("BRB", "be right back", case_sensitive=True)])
    assert expand_snippets("BRB and brb", resolver=resolver) == "be right back and brb"


def test_expand_snippets_no_resolver_and_empty() -> None:
    assert expand_snippets("text", resolver=None) == "text"
    assert expand_snippets("", resolver=_ListResolver([])) == ""


def test_expand_snippets_resolve_only_fallback() -> None:
    assert expand_snippets("omw home", resolver=_ResolveOnly()) == "on my way home"


def test_expand_snippets_max_expansions_caps_runaway() -> None:
    resolver = _ListResolver([_Snip("tk", "EXPANDED")])
    out = expand_snippets("tk tk tk tk tk", resolver=resolver, max_expansions=3)
    assert out.count("EXPANDED") == 3
    assert out.count("tk") == 2


# ---------------------------------------------------------------------- #
# Layer 5: apply_app_formatting
# ---------------------------------------------------------------------- #


def test_apply_app_formatting_raw_categories_untouched() -> None:
    raw = "weird   spacing\n\n\n"
    assert apply_app_formatting(raw, category=AppCategory.CODE) == raw
    assert apply_app_formatting(raw, category="terminal") == raw


def test_apply_app_formatting_messaging_collapses_paragraphs() -> None:
    assert apply_app_formatting("a\n\nb\n", category="messaging") == "a\nb\n"


def test_apply_app_formatting_forms_flattens_newlines() -> None:
    assert apply_app_formatting("a\nb\n\nc", category=AppCategory.FORMS) == "a b c"


def test_apply_app_formatting_docs_strips_trailing_whitespace_per_line() -> None:
    assert apply_app_formatting("a  \nb  ", category="docs") == "a\nb"


def test_apply_app_formatting_unknown_and_invalid_category() -> None:
    assert apply_app_formatting("a  \nb", category=None) == "a\nb"
    assert apply_app_formatting("a  \nb", category="not-a-category") == "a\nb"
    assert apply_app_formatting("", category="docs") == ""


# ---------------------------------------------------------------------- #
# Layer 6: speech corrections
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the meeting is at 2 actually 3 pm", "the meeting is at 3 pm"),
        ("Actually let me rephrase that. We ship Friday.", "We ship Friday."),
        ("today, no actually tomorrow morning works", "tomorrow morning works"),
        ("Morgan, actually his name is Morgan", "Morgan"),
        # Names that differ are NOT a redundancy — leave intact.
        ("Morgan, actually his name is Logan", "Morgan, actually his name is Logan"),
        # No 'actually' → fast-path no-op.
        ("plain sentence", "plain sentence"),
        ("", ""),
    ],
)
def test_resolve_backtrack(text: str, expected: str) -> None:
    assert resolve_backtrack(text) == expected


def test_strip_correction_chants() -> None:
    out = strip_correction_chants("that's accommodate with two c's and two m's and send it")
    assert "with two c's" not in out
    assert "send it" in out
    assert strip_correction_chants("") == ""
    assert strip_correction_chants("nothing here") == "nothing here"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I think like the plan is good", "I think the plan is good"),
        ("It was, you know, fine", "It was, fine"),
        # Comparison-"like" must be preserved (docstring example).
        ("I think like a fox", "I think like a fox"),
        ("", ""),
    ],
)
def test_strip_fillers(text: str, expected: str) -> None:
    assert strip_fillers(text) == expected


# ---------------------------------------------------------------------- #
# Layers 6b/7: list rendering
# ---------------------------------------------------------------------- #


def test_render_two_bullet_points() -> None:
    out = render_two_bullet_points("add two bullet points first, buy milk second, walk dog")
    assert out == "- Buy milk.\n- Walk dog."


def test_render_two_bullet_points_ignores_plain_prose() -> None:
    text = "first, buy milk second, walk dog"
    assert render_two_bullet_points(text) == text
    assert render_two_bullet_points("") == ""


def test_render_list_from_ordinals() -> None:
    out = render_list_from_ordinals(
        "My goals are first finish report, second send deck, and third book calls."
    )
    assert out == "My goals are:\n1. Finish report.\n2. Send deck.\n3. Book calls."


def test_render_list_from_ordinals_requires_three_items() -> None:
    text = "first do this, second do that."
    assert render_list_from_ordinals(text) == text
    assert render_list_from_ordinals("") == ""


def test_render_list_from_ordinal_sentences() -> None:
    out = render_list_from_ordinal_sentences(
        "Here is the plan. First, ship X. Second, ship Y. Third, ship Z."
    )
    assert out == "Here is the plan:\n1. Ship X.\n2. Ship Y.\n3. Ship Z.\n"


def test_render_list_from_ordinal_sentences_no_op() -> None:
    assert render_list_from_ordinal_sentences("Just one sentence.") == "Just one sentence."
    assert render_list_from_ordinal_sentences("") == ""


def test_render_three_things_agenda() -> None:
    out = render_three_things_agenda(
        "we will cover three things. first, scope. second, budget. third, timeline."
    )
    assert out == "we will cover three things:\n1. Scope.\n2. Budget.\n3. Timeline."


def test_render_three_things_agenda_no_op() -> None:
    assert render_three_things_agenda("nothing here") == "nothing here"
    assert render_three_things_agenda("") == ""


# ---------------------------------------------------------------------- #
# Layer 7d: explicit numbered markers
# ---------------------------------------------------------------------- #


def test_normalize_explicit_numbered_markers() -> None:
    out = normalize_explicit_numbered_markers("Update: 1. Ship X. 2. Test Y. 3. Done Z.")
    assert out == "Update:\n1. Ship X.\n2. Test Y.\n3. Done Z."


def test_normalize_explicit_numbered_markers_requires_full_sequence() -> None:
    assert normalize_explicit_numbered_markers("1. only one item here") == "1. only one item here"
    assert normalize_explicit_numbered_markers("") == ""


# ---------------------------------------------------------------------- #
# run_pipeline
# ---------------------------------------------------------------------- #


def test_run_pipeline_raw_category_bypasses_everything() -> None:
    raw = "rm -rf /  pipe stuff new line"
    assert run_pipeline(raw, app_category="terminal") == raw
    assert run_pipeline(raw, app_category=AppCategory.CODE) == raw


def test_run_pipeline_docs_normalizes_and_applies_newlines() -> None:
    out = run_pipeline("hello world new paragraph next part", app_category="docs")
    assert out == "Hello world\n\nNext part"


def test_run_pipeline_corrections_and_fillers() -> None:
    out = run_pipeline("i think like the plan is, you know, good", app_category="unknown")
    assert out == "I think the plan is, good"


def test_run_pipeline_expands_snippets() -> None:
    resolver = _ListResolver([_Snip("signoff", "Best, Juno")])
    out = run_pipeline("add my signoff", snippet_resolver=resolver, app_category="email")
    assert out == "Add my Best, Juno"


def test_run_pipeline_empty() -> None:
    assert run_pipeline("", app_category="docs") == ""


# ---------------------------------------------------------------------- #
# guards.is_no_touch_surface
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("code", True),
        ("terminal", True),
        (" TERMINAL ", True),
        ("Code", True),
        ("docs", False),
        ("messaging", False),
        ("", False),
        (None, False),
    ],
)
def test_is_no_touch_surface(category: str | None, expected: bool) -> None:
    assert is_no_touch_surface(category) is expected
