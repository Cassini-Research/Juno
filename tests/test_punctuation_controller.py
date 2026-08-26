from __future__ import annotations

from juno_v2.writer.punctuation_controller import apply_final_punctuation_floor


def _apply(text: str, **overrides: object):
    params = dict(
        app_category="docs",
        writer_mode="default_surface",
        punctuation_policy="standard",
        final_formatting_policy="minimal",
    )
    params.update(overrides)
    return apply_final_punctuation_floor(text, **params)


def test_adds_terminal_period_to_long_plain_dictation() -> None:
    result = _apply("send the brief to Mira tonight")

    assert result.text == "send the brief to Mira tonight."
    assert result.changed is True
    assert result.rules_applied == ["terminal_period"]


def test_adds_question_mark_to_clear_question() -> None:
    result = _apply("can you send the brief tomorrow")

    assert result.text == "can you send the brief tomorrow?"
    assert result.changed is True
    assert result.rules_applied == ["terminal_question"]


def test_adds_period_to_imperative_do_command() -> None:
    result = _apply(
        "do this once after everything run an E2E test and create a PR for main"
    )

    assert result.text == "do this once after everything run an E2E test and create a PR for main."
    assert result.changed is True
    assert result.rules_applied == ["terminal_period"]


def test_adds_period_to_wh_statement_fragments() -> None:
    why = _apply("why it ended the previous paste with a question mark")
    how = _apply("how this happened in the final paste")
    embedded = _apply("I don't know why it ended the previous paste with a question mark")

    assert why.text == "why it ended the previous paste with a question mark."
    assert how.text == "how this happened in the final paste."
    assert embedded.text == "I don't know why it ended the previous paste with a question mark."
    assert why.rules_applied == ["terminal_period"]
    assert how.rules_applied == ["terminal_period"]
    assert embedded.rules_applied == ["terminal_period"]


def test_keeps_question_marks_for_clear_question_shapes() -> None:
    cases = [
        "do you want me to send it",
        "does this work for the launch",
        "why did it end with a question mark",
        "how do we fix this properly",
        "what time is the review",
    ]

    for text in cases:
        result = _apply(text)
        assert result.text == text + "?"
        assert result.changed is True
        assert result.rules_applied == ["terminal_question"]


def test_keeps_question_marks_for_technical_noun_bridge_questions() -> None:
    cases = [
        "what version should we use",
        "what branch should I use",
        "what file did you change",
        "which model should we use",
        "which repo should this target",
        "what PR did you update",
    ]

    for text in cases:
        result = _apply(text)
        assert result.text == text + "?"
        assert result.changed is True
        assert result.rules_applied == ["terminal_question"]


def test_multi_sentence_buffer_uses_trailing_sentence_for_the_mark() -> None:
    # The opening sentence is a question but the trailing clause is an
    # imperative, so the buffer must close with a period (issue #70).
    result = _apply("Do you have the file? Please send it to me now")

    assert result.text == "Do you have the file? Please send it to me now."
    assert result.changed is True
    assert result.rules_applied == ["terminal_period"]


def test_multi_sentence_imperative_openers_still_get_terminal_period() -> None:
    cases = [
        "Can you check the branch? Then rebase it onto develop",
        "What broke the build! Update the readme and ping the team",
        "Is it ready? Send the brief to Mira tonight",
    ]

    for text in cases:
        result = _apply(text)
        assert result.text == text + ".", text
        assert result.rules_applied == ["terminal_period"], text


def test_multi_sentence_question_tail_still_gets_question_mark() -> None:
    cases = [
        "I sent the brief. Can you review it before Friday",
        "The build failed! What did you change in the config",
        "Update the readme. Do you want me to bump the version",
    ]

    for text in cases:
        result = _apply(text)
        assert result.text == text + "?", text
        assert result.rules_applied == ["terminal_question"], text


def test_inline_marker_shaped_prose_still_gets_terminal_period() -> None:
    cases = [
        "option a. should remain available",
        "version 1. 2 should be safe",
        "this includes b) as a literal marker",
    ]

    for text in cases:
        result = _apply(text)
        assert result.text == text + "."
        assert result.changed is True
        assert result.rules_applied == ["terminal_period"]


def test_skips_single_line_structured_marker_sequences() -> None:
    result = _apply("1. first item 2. second item")

    assert result.text == "1. first item 2. second item"
    assert result.changed is False
    assert result.skip_reason == "structured_text"


def test_skips_short_command_shaped_utterances() -> None:
    result = _apply("new paragraph")

    assert result.text == "new paragraph"
    assert result.changed is False
    assert result.skip_reason == "short_utterance"


def test_skips_selection_and_raw_surfaces() -> None:
    selected = _apply("make this clearer for launch", selected_text="rough text")
    terminal = _apply("git commit dash m fix preview", app_category="terminal")

    assert selected.changed is False
    assert selected.skip_reason == "selection_present"
    assert terminal.changed is False
    assert terminal.skip_reason == "raw_surface"


def test_skips_messaging_and_continuation_tail() -> None:
    messaging = _apply("I can send it tomorrow", app_category="messaging", final_formatting_policy="messaging")
    continuation = _apply("I think we should send it to")

    assert messaging.changed is False
    assert messaging.skip_reason == "messaging_light"
    assert continuation.changed is False
    assert continuation.skip_reason == "continuation_tail"


def test_skips_existing_punctuation_and_structured_text() -> None:
    punctuated = _apply("already done.")
    structured = _apply("- first item\n- second item")

    assert punctuated.changed is False
    assert punctuated.skip_reason == "already_terminated"
    assert structured.changed is False
    assert structured.skip_reason == "structured_text"


def test_skips_unicode_terminal_marks() -> None:
    # Text already ending in a Unicode ellipsis or full-width/CJK terminal must
    # not get a redundant ASCII "." or "?" appended.
    for text in ("let me think about it…", "are we really done？", "今日は終わりだ。"):
        result = _apply(text)
        assert result.changed is False, text
        assert result.skip_reason == "already_terminated", text
