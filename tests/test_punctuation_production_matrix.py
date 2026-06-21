from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest

from juno_core_v3.dictation.self_corrections import apply_unambiguous_retakes
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.modes import ModeSelection, ModeSource
from juno_v2.contracts.workbench import ClientSelection, CommitMode
from juno_v2.contracts.writer import (
    WriterActionKind,
    WriterTransformRequest,
    WriterTransformResult,
)
from juno_v2.itn.engine import ITNEngine, ITNProfile
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.preview.orthography import normalize_preview_orthography
from juno_v2.turn_plan import TurnPlanResult
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.punctuation_controller import apply_final_punctuation_floor
from juno_v2.writer.service import WriterService


class _Recorder:
    def __init__(self) -> None:
        self.log_dir = tempfile.mkdtemp(prefix="juno-punctuation-matrix-")
        self.events: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(self, *args: object, **kwargs: object) -> None:
        self.events.append((args, kwargs))


class _Backend:
    def __init__(self, text: str = "Transformed selected text") -> None:
        self.text = text
        self.requests: list[WriterTransformRequest] = []

    def warm(self) -> None:
        return None

    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        self.requests.append(req)
        text = "VERDICT: clean" if req.instruction == "dictation_edit" else self.text
        return WriterTransformResult(
            utterance_id=req.utterance_id,
            text=text,
            backend_name="matrix-backend",
            decode_ms=1.0,
        )


@dataclass
class _Snippet:
    trigger: str
    body: str
    scope: str = "global"
    case_sensitive: bool = False


class _SnippetResolver:
    def __init__(self, snippets: list[_Snippet]) -> None:
        self._snippets = snippets

    def list(self) -> list[_Snippet]:
        return list(self._snippets)

    def resolve(self, trigger: str, *, scope: str = "global") -> _Snippet | None:
        folded = trigger.casefold()
        for snippet in self._snippets:
            if snippet.scope in {scope, "global"} and snippet.trigger.casefold() == folded:
                return snippet
        return None


class _MemoryStore:
    def __init__(self, snippets: list[_Snippet] | None = None) -> None:
        self.snippets = _SnippetResolver(snippets or [])
        self.replacements: list[tuple[str, str, str]] = []
        self.lexicon: list[tuple[str, str, str]] = []

    def add_replacement(self, *, trigger: str, replacement: str, source: str) -> None:
        self.replacements.append((trigger, replacement, source))

    def add_lexicon_entry(self, *, term: str, canonical_form: str, aliases: list[str], source: str, **_: Any) -> None:
        self.lexicon.append((term, canonical_form, source))


def _selection(mode_name: str = "default_surface", *, source: ModeSource = ModeSource.AUTO, custom: str | None = None) -> ModeSelection:
    return ModeSelection(
        effective_mode=custom or mode_name,
        mode_source=source,
        manual_mode_name=mode_name if source == ModeSource.MANUAL else None,
        custom_mode_name=custom,
        resolved_from_surface=None,
    )


def _service(*, backend: _Backend | None = None, editor: bool = False) -> WriterService:
    return WriterService(
        config=WriterConfig(enable_model_transforms=True, dictation_editor_enabled=editor),
        recorder=_Recorder(),
        backend=backend,
    )


def _process(
    text: str,
    *,
    service: WriterService | None = None,
    context: TypedContextBundle | None = None,
    anchor_selection: ClientSelection | None = None,
    memory_store: Any = None,
    mode_policy=None,
    mode_selection: ModeSelection | None = None,
    partial_text: str | None = None,
    wake_verified: bool = False,
    turn_plan_result: TurnPlanResult | None = None,
    writer_tone_addon: str | None = None,
):
    policy = mode_policy or BUILTIN_MODES["default_surface"]
    selection = mode_selection or _selection(policy.base_mode)
    return (service or _service()).process_transcript(
        utterance_id="utt-punctuation-matrix",
        final_text=text,
        raw_text=text,
        context=context or TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=anchor_selection,
        memory_store=memory_store,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=policy,
        mode_selection=selection,
        partial_text=partial_text,
        wake_verified=wake_verified,
        turn_plan_result=turn_plan_result,
        writer_tone_addon=writer_tone_addon,
    )


HUD_CASES = [
    ("hello comma world period this is next", "", "Hello comma world period this is next", "", 2),
    ("hello new paragraph next point", "", "Hello new paragraph next point", "", 1),
    ("hello new line next point", "", "Hello new line next point", "", 1),
    ("hello line break next point", "", "Hello line break next point", "", 1),
    ("are we ready question mark yes exclamation point", "", "Are we ready question mark yes exclamation point", "", 2),
    ("status colon blocked semicolon owner pending", "", "Status colon blocked semicolon owner pending", "", 2),
    ("first comma second comma third period", "", "First comma second comma third period", "", 3),
    ("warning exclamation mark do not deploy", "", "Warning exclamation mark do not deploy", "", 1),
    ("warning exclamation point do not deploy", "", "Warning exclamation point do not deploy", "", 1),
    ("use full stop in the label", "", "Use full stop in the label", "", 0),
    ("the word period means a dot", "", "The word period means a dot", "", 0),
    ("comma separated values are hard", "", "Comma separated values are hard", "", 0),
    ("the new paragraph is short and a comma goes here", "", "The new paragraph is short and a comma goes here", "", 0),
    ("that new line looks wrong", "", "That new line looks wrong", "", 0),
    ("please write hello comma then stop", "", "Please write hello comma then stop", "", 1),
    ("the ratio is one colon two", "", "The ratio is one colon two", "", 1),
    ("open risks new paragraph mitigations comma owners period", "", "Open risks new paragraph mitigations comma owners period", "", 3),
    ("alpha semicolon beta period gamma", "", "Alpha semicolon beta period gamma", "", 2),
    ("we shipped period", "now test comma then report", "We shipped period", "now test comma then report", 2),
    ("we are ready question mark", "yes exclamation mark", "We are ready question mark", "yes exclamation mark", 2),
    ("", "hello comma tail period", "", "Hello comma tail period", 2),
    ("hello full stop next", "", "Hello full stop next", "", 1),
    ("the full stop is in the style guide", "", "The full stop is in the style guide", "", 0),
    ("send it comma if the tests pass", "", "Send it comma if the tests pass", "", 1),
    ("the release date is friday question mark", "", "The release date is friday question mark", "", 1),
    ("alpha colon beta colon gamma", "", "Alpha colon beta colon gamma", "", 2),
    ("please add a comma after hello", "", "Please add a comma after hello", "", 0),
    ("my period key broke", "", "My period key broke", "", 0),
]


@pytest.mark.parametrize(
    ("committed", "tail", "expected_committed", "expected_tail", "expected_cues"),
    HUD_CASES,
)
def test_live_hud_spoken_punctuation_matrix(
    committed: str,
    tail: str,
    expected_committed: str,
    expected_tail: str,
    expected_cues: int,
) -> None:
    got_committed, got_tail, meta = normalize_preview_orthography(committed, tail)

    assert got_committed == expected_committed
    assert got_tail == expected_tail
    assert meta["preview_spoken_punctuation_cues"] == expected_cues


FINAL_FLOOR_CASES = [
    ("period_docs", "send the brief to Mira tonight", {}, "send the brief to Mira tonight.", True, "terminal_period"),
    ("question_can", "can you send the brief tomorrow", {}, "can you send the brief tomorrow?", True, "terminal_question"),
    ("question_should", "should we ship this tonight", {}, "should we ship this tonight?", True, "terminal_question"),
    ("question_where", "where is the launch checklist now", {}, "where is the launch checklist now?", True, "terminal_question"),
    ("wh_statement", "what we need is more testing", {}, "what we need is more testing.", True, "terminal_period"),
    ("already_period", "already done.", {}, "already done.", False, "already_terminated"),
    ("already_question", "are we done?", {}, "are we done?", False, "already_terminated"),
    ("already_bang", "ship it!", {}, "ship it!", False, "already_terminated"),
    ("already_colon", "status:", {}, "status:", False, "already_terminated"),
    ("already_paren", "ship the test build)", {}, "ship the test build)", False, "already_terminated"),
    ("short_new_para", "new paragraph", {}, "new paragraph", False, "short_utterance"),
    ("short_transform", "make that shorter", {}, "make that shorter", False, "short_utterance"),
    ("continuation_to", "I think we should send it to", {}, "I think we should send it to", False, "continuation_tail"),
    ("continuation_and", "we need tests and", {}, "we need tests and", False, "continuation_tail"),
    ("newline_structured", "first line\nsecond line", {}, "first line\nsecond line", False, "structured_text"),
    ("bullet_structured", "- ship build\n- test preview", {}, "- ship build\n- test preview", False, "structured_text"),
    ("numbered_structured", "1. ship build\n2. test preview", {}, "1. ship build\n2. test preview", False, "structured_text"),
    ("selection_active", "make this clearer for launch", {"selection_active": True}, "make this clearer for launch", False, "selection_present"),
    ("selected_text", "make this clearer for launch", {"selected_text": "rough copy"}, "make this clearer for launch", False, "selection_present"),
    ("wake_verified", "set an alarm for four pm", {"wake_verified": True}, "set an alarm for four pm", False, "wake_verified"),
    ("snippet_expanded", "please add Best,\nJuno", {"snippet_expanded": True}, "please add Best,\nJuno", False, "snippet_expanded"),
    ("code_surface", "return false from the handler", {"app_category": "code"}, "return false from the handler", False, "raw_surface"),
    ("terminal_surface", "git status and then make test", {"app_category": "terminal"}, "git status and then make test", False, "raw_surface"),
    ("dev_tools_surface", "reload the console after changing flags", {"app_category": "developer_tools"}, "reload the console after changing flags", False, "raw_surface"),
    ("messaging_surface", "I can send it tomorrow", {"app_category": "messaging"}, "I can send it tomorrow", False, "messaging_light"),
    ("messaging_policy", "I can send it tomorrow", {"final_formatting_policy": "messaging"}, "I can send it tomorrow", False, "messaging_light"),
    ("light_policy", "I can send it tomorrow", {"punctuation_policy": "light"}, "I can send it tomorrow", False, "messaging_light"),
    ("none_policy", "send the brief to Mira tonight", {"punctuation_policy": "none"}, "send the brief to Mira tonight", False, "policy_no_punctuation"),
    ("literal_policy", "send the brief to Mira tonight", {"punctuation_policy": "literal_minimal"}, "send the brief to Mira tonight", False, "policy_no_punctuation"),
    ("verbatim_mode", "send the brief to Mira tonight", {"writer_mode": "verbatim"}, "send the brief to Mira tonight", False, "mode_no_punctuation"),
    ("command_mode", "send the brief to Mira tonight", {"writer_mode": "command_mode"}, "send the brief to Mira tonight", False, "mode_no_punctuation"),
    ("email_surface", "please find the launch notes attached", {"app_category": "email", "punctuation_policy": "strong"}, "please find the launch notes attached.", True, "terminal_period"),
    ("unknown_surface", "the build passed every regression gate", {"app_category": "unknown"}, "the build passed every regression gate.", True, "terminal_period"),
    ("imperative_do_command", "do this once after everything run an E2E test and create a PR for main", {}, "do this once after everything run an E2E test and create a PR for main.", True, "terminal_period"),
    ("why_statement_fragment", "why it ended the previous paste with a question mark", {}, "why it ended the previous paste with a question mark.", True, "terminal_period"),
    ("how_statement_fragment", "how this happened in the final paste", {}, "how this happened in the final paste.", True, "terminal_period"),
    ("clear_why_question", "why did it end with a question mark", {}, "why did it end with a question mark?", True, "terminal_question"),
]


@pytest.mark.parametrize(("name", "text", "overrides", "expected", "changed", "reason"), FINAL_FLOOR_CASES)
def test_final_paste_punctuation_floor_matrix(
    name: str,
    text: str,
    overrides: dict[str, object],
    expected: str,
    changed: bool,
    reason: str,
) -> None:
    params: dict[str, object] = {
        "app_category": "docs",
        "writer_mode": "default_surface",
        "punctuation_policy": "standard",
        "final_formatting_policy": "minimal",
    }
    params.update(overrides)

    result = apply_final_punctuation_floor(text, **params)  # type: ignore[arg-type]

    assert result.text == expected, name
    assert result.changed is changed, name
    if changed:
        assert result.rules_applied == [reason], name
        assert result.skip_reason is None, name
    else:
        assert result.skip_reason == reason, name


FINAL_ITN_CASES = [
    ("hello comma world period", "hello, world."),
    ("first point new line second point", "first point\nsecond point"),
    ("between models and people, new paragraph but voice is how we think.", "between models and people\n\nbut voice is how we think."),
    ("hold on, period", "hold on."),
    ("the new paragraph is short", "the new paragraph is short"),
    ("a comma goes here", "a comma goes here"),
]


@pytest.mark.parametrize(("spoken", "expected"), FINAL_ITN_CASES)
def test_final_spoken_punctuation_itn_matrix(spoken: str, expected: str) -> None:
    assert ITNEngine().run(spoken, profile=ITNProfile("prose")).text == expected


def _action_turn_plan(source: str) -> TurnPlanResult:
    return TurnPlanResult(
        plan={
            "schema_version": "turn_plan_v1",
            "utterance_kind": "actions",
            "corrected_transcript": {"text": source},
            "target": {"kind": "none", "confidence": 1.0},
            "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
            "transform": {
                "operation": "none",
                "instruction": "",
                "transformed_text": None,
                "requires_second_pass": False,
            },
            "actions": [
                {
                    "kind": "note",
                    "operation": "create",
                    "body": "launch window is tomorrow",
                    "evidence_span": "take a note launch window is tomorrow",
                    "schedule": {"kind": "none"},
                    "missing_fields": [],
                }
            ],
            "snippets": [],
            "memory_candidates": [],
            "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
            "uncertainties": [],
        },
        status="ok",
        backend_name="matrix-plan",
        decode_ms=1.0,
    )


SERVICE_CASES = [
    ("plain_docs_period", "send the brief to Mira tonight", {}, WriterActionKind.PASS_THROUGH_COMMIT, "Send the brief to Mira tonight."),
    ("plain_docs_question", "can you send the brief tomorrow", {}, WriterActionKind.PASS_THROUGH_COMMIT, "Can you send the brief tomorrow?"),
    (
        "code_surface_no_period",
        "return false from the handler",
        {"context": TypedContextBundle(app_name="Xcode", app_category="code")},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "return false from the handler",
    ),
    (
        "terminal_surface_no_period",
        "git status and then make test",
        {"context": TypedContextBundle(app_name="Terminal", app_category="terminal")},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "git status and then make test",
    ),
    (
        "messaging_surface_no_period",
        "I can send it tomorrow",
        {"context": TypedContextBundle(app_name="Messages", app_category="messaging")},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "I can send it tomorrow",
    ),
    (
        "casual_mode_no_period",
        "I can send it tomorrow",
        {
            "context": TypedContextBundle(app_name="Slack", app_category="messaging"),
            "mode_policy": BUILTIN_MODES["casual_chat"],
            "mode_selection": _selection("casual_chat", source=ModeSource.MANUAL),
        },
        WriterActionKind.PASS_THROUGH_COMMIT,
        "I can send it tomorrow",
    ),
    (
        "formal_email_period",
        "please find the launch notes attached",
        {
            "context": TypedContextBundle(app_name="Mail", app_category="email"),
            "mode_policy": BUILTIN_MODES["formal_email"],
            "mode_selection": _selection("formal_email", source=ModeSource.MANUAL),
        },
        WriterActionKind.PASS_THROUGH_COMMIT,
        "Please find the launch notes attached.",
    ),
    (
        "custom_mode_preserves_punctuation_floor",
        "capture launch blockers before the review",
        {
            "mode_policy": replace(
                BUILTIN_MODES["default_surface"],
                mode_name="Launch Notes",
                prompt_prefix="Use terse launch-note style.",
            ),
            "mode_selection": _selection("default_surface", source=ModeSource.CUSTOM, custom="Launch Notes"),
            "writer_tone_addon": "Prefer terse launch notes.",
        },
        WriterActionKind.PASS_THROUGH_COMMIT,
        "Capture launch blockers before the review.",
    ),
    (
        "verbatim_no_period",
        "send the brief to Mira tonight",
        {"mode_policy": BUILTIN_MODES["verbatim"], "mode_selection": _selection("verbatim", source=ModeSource.MANUAL)},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "send the brief to Mira tonight",
    ),
    (
        "command_mode_long_dictation_no_period",
        "this is a long fallback dictation in command mode and it should stay exact",
        {"mode_policy": BUILTIN_MODES["command_mode"], "mode_selection": _selection("command_mode", source=ModeSource.MANUAL)},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "This is a long fallback dictation in command mode and it should stay exact",
    ),
    ("new_paragraph_command", "New paragraph", {}, WriterActionKind.DIRECT_COMMIT, "\n\n"),
    ("new_line_command", "New line", {}, WriterActionKind.DIRECT_COMMIT, "\n"),
    ("next_bullet_command", "Next bullet", {}, WriterActionKind.DIRECT_COMMIT, "\n- "),
    ("start_bullet_mode", "Start bullet list", {}, WriterActionKind.STATE_MUTATION, ""),
    ("stop_bullet_mode", "Stop bullet list", {}, WriterActionKind.STATE_MUTATION, ""),
    (
        "explicit_bullet_list",
        "Start a bullet list. First verify microphone permission. Second run action combos. Third check final paste.",
        {},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "- verify microphone permission\n- run action combos\n- check final paste",
    ),
    (
        "natural_bullet_list",
        "I think we need to focus on 3 things, first is that we test the HUD and second is that we verify final paste.",
        {},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "- we test the HUD\n- we verify final paste",
    ),
    (
        "numbered_structural_list",
        "Note down 3 points. First ship the build. Second verify punctuation. Third monitor history.",
        {},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "1. ship the build\n2. verify punctuation\n3. monitor history",
    ),
    (
        "direct_snippet_insert",
        "use signoff",
        {"memory_store": _MemoryStore([_Snippet("signoff", "Best,\nJuno")])},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "Best,\nJuno",
    ),
    (
        "inline_snippet_expansion_no_extra_period",
        "please add signoff",
        {"memory_store": _MemoryStore([_Snippet("signoff", "Best,\nJuno")])},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "Please add Best, Juno",
    ),
    (
        "selection_to_bullets",
        "turn this into bullets",
        {
            "context": TypedContextBundle(app_name="Notes", app_category="docs", selected_text="alpha. beta."),
            "anchor_selection": ClientSelection(start=10, end=22),
        },
        WriterActionKind.TRANSFORM_COMMIT,
        "- alpha\n- beta",
    ),
    (
        "selection_model_transform",
        "make this clearer",
        {
            "service": _service(backend=_Backend("Selected text improved")),
            "context": TypedContextBundle(app_name="Notes", app_category="docs", selected_text="rough selected text"),
            "anchor_selection": ClientSelection(start=0, end=19),
        },
        WriterActionKind.TRANSFORM_COMMIT,
        "Selected text improved",
    ),
    (
        "recent_target_transform",
        "make that shorter",
        {
            "service": _service(backend=_Backend("Shorter text")),
            "context": TypedContextBundle(
                app_name="Notes",
                app_category="docs",
                metadata={"last_committed_text": "This is a very long sentence.", "last_committed_start": 5, "last_committed_end": 34},
            ),
        },
        WriterActionKind.TRANSFORM_COMMIT,
        "Shorter text",
    ),
    (
        "memory_replacement_command",
        "change Chino to Juno",
        {"memory_store": _MemoryStore()},
        WriterActionKind.MEMORY_MUTATION,
        "",
    ),
    (
        "prose_change_to_stays_dictation",
        "We should change these two buttons to What is Juno and Quickstart instead of the old labels",
        {},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "We should change these two buttons to What is Juno and Quickstart instead of the old labels.",
    ),
    (
        "insert_current_file_path",
        "insert current file path",
        {"context": TypedContextBundle(app_name="Xcode", app_category="code", focused_file_path="/tmp/Juno/App.swift")},
        WriterActionKind.DIRECT_COMMIT,
        "/tmp/Juno/App.swift",
    ),
    (
        "actions_plan_falls_back_to_text_not_silence",
        "take a note launch window is tomorrow",
        {"turn_plan_result": _action_turn_plan("take a note launch window is tomorrow")},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "Take a note launch window is tomorrow.",
    ),
    (
        "wake_action_fallback_does_not_add_period",
        "set an alarm for four pm to launch Juno",
        {"wake_verified": True, "turn_plan_result": _action_turn_plan("set an alarm for four pm to launch Juno")},
        WriterActionKind.PASS_THROUGH_COMMIT,
        "Set an alarm for four pm to launch Juno",
    ),
]


@pytest.mark.parametrize(("name", "spoken", "kwargs", "action", "expected"), SERVICE_CASES)
def test_writer_service_routing_matrix(
    name: str,
    spoken: str,
    kwargs: dict[str, object],
    action: WriterActionKind,
    expected: str,
) -> None:
    result = _process(spoken, **kwargs)

    assert result.action == action, name
    assert result.output_text == expected, name
    if result.action in {WriterActionKind.DIRECT_COMMIT, WriterActionKind.TRANSFORM_COMMIT}:
        assert result.metadata.get("punctuation_floor") is None, name
    if name == "custom_mode_preserves_punctuation_floor":
        assert result.custom_mode_name == "Launch Notes"
    if name == "memory_replacement_command":
        assert kwargs["memory_store"].replacements == [("Chino", "Juno", "voice_command")]  # type: ignore[index,union-attr]
    if name == "wake_action_fallback_does_not_add_period":
        assert result.metadata["punctuation_floor"]["skip_reason"] == "wake_verified"


def test_self_correction_retake_then_final_punctuation() -> None:
    corrected, applied = apply_unambiguous_retakes(
        "send it on Tuesday scratch that send it on Wednesday morning please"
    )

    assert applied
    result = _process(corrected)

    assert result.output_text == "Send it on Wednesday morning please."


def test_matrix_has_at_least_fifty_explicit_scenarios() -> None:
    total = len(HUD_CASES) + len(FINAL_FLOOR_CASES) + len(FINAL_ITN_CASES) + len(SERVICE_CASES) + 1
    assert total >= 50
