"""Dictation editor lane — parser, applier guards, service + pipeline wiring."""
from __future__ import annotations

from typing import Any

from juno_core_v3.dictation.pipeline import OneShotDictationPipeline
from juno_core_v3.dictation.transcriber import TranscribeResult
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.modes import ModeSelection, ModeSource
from juno_v2.contracts.writer import (
    WriterActionKind,
    WriterTransformRequest,
    WriterTransformResult,
)
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.transcript.adjudicator import TranscriptAdjudicatorConfig
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.dictation_editor import (
    apply_edit_script,
    parse_edit_script,
)
from juno_v2.writer.service import WriterService

import tempfile


class _Recorder:
    def __init__(self) -> None:
        self.log_dir = tempfile.mkdtemp(prefix="juno-editor-test-")
        self.events: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(self, *args: object, **kwargs: object) -> None:
        self.events.append((args, kwargs))


class _EditorBackend:
    """Returns a scripted editor output for dictation_edit_v1 requests."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[WriterTransformRequest] = []

    def warm(self) -> None:
        return None

    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        self.requests.append(req)
        return WriterTransformResult(
            utterance_id=req.utterance_id,
            text=self.output,
            backend_name="fake-editor",
            decode_ms=5.0,
        )


def _selection() -> ModeSelection:
    return ModeSelection(
        effective_mode="default_surface",
        mode_source=ModeSource.AUTO,
        manual_mode_name=None,
        custom_mode_name=None,
        resolved_from_surface=None,
    )


def _editor_service(output: str) -> tuple[WriterService, _EditorBackend, _Recorder]:
    backend = _EditorBackend(output)
    recorder = _Recorder()
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True, dictation_editor_enabled=True),
        recorder=recorder,
        backend=backend,
    )
    return service, backend, recorder


def _process(service: WriterService, text: str, **kwargs: Any):
    return service.process_transcript(
        utterance_id="utt-editor",
        final_text=text,
        raw_text=kwargs.pop("raw_text", text),
        context=kwargs.pop("context", TypedContextBundle(app_name="Notes", app_category="docs")),
        anchor_selection=None,
        memory_store=kwargs.pop("memory_store", None),
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Snippet expansion through the dictation-editor path (regression)
# --------------------------------------------------------------------------- #

from dataclasses import dataclass as _dataclass  # noqa: E402
from types import SimpleNamespace as _SimpleNamespace  # noqa: E402


@_dataclass
class _Snip:
    trigger: str
    body: str
    scope: str = "global"
    case_sensitive: bool = False


class _ListResolver:
    """Minimal SnippetResolver with the ``list()`` API the writer prefers."""

    def __init__(self, snips: list[_Snip]) -> None:
        self._snips = snips

    def list(self) -> list[_Snip]:
        return list(self._snips)

    def resolve(self, trigger: str, *, scope: str = "global") -> _Snip | None:
        return None


def _store_with_snippets(*snips: _Snip):
    return _SimpleNamespace(snippets=_ListResolver(list(snips)))


def test_dictation_editor_path_expands_user_snippets() -> None:
    """A saved snippet must still expand when the AI dictation editor is on.

    Production runs with ``dictation_editor_enabled`` (JUNO_V2_DICTATION_EDITOR=1).
    The editor path returns its result before the deterministic
    ``expand_snippets`` step, so user snippets never came up in the real app
    (they only expanded in the editor-off deterministic pipeline that the
    other tests exercise). Editor returns the spoken text unchanged
    ("VERDICT: clean"); the stored ``signoff`` snippet body must still appear.
    """
    service, _backend, _ = _editor_service("VERDICT: clean")
    store = _store_with_snippets(_Snip("signoff", "Best, Juno"))

    result = _process(service, "add my signoff", memory_store=store)

    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert "Best, Juno" in result.output_text, (
        f"snippet did not expand through the editor path: {result.output_text!r}"
    )


def test_structured_new_paragraph_dictation_skips_editor() -> None:
    service, backend, recorder = _editor_service(
        'VERDICT: edited\nEDIT: "final paste" => "HUD final"'
    )
    final_text = (
        "Here is the release plan\n\n"
        "first verify permissions\n\n"
        "second test, the HUD\n\n"
        "third test final paste"
    )
    raw_text = (
        "Here is the release plan, new paragraph, first verify permissions, "
        "new paragraph, second test, the HUD, new paragraph, third test final paste"
    )

    result = _process(service, final_text, raw_text=raw_text)

    assert backend.requests == []
    assert result.output_text == final_text
    assert result.metadata.get("reason") != "dictation_editor"
    assert any(
        args[1] == "dictation_edit_bypassed"
        and args[2]["reason"] == "structured_paragraph_text"
        for args, _kwargs in recorder.events
    )


def test_literal_new_paragraph_phrase_can_still_use_editor() -> None:
    service, backend, _recorder = _editor_service("VERDICT: clean")

    result = _process(service, "the new paragraph is short")

    assert len(backend.requests) == 1
    assert result.metadata["reason"] == "dictation_editor"
    assert result.output_text == "the new paragraph is short."


def test_same_utterance_bullet_list_command_bypasses_editor() -> None:
    service, backend, _ = _editor_service("VERDICT: clean")

    result = _process(
        service,
        "Start a bullet list. First verify microphone permission. Second run action combos. Third check final paste.",
    )

    assert backend.requests == []
    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == (
        "- verify microphone permission\n"
        "- run action combos\n"
        "- check final paste"
    )
    assert result.metadata["dictation_cleanup"]["pipeline"] == "explicit_same_utterance_bullet_list"


def test_natural_counted_ordinal_list_bypasses_editor() -> None:
    service, backend, _ = _editor_service("VERDICT: clean")

    result = _process(
        service,
        "I think we need to focus on 3 things, first is that we check everything properly "
        "before production and second is that we go to a party after we push things live.",
    )

    assert backend.requests == []
    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == (
        "- we check everything properly before production\n"
        "- we go to a party after we push things live"
    )
    cleanup = result.metadata["dictation_cleanup"]
    assert cleanup["pipeline"] == "natural_ordinal_bullet_list"
    assert cleanup["claimed_item_count"] == 3
    assert cleanup["spoken_item_count"] == 2
    assert cleanup["claimed_count_mismatch"] is True


def test_natural_counted_ordinal_list_skips_no_touch_surface() -> None:
    service, backend, _ = _editor_service("VERDICT: clean")
    spoken = (
        "I think we need to focus on 3 things, first is that we check everything properly "
        "before production and second is that we go to a party after we push things live."
    )

    result = _process(
        service,
        spoken,
        context=TypedContextBundle(app_name="Xcode", app_category="code"),
    )

    assert backend.requests == []
    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert "- " not in result.output_text
    assert "\n1." not in result.output_text
    assert "first is that" in result.output_text.lower()
    assert "second is that" in result.output_text.lower()


def test_explicit_structural_list_uses_deterministic_fallback_when_turn_planner_disabled() -> None:
    service, backend, _ = _editor_service("VERDICT: clean")

    result = _process(
        service,
        "Note down 10 points. First remove patches. Second make Qwen plan. Third validate spans.",
    )

    assert backend.requests == []
    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "1. remove patches\n2. make Qwen plan\n3. validate spans"
    assert result.metadata["dictation_cleanup"]["pipeline"] == "deterministic_structural_no_planner"
    assert result.metadata["turn_plan"]["backend"] == "deterministic_structural_no_planner"


# --------------------------------------------------------------------------- #
# Parser + applier
# --------------------------------------------------------------------------- #


def test_parse_and_apply_edit_and_delete() -> None:
    src = "I told him we got the budget approved, I mean the headcount approved, so we can hire."
    script = parse_edit_script(
        'VERDICT: edited\nDELETE: "the budget approved, I mean"\n'
    )
    assert script is not None
    out, applied = apply_edit_script(src, script)
    assert out == "I told him we got the headcount approved, so we can hire."
    assert applied["deletes"] == 1


def test_apply_word_fix_edit() -> None:
    src = "I don't know how God committed in the final text."
    script = parse_edit_script('VERDICT: edited\nEDIT: "God committed" => "got committed"')
    assert script is not None
    out, _ = apply_edit_script(src, script)
    assert out == "I don't know how got committed in the final text."


def test_lettered_structure_from_spoken_items() -> None:
    src = "I am thinking about three things, a, get the deck done, b, ship the fix, c, email Sam."
    script = parse_edit_script(
        "VERDICT: edited\n"
        "STRUCT: lettered\n"
        'ITEM: "get the deck done"\n'
        'ITEM: "ship the fix"\n'
        'ITEM: "email Sam"\n'
        'DELETE: "I am thinking about three things,"\n'
    )
    assert script is not None
    out, applied = apply_edit_script(src, script)
    assert applied["struct"] == "lettered"
    lines = out.splitlines()
    assert lines[0] == "a. get the deck done"
    assert lines[1] == "b. ship the fix"
    assert lines[2].startswith("c. email Sam")


def test_ungrounded_anchor_is_skipped_not_fatal() -> None:
    src = "ship the build tonight"
    script = parse_edit_script(
        'VERDICT: edited\nEDIT: "completely unrelated phrase" => "x"\nDELETE: "tonight"'
    )
    assert script is not None
    out, applied = apply_edit_script(src, script)
    assert out == "ship the build"
    assert applied["skipped"] == 1


def test_over_aggressive_script_is_defanged() -> None:
    src = "please send the launch summary to the whole team tonight after the standup"
    script = parse_edit_script(
        'VERDICT: edited\nEDIT: "please send the launch summary to the whole" => "x"\n'
        'DELETE: "team tonight after the standup"'
    )
    assert script is not None
    out, applied = apply_edit_script(src, script)
    # The unevidenced content DELETE is blocked; the tail survives.
    assert applied["skipped"] >= 1
    assert "team tonight after the standup" in out


def test_unparseable_output_returns_none() -> None:
    assert parse_edit_script("Sure! Here's the improved text: hello world") is None
    assert parse_edit_script("") is None


# --------------------------------------------------------------------------- #
# Service integration
# --------------------------------------------------------------------------- #


def test_editor_outcome_is_primary_for_dictation() -> None:
    service, backend, recorder = _editor_service(
        'VERDICT: edited\nEDIT: "God committed" => "got committed"'
    )
    result = _process(service, "I don't know how God committed in the final text.")
    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "I don't know how got committed in the final text."
    assert result.metadata["reason"] == "dictation_editor"
    assert backend.requests and backend.requests[0].context_payload["task"] == "dictation_edit_v1"
    names = [a[1] for a, _ in recorder.events if len(a) > 1]
    assert "dictation_edit_generated" in names


def test_editor_rejects_case_only_common_word_without_protected_evidence() -> None:
    service, _backend, _recorder = _editor_service(
        'VERDICT: edited\nEDIT: "stable" => "Stable"'
    )
    context = TypedContextBundle(
        app_name="Notes",
        app_category="docs",
        candidate_entities=["Stable Ihe"],
    )

    result = _process(
        service,
        "The first part is stable, the second part has commas",
        context=context,
    )

    assert result.output_text == "The first part is stable, the second part has commas."
    assert result.metadata["editor"]["applied"]["edits"] == 0
    assert result.metadata["editor"]["applied"]["skipped"] == 1


def test_editor_keeps_case_only_non_common_name_fix() -> None:
    script = parse_edit_script('VERDICT: edited\nEDIT: "rahul" => "Rahul"')
    assert script is not None

    out, applied = apply_edit_script(
        "ask rahul to check voting",
        script,
        protected_terms=["Rahul"],
    )

    assert out == "ask Rahul to check voting"
    assert applied["edits"] == 1


def test_editor_clean_verdict_gets_punctuation_floor() -> None:
    service, _, _ = _editor_service("VERDICT: clean")
    result = _process(service, "send the brief to Mira tonight")
    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "send the brief to Mira tonight."
    assert result.metadata["punctuation_floor"]["rules_applied"] == ["terminal_period"]


def test_editor_garbage_floors_to_legacy_lane() -> None:
    service, _, recorder = _editor_service("I rewrote everything for you!")
    result = _process(service, "send the brief to Mira tonight")
    # Floor: deterministic pass-through still delivers the text.
    assert result.output_text.strip()
    names = [a[1] for a, _ in recorder.events if len(a) > 1]
    assert "dictation_edit_floor" in names


def test_editor_skipped_for_wake_and_selection() -> None:
    service, backend, _ = _editor_service("VERDICT: clean")
    _process(service, "take a note buy milk", wake_verified=True)
    assert all(
        r.context_payload.get("task") != "dictation_edit_v1" for r in backend.requests
    )
    service2, backend2, _ = _editor_service("VERDICT: clean")
    _process(
        service2,
        "make this sharper",
        context=TypedContextBundle(app_name="Notes", app_category="docs", selected_text="rough text"),
    )
    assert all(
        r.context_payload.get("task") != "dictation_edit_v1" for r in backend2.requests
    )


# --------------------------------------------------------------------------- #
# Pipeline integration
# --------------------------------------------------------------------------- #


def test_pipeline_dictation_uses_editor_and_skips_final_adjudication() -> None:
    source = "I don't know how God committed in the final text but please check the logs"

    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript=source,
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(app_name="Notes", app_category="docs")

    import io
    import math
    import struct
    import wave

    def _wav() -> bytes:
        buf = io.BytesIO()
        frames = [
            struct.pack("<h", int(20000 * math.sin(i / 8.0))) for i in range(16000)
        ]
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(frames))
        return buf.getvalue()

    backend = _EditorBackend('VERDICT: edited\nEDIT: "God committed" => "got committed"')
    recorder = _Recorder()
    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, dictation_editor_enabled=True),
        recorder=_Recorder(),
        backend=backend,
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=True),
        itn_enabled=False,
    )
    result = pipeline.run(_wav(), utterance_id="utt-editor-pipe", save_history=False, save_audio=False)
    assert result.ok
    assert "got committed" in result.transcript
    assert "God committed" not in result.transcript
    assert result.paste_kind != "none"
    # Final 0.6B adjudication must be skipped — the editor owns correction.
    payloads = [a[2] for a, _ in recorder.events if len(a) > 2 and isinstance(a[2], dict)]
    assert any(
        p.get("skip_reason") == "dictation_editor_lane" or p.get("reason") == "dictation_editor_lane"
        for p in payloads
    ) or not any(
        a[1] == "oneshot_transcript_adjudication_started" and (a[2] or {}).get("stage") == "final"
        for a, _ in recorder.events
        if len(a) > 2
    )


# --------------------------------------------------------------------------- #
# Round-2 production regressions (2026-06-10 afternoon)
# --------------------------------------------------------------------------- #


def test_delete_without_evidence_is_skipped() -> None:
    # Production: the editor deleted the emphatic clause "that is just not
    # acceptable behavior" from a real paste. Content deletes need evidence.
    src = "it cut my note off that is just not acceptable behavior and there were issues"
    script = parse_edit_script('VERDICT: edited\nDELETE: "that is just not acceptable behavior"')
    assert script is not None
    out, applied = apply_edit_script(src, script)
    assert out == src
    assert applied["skipped"] == 1


def test_content_compressing_edit_without_evidence_is_skipped() -> None:
    # Production: the editor compressed the opening "make if there is no" to
    # "if no", making the committed text look like the first words were lost.
    src = "Thank you to make if there is no new repo exists and make a new repo"
    script = parse_edit_script('VERDICT: edited\nEDIT: "make if there is no" => "if no"')
    assert script is not None

    out, applied = apply_edit_script(src, script)

    assert out == src
    assert applied["skipped"] == 1


def test_one_token_content_drop_without_evidence_is_skipped() -> None:
    src = "Thank you to make if there is no new repo exists and make a new repo"
    script = parse_edit_script(
        'VERDICT: edited\nEDIT: "make if there is no" => "if there is no"'
    )
    assert script is not None

    out, applied = apply_edit_script(src, script)

    assert out == src
    assert applied["skipped"] == 1


def test_short_content_compressing_edit_without_evidence_is_skipped() -> None:
    src = "Please create new repo and push it"
    script = parse_edit_script('VERDICT: edited\nEDIT: "create new repo" => "create"')
    assert script is not None

    out, applied = apply_edit_script(src, script)

    assert out == src
    assert applied["skipped"] == 1


def test_stutter_collapse_edit_still_applies() -> None:
    src = "Please create create new repo and push it"
    script = parse_edit_script('VERDICT: edited\nEDIT: "create create" => "create"')
    assert script is not None

    out, applied = apply_edit_script(src, script)

    assert out == "Please create new repo and push it"
    assert applied["edits"] == 1


def test_content_compressing_edit_with_correction_marker_still_applies() -> None:
    src = "I told him we got the budget approved, I mean, so we can hire."
    script = parse_edit_script(
        'VERDICT: edited\nEDIT: "the budget approved, I mean" => "the headcount approved"'
    )
    assert script is not None

    out, applied = apply_edit_script(src, script)

    assert out == "I told him we got the headcount approved, so we can hire."
    assert applied["edits"] == 1


def test_delete_with_marker_or_restart_still_applies() -> None:
    src = "I told him we got the budget approved, I mean the headcount approved, so we can hire."
    script = parse_edit_script('VERDICT: edited\nDELETE: "the budget approved, I mean"')
    assert script is not None
    out, _ = apply_edit_script(src, script)
    assert "budget" not in out
    src2 = "send the deck to Priya send the deck to Mira tonight"
    script2 = parse_edit_script('VERDICT: edited\nDELETE: "send the deck to Priya"')
    assert script2 is not None
    out2, _ = apply_edit_script(src2, script2)
    assert out2 == "send the deck to Mira tonight"


def test_sentence_starts_capitalized_after_terminal_punct() -> None:
    from juno_v2.writer.dictation_editor import capitalize_sentence_starts

    assert (
        capitalize_sentence_starts("broke in a bunch of places. more importantly it cut my note off")
        == "broke in a bunch of places. More importantly it cut my note off"
    )
    assert capitalize_sentence_starts("use e.g. lowercase here") == "use e.g. lowercase here"


def test_note_body_sliced_from_source_when_model_truncates() -> None:
    from juno_v2.turn_plan.planner import normalize_turn_plan

    source = (
        "take a note titled What is Juno and why does this exist? Text is "
        "still the main interface between people and models but voice is how "
        "we think. Juno exists to close that gap. Set an alarm for 4pm to "
        "launch Juno on Product Hunt and remind me to post on X"
    )
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                # model truncated its re-typed body (production bug)
                "body": "What is Juno and why does this exist? Text is still the main interface between",
                "evidence_span": "take a note titled What is Juno and why does this exist? Text is still the main interface between",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }
    normalized, notes = normalize_turn_plan(plan, source_text=source)
    body = str(normalized["actions"][0]["body"])
    assert "note_body_sliced_from_source" in notes
    assert body.endswith("close that gap"), body
    assert "Set an alarm" not in body


def test_hud_window_join_smoothing_render_helper() -> None:
    # Render-layer helper only: never applied to the emitted committed
    # stream (that broke the HUD's append-only contract in production).
    from juno_v2.preview.orthography import _smooth_window_joins

    out = _smooth_window_joins("voice is how we think. ask, interrupt ourselves. and move")
    assert ". ask" not in out and ". and" not in out
    out2 = _smooth_window_joins("why does this exist? new Line Text is still")
    assert "new Line" not in out2 and "\n" in out2
    assert _smooth_window_joins("the new line is short") == "the new line is short"


# --------------------------------------------------------------------------- #
# Route ordering (2026-06-11 production matrix failures)
# --------------------------------------------------------------------------- #


def test_commands_bypass_editor_and_produce_results() -> None:
    expected = {
        "New paragraph": "\n\n",
        "Next bullet": "\n- ",
        "Make that shorter": None,  # routed to recent-target lane, not literal paste
    }
    for spoken, want in expected.items():
        service, backend, recorder = _editor_service("VERDICT: clean")
        result = _process(service, spoken)
        names = [a[1] for a, _ in recorder.events if len(a) > 1]
        assert "dictation_edit_bypassed" in names, spoken
        assert not backend.requests, f"editor must not run for {spoken!r}"
        assert result.output_text != spoken, f"{spoken!r} pasted literally"
        if want is not None:
            assert result.output_text == want


def test_pipeline_new_paragraph_survives_itn_and_executes() -> None:
    # Production case 7: ITN collapsed "New Paragraph" to "\n\n" before the
    # parser, the writer saw an empty command, and the turn died as
    # unsupported_intent. The pure-command claim must run before ITN.
    source = "New Paragraph"

    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript=source,
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=600.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(app_name="Notes", app_category="docs")

    import io
    import math
    import struct
    import wave

    def _wav() -> bytes:
        buf = io.BytesIO()
        frames = [struct.pack("<h", int(18000 * math.sin(i / 9.0))) for i in range(9600)]
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(frames))
        return buf.getvalue()

    recorder = _Recorder()
    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, dictation_editor_enabled=True),
        recorder=_Recorder(),
        backend=_EditorBackend("VERDICT: clean"),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=True,
    )
    result = pipeline.run(_wav(), utterance_id="utt-new-paragraph", save_history=False, save_audio=False)
    assert result.ok
    assert result.transcript == "\n\n"
    assert result.paste_kind != "none"
    names = [a[1] for a, _ in recorder.events if len(a) > 1]
    assert "oneshot_itn_skipped_for_command" in names


def test_rejected_action_response_carries_recoverable_transcript() -> None:
    # Safety requirement: a rejected wake/action turn must never silently
    # erase the words — the response payload carries them for the shell's
    # copy surface, in addition to the History row.
    source = "juno set an alarm to publish the changelog"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": "set an alarm to publish the changelog"},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "alarm",
                "operation": "create",
                "body": "publish the changelog",
                "evidence_span": "set an alarm to publish the changelog",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript=source,
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=900.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(app_name="Notes", app_category="docs")

    import io
    import math
    import struct
    import wave

    def _wav() -> bytes:
        buf = io.BytesIO()
        frames = [struct.pack("<h", int(18000 * math.sin(i / 9.0))) for i in range(9600)]
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(frames))
        return buf.getvalue()

    from juno_v2.writer.service import WriterService as _WS

    writer = _WS(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=None,
    )
    # Force the turn-plan path with a fake planner backend.
    from tests.test_qwen_turn_planner import _TurnPlanBackend  # type: ignore

    writer.backend = _TurnPlanBackend(plan)
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=_Recorder(),
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )
    result = pipeline.run(_wav(), utterance_id="utt-recoverable", save_history=False, save_audio=False)
    assert result.ok
    assert result.paste_kind == "none"
    assert result.transcript == ""
    assert "publish the changelog" in result.recoverable_transcript
    assert result.to_dict().get("recoverable_transcript")
    # Issue-2: the recoverable transcript is wake-stripped so the shell pastes
    # the user's actual words ("set an alarm to publish the changelog") into the
    # focused surface, not "juno set an alarm ...". The wake word was only the
    # address, never content.
    assert "juno" not in result.recoverable_transcript.lower().split()
    assert result.recoverable_transcript.lower().startswith("set an alarm")
