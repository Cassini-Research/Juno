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
        raw_text=text,
        context=kwargs.pop("context", TypedContextBundle(app_name="Notes", app_category="docs")),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
        **kwargs,
    )


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


def test_editor_clean_verdict_passes_text_through() -> None:
    service, _, _ = _editor_service("VERDICT: clean")
    result = _process(service, "send the brief to Mira tonight")
    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "send the brief to Mira tonight"


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
