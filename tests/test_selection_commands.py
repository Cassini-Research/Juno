from __future__ import annotations

from dataclasses import replace
import tempfile

import pytest

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.modes import ModeSelection, ModeSource
from juno_v2.contracts.workbench import ClientSelection
from juno_v2.contracts.writer import (
    WriterActionKind,
    WriterIntentKind,
    WriterTransformRequest,
    WriterTransformResult,
)
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.transforms.catalog import BUILTIN_CATALOG
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.parser import WriterIntentParser
from juno_v2.writer.selection_commands import (
    looks_like_selection_edit_command,
    recognize_selection_transform_command,
    selection_command_allows_list_output,
)
from juno_v2.writer.service import WriterService


class _Recorder:
    def __init__(self) -> None:
        self.log_dir = tempfile.mkdtemp(prefix="juno-selection-command-test-")
        self.events: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(self, *args: object, **kwargs: object) -> None:
        self.events.append((args, kwargs))


class _RewriteBackend:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[WriterTransformRequest] = []

    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        self.requests.append(req)
        return WriterTransformResult(
            utterance_id=req.utterance_id,
            text=self.output,
            backend_name="fake-selection-backend",
        )


class _ClassifyingBackend(_RewriteBackend):
    def __init__(self, output: str, classification: dict[str, object]) -> None:
        super().__init__(output)
        self.classification = classification
        self.classifier_calls: list[tuple[str, str]] = []

    def classify_dictation_vs_edit_selection(
        self,
        *,
        spoken: str,
        selection_excerpt: str,
    ) -> dict[str, object]:
        self.classifier_calls.append((spoken, selection_excerpt))
        return self.classification


def _mode_selection() -> ModeSelection:
    return ModeSelection(
        effective_mode="default_surface",
        mode_source=ModeSource.AUTO,
        manual_mode_name=None,
        custom_mode_name=None,
        resolved_from_surface=None,
    )


def _process_selected(
    service: WriterService,
    spoken: str,
    selected: str,
):
    return service.process_transcript(
        utterance_id="utt-selection-command",
        final_text=spoken,
        raw_text=spoken,
        context=TypedContextBundle(
            app_name="Notes",
            app_category="docs",
            selected_text=selected,
        ),
        anchor_selection=ClientSelection(start=0, end=len(selected)),
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=replace(BUILTIN_MODES["default_surface"], final_formatting_policy="minimal"),
        mode_selection=_mode_selection(),
    )


@pytest.mark.parametrize(
    ("transform_id", "spoken"),
    [
        ("polish", "Polish this"),
        ("fix_grammar", "Fix the grammar"),
        ("make_shorter", "Make this shorter"),
        ("make_longer", "Make this longer"),
        ("make_clearer", "Make this clearer"),
        ("make_more_formal", "Make this formal"),
        ("make_more_casual", "Make this friendlier"),
        ("bulletize", "Turn this into bullets"),
        ("numbered_list", "Make this a numbered list"),
        ("summarize", "Summarize this in five points"),
        ("simplify", "Simplify this"),
        ("translate_preserve_meaning", "Translate this to French"),
        ("email_rewrite", "Rewrite this as an email"),
        ("slack_rewrite", "Rewrite this for Slack"),
        ("notes_rewrite", "Turn this into notes"),
        ("checklist_rewrite", "Turn this into a checklist"),
    ],
)
def test_every_builtin_transform_has_a_bounded_selection_command(
    transform_id: str,
    spoken: str,
) -> None:
    assert transform_id in BUILTIN_CATALOG

    command = recognize_selection_transform_command(spoken)

    assert command is not None
    assert command.transform_id == transform_id


@pytest.mark.parametrize(
    "spoken",
    [
        "To make this shorter.",
        "Please make this shorter.",
        "Can you shorten this?",
        "Could you make this shorter?",
        "I want you to make this shorter.",
        "I need you to turn this into bullets.",
        "Let's make this clearer.",
        "Make it concise, please.",
    ],
)
def test_command_frames_and_asr_leading_to_are_recognized(spoken: str) -> None:
    assert recognize_selection_transform_command(spoken) is not None
    assert WriterIntentParser().parse(spoken, selection_present=True).kind != WriterIntentKind.DICTATE


@pytest.mark.parametrize(
    "prose",
    [
        "To make this shorter, I removed two paragraphs.",
        "I want to make this shorter eventually.",
        "Can you believe this is shorter?",
        "Please note that this is shorter.",
        "Make this shorter is what she said.",
        "We should make this shorter.",
        "I asked you to make this shorter yesterday.",
        "To turn this into bullets, select the text first.",
        "This paragraph discusses bullet points.",
    ],
)
def test_complete_match_keeps_command_shaped_prose_as_dictation(prose: str) -> None:
    assert recognize_selection_transform_command(prose) is None
    assert WriterIntentParser().parse(prose, selection_present=True).kind == WriterIntentKind.DICTATE


@pytest.mark.parametrize(
    "spoken",
    [
        "Make this shorter",
        "Turn this into bullets",
        "Summarize this in five points",
        "Rewrite this as an email",
    ],
)
def test_targetless_commands_remain_dictation_for_the_focused_app(spoken: str) -> None:
    assert WriterIntentParser().parse(spoken, selection_present=False).kind == WriterIntentKind.DICTATE


def test_asr_leading_to_shorter_bypasses_turn_planner_and_transforms_selection() -> None:
    backend = _RewriteBackend("A shorter selected paragraph.")
    service = WriterService(
        config=WriterConfig(
            enable_model_transforms=True,
            enable_turn_planner=True,
            dictation_editor_enabled=True,
        ),
        recorder=_Recorder(),
        backend=backend,
    )

    result = _process_selected(
        service,
        "to make this shorter.",
        "This is a much longer selected paragraph with details that can be trimmed.",
    )

    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.output_text == "A shorter selected paragraph."
    assert len(backend.requests) == 1
    assert backend.requests[0].context_payload["task"] == "selection_transform_v1"


def test_unresolved_command_shaped_text_never_overwrites_selection() -> None:
    backend = _RewriteBackend("should never be used")
    service = WriterService(
        config=WriterConfig(
            enable_model_transforms=True,
            enable_turn_planner=False,
            dictation_editor_enabled=False,
        ),
        recorder=_Recorder(),
        backend=backend,
    )

    result = _process_selected(service, "Make this sound like me", "Keep this selected text safe.")

    assert looks_like_selection_edit_command("Make this sound like me")
    assert result.action == WriterActionKind.NOOP
    assert result.output_text == ""
    assert result.metadata["reason"] == "selection_intent_ambiguous"
    assert backend.requests == []


def test_free_form_selection_edit_uses_the_real_backend_classifier() -> None:
    backend = _ClassifyingBackend(
        "Warm, natural selected copy.",
        {
            "intent": "edit",
            "confidence": 0.93,
            "instruction": "Give the selected text a warmer, more natural voice.",
        },
    )
    service = WriterService(
        config=WriterConfig(
            enable_model_transforms=True,
            enable_turn_planner=False,
            dictation_editor_enabled=False,
        ),
        recorder=_Recorder(),
        backend=backend,
    )

    result = _process_selected(service, "Make this sound like me", "Cold selected copy.")

    assert backend.classifier_calls == [("Make this sound like me", "Cold selected copy.")]
    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.output_text == "Warm, natural selected copy."


def test_malformed_classifier_confidence_cannot_overwrite_selection() -> None:
    backend = _ClassifyingBackend(
        "should never be used",
        {"intent": "edit", "confidence": "not-a-number", "instruction": "Rewrite it."},
    )
    service = WriterService(
        config=WriterConfig(
            enable_model_transforms=True,
            enable_turn_planner=False,
            dictation_editor_enabled=False,
        ),
        recorder=_Recorder(),
        backend=backend,
    )

    result = _process_selected(service, "Make this sound like me", "Keep this selected text safe.")

    assert result.action == WriterActionKind.NOOP
    assert result.metadata["reason"] == "selection_intent_ambiguous"
    assert backend.requests == []


@pytest.mark.parametrize(
    "instruction",
    [
        "Summarize this as a single paragraph",
        "Summarize this without bullets",
        "Make this shorter; do not use a list",
    ],
)
def test_explicit_non_list_request_rejects_list_output(instruction: str) -> None:
    assert not selection_command_allows_list_output(instruction, instruction=instruction)


@pytest.mark.parametrize(
    ("model_output", "reason"),
    [
        ("", "selection_transform_empty_output"),
        ("Make this shorter.", "selection_transform_command_echo"),
        ("- First point\n- Second point", "selection_transform_structure_drift"),
    ],
)
def test_model_transform_rejects_destructive_or_wrong_shape_output(
    model_output: str,
    reason: str,
) -> None:
    backend = _RewriteBackend(model_output)
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=False),
        recorder=_Recorder(),
        backend=backend,
    )

    result = _process_selected(
        service,
        "Make this shorter.",
        "This selected paragraph must not disappear or silently become a list.",
    )

    assert result.action == WriterActionKind.NOOP
    assert result.output_text == ""
    assert result.metadata["reason"] == reason


def test_bullet_conversion_is_deterministic_and_preserves_every_selected_item() -> None:
    backend = _RewriteBackend("model must not run")
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True),
        recorder=_Recorder(),
        backend=backend,
    )
    selected = "Protect the opening sentence. Keep the middle sentence. Preserve the final sentence."

    result = _process_selected(service, "To turn this into bullets.", selected)

    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.output_text == (
        "- Protect the opening sentence\n"
        "- Keep the middle sentence\n"
        "- Preserve the final sentence"
    )
    assert all(phrase in result.output_text for phrase in ("Protect the opening", "Keep the middle", "Preserve the final"))
    assert backend.requests == []
