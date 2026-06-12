from __future__ import annotations

import io
import json
import math
import struct
import tempfile
import wave
from datetime import datetime
from typing import Any

from juno_core_v3.actions.llm_extractor import set_llm_extractor
from juno_core_v3.actions.pipeline_hook import detect_actions_for_pipeline
from juno_core_v3.dictation.pipeline import OneShotDictationPipeline
from juno_core_v3.dictation.transcriber import TranscribeResult
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.modes import ModeSelection, ModeSource
from juno_v2.contracts.workbench import ClientSelection, CommitMode
from juno_v2.contracts.writer import (
    WriterActionKind,
    WriterMode,
    WriterTransformRequest,
    WriterTransformResult,
)
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.transcript.adjudicator import TranscriptAdjudicatorConfig
from juno_v2.turn_plan import actions_from_turn_plan, render_turn_plan, validate_turn_plan
from juno_v2.turn_plan.planner import TurnPlanPacket, TurnPlanResult, TurnPlanner
from juno_v2.turn_plan.planner import normalize_turn_plan
from juno_v2.turn_plan.planner import _fallback_structural_turn_plan, _json_object, _result_from_backend
from juno_v2.writer.backends.mlx_lm import (
    _build_writer_prompt,
    _filter_memory_extraction_candidates,
    _system_prompt,
)
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.service import WriterService, _memory_learning_source_text, _spelled_vocab_memory_candidates


class _Recorder:
    def __init__(self) -> None:
        self.log_dir = tempfile.mkdtemp(prefix="juno-turn-plan-test-")
        self.events: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(self, *args: object, **kwargs: object) -> None:
        self.events.append((args, kwargs))


class _TurnPlanBackend:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.requests: list[WriterTransformRequest] = []

    def warm(self) -> None:
        return None

    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        self.requests.append(req)
        task = str((req.context_payload or {}).get("task") or req.metadata.get("kind") or "")
        if task == "transcript_adjudication_v1":
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text=json.dumps(
                    {
                        "schema_version": "transcript_adjudication_v1",
                        "corrected_text": req.source_text,
                        "ops": [],
                        "confidence": 1.0,
                        "protected_terms_used": [],
                    }
                ),
                backend_name="fake-qwen",
            )
        return WriterTransformResult(
            utterance_id=req.utterance_id,
            text=json.dumps(self.plan),
            backend_name="fake-qwen",
        )


class _TaskBackend:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.requests: list[WriterTransformRequest] = []

    def warm(self) -> None:
        return None

    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        self.requests.append(req)
        task = str((req.context_payload or {}).get("task") or req.metadata.get("kind") or "")
        value = self.responses[task]
        text = value if isinstance(value, str) else json.dumps(value)
        return WriterTransformResult(utterance_id=req.utterance_id, text=text, backend_name="fake-qwen")


class _ExtractingTaskBackend(_TaskBackend):
    def __init__(self, responses: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
        super().__init__(responses)
        self.candidates = candidates

    def extract_memory_candidates(self, *, text: str, kind: str, limit: int = 6) -> list[dict[str, Any]]:
        return self.candidates[:limit]


def _selection() -> ModeSelection:
    return ModeSelection(
        effective_mode="default_surface",
        mode_source=ModeSource.AUTO,
        manual_mode_name=None,
        custom_mode_name=None,
        resolved_from_surface=None,
    )


def _service(plan: dict[str, Any]) -> WriterService:
    return WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=_TurnPlanBackend(plan),
    )


def test_turn_planner_renders_only_spoken_numbered_items_when_claim_count_is_larger() -> None:
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "format_dictation",
        "corrected_transcript": {"text": "note down ten points remove patches make qwen plan validate spans"},
        "target": {"kind": "cursor", "confidence": 0.9},
        "render_plan": {
            "render_kind": "numbered_list",
            "markdown_allowed": False,
            "claimed_item_count": 10,
            "spoken_item_count": 3,
            "content_units": [
                {"kind": "item", "text": "remove patches", "source_span": "remove patches", "order": 1},
                {"kind": "item", "text": "make Qwen plan", "source_span": "make qwen plan", "order": 2},
                {"kind": "item", "text": "validate spans", "source_span": "validate spans", "order": 3},
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [{"type": "claimed_count_mismatch", "claimed": 10, "spoken": 3}],
    }

    result = _service(plan).process_transcript(
        utterance_id="utt-list",
        final_text="note down ten points remove patches make qwen plan validate spans",
        raw_text="note down ten points remove patches make qwen plan validate spans",
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "1. remove patches\n2. make Qwen plan\n3. validate spans"
    assert "\n4." not in result.output_text
    assert result.learn_from_commit is False
    assert result.metadata["turn_plan"]["utterance_kind"] == "format_dictation"


def test_structural_fallback_handles_natural_counted_ordinal_list() -> None:
    plan = _fallback_structural_turn_plan(
        "I think we need to focus on 3 things, first is that we check everything properly "
        "before production and second is that we go to a party after we push things live."
    )

    assert plan is not None
    render = plan["render_plan"]
    assert render["claimed_item_count"] == 3
    assert render["spoken_item_count"] == 2
    assert [unit["text"] for unit in render["content_units"]] == [
        "we check everything properly before production",
        "we go to a party after we push things live",
    ]
    assert plan["uncertainties"] == [{"type": "claimed_count_mismatch", "claimed": 3, "spoken": 2}]


def test_turn_plan_json_parser_recovers_misnested_render_metadata() -> None:
    raw = (
        '{"schema_version":"turn_plan_v1","utterance_kind":"dictation",'
        '"corrected_transcript":"Note down 3 points. First remove patches. Second make Qwen plan. Third validate spans.",'
        '"target":{"kind":"cursor","confidence":1.0},'
        '"render_plan":{"render_kind":"numbered_list","markdown_allowed":true,'
        '"content_units":['
        '{"kind":"item","text":"First remove patches","source_span":{"start":20,"end":40},"order":1},'
        '{"kind":"item","text":"Second make Qwen plan","source_span":{"start":42,"end":63},"order":2},'
        '{"kind":"item","text":"Third validate spans","source_span":{"start":65,"end":85},"order":3},'
        '"claimed_item_count":3,"spoken_item_count":3},'
        '"transform":{"operation":"none","instruction":"","transformed_text":null,"requires_second_pass":false},'
        '"actions":[],"snippets":[],"memory_candidates":[],'
        '"safety":{"commit_policy":"commit","execute_policy":"no_execute"},'
        '"uncertainties":[]}'
    )

    parsed = _json_object(raw)

    assert parsed is not None
    render = parsed["render_plan"]
    assert render["claimed_item_count"] == 3
    assert render["spoken_item_count"] == 3
    assert [unit["text"] for unit in render["content_units"]] == [
        "First remove patches",
        "Second make Qwen plan",
        "Third validate spans",
    ]


def test_turn_plan_rejects_repair_request_wrapper_as_plan() -> None:
    result = _result_from_backend(
        WriterTransformResult(
            utterance_id="utt-wrapper",
            text=json.dumps(
                {
                    "task": "turn_repair_v1",
                    "utterance_id": "utt-wrapper",
                    "source_text": "remind me to call mom at 6pm",
                    "invalid": {"raw_output": "{}"},
                }
            ),
            backend_name="fake-qwen",
        )
    )

    assert result.plan is None
    assert result.status == "invalid_plan_object"
    assert result.errors == ["invalid_turn_plan_object"]


def test_turn_plan_validation_allows_normalized_clock_evidence() -> None:
    source = "Juno remind me to call mom at 6pm and write running 10 minutes late."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {
            "text": "remind me to call mom at 6:00 PM and write running 10 minutes late."
        },
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": "running 10 minutes late", "source_span": "running 10 minutes late", "order": 1}
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call mom",
                "evidence_span": "call mom",
                "schedule": {"kind": "instant", "source_span": "6pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    validation = validate_turn_plan(plan, source_text=source, context=TypedContextBundle(app_category="docs"))

    assert validation.ok
    assert validation.errors == []


def test_turn_planner_selection_transform_uses_selected_target() -> None:
    selected = "this launch note is too long and too soft"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "transform",
        "corrected_transcript": {"text": "make this sharper"},
        "target": {"kind": "selection", "confidence": 0.98},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {
            "operation": "rewrite",
            "instruction": "Make the selected text sharper.",
            "transformed_text": "This launch note is direct and ready.",
            "requires_second_pass": False,
        },
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

    result = _service(plan).process_transcript(
        utterance_id="utt-transform-plan",
        final_text="make this sharper",
        raw_text="make this sharper",
        context=TypedContextBundle(app_name="Notes", app_category="docs", selected_text=selected),
        anchor_selection=ClientSelection(start=5, end=5 + len(selected)),
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.commit_mode == CommitMode.REPLACE_SELECTION
    assert result.selection_override == ClientSelection(start=5, end=5 + len(selected))
    assert result.output_text == "This launch note is direct and ready."


def test_turn_plan_actions_support_multiple_native_actions_without_template_phrases() -> None:
    source = "take a note Project Atlas is ready then remind me to call mom at 6 PM and set an alarm for 7 AM"
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
                "body": "Project Atlas is ready",
                "evidence_span": "Project Atlas is ready",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call mom",
                "evidence_span": "call mom",
                "schedule": {"kind": "instant", "source_span": "at 6 PM"},
                "missing_fields": [],
            },
            {
                "kind": "alarm",
                "operation": "create",
                "body": "",
                "evidence_span": "set an alarm for 7 AM",
                "schedule": {"kind": "instant", "source_span": "7 AM"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    validation = validate_turn_plan(plan, source_text=source, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 8, 12, 0))

    assert validation.ok
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["note", "reminder", "alarm"]
    assert parsed.actions[1].body == "call mom"
    assert parsed.actions[1].when is not None
    assert parsed.actions[2].body == "Alarm"


def test_turn_plan_actions_support_launch_video_note_alarm_and_unscheduled_reminder() -> None:
    note_body = (
        "We are recording a sharper video titled What is Juno and why does it exist. "
        "Text is the primary interface between a model and us, and voice is how we communicate. "
        "Juno is private, sends no data to the cloud, and runs Whisper and Qwen locally."
    )
    source = (
        f"take a note {note_body} "
        "set an alarm for 4pm to launch Juno on Product Hunt "
        "and remind me to put an X post looking for a hardware engineer"
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
                "body": note_body,
                "evidence_span": note_body,
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
            {
                "kind": "alarm",
                "operation": "create",
                "body": "launch Juno on Product Hunt",
                "evidence_span": "set an alarm for 4pm to launch Juno on Product Hunt",
                "schedule": {"kind": "instant", "source_span": "4pm"},
                "missing_fields": [],
            },
            {
                "kind": "reminder",
                "operation": "create",
                "body": "put an X post looking for a hardware engineer",
                "evidence_span": "remind me to put an X post looking for a hardware engineer",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="docs"))

    assert validation.ok
    assert "action_command_render_collapsed" not in notes
    assert rendered.text == ""
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["note", "alarm", "reminder"]
    assert parsed.actions[0].body == note_body
    assert parsed.actions[1].body == "launch Juno on Product Hunt"
    assert parsed.actions[1].when is not None
    assert parsed.actions[2].when is None
    assert parsed.actions[2].body == "put an X post looking for a hardware engineer"


def test_turn_plan_action_converter_infers_minimal_grounded_alarm_time() -> None:
    source = "Juno set an alarm for 7am"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "alarm",
                "operation": "create",
                "body": "",
                "evidence_span": source,
                "schedule": {"kind": "instant", "source_span": "for 7am"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert parsed.actions is not None
    assert parsed.rejected_reason is None
    assert parsed.actions[0].kind.value == "alarm"
    assert parsed.actions[0].body == "Alarm"
    assert parsed.actions[0].when is not None
    assert parsed.actions[0].when.iso.startswith("2026-06-10T07:00:00")


def test_turn_plan_action_converter_replaces_schedule_only_alarm_body() -> None:
    source = "set an alarm for 4pm"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "alarm",
                "operation": "create",
                "body": "4pm",
                "evidence_span": source,
                "schedule": {"kind": "instant", "source_span": "4pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert parsed.actions is not None
    assert parsed.actions[0].body == "Alarm"


def test_turn_plan_normalizer_repairs_schedule_only_alarm_body_from_evidence() -> None:
    source = "set an alarm for 4pm to launch Juno on Product Hunt"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "alarm",
                "operation": "create",
                "body": "4pm",
                "evidence_span": source,
                "schedule": {"kind": "instant", "source_span": "4pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert "action_body_repaired_from_evidence" in notes
    assert parsed.actions is not None
    assert parsed.actions[0].body == "launch Juno on Product Hunt"


def test_turn_plan_normalizer_recovers_from_bad_numeric_schedule_spans() -> None:
    source = (
        "set an alarm for 5pm to launch Juno on Product Hunt and "
        "remind me tomorrow at 10am to post the hardware engineer request"
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
                "kind": "alarm",
                "operation": "create",
                "body": "5pm",
                "evidence_span": [0, 10],
                "schedule": {"kind": "instant", "source_span": [0, 10]},
                "missing_fields": ["time"],
            },
            {
                "kind": "reminder",
                "operation": "create",
                "body": "post the hardware engineer request",
                "evidence_span": "post the hardware engineer request",
                "schedule": {
                    "kind": "instant",
                    "source_span": "on Product Hunt and re",
                    "time": "tomorrow at 10am",
                },
                "missing_fields": ["time"],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_schedule_span_replaced_with_temporal_candidate" in notes
    assert "action_schedule_span_replaced_with_time" in notes
    assert parsed.actions is not None
    assert parsed.missing_fields == []
    assert [(action.kind.value, action.body) for action in parsed.actions] == [
        ("alarm", "launch Juno on Product Hunt"),
        ("reminder", "post the hardware engineer request"),
    ]
    assert parsed.actions[0].when is not None
    assert parsed.actions[0].when.iso.startswith("2026-06-09T17:00:00")
    assert parsed.actions[1].when is not None
    assert parsed.actions[1].when.iso.startswith("2026-06-10T10:00:00")


def test_turn_plan_normalizer_recovers_alarm_body_from_later_action_clause() -> None:
    source = (
        "take a note Juno is private and local. "
        "Set an alarm for 4pm to launch Juno on Product Hunt "
        "and remind me to put an X post looking for a hardware engineer"
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
                "body": "Juno is private and local.",
                "evidence_span": "Juno is private and local.",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
            {
                "kind": "alarm",
                "operation": "create",
                "body": "4pm",
                "evidence_span": "4pm",
                "schedule": {"kind": "instant", "source_span": "4pm"},
                "missing_fields": ["time"],
            },
            {
                "kind": "reminder",
                "operation": "create",
                "body": "put an X post looking for a hardware engineer",
                "evidence_span": "put an X post looking for a hardware engineer",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert "action_body_repaired_from_evidence" in notes
    assert parsed.actions is not None
    assert parsed.actions[1].body == "launch Juno on Product Hunt"


def test_turn_plan_confirm_execute_policy_marks_actions_for_confirmation() -> None:
    source = "remind me to call mom at 6 PM"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call mom",
                "evidence_span": "call mom",
                "schedule": {"kind": "instant", "source_span": "at 6 PM"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "confirm"},
        "uncertainties": [{"type": "confirm_before_execute"}],
    }

    validation = validate_turn_plan(plan, source_text=source, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 8, 12, 0))

    assert validation.ok
    assert parsed.actions is not None
    assert parsed.actions[0].needs_confirmation is True


def test_turn_plan_no_execute_policy_rejects_actions() -> None:
    source = "take a note Project Atlas is ready"
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
                "body": "Project Atlas is ready",
                "evidence_span": "Project Atlas is ready",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

    validation = validate_turn_plan(plan, source_text=source, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 8, 12, 0))

    assert not validation.ok
    assert "actions_with_no_execute_policy" in validation.errors
    assert parsed.actions is None
    assert parsed.rejected_reason == "execute_policy_no_execute"


def test_turn_plan_rejects_incomplete_reminder_without_executing_or_pasting_body() -> None:
    source = "remind me at 3"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "",
                "evidence_span": source,
                "schedule": {"kind": "instant", "source_span": "at 3"},
                "missing_fields": ["body"],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "confirm"},
        "uncertainties": [{"field": "body"}],
    }

    validation = validate_turn_plan(plan, source_text=source, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 8, 12, 0))

    # Per-action problems are validation warnings (plan survives), but the
    # incomplete reminder must still neither execute nor paste: coercion
    # rejects it.
    assert validation.ok
    assert "action_0_missing_body" in validation.warnings
    assert parsed.actions is None
    assert parsed.rejected_reason == "action_0_missing_body"


def test_turn_plan_rejects_ungrounded_rendered_content() -> None:
    source = "ship product today"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "format_dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "bulleted_list",
            "markdown_allowed": False,
            "content_units": [{"kind": "item", "text": "finally call mom", "source_span": "", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

    validation = validate_turn_plan(plan, source_text=source, context=TypedContextBundle(app_category="docs"))

    assert not validation.ok
    assert "content_unit_0_ungrounded" in validation.errors


def test_turn_plan_normalizer_repairs_schema_shape_and_drops_stray_dictation_actions() -> None:
    source = "Note down 3 points, first remove patches, second make Qwen plan, third validate spans"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": source,
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "numbered_list",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "item", "text": "first remove patches", "source_span": {"start": 20, "end": 40}, "order": 1},
                {"kind": "item", "text": "second make Qwen plan", "source_span": {"start": 42, "end": 63}, "order": 2},
                {"kind": "item", "text": "third validate spans", "source_span": {"start": 65, "end": 85}, "order": 3},
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "first remove patches second make Qwen plan third validate spans",
                "evidence_span": {"start": 0, "end": 5},
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="docs"))

    assert validation.ok
    assert "corrected_transcript_string_to_object" in notes
    assert "dictation_actions_dropped" in notes
    assert "execute_policy_cleared_without_actions" in notes
    assert normalized["utterance_kind"] == "format_dictation"
    assert normalized["actions"] == []
    assert normalized["safety"]["execute_policy"] == "no_execute"
    assert rendered.text == "1. first remove patches\n2. second make Qwen plan\n3. third validate spans"


def test_turn_plan_normalizer_does_not_trim_plain_dictation_with_stray_action() -> None:
    source = "Juno should write the product name in this sentence."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": source, "source_span": {"start": 0, "end": len(source)}, "order": 1},
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "the product name in this sentence",
                "evidence_span": {"start": 18, "end": len(source) - 1},
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="docs"))

    assert validation.ok
    assert "mixed_render_trimmed_to_write_clause" not in notes
    assert "dictation_actions_dropped" in notes
    assert "execute_policy_cleared_without_actions" in notes
    assert normalized["utterance_kind"] == "dictation"
    assert normalized["actions"] == []
    assert normalized["safety"]["execute_policy"] == "no_execute"
    assert rendered.text == source


def test_turn_plan_normalizer_maps_command_actions_into_action_contract() -> None:
    source = "remind me at 3pm to call Ishida"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "command",
        "corrected_transcript": source,
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Ishida",
                "evidence_span": {"start": 16, "end": 27},
                "schedule": {"kind": "instant", "source_span": "3pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="messaging"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "command_kind_mapped" in notes
    assert normalized["utterance_kind"] == "actions"
    assert parsed.actions is not None
    assert parsed.actions[0].kind.value == "reminder"
    assert parsed.actions[0].body == "call Ishida"


def test_turn_plan_normalizer_collapses_duplicate_native_note_render() -> None:
    source = "take a note Project Atlas is ready"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "command",
        "corrected_transcript": source,
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "note",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": "Project Atlas is ready", "source_span": {"start": 12, "end": 34}, "order": 1},
                {"kind": "paragraph", "text": "Take a note. Project Atlas is ready.", "source_span": source, "order": 2},
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "Project Atlas is ready",
                "evidence_span": {"start": 12, "end": 34},
                "schedule": {"kind": "none", "source_span": {"start": 0, "end": 0}},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_duplicate_render_collapsed" in notes
    assert normalized["utterance_kind"] == "actions"
    assert rendered.text == ""
    assert parsed.actions is not None
    assert parsed.actions[0].kind.value == "note"
    assert parsed.actions[0].body == "Project Atlas is ready"


def test_turn_plan_normalizer_recovers_native_note_action_mislabeled_as_dictation() -> None:
    source = "Take A Note Project Atlas is ready."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "note",
            "markdown_allowed": True,
            "content_units": [
                {"kind": "paragraph", "text": "Project Atlas is ready", "source_span": {"start": 12, "end": 34}, "order": 1},
                {"kind": "paragraph", "text": "Take A Note Project Atlas is ready", "source_span": {"start": 0, "end": len(source) - 1}, "order": 2},
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "Project Atlas is ready",
                "evidence_span": {"start": 12, "end": 34},
                "schedule": {"kind": "none", "source_span": {"start": 0, "end": 0}},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_duplicate_render_collapsed" in notes
    assert "dictation_native_note_mapped_to_actions" in notes
    assert "dictation_actions_dropped" not in notes
    assert normalized["utterance_kind"] == "actions"
    assert rendered.text == ""
    assert parsed.actions is not None
    assert parsed.actions[0].kind.value == "note"
    assert parsed.actions[0].body == "Project Atlas is ready"


def test_turn_plan_normalizer_expands_native_note_body_from_grounded_evidence() -> None:
    source = "Juno take a note Project Atlas is ready"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "note",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "Project Atlas is ready", "source_span": "Project Atlas is ready", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "Project",
                "evidence_span": "Project Atlas is ready",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_body_repaired_from_evidence" in notes
    assert parsed.actions is not None
    assert parsed.actions[0].kind.value == "note"
    assert parsed.actions[0].body == "Project Atlas is ready"


def test_turn_plan_normalizer_does_not_strip_note_body_from_schedule_shape() -> None:
    source = "Juno take a note Project Atlas is ready"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "note",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "Project Atlas is ready", "source_span": "Project Atlas is ready", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "Project Atlas is ready",
                "evidence_span": "Project Atlas is ready",
                "schedule": {"kind": "none", "source_span": "Atlas is ready"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert "action_schedule_removed_from_body" not in notes
    assert parsed.actions is not None
    assert parsed.actions[0].body == "Project Atlas is ready"


def test_turn_plan_normalizer_recovers_native_reminder_action_mislabeled_as_dictation() -> None:
    source = "remind me to go to Disco with Ishida at 9pm"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": "remind me to go to Disco with Ishida at 9pm.",
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": "remind me to go to Disco with Ishida at 9pm", "source_span": {"start": 0, "end": len(source)}, "order": 1}
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "go to Disco with Ashida at 9 PM",
                "evidence_span": {"start": 0, "end": len(source)},
                "schedule": {"kind": "instant", "source_span": {"start": len(source) - 4, "end": len(source)}},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_body_repaired_from_evidence" in notes
    assert "dictation_native_action_mapped_to_actions" in notes
    assert "dictation_actions_dropped" not in notes
    assert normalized["utterance_kind"] == "actions"
    assert rendered.text == ""
    assert parsed.actions is not None
    assert parsed.actions[0].kind.value == "reminder"
    assert parsed.actions[0].body == "go to Disco with Ishida"


def test_turn_plan_normalizer_collapses_action_only_command_render() -> None:
    source = "remind me to go to Disco with Ishida at 9pm"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "command",
        "corrected_transcript": "remind me to go to Disco with Ishida at 9pm.",
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": "Remind me to go to Disco with Ishida at 9pm.", "source_span": source, "order": 1}
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "go to Disco with Ishida at 9pm",
                "evidence_span": "go to Disco with Ishida",
                "schedule": {"kind": "instant", "source_span": "9pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_duplicate_render_collapsed" in notes or "action_command_render_collapsed" in notes
    assert normalized["utterance_kind"] == "actions"
    assert rendered.text == ""
    assert parsed.actions is not None
    assert parsed.actions[0].kind.value == "reminder"
    assert parsed.actions[0].body == "go to Disco with Ishida"


def test_turn_plan_normalizer_collapses_source_command_when_render_still_has_action_text() -> None:
    source = "Remind me to go to disco with Ishida at 9pm."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": source, "source_span": "Remind me to go to disco with Ishida at 9pm", "order": 1}
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "go to disco with Ishida",
                "evidence_span": "go to disco with Ishida",
                "schedule": {"kind": "instant", "source_span": "9pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="docs"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="docs"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_duplicate_render_collapsed" in notes or "action_command_render_collapsed" in notes
    assert normalized["utterance_kind"] == "actions"
    assert rendered.text == ""
    assert parsed.actions is not None
    assert parsed.actions[0].body == "go to disco with Ishida"


def test_actions_strip_native_invocation_prefix_from_grounded_note_body() -> None:
    source = "Take A Note Project Atlas is ready."
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
                "body": "Take A Note: Project Atlas is ready",
                "evidence_span": "Take A Note Project Atlas is ready",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert parsed.actions is not None
    assert parsed.actions[0].body == "Project Atlas is ready"


def test_turn_plan_normalizer_keeps_write_clause_as_cursor_text_not_note_action() -> None:
    source = "remind me to call mom at 6pm and write running 10 minutes late"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {
            "render_kind": "paragraph",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": "Remind me to call mom at 6pm and write running 10 minutes late", "source_span": {"start": 0, "end": len(source)}, "order": 1},
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call mom at 6pm",
                "evidence_span": {"start": 13, "end": 27},
                "schedule": {"kind": "instant", "source_span": {"start": 22, "end": 26}},
                "missing_fields": [],
            },
            {
                "kind": "note",
                "operation": "create",
                "body": "running 10 minutes late",
                "evidence_span": {"start": 37, "end": len(source)},
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="messaging"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="messaging"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "render_kind_alias_normalized" in notes
    assert "mixed_render_trimmed_to_write_clause" in notes
    assert "render_only_note_actions_dropped" in notes
    assert "action_schedule_removed_from_body" in notes
    assert normalized["utterance_kind"] == "mixed"
    assert rendered.text == "running 10 minutes late"
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["reminder"]
    assert parsed.actions[0].body == "call mom"


def test_actions_strip_inferred_schedule_before_write_clause_from_body() -> None:
    source = "remind me to call Bob at 6pm and write running 10 minutes late"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "running 10 minutes late", "source_span": "running 10 minutes late", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Bob at 6pm and w",
                "evidence_span": "remind me to call Bob at 6pm and w",
                "schedule": {"kind": "instant"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, _notes = normalize_turn_plan(plan, source_text=source)
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert parsed.actions is not None
    assert parsed.actions[0].kind.value == "reminder"
    assert parsed.actions[0].body == "call Bob"
    assert parsed.actions[0].when is not None


def test_turn_plan_normalizer_lifts_write_clause_note_action_into_render_when_render_missing() -> None:
    source = "remind me to call mom at 6pm and write running 10 minutes late"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call mom at 6pm",
                "evidence_span": "remind me to call mom at 6pm",
                "schedule": {"kind": "instant", "source_span": "6pm"},
                "missing_fields": [],
            },
            {
                "kind": "note",
                "operation": "create",
                "body": "running 10 minutes late",
                "evidence_span": "running 10 minutes late",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="messaging"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="messaging"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "write_clause_note_lifted_to_render" in notes
    assert normalized["utterance_kind"] == "mixed"
    assert rendered.text == "running 10 minutes late"
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["reminder"]


def test_turn_plan_normalizer_repairs_ungrounded_write_clause_render_from_source() -> None:
    source = "remind me to call Bob at 6pm and write running 10 minutes late"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {
            "render_kind": "message",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "running 0 minutes late", "source_span": "running 0 minutes late", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Bob at 6pm",
                "evidence_span": "remind me to call Bob at 6pm",
                "schedule": {"kind": "instant", "source_span": "6pm"},
                "missing_fields": [],
            },
            {
                "kind": "note",
                "operation": "create",
                "body": "running 10 minutes late",
                "evidence_span": "running 10 minutes late",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="messaging"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="messaging"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "write_clause_render_repaired_from_source" in notes
    assert "render_only_note_actions_dropped" in notes
    assert rendered.text == "running 10 minutes late"
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["reminder"]


def test_turn_plan_normalizer_repairs_under_scoped_reminder_body_from_source() -> None:
    source = "remind me to call Bob at 6pm and write running 10 minutes late"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": "running 10 minutes late", "source_span": "running 10 minutes late", "order": 1}
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call",
                "evidence_span": "remind me to call Bob at 6pm and w",
                "schedule": {"kind": "instant", "source_span": "6pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="messaging"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="messaging"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_body_repaired_from_evidence" in notes
    assert rendered.text == "running 10 minutes late"
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["reminder"]
    assert parsed.actions[0].body == "call Bob"


def test_turn_plan_normalizer_repairs_over_scoped_reminder_body_from_schedule_write_boundary() -> None:
    source = "Juno remind me to call Bob at 6pm and write running 10 minutes late."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": "running 10 minutes late", "source_span": "running 10 minutes late", "order": 1}
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Bob at 6pm and w",
                "evidence_span": "remind me to call Bob at 6pm and w",
                "schedule": {"kind": "instant", "source_span": "6pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="messaging"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="messaging"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "action_body_repaired_from_evidence" in notes
    assert rendered.text == "running 10 minutes late"
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["reminder"]
    assert parsed.actions[0].body == "call Bob"


def test_turn_plan_normalizer_drops_false_note_action_for_write_tail_even_when_note_body_is_ungrounded() -> None:
    source = "remind me to call Bob at 6pm and write running 10 minutes late"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "message",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "running 0 minutes late", "source_span": "running 0 minutes late", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Bob at 6pm a",
                "evidence_span": "remind me to call Bob at 6pm",
                "schedule": {"kind": "instant", "source_span": "6pm"},
                "missing_fields": [],
            },
            {
                "kind": "note",
                "operation": "create",
                "body": "running 0 minutes late",
                "evidence_span": "6pm and write",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=source)
    validation = validate_turn_plan(normalized, source_text=source, context=TypedContextBundle(app_category="messaging"))
    rendered = render_turn_plan(normalized, context=TypedContextBundle(app_category="messaging"))
    parsed = actions_from_turn_plan(normalized, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert validation.ok
    assert "write_clause_render_repaired_from_source" in notes
    assert "render_only_note_actions_dropped" in notes
    assert rendered.text == "running 10 minutes late"
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["reminder"]
    assert parsed.actions[0].body == "call Bob"


def test_actions_strip_schedule_fragment_from_reminder_body_payload() -> None:
    source = "Juno remind me to call Bob at 6pm and write running 10 minutes late."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [
                {"kind": "paragraph", "text": "running 10 minutes late", "source_span": "running 10 minutes late", "order": 1}
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Bob at 6pm and w",
                "evidence_span": "remind me to call Bob at 6pm",
                "schedule": {"kind": "instant", "source_span": "6pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 9, 12, 0))

    assert parsed.actions is not None
    assert parsed.actions[0].body == "call Bob"


def test_text_turn_commit_policy_no_commit_falls_back_to_text_rules() -> None:
    source = "write hello world"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "format_dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "message",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "hello world", "source_span": "hello world", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

    result = _service(plan).process_transcript(
        utterance_id="utt-no-commit",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Messages", app_category="messaging"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result.action != WriterActionKind.NOOP
    assert result.output_text.strip()


def test_turn_plan_action_kind_without_wake_or_actions_falls_back_to_text() -> None:
    source = "write this as an email to Lena saying the install flow is fixed"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 0.6},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

    result = _service(plan).process_transcript(
        utterance_id="utt-action-kind-fallback",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Mail", app_category="email"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
        wake_verified=False,
    )

    assert result.action != WriterActionKind.NOOP
    assert result.output_text.strip()


def test_wake_verified_mixed_native_actions_reaching_writer_still_deliver_text() -> None:
    # Contract change 2026-06-10: action dispatch is pipeline-owned. The
    # writer is only reached when the pipeline dispatched NO actions, so a
    # paste-suppressing NOOP here meant "no paste AND no action" — the
    # production data-loss hole (failure_reason=turn_plan_action_only with an
    # empty transcript). A mixed/actions plan reaching the writer must
    # deliver the spoken text instead of silently dropping it.
    source = "take a note launch checklist and remind me tomorrow at 9am to send it"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 0.8},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "launch checklist", "source_span": "launch checklist", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "launch checklist",
                "evidence_span": "take a note launch checklist",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    result = _service(plan).process_transcript(
        utterance_id="utt-mixed-action-no-paste",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Slack", app_category="messaging"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
        wake_verified=True,
    )

    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text.strip()
    assert "launch checklist" in result.output_text


def test_turn_plan_snippet_insert_resolves_exact_trigger_from_memory() -> None:
    with tempfile.TemporaryDirectory(prefix="juno-turn-snippet-") as tmp:
        memory = JsonMemoryStore(tmp)
        memory.add_snippet(
            trigger="customer follow up snippet",
            body="Customer Follow-Up\nContext:\nPain:\nNext step:\nOwner:\nDeadline:",
            scope="global",
        )
        plan = {
            "schema_version": "turn_plan_v1",
            "utterance_kind": "dictation",
            "corrected_transcript": {"text": "insert customer follow up snippet"},
            "target": {"kind": "cursor", "confidence": 1.0},
            "render_plan": {"render_kind": "plain", "markdown_allowed": False, "content_units": []},
            "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
            "actions": [],
            "snippets": [{"operation": "insert", "trigger": "customer follow up snippet"}],
            "memory_candidates": [],
            "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
            "uncertainties": [],
        }

        rendered = render_turn_plan(plan, context=TypedContextBundle(app_category="docs"), memory_store=memory)

    assert rendered.rendered
    assert rendered.reason == "snippet_insert"
    assert rendered.text == "Customer Follow-Up\nContext:\nPain:\nNext step:\nOwner:\nDeadline:"


def test_direct_snippet_insert_resolves_exact_trigger_without_saying_snippet() -> None:
    with tempfile.TemporaryDirectory(prefix="juno-direct-snippet-") as tmp:
        memory = JsonMemoryStore(tmp)
        memory.add_snippet(
            trigger="launch footer aurora temp",
            body="Best,\nJas\nFounder, Juno",
            scope="global",
        )

        result = _service({
            "schema_version": "turn_plan_v1",
            "utterance_kind": "dictation",
            "corrected_transcript": {"text": "add launch footer aurora temp"},
            "target": {"kind": "cursor", "confidence": 1.0},
            "render_plan": {"render_kind": "plain", "markdown_allowed": False, "content_units": []},
            "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
            "actions": [],
            "snippets": [],
            "memory_candidates": [],
            "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
            "uncertainties": [],
        }).process_transcript(
            utterance_id="utt-direct-snippet-no-word",
            final_text="add launch footer aurora temp",
            raw_text="add launch footer aurora temp",
            context=TypedContextBundle(app_name="Notes", app_category="docs"),
            anchor_selection=None,
            memory_store=memory,
            memory_snapshot=memory.snapshot(),
            memory_packet=memory.serving_packet().to_dict(),
            mode_policy=BUILTIN_MODES["default_surface"],
            mode_selection=_selection(),
        )

    assert result.output_text == "Best,\nJas\nFounder, Juno"
    assert result.metadata["snippet_expanded"] is True


def test_memory_serving_packet_carries_ranked_snippets() -> None:
    with tempfile.TemporaryDirectory(prefix="juno-memory-snippets-") as tmp:
        memory = JsonMemoryStore(tmp)
        memory.add_snippet(trigger="launch footer", body="Best,\nJas", scope="global")
        packet = memory.serving_packet()

    assert packet.snippets == [
        {
            "trigger": "launch footer",
            "scope": "global",
            "body_preview": "Best,\nJas",
            "body_chars": 9,
            "case_sensitive": False,
        }
    ]


def test_actions_apply_self_correction_before_body_and_schedule_parse() -> None:
    source = "set an alarm for 3pm no no scratch that 4:15pm to launch Juno"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "alarm",
                "operation": "create",
                "body": "3pm no no scratch that 4:15pm to launch Juno",
                "evidence_span": source,
                "schedule": {"kind": "instant", "source_span": "3pm no no scratch that 4:15pm"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 10, 12, 0))

    assert parsed.actions is not None
    assert parsed.actions[0].when is not None
    assert parsed.actions[0].when.iso.startswith("2026-06-10T16:15")
    assert parsed.actions[0].body == "launch Juno"


def test_actions_do_not_borrow_schedule_from_another_action_clause() -> None:
    source = "remind me at 3 to call Tara and set an alarm in 25 minutes"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Tara",
                "evidence_span": "remind me at 3 to call Tara",
                "schedule": {"kind": "instant", "source_span": "at 3"},
                "missing_fields": [],
            },
            {
                "kind": "alarm",
                "operation": "create",
                "body": "Alarm",
                "evidence_span": "set an alarm in 25 minutes",
                "schedule": {"kind": "instant", "source_span": "in 25 minutes"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 10, 12, 0))

    # The ambiguous "at 3" reminder must not borrow the alarm clause's
    # "in 25 minutes" — it is skipped. The well-formed alarm still ships.
    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["alarm"]
    assert parsed.actions[0].when is not None
    assert parsed.skipped_reasons == ["action_0_time_parse_failed"]
    assert "schedule" in parsed.missing_fields


def test_one_invalid_action_does_not_reject_valid_siblings() -> None:
    # Production 2026-06-11: a six-action utterance ("set up 3 alarms…,
    # 4th call Parth at 11 p.m. today, add a note…, remind me…") was fully
    # rejected because the planner emitted the 4th alarm without a usable
    # schedule span. Valid siblings must survive.
    source = (
        "Hey Juno, Set up 3 alarms. First, Call Aarti at 5 PM today. "
        "Second, Call Atharva at 7 PM today. Third, Call Darpan at 9 PM today. "
        "4th call Parth at 11 p.m. today. Add a note that we need to release Juno today. "
        "Remind me to call my mom tomorrow 5 p.m."
    )

    def alarm(body: str, span: str, evidence: str) -> dict:
        return {
            "kind": "alarm",
            "operation": "create",
            "body": body,
            "evidence_span": evidence,
            "schedule": {"kind": "instant", "source_span": span},
            "missing_fields": [],
        }

    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            alarm("Call Aarti", "5 PM today", "Call Aarti at 5 PM today"),
            alarm("Call Atharva", "7 PM today", "Call Atharva at 7 PM today"),
            alarm("Call Darpan", "9 PM today", "Call Darpan at 9 PM today"),
            # Planner dropped the time for the irregular "4th call …" clause.
            alarm("Call Parth", "", "4th call Parth"),
            {
                "kind": "note",
                "operation": "create",
                "body": "we need to release Juno today",
                "evidence_span": "Add a note that we need to release Juno today",
                "missing_fields": [],
            },
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call my mom",
                "evidence_span": "Remind me to call my mom tomorrow 5 p.m.",
                "schedule": {"kind": "instant", "source_span": "tomorrow 5 p.m."},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 11, 12, 6))

    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["alarm", "alarm", "alarm", "note", "reminder"]
    assert parsed.rejected_reason is None
    assert parsed.skipped_reasons == ["action_3_time_parse_failed"]
    assert "schedule" in parsed.missing_fields


def test_all_invalid_actions_still_reject_the_batch() -> None:
    source = "remind me at 3 to call Tara"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Tara",
                "evidence_span": "remind me at 3 to call Tara",
                "schedule": {"kind": "instant", "source_span": "at 3"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 11, 12, 6))

    assert parsed.actions is None
    assert parsed.rejected_reason == "action_0_time_parse_failed"
    assert parsed.skipped_reasons == ["action_0_time_parse_failed"]


def test_meridiem_spelling_grounds_across_asr_and_planner_forms() -> None:
    from juno_v2.turn_plan.validators import span_present

    source = "4th call Parth at 11 p.m. today."
    # Planner re-types spans in normalized form; both spellings must ground.
    assert span_present("11 p.m. today", source)
    assert span_present("11 PM today", source)
    assert span_present("11 P.M. today", source)
    # And the reverse: dotted span against a normalized source.
    assert span_present("11 p.m. today", "call Parth at 11 PM today")
    # Ordinary "a"/"m" word sequences must not be collapsed.
    assert not span_present("am", "got a message")

    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "alarm",
                "operation": "create",
                "body": "Call Parth",
                # Normalized re-typing of the dotted source time.
                "evidence_span": "4th call Parth at 11 PM today",
                "schedule": {"kind": "instant", "source_span": "11 PM today"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 11, 12, 6))

    assert parsed.actions is not None
    assert parsed.skipped_reasons == []
    assert parsed.actions[0].when is not None
    assert parsed.actions[0].when.iso.startswith("2026-06-11T23:00")


def test_turn_plan_validation_rejects_common_memory_candidates_in_secure_fields() -> None:
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": "finally ship this"},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "finally ship this", "source_span": "finally ship this", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [{"surface": "finally", "canonical": "finally"}],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

    validation = validate_turn_plan(
        plan,
        source_text="finally ship this",
        context=TypedContextBundle(metadata={"focused_is_secure": True}),
    )

    assert not validation.ok
    assert "memory_candidates_in_secure_field" in validation.errors
    assert "memory_candidate_0_common_word" in validation.warnings


def test_memory_extraction_splits_composite_vocab_and_preserves_acronym_phrases() -> None:
    source = "Teach Juno Lumare Ishida MLX LM Qwen should be remembered as product terms"
    candidates = [
        {"term": "Juno Lumare Ishida", "note": "adjacent names"},
        {"term": "MLX", "note": "acronym phrase"},
        {"term": "Qwen", "note": "model name"},
    ]

    filtered = _filter_memory_extraction_candidates(
        kind="vocab",
        source_text=source,
        candidates=candidates,
        limit=8,
    )

    terms = [item["term"] for item in filtered]
    assert "Lumare" in terms
    assert "Ishida" in terms
    assert "MLX LM" in terms
    assert "Qwen" in terms
    assert "Juno Lumare Ishida" not in terms


def test_memory_extraction_expands_grounded_descriptive_terms_in_teach_context() -> None:
    source = "Teach Juno Lumare Ishida MLX LM Qwen adjudicator Cassini actions layer as product terms"
    candidates = [
        {"term": "Juno", "note": "product"},
        {"term": "Lumare", "note": "name"},
        {"term": "Ishida", "note": "name"},
        {"term": "MLX", "note": "acronym"},
        {"term": "Qwen", "note": "model"},
        {"term": "Cassini", "note": "project"},
    ]

    filtered = _filter_memory_extraction_candidates(
        kind="vocab",
        source_text=source,
        candidates=candidates,
        limit=10,
    )

    terms = [item["term"] for item in filtered]
    assert "Lumare" in terms
    assert "Ishida" in terms
    assert "MLX LM" in terms
    assert "Qwen adjudicator" in terms
    assert "Cassini actions layer" in terms
    assert "Qwen adjudicator Cassini" not in terms


def test_memory_extraction_trims_spoken_learning_relation_tail() -> None:
    source = "Teach Juno that MLX LM may sound like em-lex and Lumare is pronounced Loo-mah-ree"
    candidates = [
        {"term": "MLX LM may sound", "note": "em-lex"},
        {"term": "Lumare is pronounced", "note": "Loo-mah-ree"},
    ]

    filtered = _filter_memory_extraction_candidates(
        kind="vocab",
        source_text=source,
        candidates=candidates,
        limit=8,
    )

    terms = [item["term"] for item in filtered]
    assert "MLX LM" in terms
    assert "Lumare" in terms
    assert "MLX LM may sound" not in terms
    assert "Lumare is pronounced" not in terms


def test_memory_extraction_promotes_grounded_descriptor_from_teach_relation_note() -> None:
    source = (
        "Teach Juno that Qwen should mean Qwen adjudicator "
        "and Cassini should mean Cassini actions layer."
    )
    filtered = _filter_memory_extraction_candidates(
        kind="vocab",
        source_text=source,
        candidates=[
            {"term": "Qwen", "note": "Qwen adjudicator, not Qwen"},
            {"term": "Cassini", "note": "Cassini actions layer, not just Cassini"},
        ],
        limit=8,
    )

    terms = [item["term"] for item in filtered]
    assert "Qwen adjudicator" in terms
    assert "Cassini actions layer" in terms
    assert "Qwen" not in terms
    assert "Cassini" not in terms


def test_memory_extraction_stops_descriptor_at_conjunction_boundary() -> None:
    source = "Teach Juno that Qwen adjudicator and Cassini actions layer are product terms."
    filtered = _filter_memory_extraction_candidates(
        kind="vocab",
        source_text=source,
        candidates=[
            {"term": "Qwen", "note": "Qwen adjudicator and Cassini actions layer"},
            {"term": "Cassini", "note": "Cassini actions layer"},
        ],
        limit=8,
    )

    terms = [item["term"] for item in filtered]
    assert "Qwen adjudicator" in terms
    assert "Cassini actions layer" in terms
    assert "Qwen adjudicator and" not in terms


def test_turn_plan_memory_mutation_applies_grounded_candidates_without_paste() -> None:
    source = "Teach Juno these terms: Lumare, Qwen adjudicator, and finally."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "memory_mutation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [
            {"term": "Lumare", "note": "name"},
            {"term": "Qwen adjudicator", "note": "product term"},
            {"term": "finally", "note": "common word"},
            {"term": "Teach Juno", "note": "command"},
        ],
        "safety": {"commit_policy": "no_commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }
    memory = JsonMemoryStore(tempfile.mkdtemp(prefix="juno-memory-turn-plan-"))

    result = _service(plan).process_transcript(
        utterance_id="utt-memory-mutation",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=memory,
        memory_snapshot=memory.snapshot(),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    learned = {str(item.get("canonical_form") or item.get("term")) for item in memory.vocabulary.raw()}
    assert result.action == WriterActionKind.MEMORY_MUTATION
    assert result.output_text == ""
    assert result.memory_updated is True
    assert "Lumare" in learned
    assert "Qwen adjudicator" in learned
    assert "finally" not in learned
    assert "Teach Juno" not in learned


def test_explicit_memory_learning_uses_extractor_when_turn_plan_json_fails() -> None:
    source = "Teach Juno these terms: Lumare and Qwen adjudicator."
    memory = JsonMemoryStore(tempfile.mkdtemp(prefix="juno-memory-turn-plan-"))
    backend = _ExtractingTaskBackend(
        {"turn_planning_v1": "not json", "turn_repair_v1": "still not json"},
        [
            {"term": "Lumare", "note": "name"},
            {"term": "Qwen adjudicator", "note": "product term"},
        ],
    )

    result = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    ).process_transcript(
        utterance_id="utt-memory-extractor-fallback",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=memory,
        memory_snapshot=memory.snapshot(),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    learned = {str(item.get("canonical_form") or item.get("term")) for item in memory.vocabulary.raw()}
    assert result.action == WriterActionKind.MEMORY_MUTATION
    assert result.output_text == ""
    assert result.memory_updated is True
    assert "Lumare" in learned
    assert "Qwen adjudicator" in learned
    assert result.metadata["memory_action"] == "extractor_memory_candidates"


def test_explicit_memory_learning_uses_extractor_when_turn_plan_has_no_candidates() -> None:
    source = "Teach Juno these terms: Lumare and Qwen adjudicator."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": source, "source_span": source, "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }
    memory = JsonMemoryStore(tempfile.mkdtemp(prefix="juno-memory-turn-plan-"))
    backend = _ExtractingTaskBackend(
        {"turn_planning_v1": plan},
        [
            {"term": "Lumare", "note": "name"},
            {"term": "Qwen adjudicator", "note": "product term"},
        ],
    )

    result = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    ).process_transcript(
        utterance_id="utt-memory-extractor-no-candidates",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=memory,
        memory_snapshot=memory.snapshot(),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    learned = {str(item.get("canonical_form") or item.get("term")) for item in memory.vocabulary.raw()}
    assert result.action == WriterActionKind.MEMORY_MUTATION
    assert result.output_text == ""
    assert "Lumare" in learned
    assert "Qwen adjudicator" in learned


def test_explicit_memory_learning_uses_spelled_terms_before_extractor_candidates() -> None:
    source = (
        "Teach Juno these terms: Loomer spelled L U M A R E, "
        "Ishida spelled I S H I D A, Qwen adjudicator spelled Q W E N adjudicator, "
        "and Cassini actions layer."
    )
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": source, "source_span": source, "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }
    memory = JsonMemoryStore(tempfile.mkdtemp(prefix="juno-memory-spelled-"))
    backend = _ExtractingTaskBackend(
        {"turn_planning_v1": plan},
        [
            {"term": "ISHIDA", "note": "spelled name"},
            {"term": "Qwen adjudicator", "note": "product term"},
            {"term": "Cassini actions layer", "note": "product term"},
        ],
    )

    result = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    ).process_transcript(
        utterance_id="utt-memory-spelled",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=memory,
        memory_snapshot=memory.snapshot(),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    rows = memory.vocabulary.raw()
    learned = {str(item.get("canonical_form") or item.get("term")): item for item in rows}
    assert result.action == WriterActionKind.MEMORY_MUTATION
    assert result.output_text == ""
    assert result.memory_updated is True
    assert "Lumare" in learned
    assert "Ishida" in learned
    assert "ISHIDA" not in learned
    assert "Qwen adjudicator" in learned
    assert "Cassini actions layer" in learned
    assert "Loomer" in learned["Lumare"].get("aliases", [])


def test_spelled_vocab_candidates_are_generic_and_grounded_by_surface() -> None:
    source = "Teach Juno these terms: Loomer spelled L U M A R E, Ishida spelled I S H I D A."
    candidates = _spelled_vocab_memory_candidates(source)

    assert candidates == [
        {
            "term": "Loomer",
            "canonical_form": "Lumare",
            "aliases": ["Loomer"],
            "pronunciation_hint": "spelled L U M A R E",
        },
        {
            "term": "Ishida",
            "canonical_form": "Ishida",
            "aliases": [],
            "pronunciation_hint": "spelled I S H I D A",
        },
    ]


def test_spelled_vocab_candidates_stop_before_adjacent_acronym_run() -> None:
    source = (
        "teach Juno these terms. Lumar is spelled L-U-M-A-R-E-M-L-X-L-M. "
        "Qwen Adjudicator is spelled Q-W-E-N, Cassini Actionslayer."
    )
    candidates = _spelled_vocab_memory_candidates(source)

    assert candidates == [
        {
            "term": "Qwen Adjudicator",
            "canonical_form": "Qwen Adjudicator",
            "aliases": [],
            "pronunciation_hint": "spelled Q W E N",
        },
    ]


def test_spelled_vocab_candidates_reject_ambiguous_hyphenated_asr_spelling_chains() -> None:
    source = (
        "Teach Juno these terms. Luma is spelled L-U-M-A-R-E-M-L-X-L-M "
        "is spelled M-L-X-L-M-Q-N-Adjudicator is spelled Q-W-E-N-Cassini-Actions-Layer."
    )
    candidates = _spelled_vocab_memory_candidates(source)

    assert candidates == []


def test_explicit_memory_learning_handles_asr_collapsed_spelling_without_overmerge() -> None:
    source = (
        "teach Juno these terms. Lumar is spelled L-U-M-A-R-E-M-L-X-L-M. "
        "Qwen Adjudicator is spelled Q-W-E-N, Cassini Actionslayer."
    )
    memory = JsonMemoryStore(tempfile.mkdtemp(prefix="juno-memory-collapsed-spelling-"))
    backend = _ExtractingTaskBackend(
        {"turn_planning_v1": "not json", "turn_repair_v1": "still not json"},
        [
            {"term": "Lumare-MLX LM", "note": "over-merged spelled term"},
            {"term": "Qwen Adjudicator", "note": "product term"},
        ],
    )

    result = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    ).process_transcript(
        utterance_id="utt-memory-collapsed-spelling",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=memory,
        memory_snapshot=memory.snapshot(),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    learned = {
        str(item.get("canonical_form") or item.get("term")): item
        for item in memory.vocabulary.raw()
        if str(item.get("canonical_form") or item.get("term")) != "Juno"
    }
    assert result.action == WriterActionKind.MEMORY_MUTATION
    assert result.output_text == ""
    assert result.memory_updated is True
    assert "Qwen Adjudicator" in learned
    assert "Cassini Actionslayer" in learned
    assert "Lumare-MLX LM" not in learned
    assert "Lumare" not in learned


def test_explicit_memory_learning_uses_clean_live_hint_when_final_asr_spelling_is_mangled() -> None:
    final_text = (
        "Teach Juno these terms. Luma is spelled L-U-M-A-R-E-M-L-X-L-M "
        "is spelled M-L-X-L-M-Q-N-Adjudicator is spelled Q-W-E-N-Cassini-Actions-Layer."
    )
    partial_text = (
        "Teach Juno these terms. Loomer is spelled L U M A R E. "
        "MLX LM is spelled M L X L M. "
        "Quinn Adjudicator is spelled Q W E N. Cassini Actions Layer."
    )
    memory = JsonMemoryStore(tempfile.mkdtemp(prefix="juno-memory-live-hint-spelling-"))
    backend = _ExtractingTaskBackend(
        {"turn_planning_v1": "not json", "turn_repair_v1": "still not json"},
        [
            {"term": "Luma", "canonical_form": "Lumaremlxlm", "note": "over-merged final ASR term"},
            {"term": "Qwen", "note": "partial product term"},
            {"term": "Cassini", "note": "partial product term"},
        ],
    )

    result = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    ).process_transcript(
        utterance_id="utt-memory-live-hint-spelling",
        final_text=final_text,
        raw_text=final_text,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=memory,
        memory_snapshot=memory.snapshot(),
        memory_packet={},
        partial_text=partial_text,
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    learned = {
        str(item.get("canonical_form") or item.get("term")): item
        for item in memory.vocabulary.raw()
        if str(item.get("canonical_form") or item.get("term")) != "Juno"
    }
    assert _memory_learning_source_text(final_text, partial_text) == partial_text
    assert result.action == WriterActionKind.MEMORY_MUTATION
    assert result.output_text == ""
    assert result.memory_updated is True
    assert set(learned) == {"Lumare", "MLX LM", "Qwen Adjudicator", "Cassini Actions Layer"}
    assert "Loomer" in learned["Lumare"].get("aliases", [])
    assert "Quinn Adjudicator" in learned["Qwen Adjudicator"].get("aliases", [])


def test_explicit_memory_learning_handles_asr_sentence_separated_spelling() -> None:
    source = (
        "teach Juno these terms. Loomer is spelled L U M A R E. "
        "MLX LM is spelled M L X L M. "
        "Quinn Adjudicator is spelled Q W E N. Cassini Actions Layer."
    )
    memory = JsonMemoryStore(tempfile.mkdtemp(prefix="juno-memory-sentence-spelling-"))
    backend = _ExtractingTaskBackend(
        {"turn_planning_v1": "not json", "turn_repair_v1": "still not json"},
        [
            {"term": "Loomer", "canonical_form": "Lumaremlxlm", "note": "over-merged spelled term"},
            {"term": "Qwen", "note": "partial product term"},
            {"term": "Cassini", "note": "partial product term"},
            {"term": "Actions", "note": "partial product term"},
        ],
    )

    result = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    ).process_transcript(
        utterance_id="utt-memory-sentence-spelling",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=memory,
        memory_snapshot=memory.snapshot(),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    learned = {
        str(item.get("canonical_form") or item.get("term")): item
        for item in memory.vocabulary.raw()
        if str(item.get("canonical_form") or item.get("term")) != "Juno"
    }
    assert result.action == WriterActionKind.MEMORY_MUTATION
    assert result.output_text == ""
    assert result.memory_updated is True
    assert set(learned) == {"Lumare", "MLX LM", "Qwen Adjudicator", "Cassini Actions Layer"}
    assert "Loomer" in learned["Lumare"].get("aliases", [])
    assert "Quinn Adjudicator" in learned["Qwen Adjudicator"].get("aliases", [])


def test_turn_plan_memory_candidates_do_not_silently_apply_without_learning_request() -> None:
    source = "Lumare and Qwen adjudicator are in the roadmap."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": source, "source_span": source, "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [
            {"term": "Lumare", "note": "name"},
            {"term": "Qwen adjudicator", "note": "product term"},
        ],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }
    memory = JsonMemoryStore(tempfile.mkdtemp(prefix="juno-memory-turn-plan-"))

    result = _service(plan).process_transcript(
        utterance_id="utt-memory-no-auto",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=memory,
        memory_snapshot=memory.snapshot(),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    learned = {str(item.get("canonical_form") or item.get("term")) for item in memory.vocabulary.raw()}
    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.memory_updated is False
    assert "Lumare" not in learned
    assert "Qwen adjudicator" not in learned


def test_turn_plan_terminal_render_preserves_command_text_without_markdown_cleanup() -> None:
    command = "git commit -m 'fix: preserve Qwen turn_plan_v1'"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": command},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "terminal",
            "markdown_allowed": False,
            "content_units": [{"kind": "command", "text": command, "source_span": command, "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

    rendered = render_turn_plan(plan, context=TypedContextBundle(app_category="terminal"), memory_store=None)

    assert rendered.text == command


def test_mlx_turn_plan_prompt_contract_is_json_and_no_fabrication() -> None:
    req = WriterTransformRequest(
        utterance_id="utt-prompt-contract",
        instruction="Plan the turn.",
        source_text="note down three points one alpha two beta",
        mode=WriterMode.DEFAULT_SURFACE,
        context_payload={"task": "turn_planning_v1", "payload": {"asr": {"final_text": "note down three points one alpha two beta"}}},
        metadata={"kind": "turn_planning_v1"},
    )

    prompt = _system_prompt(req).lower()
    user_payload = _build_writer_prompt(req)

    assert "strict json object" in prompt
    assert "do not add facts" in prompt
    assert "include only spoken items" in prompt
    assert json.loads(user_payload)["asr"]["final_text"] == "note down three points one alpha two beta"

    repair_req = WriterTransformRequest(
        utterance_id="utt-prompt-repair-contract",
        instruction="Repair the plan.",
        source_text="write a numbered list with 10 points first alpha second beta third gamma",
        mode=WriterMode.DEFAULT_SURFACE,
        context_payload={"task": "turn_repair_v1", "payload": {"source_text": "write a numbered list"}},
        metadata={"kind": "turn_repair_v1"},
    )
    repair_prompt = _system_prompt(repair_req).lower()
    assert "never invent missing entries" in repair_prompt
    assert "keep only spoken source items" in repair_prompt


def test_turn_planner_repairs_invalid_json_before_writer_fallback() -> None:
    source = "write hello world"
    repaired_plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": source},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "message",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "hello world", "source_span": "hello world", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }
    backend = _TaskBackend({"turn_planning_v1": "not json", "turn_repair_v1": repaired_plan})
    recorder = _Recorder()
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=recorder,
        backend=backend,
    )

    result = service.process_transcript(
        utterance_id="utt-repair",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Messages", app_category="messaging"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "hello world"
    assert result.metadata["turn_plan"]["repair_attempted"] is True
    assert [str((req.context_payload or {}).get("task")) for req in backend.requests] == [
        "turn_planning_v1",
        "turn_repair_v1",
    ]
    repair_payload = (backend.requests[-1].context_payload or {}).get("payload") or {}
    assert "source_payload" not in repair_payload
    assert repair_payload["source_text"] == source
    assert "allowed_values" in repair_payload


def test_turn_planner_uses_structural_fallback_after_invalid_repaired_plan() -> None:
    source = "Write a numbered list with 10 points. First alpha, second beta, third gamma."
    repaired_plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "format_dictation",
        "corrected_transcript": {
            "text": "1. alpha\n2. beta\n3. gamma\n4. delta\n5. epsilon\n6. zeta\n7. eta\n8. theta\n9. iota\n10. kappa"
        },
        "target": {"kind": "cursor", "confidence": 0.9},
        "render_plan": {
            "render_kind": "numbered_list",
            "markdown_allowed": False,
            "claimed_item_count": 10,
            "spoken_item_count": 10,
            "content_units": [
                {"kind": "item", "text": "1. alpha", "source_span": "First alpha", "order": 1},
                {"kind": "item", "text": "2. beta", "source_span": "second beta", "order": 2},
                {"kind": "item", "text": "3. gamma", "source_span": "third gamma", "order": 3},
                {"kind": "item", "text": "4. delta", "source_span": "delta", "order": 4},
                {"kind": "item", "text": "5. epsilon", "source_span": "epsilon", "order": 5},
                {"kind": "item", "text": "6. zeta", "source_span": "zeta", "order": 6},
                {"kind": "item", "text": "7. eta", "source_span": "eta", "order": 7},
                {"kind": "item", "text": "8. theta", "source_span": "theta", "order": 8},
                {"kind": "item", "text": "9. iota", "source_span": "iota", "order": 9},
                {"kind": "item", "text": "10. kappa", "source_span": "kappa", "order": 10},
            ],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }
    backend = _TaskBackend({"turn_planning_v1": "not json", "turn_repair_v1": repaired_plan})
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    )

    result = service.process_transcript(
        utterance_id="utt-list-invalid-repair-fallback",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "1. alpha\n2. beta\n3. gamma"
    assert "\n4." not in result.output_text
    assert result.metadata["turn_plan"]["repair_attempted"] is True
    assert result.metadata["turn_plan"]["repair_status"] == "fallback"
    assert result.metadata["turn_plan"]["validation_errors_before_repair"]
    assert result.metadata["turn_plan"]["render"]["claimed_item_count"] == 10
    assert result.metadata["turn_plan"]["render"]["spoken_item_count"] == 3


def test_turn_plan_missing_recent_deterministic_target_falls_back_to_parser() -> None:
    source = "Turn that into bullets"
    recent = "Verify microphone permission. Run action combos. Check final paste."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "transform",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 0.5},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {
            "operation": "bullets",
            "instruction": "Turn into bullets",
            "transformed_text": None,
            "requires_second_pass": False,
        },
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

    result = _service(plan).process_transcript(
        utterance_id="utt-recent-bullets-planner-shadow",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(
            app_name="Notes",
            app_category="docs",
            metadata={
                "last_committed_text": recent,
                "last_committed_start": 4,
                "last_committed_end": 4 + len(recent),
            },
        ),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.output_text == "- Verify microphone permission\n- Run action combos\n- Check final paste"
    assert result.metadata["target"] == "recent_commit"


def test_turn_plan_missing_recent_model_target_uses_recent_commit_for_generation() -> None:
    source = "Make that shorter"
    recent = "This is a long launch-readiness update with repeated details about permissions, actions, and paste checks."
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "transform",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 0.5},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {
            "operation": "rewrite",
            "instruction": "Make the recent text more concise.",
            "transformed_text": None,
            "requires_second_pass": True,
        },
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }
    backend = _TaskBackend({
        "turn_planning_v1": plan,
        "transform_generation_v1": {"transformed_text": "Short launch update."},
    })
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    )

    result = service.process_transcript(
        utterance_id="utt-recent-model-planner-shadow",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(
            app_name="Notes",
            app_category="docs",
            metadata={
                "last_committed_text": recent,
                "last_committed_start": 8,
                "last_committed_end": 8 + len(recent),
            },
        ),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.output_text == "Short launch update."
    assert result.commit_mode == CommitMode.REPLACE_SELECTION
    assert result.selection_override == ClientSelection(start=8, end=8 + len(recent))
    assert result.metadata["target"] == "recent_commit"
    assert result.metadata["target_text_chars"] == len(recent)


def _actions_plan_with_one_ungrounded_body(source: str) -> dict[str, Any]:
    return {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "alarm",
                "operation": "create",
                "body": "call Atharv",
                "evidence_span": "tomorrow 9pm to call Atharv",
                "schedule": {"kind": "instant", "source_span": "tomorrow 9pm"},
                "missing_fields": [],
            },
            {
                # Planner rewrote the note body so it is no longer a source
                # span, and gave no evidence (production 2026-06-11:
                # action_4_body_not_grounded rejected the whole plan).
                "kind": "note",
                "operation": "create",
                "body": "Fix every bug in the Juno launch build",
                "evidence_span": "",
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }


def test_per_action_validation_failures_are_warnings_not_plan_fatal() -> None:
    source = "Hey Juno, set an alarm tomorrow 9pm to call Atharv, add a note about bugs"
    plan = _actions_plan_with_one_ungrounded_body(source)

    validation = validate_turn_plan(plan, source_text=source, context=TypedContextBundle(app_category="unknown"))

    assert validation.ok
    assert validation.errors == []
    assert "action_1_body_not_grounded" in validation.warnings

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 11, 13, 33))

    assert parsed.actions is not None
    assert [a.kind.value for a in parsed.actions] == ["alarm"]
    assert parsed.skipped_reasons == ["action_1_body_not_grounded"]


def test_single_bad_action_does_not_trigger_repair_pass() -> None:
    source = "Hey Juno, set an alarm tomorrow 9pm to call Atharv, add a note about bugs"
    # No turn_repair_v1 response registered: a repair attempt would KeyError.
    backend = _TaskBackend({"turn_planning_v1": _actions_plan_with_one_ungrounded_body(source)})
    recorder = _Recorder()
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=recorder,
        backend=backend,
    )

    result = service.process_transcript(
        utterance_id="utt-partial-actions",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="System Settings", app_category="unknown"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result is not None
    assert [str((req.context_payload or {}).get("task")) for req in backend.requests] == ["turn_planning_v1"]
    generated = [
        args[2]
        for args, _kwargs in recorder.events
        if len(args) > 2 and args[1] == "turn_plan_generated"
    ]
    assert len(generated) == 1
    payload = generated[0]
    assert payload["repair_attempted"] is False
    assert payload["validation_ok"] is True
    # Contract change 2026-06-11: compound segments are authoritative — the
    # ungrounded sibling body is re-grounded inside its own segment during
    # normalization instead of surviving as a validation warning.
    assert "action_1_body_not_grounded" not in payload["validation_warnings"]
    assert any("segment" in n for n in payload["normalization_notes"])


def test_unusable_repair_restores_initial_plan() -> None:
    source = "Hey Juno, set an alarm tomorrow 9pm to call Atharv, add a note about bugs"
    initial_plan = _actions_plan_with_one_ungrounded_body(source)
    # The repair decode echoes the turn_repair_v1 request back instead of a
    # plan — exactly what production logged on 2026-06-11.
    echoed_request = {
        "task": "turn_repair_v1",
        "utterance_id": "utt-echo",
        "source_text": source,
        "allowed_values": {"utterance_kind": ["dictation", "actions"]},
        "invalid": {"status": "ok", "errors": []},
    }
    backend = _TaskBackend({"turn_repair_v1": echoed_request})
    planner = TurnPlanner(backend)
    packet = TurnPlanPacket(
        utterance_id="utt-echo",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="System Settings", app_category="unknown"),
    )
    prior = TurnPlanResult(plan=initial_plan, status="ok")

    repaired = planner.repair(packet, prior, validation_errors=["action_1_body_not_grounded"])

    assert repaired.ok
    assert repaired.plan is initial_plan
    assert repaired.repair_attempted is True
    assert str(repaired.repair_status).startswith("unusable_repair_kept_initial:")
    assert "repair_unusable_initial_plan_restored" in repaired.normalization_notes


def test_turn_planner_uses_generic_structural_fallback_after_bad_qwen_json() -> None:
    source = "Note down 10 points First remove patches Second make Qwen plan Third validate spans"
    repair_wrapper = {
        "task": "turn_repair_v1",
        "utterance_id": "utt-list-fallback",
        "source_text": source,
        "invalid": {"raw_output": "not json"},
    }
    backend = _TaskBackend({"turn_planning_v1": "not json", "turn_repair_v1": repair_wrapper})
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    )

    result = service.process_transcript(
        utterance_id="utt-list-fallback",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
    )

    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "1. remove patches\n2. make Qwen plan\n3. validate spans"
    assert result.learn_from_commit is False
    assert result.metadata["turn_plan"]["repair_attempted"] is True
    assert result.metadata["turn_plan"]["repair_status"] == "fallback"
    assert result.metadata["turn_plan"]["render"]["claimed_item_count"] == 10
    assert result.metadata["turn_plan"]["render"]["spoken_item_count"] == 3
    assert [str((req.context_payload or {}).get("task")) for req in backend.requests] == [
        "turn_planning_v1",
        "turn_repair_v1",
    ]


def test_structural_fallback_supports_spoken_cardinal_markers() -> None:
    plan = _fallback_structural_turn_plan("write down three items one alpha two beta three gamma")

    assert plan is not None
    render = plan["render_plan"]
    assert render["claimed_item_count"] == 3
    assert render["spoken_item_count"] == 3
    assert [unit["text"] for unit in render["content_units"]] == ["alpha", "beta", "gamma"]


def test_pipeline_turn_plan_controls_mixed_actions_and_writer_paste() -> None:
    source = "juno remind me to call mom at 6 PM and write running ten minutes late"
    post_wake = "remind me to call mom at 6 PM and write running ten minutes late"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "mixed",
        "corrected_transcript": {"text": post_wake},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "message",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": "Running ten minutes late.", "source_span": "running ten minutes late", "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call mom",
                "evidence_span": "call mom",
                "schedule": {"kind": "instant", "source_span": "at 6 PM"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

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
            return TypedContextBundle(app_name="Messages", app_category="messaging")

    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=_TurnPlanBackend(plan),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=_Recorder(),
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-mixed-turn-plan",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.actions is not None
    assert result.actions[0]["kind"] == "reminder"
    assert result.transcript == ""
    assert result.paste_kind == "none"
    assert result.is_action is True
    assert result.metadata["turn_plan"]["mixed_paste_allowed"] is False


def test_pipeline_wake_action_is_not_demoted_by_question_words_inside_note_body() -> None:
    post_wake = (
        "take a note title What is Juno and why does it exist "
        "and set an alarm for 4pm to launch Juno on Product Hunt"
    )
    source = f"Hey Juno, {post_wake}"
    recorder = _Recorder()
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": post_wake},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "title What is Juno and why does it exist",
                "evidence_span": "title What is Juno and why does it exist",
                "schedule": {"kind": "none"},
                "missing_fields": [],
            },
            {
                "kind": "alarm",
                "operation": "create",
                "body": "launch Juno on Product Hunt",
                "evidence_span": "set an alarm for 4pm to launch Juno on Product Hunt",
                "schedule": {"kind": "instant", "source_span": "4pm"},
                "missing_fields": [],
            },
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
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=recorder,
        backend=_TaskBackend({"turn_planning_v1": plan}),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-question-word-action-note",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.transcript == ""
    assert result.paste_kind == "none"
    assert result.actions is not None
    assert [action["kind"] for action in result.actions] == ["note", "alarm"]
    event_names = [args[1] for args, _ in recorder.events if len(args) >= 2]
    assert "action_wake_gate_demoted" not in event_names
    assert "turn_plan_actions_detected" in event_names


def test_pipeline_invalid_turn_plan_routes_valid_action_to_regex_fallback() -> None:
    source = "juno remind me at 3 PM to call Ishida"
    recorder = _Recorder()

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
            return TypedContextBundle(app_name="Messages", app_category="messaging")

    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=recorder,
        backend=_TaskBackend({"turn_planning_v1": "not json", "turn_repair_v1": "still not json"}),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-invalid-plan-action-rejected",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.transcript == ""
    assert result.paste_kind == "none"
    assert result.is_action is True
    assert result.actions is not None
    assert result.actions[0]["kind"] == "reminder"
    assert result.actions[0]["body"] == "call Ishida"
    event_names = [args[1] for args, _ in recorder.events if len(args) >= 2]
    assert "turn_plan_action_routed_to_fallback" in event_names
    assert "turn_plan_action_fallback_used" in event_names


def test_pipeline_valid_actionless_turn_plan_is_respected_without_regex_fallback() -> None:
    source = "juno remind me to go to disco with Ishida at 9pm"
    post_wake = "remind me to go to disco with Ishida at 9pm"
    recorder = _Recorder()
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "dictation",
        "corrected_transcript": {"text": post_wake},
        "target": {"kind": "cursor", "confidence": 1.0},
        "render_plan": {
            "render_kind": "plain",
            "markdown_allowed": False,
            "content_units": [{"kind": "paragraph", "text": post_wake, "source_span": post_wake, "order": 1}],
        },
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "no_execute"},
        "uncertainties": [],
    }

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
            return TypedContextBundle(app_name="Messages", app_category="messaging")

    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=recorder,
        backend=_TaskBackend({"turn_planning_v1": plan}),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-valid-actionless-plan",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.actions is None
    assert result.transcript == post_wake
    assert result.paste_kind == "insert"
    assert result.noop_reason is None
    assert result.is_action is False
    event_names = [args[1] for args, _ in recorder.events if len(args) >= 2]
    assert "turn_plan_action_fallback_used" not in event_names


def test_action_detection_does_not_let_llm_veto_grounded_wake_action_grammar() -> None:
    recorder = _Recorder()

    def fake_non_action_extractor(text: str, now: datetime | None) -> dict[str, Any]:
        return {
            "schema_version": "actions_intent_v2",
            "intent": "dictation",
            "should_execute": False,
            "confidence": 1.0,
            "decision_evidence_span": text,
            "actions": [],
        }

    set_llm_extractor(fake_non_action_extractor)
    try:
        actions = detect_actions_for_pipeline(
            utterance_id="utt-llm-veto-note",
            normalized_text="take a note Project Atlas is ready",
            recorder=recorder,
            trace_kind="system",
            now=datetime(2026, 6, 9, 12, 0),
            wake_verified=True,
            raw_wake_text="Juno take a note Project Atlas is ready",
            context_packet=None,
        )
    finally:
        set_llm_extractor(None)

    assert actions is not None
    assert actions[0].kind.value == "note"
    assert actions[0].body == "Project Atlas is ready"
    event_names = [args[1] for args, _ in recorder.events if len(args) >= 2]
    assert "action_extraction_llm_veto_overridden" in event_names


def test_pipeline_preserves_leading_juno_when_not_action_intent() -> None:
    source = "Juno should write the product name in this sentence."
    recorder = _Recorder()

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

    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-leading-juno-dictation",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.actions is None
    assert result.transcript == source
    assert result.paste_kind == "insert"
    assert result.is_action is False
    event_names = [args[1] for args, _ in recorder.events if len(args) >= 2]
    assert "action_wake_gate_demoted" not in event_names


def test_pipeline_recovers_note_action_source_from_live_hint_near_miss() -> None:
    final_asr = "Juno Taker Note Project Atlas is ready for launch."
    live_hint = "Juno take a note Project Atlas is ready for launch."
    post_wake = "take a note Project Atlas is ready for launch."
    recorder = _Recorder()
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": post_wake},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": "Project Atlas is ready for launch.",
                "evidence_span": "Project Atlas is ready for launch",
                "schedule": {"kind": "none", "source_span": ""},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript=final_asr,
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=recorder,
        backend=_TaskBackend({"turn_planning_v1": plan}),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-live-hint-note-action",
        transcript_hint=live_hint,
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.transcript == ""
    assert result.paste_kind == "none"
    assert result.is_action is True
    assert result.actions is not None
    assert result.actions[0]["kind"] == "note"
    assert result.actions[0]["body"] == "Project Atlas is ready for launch."
    event_names = [args[1] for args, _ in recorder.events if len(args) >= 2]
    assert "action_source_recovered_from_live_hint" in event_names


def test_memory_extraction_filters_common_words_and_ungrounded_snippets() -> None:
    common = _filter_memory_extraction_candidates(
        kind="vocab",
        source_text="finally earlier also tomorrow please write this sentence clearly",
        candidates=[
            {"term": "tomorrow", "note": "common"},
            {"term": "clearly", "note": "common"},
            {"term": "write this sentence", "note": "generic"},
        ],
        limit=6,
    )
    assert common == []

    terms = _filter_memory_extraction_candidates(
        kind="vocab",
        source_text="Teach Juno these terms: Lumare, Ishida, MLX LM, Qwen adjudicator, Cassini actions layer.",
        candidates=[
            {"term": "Teach Juno", "note": "command"},
            {"term": "Lumare", "note": "name"},
            {"term": "Ishida", "note": "name"},
            {"term": "MLX LM", "note": "acronym"},
            {"term": "Qwen adjudicator", "note": "product term"},
            {"term": "Cassini actions layer", "note": "product phrase"},
        ],
        limit=8,
    )
    assert [item["term"] for item in terms] == [
        "Lumare",
        "Ishida",
        "MLX LM",
        "Qwen adjudicator",
        "Cassini actions layer",
    ]

    snippets = _filter_memory_extraction_candidates(
        kind="snippet",
        source_text="Teach Juno these terms: Lumare and Ishida.",
        candidates=[
            {
                "trigger": "lumare",
                "body": "Lumare is a proprietary AI platform for decision support.",
            }
        ],
        limit=4,
    )
    assert snippets == []


def _loud_wav_bytes() -> bytes:
    sample_rate = 16_000
    frames = []
    for i in range(sample_rate // 2):
        sample = int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        frames.append(struct.pack("<h", sample))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(frames))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Span grounding regressions (2026-06-10 "e titled," production corruption)
# --------------------------------------------------------------------------- #
#
# The model re-types spans inside the JSON plan. Production shipped an Apple
# Note whose body began "e titled, …" because a mid-word fragment of
# "…take a not|e titled…" passed the character-level grounding check. Span
# containment must align to token boundaries, and a misaligned retyped span
# must be repaired into a clean grounded body, not accepted verbatim.

_NOTE_SOURCE = (
    "take a note titled, What is Juno and why does it exist? New paragraph, "
    "text is still the main interface between models and people, but voice is "
    "how we actually think, interrupt ourselves, and move fast."
)


def test_span_present_requires_token_boundaries() -> None:
    from juno_v2.turn_plan.validators import span_present

    assert span_present("note titled, What is Juno", _NOTE_SOURCE)
    # mid-word fragment of "…not|e titled…" must NOT count as grounded
    assert not span_present("e titled, what is Juno and why does it exist", _NOTE_SOURCE)
    # single mid-word token
    assert not span_present("nterface between models", _NOTE_SOURCE)


def test_misaligned_retyped_note_body_is_repaired_to_clean_grounded_body() -> None:
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": _NOTE_SOURCE},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                # model-retyped: starts mid-word, case drifted
                "body": (
                    "e titled, what is Juno and why does it exist? New paragraph, "
                    "text is still the main interface between models and people, but "
                    "voice is how we actually think, interrupt ourselves, and move fast."
                ),
                "evidence_span": _NOTE_SOURCE,
                "schedule": {"kind": "none"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    normalized, notes = normalize_turn_plan(plan, source_text=_NOTE_SOURCE)
    body = str(normalized["actions"][0]["body"])
    assert body.startswith("What is Juno and why does it exist?"), body
    assert "e titled" not in body
    assert "action_body_repaired_from_evidence" in notes

    parsed = actions_from_turn_plan(normalized, source_text=_NOTE_SOURCE, now=datetime(2026, 6, 10, 9, 0))
    assert parsed.actions is not None
    assert parsed.actions[0].kind.value == "note"
    assert parsed.actions[0].body.startswith("What is Juno and why does it exist?")


def test_truncated_retyped_evidence_is_snapped_not_discarded() -> None:
    from juno_v2.turn_plan.planner import _snap_span_to_source

    source = "remind me to call Bob at 6pm and write running 10 minutes late"
    # tail truncation ("…and w" for "…and write") keeps the grounded prefix
    assert (
        _snap_span_to_source("remind me to call Bob at 6pm and w", source)
        == "remind me to call Bob at 6pm and"
    )
    # leading mid-word token is trimmed from the start
    assert _snap_span_to_source("e to call Bob at 6pm", source) == "to call Bob at 6pm"
    # hopeless spans yield nothing
    assert _snap_span_to_source("completely unrelated words here", source) == ""


# --------------------------------------------------------------------------- #
# Action-lane safety regressions (2026-06-10 production failures)
# --------------------------------------------------------------------------- #


def test_pipeline_ignores_planner_actions_without_action_verb() -> None:
    # Production U3: "Hey Juno, I did not reinstall…" — no action verb, but the
    # planner invented a note and the user's dictation was paste-suppressed
    # into Notes. Wake alone is not consent; without a deterministic action
    # signal the utterance must stay dictation and be pasted.
    source = "juno I did not reinstall the app in the past and I did not run installed app UI interactions"
    post_wake = "I did not reinstall the app in the past and I did not run installed app UI interactions"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": post_wake},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "note",
                "operation": "create",
                "body": post_wake,
                "evidence_span": post_wake,
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
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(app_name="Terminal", app_category="developer_tools")

    recorder = _Recorder()
    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=_TurnPlanBackend(plan),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-no-verb-hallucinated-note",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.actions is None
    assert result.is_action is False
    assert result.paste_kind != "none"
    assert "did not reinstall the app" in result.transcript
    event_names = [args[1] for args, _ in recorder.events if len(args) > 1]
    assert "turn_plan_actions_ignored_without_verb" in event_names


def test_actions_plan_reaching_writer_never_suppresses_text() -> None:
    # The writer is reached only when the pipeline dispatched no actions, so
    # an actions-kind plan must come back as text delivery, never as an
    # empty-output NOOP (failure_reason=turn_plan_action_only data loss).
    source = "send the launch summary to the team tonight"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                "kind": "reminder",
                "operation": "create",
                "body": "send the launch summary to the team",
                "evidence_span": "send the launch summary to the team tonight",
                "schedule": {"kind": "instant", "source_span": "tonight"},
                "missing_fields": [],
            }
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    result = _service(plan).process_transcript(
        utterance_id="utt-writer-actions-text-guarantee",
        final_text=source,
        raw_text=source,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=MemorySnapshot(schema_version=1),
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=_selection(),
        wake_verified=True,
    )

    assert result.output_text.strip(), "writer must never return empty output for an actions plan"
    assert result.action != WriterActionKind.NOOP


def test_rejected_action_attempt_keeps_transcript_in_history() -> None:
    # A genuine failed command ("set an alarm" with an unparseable time)
    # still suppresses the paste, but the spoken words must survive in
    # History for recovery instead of vanishing.
    source = "juno set an alarm to publish the changelog"
    post_wake = "set an alarm to publish the changelog"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": post_wake},
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
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(app_name="Notes", app_category="docs")

    recorder = _Recorder()
    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=_TurnPlanBackend(plan),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-rejected-alarm-history",
        save_history=True,
        save_audio=False,
    )

    assert result.ok
    assert result.paste_kind == "none"

    import pathlib
    import sqlite3

    db_path = pathlib.Path(recorder.log_dir) / "product_history.sqlite"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT transcript, failure_reason FROM utterances WHERE utterance_id = ?",
            ("utt-rejected-alarm-history",),
        ).fetchone()
    assert row is not None
    transcript, failure_reason = row
    assert "publish the changelog" in (transcript or "")
    assert failure_reason == "action_rejected"


def test_turn_plan_unsupported_operation_routes_to_extractor_fallback() -> None:
    # The turn-plan lane only creates; operations (complete/update/delete)
    # belong to the extractor lane. A planner operation must route there
    # instead of failing the turn outright.
    source = "juno remind me tomorrow at 9.15am to send the brief"
    post_wake = "remind me tomorrow at 9.15am to send the brief"
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": post_wake},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                # Planner mis-tags this as an update on an existing item.
                "kind": "reminder",
                "operation": "update",
                "body": "send the brief",
                "evidence_span": "remind me tomorrow at 9.15am to send the brief",
                "schedule": {"kind": "instant", "source_span": "tomorrow at 9.15am"},
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
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(app_name="Notes", app_category="docs")

    def broken_extractor(text: str, now: datetime | None) -> dict[str, Any]:
        raise RuntimeError("model unavailable")  # → allow_regex_fallback path

    recorder = _Recorder()
    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=_TurnPlanBackend(plan),
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    set_llm_extractor(broken_extractor)
    try:
        result = pipeline.run(
            _loud_wav_bytes(),
            utterance_id="utt-operation-fallback",
            save_history=False,
            save_audio=False,
        )
    finally:
        set_llm_extractor(None)

    assert result.ok
    assert result.actions is not None, "operation rejection must fall back to the extractor lane"
    assert result.actions[0]["kind"] == "reminder"
    assert result.is_action is True
    assert result.paste_kind == "none"
    event_names = [args[1] for args, _ in recorder.events if len(args) >= 2]
    assert "turn_plan_operation_routed_to_fallback" in event_names
    assert "turn_plan_action_fallback_used" in event_names


def test_action_invalid_json_turn_plan_routes_to_extractor_fallback() -> None:
    source = "Juno take a note Cassini launches on Monday."

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

    backend = _TaskBackend({"turn_planning_v1": "not json", "turn_repair_v1": "still not json"})
    recorder = _Recorder()
    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    set_llm_extractor(None)
    result = pipeline.run(_loud_wav_bytes(), utterance_id="utt-action-invalid-json-fallback", save_history=False, save_audio=False)

    assert result.ok
    assert result.paste_kind == "none"
    assert result.is_action is True
    assert result.actions is not None
    assert result.actions[0]["kind"] == "note"
    assert result.actions[0]["body"] == "Cassini launches on Monday"
    assert result.metadata["turn_plan"]["status"] == "invalid_json"
    event_names = [args[1] for args, _ in recorder.events if len(args) >= 2]
    assert "turn_plan_action_routed_to_fallback" in event_names
    assert "turn_plan_action_fallback_used" in event_names


def test_scratched_at_asr_variant_retake_runs_before_action_fallback() -> None:
    source = "Juno remind me at 3 p.m. scratched at 4.15 p.m. to call Sam."

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

    backend = _TaskBackend({"turn_planning_v1": "not json", "turn_repair_v1": "still not json"})
    writer = WriterService(
        config=WriterConfig(enable_model_transforms=True, enable_turn_planner=True, dictation_editor_enabled=False),
        recorder=_Recorder(),
        backend=backend,
    )
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=_Recorder(),
        context_provider=FakeContextProvider(),
        writer_service=writer,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    set_llm_extractor(None)
    result = pipeline.run(_loud_wav_bytes(), utterance_id="utt-scratched-at-action", save_history=False, save_audio=False)

    assert result.ok
    assert result.paste_kind == "none"
    assert result.actions is not None
    assert result.actions[0]["kind"] == "reminder"
    assert result.actions[0]["body"] == "call Sam"
    assert result.actions[0]["when"] is not None
    assert "4:15" in result.actions[0]["when"]["iso"] or "16:15" in result.actions[0]["when"]["iso"]
    assert any(
        applied.get("rule") == "self_correction_retakes"
        for applied in result.normalization_applied
    )


def test_compound_six_actions_segment_authoritative_semantics() -> None:
    # Production 2026-06-11 case 19: six actions passed count checks while
    # bodies/spans/times bled across siblings. Segments are now authoritative:
    # exact bodies, kinds, and times — from a deliberately corrupted batch.
    src = (
        "take a note checkpoint alpha and take a note checkpoint beta and remind me tomorrow at 10am "
        "to call Sam and remind me Friday at 2pm to review metrics and set an alarm for 4.30pm to publish "
        "and set an alarm in 45 minutes to stand up"
    )
    corrupted = [
        {"kind": "note", "operation": "create", "body": "checkpoint alpha and take a note checkpoint beta",
         "evidence_span": "take a note checkpoint alpha and take a note checkpoint beta", "schedule": {"kind": "none"}, "missing_fields": []},
        {"kind": "note", "operation": "create", "body": "checkpoint alpha and take a note checkpoint beta",
         "evidence_span": "take a", "schedule": {"kind": "none"}, "missing_fields": []},
        {"kind": "reminder", "operation": "create", "body": "call Sam",
         "evidence_span": "beta and", "schedule": {"kind": "instant", "source_span": "tomorrow at 10am"}, "missing_fields": []},
        {"kind": "reminder", "operation": "create", "body": "Friday at 2pm to review metrics",
         "evidence_span": "remind me", "schedule": {"kind": "instant", "source_span": "tomorrow at 10am"}, "missing_fields": []},
        {"kind": "alarm", "operation": "create", "body": "publish",
         "evidence_span": "and set", "schedule": {"kind": "instant", "source_span": "4.30pm"}, "missing_fields": []},
        {"kind": "alarm", "operation": "create", "body": "stand up",
         "evidence_span": "and set", "schedule": {"kind": "instant", "source_span": "in 45 minutes"}, "missing_fields": []},
    ]
    plan = {
        "schema_version": "turn_plan_v1", "utterance_kind": "actions",
        "corrected_transcript": {"text": src},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": corrupted, "snippets": [], "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"}, "uncertainties": [],
    }
    from datetime import timedelta, timezone

    normalized, notes = normalize_turn_plan(plan, source_text=src)
    now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone(timedelta(hours=7)))  # Thursday
    parsed = actions_from_turn_plan(normalized, source_text=src, now=now)
    assert parsed.actions is not None and len(parsed.actions) == 6
    expected = [
        ("note", "checkpoint alpha", None),
        ("note", "checkpoint beta", None),
        ("reminder", "call Sam", "2026-06-12T10:00:00+07:00"),
        ("reminder", "review metrics", "2026-06-12T14:00:00+07:00"),
        ("alarm", "publish", "2026-06-11T16:30:00+07:00"),
        ("alarm", "stand up", "2026-06-11T12:45:00+07:00"),
    ]
    for action, (kind, body, when_iso) in zip(parsed.actions, expected):
        assert action.kind.value == kind
        assert action.body == body
        assert (action.when.iso if action.when else None) == when_iso
        # No body may carry a sibling's native-action anchor.
        for anchor in ("take a note", "remind me", "set an alarm"):
            assert anchor not in action.body

def _complex_five_action_source() -> str:
    # Production 2026-06-11 23:40 (utterance macshell-44FD9D72): five actions
    # in one breath, including a dotted clock and a trailing compound.
    return (
        "Hey Juno, set up two alarms. First, to call my wife at 10 pm tomorrow. "
        "Second, to call my brother at 11 pm tomorrow. Third, to call my friend "
        "at 11.30 pm tomorrow. Add a note to fix all the issues and put Juno on "
        "product hunt tomorrow 12pm. And remind me to call Darpan tomorrow 3pm."
    )


def test_complex_compound_alarms_all_carry_correct_times() -> None:
    # The well-formed plan for the production utterance: every alarm must
    # dispatch with the exact spoken time — including the dotted "11.30 pm".
    source = _complex_five_action_source()

    def alarm(body: str, span: str, evidence: str) -> dict:
        return {
            "kind": "alarm",
            "operation": "create",
            "body": body,
            "evidence_span": evidence,
            "schedule": {"kind": "instant", "source_span": span},
            "missing_fields": [],
        }

    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            alarm("call my wife", "10 pm tomorrow", "call my wife at 10 pm tomorrow"),
            alarm("call my brother", "11 pm tomorrow", "call my brother at 11 pm tomorrow"),
            alarm("call my friend", "11.30 pm tomorrow", "call my friend at 11.30 pm tomorrow"),
            {
                "kind": "note",
                "operation": "create",
                "body": "fix all the issues and put Juno on product hunt tomorrow 12pm",
                "evidence_span": "Add a note to fix all the issues and put Juno on product hunt tomorrow 12pm",
                "missing_fields": [],
            },
            {
                "kind": "reminder",
                "operation": "create",
                "body": "call Darpan",
                "evidence_span": "remind me to call Darpan tomorrow 3pm",
                "schedule": {"kind": "instant", "source_span": "tomorrow 3pm"},
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 11, 23, 40))

    assert parsed.actions is not None
    assert parsed.skipped_reasons == []
    assert [a.kind.value for a in parsed.actions] == ["alarm", "alarm", "alarm", "note", "reminder"]
    times = [(a.when.iso if a.when else None) for a in parsed.actions]
    # Every alarm and the reminder carry an exact instant; only the note is untimed.
    assert times[0] is not None and times[0].startswith("2026-06-12T22:00")
    assert times[1] is not None and times[1].startswith("2026-06-12T23:00")
    assert times[2] is not None and times[2].startswith("2026-06-12T23:30"), (
        f"dotted '11.30 pm' must parse exactly: {times[2]}"
    )
    assert times[3] is None
    assert times[4] is not None and times[4].startswith("2026-06-12T15:00")
    # The invariant behind all of this: no alarm ever dispatches timeless.
    for action in parsed.actions:
        if action.kind.value == "alarm":
            assert action.when is not None


def test_sloppy_planner_alarms_never_dispatch_timeless() -> None:
    # The model variant production actually emitted that night: mangled
    # bodies, "vague" schedule kinds, and time spans reduced to "tomorrow".
    # The shell hard-fails timeless alarms ("An alarm needs a time."), so
    # coercion must SKIP them — never ship them — while valid siblings and
    # untimed notes still go through.
    source = _complex_five_action_source()
    plan = {
        "schema_version": "turn_plan_v1",
        "utterance_kind": "actions",
        "corrected_transcript": {"text": source},
        "target": {"kind": "none", "confidence": 1.0},
        "render_plan": {"render_kind": "none", "markdown_allowed": False, "content_units": []},
        "transform": {"operation": "none", "instruction": "", "transformed_text": None, "requires_second_pass": False},
        "actions": [
            {
                # Mangled body, vague schedule, no span: must SKIP, not ship.
                "kind": "alarm",
                "operation": "create",
                "body": "call my",
                "evidence_span": "call my",
                "schedule": {"kind": "vague", "source_span": ""},
                "missing_fields": [],
            },
            {
                # Valid sibling: must still dispatch with its exact time.
                "kind": "alarm",
                "operation": "create",
                "body": "call my brother",
                "evidence_span": "call my brother at 11 pm tomorrow",
                "schedule": {"kind": "instant", "source_span": "11 pm tomorrow"},
                "missing_fields": [],
            },
            {
                # Mangled body but a real time hiding in the evidence: the
                # grounded-span inference may rescue it — and if it cannot,
                # the action must skip rather than ship timeless.
                "kind": "alarm",
                "operation": "create",
                "body": "all the issues",
                "evidence_span": "all the issues",
                "schedule": {"kind": "vague", "source_span": ""},
                "missing_fields": [],
            },
            {
                "kind": "note",
                "operation": "create",
                "body": "fix all the issues",
                "evidence_span": "fix all the issues",
                "missing_fields": [],
            },
        ],
        "snippets": [],
        "memory_candidates": [],
        "safety": {"commit_policy": "no_commit", "execute_policy": "execute"},
        "uncertainties": [],
    }

    parsed = actions_from_turn_plan(plan, source_text=source, now=datetime(2026, 6, 11, 23, 40))

    assert parsed.actions is not None
    # The hard invariant: NO dispatched alarm may lack a time.
    for action in parsed.actions:
        if action.kind.value == "alarm":
            assert action.when is not None, f"timeless alarm dispatched: {action.body!r}"
    # The valid alarm and the note survived; the unparseable alarms skipped.
    kinds = [a.kind.value for a in parsed.actions]
    assert "note" in kinds
    assert kinds.count("alarm") >= 1
    assert any("time_parse_failed" in r for r in parsed.skipped_reasons)
