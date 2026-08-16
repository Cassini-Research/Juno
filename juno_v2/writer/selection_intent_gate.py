"""Second-stage intent when the parser returned DICTATE but text is selected.

Avoids pasting free-form edit instructions into the document when the user
highlighted a span and spoke a natural-language transformation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.modes import ModePolicy
from juno_v2.contracts.tracing import TraceKind
from juno_v2.contracts.workbench import ClientSelection
from juno_v2.writer.selection_commands import (
    looks_like_selection_edit_command,
    recognize_selection_transform_command,
)

if TYPE_CHECKING:
    from juno_v2.observability.tracing import TraceRecorder
    from juno_v2.writer.backends.base import WriterBackend

ConfidenceEdit = 0.72
ConfidenceDictate = 0.7
AmbiguousBand = 0.08
LongDictationWordFloor = 24


@dataclass(slots=True)
class SelectionIntentResolution:
    kind: Literal["dictate", "edit", "ambiguous"]
    instruction: str | None = None
    confidence: float | None = None
    source: str = "none"


def resolve_selection_intent_after_dictate_parse(
    *,
    recorder: TraceRecorder,
    utterance_id: str,
    spoken_final: str,
    context: TypedContextBundle,
    anchor_selection: ClientSelection | None,
    mode_policy: ModePolicy | None,
    backend: WriterBackend | None,
) -> SelectionIntentResolution:
    sel = (context.selected_text or "").strip()
    if not sel or anchor_selection is None or anchor_selection.start == anchor_selection.end:
        return SelectionIntentResolution(kind="dictate", source="no_selection_anchor")

    if mode_policy is not None and not mode_policy.allow_selection_commands:
        return SelectionIntentResolution(kind="dictate", source="mode_disallows_selection_commands")

    spoken = (spoken_final or "").strip()
    if not spoken:
        return SelectionIntentResolution(kind="dictate", source="empty_spoken")
    spoken_words = spoken.split()
    bounded_command = recognize_selection_transform_command(spoken)
    if bounded_command is not None:
        recorder.record(
            TraceKind.WRITER,
            "selection_intent_gate",
            {
                "utterance_id": utterance_id,
                "kind": "edit",
                "source": "bounded_command",
                "transform_id": bounded_command.transform_id,
                "confidence": 1.0,
            },
        )
        return SelectionIntentResolution(
            kind="edit",
            instruction=bounded_command.instruction,
            confidence=1.0,
            source="bounded_command",
        )
    command_overwrite_risk = looks_like_selection_edit_command(spoken)
    has_selection_target = re.search(
        r"\b(?:this|it|selection|selected|highlighted)\b|"
        r"\b(?:selected|highlighted|this|that)\s+text\b",
        spoken,
        re.I,
    ) is not None

    if len(spoken_words) > LongDictationWordFloor and not has_selection_target:
        recorder.record(
            TraceKind.WRITER,
            "selection_intent_gate",
            {
                "utterance_id": utterance_id,
                "kind": "dictate",
                "source": "long_spoken_no_selection_target",
                "confidence": 0.86,
            },
        )
        return SelectionIntentResolution(kind="dictate", confidence=0.86, source="long_spoken_no_selection_target")

    classify = getattr(backend, "classify_dictation_vs_edit_selection", None)
    if callable(classify):
        excerpt = sel[:400]
        try:
            raw = classify(spoken=spoken, selection_excerpt=excerpt)
        except Exception as exc:  # noqa: BLE001
            recorder.record(
                TraceKind.WRITER,
                "selection_intent_gate_error",
                {"utterance_id": utterance_id, "error": str(exc)},
            )
            return SelectionIntentResolution(
                kind="ambiguous" if command_overwrite_risk else "dictate",
                instruction=spoken if command_overwrite_risk else None,
                source="classifier_error",
            )

        if not isinstance(raw, dict):
            return SelectionIntentResolution(
                kind="ambiguous" if command_overwrite_risk else "dictate",
                instruction=spoken if command_overwrite_risk else None,
                source="classifier_bad_shape",
            )

        intent = str(raw.get("intent") or "").strip().lower()
        try:
            conf = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        instr = str(raw.get("instruction") or spoken).strip()

        recorder.record(
            TraceKind.WRITER,
            "selection_intent_gate",
            {
                "utterance_id": utterance_id,
                "kind": intent,
                "confidence": conf,
                "source": "mlx_classifier",
            },
        )

        if intent == "edit" and conf >= ConfidenceEdit and instr:
            return SelectionIntentResolution(kind="edit", instruction=instr, confidence=conf, source="mlx_classifier")
        if intent == "dictate" and conf >= ConfidenceDictate:
            return SelectionIntentResolution(kind="dictate", confidence=conf, source="mlx_classifier")
        if intent == "edit" and instr and conf >= ConfidenceEdit - AmbiguousBand:
            return SelectionIntentResolution(kind="ambiguous", instruction=instr, confidence=conf, source="mlx_classifier")

        return SelectionIntentResolution(kind="ambiguous", confidence=conf, source="mlx_classifier")

    resolution = "ambiguous" if command_overwrite_risk else "dictate"
    recorder.record(
        TraceKind.WRITER,
        "selection_intent_gate",
        {
            "utterance_id": utterance_id,
            "kind": resolution,
            "source": "no_classifier",
            "selection_overwrite_risk": command_overwrite_risk,
        },
    )
    return SelectionIntentResolution(
        kind=resolution,
        instruction=spoken if command_overwrite_risk else None,
        source="no_classifier",
    )
