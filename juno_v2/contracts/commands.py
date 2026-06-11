from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias, TypedDict


class CommandTargetClass(str, Enum):
    ACTIVE_UTTERANCE = "active_utterance"
    RECENT_COMMIT = "recent_commit"
    SELECTED_TEXT = "selected_text"
    EXPLICIT_SPAN = "explicit_span"
    # Last paragraph of the focused field's text before the caret — the
    # fallback target when no selection and no tracked commit exist.
    FOCUSED_TEXT = "focused_text"
    NONE = "none"


@dataclass(slots=True)
class SemanticCommandIntent:
    intent_name: str
    target_class: CommandTargetClass
    target_confidence: float
    rewrite_instruction: str
    requires_confirmation: bool
    ambiguity_reason: str | None

    def to_dict(self) -> SemanticCommandIntentPayload:
        return {
            "intent_name": self.intent_name,
            "target_class": self.target_class.value,
            "target_confidence": self.target_confidence,
            "rewrite_instruction": self.rewrite_instruction,
            "requires_confirmation": self.requires_confirmation,
            "ambiguity_reason": self.ambiguity_reason,
        }


DeterministicCommandKind = Literal[
    "discard_utterance",
    "undo_last",
    "delete_last",
    "delete_words",
    "delete_sentence",
    "insert",
    "structure",
    "quote",
    "recent_edit",
    "selected_edit",
    "translate",
    "replace",
]


CommandPayloadValue: TypeAlias = str | int
CommandPayload: TypeAlias = dict[str, CommandPayloadValue]


class SemanticCommandIntentPayload(TypedDict):
    intent_name: str
    target_class: str
    target_confidence: float
    rewrite_instruction: str
    requires_confirmation: bool
    ambiguity_reason: str | None


class DeterministicCommandPayload(TypedDict):
    name: str
    kind: DeterministicCommandKind
    payload: CommandPayload


@dataclass(slots=True)
class DeterministicCommand:
    name: str
    kind: DeterministicCommandKind
    payload: CommandPayload

    def to_dict(self) -> DeterministicCommandPayload:
        return {
            "name": self.name,
            "kind": self.kind,
            "payload": dict(self.payload),
        }


CommandPhase = Literal["II"]
