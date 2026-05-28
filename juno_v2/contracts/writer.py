from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict

from juno_v2.contracts.workbench import ClientSelection, CommitMode


class WriterMode(str, Enum):
    DEFAULT_SURFACE = "default_surface"
    FORMAL_EMAIL = "formal_email"
    CASUAL_CHAT = "casual_chat"
    VERBATIM = "verbatim"
    TECHNICAL_PRECISE = "technical_precise"
    STRUCTURED_NOTES = "structured_notes"
    EXPLICIT_REWRITE = "explicit_rewrite"
    COMMAND_MODE = "command_mode"


class WriterActionKind(str, Enum):
    PASS_THROUGH_COMMIT = "pass_through_commit"
    DIRECT_COMMIT = "direct_commit"
    TRANSFORM_COMMIT = "transform_commit"
    MODE_SWITCH = "mode_switch"
    STATE_MUTATION = "state_mutation"
    MEMORY_MUTATION = "memory_mutation"
    NOOP = "noop"


class WriterIntentKind(str, Enum):
    DICTATE = "dictate"
    INSERT_FORMATTING = "insert_formatting"
    SET_STRUCTURE_MODE = "set_structure_mode"
    DETERMINISTIC_TRANSFORM = "deterministic_transform"
    MODEL_TRANSFORM = "model_transform"
    RECENT_DETERMINISTIC_TRANSFORM = "recent_deterministic_transform"
    RECENT_MODEL_TRANSFORM = "recent_model_transform"
    SWITCH_MODE = "switch_mode"
    ADD_TERM = "add_term"
    ADD_REPLACEMENT = "add_replacement"
    COMMAND_RESULT = "command_result"
    ACTIVE_UTTERANCE_EDIT = "active_utterance_edit"
    INSERT_CONTEXT_FIELD = "insert_context_field"
    NOOP = "noop"


@dataclass(slots=True)
class WriterIntent:
    kind: WriterIntentKind
    raw_text: str
    instruction: str | None = None
    insert_text: str = ""
    transform_kind: str | None = None
    mode: WriterMode | None = None
    structure_mode: str | None = None
    term: str | None = None
    trigger: str | None = None
    replacement: str | None = None
    context_field: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["mode"] = self.mode.value if self.mode else None
        return data


@dataclass(slots=True)
class WriterTransformRequest:
    utterance_id: str
    instruction: str
    source_text: str
    mode: WriterMode
    target_selection: ClientSelection | None = None
    context_payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        return data


@dataclass(slots=True)
class WriterTransformResult:
    utterance_id: str
    text: str
    backend_name: str
    decode_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WriterOutcome:
    utterance_id: str
    action: WriterActionKind
    output_text: str = ""
    commit_mode: CommitMode | None = None
    selection_override: ClientSelection | None = None
    learn_from_commit: bool = False
    writer_mode: WriterMode | None = None
    structure_mode: str | None = None
    deterministic_used: bool = False
    model_used: bool = False
    memory_updated: bool = False
    # Mode trace (voice-writing): effective selection vs legacy writer_mode enum
    effective_mode: str | None = None
    mode_source: str | None = None
    base_mode: str | None = None
    manual_mode_name: str | None = None
    custom_mode_name: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        data["commit_mode"] = self.commit_mode.value if self.commit_mode else None
        data["writer_mode"] = self.writer_mode.value if self.writer_mode else None
        return data
