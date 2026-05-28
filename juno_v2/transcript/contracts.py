from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PatchOpType = Literal["replace", "insert", "delete", "punctuate", "case"]
PatchReason = Literal[
    "asr_correction",
    "memory_alias",
    "user_replacement",
    "screen_term",
    "file_or_symbol",
    "self_correction",
    "spoken_punctuation",
    "itn",
    "capitalization",
    "spacing",
]


@dataclass(slots=True)
class TranscriptPatchOp:
    op: PatchOpType
    start_char: int
    end_char: int
    text: str
    reason: PatchReason
    confidence: float
    source_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptAdjudicationResult:
    utterance_id: str
    stage: Literal["live", "final"]
    corrected_text: str
    ops: tuple[TranscriptPatchOp, ...]
    confidence: float
    base_visible_revision: int | None
    base_text_hash: str | None
    stable_prefix_chars: int | None
    protected_terms_used: tuple[str, ...]
    rejected: bool = False
    rejected_reason: str | None = None
    backend_name: str | None = None
    decode_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Snapshot text the engine adjudicated against. Shells use this to
    # reconcile patches against drift: the live preview keeps appending while
    # Qwen is decoding, so by the time the patch arrives the
    # shell's visible text has grown past the snapshot. The shell accepts
    # the patch when current_text == base_visible_text OR starts with it
    # (append-only growth, op positions still valid in the stable prefix).
    base_visible_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "transcript_patch_v1",
            "utterance_id": self.utterance_id,
            "stage": self.stage,
            "corrected_text": self.corrected_text,
            "ops": [op.to_dict() for op in self.ops],
            "confidence": self.confidence,
            "base_visible_revision": self.base_visible_revision,
            "base_text_hash": self.base_text_hash,
            "base_visible_text": self.base_visible_text,
            "stable_prefix_chars": self.stable_prefix_chars,
            "protected_terms_used": list(self.protected_terms_used),
            "rejected": self.rejected,
            "rejected_reason": self.rejected_reason,
            "backend_name": self.backend_name,
            "decode_ms": self.decode_ms,
            "metadata": dict(self.metadata),
        }
