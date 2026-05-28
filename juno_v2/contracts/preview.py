from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from juno_v2.contracts.audio import AudioSamples


@dataclass(slots=True)
class PreviewDecodeRequest:
    utterance_id: str
    audio: AudioSamples
    sample_rate_hz: int
    start_ms: float
    end_ms: float
    is_final: bool = False
    language: str | None = None
    allowed_languages: list[str] = field(default_factory=list)
    language_policy: str | None = None
    initial_prompt: str | None = None
    decode_seq: int = 0
    reset_decoder_state: bool = False
    bias_phrases: list[str] = field(default_factory=list)
    context_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def audio_duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass(slots=True)
class PreviewDecodeResult:
    utterance_id: str
    text: str
    start_ms: float
    end_ms: float
    audio_duration_ms: float
    is_final: bool
    backend_name: str
    language: str | None = None
    decode_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreviewEmission:
    utterance_id: str
    text: str
    start_ms: float
    end_ms: float
    is_final: bool
    backend_name: str
    language: str | None = None
    decode_ms: float = 0.0
    stability_delta_chars: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreviewSessionSummary:
    session_id: str
    utterance_count: int
    preview_decode_count: int
    partial_emit_count: int
    final_emit_count: int
    duplicate_partial_count: int
    stability_delta_chars_total: int
    total_audio_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
