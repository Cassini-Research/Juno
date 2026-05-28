from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict


class SpeechPhase(str, Enum):
    SILENT = "silent"
    MAYBE_SPEECH = "maybe_speech"
    IN_SPEECH = "in_speech"
    PAUSED = "paused"
    ENDED = "ended"


class SpeechEventKind(str, Enum):
    SESSION_STARTED = "session_started"
    FRAME_PROCESSED = "frame_processed"
    SPEECH_STARTED = "speech_started"
    SPEECH_CONFIRMED = "speech_confirmed"
    SPEECH_PAUSED = "speech_paused"
    SPEECH_RESUMED = "speech_resumed"
    SPEECH_ENDED = "speech_ended"
    SESSION_COMPLETED = "session_completed"
    SESSION_ABORTED = "session_aborted"


@dataclass(slots=True)
class VadDecision:
    frame_index: int
    start_ms: float
    end_ms: float
    webrtc_speech: bool
    silero_speech: bool
    energy_speech: bool
    energy_rms: float
    decision: bool
    # Asymmetric silence gate: true only when both speech-trained detectors
    # (silero + webrtc) agree this frame is not speech. Advances the
    # silence-run counter that drives pause/end triggers, so noisy energy
    # alone cannot keep an utterance alive past a genuine stop.
    is_silent: bool = False
    silero_score: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpeechEvent:
    kind: SpeechEventKind
    phase: SpeechPhase
    frame_index: int
    start_ms: float
    end_ms: float
    utterance_id: str | None = None
    reason: str | None = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["phase"] = self.phase.value
        return data


@dataclass(slots=True)
class SpeechSessionSummary:
    session_id: str
    total_frames: int
    total_audio_ms: float
    utterance_count: int
    speech_frame_count: int
    silence_frame_count: int
    phase: SpeechPhase
    completed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["phase"] = self.phase.value
        return data
