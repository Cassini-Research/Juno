from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from juno_v2.audio.ring_buffer import AudioRingBuffer
from juno_v2.contracts.audio import AudioFrame
from juno_v2.contracts.speech import SpeechEvent, SpeechEventKind, SpeechSessionSummary
from juno_v2.contracts.tracing import TraceKind
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.speech.config import SpeechStateConfig, preprocess_audio_frame
from juno_v2.speech.state_machine import SpeechStateMachine
from juno_v2.vad.probes import DualVadPolicy


@dataclass(slots=True)
class SpeechSessionRunner:
    state_config: SpeechStateConfig
    vad_policy: DualVadPolicy
    recorder: TraceRecorder
    session_id: str

    @classmethod
    def create(cls, state_config: SpeechStateConfig, vad_policy: DualVadPolicy, recorder: TraceRecorder) -> "SpeechSessionRunner":
        return cls(state_config=state_config, vad_policy=vad_policy, recorder=recorder, session_id=recorder.session_id)

    def run(self, frames: Iterable[AudioFrame]) -> SpeechSessionSummary:
        machine = SpeechStateMachine(self.state_config)
        ring = AudioRingBuffer(max_frames=self.state_config.ring_buffer_frames())
        total_audio_ms = 0.0
        self.recorder.record(
            TraceKind.SYSTEM,
            "speech_session_started",
            {
                "session_id": self.session_id,
                "frame_ms": self.state_config.frame_ms,
                "sample_rate_hz": self.state_config.sample_rate_hz,
                "ring_buffer_frames": self.state_config.ring_buffer_frames(),
                "speech_profile": self.state_config.profile_name,
                "input_gain_db": self.state_config.input_gain_db,
            },
        )
        for frame in frames:
            prepared_frame = preprocess_audio_frame(frame, self.state_config)
            total_audio_ms = prepared_frame.end_ms
            ring.append(prepared_frame)
            decision = self.vad_policy.decide(prepared_frame)
            self.recorder.record(
                TraceKind.SYSTEM,
                "vad_frame_decision",
                {
                    "frame_index": frame.index,
                    "start_ms": frame.start_ms,
                    "end_ms": frame.end_ms,
                    "decision": decision.decision,
                    "webrtc_speech": decision.webrtc_speech,
                    "silero_speech": decision.silero_speech,
                    "energy_speech": decision.energy_speech,
                    "energy_rms": decision.energy_rms,
                    "silero_score": decision.silero_score,
                    "ring_buffer_size": len(ring),
                    "ring_buffer_ms": ring.duration_ms(),
                },
            )
            events = machine.process(frame, decision)
            for event in events:
                self._record_event(event)
        summary = machine.summary(session_id=self.session_id, total_audio_ms=total_audio_ms)
        self.recorder.record(TraceKind.SYSTEM, "speech_session_completed", summary.to_dict())
        return summary

    def _record_event(self, event: SpeechEvent) -> None:
        kind = TraceKind.SYSTEM if event.kind != SpeechEventKind.FRAME_PROCESSED else TraceKind.WORKBENCH
        payload = event.to_dict()
        self.recorder.record(kind, event.kind.value, payload)
