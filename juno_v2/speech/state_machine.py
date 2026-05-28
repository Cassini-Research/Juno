from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from juno_v2.contracts.audio import AudioFrame
from juno_v2.contracts.speech import SpeechEvent, SpeechEventKind, SpeechPhase, SpeechSessionSummary, VadDecision
from juno_v2.contracts.tracing import new_trace_id
from juno_v2.speech.config import SpeechStateConfig


@dataclass(slots=True)
class SpeechStateMachine:
    config: SpeechStateConfig
    phase: SpeechPhase = SpeechPhase.SILENT
    current_utterance_id: str | None = None
    speech_run_frames: int = 0
    silence_run_frames: int = 0
    total_frames: int = 0
    speech_frame_count: int = 0
    silence_frame_count: int = 0
    utterance_count: int = 0
    _events: List[SpeechEvent] = field(default_factory=list)

    def _emit(self, event: SpeechEvent) -> None:
        self._events.append(event)

    def _next_utterance_id(self) -> str:
        self.utterance_count += 1
        return f"utt_{new_trace_id()[:12]}"

    def process(self, frame: AudioFrame, decision: VadDecision) -> List[SpeechEvent]:
        self._events = []
        self.total_frames += 1
        # Asymmetric counters: `decision` (liberal, may include energy) drives
        # speech-side transitions; `is_silent` (strict — silero+webrtc both
        # negative) drives silence-side transitions. Frames that are neither
        # (e.g. ambiguous noise) hold the state without advancing either
        # counter, so noisy rooms no longer prevent end-of-turn detection.
        if decision.decision:
            self.speech_frame_count += 1
            self.speech_run_frames += 1
            self.silence_run_frames = 0
        elif decision.is_silent:
            self.silence_frame_count += 1
            self.silence_run_frames += 1
            self.speech_run_frames = 0

        self._emit(
            SpeechEvent(
                kind=SpeechEventKind.FRAME_PROCESSED,
                phase=self.phase,
                frame_index=frame.index,
                start_ms=frame.start_ms,
                end_ms=frame.end_ms,
                utterance_id=self.current_utterance_id,
                payload={"vad": decision.to_dict()},
            )
        )

        if self.phase == SpeechPhase.SILENT:
            if decision.decision:
                self.phase = SpeechPhase.MAYBE_SPEECH
                self.current_utterance_id = self._next_utterance_id()
                self._emit(
                    SpeechEvent(
                        kind=SpeechEventKind.SPEECH_STARTED,
                        phase=self.phase,
                        frame_index=frame.index,
                        start_ms=frame.start_ms,
                        end_ms=frame.end_ms,
                        utterance_id=self.current_utterance_id,
                        reason="speech_gate_opened",
                    )
                )
        elif self.phase == SpeechPhase.MAYBE_SPEECH:
            if decision.decision and self.speech_run_frames >= self.config.start_trigger_frames:
                self.phase = SpeechPhase.IN_SPEECH
                self._emit(
                    SpeechEvent(
                        kind=SpeechEventKind.SPEECH_CONFIRMED,
                        phase=self.phase,
                        frame_index=frame.index,
                        start_ms=frame.start_ms,
                        end_ms=frame.end_ms,
                        utterance_id=self.current_utterance_id,
                        reason="start_trigger_reached",
                    )
                )
            elif not decision.decision and self.silence_run_frames >= self.config.pause_trigger_frames:
                # Reject weak start.
                self.phase = SpeechPhase.SILENT
                self._emit(
                    SpeechEvent(
                        kind=SpeechEventKind.SESSION_ABORTED,
                        phase=self.phase,
                        frame_index=frame.index,
                        start_ms=frame.start_ms,
                        end_ms=frame.end_ms,
                        utterance_id=self.current_utterance_id,
                        reason="weak_start_rejected",
                    )
                )
                self.current_utterance_id = None
        elif self.phase == SpeechPhase.IN_SPEECH:
            if not decision.decision and self.silence_run_frames >= self.config.pause_trigger_frames:
                self.phase = SpeechPhase.PAUSED
                self._emit(
                    SpeechEvent(
                        kind=SpeechEventKind.SPEECH_PAUSED,
                        phase=self.phase,
                        frame_index=frame.index,
                        start_ms=frame.start_ms,
                        end_ms=frame.end_ms,
                        utterance_id=self.current_utterance_id,
                        reason="pause_trigger_reached",
                    )
                )
        elif self.phase == SpeechPhase.PAUSED:
            if decision.decision:
                self.phase = SpeechPhase.IN_SPEECH
                self._emit(
                    SpeechEvent(
                        kind=SpeechEventKind.SPEECH_RESUMED,
                        phase=self.phase,
                        frame_index=frame.index,
                        start_ms=frame.start_ms,
                        end_ms=frame.end_ms,
                        utterance_id=self.current_utterance_id,
                        reason="speech_resumed",
                    )
                )
            elif self.silence_run_frames >= self.config.end_trigger_frames:
                self.phase = SpeechPhase.ENDED
                self._emit(
                    SpeechEvent(
                        kind=SpeechEventKind.SPEECH_ENDED,
                        phase=self.phase,
                        frame_index=frame.index,
                        start_ms=frame.start_ms,
                        end_ms=frame.end_ms,
                        utterance_id=self.current_utterance_id,
                        reason="end_trigger_reached",
                    )
                )
                self.phase = SpeechPhase.SILENT
                self.current_utterance_id = None
        return list(self._events)

    def summary(self, session_id: str, total_audio_ms: float) -> SpeechSessionSummary:
        return SpeechSessionSummary(
            session_id=session_id,
            total_frames=self.total_frames,
            total_audio_ms=total_audio_ms,
            utterance_count=self.utterance_count,
            speech_frame_count=self.speech_frame_count,
            silence_frame_count=self.silence_frame_count,
            phase=self.phase,
            completed=True,
        )
