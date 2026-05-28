from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from juno_v2.asr.utterance_buffer import UtteranceAudioBuffer
from juno_v2.audio.ring_buffer import AudioRingBuffer
from juno_v2.contracts.audio import AudioFrame
from juno_v2.contracts.final import FinalDecodeRequest, FinalSessionSummary, FinalTranscript
from juno_v2.contracts.speech import SpeechEventKind
from juno_v2.contracts.tracing import TraceKind
from juno_v2.final.backends.base import FinalAsrBackend
from juno_v2.final.config import FinalAsrConfig
from juno_v2.final.sinks import FinalTranscriptSink
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.speech.config import SpeechStateConfig
from juno_v2.speech.state_machine import SpeechStateMachine
from juno_v2.vad.probes import DualVadPolicy


@dataclass(slots=True)
class FinalAsrSessionRunner:
    state_config: SpeechStateConfig
    final_config: FinalAsrConfig
    vad_policy: DualVadPolicy
    backend: FinalAsrBackend
    recorder: TraceRecorder
    sink: FinalTranscriptSink

    def run(self, frames: Iterable[AudioFrame]) -> FinalSessionSummary:
        machine = SpeechStateMachine(self.state_config)
        ring = AudioRingBuffer(max_frames=self.state_config.ring_buffer_frames())
        active: UtteranceAudioBuffer | None = None
        decode_count = 0
        emitted_count = 0
        total_decode_ms = 0.0
        total_eot_ms = 0.0
        total_audio_ms = 0.0

        self.backend.warm()
        self.recorder.record(
            TraceKind.ASR_FINAL,
            "final_session_started",
            {
                "frame_ms": self.state_config.frame_ms,
                "backend": self.backend.backend_name,
            },
        )

        for frame in frames:
            total_audio_ms = frame.end_ms
            ring.append(frame)
            decision = self.vad_policy.decide(frame)
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
                    "ring_buffer_ms": ring.duration_ms(),
                },
            )
            events = machine.process(frame, decision)
            for event in events:
                self.recorder.record(TraceKind.SYSTEM, event.kind.value, event.to_dict())
                if event.kind == SpeechEventKind.SPEECH_STARTED and event.utterance_id is not None:
                    active = UtteranceAudioBuffer(utterance_id=event.utterance_id)
                    active.seed(ring.snapshot()[-self.state_config.pre_roll_frames :])
                    self.sink.clear()
                elif event.kind == SpeechEventKind.SESSION_ABORTED:
                    active = None
                    self.sink.clear()
                elif event.kind == SpeechEventKind.SPEECH_ENDED and active is not None:
                    active.append(frame)
                    transcript = self._decode(active)
                    decode_count += 1
                    total_decode_ms += transcript.decode_ms
                    total_eot_ms += transcript.end_of_turn_latency_ms
                    emitted_count += 1
                    self.sink.emit(transcript)
                    active = None

            if active is not None:
                active.append(frame)

        summary = FinalSessionSummary(
            session_id=self.recorder.session_id,
            utterance_count=machine.utterance_count,
            decode_count=decode_count,
            emitted_count=emitted_count,
            total_audio_ms=total_audio_ms,
            average_decode_ms=(total_decode_ms / decode_count) if decode_count else 0.0,
            average_end_of_turn_latency_ms=(total_eot_ms / decode_count) if decode_count else 0.0,
            metadata={"backend": self.backend.backend_name},
        )
        self.recorder.record(TraceKind.ASR_FINAL, "final_session_completed", summary.to_dict())
        return summary

    def _decode(self, active: UtteranceAudioBuffer) -> FinalTranscript:
        req = FinalDecodeRequest(
            utterance_id=active.utterance_id,
            audio=active.audio(),
            sample_rate_hz=self.state_config.sample_rate_hz,
            start_ms=active.start_ms,
            end_ms=active.end_ms,
            language=self.final_config.language,
            initial_prompt=self.final_config.initial_prompt,
        )
        self.recorder.record(
            TraceKind.ASR_FINAL,
            "final_decode_started",
            {
                "utterance_id": req.utterance_id,
                "start_ms": req.start_ms,
                "end_ms": req.end_ms,
                "audio_duration_ms": req.audio_duration_ms,
            },
        )
        result = self.backend.decode(req)
        self.recorder.record(TraceKind.ASR_FINAL, "final_decode_completed", result.to_dict())
        transcript = FinalTranscript(
            utterance_id=result.utterance_id,
            text=result.text.strip(),
            start_ms=result.start_ms,
            end_ms=result.end_ms,
            backend_name=result.backend_name,
            model_path=getattr(result, "model_path", "") or "",
            language=result.language,
            decode_ms=result.decode_ms,
            end_of_turn_latency_ms=result.end_of_turn_latency_ms,
            metadata={
                **result.metadata,
                "segment_count": len(result.segments),
                # Per-segment audio-side signals required by the commit-side
                # trailing-silence-hallucination guard. Stored as a tuple of
                # dicts (rather than the FinalSegment dataclass) to keep
                # ``final_metadata`` JSON-serialisable for tracing.
                "segments": tuple(seg.to_dict() for seg in result.segments),
            },
        )
        self.recorder.record(TraceKind.ASR_FINAL, "final_transcript_emitted", transcript.to_dict())
        return transcript
