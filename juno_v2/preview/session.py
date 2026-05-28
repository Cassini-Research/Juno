from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from juno_v2.audio.ring_buffer import AudioRingBuffer
from juno_v2.contracts.audio import AudioFrame
from juno_v2.contracts.preview import PreviewDecodeRequest, PreviewEmission, PreviewSessionSummary
from juno_v2.contracts.speech import SpeechEventKind, SpeechPhase
from juno_v2.contracts.tracing import TraceKind
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.preview.backends.base import PreviewAsrBackend
from juno_v2.preview.buffer import UtteranceAudioBuffer
from juno_v2.preview.config import PreviewAsrConfig
from juno_v2.preview.sinks import PreviewSink
from juno_v2.speech.config import SpeechStateConfig
from juno_v2.speech.state_machine import SpeechStateMachine
from juno_v2.vad.probes import DualVadPolicy


@dataclass(slots=True)
class PreviewSessionRunner:
    state_config: SpeechStateConfig
    preview_config: PreviewAsrConfig
    vad_policy: DualVadPolicy
    backend: PreviewAsrBackend
    recorder: TraceRecorder
    sink: PreviewSink

    def run(self, frames: Iterable[AudioFrame]) -> PreviewSessionSummary:
        machine = SpeechStateMachine(self.state_config)
        ring = AudioRingBuffer(max_frames=self.state_config.ring_buffer_frames())
        active: UtteranceAudioBuffer | None = None
        preview_decode_count = 0
        partial_emit_count = 0
        final_emit_count = 0
        duplicate_partial_count = 0
        stability_delta_chars_total = 0
        total_audio_ms = 0.0

        self.backend.warm()
        self.recorder.record(
            TraceKind.ASR_PREVIEW,
            "preview_session_started",
            {
                "frame_ms": self.state_config.frame_ms,
                "partial_decode_interval_ms": self.preview_config.partial_decode_interval_ms,
                "min_decode_audio_ms": self.preview_config.min_decode_audio_ms,
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
                self.recorder.record(
                    TraceKind.SYSTEM,
                    event.kind.value,
                    event.to_dict(),
                )
                if event.kind == SpeechEventKind.SPEECH_STARTED and event.utterance_id is not None:
                    active = UtteranceAudioBuffer(utterance_id=event.utterance_id)
                    active.seed(ring.snapshot()[-self.state_config.pre_roll_frames :])
                elif event.kind == SpeechEventKind.SESSION_ABORTED:
                    active = None
                    self.sink.clear()
                elif event.kind == SpeechEventKind.SPEECH_ENDED and active is not None:
                    emission = self._decode(active, is_final=True)
                    preview_decode_count += 1
                    if emission is not None:
                        final_emit_count += 1
                        stability_delta_chars_total += emission.stability_delta_chars
                        self.sink.emit(emission)
                    active = None

            if active is not None:
                active.append(frame)
                if machine.phase in {SpeechPhase.IN_SPEECH, SpeechPhase.PAUSED} and active.should_decode_partial(
                    self.preview_config.partial_decode_interval_ms,
                    self.preview_config.min_decode_audio_ms,
                ):
                    emission = self._decode(active, is_final=False)
                    preview_decode_count += 1
                    active.mark_partial_decode()
                    if emission is not None:
                        if emission.stability_delta_chars == 0 and emission.text:
                            duplicate_partial_count += 1
                        else:
                            partial_emit_count += 1
                            stability_delta_chars_total += emission.stability_delta_chars
                            self.sink.emit(emission)

        summary = PreviewSessionSummary(
            session_id=self.recorder.session_id,
            utterance_count=machine.utterance_count,
            preview_decode_count=preview_decode_count,
            partial_emit_count=partial_emit_count,
            final_emit_count=final_emit_count,
            duplicate_partial_count=duplicate_partial_count,
            stability_delta_chars_total=stability_delta_chars_total,
            total_audio_ms=total_audio_ms,
            metadata={"backend": self.backend.backend_name},
        )
        self.recorder.record(TraceKind.ASR_PREVIEW, "preview_session_completed", summary.to_dict())
        return summary

    def _decode(self, active: UtteranceAudioBuffer, *, is_final: bool) -> PreviewEmission | None:
        req = PreviewDecodeRequest(
            utterance_id=active.utterance_id,
            audio=active.audio(),
            sample_rate_hz=self.state_config.sample_rate_hz,
            start_ms=active.start_ms,
            end_ms=active.end_ms,
            is_final=is_final,
            language=self.preview_config.language,
            initial_prompt=self.preview_config.initial_prompt,
        )
        self.recorder.record(
            TraceKind.ASR_PREVIEW,
            "preview_decode_started",
            {
                "utterance_id": req.utterance_id,
                "start_ms": req.start_ms,
                "end_ms": req.end_ms,
                "audio_duration_ms": req.audio_duration_ms,
                "is_final": is_final,
            },
        )
        result = self.backend.decode(req)
        text = result.text.strip()
        self.recorder.record(
            TraceKind.ASR_PREVIEW,
            "preview_decode_completed",
            result.to_dict(),
        )
        if not text:
            return None
        stability_delta = active.update_last_emitted_text(text)
        emission = PreviewEmission(
            utterance_id=result.utterance_id,
            text=text,
            start_ms=result.start_ms,
            end_ms=result.end_ms,
            is_final=result.is_final,
            backend_name=result.backend_name,
            language=result.language,
            decode_ms=result.decode_ms,
            stability_delta_chars=stability_delta,
            metadata=result.metadata,
        )
        self.recorder.record(
            TraceKind.ASR_PREVIEW,
            "preview_emitted",
            emission.to_dict(),
        )
        return emission
