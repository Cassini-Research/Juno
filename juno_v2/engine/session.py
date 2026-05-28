from __future__ import annotations

import os
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable


# Live-dictation latency logging. When JUNO_V2_LOG_LATENCY is set to
# a truthy value (1 / true / yes), every committed utterance emits a
# one-line human-readable latency summary to stderr right after the
# commit event lands on its metric object. Default is off so batch
# runs stay quiet. Scoped here at module load because the relevant
# env var should be set before the service boots and should not
# change mid-run. The flag controls format (1) from the Option B
# discussion: rich single line per commit, scoped behind env var,
# cheap to scan while watching the dictation window.
_LIVE_LATENCY_LOG_ENABLED = os.environ.get("JUNO_V2_LOG_LATENCY", "").strip().lower() in {"1", "true", "yes", "on"}


def _format_latency_ms(value: float | None) -> str:
    """Render a latency field for the stderr log line. None → 'n/a'
    so the output shape stays constant even when a metric is missing
    (e.g. TTFT is unmeasured when a short utterance finalizes before
    any partial decode emission)."""
    if value is None:
        return "n/a"
    return f"{value:.0f}ms"


def _emit_live_latency_line(
    metrics: UtteranceRuntimeMetrics,
    *,
    committed_text: str | None,
    preview_backend: str,
    final_backend: str,
    repetition_collapse: dict | None = None,
) -> None:
    """Print a single human-readable latency summary to stderr.

    Called once per committed utterance from inside the writer-commit
    flush, at the exact moment speech_end_to_commit_ms is populated on
    the metric object. No-ops when JUNO_V2_LOG_LATENCY is unset so
    the cost is a module-level boolean check per commit.

    Output format::

        [juno-latency] ttft=286ms final=950ms commit=953ms preview=streaming_local_http_json final_backend=mlx_whisper text="Hello." utt=010d534be6d9

    When the repetition collapser fired on this utterance, an extra
    ``collapsed=<removed_words>w`` field is appended so the operator
    can see mid-dictation when the anti-repetition post-filter saved
    them from a runaway hallucination:

        [juno-latency] ttft=286ms ... text='Some can be saved.' collapsed=8w utt=...

    The preview+final backend names are included so a user running
    multiple stacks can tell which combination produced each number
    at a glance. The committed text is truncated to 60 characters.
    """
    if not _LIVE_LATENCY_LOG_ENABLED:
        return
    text = (committed_text or "").strip().replace("\n", " ")
    if len(text) > 60:
        text = text[:57] + "..."
    parts = [
        "[juno-latency]",
        f"ttft={_format_latency_ms(metrics.ttft_ms)}",
        f"final={_format_latency_ms(metrics.speech_end_to_final_ms)}",
        f"commit={_format_latency_ms(metrics.speech_end_to_commit_ms)}",
        f"preview={preview_backend}",
        f"final_backend={final_backend}",
        f"text={text!r}",
    ]
    if repetition_collapse and repetition_collapse.get("collapsed"):
        removed = repetition_collapse.get("removed_words", 0)
        parts.append(f"collapsed={removed}w")
    parts.append(f"utt={metrics.utterance_id}")
    print(" ".join(parts), file=sys.stderr, flush=True)

from juno_v2.asr.utterance_buffer import UtteranceAudioBuffer
from juno_v2.audio.ring_buffer import AudioRingBuffer
from juno_v2.commit.controller import CommitController
from juno_v2.context.provider import ContextProvider
from juno_v2.contracts.audio import AudioFrame, AudioSamples
from juno_v2.contracts.commit import CommitDecision
from juno_v2.context.clipboard_enrichment import inject_clipboard_ring
from juno_core_v3.context.clipboard_ring import ClipboardRingBuffer
from juno_v2.context.session_memory import SessionContextMemory
from juno_v2.contracts.context import RecognitionBiasPlan, TypedContextBundle
from juno_v2.contracts.final import FinalDecodeRequest, FinalTranscript
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.preview import PreviewDecodeRequest, PreviewEmission
from juno_v2.contracts.speech import SpeechEventKind, SpeechPhase
from juno_v2.contracts.tracing import TraceKind
from juno_v2.final.backends.base import FinalAsrBackend
from juno_v2.final.config import FinalAsrConfig
from juno_v2.memory.bias import RecognitionBiasEngine
from juno_v2.memory.repetition import collapse_tail_repetition
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.memory.term_policy import learned_term_allowed
from juno_v2.contracts.modes import ModePolicy, ModeSelection
from juno_v2.modes.store import CustomModeStore, default_modes_data_path
from juno_v2.presets.surface_presets import (
    SurfacePresetStore,
    build_surface_context_line,
    default_surface_presets_path,
    resolve_mode_with_surface_presets,
)
from juno_v2.memory.ranking import rank_memory_for_context
from juno_v2.personalization.seed.runtime import JunoSeedPersonalizationRuntime
from juno_v2.personalization.seed.suppressed import is_durable_memory_suppressed
from juno_v2.contracts.writer import WriterActionKind, WriterOutcome
from juno_v2.itn.engine import ITNEngine
from juno_v2.itn.format_policy import resolve_itn_format_policy
from juno_v2.language.normalize import LanguageAwareNormalizer
from juno_core_v3.context.plane import ContextPlane, ContextPlaneConfig
from juno_v2.language.policy import LanguagePlanner
from juno_v2.writer.service import WriterService
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.preview.backends.base import PreviewAsrBackend
from juno_v2.preview.config import PreviewAsrConfig
from juno_v2.preview.personalization_repair import preview_personalization_terms_from_plan
from juno_v2.speech.config import SpeechStateConfig, preprocess_audio_frame
from juno_v2.speech.state_machine import SpeechStateMachine
from juno_v2.vad.probes import DualVadPolicy
from juno_v2.runtime.execution import StageExecutor, StageTask
from juno_v2.runtime.lifecycle import BackendLifecycleManager
from juno_v2.runtime.truth import build_runtime_truth_report


@dataclass(slots=True)
class UtteranceRuntimeMetrics:
    utterance_id: str
    ttft_ms: float | None = None
    first_preview_decode_ms: float | None = None
    first_preview_stability_delta_chars: int | None = None
    first_preview_queue_wait_ms: float | None = None
    first_preview_worker_service_ms: float | None = None
    speech_end_to_final_ms: float | None = None
    speech_end_to_commit_ms: float | None = None
    final_decode_ms: float | None = None
    final_queue_wait_ms: float | None = None
    final_worker_service_ms: float | None = None
    writer_queue_wait_ms: float | None = None
    writer_worker_service_ms: float | None = None
    preview_emit_count: int = 0
    preview_duplicate_count: int = 0
    preview_regression_count: int = 0
    preview_churn_chars_total: int = 0
    preview_low_quality_suppression_count: int = 0
    preview_low_quality_emit_count: int = 0
    committed: bool = False
    conflict_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            'utterance_id': self.utterance_id,
            'ttft_ms': self.ttft_ms,
            'first_preview_decode_ms': self.first_preview_decode_ms,
            'first_preview_stability_delta_chars': self.first_preview_stability_delta_chars,
            'first_preview_queue_wait_ms': self.first_preview_queue_wait_ms,
            'first_preview_worker_service_ms': self.first_preview_worker_service_ms,
            'speech_end_to_final_ms': self.speech_end_to_final_ms,
            'speech_end_to_commit_ms': self.speech_end_to_commit_ms,
            'final_decode_ms': self.final_decode_ms,
            'final_queue_wait_ms': self.final_queue_wait_ms,
            'final_worker_service_ms': self.final_worker_service_ms,
            'writer_queue_wait_ms': self.writer_queue_wait_ms,
            'writer_worker_service_ms': self.writer_worker_service_ms,
            'preview_emit_count': self.preview_emit_count,
            'preview_duplicate_count': self.preview_duplicate_count,
            'preview_regression_count': self.preview_regression_count,
            'preview_churn_chars_total': self.preview_churn_chars_total,
            'preview_low_quality_suppression_count': self.preview_low_quality_suppression_count,
            'preview_low_quality_emit_count': self.preview_low_quality_emit_count,
            'committed': self.committed,
            'conflict_reason': self.conflict_reason,
        }


@dataclass(slots=True)
class BufferedUtteranceSnapshot:
    utterance_id: str
    audio: AudioSamples
    start_ms: float
    end_ms: float
    partial_decode_seq: int


@dataclass(slots=True)
class PreviewJobOutcome:
    emission: PreviewEmission | None
    normalization_change_count: int
    low_quality_suppressed: bool = False


@dataclass(slots=True)
class FinalJobOutcome:
    transcript: FinalTranscript
    normalization_change_count: int


@dataclass(slots=True)
class WriterCommitJobOutcome:
    final_transcript: FinalTranscript
    writer_outcome: WriterOutcome | None
    decision_result: CommitDecision | None
    learned_from_commit: bool


@dataclass(slots=True)
class DictationSessionSummary:
    session_id: str
    utterance_count: int
    preview_decode_count: int
    final_decode_count: int
    committed_count: int
    conflict_count: int
    preview_emit_count: int
    total_audio_ms: float
    average_ttft_ms: float | None = None
    average_speech_end_to_final_ms: float | None = None
    average_speech_end_to_commit_ms: float | None = None
    average_final_decode_ms: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'utterance_count': self.utterance_count,
            'preview_decode_count': self.preview_decode_count,
            'final_decode_count': self.final_decode_count,
            'committed_count': self.committed_count,
            'conflict_count': self.conflict_count,
            'preview_emit_count': self.preview_emit_count,
            'total_audio_ms': self.total_audio_ms,
            'average_ttft_ms': self.average_ttft_ms,
            'average_speech_end_to_final_ms': self.average_speech_end_to_final_ms,
            'average_speech_end_to_commit_ms': self.average_speech_end_to_commit_ms,
            'average_final_decode_ms': self.average_final_decode_ms,
            'metadata': self.metadata,
        }


@dataclass(slots=True)
class DictationSessionRunner:
    state_config: SpeechStateConfig
    preview_config: PreviewAsrConfig
    final_config: FinalAsrConfig
    vad_policy: DualVadPolicy
    preview_backend: PreviewAsrBackend
    final_backend: FinalAsrBackend
    recorder: TraceRecorder
    controller: CommitController
    context_provider: ContextProvider | None = None
    memory_store: JsonMemoryStore | None = None
    custom_mode_store: CustomModeStore | None = None
    surface_preset_store: SurfacePresetStore | None = None
    bias_engine: RecognitionBiasEngine = field(default_factory=RecognitionBiasEngine)
    writer_service: WriterService | None = None
    language_planner: LanguagePlanner = field(default_factory=LanguagePlanner)
    transcript_normalizer: LanguageAwareNormalizer = field(default_factory=LanguageAwareNormalizer)
    itn_engine: ITNEngine = field(default_factory=ITNEngine)
    lifecycle_manager: BackendLifecycleManager | None = None
    engine_mode: str = 'canonical_unmarked'
    audio_save_dir: Path | None = None
    # Optional shared clipboard ring. When set, every utterance's
    # TypedContextBundle is enriched with the most recent clipboard
    # entries (newest first) so the writer / bias engine / tools can
    # reference recent pastes. ``ProductionServiceRunner`` and the
    # registry launcher hand the same ring to both this runner and
    # the workbench app so one-shot and streaming paths see a
    # unified clipboard history.
    clipboard_ring: ClipboardRingBuffer | None = None
    context_plane: ContextPlane = field(default_factory=lambda: ContextPlane(ContextPlaneConfig()))
    session_context_memory: SessionContextMemory = field(default_factory=SessionContextMemory)
    juno_seed_runtime: JunoSeedPersonalizationRuntime | None = None
    # Frozen-per-session gate for the preview-lane HUD caption. When False,
    # the listening loop skips per-utterance ``preview_backend.decode``
    # invocations and emits no preview events. Final lane / writer / commit /
    # insertion are unaffected. Mutating mid-session is undefined; the broker
    # setter only takes effect on the *next* session.
    preview_decode_enabled: bool = True

    def run(self, frames: Iterable[AudioFrame], *, allow_interrupt: bool = False) -> DictationSessionSummary:
        machine = SpeechStateMachine(self.state_config)
        ring = AudioRingBuffer(max_frames=self.state_config.ring_buffer_frames())
        active: UtteranceAudioBuffer | None = None
        preview_decode_count = 0
        final_decode_count = 0
        preview_emit_count = 0
        preview_duplicate_count = 0
        preview_regression_count = 0
        preview_churn_chars_total = 0
        preview_low_quality_suppression_count = 0
        preview_low_quality_emit_count = 0
        committed_count = 0
        conflict_count = 0
        normalization_change_count = 0
        writer_action_count = 0
        writer_model_action_count = 0
        writer_deterministic_action_count = 0
        writer_memory_action_count = 0
        writer_mode_switch_count = 0
        writer_noop_count = 0
        total_audio_ms = 0.0
        metrics: dict[str, UtteranceRuntimeMetrics] = {}
        speech_start_ns: dict[str, int] = {}
        speech_end_ns: dict[str, int] = {}
        utterance_plans: dict[str, RecognitionBiasPlan] = {}
        interrupted = False
        requested_language_counts: dict[str, int] = {}
        observed_language_counts: dict[str, int] = {}
        language_policy_counts: dict[str, int] = {}
        code_switch_count = 0
        language_mismatch_count = 0
        utterance_records: list[dict] = []
        speech_start_deferred_count = 0

        lifecycle = self.lifecycle_manager or BackendLifecycleManager()
        self.lifecycle_manager = lifecycle
        if self.lifecycle_manager is lifecycle and lifecycle.snapshot()['component_count'] == 0:
            # Streaming MLX preview backends bind state to the warming
            # thread. The workbench server warms them on a dedicated worker;
            # flag the registration so warm_all() on the main thread skips
            # them.
            _preview_warm_on_main_thread = self.preview_config.backend_name not in {
                'qwen_asr',
            }
            lifecycle.register_backend('preview_asr', self.preview_backend, metadata={
                'configured_backend': self.preview_config.backend_name,
                'model_path': str(self.preview_config.model_path),
                'language': self.preview_config.language,
                'device': self.preview_config.device,
                'compute_type': self.preview_config.compute_type,
            }, warm_on_main_thread=_preview_warm_on_main_thread)
            lifecycle.register_backend('final_asr', self.final_backend, metadata={
                'configured_backend': self.final_config.backend_name,
                'model_path': str(self.final_config.model_path),
                'language': self.final_config.language,
                'device': self.final_config.device,
                'compute_type': self.final_config.compute_type,
            })
            if self.writer_service is not None:
                if self.writer_service.backend is not None:
                    lifecycle.register_backend(
                        'writer',
                        self.writer_service.backend,
                        metadata={'configured_backend': self.writer_service.backend.backend_name},
                        residency_policy=self.writer_service.config.residency_policy,
                        idle_unload_ttl_s=self.writer_service.config.idle_unload_ttl_s,
                    )
                    self.writer_service.backend_acquire = lambda: lifecycle.acquire('writer')
                    self.writer_service.backend_release = lambda: lifecycle.release('writer')
                lifecycle.register_component('writer_service', self.writer_service.state.mode.value, lambda: None, metadata={
                    'backend_enabled': self.writer_service.backend is not None,
                })
        lifecycle.warm_all()
        if self.writer_service is not None:
            self.controller.store.set_writer_mode(self.writer_service.state.mode.value)
        lifecycle_snapshot = lifecycle.snapshot()
        self.recorder.record(
            TraceKind.SYSTEM,
            'dictation_session_started',
            {
                'engine_mode': self.engine_mode,
                'preview_backend': self.preview_backend.backend_name,
                'final_backend': self.final_backend.backend_name,
                'writer_backend': None if self.writer_service is None or self.writer_service.backend is None else self.writer_service.backend.backend_name,
                'writer_model_path': None if self.writer_service is None else self.writer_service.config.model_path,
                'allow_interrupt': allow_interrupt,
                'memory_enabled': self.memory_store is not None,
                'context_enabled': self.context_provider is not None,
                'writer_enabled': self.writer_service is not None,
                'lifecycle': lifecycle_snapshot,
                'execution_model': 'threaded_stage_workers',
                'speech_profile': self.state_config.profile_name,
                'input_gain_db': self.state_config.input_gain_db,
            },
        )

        preview_executor = StageExecutor('preview', max_workers=1)
        final_executor = StageExecutor('final', max_workers=1)
        writer_executor = StageExecutor('writer', max_workers=1)
        preview_task: StageTask[PreviewJobOutcome] | None = None
        final_tasks: dict[str, StageTask[FinalJobOutcome]] = {}
        writer_tasks: dict[str, StageTask[WriterCommitJobOutcome]] = {}
        def _schedule_final_for_snapshot(snapshot: BufferedUtteranceSnapshot, plan: RecognitionBiasPlan) -> None:
            self._save_utterance_wav(snapshot)
            final_tasks[snapshot.utterance_id] = final_executor.submit(snapshot.utterance_id, self._decode_final, snapshot, plan)

        def _flush_completed_tasks() -> None:
            nonlocal preview_decode_count, preview_emit_count, normalization_change_count
            nonlocal preview_duplicate_count, preview_regression_count, preview_churn_chars_total
            nonlocal preview_low_quality_suppression_count, preview_low_quality_emit_count
            nonlocal final_decode_count, committed_count, conflict_count
            nonlocal writer_action_count, writer_model_action_count, writer_deterministic_action_count
            nonlocal writer_memory_action_count, writer_mode_switch_count, writer_noop_count
            nonlocal code_switch_count, language_mismatch_count
            nonlocal preview_task, active

            if preview_task is not None and preview_task.future.done():
                stage_result = preview_task.future.result()
                outcome = stage_result.result
                preview_decode_count += 1
                normalization_change_count += outcome.normalization_change_count
                emission = outcome.emission
                m = metrics.setdefault(preview_task.utterance_id, UtteranceRuntimeMetrics(utterance_id=preview_task.utterance_id))
                if outcome.low_quality_suppressed:
                    preview_low_quality_suppression_count += 1
                    m.preview_low_quality_suppression_count += 1
                if emission is not None and not (
                    active is not None and active.utterance_id != preview_task.utterance_id
                ):
                    stability_delta = 0
                    duplicate = False
                    regression = False
                    if active is not None and active.utterance_id == preview_task.utterance_id:
                        previous_text = active.last_emitted_text
                        stability_delta = active.update_last_emitted_text(emission.text)
                        duplicate = previous_text == emission.text
                        regression = bool(previous_text and emission.text and len(emission.text) < len(previous_text))
                    emitted = PreviewEmission(
                        utterance_id=emission.utterance_id,
                        text=emission.text,
                        start_ms=emission.start_ms,
                        end_ms=emission.end_ms,
                        is_final=emission.is_final,
                        backend_name=emission.backend_name,
                        language=emission.language,
                        decode_ms=emission.decode_ms,
                        stability_delta_chars=stability_delta,
                        metadata={
                            **emission.metadata,
                            'queue_wait_ms': stage_result.queue_wait_ms,
                            'worker_service_ms': stage_result.worker_service_ms,
                            'duplicate_preview': duplicate,
                            'regression_preview': regression,
                        },
                    )
                    preview_emit_count += 1
                    m.preview_emit_count += 1
                    m.preview_churn_chars_total += stability_delta
                    preview_churn_chars_total += stability_delta
                    if duplicate:
                        preview_duplicate_count += 1
                        m.preview_duplicate_count += 1
                    if regression:
                        preview_regression_count += 1
                        m.preview_regression_count += 1
                    if emitted.metadata.get('low_quality_candidate'):
                        preview_low_quality_emit_count += 1
                        m.preview_low_quality_emit_count += 1
                    if m.ttft_ms is None and emitted.utterance_id in speech_start_ns:
                        m.ttft_ms = (time.perf_counter_ns() - speech_start_ns[emitted.utterance_id]) / 1_000_000.0
                        m.first_preview_decode_ms = emitted.decode_ms
                        m.first_preview_stability_delta_chars = emitted.stability_delta_chars
                        m.first_preview_queue_wait_ms = stage_result.queue_wait_ms
                        m.first_preview_worker_service_ms = stage_result.worker_service_ms
                    self.controller.store.set_language_state(
                        requested_language=emitted.metadata.get('requested_language'),
                        observed_language=emitted.language,
                        language_policy=emitted.metadata.get('language_policy'),
                    )
                    self.controller.apply_preview(emitted)
                else:
                    self.recorder.record(TraceKind.ASR_PREVIEW, 'preview_emission_dropped', {
                        'utterance_id': preview_task.utterance_id,
                        'reason': 'stale_or_inactive',
                    })
                preview_task = None

            completed_final_ids = [uid for uid, task in final_tasks.items() if task.future.done()]
            for utterance_id in completed_final_ids:
                stage_result = final_tasks.pop(utterance_id).future.result()
                outcome = stage_result.result
                transcript = outcome.transcript
                transcript.metadata['queue_wait_ms'] = stage_result.queue_wait_ms
                transcript.metadata['worker_service_ms'] = stage_result.worker_service_ms
                final_decode_count += 1
                normalization_change_count += outcome.normalization_change_count
                requested_language = transcript.metadata.get('requested_language')
                observed_language = transcript.language
                language_policy = transcript.metadata.get('language_policy')
                _bump_counter(requested_language_counts, requested_language)
                _bump_counter(observed_language_counts, observed_language)
                _bump_counter(language_policy_counts, language_policy)
                if transcript.metadata.get('script_summary', {}).get('code_switch_detected'):
                    code_switch_count += 1
                if requested_language and observed_language and requested_language != observed_language:
                    language_mismatch_count += 1
                self.controller.store.set_language_state(
                    requested_language=requested_language,
                    observed_language=observed_language,
                    language_policy=language_policy,
                )
                m = metrics.setdefault(utterance_id, UtteranceRuntimeMetrics(utterance_id=utterance_id))
                m.final_decode_ms = transcript.decode_ms
                m.final_queue_wait_ms = stage_result.queue_wait_ms
                m.final_worker_service_ms = stage_result.worker_service_ms
                if utterance_id in speech_end_ns:
                    m.speech_end_to_final_ms = (time.perf_counter_ns() - speech_end_ns[utterance_id]) / 1_000_000.0
                plan = utterance_plans[utterance_id]
                writer_tasks[utterance_id] = writer_executor.submit(utterance_id, self._run_writer_commit_job, transcript, plan)

            completed_writer_ids = [uid for uid, task in writer_tasks.items() if task.future.done()]
            for utterance_id in completed_writer_ids:
                stage_result = writer_tasks.pop(utterance_id).future.result()
                outcome = stage_result.result
                transcript = outcome.final_transcript
                writer_outcome = outcome.writer_outcome
                decision_result = outcome.decision_result
                m = metrics.setdefault(utterance_id, UtteranceRuntimeMetrics(utterance_id=utterance_id))
                m.writer_queue_wait_ms = stage_result.queue_wait_ms
                m.writer_worker_service_ms = stage_result.worker_service_ms
                if writer_outcome is not None:
                    writer_action_count += 1
                    writer_model_action_count += int(writer_outcome.model_used)
                    writer_deterministic_action_count += int(writer_outcome.deterministic_used)
                    writer_memory_action_count += int(writer_outcome.memory_updated)
                    writer_mode_switch_count += int(writer_outcome.action == WriterActionKind.MODE_SWITCH)
                    writer_noop_count += int(writer_outcome.action == WriterActionKind.NOOP)
                plan = utterance_plans[utterance_id]
                if decision_result is not None and decision_result.committed:
                    committed_count += 1
                    m.committed = True
                    if utterance_id in speech_end_ns:
                        m.speech_end_to_commit_ms = (time.perf_counter_ns() - speech_end_ns[utterance_id]) / 1_000_000.0
                    # Persistent history (P0). Append a compact record so the macOS
                    # app's History survives broker restarts.
                    from juno_v2.observability.history_store import append_history_record

                    ms_raw = plan.metadata.get('mode_selection') if isinstance(plan.metadata, dict) else None
                    effective_mode = None
                    if isinstance(ms_raw, dict):
                        effective_mode = ms_raw.get('effective_mode') or ms_raw.get('selected_mode') or ms_raw.get('mode')
                    committed_text = (decision_result.committed_text or "").strip()
                    raw_text = (transcript.metadata.get('raw_text') or transcript.text or "").strip()
                    words = len([tok for tok in committed_text.split() if tok])
                    ctx = plan.context
                    bundle_id = (getattr(ctx, 'metadata', None) or {}).get('app_bundle_id')
                    append_history_record(
                        self.recorder.log_dir,
                        {
                            "utterance_id": utterance_id,
                            "ts_unix_ms": int(time.time() * 1000),
                            "transcript": committed_text,
                            "raw_transcript": raw_text,
                            "mode": str(effective_mode or ""),
                            "final_backend": str(transcript.backend_name or ""),
                            "model_path": str(getattr(transcript, "model_path", "") or ""),
                            "context": {
                                "app_name": getattr(ctx, "app_name", None),
                                "app_bundle_id": bundle_id,
                                "window_title": getattr(ctx, "window_title", None),
                                "app_category": getattr(ctx, "app_category", None),
                            },
                            "failure_reason": None,
                            "session_class": "insert",
                            "processing_ms": int(m.speech_end_to_commit_ms or 0),
                            "words": words,
                            "replay_available": False,
                        },
                    )
                    # Live-dictation latency feedback. No-op unless
                    # JUNO_V2_LOG_LATENCY is set; when enabled, one
                    # stderr line per committed utterance so the
                    # operator can watch TTFT and speech-end-to-commit
                    # while dictating without having to post-process
                    # the trace JSONL.
                    _emit_live_latency_line(
                        m,
                        committed_text=decision_result.committed_text,
                        preview_backend=self.preview_backend.backend_name,
                        final_backend=self.final_backend.backend_name,
                        repetition_collapse=transcript.metadata.get('repetition_collapse'),
                    )
                elif decision_result is not None and decision_result.conflict_reason:
                    conflict_count += 1
                    m.conflict_reason = decision_result.conflict_reason
                utterance_records.append({
                    'utterance_id': utterance_id,
                    'requested_language': transcript.metadata.get('requested_language'),
                    'observed_language': transcript.language,
                    'language_policy': transcript.metadata.get('language_policy'),
                    'raw_text': transcript.metadata.get('raw_text', transcript.text),
                    'final_text': transcript.text,
                    'committed_text': None if decision_result is None else decision_result.committed_text,
                    'committed': bool(decision_result is not None and decision_result.committed),
                    'conflict_reason': None if decision_result is None else decision_result.conflict_reason,
                    'writer_action': None if writer_outcome is None else writer_outcome.action.value,
                    'writer_mode': None if writer_outcome is None or writer_outcome.writer_mode is None else writer_outcome.writer_mode.value,
                    'context_candidate_count': len(plan.context.candidate_entities),
                    'context_redaction': plan.context.redaction.to_dict(),
                    'bias_phrase_count': len(plan.bias_phrases),
                    'memory_packet_summary': dict(plan.metadata.get('memory_packet_summary', {})),
                    'normalization_applied_count': len(transcript.metadata.get('normalization', {}).get('applied', [])),
                    'ttft_ms': m.ttft_ms,
                    'speech_end_to_final_ms': m.speech_end_to_final_ms,
                    'speech_end_to_commit_ms': m.speech_end_to_commit_ms,
                    'final_decode_ms': m.final_decode_ms,
                    'queue_wait_ms': transcript.metadata.get('queue_wait_ms'),
                    'worker_service_ms': transcript.metadata.get('worker_service_ms'),
                    'writer_queue_wait_ms': stage_result.queue_wait_ms,
                    'writer_worker_service_ms': stage_result.worker_service_ms,
                })

        try:
            for frame in frames:
                _flush_completed_tasks()
                prepared_frame = preprocess_audio_frame(frame, self.state_config)
                total_audio_ms = prepared_frame.end_ms
                ring.append(prepared_frame)
                decision = self.vad_policy.decide(prepared_frame)
                self.recorder.record(
                    TraceKind.SYSTEM,
                    'vad_frame_decision',
                    {
                        'frame_index': frame.index,
                        'start_ms': frame.start_ms,
                        'end_ms': frame.end_ms,
                        'decision': decision.decision,
                        'webrtc_speech': decision.webrtc_speech,
                        'silero_speech': decision.silero_speech,
                        'energy_speech': decision.energy_speech,
                        'energy_rms': decision.energy_rms,
                        'silero_score': decision.silero_score,
                        'ring_buffer_ms': ring.duration_ms(),
                    },
                )
                events = machine.process(prepared_frame, decision)
                for event in events:
                    self.recorder.record(TraceKind.SYSTEM, event.kind.value, event.to_dict())
                    if event.kind == SpeechEventKind.SPEECH_STARTED and event.utterance_id is not None:
                        active = UtteranceAudioBuffer(utterance_id=event.utterance_id)
                        active.seed(ring.snapshot()[-self.state_config.pre_roll_frames :])
                        # Selection anchor for preview / replace-selection commits.
                        # macOS one-shot (``ingest_wav``) uses client-frozen juno-capability
                        # JSON merged in :class:`~juno_core_v3.dictation.pipeline.OneShotDictationPipeline`;
                        # streaming relies on this AX snapshot at utterance start instead.
                        editable_sync = None
                        editable_sync_fn = getattr(self.context_provider, 'editable_sync_request', None) if self.context_provider is not None else None
                        if callable(editable_sync_fn):
                            try:
                                editable_sync = editable_sync_fn()
                            except Exception as exc:
                                self.recorder.record(TraceKind.CONTEXT, 'editable_sync_failed', {'error': str(exc)})
                        if editable_sync is not None:
                            self.controller.sync_client_state(editable_sync)
                            self.recorder.record(TraceKind.CONTEXT, 'editable_sync_applied', {
                                'buffer_length': len(editable_sync.buffer_text),
                                'selection_start': editable_sync.selection_start,
                                'selection_end': editable_sync.selection_end,
                            })
                        self.controller.begin_utterance(event.utterance_id)
                        metrics[event.utterance_id] = UtteranceRuntimeMetrics(utterance_id=event.utterance_id)
                        speech_start_ns[event.utterance_id] = time.perf_counter_ns()
                        utterance_plans[event.utterance_id] = self._plan_for_utterance(event.utterance_id)
                    elif event.kind == SpeechEventKind.SESSION_ABORTED:
                        active = None
                        self.controller.abort_active('weak_start_rejected')
                    elif event.kind == SpeechEventKind.SPEECH_ENDED and active is not None:
                        # Include the frame that satisfied end-of-utterance silence before snapshot.
                        active.append(frame)
                        speech_end_ns[active.utterance_id] = time.perf_counter_ns()
                        plan = utterance_plans.get(active.utterance_id) or self._plan_for_utterance(active.utterance_id)
                        _schedule_final_for_snapshot(self._snapshot_active(active), plan)
                        active = None

                if active is not None:
                    active.append(frame)
                    if machine.phase in {SpeechPhase.MAYBE_SPEECH, SpeechPhase.IN_SPEECH, SpeechPhase.PAUSED} and active.should_decode_partial(
                        self.preview_config.partial_decode_interval_ms,
                        self.preview_config.min_decode_audio_ms,
                    ) and preview_task is None:
                        plan = utterance_plans.get(active.utterance_id) or self._plan_for_utterance(active.utterance_id)
                        preview_task = preview_executor.submit(active.utterance_id, self._decode_preview, self._snapshot_active(active), plan, False)
                        active.mark_partial_decode()

            # Replay / finite sources: if the stream ended before the speech state machine
            # emitted SPEECH_ENDED (e.g. trailing silence shorter than end_trigger), finalize.
            if active is not None:
                speech_end_ns.setdefault(active.utterance_id, time.perf_counter_ns())
                plan = utterance_plans.get(active.utterance_id) or self._plan_for_utterance(active.utterance_id)
                _schedule_final_for_snapshot(self._snapshot_active(active), plan)
                active = None

            while preview_task is not None or final_tasks or writer_tasks:
                _flush_completed_tasks()
                time.sleep(0.001)
        except KeyboardInterrupt:
            if not allow_interrupt:
                raise
            interrupted = True
            self.recorder.record(TraceKind.SYSTEM, 'dictation_session_interrupted', {})
            if active is not None:
                self.controller.abort_active('keyboard_interrupt')
        finally:
            preview_executor.shutdown(wait=True)
            final_executor.shutdown(wait=True)
            writer_executor.shutdown(wait=True)

        memory_snapshot = self.memory_store.snapshot() if self.memory_store is not None else None
        utterance_metrics_payload = [m.to_dict() for m in metrics.values()]
        runtime_truth = build_runtime_truth_report(
            utterance_metrics=utterance_metrics_payload,
            utterance_count=machine.utterance_count,
            committed_count=committed_count,
            conflict_count=conflict_count,
            preview_decode_count=preview_decode_count,
            final_decode_count=final_decode_count,
            preview_emit_count=preview_emit_count,
            preview_duplicate_count=preview_duplicate_count,
            preview_regression_count=preview_regression_count,
            preview_churn_chars_total=preview_churn_chars_total,
            preview_low_quality_suppression_count=preview_low_quality_suppression_count,
            preview_low_quality_emit_count=preview_low_quality_emit_count,
            normalization_change_count=normalization_change_count,
            writer_action_count=writer_action_count,
            writer_model_action_count=writer_model_action_count,
            writer_deterministic_action_count=writer_deterministic_action_count,
            writer_noop_count=writer_noop_count,
            memory_snapshot=memory_snapshot,
            memory_packet_summaries=[record.get('memory_packet_summary', {}) for record in utterance_records],
            requested_language_counts=requested_language_counts,
            observed_language_counts=observed_language_counts,
            language_policy_counts=language_policy_counts,
            code_switch_utterance_count=code_switch_count,
            language_mismatch_count=language_mismatch_count,
        )
        summary = DictationSessionSummary(
            session_id=self.recorder.session_id,
            utterance_count=machine.utterance_count,
            preview_decode_count=preview_decode_count,
            final_decode_count=final_decode_count,
            committed_count=committed_count,
            conflict_count=conflict_count,
            preview_emit_count=preview_emit_count,
            total_audio_ms=total_audio_ms,
            average_ttft_ms=_avg([m.ttft_ms for m in metrics.values()]),
            average_speech_end_to_final_ms=_avg([m.speech_end_to_final_ms for m in metrics.values()]),
            average_speech_end_to_commit_ms=_avg([m.speech_end_to_commit_ms for m in metrics.values()]),
            average_final_decode_ms=_avg([m.final_decode_ms for m in metrics.values()]),
            metadata={
                'engine_mode': self.engine_mode,
                'preview_backend': self.preview_backend.backend_name,
                'final_backend': self.final_backend.backend_name,
                'writer_backend': None if self.writer_service is None or self.writer_service.backend is None else self.writer_service.backend.backend_name,
                'writer_model_path': None if self.writer_service is None else self.writer_service.config.model_path,
                'interrupted': interrupted,
                'utterance_metrics': utterance_metrics_payload,
                'memory_enabled': self.memory_store is not None,
                'context_enabled': self.context_provider is not None,
                'writer_enabled': self.writer_service is not None,
                'writer_action_count': writer_action_count,
                'writer_model_action_count': writer_model_action_count,
                'writer_deterministic_action_count': writer_deterministic_action_count,
                'writer_memory_action_count': writer_memory_action_count,
                'writer_mode_switch_count': writer_mode_switch_count,
                'writer_noop_count': writer_noop_count,
                'normalization_change_count': normalization_change_count,
                'requested_language_counts': requested_language_counts,
                'observed_language_counts': observed_language_counts,
                'language_policy_counts': language_policy_counts,
                'code_switch_utterance_count': code_switch_count,
                'language_mismatch_count': language_mismatch_count,
                'speech_start_deferred_count': speech_start_deferred_count,
                'memory_snapshot': memory_snapshot.to_dict() if memory_snapshot is not None else None,
                'writer_mode': self.writer_service.state.mode.value if self.writer_service is not None else None,
                'lifecycle': lifecycle.snapshot(),
                'runtime_truth': runtime_truth.to_dict(),
                'utterance_records': utterance_records,
                'memory_packet_summaries': [record.get('memory_packet_summary', {}) for record in utterance_records],
                'execution_model': 'threaded_stage_workers',
                'speech_profile': self.state_config.profile_name,
                'input_gain_db': self.state_config.input_gain_db,
            },
        )
        self.recorder.record(TraceKind.METRICS, 'dictation_runtime_metrics', summary.to_dict())
        self.recorder.record(TraceKind.SYSTEM, 'dictation_session_completed', summary.to_dict())
        return summary

    def _snapshot_active(self, active: UtteranceAudioBuffer) -> BufferedUtteranceSnapshot:
        return BufferedUtteranceSnapshot(
            utterance_id=active.utterance_id,
            audio=active.audio().copy(),
            start_ms=active.start_ms,
            end_ms=active.end_ms,
            partial_decode_seq=active.partial_decode_seq,
        )

    def _save_utterance_wav(self, snapshot: BufferedUtteranceSnapshot) -> None:
        if self.audio_save_dir is None:
            return
        try:
            import numpy as np
            audio_dir = self.audio_save_dir / self.recorder.session_id
            audio_dir.mkdir(parents=True, exist_ok=True)
            wav_path = audio_dir / f"{snapshot.utterance_id}.wav"
            pcm = np.clip(snapshot.audio, -1.0, 1.0)
            pcm_int16 = (pcm * 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.state_config.sample_rate_hz)
                wf.writeframes(pcm_int16.tobytes())
            self.recorder.record(
                TraceKind.SYSTEM,
                "utterance_audio_saved",
                {"utterance_id": snapshot.utterance_id, "path": str(wav_path),
                 "duration_ms": snapshot.end_ms - snapshot.start_ms},
            )
        except Exception as exc:
            self.recorder.record(
                TraceKind.SYSTEM,
                "utterance_audio_save_failed",
                {"utterance_id": snapshot.utterance_id, "error": str(exc)},
            )

    def _apply_context_plane(self, utterance_id: str, context: TypedContextBundle) -> tuple[TypedContextBundle, dict[str, object]]:
        """Apply ContextPlane to the live/streaming context bundle.

        The v2 session runner still consumes ``TypedContextBundle`` downstream, so
        we back-apply the plane-authoritative fields after suppression / budgets /
        degradation are computed.  This keeps the downstream bias, writer, and
        normalization logic unchanged while making ContextPlane the runtime truth.
        """
        if self.clipboard_ring is not None and self.context_plane.config.clipboard_ring is None:
            self.context_plane.config.clipboard_ring = self.clipboard_ring
        packet = self.context_plane.build_from_typed_bundle(context, surface_id="workbench_dev")
        context.selected_text = packet.selected_text
        context.focused_text_before = packet.focused_text_before
        context.focused_text_after = packet.focused_text_after
        context.clipboard_text = packet.clipboard_text
        context.field_text_excerpt = packet.field_text_excerpt
        context.app_name = packet.app_name
        context.window_title = packet.window_title
        plane_meta = {
            'suppression': packet.metadata.get('suppression', 'none'),
            'degradation': packet.metadata.get('degradation', 'none'),
            'truncation_applied': dict(packet.truncation_applied),
            'budget_exceeded': bool(packet.metadata.get('budget_exceeded', False)),
            'provenance': {k: v.value if hasattr(v, 'value') else v for k, v in packet.provenance.items()},
            'budgets': dict(packet.metadata.get('budgets', {})),
            'ordered_sources': list(packet.metadata.get('ordered_sources', [])),
            'surface_id': packet.metadata.get('surface_id', 'workbench_dev'),
        }
        self.recorder.record(
            TraceKind.CONTEXT,
            'streaming_context_plane_applied',
            {'utterance_id': utterance_id, **plane_meta},
        )
        return context, plane_meta

    def _plan_for_utterance(self, utterance_id: str) -> RecognitionBiasPlan:
        context = self.context_provider.snapshot() if self.context_provider is not None else TypedContextBundle()
        # Enrich the context bundle with recent clipboard entries before
        # bias/writer/memory stages read it. inject_clipboard_ring is a
        # no-op when no ring is wired (tests / standalone workbench).
        inject_clipboard_ring(context, self.clipboard_ring, limit=5)
        context, context_plane_meta = self._apply_context_plane(utterance_id, context)
        new_screen_terms = self.session_context_memory.observe(context)
        recent_screen_terms = self.session_context_memory.snapshot(limit=24)
        if recent_screen_terms:
            context.metadata['recent_screen_terms'] = recent_screen_terms
            merged_candidates = list(context.candidate_entities)
            seen_candidates = {item.casefold() for item in merged_candidates}
            for term in recent_screen_terms[:12]:
                key = term.casefold()
                if key not in seen_candidates:
                    merged_candidates.append(term)
                    seen_candidates.add(key)
            context.candidate_entities = merged_candidates[:36]
        self.recorder.record(
            TraceKind.CONTEXT,
            'session_context_memory_updated',
            {
                'utterance_id': utterance_id,
                'new_terms': new_screen_terms[:12],
                'recent_terms': recent_screen_terms[:24],
                'candidate_count': len(context.candidate_entities),
            },
        )
        snapshot = self.memory_store.snapshot() if self.memory_store is not None else MemorySnapshot(schema_version=1)
        language_decision = self.language_planner.plan_utterance(
            utterance_id=utterance_id,
            context=context,
            configured_preview_language=self.preview_config.language,
            configured_final_language=self.final_config.language,
        )
        mode_store = self.custom_mode_store or CustomModeStore(default_modes_data_path(self.recorder.log_dir))
        preset_store = self.surface_preset_store or SurfacePresetStore(
            default_surface_presets_path(self.recorder.log_dir)
        )
        manual_m = getattr(self.controller.store.state, 'manual_writer_mode', None)
        custom_m = getattr(self.controller.store.state, 'custom_writer_mode', None)
        custom_rec = mode_store.get(custom_m) if custom_m else None
        bundle_raw = (context.metadata or {}).get('app_bundle_id')
        bundle_id = bundle_raw.strip() if isinstance(bundle_raw, str) and bundle_raw.strip() else None
        mode_sel, mode_pol, active_preset = resolve_mode_with_surface_presets(
            manual_mode_name=manual_m,
            custom_mode_name=custom_m,
            custom_record=custom_rec,
            surface_hint=context.app_category,
            surface_bundle_id=bundle_id,
            preset_store=preset_store,
            custom_mode_store=mode_store,
        )
        memory_packet = rank_memory_for_context(
            snapshot,
            context=context,
            mode_policy=mode_pol,
            effective_mode=mode_sel.effective_mode,
        )
        include_title = bool(active_preset and active_preset.include_window_title_in_asr)
        surf_line = build_surface_context_line(
            app_name=context.app_name,
            window_title=context.window_title,
            app_category=context.app_category,
            include_window_title=include_title,
        )
        preset_addon = (active_preset.asr_addon or '').strip() if active_preset else None
        base_prompt = _merge_prompts(
            language_decision.initial_prompt,
            self.preview_config.initial_prompt,
            (mode_pol.prompt_prefix or '').strip() or None,
            preset_addon,
            surf_line,
        )
        seed_attachment = None
        if self.juno_seed_runtime is not None:
            seed_attachment = self.juno_seed_runtime.build_seed_attachment(
                snapshot=snapshot,
                context=context,
                context_plane_suppression=str(context_plane_meta.get('suppression'))
                if context_plane_meta.get('suppression') is not None
                else None,
            )
        plan = self.bias_engine.build_plan(
            utterance_id=utterance_id,
            snapshot=snapshot,
            context=context,
            base_prompt=base_prompt,
            memory_packet=memory_packet,
            mode_policy=mode_pol,
            effective_mode=mode_sel.effective_mode,
            seed_attachment=seed_attachment,
        )
        plan.metadata['language_decision'] = language_decision.to_dict()
        plan.metadata['mode_selection'] = mode_sel.to_dict()
        plan.metadata['mode_policy_snapshot'] = mode_pol.to_dict()
        if active_preset is not None:
            plan.metadata['surface_preset'] = {
                'id': active_preset.id,
                'bundle_id': active_preset.bundle_id,
                'asr_addon_len': len(active_preset.asr_addon or ''),
            }
            if (active_preset.writer_tone_addon or '').strip():
                plan.metadata['surface_preset_writer_tone'] = active_preset.writer_tone_addon.strip()
        plan.metadata['context_plane'] = context_plane_meta
        packet_summary = {
            'bias_phrase_count': len(plan.bias_phrases),
            'lexicon_terms': len(plan.metadata.get('memory_serving_packet', {}).get('lexicon_terms', [])),
            'replacements': len(plan.metadata.get('memory_serving_packet', {}).get('replacements', [])),
            'corrections': len(plan.metadata.get('memory_serving_packet', {}).get('corrections', [])),
            'session_entities': len(plan.metadata.get('memory_serving_packet', {}).get('session_entities', [])),
        }
        plan.metadata['memory_packet_summary'] = packet_summary
        self.recorder.record(TraceKind.MEMORY, 'memory_serving_packet_built', {'utterance_id': utterance_id, **packet_summary})
        self.recorder.record(TraceKind.CONTEXT, 'utterance_context_planned', plan.to_dict())
        return plan

    def _decode_preview(self, snapshot: BufferedUtteranceSnapshot, plan: RecognitionBiasPlan, is_final: bool = False) -> PreviewJobOutcome:
        # When the user has live transcriptions disabled the engine session is
        # configured with ``preview_decode_enabled=False``. Bail out *before*
        # touching the lifecycle manager so we don't flap the resident-model
        # refcount, and *before* running any of the request building so the
        # preview backend never sees this utterance. Final lane / writer /
        # commit / insertion are completely unaffected — they consume the
        # final decode path which lives below.
        if not self.preview_decode_enabled:
            self.recorder.record(TraceKind.ASR_PREVIEW, 'preview_decode_disabled', {
                'utterance_id': snapshot.utterance_id,
                'start_ms': snapshot.start_ms,
                'end_ms': snapshot.end_ms,
                'is_final': is_final,
                'reason': 'live_caption_disabled',
            })
            return PreviewJobOutcome(
                emission=None,
                normalization_change_count=0,
                low_quality_suppressed=False,
            )
        language_decision = _language_decision_dict(plan)
        req = PreviewDecodeRequest(
            utterance_id=snapshot.utterance_id,
            audio=snapshot.audio,
            sample_rate_hz=self.state_config.sample_rate_hz,
            start_ms=snapshot.start_ms,
            end_ms=snapshot.end_ms,
            is_final=is_final,
            language=language_decision.get('request_language') or self.preview_config.language,
            allowed_languages=list(language_decision.get('allowed_languages', [])),
            language_policy=language_decision.get('policy_name'),
            initial_prompt=None,
            decode_seq=snapshot.partial_decode_seq,
            reset_decoder_state=(snapshot.partial_decode_seq == 0),
            bias_phrases=[],
            context_payload={
                **plan.context.to_dict(),
                'language_decision': language_decision,
                'preview_prompt_mode': 'personalization_repair_only',
                'preview_personalization_terms': preview_personalization_terms_from_plan(plan),
                'preview_display_orthography': True,
            },
        )
        self.recorder.record(TraceKind.ASR_PREVIEW, 'preview_decode_started', {
            'utterance_id': req.utterance_id,
            'start_ms': req.start_ms,
            'end_ms': req.end_ms,
            'audio_duration_ms': req.audio_duration_ms,
            'is_final': is_final,
            'language': req.language,
            'allowed_languages': req.allowed_languages,
            'language_policy': req.language_policy,
            'bias_phrase_count': len(req.bias_phrases),
            'decode_seq': req.decode_seq,
        })
        lifecycle = self.lifecycle_manager
        if lifecycle is not None:
            lifecycle.acquire('preview_asr')
        try:
            result = self.preview_backend.decode(req)
        finally:
            if lifecycle is not None:
                lifecycle.release('preview_asr')
        normalization = self._normalize_transcript(
            result.text.strip(),
            plan=plan,
            observed_language=result.language,
            requested_language=req.language,
            policy_name=req.language_policy,
            scope='preview',
        )
        self.recorder.record(TraceKind.ASR_PREVIEW, 'preview_decode_completed', {
            **result.to_dict(),
            'requested_language': req.language,
            'allowed_languages': req.allowed_languages,
            'language_policy': req.language_policy,
            'normalization': normalization.to_dict(),
        })
        text = normalization.normalized_text.strip()
        low_quality_suppressed = bool(result.metadata.get('low_quality_candidate')) and not text
        preview_itn = {
            'applied': False,
            'profile': None,
            'rules_applied': [],
            'reason': 'preview_partial',
        }
        self.recorder.record(TraceKind.SYSTEM, 'streaming_preview_itn_skipped', {
            'utterance_id': result.utterance_id,
            'reason': 'preview_partial',
        })
        if not text:
            return PreviewJobOutcome(emission=None, normalization_change_count=len(normalization.applied), low_quality_suppressed=low_quality_suppressed)
        emission = PreviewEmission(
            utterance_id=result.utterance_id,
            text=text,
            start_ms=result.start_ms,
            end_ms=result.end_ms,
            is_final=result.is_final,
            backend_name=result.backend_name,
            language=result.language,
            decode_ms=result.decode_ms,
            stability_delta_chars=0,
            metadata={
                **result.metadata,
                'raw_text': normalization.raw_text,
                'normalization': normalization.to_dict(),
                'requested_language': req.language,
                'allowed_languages': req.allowed_languages,
                'language_policy': req.language_policy,
                'script_summary': normalization.metadata.get('script_summary'),
                'decode_seq': req.decode_seq,
                'itn': preview_itn,
            },
        )
        self.recorder.record(TraceKind.ASR_PREVIEW, 'preview_emitted', emission.to_dict())
        return PreviewJobOutcome(emission=emission, normalization_change_count=len(normalization.applied))

    def _decode_final(self, snapshot: BufferedUtteranceSnapshot, plan: RecognitionBiasPlan) -> FinalJobOutcome:
        language_decision = _language_decision_dict(plan)
        req = FinalDecodeRequest(
            utterance_id=snapshot.utterance_id,
            audio=snapshot.audio,
            sample_rate_hz=self.state_config.sample_rate_hz,
            start_ms=snapshot.start_ms,
            end_ms=snapshot.end_ms,
            language=language_decision.get('request_language') or self.final_config.language,
            allowed_languages=list(language_decision.get('allowed_languages', [])),
            language_policy=language_decision.get('policy_name'),
            initial_prompt=_merge_prompts(language_decision.get('initial_prompt'), self.final_config.initial_prompt, plan.initial_prompt),
            bias_phrases=plan.bias_phrases,
            context_payload={**plan.context.to_dict(), 'language_decision': language_decision},
        )
        self.recorder.record(TraceKind.ASR_FINAL, 'final_decode_started', {
            'utterance_id': req.utterance_id,
            'start_ms': req.start_ms,
            'end_ms': req.end_ms,
            'audio_duration_ms': req.audio_duration_ms,
            'language': req.language,
            'allowed_languages': req.allowed_languages,
            'language_policy': req.language_policy,
            'bias_phrase_count': len(req.bias_phrases),
        })
        lifecycle = self.lifecycle_manager
        if lifecycle is not None:
            lifecycle.acquire('final_asr')
        try:
            result = self.final_backend.decode(req)
        finally:
            if lifecycle is not None:
                lifecycle.release('final_asr')
        raw_for_observe = (result.text or "").strip()
        if self.juno_seed_runtime is not None and raw_for_observe:
            cp_sup = None
            if isinstance(plan.metadata, dict):
                cp = plan.metadata.get('context_plane')
                if isinstance(cp, dict):
                    v = cp.get('suppression')
                    cp_sup = str(v) if v is not None else None
            suppressed_ob = self.juno_seed_runtime.durable_memory_suppressed(
                plan.context,
                context_plane_suppression=cp_sup,
            )
            self.juno_seed_runtime.observe_transcript_for_context_entities(
                raw_for_observe,
                plan.context,
                durable_memory_suppressed=suppressed_ob,
            )
        normalization = self._normalize_transcript(
            result.text.strip(),
            plan=plan,
            observed_language=result.language,
            requested_language=req.language,
            policy_name=req.language_policy,
            scope='final',
        )
        # ITN — applied after language-aware normalization, before commit.
        # Profile is derived from app_category in the utterance context so
        # code/terminal sessions get the right rule set automatically.
        _itn_text = normalization.normalized_text.strip()
        _itn_app_category = getattr(plan.context, "app_category", None) if plan.context is not None else None
        _itn_profile = self.itn_engine.profile_for_category(_itn_app_category)
        _itn_fmt = resolve_itn_format_policy(plan.context)
        _itn_result = self.itn_engine.run(_itn_text, profile=_itn_profile, format_policy=_itn_fmt)
        if _itn_result.changed:
            self.recorder.record(TraceKind.SYSTEM, 'streaming_itn_applied', {
                'utterance_id': result.utterance_id,
                'profile': _itn_result.profile,
                'rules_applied': _itn_result.rules_applied,
                'original_chars': len(_itn_result.original_text),
                'output_chars': len(_itn_result.text),
                'format': dict(_itn_result.format_snapshot),
            })
        _itn_metadata = {
            'applied': _itn_result.changed,
            'profile': _itn_result.profile,
            'rules_applied': list(_itn_result.rules_applied),
            'format': dict(_itn_result.format_snapshot),
        }
        _post_itn_text = _itn_result.text if _itn_result.changed else _itn_text

        self.recorder.record(TraceKind.ASR_FINAL, 'final_decode_completed', {
            **result.to_dict(),
            'requested_language': req.language,
            'allowed_languages': req.allowed_languages,
            'language_policy': req.language_policy,
            'normalization': normalization.to_dict(),
            'itn': _itn_metadata,
        })
        # Collapse autoregressive repetition loops ("Some can be saved.
        # Some can be saved. Some can be saved. Som") before staging
        # for commit. Gated on the words-per-second ratio against the
        # audio duration so legitimate user dictation repetition is
        # preserved (see juno_v2/memory/repetition.py for the
        # discrimination logic and the live-session failure cases
        # that motivated it).
        collapsed_text, repetition_diag = collapse_tail_repetition(
            _post_itn_text,
            audio_duration_ms=result.audio_duration_ms,
        )
        if repetition_diag.collapsed:
            self.recorder.record(TraceKind.ASR_FINAL, 'final_repetition_collapsed', {
                'utterance_id': result.utterance_id,
                **repetition_diag.to_dict(),
                'before_text': normalization.normalized_text.strip(),
                'after_text': collapsed_text,
            })
        transcript = FinalTranscript(
            utterance_id=result.utterance_id,
            text=collapsed_text,
            start_ms=result.start_ms,
            end_ms=result.end_ms,
            backend_name=result.backend_name,
            model_path=getattr(result, 'model_path', '') or '',
            language=result.language,
            decode_ms=result.decode_ms,
            end_of_turn_latency_ms=result.end_of_turn_latency_ms,
            metadata={
                **result.metadata,
                'segment_count': len(result.segments),
                # Per-segment audio-side signals required by the commit-side
                # trailing-silence-hallucination guard. Tuples of dicts keep
                # ``final_metadata`` JSON-serialisable.
                'segments': tuple(seg.to_dict() for seg in result.segments),
                'raw_text': normalization.raw_text,
                'normalization': normalization.to_dict(),
                'repetition_collapse': repetition_diag.to_dict(),
                'audio_duration_ms': result.audio_duration_ms,
                'requested_language': req.language,
                'allowed_languages': req.allowed_languages,
                'language_policy': req.language_policy,
                'script_summary': normalization.metadata.get('script_summary'),
                'itn': _itn_metadata,
            },
        )
        self.recorder.record(TraceKind.ASR_FINAL, 'final_transcript_emitted', transcript.to_dict())
        return FinalJobOutcome(transcript=transcript, normalization_change_count=len(normalization.applied))

    def _apply_writer(self, *, final_transcript: FinalTranscript, plan: RecognitionBiasPlan) -> WriterOutcome | None:
        if self.writer_service is None:
            return None
        active_selection = self.controller.session_selection(final_transcript.utterance_id)
        ms_raw = plan.metadata.get('mode_selection')
        mp_raw = plan.metadata.get('mode_policy_snapshot')
        tone_raw = plan.metadata.get('surface_preset_writer_tone')
        writer_tone: str | None = tone_raw.strip() if isinstance(tone_raw, str) and tone_raw.strip() else None
        if isinstance(ms_raw, dict) and isinstance(mp_raw, dict):
            mode_sel = ModeSelection.from_dict(ms_raw)
            mode_pol = ModePolicy.from_dict(mp_raw)
        else:
            mode_store = self.custom_mode_store or CustomModeStore(default_modes_data_path(self.recorder.log_dir))
            preset_store = self.surface_preset_store or SurfacePresetStore(
                default_surface_presets_path(self.recorder.log_dir)
            )
            manual_m = getattr(self.controller.store.state, 'manual_writer_mode', None)
            custom_m = getattr(self.controller.store.state, 'custom_writer_mode', None)
            custom_rec = mode_store.get(custom_m) if custom_m else None
            br = (plan.context.metadata or {}).get('app_bundle_id')
            bid = br.strip() if isinstance(br, str) and br.strip() else None
            mode_sel, mode_pol, _ap = resolve_mode_with_surface_presets(
                manual_mode_name=manual_m,
                custom_mode_name=custom_m,
                custom_record=custom_rec,
                surface_hint=plan.context.app_category,
                surface_bundle_id=bid,
                preset_store=preset_store,
                custom_mode_store=mode_store,
            )
            if writer_tone is None and _ap is not None and (_ap.writer_tone_addon or '').strip():
                writer_tone = _ap.writer_tone_addon.strip()
        mp = plan.metadata.get('memory_serving_packet')
        if isinstance(mp, dict):
            memory_packet_dict = mp
        else:
            to_dict = getattr(mp, "to_dict", None)
            converted = to_dict() if callable(to_dict) else None
            memory_packet_dict = converted if isinstance(converted, dict) else None
        partial_text = getattr(self.controller.store.state, 'partial_text', '') or ''
        outcome = self.writer_service.process_transcript(
            utterance_id=final_transcript.utterance_id,
            final_text=final_transcript.text,
            raw_text=final_transcript.metadata.get('raw_text', final_transcript.text),
            context=plan.context,
            anchor_selection=active_selection,
            memory_store=self.memory_store,
            memory_snapshot=self.memory_store.snapshot() if self.memory_store is not None else None,
            memory_packet=memory_packet_dict,
            language_hint=final_transcript.language or final_transcript.metadata.get('requested_language'),
            mode_policy=mode_pol,
            mode_selection=mode_sel,
            partial_text=partial_text,
            writer_tone_addon=writer_tone,
        )
        prof = plan.metadata.get('personalization_profile')
        if isinstance(prof, dict):
            outcome.metadata['personalization_applied'] = True
            outcome.metadata['top_vocab_hits'] = prof.get('vocabulary_candidates', [])[:8]
            outcome.metadata['top_replacements'] = prof.get('replacements', [])[:8]
            outcome.metadata['top_snippets'] = prof.get('snippets', [])[:8]
            outcome.metadata['top_entities'] = prof.get('entities', [])[:8]
            outcome.metadata['top_styles'] = prof.get('style_cards', [])[:8]
            outcome.metadata['recent_correction_hits'] = prof.get('recent_accepted_corrections', [])[:8]
        self.recorder.record(TraceKind.WRITER, 'writer_outcome', outcome.to_dict())
        eff = outcome.effective_mode or (
            outcome.writer_mode.value if outcome.writer_mode is not None else (
                self.writer_service.state.mode.value if self.writer_service else 'default_surface'
            )
        )
        self.controller.store.set_writer_mode(eff)
        if outcome.action == WriterActionKind.MODE_SWITCH and outcome.metadata.get('set_manual_writer_mode'):
            self.controller.store.set_manual_writer_mode(str(outcome.metadata['set_manual_writer_mode']))
        if outcome.action == WriterActionKind.STATE_MUTATION and outcome.metadata.get('state_action') == 'discard_active_partial':
            self.controller.store.clear_partial()
        elif outcome.action == WriterActionKind.STATE_MUTATION and outcome.metadata.get('pending_partial_text') is not None:
            from juno_v2.contracts.workbench import PartialCommitRequest

            self.controller.store.apply_partial(PartialCommitRequest(text=str(outcome.metadata.get('pending_partial_text', ''))))
        self.controller.store.set_last_writer_action(outcome.action.value, payload=outcome.metadata)
        return outcome

    def _run_writer_commit_job(self, final_transcript: FinalTranscript, plan: RecognitionBiasPlan) -> WriterCommitJobOutcome:
        writer_outcome = self._apply_writer(final_transcript=final_transcript, plan=plan)
        decision_result = self._commit_from_writer(final_transcript=final_transcript, writer_outcome=writer_outcome)
        learned = False
        if decision_result is not None and decision_result.committed and (writer_outcome is None or writer_outcome.learn_from_commit):
            raw_text = final_transcript.metadata.get('raw_text', final_transcript.text)
            committed_text = decision_result.committed_text or final_transcript.text
            if self._should_learn_from_commit(raw_text=raw_text, committed_text=committed_text, final_transcript=final_transcript, writer_outcome=writer_outcome):
                self._learn_from_commit(
                    raw_text=raw_text,
                    committed_text=committed_text,
                    plan=plan,
                )
                learned = True
            else:
                self.recorder.record(TraceKind.MEMORY, 'memory_update_skipped_from_commit', {
                    'utterance_id': final_transcript.utterance_id,
                    'raw_text': raw_text,
                    'committed_text': committed_text,
                })
        return WriterCommitJobOutcome(
            final_transcript=final_transcript,
            writer_outcome=writer_outcome,
            decision_result=decision_result,
            learned_from_commit=learned,
        )

    def _normalize_transcript(
        self,
        text: str,
        *,
        plan: RecognitionBiasPlan,
        observed_language: str | None,
        requested_language: str | None,
        policy_name: str | None,
        scope: str,
    ):
        snapshot = self.memory_store.snapshot() if self.memory_store is not None else MemorySnapshot(schema_version=1)
        memory_norm = self.bias_engine.normalize_transcript(text, snapshot=snapshot, plan=plan, scope=scope)
        language_norm = self.transcript_normalizer.normalize_transcript(
            memory_norm.normalized_text,
            requested_language=requested_language,
            observed_language=observed_language,
            policy_name=policy_name,
            scope=scope,
        )
        combined = memory_norm.applied + language_norm.applied
        return type(memory_norm)(
            raw_text=memory_norm.raw_text,
            normalized_text=language_norm.normalized_text,
            applied=combined,
            metadata={
                **memory_norm.metadata,
                'language_requested': requested_language,
                'language_observed': observed_language,
                'language_policy': policy_name,
                'script_summary': language_norm.metadata.get('script_summary'),
            },
        )

    def _commit_from_writer(
        self,
        *,
        final_transcript: FinalTranscript,
        writer_outcome: WriterOutcome | None,
    ) -> CommitDecision | None:
        if writer_outcome is None:
            return self.controller.stage_and_maybe_commit(final_transcript, auto_commit=True)
        if writer_outcome.action == WriterActionKind.PASS_THROUGH_COMMIT:
            transcript = FinalTranscript(
                utterance_id=final_transcript.utterance_id,
                text=writer_outcome.output_text,
                start_ms=final_transcript.start_ms,
                end_ms=final_transcript.end_ms,
                backend_name=final_transcript.backend_name,
                model_path=getattr(final_transcript, 'model_path', '') or '',
                language=final_transcript.language,
                decode_ms=final_transcript.decode_ms,
                end_of_turn_latency_ms=final_transcript.end_of_turn_latency_ms,
                metadata={**final_transcript.metadata, 'writer': writer_outcome.to_dict()},
            )
            return self.controller.stage_and_maybe_commit(transcript, auto_commit=True)
        if writer_outcome.action in {WriterActionKind.DIRECT_COMMIT, WriterActionKind.TRANSFORM_COMMIT}:
            return self.controller.commit_text_for_active(
                utterance_id=final_transcript.utterance_id,
                text=writer_outcome.output_text,
                auto_commit=True,
                commit_mode_override=writer_outcome.commit_mode,
                selection_override=writer_outcome.selection_override,
                metadata=writer_outcome.to_dict(),
            )
        if writer_outcome.action in {WriterActionKind.MODE_SWITCH, WriterActionKind.STATE_MUTATION, WriterActionKind.MEMORY_MUTATION, WriterActionKind.NOOP}:
            self.controller.complete_active_without_commit(reason=writer_outcome.action.value, metadata=writer_outcome.to_dict())
            return None
        return self.controller.stage_and_maybe_commit(final_transcript, auto_commit=True)

    def _learn_from_commit(self, *, raw_text: str, committed_text: str, plan: RecognitionBiasPlan) -> None:
        if self.memory_store is None:
            return
        cp_sup = None
        if isinstance(plan.metadata, dict):
            cp = plan.metadata.get('context_plane')
            if isinstance(cp, dict):
                v = cp.get('suppression')
                cp_sup = str(v) if v is not None else None
        seed_layer = self.juno_seed_runtime.seed_layer if self.juno_seed_runtime is not None else None
        if is_durable_memory_suppressed(seed_layer, plan.context, context_plane_suppression=cp_sup):
            self.recorder.record(
                TraceKind.MEMORY,
                'memory_update_suppressed_from_commit',
                {'reason': 'durable_memory_suppressed'},
            )
            return
        self.memory_store.record_correction(raw_text, committed_text)
        if self.juno_seed_runtime is not None:
            self.juno_seed_runtime.promotion.maybe_promote_correction_to_lexicon(
                observed=raw_text,
                corrected=committed_text,
                durable_memory_suppressed=False,
            )
        entities = self.bias_engine.extract_session_entities(committed_text)
        entities.extend(plan.context.candidate_entities[:8])
        self.memory_store.upsert_session_entities(entities, source='commit')
        if self.juno_seed_runtime is not None:
            committed_lower = committed_text.casefold()
            for token in dict.fromkeys(entities):
                t = (token or '').strip()
                if not learned_term_allowed(t) or t.casefold() not in committed_lower:
                    continue
                self.juno_seed_runtime.learned_store.increment_acceptance(
                    t,
                    from_suppressed_context=False,
                )
                self.juno_seed_runtime.promotion.maybe_promote_context_entity_to_lexicon(
                    token=t,
                    durable_memory_suppressed=False,
                )
        self.recorder.record(TraceKind.MEMORY, 'memory_updated_from_commit', {
            'raw_text': raw_text,
            'committed_text': committed_text,
            'entities': entities,
        })

    def _should_learn_from_commit(
        self,
        *,
        raw_text: str,
        committed_text: str,
        final_transcript: FinalTranscript,
        writer_outcome: WriterOutcome | None,
    ) -> bool:
        if self.memory_store is None:
            return False
        text = (committed_text or '').strip()
        if len(text) < 4:
            return False
        words = [token for token in text.replace('\n', ' ').split() if token]
        if len(words) < 2:
            return False
        lowered = [token.casefold().strip('.,!?;:-') for token in words if token.strip('.,!?;:-')]
        if lowered:
            repeated = max(lowered.count(token) for token in set(lowered))
            if repeated >= max(3, int(len(lowered) * 0.5)):
                return False
        alpha_chars = [ch for ch in text if ch.isalpha()]
        if alpha_chars:
            unique_alpha_ratio = len(set(ch.casefold() for ch in alpha_chars)) / max(1, len(alpha_chars))
            if unique_alpha_ratio < 0.18:
                return False
        avg_logprob = final_transcript.metadata.get('avg_logprob')
        if isinstance(avg_logprob, (int, float)) and avg_logprob < -1.15:
            return False
        if writer_outcome is not None and writer_outcome.action in {WriterActionKind.NOOP, WriterActionKind.MODE_SWITCH}:
            return False
        normalized_raw = ' '.join(raw_text.split()).casefold()
        normalized_committed = ' '.join(text.split()).casefold()
        if normalized_raw == normalized_committed:
            # No correction benefit; only learn if we can extract at least one likely entity.
            return bool(self.bias_engine.extract_session_entities(text))
        return True


def _avg(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return mean(nums)


def _bump_counter(counter: dict[str, int], value: str | None) -> None:
    if not value:
        return
    counter[value] = counter.get(value, 0) + 1


def _merge_prompts(*parts: str | None) -> str | None:
    clean: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = (part or '').strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        clean.append(value)
    return ' | '.join(clean) if clean else None


def _language_decision_dict(plan: RecognitionBiasPlan) -> dict:
    value = plan.metadata.get('language_decision', {}) if isinstance(plan.metadata, dict) else {}
    return value if isinstance(value, dict) else {}
