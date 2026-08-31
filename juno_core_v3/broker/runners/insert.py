"""Insert-class session runner.

An Insert session takes audio (a WAV buffer) and commits the resulting
transcript into the active field. Before B2, the broker plane had no
``InsertRunner`` class at all — the ``runners/`` module documented one
but callers reached directly into
:class:`~juno_core_v3.dictation.pipeline.OneShotDictationPipeline`.
That left the broker plane asymmetric: ``SessionKind.TRANSFORM`` had a
typed runner while ``SessionKind.INSERT``
was a shape-less passthrough.

This module closes that gap. ``InsertRunner`` is a thin adapter over
the already-v3 :class:`OneShotDictationPipeline`:

* It accepts a broker-shaped :class:`InsertRequest` (not the pipeline's
  keyword-soup ``run(wav_bytes, *, language=..., app_bundle_id=..., ...)``).
* It applies the single broker-policy decision the workbench already
  makes today — ``reduce_optional_lanes`` temporarily disables the
  writer service so we fall back to the deterministic writer when the
  host is memory-pressured.
* It returns a broker-shaped :class:`InsertResult` whose ``to_dict()``
  is a passthrough of the underlying pipeline result. That keeps the
  HTTP / trace wire format identical to the pre-B2 surface, so every
  existing caller keeps working.

What it *doesn't* do:

* It does **not** own the live-streaming engine
  (:class:`~juno_v2.engine.session.DictationSessionRunner`). The
  streaming engine has its own loop around VAD, the audio ring buffer,
  Metal serialisation, writer, commit, memory, trace — lifting it is
  a much larger concern (see B2-maximal in the diagram-vs-code audit).
* It does **not** create or teardown the ``OneShotDictationPipeline``.
  Callers inject one. This keeps the runner test-friendly and avoids
  hiding the pipeline's 15-field construction inside the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from juno_core_v3.dictation.pipeline import (
        OneShotDictationPipeline,
        OneShotDictationResult,
    )


# ---------------------------------------------------------------- #
# Wire contracts
# ---------------------------------------------------------------- #


@dataclass(slots=True)
class InsertRequest:
    """Input to :meth:`InsertRunner.run`.

    ``wav_bytes`` is a full RIFF WAV payload (16 kHz mono, signed 16-bit
    PCM) — the same contract the pipeline accepts today. Every other
    field has a safe default so simple callers can say
    ``InsertRequest(wav_bytes=…)`` and get sane behaviour.

    ``reduce_optional_lanes`` is the one broker-policy lever the runner
    honours today. When set, the writer service is temporarily detached
    from the pipeline so the run falls back to the deterministic writer.
    This matches the existing semantics of the workbench's
    ``_run_oneshot_with_policy`` helper — the policy decision itself
    still lives with the caller (they have the full policy dict); the
    runner only enacts the resolved flag.
    """

    wav_bytes: bytes
    language: str | None = None
    app_bundle_id: str | None = None
    window_title_hint: str | None = None
    utterance_id: str | None = None
    manual_writer_mode: str | None = None
    custom_writer_mode: str | None = None
    reduce_optional_lanes: bool = False
    # Client-frozen juno-capability JSON (macOS shell at hotkey press).
    # When set, :class:`OneShotDictationPipeline` merges into the server
    # snapshot before the context plane.
    frozen_context: dict[str, Any] | None = None
    save_history: bool = True
    save_audio: bool = True
    transcript_stage: str = "final_delivery"
    session_context_tape: dict[str, Any] | list[Any] | None = None
    transcript_hint: str | None = None
    language_mode: str | None = None
    shell_timeline: dict[str, Any] | None = None
    # User-facing pause-sensitivity slider value (seconds of trailing silence
    # before the speech state machine declares an utterance done). ``None``
    # means "use the speech-profile default" — required for back-compat with
    # callers / older shells that never pass the field. The pipeline / engine
    # converts seconds → ``pause_trigger_frames`` via
    # ``juno_v2.speech.config.pause_trigger_frames_for_seconds``.
    pause_sensitivity_seconds: float | None = None
    # Provenance fields — not consumed by the pipeline today, but
    # carried on the request so a future migration can stamp them into
    # the trace without a contract change.
    broker_session_id: str | None = None
    surface_id: str | None = None
    # Optional opaque correlation tags supplied only by reliability harnesses.
    # Normal product requests leave both unset.
    test_run_id: str | None = None
    test_case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InsertResult:
    """Output of :meth:`InsertRunner.run`.

    Wraps the underlying :class:`OneShotDictationResult` so callers can
    reach typed broker-shaped fields directly (``ok``, ``transcript``,
    ``backend_name``, ``utterance_id``, ``error``) without fishing
    through ``to_dict()``. For every other field, delegate to the raw
    result — the workbench path keeps its existing wire format.
    """

    raw: "OneShotDictationResult"

    @property
    def ok(self) -> bool:
        return bool(self.raw.ok)

    @property
    def utterance_id(self) -> str:
        return self.raw.utterance_id

    @property
    def transcript(self) -> str:
        return self.raw.transcript or ""

    @property
    def backend_name(self) -> str:
        return self.raw.backend_name or ""

    @property
    def language(self) -> str | None:
        return self.raw.language

    @property
    def error(self) -> str | None:
        return self.raw.error

    @property
    def error_code(self) -> str | None:
        return self.raw.error_code

    def to_dict(self) -> dict[str, Any]:
        """Return the wire-format dict the HTTP / trace layer consumes.

        Intentionally a passthrough to the wrapped pipeline result so
        the pre-B2 workbench response shape is preserved byte-for-byte.
        """
        return self.raw.to_dict()


# ---------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------- #


@dataclass
class InsertRunner:
    """Broker-plane runner for :class:`~juno_core_v3.contracts.session.SessionKind.INSERT`.

    Injected with an :class:`OneShotDictationPipeline` the caller already
    configured (transcriber, context, memory, writer, capability gate,
    audio retention, …). The runner never builds or tears down the
    pipeline; it only drives a single ``run`` through it with
    broker-shaped inputs and outputs.
    """

    pipeline: "OneShotDictationPipeline"

    def run(self, req: InsertRequest) -> InsertResult:
        """Execute a one-shot Insert session.

        Honours ``reduce_optional_lanes`` by temporarily detaching the
        pipeline's writer service for the duration of the call. The
        original writer is restored in a ``finally`` block so an
        exception during ``pipeline.run`` never strands the pipeline
        in a writerless state.
        """
        restore_writer = False
        original_writer = None
        degraded_lane = bool(req.reduce_optional_lanes and self.pipeline.writer_service is not None)
        if degraded_lane:
            original_writer = self.pipeline.writer_service
            self.pipeline.writer_service = None
            restore_writer = True
        try:
            kwargs = {
                "language": req.language,
                "app_bundle_id": req.app_bundle_id,
                "window_title_hint": req.window_title_hint,
                "utterance_id": req.utterance_id,
                "manual_writer_mode": req.manual_writer_mode,
                "custom_writer_mode": req.custom_writer_mode,
                "frozen_context": req.frozen_context,
                "degraded_writer_lane": degraded_lane,
                "save_history": req.save_history,
                "save_audio": req.save_audio,
                "transcript_stage": req.transcript_stage,
                "session_context_tape": req.session_context_tape,
                "transcript_hint": req.transcript_hint,
                "language_mode": req.language_mode,
                "shell_timeline": req.shell_timeline,
                "pause_sensitivity_seconds": req.pause_sensitivity_seconds,
                "test_run_id": req.test_run_id,
                "test_case_id": req.test_case_id,
            }
            raw = self.pipeline.run(req.wav_bytes, **self._accepted_run_kwargs(kwargs))
        finally:
            if restore_writer:
                self.pipeline.writer_service = original_writer
        return InsertResult(raw=raw)

    def _accepted_run_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            sig = inspect.signature(self.pipeline.run)
        except (TypeError, ValueError):
            return kwargs
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}


__all__ = ["InsertRequest", "InsertResult", "InsertRunner"]
