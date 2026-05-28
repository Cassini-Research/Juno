"""Dictation transcription service.

The broker exposes HTTP endpoints (``POST /api/broker/dictation/ingest_wav``
and the OpenAI-compatible ``POST /v1/audio/transcriptions``) for any
surface to submit a recorded WAV and receive a final transcript in one
call. The actual model invocation sits behind a narrow protocol so:

- Production wires a real ASR backend via
  :class:`FinalBackendTranscriber`, which reuses the engine's already-
  loaded final ASR backend.
- Standalone workbench runs resolve a transcriber from environment
  variables via :func:`resolve_transcriber_from_env`.
- Tests pass :class:`StubTranscriber` explicitly to exercise the HTTP
  plumbing without a model.

**Never** silently fall back to a stub in a code path a user could hit.
If a transcriber isn't configured, use :class:`UnavailableTranscriber`,
which returns ``ok=False`` with a structured error code. Pasting a fake
"transcript" into the user's document is worse than a loud failure.
"""

from __future__ import annotations

import io
import os
import re
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class TranscribeUnavailable(RuntimeError):
    """Raised (or wrapped) when no ASR backend is configured.

    Carries a stable ``code`` the HTTP surface uses as a machine-readable
    error key. The Mac shell surfaces this to the user; it must never
    end up in a ``transcript`` field.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TranscribeResult:
    transcript: str
    language: str | None
    backend_name: str
    audio_duration_ms: float
    decode_ms: float
    # Absolute model path or HF repo id the backend actually used.
    # ``""`` when the backend has no notion of a path (the
    # :class:`StubTranscriber` used by tests) — callers that care about
    # provenance must branch on the empty string, not on ``None``.
    model_path: str = ""
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    segments: tuple[Any, ...] = ()

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "language": self.language,
            "backend_name": self.backend_name,
            "audio_duration_ms": self.audio_duration_ms,
            "decode_ms": self.decode_ms,
            "model_path": self.model_path,
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
            "compression_ratio": self.compression_ratio,
            "segments": [seg.to_dict() if hasattr(seg, "to_dict") else seg for seg in self.segments],
        }


class DictationTranscriber(Protocol):
    """Transcribe a single WAV blob end-to-end.

    Implementations MUST raise :class:`TranscribeUnavailable` rather than
    returning a placeholder transcript when no backend is configured.

    ``initial_prompt`` and ``bias_phrases`` are optional decode hints that
    the :class:`~juno_core_v3.dictation.pipeline.OneShotDictationPipeline`
    derives from memory + context. Implementations that cannot honour them
    (e.g. :class:`StubTranscriber`) MUST accept them as kwargs and ignore
    them rather than raising, so the pipeline stays transport-complete.
    """

    backend_name: str

    def transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        bias_phrases: list[str] | None = None,
    ) -> TranscribeResult: ...


# ------------------------------------------------------------------ #
# WAV decoder
# ------------------------------------------------------------------ #

def _decode_wav_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a PCM16 mono/stereo WAV into a 1-D float32 array, plus sample rate.

    The Mac shell records 16 kHz mono PCM16, so the common case is zero
    resampling. If a caller sends stereo we fold down to mono; if it sends
    a sample rate other than 16 kHz we resample naively so the backend
    still gets something sensible. This is intentionally dependency-light.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"unsupported WAV sample width: {sample_width} bytes (need 16-bit PCM)")

    audio_i16 = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio_i16 = audio_i16.reshape(-1, channels).mean(axis=1).astype(np.int16)

    audio = audio_i16.astype(np.float32) / 32768.0

    target_sr = 16_000
    if sample_rate != target_sr:
        # Linear resampling — quality is "fine" for whisper-class models.
        ratio = target_sr / float(sample_rate)
        out_len = int(round(audio.shape[0] * ratio))
        if out_len <= 0:
            audio = np.zeros(0, dtype=np.float32)
        else:
            xp = np.arange(audio.shape[0], dtype=np.float32)
            xq = np.linspace(0.0, audio.shape[0] - 1, out_len, dtype=np.float32)
            audio = np.interp(xq, xp, audio).astype(np.float32)
        sample_rate = target_sr

    return audio, sample_rate


# ------------------------------------------------------------------ #
# Implementations
# ------------------------------------------------------------------ #

@dataclass
class StubTranscriber:
    """Deterministic stub for tests & for workbench runs without a model.

    Returns a canned transcript based on audio duration; the shape of the
    response mirrors the real backend so downstream code is transport-
    complete even when no ASR is wired.
    """

    canned_transcript: str = ""
    backend_name: str = "stub"

    def transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        bias_phrases: list[str] | None = None,
    ) -> TranscribeResult:
        del initial_prompt, bias_phrases  # stub ignores decode hints
        audio, sr = _decode_wav_to_float32(wav_bytes)
        duration_ms = (audio.shape[0] / float(sr)) * 1000.0 if sr else 0.0
        return TranscribeResult(
            transcript=self.canned_transcript,
            language=language,
            backend_name=self.backend_name,
            audio_duration_ms=duration_ms,
            decode_ms=0.0,
        )


@dataclass
class FinalBackendTranscriber:
    """Bridge to :class:`juno_v2.final.backends.base.FinalAsrBackend`.

    Keeps a single ``FinalDecodeRequest`` allocation per call; the backend
    owns its own model state. The caller is responsible for calling
    ``backend.warm()`` out-of-band if cold-start cost is a concern — we
    do lazy warming on first transcribe so the workbench starts fast even
    when a model is configured but never used.
    """

    backend: object  # FinalAsrBackend — kept generic to avoid a hard import cycle.
    language: str | None = None
    # Private state: warm status + a lock to serialize warm across
    # concurrent callers. ``init=False`` keeps these out of the generated
    # __init__ signature so callers only pass ``backend`` and ``language``.
    _warmed: bool = field(init=False, default=False, repr=False)
    _warm_lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False
    )

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "backend_name", "unknown")

    @property
    def model_path(self) -> str:
        """Path / repo id of the wrapped final backend.

        Sourced from ``backend.config.model_path`` when the backend
        follows the :class:`~juno_v2.final.config.FinalAsrConfig` shape,
        falling back to an empty string for backends that don't carry a
        config (tests, synthetic doubles). Callers use this for
        per-utterance trace attribution — never for decode logic.
        """
        config = getattr(self.backend, "config", None)
        return str(getattr(config, "model_path", "") or "")

    def _ensure_warm(self) -> None:
        # Fast path: no lock when already warm. Safe because the only
        # transition is False -> True and the write under the lock uses
        # a memory barrier on CPython (bool assignment).
        if self._warmed:
            return
        with self._warm_lock:
            if self._warmed:
                return
            try:
                self.backend.warm()  # type: ignore[attr-defined]
            except Exception as exc:
                raise TranscribeUnavailable(
                    "asr_backend_warm_failed",
                    f"failed to warm {self.backend_name}: {exc}",
                ) from exc
            self._warmed = True

    def transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        bias_phrases: list[str] | None = None,
    ) -> TranscribeResult:
        from juno_v2.contracts.final import FinalDecodeRequest

        self._ensure_warm()

        audio, sample_rate = _decode_wav_to_float32(wav_bytes)
        duration_ms = (audio.shape[0] / float(sample_rate)) * 1000.0 if sample_rate else 0.0
        req = FinalDecodeRequest(
            utterance_id="mac_oneshot",
            audio=audio,
            sample_rate_hz=sample_rate,
            start_ms=0.0,
            end_ms=duration_ms,
            language=language or self.language,
            allowed_languages=[],
            language_policy="auto",
            initial_prompt=initial_prompt,
            bias_phrases=list(bias_phrases or []),
        )
        result = self.backend.decode(req)  # type: ignore[attr-defined]
        # Prefer the model_path the backend stamped on the decode result
        # (remote backends may echo a different ref than the local
        # config). Fall back to our ``model_path`` property so the wire
        # always carries *something* for provenance.
        resolved_model_path = str(getattr(result, "model_path", "") or "") or self.model_path
        metadata = dict(getattr(result, "metadata", {}) or {})
        return TranscribeResult(
            transcript=result.text,
            language=result.language,
            backend_name=result.backend_name,
            audio_duration_ms=result.audio_duration_ms,
            decode_ms=result.decode_ms,
            model_path=resolved_model_path,
            avg_logprob=_optional_float(metadata.get("avg_logprob")),
            no_speech_prob=_optional_float(metadata.get("no_speech_prob")),
            compression_ratio=_optional_float(metadata.get("compression_ratio")),
            segments=tuple(getattr(result, "segments", ()) or ()),
        )


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class UnavailableTranscriber:
    """Explicit "no ASR configured" transcriber.

    The workbench uses this as the default in code paths where a real
    backend is not present. It *raises* :class:`TranscribeUnavailable`
    so the HTTP layer can translate it into an ``ok=False`` JSON body
    with a stable error code. It never returns a fake transcript string.

    The ``code`` field lets callers distinguish the failure mode:
    ``"asr_backend_not_configured"`` (default — no env / config),
    ``"warming"`` (engine is still loading models — runtime.service
    swaps the real transcriber in once ``warm_all`` completes), or
    any other stable string the caller wants. The Mac shell looks
    at this code to decide whether the broker snapshot failure is a
    setup issue (show install card) or transient (show warming
    card / silently retry).
    """

    reason: str = (
        "No ASR backend configured. Set JUNO_FINAL_MODEL_PATH + "
        "JUNO_FINAL_BACKEND, or run the full stack via scripts/run_live.sh "
        "so the workbench can reuse the loaded model."
    )
    backend_name: str = "unavailable"
    code: str = "asr_backend_not_configured"

    def transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        bias_phrases: list[str] | None = None,
    ) -> TranscribeResult:
        del language, initial_prompt, bias_phrases
        raise TranscribeUnavailable(self.code, self.reason)


# ------------------------------------------------------------------ #
# Env resolver for standalone workbench runs without the full runtime stack.
# ------------------------------------------------------------------ #

# Heuristic: HF repo ids are ``org/name`` with a single slash, no path
# separators beyond that, and never start with ``/``. This catches
# ``mlx-community/whisper-large-v3-turbo`` while letting absolute paths
# (``/Users/...``) and relative paths (``./model``) fall through unchanged.
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")


def _resolve_model_path(raw: str) -> str:
    """Return a local filesystem path for ``raw``.

    If ``raw`` is already an existing path, returns it unchanged. If it
    looks like an HF repo id we ask ``snapshot_download`` for the cached
    snapshot directory **without contacting the network**
    (``local_files_only=True``). On a cache hit we return that path; on a
    miss (or any other error) we return the repo id unchanged so the
    caller can decide what to do — typically schedule a background
    pre-warm that does the full download while ``/healthz`` reports a
    warming state.

    Returning the bare repo id on miss is safe for backends that accept
    HF identifiers natively (mlx_whisper, faster_whisper). For backends
    that strictly require a path, the caller's existing
    ``Path(...).exists()`` check will fire a clear "does not exist"
    error rather than block the workbench startup on a multi-GB pull.
    """
    if not raw:
        return raw
    if Path(raw).exists():
        return raw
    if not _HF_REPO_ID.match(raw):
        return raw
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        return snapshot_download(raw, local_files_only=True)
    except Exception:
        # Cache miss, network issue, or huggingface_hub import failure —
        # any of these mean we can't resolve synchronously. Return the
        # original repo id and let the caller schedule a pre-warm.
        return raw


def is_hf_repo_id(raw: str) -> bool:
    """``True`` when ``raw`` looks like an HF ``org/name`` identifier and
    isn't an existing local path. Used by the workbench to decide whether
    to schedule a pre-warm thread."""
    if not raw:
        return False
    if Path(raw).exists():
        return False
    return bool(_HF_REPO_ID.match(raw))


def resolve_transcriber_from_env(env: dict[str, str] | None = None) -> DictationTranscriber:
    """Build a transcriber from environment variables.

    Recognised vars (all optional; missing required ones -> unavailable
    transcriber with a helpful error):

    - ``JUNO_FINAL_BACKEND``: ``faster_whisper`` | ``mlx_whisper`` |
      ``local_http_json``.
    - ``JUNO_FINAL_MODEL_PATH``: required for ``faster_whisper`` and
      ``mlx_whisper``. Accepts either a local filesystem path or a
      Hugging Face repo id (e.g. ``mlx-community/whisper-large-v3-turbo``);
      repo ids are resolved via ``huggingface_hub.snapshot_download`` so
      callers don't have to pre-resolve the cache layout.
    - ``JUNO_FINAL_ENDPOINT``: required for ``local_http_json``.
    - ``JUNO_FINAL_DEVICE``, ``JUNO_FINAL_COMPUTE_TYPE``: optional
      tuning for ``faster_whisper``.
    - ``JUNO_FINAL_LANGUAGE``: preset language (e.g. ``"en"``).

    For backwards compatibility every var is also recognised under the
    ``JUNO_V2_FINAL_*`` prefix used by ``juno_v2.runtime.deployment``.
    The unprefixed forms win when both are set; the V2 forms exist so
    bundled-engine launchers (``run_engine.sh``) and tests configured for
    the production runtime path don't have to duplicate env vars when
    pointed at standalone workbench mode. See PR #28's followup notes.

    The function does **not** load models eagerly — the backend is
    constructed but ``warm()`` is deferred until the first transcribe.
    That keeps workbench startup cheap even when the user doesn't plan
    to use the endpoint.
    """
    e = dict(os.environ if env is None else env)

    def _lookup(*names: str) -> str:
        for n in names:
            v = e.get(n)
            if v:
                return v
        return ""

    backend = _lookup("JUNO_FINAL_BACKEND", "JUNO_V2_FINAL_BACKEND").strip().lower()
    if not backend:
        return UnavailableTranscriber()

    try:
        from juno_v2.final.config import FinalAsrConfig
    except Exception as exc:  # pragma: no cover — juno_v2 must be importable
        return UnavailableTranscriber(
            reason=f"juno_v2 import failed while resolving transcriber: {exc}"
        )

    raw_model_path = _lookup("JUNO_FINAL_MODEL_PATH", "JUNO_V2_FINAL_MODEL_PATH")
    endpoint = _lookup("JUNO_FINAL_ENDPOINT", "JUNO_V2_FINAL_ENDPOINT") or None
    device = _lookup("JUNO_FINAL_DEVICE", "JUNO_V2_FINAL_DEVICE") or "auto"
    compute_type = _lookup("JUNO_FINAL_COMPUTE_TYPE", "JUNO_V2_FINAL_COMPUTE_TYPE") or "default"
    language = _lookup("JUNO_FINAL_LANGUAGE", "JUNO_V2_FINAL_LANGUAGE") or None
    model_path = _resolve_model_path(raw_model_path) if raw_model_path else ""

    needs_path = backend in {"faster_whisper", "mlx_whisper"}
    if needs_path:
        if not model_path:
            return UnavailableTranscriber(
                reason=f"JUNO_FINAL_BACKEND={backend} requires JUNO_FINAL_MODEL_PATH"
            )
        if not Path(model_path).exists():
            return UnavailableTranscriber(
                reason=f"JUNO_FINAL_MODEL_PATH does not exist: {model_path}"
            )
    if backend == "local_http_json" and not endpoint:
        return UnavailableTranscriber(
            reason="JUNO_FINAL_BACKEND=local_http_json requires JUNO_FINAL_ENDPOINT"
        )

    config = FinalAsrConfig(
        backend_name=backend,
        model_path=model_path,
        local_http_endpoint=endpoint,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    try:
        if backend == "faster_whisper":
            from juno_v2.final.backends.faster_whisper import FasterWhisperFinalBackend
            ff = FasterWhisperFinalBackend(config)
        elif backend == "mlx_whisper":
            from juno_v2.final.backends.mlx_whisper import MlxWhisperFinalBackend
            ff = MlxWhisperFinalBackend(config)
        elif backend == "local_http_json":
            from juno_v2.final.backends.local_http_json import LocalHttpJsonFinalBackend
            ff = LocalHttpJsonFinalBackend(config)
        else:
            return UnavailableTranscriber(
                reason=f"JUNO_FINAL_BACKEND={backend!r} is not a known backend"
            )
    except Exception as exc:
        return UnavailableTranscriber(
            reason=f"failed to construct {backend} backend: {exc}"
        )

    return FinalBackendTranscriber(backend=ff, language=language)


__all__ = [
    "DictationTranscriber",
    "FinalBackendTranscriber",
    "StubTranscriber",
    "TranscribeResult",
    "TranscribeUnavailable",
    "UnavailableTranscriber",
    "resolve_transcriber_from_env",
]
