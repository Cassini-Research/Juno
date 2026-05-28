from __future__ import annotations

import sys
import time
from pathlib import Path
import gc
import inspect
import threading
from typing import Any, TYPE_CHECKING

import numpy as np

from juno_v2.contracts.final import FinalDecodeRequest, FinalDecodeResult, FinalSegment
from juno_v2.contracts.tracing import TraceKind
from juno_v2.final.backends.base import FinalAsrBackend
from juno_v2.final.config import FinalAsrConfig
from juno_v2.runtime.mlx_lock import mlx_decode_guard

if TYPE_CHECKING:
    from juno_v2.observability.tracing import TraceRecorder


class MlxWhisperFinalBackend(FinalAsrBackend):
    """Apple Silicon-native final ASR backend using mlx-whisper."""

    backend_name = "mlx_whisper"

    def __init__(self, config: FinalAsrConfig) -> None:
        self.config = config
        self._mlx_whisper = None
        self._loaded_model = None
        self._model_thread_id: int | None = None
        self._model_lock = threading.RLock()
        self._model_ref: str | None = None  # resolved path or HF repo ID used at runtime
        self._model_source_type: str | None = None  # local_path | hf_repo | unknown
        self._warm_logged = False
        self._tracer: TraceRecorder | None = None
        # Sub-stage timings captured during warm() so the service layer can
        # attribute cold-start cost to either the MLX module import or the
        # actual model load (which may include an HF hub download on first run).
        self._warm_timings: dict[str, float] = {}

    def warm_timings(self) -> dict[str, float]:
        """Return a copy of the last warm() sub-stage timings.

        Keys:
          - ``final_backend_module_import_ms``: time to import ``mlx_whisper``.
          - ``final_backend_load_ms``: time for ``load_models.load_model()``,
            which on a cold cache includes downloading the model artifacts from
            the HuggingFace hub.
        """
        return dict(self._warm_timings)

    def set_tracer(self, tracer: TraceRecorder | None) -> None:
        """Route structured warm/unload events through this TraceRecorder.

        When no tracer is attached the backend falls back to stdout so cold
        starts remain visible to operators running the service by hand.
        """
        self._tracer = tracer

    def _emit_warm_event(self, name: str, payload: dict[str, Any]) -> None:
        if self._tracer is not None:
            try:
                self._tracer.record(TraceKind.ASR_FINAL, name, payload)
                return
            except Exception:
                pass
        # Tracer unavailable — fall back to stdout so operators running
        # without structured observability still see the event.
        rendered = " ".join(f"{k}={v}" for k, v in payload.items())
        print(f"[juno_v2][final][mlx_whisper] {name} {rendered}".rstrip())

    @staticmethod
    def _has_mlx_weights(model_path: Path) -> bool:
        """Return True only when the directory contains native MLX weight files."""
        return (model_path / "weights.safetensors").exists() or (model_path / "weights.npz").exists()

    def _resolve_model_ref(self) -> tuple[str, str]:
        """Return (ref, source_type) to hand to mlx_whisper.

        mlx_whisper.load_models treats its argument as a local path first; if
        the path does not exist it falls back to snapshot_download.  We use the
        HF repo ID when the pre-downloaded directory does not contain the native
        MLX weight files (weights.safetensors / weights.npz) to let mlx_whisper
        download the correct format into its own HF-hub cache on first use.
        """
        model_path = Path(self.config.model_path)
        if model_path.exists() and self._has_mlx_weights(model_path):
            return (str(model_path), "local_path")
        # Local path missing MLX weights — use HF repo ID so mlx_whisper
        # handles download/caching itself.
        if self.config.hf_repo_id:
            return (self.config.hf_repo_id, "hf_repo")
        # No repo ID available; fall back to local path and let mlx_whisper
        # surface a meaningful error.
        return (str(model_path), "unknown")

    @staticmethod
    def _patch_model_dimensions(mlx_whisper_module: object) -> None:  # type: ignore[type-arg]
        """Strip unknown config keys before they reach ModelDimensions(**config).

        mlx-community model artifacts include extra fields (alignment_heads,
        lang_ids, suppress_ids, …) in config.json alongside the 10 architecture
        dimension fields.  mlx_whisper.load_models passes the full dict to
        ModelDimensions(**config), which is a strict dataclass and rejects
        unexpected kwargs.  This one-time patch drops unknown keys silently.
        """
        try:
            w = mlx_whisper_module.whisper  # type: ignore[attr-defined]
            if getattr(w.ModelDimensions, '_juno_compat_patched', False):
                return
            orig_init = w.ModelDimensions.__init__
            sig = inspect.signature(orig_init)
            allowed = frozenset(
                name
                for name, param in sig.parameters.items()
                if name != 'self' and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            )

            def _tolerant_init(self_inner: object, *args: object, **kwargs: object) -> None:
                if args:
                    # Preserve positional calling semantics.
                    orig_init(self_inner, *args, **kwargs)
                    return
                if kwargs:
                    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
                orig_init(self_inner, **kwargs)

            w.ModelDimensions.__init__ = _tolerant_init  # type: ignore[method-assign]
            w.ModelDimensions._juno_compat_patched = True  # type: ignore[attr-defined]
        except Exception:
            pass

    @staticmethod
    def _transcribe_fp16_dtype(mlx_whisper_module: object) -> Any | None:
        """Return the dtype used by ``mlx_whisper.transcribe(fp16=True)``.

        ``mlx_whisper.transcribe`` defaults to fp16 and looks up its model
        through a global ``ModelHolder``. When we warm and seed that holder,
        the preloaded model must use the same dtype or newer mlx-whisper
        versions reject the encoder output at decode time.
        """
        transcribe_fn = getattr(mlx_whisper_module, "transcribe", None)
        globals_dict = getattr(transcribe_fn, "__globals__", None)
        if globals_dict is None:
            globals_dict = getattr(getattr(transcribe_fn, "__func__", None), "__globals__", None)
        mx_module = globals_dict.get("mx") if isinstance(globals_dict, dict) else None
        dtype = getattr(mx_module, "float16", None)
        if dtype is not None:
            return dtype
        try:
            import mlx.core as mx  # type: ignore

            return mx.float16
        except Exception:
            return None

    def warm(self) -> None:
        with self._model_lock:
            if self._mlx_whisper is not None:
                return
            self._warm_timings = {}
            import_started = time.perf_counter_ns()
            try:
                import mlx_whisper  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "mlx-whisper is required for the MLX final backend on Apple Silicon; install project dependencies first"
                ) from exc
            self._warm_timings['final_backend_module_import_ms'] = (time.perf_counter_ns() - import_started) / 1_000_000.0
            self._patch_model_dimensions(mlx_whisper)
            self._mlx_whisper = mlx_whisper
            model_ref, source_type = self._resolve_model_ref()
            self._model_ref = model_ref
            self._model_source_type = source_type
            if not self._warm_logged:
                # Operator truth: log exactly what we are handing to mlx_whisper.
                self._emit_warm_event(
                    "warm_start",
                    {"model_source_type": source_type, "model_ref": model_ref},
                )
                self._warm_logged = True
            try:
                from mlx_whisper import load_models  # type: ignore
                load_started = time.perf_counter_ns()
                load_kwargs: dict[str, Any] = {"path_or_hf_repo": model_ref}
                model_dtype = self._transcribe_fp16_dtype(mlx_whisper)
                if model_dtype is not None:
                    load_kwargs["dtype"] = model_dtype
                self._loaded_model = load_models.load_model(**load_kwargs)
                self._model_thread_id = threading.get_ident()
                # Issue #3: ``mlx_whisper.transcribe`` resolves the model via
                # its own class-level ``ModelHolder`` cache (keyed on
                # path_or_hf_repo). Without seeding that cache, the very
                # first ``decode()`` call would trigger a SECOND
                # ``load_model`` — defeating the whole point of warm() and
                # leaving ``self._loaded_model`` orphaned. Seed the cache
                # with the model we just loaded so every subsequent
                # transcribe() hits the cache and reuses our resident copy.
                try:
                    from mlx_whisper.transcribe import ModelHolder  # type: ignore
                    ModelHolder.model = self._loaded_model
                    ModelHolder.model_path = model_ref
                except Exception:
                    # If the upstream layout changes, skip the seed silently
                    # — correctness still holds (load on first decode), only
                    # the latency win is lost.
                    pass
                self._warm_timings['final_backend_load_ms'] = (time.perf_counter_ns() - load_started) / 1_000_000.0
                self._emit_warm_event(
                    "warm_ok",
                    {
                        "model_ref": model_ref,
                        **self._warm_timings,
                    },
                )
            except Exception as exc:
                self._loaded_model = None
                self._model_thread_id = None
                raise RuntimeError(
                    f"MLX Whisper final lane failed to load model (source_type={source_type}, ref={model_ref}): {type(exc).__name__}: {exc}"
                ) from exc

    def _ensure_warm_on_current_thread(self) -> None:
        """Ensure cached MLX arrays belong to the decode thread.

        MLX streams are thread-local. A service can warm the backend on
        its startup thread, while Juno broker requests decode on socket
        handler threads. Reusing the startup-thread model there raises
        ``RuntimeError: There is no Stream(gpu, N) in current thread``.
        """
        self.warm()
        if self._model_thread_id == threading.get_ident():
            return
        with self._model_lock:
            if self._model_thread_id == threading.get_ident():
                return
            self.unload()
            self.warm()

    def decode(self, req: FinalDecodeRequest) -> FinalDecodeResult:
        self._ensure_warm_on_current_thread()
        assert self._mlx_whisper is not None
        model_ref = self._model_ref or self._resolve_model_ref()[0]
        audio = np.asarray(req.audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        audio_ms = (audio.shape[0] / max(req.sample_rate_hz, 1)) * 1000.0
        print(
            f"[ASR_FINAL]   decode_start utt={req.utterance_id[:8]} "
            f"audio_ms={audio_ms:.0f} sr={req.sample_rate_hz} "
            f"lang={req.language or self.config.language} model={model_ref}",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        # Issue #16: mlx_whisper.transcribe accepts (str | np.ndarray |
        # mx.array) for its first positional arg. Pre-fix this code wrote
        # the audio to a tempfile WAV every decode (~10-30ms of disk I/O
        # plus APFS variance). Pass the float32 ndarray directly — same
        # data path the upstream library would take after reading the
        # WAV back. Audio is already clipped to [-1, 1] above.
        with mlx_decode_guard():
            result = self._mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=model_ref,
                language=req.language or self.config.language,
                initial_prompt=req.initial_prompt or self.config.initial_prompt,
                fp16=True,
                word_timestamps=False,
                condition_on_previous_text=self.config.condition_on_previous_text,
            )
        decode_ms = (time.perf_counter() - started) * 1000.0
        text = str(result.get('text', '')).strip()
        detected_language = result.get('language')
        segments_raw = result.get('segments') or []
        segments: list[FinalSegment] = []
        # Capture per-segment confidence/silence signals so the commit-side
        # hallucination guard can distinguish real speech from
        # whisper-on-silence one-shots like "Thank you." (avg_logprob and
        # no_speech_prob are produced per segment by mlx_whisper; we average
        # them here and weight by segment duration so a long real segment
        # outweighs a short hallucinated trailer).
        weighted_logprob = 0.0
        weighted_no_speech = 0.0
        weighted_compression = 0.0
        total_weight_s = 0.0
        for seg in segments_raw:
            seg_text = str(seg.get('text', '')).strip()
            if not seg_text:
                continue
            start_s = float(seg.get('start', 0.0) or 0.0)
            end_s = float(seg.get('end', 0.0) or 0.0)
            seg_avg_logprob = seg.get('avg_logprob')
            seg_no_speech = seg.get('no_speech_prob')
            seg_compression = seg.get('compression_ratio')
            # Per-segment audio-side signals are required by the commit-side
            # trailing-silence-hallucination guard to corroborate stock-phrase
            # matches on individual tail segments (a hallucinated "Thank you."
            # appended to real text has its own high no_speech_prob even when
            # the utterance-level aggregate looks confident).
            try:
                seg_no_speech_float = float(seg_no_speech) if seg_no_speech is not None else None
            except (TypeError, ValueError):
                seg_no_speech_float = None
            try:
                seg_avg_logprob_float = float(seg_avg_logprob) if seg_avg_logprob is not None else None
            except (TypeError, ValueError):
                seg_avg_logprob_float = None
            try:
                seg_compression_float = float(seg_compression) if seg_compression is not None else None
            except (TypeError, ValueError):
                seg_compression_float = None
            segments.append(
                FinalSegment(
                    start_ms=req.start_ms + (start_s * 1000.0),
                    end_ms=req.start_ms + (end_s * 1000.0),
                    text=seg_text,
                    no_speech_prob=seg_no_speech_float,
                    avg_logprob=seg_avg_logprob_float,
                    compression_ratio=seg_compression_float,
                )
            )
            weight = max(end_s - start_s, 0.05)
            try:
                if seg_avg_logprob is not None:
                    weighted_logprob += float(seg_avg_logprob) * weight
                if seg_no_speech is not None:
                    weighted_no_speech += float(seg_no_speech) * weight
                if seg_compression is not None:
                    weighted_compression += float(seg_compression) * weight
                total_weight_s += weight
            except (TypeError, ValueError):
                continue
        avg_logprob = (weighted_logprob / total_weight_s) if total_weight_s > 0 else None
        no_speech_prob = (weighted_no_speech / total_weight_s) if total_weight_s > 0 else None
        compression_ratio = (weighted_compression / total_weight_s) if total_weight_s > 0 else None
        text_preview = (text or "").replace("\n", " ")[:120]
        print(
            f"[ASR_FINAL]   decode_done  utt={req.utterance_id[:8]} "
            f"decode_ms={decode_ms:.1f} chars={len(text or '')} "
            f"segs={len(segments)} no_speech={no_speech_prob} "
            f"avg_logprob={avg_logprob} text={text_preview!r}",
            file=sys.stderr,
            flush=True,
        )
        return FinalDecodeResult(
            utterance_id=req.utterance_id,
            text=text,
            start_ms=req.start_ms,
            end_ms=req.end_ms,
            audio_duration_ms=req.audio_duration_ms,
            backend_name=self.backend_name,
            model_path=model_ref or str(self.config.model_path or ""),
            language=detected_language,
            decode_ms=decode_ms,
            end_of_turn_latency_ms=decode_ms,
            segments=segments,
            metadata={
                'requested_language': req.language or self.config.language,
                'allowed_languages': list(req.allowed_languages),
                'language_policy': req.language_policy,
                'mlx_model_path': str(self.config.model_path),
                'mlx_model_ref': model_ref,
                'mlx_model_source_type': self._model_source_type,
                'avg_logprob': avg_logprob,
                'no_speech_prob': no_speech_prob,
                'compression_ratio': compression_ratio,
                # Issue #17: required by the silence-hallucination guard in
                # commit/controller.py to corroborate avg_logprob with audio
                # length when the decoded text is a clean one-shot like
                # "Thank you." on silent audio.
                'audio_duration_ms': req.audio_duration_ms,
            },
        )


    def unload(self) -> None:
        with self._model_lock:
            self._loaded_model = None
            self._mlx_whisper = None
            self._model_thread_id = None
            try:
                from mlx_whisper.transcribe import ModelHolder  # type: ignore

                ModelHolder.model = None
                ModelHolder.model_path = None
            except Exception:
                pass
            try:
                import mlx.core as mx  # type: ignore
                mx.clear_cache()
            except Exception:
                pass
            gc.collect()
