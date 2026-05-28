"""Qwen3-ASR final backend (Apple Silicon, MLX).

Wraps the ``moona3k/mlx-qwen3-asr`` Python port (PyPI: ``mlx-qwen3-asr``).
Feeds the 1.7B model by default; the 0.6B variant drives the streaming
preview backend instead.

Why a dedicated backend rather than reusing ``MlxWhisperFinalBackend``:
the Qwen3 port exposes a ``Session`` API that owns the model + tokenizer
lifecycle explicitly (no hidden globals) and accepts numpy float32 audio
directly — no temp-WAV round-trip, no ``ModelDimensions`` compatibility
shim. That simpler surface is what lets this backend stay a thin mapping
between :class:`~juno_v2.contracts.final.FinalDecodeRequest` and the
port's ``TranscriptionResult``.

Metal serialisation still goes through
:func:`~juno_v2.runtime.mlx_lock.mlx_decode_guard` so Qwen3 Final and
any other MLX user (MLX Whisper final, MLX LM writer, MLX preview)
can't stomp on each other's Metal command buffers.

Runtime gate: Apple Silicon only (``sys_platform == 'darwin' and
platform_machine == 'arm64'``). The ``mlx-qwen3-asr`` dep is declared
with the same gate in ``pyproject.toml``, so a Linux / CI install never
pulls the wheel. Importing this module on the wrong platform is still
safe — we defer the ``mlx_qwen3_asr`` import to ``warm()`` and surface
an actionable error there.
"""
from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np

from juno_v2.contracts.final import FinalDecodeRequest, FinalDecodeResult, FinalSegment
from juno_v2.final.backends.base import FinalAsrBackend
from juno_v2.final.config import FinalAsrConfig
from juno_v2.runtime.mlx_lock import mlx_decode_guard


class QwenAsrFinalBackend(FinalAsrBackend):
    """Apple-Silicon-native final ASR backend using the MLX Qwen3-ASR port."""

    backend_name = "qwen_asr"

    # Default to the 1.7B model for the final slot. The 0.6B variant is
    # faster but has higher WER; it's used by the streaming preview
    # backend instead.
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"

    def __init__(self, config: FinalAsrConfig) -> None:
        self.config = config
        self._mlx_qwen3_asr: Any = None
        self._session: Any = None
        self._model_ref: str | None = None
        self._warm_logged = False

    # ------------------------------------------------------------------
    # Model-ref resolution
    # ------------------------------------------------------------------

    def _resolve_model_ref(self) -> str:
        """Pick the Qwen3-ASR model id to load.

        Precedence:
        1. ``config.hf_repo_id`` — operator-specified HF repo (matches
           the registry-defaults convention).
        2. ``config.model_path`` — legacy field; if it looks like an HF
           repo id (``owner/name``) we use it, otherwise fall back.
        3. ``self._DEFAULT_MODEL`` — Qwen3-ASR-1.7B.
        """
        if self.config.hf_repo_id:
            return self.config.hf_repo_id
        model_path = str(self.config.model_path or "").strip()
        if "/" in model_path and not model_path.startswith("."):
            return model_path
        return self._DEFAULT_MODEL

    # ------------------------------------------------------------------
    # Warm / unload
    # ------------------------------------------------------------------

    def warm(self) -> None:
        if self._session is not None:
            return
        try:
            import mlx_qwen3_asr  # type: ignore
        except Exception as exc:  # pragma: no cover - platform-gated dep
            raise RuntimeError(
                "mlx-qwen3-asr is required for the Qwen3 final backend on "
                "Apple Silicon; install project deps (`pip install -e .`) "
                "on an arm64 macOS machine"
            ) from exc
        self._mlx_qwen3_asr = mlx_qwen3_asr
        model_ref = self._resolve_model_ref()
        self._model_ref = model_ref
        if not self._warm_logged:
            # Operator truth: log exactly what we hand to the port so
            # runtime doctor can attribute a transcript to a checkpoint.
            print(f"[juno_v2][final][qwen_asr] model_ref={model_ref}")
            self._warm_logged = True
        try:
            self._session = mlx_qwen3_asr.Session(model=model_ref)
        except Exception as exc:
            self._session = None
            raise RuntimeError(
                f"Qwen3-ASR final lane failed to load model "
                f"(ref={model_ref}): {type(exc).__name__}: {exc}"
            ) from exc
        print("[juno_v2][final][qwen_asr] warm=ok")

    def unload(self) -> None:
        self._session = None
        self._mlx_qwen3_asr = None
        try:
            import mlx.core as mx  # type: ignore

            mx.clear_cache()
        except Exception:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, req: FinalDecodeRequest) -> FinalDecodeResult:
        self.warm()
        assert self._session is not None
        model_ref = self._model_ref or self._resolve_model_ref()

        audio = np.asarray(req.audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        # Clip to the port's expected range; callers occasionally hand us
        # audio with transient peaks above 1.0 (e.g. from a hot mic).
        audio = np.clip(audio, -1.0, 1.0)

        # Language hint — the port accepts a plain language name
        # ("English", "Chinese", ...) or the short ISO-ish form
        # ("en", "zh"). We forward whatever the request has verbatim so
        # the canonicaliser inside the port can do its job; None (the
        # request's default) asks the model to auto-detect.
        requested_language = req.language or self.config.language
        started = time.perf_counter()
        with mlx_decode_guard():
            result = self._session.transcribe(
                audio,
                language=requested_language,
                return_chunks=True,
            )
        decode_ms = (time.perf_counter() - started) * 1000.0

        text = str(getattr(result, "text", "") or "").strip()
        detected_language = getattr(result, "language", None)

        # Map chunk-level timestamps (port's "chunks" attribute, which is
        # a list of {"text": ..., "timestamp": [start_s, end_s]}) into
        # our FinalSegment shape. Some releases expose .segments instead,
        # so we check both and prefer chunks when present.
        segments = self._extract_segments(result, req)

        return FinalDecodeResult(
            utterance_id=req.utterance_id,
            text=text,
            start_ms=req.start_ms,
            end_ms=req.end_ms,
            audio_duration_ms=req.audio_duration_ms,
            backend_name=self.backend_name,
            model_path=model_ref or "",
            language=detected_language,
            decode_ms=decode_ms,
            end_of_turn_latency_ms=decode_ms,
            segments=segments,
            metadata={
                "requested_language": requested_language,
                "allowed_languages": list(req.allowed_languages),
                "language_policy": req.language_policy,
                "qwen_asr_model_ref": model_ref,
            },
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_segments(
        self, result: Any, req: FinalDecodeRequest
    ) -> list[FinalSegment]:
        segments: list[FinalSegment] = []
        chunks = getattr(result, "chunks", None) or []
        if chunks:
            for chunk in chunks:
                seg_text = str(chunk.get("text", "") or "").strip()
                if not seg_text:
                    continue
                ts = chunk.get("timestamp") or (0.0, 0.0)
                try:
                    start_s = float(ts[0] or 0.0)
                    end_s = float(ts[1] or 0.0)
                except (TypeError, ValueError, IndexError):
                    start_s, end_s = 0.0, 0.0
                segments.append(
                    FinalSegment(
                        start_ms=req.start_ms + (start_s * 1000.0),
                        end_ms=req.start_ms + (end_s * 1000.0),
                        text=seg_text,
                    )
                )
            return segments

        raw = getattr(result, "segments", None) or []
        for seg in raw:
            seg_text = str(
                (seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", ""))
                or ""
            ).strip()
            if not seg_text:
                continue
            start_s = float(
                (seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0.0))
                or 0.0
            )
            end_s = float(
                (seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", 0.0))
                or 0.0
            )
            segments.append(
                FinalSegment(
                    start_ms=req.start_ms + (start_s * 1000.0),
                    end_ms=req.start_ms + (end_s * 1000.0),
                    text=seg_text,
                )
            )
        return segments
