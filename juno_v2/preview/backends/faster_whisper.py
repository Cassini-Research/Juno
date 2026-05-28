from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from juno_v2.contracts.preview import PreviewDecodeRequest, PreviewDecodeResult
from juno_v2.preview.backends.base import PreviewAsrBackend
from juno_v2.preview.config import PreviewAsrConfig


class FasterWhisperPreviewBackend(PreviewAsrBackend):
    backend_name = "faster_whisper"

    def __init__(self, config: PreviewAsrConfig) -> None:
        self.config = config
        self._model = None

    def warm(self) -> None:
        if self._model is not None:
            return
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Preview ASR model path does not exist: {model_path}. "
                "Juno v2 preview lane requires a local model artifact; it will not auto-download weights."
            )
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # pragma: no cover - import depends on local runtime
            raise RuntimeError(
                "faster-whisper is required for the preview baseline backend; install project dependencies first"
            ) from exc
        self._model = WhisperModel(
            str(model_path),
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def decode(self, req: PreviewDecodeRequest) -> PreviewDecodeResult:
        self.warm()
        assert self._model is not None
        audio = np.asarray(req.audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=req.language or self.config.language,
            initial_prompt=req.initial_prompt or self.config.initial_prompt,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            vad_filter=False,
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=False,
            without_timestamps=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()).strip()
        decode_ms = (time.perf_counter() - started) * 1000.0
        detected_language = getattr(info, "language", None)
        return PreviewDecodeResult(
            utterance_id=req.utterance_id,
            text=text,
            start_ms=req.start_ms,
            end_ms=req.end_ms,
            audio_duration_ms=req.audio_duration_ms,
            is_final=req.is_final,
            backend_name=self.backend_name,
            language=detected_language,
            decode_ms=decode_ms,
            metadata={
                "avg_logprob": getattr(info, "avg_logprob", None),
                "duration": getattr(info, "duration", None),
                "requested_language": req.language or self.config.language,
                "allowed_languages": list(req.allowed_languages),
                "language_policy": req.language_policy,
            },
        )


    def unload(self) -> None:
        self._model = None
