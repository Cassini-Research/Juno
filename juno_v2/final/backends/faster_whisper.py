from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from juno_v2.contracts.final import FinalDecodeRequest, FinalDecodeResult, FinalSegment
from juno_v2.final.backends.base import FinalAsrBackend
from juno_v2.final.config import FinalAsrConfig


class FasterWhisperFinalBackend(FinalAsrBackend):
    backend_name = "faster_whisper"

    def __init__(self, config: FinalAsrConfig) -> None:
        self.config = config
        self._model = None

    def warm(self) -> None:
        if self._model is not None:
            return
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Final ASR model path does not exist: {model_path}. "
                "Juno v2 final lane requires a local model artifact; it will not auto-download weights."
            )
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is required for the final baseline backend; install project dependencies first"
            ) from exc
        self._model = WhisperModel(
            str(model_path),
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def decode(self, req: FinalDecodeRequest) -> FinalDecodeResult:
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
            without_timestamps=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        segment_list: list[FinalSegment] = []
        texts: list[str] = []
        # Aggregate per-segment confidence/silence signals (duration-weighted)
        # so the commit-side hallucination guard has audio-side corroboration
        # for whisper-on-silence one-shots like "Thank you.".
        weighted_logprob = 0.0
        weighted_no_speech = 0.0
        weighted_compression = 0.0
        total_weight_s = 0.0
        for seg in segments:
            seg_text = getattr(seg, "text", "").strip()
            if not seg_text:
                continue
            texts.append(seg_text)
            start_s = float(getattr(seg, "start", 0.0) or 0.0)
            end_s = float(getattr(seg, "end", 0.0) or 0.0)
            seg_avg_logprob = getattr(seg, "avg_logprob", None)
            seg_no_speech = getattr(seg, "no_speech_prob", None)
            seg_compression = getattr(seg, "compression_ratio", None)
            # Per-segment audio-side signals are required by the commit-side
            # trailing-silence-hallucination guard to corroborate stock-phrase
            # matches on individual tail segments.
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
            segment_list.append(
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
        decode_ms = (time.perf_counter() - started) * 1000.0
        detected_language = getattr(info, "language", None)
        seg_avg_logprob_aggregate = (
            weighted_logprob / total_weight_s if total_weight_s > 0 else None
        )
        no_speech_prob_aggregate = (
            weighted_no_speech / total_weight_s if total_weight_s > 0 else None
        )
        compression_ratio_aggregate = (
            weighted_compression / total_weight_s if total_weight_s > 0 else None
        )
        # Prefer info.avg_logprob when available (matches existing behavior);
        # fall back to the segment-weighted aggregate otherwise so the
        # commit-side guard always has the field populated when segments
        # carried the signal.
        info_avg_logprob = getattr(info, "avg_logprob", None)
        avg_logprob_value = info_avg_logprob if info_avg_logprob is not None else seg_avg_logprob_aggregate
        return FinalDecodeResult(
            utterance_id=req.utterance_id,
            text=" ".join(texts).strip(),
            start_ms=req.start_ms,
            end_ms=req.end_ms,
            audio_duration_ms=req.audio_duration_ms,
            backend_name=self.backend_name,
            model_path=str(self.config.model_path or ""),
            language=detected_language,
            decode_ms=decode_ms,
            end_of_turn_latency_ms=decode_ms,
            segments=segment_list,
            metadata={
                "avg_logprob": avg_logprob_value,
                "no_speech_prob": no_speech_prob_aggregate,
                "compression_ratio": compression_ratio_aggregate,
                "duration": getattr(info, "duration", None),
                "requested_language": req.language or self.config.language,
                "allowed_languages": list(req.allowed_languages),
                "language_policy": req.language_policy,
            },
        )


    def unload(self) -> None:
        self._model = None
