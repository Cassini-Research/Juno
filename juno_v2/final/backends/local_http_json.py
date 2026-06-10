from __future__ import annotations

import base64
import json
import time
from urllib import request

import numpy as np

from juno_v2.asr.wav import encode_wav_bytes
from juno_v2.contracts.final import FinalDecodeRequest, FinalDecodeResult, FinalSegment
from juno_v2.final.backends.base import FinalAsrBackend, effective_decode_language
from juno_v2.final.config import FinalAsrConfig


class LocalHttpJsonFinalBackend(FinalAsrBackend):
    """Generic local HTTP path for an ASR service.

    Expects a local HTTP endpoint that accepts WAV bytes and returns JSON like:
    {
      "text": "...",
      "language": "en",
      "decode_ms": 123.4,
      "segments": [{"start_ms": 0.0, "end_ms": 120.0, "text": "..."}]
    }
    """

    backend_name = "local_http_json"

    def __init__(self, config: FinalAsrConfig) -> None:
        self.config = config
        if not config.local_http_endpoint:
            raise ValueError("local_http_endpoint is required for LocalHttpJsonFinalBackend")

    def warm(self) -> None:
        health_url = self.config.local_http_endpoint.rstrip("/") + "/healthz"
        req = request.Request(health_url, method="GET")
        with request.urlopen(req, timeout=self.config.local_http_timeout_sec) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("ok", False):
            raise RuntimeError(f"Local ASR service health check failed: {payload}")

    def decode(self, req: FinalDecodeRequest) -> FinalDecodeResult:
        audio = np.asarray(req.audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        wav_bytes = encode_wav_bytes(audio, req.sample_rate_hz)
        started = time.perf_counter()
        http_req = request.Request(
            self.config.local_http_endpoint.rstrip("/") + "/transcribe",
            data=wav_bytes,
            method="POST",
            headers={
                "Content-Type": "audio/wav",
                "X-Juno-Language": effective_decode_language(req, self.config.language) or "",
                "X-Juno-Allowed-Languages": base64.b64encode(json.dumps(req.allowed_languages).encode("utf-8")).decode("ascii"),
                "X-Juno-Language-Policy": req.language_policy or "",
                "X-Juno-Utterance-Id": req.utterance_id,
                "X-Juno-Bias-Phrases": base64.b64encode(json.dumps(req.bias_phrases).encode("utf-8")).decode("ascii"),
                "X-Juno-Context": base64.b64encode(json.dumps(req.context_payload).encode("utf-8")).decode("ascii"),
            },
        )
        with request.urlopen(http_req, timeout=self.config.local_http_timeout_sec) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        segments = [
            FinalSegment(
                start_ms=float(segment.get("start_ms", req.start_ms)),
                end_ms=float(segment.get("end_ms", req.end_ms)),
                text=str(segment.get("text", "")).strip(),
            )
            for segment in payload.get("segments", [])
            if str(segment.get("text", "")).strip()
        ]
        decode_ms = float(payload.get("decode_ms", roundtrip_ms))
        # Remote backends dispatch by endpoint rather than a file path.
        # We prefer an explicit ``model`` echoed in the response payload
        # (so the server can surface the exact checkpoint it used) and
        # fall back to the endpoint URL so the trace always has *some*
        # provenance string.
        model_path = str(
            payload.get("model")
            or payload.get("model_path")
            or self.config.local_http_endpoint
            or self.config.model_path
            or ""
        )
        return FinalDecodeResult(
            utterance_id=req.utterance_id,
            text=str(payload.get("text", "")).strip(),
            start_ms=req.start_ms,
            end_ms=req.end_ms,
            audio_duration_ms=req.audio_duration_ms,
            backend_name=self.backend_name,
            model_path=model_path,
            language=payload.get("language", req.language or self.config.language),
            decode_ms=decode_ms,
            end_of_turn_latency_ms=roundtrip_ms,
            segments=segments,
            metadata={"roundtrip_ms": roundtrip_ms, "raw": payload},
        )
