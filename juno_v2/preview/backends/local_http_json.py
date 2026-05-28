from __future__ import annotations

import base64
import json
import time
from urllib import request

import numpy as np

from juno_v2.asr.wav import encode_wav_bytes
from juno_v2.contracts.preview import PreviewDecodeRequest, PreviewDecodeResult
from juno_v2.preview.backends.base import PreviewAsrBackend
from juno_v2.preview.config import PreviewAsrConfig


class LocalHttpJsonPreviewBackend(PreviewAsrBackend):
    """Local HTTP path for a streaming-oriented preview service.

    The service contract is intentionally local-first. Juno does not send audio
    to remote hosted APIs here. The backend may internally maintain decoder
    state keyed by utterance id, but the core engine contract stays simple.
    """

    backend_name = "local_http_json"

    def __init__(self, config: PreviewAsrConfig) -> None:
        self.config = config
        if not config.local_http_endpoint:
            raise ValueError("local_http_endpoint is required for LocalHttpJsonPreviewBackend")

    def warm(self) -> None:
        health_url = self.config.local_http_endpoint.rstrip("/") + "/healthz"
        req = request.Request(health_url, method="GET")
        with request.urlopen(req, timeout=self.config.local_http_timeout_sec) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("ok", False):
            raise RuntimeError(f"Local preview ASR health check failed: {payload}")

    def decode(self, req: PreviewDecodeRequest) -> PreviewDecodeResult:
        audio = np.asarray(req.audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        wav_bytes = encode_wav_bytes(audio, req.sample_rate_hz)
        started = time.perf_counter()
        http_req = request.Request(
            self.config.local_http_endpoint.rstrip("/") + "/preview",
            data=wav_bytes,
            method="POST",
            headers={
                "Content-Type": "audio/wav",
                "X-Juno-Language": req.language or self.config.language or "",
                "X-Juno-Allowed-Languages": base64.b64encode(json.dumps(req.allowed_languages).encode("utf-8")).decode("ascii"),
                "X-Juno-Language-Policy": req.language_policy or "",
                "X-Juno-Utterance-Id": req.utterance_id,
                "X-Juno-Is-Final": "true" if req.is_final else "false",
                "X-Juno-Decode-Seq": str(req.decode_seq),
                "X-Juno-Reset-Decoder-State": "true" if req.reset_decoder_state else "false",
                "X-Juno-Bias-Phrases": base64.b64encode(json.dumps(req.bias_phrases).encode("utf-8")).decode("ascii"),
                "X-Juno-Context": base64.b64encode(json.dumps(req.context_payload).encode("utf-8")).decode("ascii"),
            },
        )
        with request.urlopen(http_req, timeout=self.config.local_http_timeout_sec) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        return PreviewDecodeResult(
            utterance_id=req.utterance_id,
            text=str(payload.get("text", "")).strip(),
            start_ms=req.start_ms,
            end_ms=req.end_ms,
            audio_duration_ms=req.audio_duration_ms,
            is_final=req.is_final,
            backend_name=self.backend_name,
            language=payload.get("language", req.language or self.config.language),
            decode_ms=float(payload.get("decode_ms", roundtrip_ms)),
            metadata={"roundtrip_ms": roundtrip_ms, "raw": payload, "decode_seq": req.decode_seq},
        )
