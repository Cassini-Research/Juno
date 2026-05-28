from __future__ import annotations

import json
import time
from urllib import request

from juno_v2.contracts.writer import WriterTransformRequest, WriterTransformResult
from juno_v2.writer.backends.base import WriterBackend
from juno_v2.writer.config import WriterConfig


class LocalHttpJsonWriterBackend(WriterBackend):
    backend_name = "local_http_json"

    def __init__(self, config: WriterConfig) -> None:
        self.config = config
        if not config.local_http_endpoint:
            raise ValueError("local_http_endpoint is required for LocalHttpJsonWriterBackend")

    def warm(self) -> None:
        health_url = self.config.local_http_endpoint.rstrip("/") + "/healthz"
        req = request.Request(health_url, method="GET")
        with request.urlopen(req, timeout=self.config.local_http_timeout_sec) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("ok", False):
            raise RuntimeError(f"Local writer backend health check failed: {payload}")

    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        payload = json.dumps(req.to_dict()).encode("utf-8")
        started = time.perf_counter()
        http_req = request.Request(
            self.config.local_http_endpoint.rstrip("/") + "/rewrite",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(http_req, timeout=self.config.local_http_timeout_sec) as resp:  # noqa: S310
            response_payload = json.loads(resp.read().decode("utf-8"))
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        return WriterTransformResult(
            utterance_id=req.utterance_id,
            text=str(response_payload.get("text", "")).strip(),
            backend_name=self.backend_name,
            decode_ms=float(response_payload.get("decode_ms", roundtrip_ms)),
            metadata={"roundtrip_ms": roundtrip_ms, "raw": response_payload},
        )
