"""OpenAI-compatible transcription endpoint helpers.

We expose ``POST /v1/audio/transcriptions`` as a mirror of OpenAI's
Whisper API so compatible clients can plug into the local broker.

This module deliberately stays transport-neutral. It takes the HTTP
body + content-type header and hands back a ``DictationTranscriber``-
compatible result plus the JSON/text/verbose_json shape OpenAI returns.
The workbench HTTP handler wires it to the configured transcriber.

We do NOT try to cover the full OpenAI surface; the subset below is what
compatible clients send:

    file (required)        — audio bytes
    model (ignored)        — we route to the configured backend
    language               — ISO-639-1 hint; passed through
    response_format        — json (default) | text | verbose_json
    prompt, temperature    — accepted and ignored for now
"""

from __future__ import annotations

from email.parser import BytesParser
from email.policy import default as email_default_policy
from typing import Any, Mapping


class MultipartError(ValueError):
    """Raised when the client posts a malformed multipart body."""


def parse_multipart(content_type: str, body: bytes) -> dict[str, Any]:
    """Return ``{field_name: {"data": bytes, "filename": str | None}}``.

    Uses stdlib ``email`` to handle boundary parsing properly. We reject
    bodies that aren't ``multipart/form-data`` so the caller can fall
    back to a JSON handler with a clear error message.
    """
    if not content_type or "multipart/form-data" not in content_type.lower():
        raise MultipartError(f"expected multipart/form-data, got {content_type!r}")

    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    parser = BytesParser(policy=email_default_policy)
    msg = parser.parsebytes(header + body)
    if not msg.is_multipart():
        raise MultipartError("multipart body did not parse into parts")

    parts: dict[str, Any] = {}
    for part in msg.iter_parts():
        cd = part.get("Content-Disposition", "")
        name = _extract_param(cd, "name")
        if not name:
            continue
        filename = _extract_param(cd, "filename")
        payload = part.get_payload(decode=True) or b""
        parts[name] = {"data": payload, "filename": filename}
    return parts


def _extract_param(header_value: str, param: str) -> str | None:
    if not header_value:
        return None
    marker = f"{param}="
    for piece in header_value.split(";"):
        piece = piece.strip()
        if piece.lower().startswith(marker):
            value = piece[len(marker):]
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value
    return None


def build_openai_response(
    *,
    result: Mapping[str, Any],
    response_format: str,
) -> tuple[str, str]:
    """Turn an internal transcribe result into (content_type, body).

    ``result`` is a :meth:`TranscribeResult.to_dict` dict or the ``{"ok":
    True, ...}`` envelope the workbench returns. We look up ``transcript``
    regardless.
    """
    fmt = (response_format or "json").lower()
    text = str(result.get("transcript") or result.get("text") or "")

    if fmt == "text":
        return "text/plain; charset=utf-8", text

    if fmt == "verbose_json":
        duration_ms = float(result.get("audio_duration_ms", 0.0) or 0.0)
        payload = {
            "task": "transcribe",
            "language": result.get("language"),
            "duration": duration_ms / 1000.0,
            "text": text,
            # We don't emit per-word timestamps yet; the field is optional
            # in the OpenAI schema and most clients tolerate its absence.
            "segments": [],
        }
        return "application/json; charset=utf-8", _json_dumps(payload)

    # Default: json — OpenAI's simplest shape.
    return "application/json; charset=utf-8", _json_dumps({"text": text})


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


__all__ = ["MultipartError", "build_openai_response", "parse_multipart"]
