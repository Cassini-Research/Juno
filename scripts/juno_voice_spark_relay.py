#!/usr/bin/env python3
"""Authenticated, ephemeral DGX front door for a DGX-local speech backend.

The service offers the OpenAI batch transcription route plus a small live PCM
session API. Audio and transcripts live only in memory and expire after the
session TTL. Inference stays on the Spark: the relay adapts Juno's public
contract to the already-installed Parakeet service and applies Juno's
deterministic, conservative spoken-retake pass to the final text.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default as email_default_policy
import hmac
import http.client
import ipaddress
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit
import uuid
import wave

from self_corrections import apply_unambiguous_retakes


MAX_BATCH_BYTES = 64 * 1024 * 1024
MAX_PCM_BYTES = 20 * 1024 * 1024
PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
SESSION_RE = re.compile(r"^/v1/realtime/transcription_sessions/([a-f0-9]{32})(?:/(audio|commit|events))?$")


class RelayError(RuntimeError):
    pass


def _wav_from_pcm16(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _multipart_parts(content_type: str, body: bytes) -> dict[str, bytes]:
    if "multipart/form-data" not in content_type.lower():
        raise RelayError("expected multipart/form-data")
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = BytesParser(policy=email_default_policy).parsebytes(header + body)
    if not message.is_multipart():
        raise RelayError("malformed multipart body")
    parts: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            parts[str(name)] = part.get_payload(decode=True) or b""
    return parts


def _multipart_audio(wav: bytes, language: str) -> tuple[str, bytes]:
    # Parakeet's English-only gateway intentionally accepts exactly `model`,
    # `response_format`, and `file`; retain the argument for the public Juno
    # contract but do not forward it as an unsupported multipart field.
    _ = language
    boundary = "juno-relay-" + uuid.uuid4().hex
    pieces = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{PARAKEET_MODEL}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
        wav,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return f"multipart/form-data; boundary={boundary}", b"".join(pieces)


class JunoUpstream:
    def __init__(self, base_url: str, api_key: str, timeout: int = 360) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Juno upstream must be loopback HTTP")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.api_key = api_key
        self.timeout = timeout

    def request(self, path: str, content_type: str, body: bytes) -> tuple[int, str, bytes]:
        if path == "/v1/audio/transcriptions":
            parts = _multipart_parts(content_type, body)
            audio = parts.get("file", b"")
            if not audio:
                raise RelayError("the file field is required")
            language = parts.get("language", b"en").decode("utf-8", errors="replace").strip() or "en"
            content_type, body = _multipart_audio(audio, language)
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request(
                "POST",
                path,
                body=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": content_type,
                    "Content-Length": str(len(body)),
                    "Accept": "application/json",
                },
            )
            response = connection.getresponse()
            raw = response.read(MAX_BATCH_BYTES + 1)
            if len(raw) > MAX_BATCH_BYTES:
                raise RelayError("speech response exceeded the relay limit")
            response_type = response.getheader("Content-Type", "application/json")
            if path == "/v1/audio/transcriptions" and response.status == 200:
                payload = json.loads(raw or b"{}")
                text = str(payload.get("text") or "").strip()
                if not text:
                    raise RelayError("speech backend returned an empty transcript")
                corrected, corrections = apply_unambiguous_retakes(text)
                payload["text"] = corrected.strip()
                payload["language"] = str(payload.get("language") or "en")
                # Counts make the transform observable without exposing either
                # the transcript or correction contents to logs/health calls.
                payload["juno_corrections_applied"] = len(corrections)
                raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
                response_type = "application/json; charset=utf-8"
            return response.status, response_type, raw
        finally:
            connection.close()

    def health(self) -> bool:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            response.read()
            return response.status == 200
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()

    def transcribe_pcm(self, pcm: bytes, sample_rate: int, language: str) -> str:
        wav = _wav_from_pcm16(pcm, sample_rate)
        content_type, body = _multipart_audio(wav, language)
        status, _, raw = self.request("/v1/audio/transcriptions", content_type, body)
        if status != 200:
            raise RelayError(f"speech backend returned HTTP {status}")
        payload = json.loads(raw or b"{}")
        text = str(payload.get("text") or "").strip()
        if not text:
            raise RelayError("speech backend returned an empty transcript")
        return text


class Session:
    def __init__(self, language: str, sample_rate: int, ttl_seconds: int) -> None:
        self.id = uuid.uuid4().hex
        self.language = language
        self.sample_rate = sample_rate
        self.ttl_seconds = ttl_seconds
        self.pcm = bytearray()
        self.lock = threading.RLock()
        self.subscribers: list[queue.Queue[dict[str, Any]]] = []
        self.last_text = ""
        self.last_transcribed_bytes = 0
        self.worker_active = False
        self.dirty = False
        self.final_requested = False
        self.closed = False
        self.updated_at = time.monotonic()

    def publish(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.updated_at = time.monotonic()
            for subscriber in list(self.subscribers):
                subscriber.put(dict(event))

    def zero(self) -> None:
        with self.lock:
            for index in range(len(self.pcm)):
                self.pcm[index] = 0
            self.pcm.clear()
            self.last_text = ""
            self.closed = True
            self.updated_at = time.monotonic()


class SessionStore:
    def __init__(self, upstream: JunoUpstream, max_sessions: int, ttl_seconds: int, preview_seconds: float) -> None:
        self.upstream = upstream
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.preview_seconds = preview_seconds
        self.lock = threading.RLock()
        self.sessions: dict[str, Session] = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="juno-live")
        self.stopping = threading.Event()
        threading.Thread(target=self._cleaner, name="juno-session-cleaner", daemon=True).start()

    def create(self, language: str, sample_rate: int) -> Session:
        if sample_rate != 16000:
            raise RelayError("live sessions require mono PCM16 at 16000 Hz")
        with self.lock:
            closed = [session_id for session_id, session in self.sessions.items() if session.closed]
            for session_id in closed:
                self.sessions.pop(session_id, None)
            if len(self.sessions) >= self.max_sessions:
                raise RelayError("live transcription session limit reached")
            session = Session(language or "auto", sample_rate, self.ttl_seconds)
            self.sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Session:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None or session.closed:
            raise RelayError("transcription session not found")
        return session

    def delete(self, session_id: str) -> bool:
        with self.lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        session.publish({"type": "session.closed", "session_id": session.id})
        session.zero()
        return True

    def append(self, session: Session, pcm: bytes) -> None:
        if not pcm or len(pcm) % 2:
            raise RelayError("audio chunks must contain complete PCM16 samples")
        with session.lock:
            if len(session.pcm) + len(pcm) > MAX_PCM_BYTES:
                raise RelayError("live audio exceeded the in-memory session limit")
            session.pcm.extend(pcm)
            session.dirty = True
            session.updated_at = time.monotonic()
            threshold = int(session.sample_rate * 2 * self.preview_seconds)
            should_preview = len(session.pcm) - session.last_transcribed_bytes >= threshold
        if should_preview:
            self._schedule(session, final=False)

    def commit(self, session: Session) -> None:
        with session.lock:
            if not session.pcm:
                raise RelayError("cannot commit an empty audio buffer")
            session.final_requested = True
            session.dirty = True
        self._schedule(session, final=True)

    def _schedule(self, session: Session, final: bool) -> None:
        with session.lock:
            if final:
                session.final_requested = True
            if session.worker_active:
                return
            session.worker_active = True
        self.executor.submit(self._worker, session)

    def _worker(self, session: Session) -> None:
        try:
            while True:
                with session.lock:
                    if session.closed:
                        return
                    snapshot = bytes(session.pcm)
                    final = session.final_requested
                    session.final_requested = False
                    session.dirty = False
                session.publish({"type": "transcription.started", "session_id": session.id, "final": final})
                try:
                    text = self.upstream.transcribe_pcm(snapshot, session.sample_rate, session.language)
                    with session.lock:
                        previous = session.last_text
                        session.last_text = text
                        session.last_transcribed_bytes = len(snapshot)
                    event_type = "conversation.item.input_audio_transcription.completed" if final else "transcription.partial"
                    session.publish({"type": event_type, "session_id": session.id, "text": text})
                    if final:
                        session.publish({"type": "response.done", "session_id": session.id})
                        session.zero()
                        return
                    if text.startswith(previous):
                        delta = text[len(previous):]
                        if delta:
                            session.publish({"type": "conversation.item.input_audio_transcription.delta", "session_id": session.id, "delta": delta})
                except Exception as exc:
                    session.publish({"type": "error", "session_id": session.id, "message": str(exc)})
                    if final:
                        session.zero()
                        return
                with session.lock:
                    if not session.dirty and not session.final_requested:
                        return
        finally:
            with session.lock:
                session.worker_active = False

    def _cleaner(self) -> None:
        while not self.stopping.wait(10):
            now = time.monotonic()
            with self.lock:
                expired = [session_id for session_id, session in self.sessions.items() if now - session.updated_at > session.ttl_seconds]
            for session_id in expired:
                self.delete(session_id)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "JunoSparkRelay/1.0"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    @property
    def api_key(self) -> str:
        return self.server.api_key  # type: ignore[attr-defined]

    @property
    def upstream(self) -> JunoUpstream:
        return self.server.upstream  # type: ignore[attr-defined]

    @property
    def sessions(self) -> SessionStore:
        return self.server.sessions  # type: ignore[attr-defined]

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        if supplied.lower().startswith("bearer "):
            supplied = supplied[7:].strip()
        return hmac.compare_digest(supplied.encode(), self.api_key.encode())

    def _headers(self, status: int, content_type: str, length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if length is not None:
            self.send_header("Content-Length", str(length))
        if status >= 400 or length is None:
            # Several error paths intentionally reject a request before
            # reading its body. Closing prevents a reverse proxy from reusing
            # that connection and parsing the unread body as a new method.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self._headers(status, "application/json", len(body))
        self.wfile.write(body)

    def _read(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RelayError("invalid Content-Length") from exc
        if length < 1 or length > limit:
            raise RelayError("request body is empty or too large")
        return self.rfile.read(length)

    def _session_match(self) -> tuple[Session, str | None] | None:
        match = SESSION_RE.match(urlsplit(self.path).path)
        if not match:
            return None
        return self.sessions.get(match.group(1)), match.group(2)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/healthz":
            healthy = self.upstream.health()
            self._json(200 if healthy else 503, {"status": "ok" if healthy else "degraded", "backend": "DGX-local Parakeet + Juno final-text", "storage": "memory-only"})
            return
        if not self._authorized():
            self._json(401, {"error": {"message": "Unauthorized"}})
            return
        try:
            matched = self._session_match()
            if matched is None or matched[1] != "events":
                self._json(404, {"error": {"message": "Not found"}})
                return
            session = matched[0]
            subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
            with session.lock:
                session.subscribers.append(subscriber)
            self._headers(200, "text/event-stream")
            self.wfile.write(f"event: session.created\ndata: {{\"session_id\":\"{session.id}\"}}\n\n".encode())
            self.wfile.flush()
            try:
                while True:
                    if session.closed and subscriber.empty():
                        break
                    try:
                        event = subscriber.get(timeout=15)
                        self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            finally:
                with session.lock:
                    if subscriber in session.subscribers:
                        session.subscribers.remove(subscriber)
        except (BrokenPipeError, ConnectionResetError):
            return
        except RelayError as exc:
            self._json(404, {"error": {"message": str(exc)}})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(401, {"error": {"message": "Unauthorized"}})
            return
        path = urlsplit(self.path).path.rstrip("/")
        try:
            if path == "/v1/audio/transcriptions":
                body = self._read(MAX_BATCH_BYTES)
                content_type = self.headers.get("Content-Type", "application/octet-stream")
                status, response_type, raw = self.upstream.request("/v1/audio/transcriptions", content_type, body)
                wants_stream = "text/event-stream" in self.headers.get("Accept", "") or parse_qs(urlsplit(self.path).query).get("stream") == ["true"]
                if wants_stream and status == 200:
                    payload = json.loads(raw or b"{}")
                    text = str(payload.get("text") or "")
                    events = [
                        ("transcription.started", {"type": "transcription.started"}),
                        ("conversation.item.input_audio_transcription.delta", {"type": "conversation.item.input_audio_transcription.delta", "delta": text}),
                        ("conversation.item.input_audio_transcription.completed", {"type": "conversation.item.input_audio_transcription.completed", "text": text}),
                    ]
                    encoded = b"".join(f"event: {name}\ndata: {json.dumps(value, separators=(',', ':'))}\n\n".encode() for name, value in events)
                    self._headers(200, "text/event-stream", len(encoded))
                    self.wfile.write(encoded)
                else:
                    self._headers(status, response_type, len(raw))
                    self.wfile.write(raw)
                return
            if path == "/v1/realtime/transcription_sessions":
                body = self._read(64 * 1024)
                payload = json.loads(body)
                session = self.sessions.create(str(payload.get("language") or "auto"), int(payload.get("sample_rate") or 16000))
                self._json(201, {"id": session.id, "object": "realtime.transcription_session", "audio_format": "pcm16", "sample_rate": session.sample_rate, "expires_in": session.ttl_seconds})
                return
            matched = self._session_match()
            if matched is None:
                self._json(404, {"error": {"message": "Not found"}})
                return
            session, action = matched
            if action == "audio":
                body = self._read(MAX_PCM_BYTES)
                if self.headers.get("Content-Type", "").startswith("application/json"):
                    body = base64.b64decode(json.loads(body).get("audio", ""), validate=True)
                self.sessions.append(session, body)
                self._json(202, {"status": "accepted", "session_id": session.id, "buffered_bytes": len(session.pcm)})
                return
            if action == "commit":
                self.sessions.commit(session)
                self._json(202, {"status": "transcribing", "session_id": session.id})
                return
            self._json(404, {"error": {"message": "Not found"}})
        except (RelayError, ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": {"message": str(exc)}})
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            self._json(502, {"error": {"message": f"speech upstream unavailable: {type(exc).__name__}"}})

    def do_DELETE(self) -> None:
        if not self._authorized():
            self._json(401, {"error": {"message": "Unauthorized"}})
            return
        match = SESSION_RE.match(urlsplit(self.path).path)
        if match and self.sessions.delete(match.group(1)):
            self._json(200, {"status": "deleted"})
        else:
            self._json(404, {"error": {"message": "Not found"}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18790)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--upstream-key-file", required=True)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:18797")
    parser.add_argument("--session-ttl-seconds", type=int, default=300)
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument("--preview-seconds", type=float, default=2.0)
    parser.add_argument(
        "--allow-tailnet-bind",
        action="store_true",
        help="allow an explicit Tailscale CGNAT/ULA address when HTTPS Serve cannot be configured",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        try:
            address = ipaddress.ip_address(args.host)
        except ValueError as exc:
            raise SystemExit("relay host must be loopback or an explicit Tailscale address") from exc
        tailnet_v4 = address.version == 4 and address in ipaddress.ip_network("100.64.0.0/10")
        tailnet_v6 = address.version == 6 and address in ipaddress.ip_network("fd7a:115c:a1e0::/48")
        if not args.allow_tailnet_bind or not (tailnet_v4 or tailnet_v6):
            raise SystemExit("non-loopback bind refused; use Tailscale Serve or --allow-tailnet-bind with the node's Tailscale IP")
    api_key = Path(args.api_key_file).read_text().strip()
    upstream_key = Path(args.upstream_key_file).read_text().strip()
    if not api_key or not upstream_key or hmac.compare_digest(api_key, upstream_key):
        raise SystemExit("public and upstream keys must be present and distinct")
    upstream = JunoUpstream(args.upstream_url, upstream_key)
    sessions = SessionStore(upstream, max(1, args.max_sessions), max(30, args.session_ttl_seconds), max(0.5, args.preview_seconds))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.api_key = api_key  # type: ignore[attr-defined]
    server.upstream = upstream  # type: ignore[attr-defined]
    server.sessions = sessions  # type: ignore[attr-defined]
    print(f"Juno relay ready on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        sessions.stopping.set()
        server.server_close()


if __name__ == "__main__":
    main()
