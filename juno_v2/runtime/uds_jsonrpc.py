"""Framed JSON-RPC over Unix domain sockets — Juno engine transport.

This is the production IPC transport between the Juno macOS shell and
the local engine. It
replaces "HTTP on a well-known TCP port" with the IPC pattern that
shipping macOS products actually use:

* A UDS at ``<support-root>/runtime/engine.sock``, mode 0600, in a
  directory mode 0700. Only the user's UID can connect — eliminates
  cross-process impersonation on a shared loopback port.
* Length-prefixed framed messages (4-byte big-endian + UTF-8 JSON).
  Optional binary tail for audio bytes — no base64, no multipart.
* JSON-RPC 2.0-shaped requests/responses plus server-pushed events.

The HTTP workbench server stays available for the dev tool (browser
UI), but it's now an opt-in dev affordance — the shipping ``.app``
talks to the engine over UDS only.

This module is **transport-only**. Method dispatch (routing
``broker.dictation.ingest_wav`` → ``WorkbenchApp.broker_ingest_wav``)
lives in ``juno_v2/runtime/uds_dispatch.py``.
"""

from __future__ import annotations

import json
import logging
import os
import selectors
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# 4-byte big-endian length prefix. 16 MiB cap (more than enough for any
# single broker request — audio is chunked at the source).
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
LENGTH_PREFIX = struct.Struct(">I")


class FramingError(Exception):
    """Wire-protocol violation — caller should drop the connection."""


# ---------------------------------------------------------------------------
# Wire framing
# ---------------------------------------------------------------------------

def _read_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock`` or raise EOFError on close."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        buf = sock.recv(remaining)
        if not buf:
            raise EOFError("socket closed mid-frame")
        chunks.append(buf)
        remaining -= len(buf)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> Tuple[Dict[str, Any], Optional[bytes]]:
    """Read one framed message. Returns ``(json_obj, binary_payload_or_None)``.

    Wire layout:
        4 bytes BE length L1   — JSON length
        L1 bytes UTF-8 JSON    — control message
        if json["binary_size"] is a positive int:
            4 bytes BE length L2   — binary length (must equal binary_size)
            L2 bytes raw           — binary payload
    """
    header = _read_exact(sock, LENGTH_PREFIX.size)
    (json_len,) = LENGTH_PREFIX.unpack(header)
    if json_len <= 0 or json_len > MAX_MESSAGE_BYTES:
        raise FramingError(f"json frame size out of bounds: {json_len}")
    json_bytes = _read_exact(sock, json_len)
    try:
        obj = json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FramingError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(obj, dict):
        raise FramingError(f"JSON frame must be object, got {type(obj).__name__}")
    binary: Optional[bytes] = None
    bsize = obj.get("binary_size")
    if isinstance(bsize, int) and bsize > 0:
        if bsize > MAX_MESSAGE_BYTES:
            raise FramingError(f"binary tail too large: {bsize}")
        bin_header = _read_exact(sock, LENGTH_PREFIX.size)
        (declared,) = LENGTH_PREFIX.unpack(bin_header)
        if declared != bsize:
            raise FramingError(
                f"binary_size ({bsize}) disagrees with binary frame header ({declared})"
            )
        binary = _read_exact(sock, declared)
    return obj, binary


def write_frame(
    sock: socket.socket,
    obj: Dict[str, Any],
    binary: Optional[bytes] = None,
) -> None:
    """Write one framed message. ``binary`` is sent in a separate frame
    after the JSON, with its own 4-byte length prefix."""
    payload = dict(obj)
    if binary is not None and len(binary) > 0:
        payload["binary_size"] = len(binary)
    json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(json_bytes) > MAX_MESSAGE_BYTES:
        raise FramingError(f"outgoing JSON too large: {len(json_bytes)}")
    sock.sendall(LENGTH_PREFIX.pack(len(json_bytes)))
    sock.sendall(json_bytes)
    if binary is not None and len(binary) > 0:
        sock.sendall(LENGTH_PREFIX.pack(len(binary)))
        sock.sendall(binary)


# ---------------------------------------------------------------------------
# JSON-RPC envelope helpers
# ---------------------------------------------------------------------------

ERR_PARSE = -32700  # malformed frame
ERR_INVALID_REQUEST = -32600  # missing/invalid id/method
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_AUTH = -32001  # custom — token mismatch


def _make_response(
    request_id: Any,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result or {}
    return out


def _make_error(code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return err


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

# Handler signature: (params, binary) -> (result_dict, binary_or_None) | result_dict
HandlerResult = Any  # Dict[str, Any] | Tuple[Dict[str, Any], Optional[bytes]]
Handler = Callable[[Dict[str, Any], Optional[bytes]], HandlerResult]


@dataclass
class JsonRpcServer:
    """Threaded UDS JSON-RPC server.

    One thread per connection (matches the existing
    ``ThreadingHTTPServer`` pattern in ``workbench/server.py`` so we
    don't change the concurrency model under the broker handlers).
    """

    socket_path: Path
    handlers: Dict[str, Handler] = field(default_factory=dict)
    auth_check: Optional[Callable[[Optional[str]], bool]] = None
    on_event_subscriber: Optional[Callable[[Callable[[str, Dict[str, Any]], None]], None]] = None

    _listener: Optional[socket.socket] = None
    _thread: Optional[threading.Thread] = None
    _running: bool = False
    _conn_threads: list = field(default_factory=list)
    _conn_threads_lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self, method: str, handler: Handler) -> None:
        self.handlers[method] = handler

    def start(self) -> None:
        if self._running:
            return
        self._listener = self._make_listener()
        self._running = True
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="juno-engine-uds-accept",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _make_listener(self) -> socket.socket:
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        # Stale socket from a prior run that didn't clean up.
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        # macOS / BSD enforces a 104-byte limit on AF_UNIX paths
        # (sun_path[104] in <sys/un.h>). Raising a clear error here is
        # better than ``OSError: AF_UNIX path too long`` from bind().
        if len(str(path).encode("utf-8")) >= 104:
            raise OSError(
                90,
                f"engine socket path is too long for AF_UNIX (>= 104 bytes): {path}",
            )
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(str(path))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        sock.listen(64)
        sock.setblocking(True)
        LOGGER.info("juno engine listening on %s", path)
        return sock

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._listener is not None
        sel = selectors.DefaultSelector()
        sel.register(self._listener, selectors.EVENT_READ)
        while self._running:
            try:
                events = sel.select(timeout=0.5)
            except OSError:
                break
            if not events:
                continue
            try:
                conn, _ = self._listener.accept()
            except OSError:
                break
            t = threading.Thread(
                target=self._serve_connection,
                args=(conn,),
                name="juno-engine-uds-conn",
                daemon=True,
            )
            with self._conn_threads_lock:
                # Reap finished threads before appending the new one.
                # Without this, every accepted connection leaves a dead
                # ``Thread`` object pinned in the list for the engine's
                # lifetime. With the Swift shell opening a fresh socket
                # for every health ping (every 3 s) and every workbench
                # poll (every 0.18 s during dictation), the list grew
                # without bound — the dominant contributor to long-
                # runtime memory creep of the engine process.
                if self._conn_threads:
                    self._conn_threads = [
                        x for x in self._conn_threads if x.is_alive()
                    ]
                self._conn_threads.append(t)
            t.start()

    def _serve_connection(self, conn: socket.socket) -> None:
        emit_event = self._build_event_emitter(conn)
        if self.on_event_subscriber is not None:
            try:
                self.on_event_subscriber(emit_event)
            except Exception:  # pragma: no cover — defensive
                LOGGER.exception("event subscriber raised")
        try:
            with conn:
                while self._running:
                    try:
                        request, binary = read_frame(conn)
                    except EOFError:
                        return
                    except FramingError as exc:
                        try:
                            write_frame(
                                conn,
                                _make_response(
                                    None,
                                    error=_make_error(ERR_PARSE, str(exc)),
                                ),
                            )
                        except OSError:
                            pass
                        return
                    response, response_binary = self._handle_request(request, binary)
                    if response is None:
                        # Notification — no reply.
                        continue
                    try:
                        write_frame(conn, response, response_binary)
                    except OSError:
                        return
        except Exception:  # pragma: no cover — defensive
            LOGGER.exception("connection handler crashed")

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def _handle_request(
        self,
        request: Dict[str, Any],
        binary: Optional[bytes],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
        rid = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(method, str) or not method:
            return _make_response(rid, error=_make_error(ERR_INVALID_REQUEST, "missing method")), None
        if not isinstance(params, dict):
            return _make_response(rid, error=_make_error(ERR_INVALID_PARAMS, "params must be object")), None

        if self.auth_check is not None:
            token = params.pop("_auth", None) if isinstance(params.get("_auth"), str) else None
            if not self.auth_check(token):
                return _make_response(rid, error=_make_error(ERR_AUTH, "auth failed")), None

        handler = self.handlers.get(method)
        if handler is None:
            return _make_response(rid, error=_make_error(ERR_METHOD_NOT_FOUND, f"unknown method: {method}")), None

        try:
            result = handler(params, binary)
        except Exception as exc:  # pragma: no cover — surfaced as RPC error
            LOGGER.exception("handler %s raised", method)
            return (
                _make_response(rid, error=_make_error(ERR_INTERNAL, str(exc), data={"method": method})),
                None,
            )

        result_dict, result_binary = _normalize_result(result)
        if rid is None:
            # Notification semantics — no reply.
            return None, None
        return _make_response(rid, result=result_dict), result_binary

    def _build_event_emitter(
        self,
        conn: socket.socket,
    ) -> Callable[[str, Dict[str, Any]], None]:
        lock = threading.Lock()

        def emit(event_name: str, data: Dict[str, Any]) -> None:
            payload = {"jsonrpc": "2.0", "event": event_name, "data": data}
            with lock:
                try:
                    write_frame(conn, payload, None)
                except OSError:
                    pass

        return emit


def _normalize_result(result: Any) -> Tuple[Dict[str, Any], Optional[bytes]]:
    if isinstance(result, tuple) and len(result) == 2:
        body, binary = result
        if not isinstance(body, dict):
            body = {"value": body}
        return body, binary
    if isinstance(result, dict):
        return result, None
    return {"value": result}, None


# ---------------------------------------------------------------------------
# Convenience client (for tests; production client is Swift)
# ---------------------------------------------------------------------------


class JsonRpcClient:
    """Synchronous reference client. Used by the Python test suite to
    exercise the server end-to-end. The shipping client is the Swift
    ``JunoEngineClient``."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = Path(socket_path)
        self._sock: Optional[socket.socket] = None
        self._next_id = 1
        self._lock = threading.Lock()

    def connect(self, *, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        last_err: Optional[OSError] = None
        while time.time() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(self.socket_path))
                self._sock = sock
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.05)
        raise ConnectionError(f"could not connect to {self.socket_path}: {last_err}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        binary: Optional[bytes] = None,
    ) -> Tuple[Dict[str, Any], Optional[bytes]]:
        if self._sock is None:
            raise ConnectionError("not connected")
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": rid,
                "method": method,
                "params": params or {},
            }
            write_frame(self._sock, request, binary)
            while True:
                response, response_binary = read_frame(self._sock)
                if response.get("event") is not None:
                    # Skip server-pushed events for synchronous calls.
                    continue
                if response.get("id") != rid:
                    continue
                if "error" in response:
                    err = response["error"]
                    raise JsonRpcError(err.get("code", ERR_INTERNAL), err.get("message", ""), err.get("data"))
                return response.get("result", {}), response_binary


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"JSON-RPC {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


__all__ = [
    "ERR_AUTH",
    "ERR_INTERNAL",
    "ERR_INVALID_PARAMS",
    "ERR_INVALID_REQUEST",
    "ERR_METHOD_NOT_FOUND",
    "ERR_PARSE",
    "FramingError",
    "Handler",
    "JsonRpcClient",
    "JsonRpcError",
    "JsonRpcServer",
    "MAX_MESSAGE_BYTES",
    "read_frame",
    "write_frame",
]
