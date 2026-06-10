from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future
import io
import json
import queue
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

from juno_v2.contracts.preview import PreviewDecodeRequest
from juno_v2.preview.streaming_core import (
    FasterWhisperStreamingDecoder,
    MlxWhisperStreamingDecoder,
    StreamingPreviewDecoder,
    StreamingPreviewSessionManager,
)


class StreamingPreviewService:
    def __init__(
        self,
        processor: StreamingPreviewSessionManager | None = None,
        *,
        processor_factory: Callable[[], StreamingPreviewSessionManager] | None = None,
        service_metadata: dict[str, Any] | None = None,
        eager_load: bool = False,
    ) -> None:
        if processor_factory is None:
            if processor is None:
                raise ValueError("processor or processor_factory is required")
            processor_factory = lambda: processor
        self._processor_factory = processor_factory
        self._decode_owner: _PreviewDecodeOwner | None = None
        self._decode_owner_guard = threading.Lock()
        self._service_metadata = dict(service_metadata or {})
        if eager_load:
            self._ensure_decode_owner()

    def handle_health(self) -> dict[str, Any]:
        owner = self._decode_owner
        health = (
            owner.health()
            if owner is not None
            else {
                "ok": True,
                "decode_owner_ready": False,
                "decode_owner_thread_id": None,
                "queue_depth": 0,
                "lazy_load": True,
                "error": None,
            }
        )
        if self._service_metadata:
            health["service"] = dict(self._service_metadata)
        return health

    def handle_warm(self) -> dict[str, Any]:
        owner = self._ensure_decode_owner()
        health = owner.health()
        health["warmed"] = bool(health.get("ok"))
        if self._service_metadata:
            health["service"] = dict(self._service_metadata)
        return health

    def handle_preview_stream(self, body: bytes, headers: dict[str, str]) -> dict[str, Any]:
        req = PreviewDecodeRequest(
            utterance_id=headers.get('x-juno-utterance-id', 'unknown'),
            audio=_decode_wav_bytes(body),
            sample_rate_hz=_extract_sample_rate_hz(body) or 16000,
            start_ms=0.0,
            end_ms=0.0,
            language=(headers.get('x-juno-language') or '').strip() or None,
            allowed_languages=_decode_b64_json(headers.get('x-juno-allowed-languages'), default=[]),
            language_policy=(headers.get('x-juno-language-policy') or '').strip() or None,
            is_final=_header_bool(headers.get('x-juno-is-final')),
            decode_seq=_header_int(headers.get('x-juno-decode-seq')),
            reset_decoder_state=_header_bool(headers.get('x-juno-reset-decoder-state')),
            bias_phrases=_decode_b64_json(headers.get('x-juno-bias-phrases'), default=[]),
            context_payload=_decode_b64_json(headers.get('x-juno-context'), default={}),
        )
        duration_ms = (len(req.audio) / float(req.sample_rate_hz)) * 1000.0 if req.sample_rate_hz > 0 else 0.0
        req.start_ms = 0.0
        req.end_ms = duration_ms
        response = self._ensure_decode_owner().decode(req)
        # New contract carries committed/tail split. Older clients can still
        # read ``text`` (the combined string from response.text).
        committed_text = getattr(response, "committed_text", None)
        tail_text = getattr(response, "tail_text", None)
        payload = {
            'text': response.text,
            'committed_text': committed_text if committed_text is not None else response.text,
            'tail_text': tail_text if tail_text is not None else "",
            'language': response.language,
            'decode_ms': response.decode_ms,
            **response.metadata,
        }
        if self._service_metadata:
            payload["service"] = dict(self._service_metadata)
            payload.setdefault("service_backend", self._service_metadata.get("backend"))
            payload.setdefault("service_model_path", self._service_metadata.get("model_path"))
            payload.setdefault("service_device", self._service_metadata.get("device"))
            payload.setdefault("service_compute_type", self._service_metadata.get("compute_type"))
        return payload

    def close(self) -> None:
        owner = self._decode_owner
        if owner is not None:
            owner.stop()

    def _ensure_decode_owner(self) -> _PreviewDecodeOwner:
        owner = self._decode_owner
        if owner is not None:
            return owner
        with self._decode_owner_guard:
            owner = self._decode_owner
            if owner is None:
                owner = _PreviewDecodeOwner(self._processor_factory)
                self._decode_owner = owner
        return owner


class _DecodeJob:
    __slots__ = ("req", "future", "enqueued_at", "queue_depth_at_enqueue")

    def __init__(self, req: PreviewDecodeRequest, future: Future, queue_depth_at_enqueue: int) -> None:
        self.req = req
        self.future = future
        self.enqueued_at = time.perf_counter()
        self.queue_depth_at_enqueue = queue_depth_at_enqueue


class _PreviewDecodeOwner:
    """Own the preview model on one thread for its full lifetime.

    MLX preview state is thread-sensitive. Building or warming the backend on
    one request thread and decoding on another can surface runtime errors such
    as "There is no Stream(gpu, 0) in current thread". This worker gives the
    preview service a single decode owner while the HTTP server can remain
    threaded for socket responsiveness.
    """

    def __init__(self, processor_factory: Callable[[], StreamingPreviewSessionManager]) -> None:
        self._processor_factory = processor_factory
        self._jobs: queue.Queue[_DecodeJob | None] = queue.Queue()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._processor: StreamingPreviewSessionManager | None = None
        self._startup_error: BaseException | None = None
        self._worker_thread_id: int | None = None
        self._thread = threading.Thread(target=self._run, name="juno-preview-decode-owner", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=120.0)

    def decode(self, req: PreviewDecodeRequest):
        if self._startup_error is not None:
            raise RuntimeError(f"preview decode owner failed to start: {self._startup_error}") from self._startup_error
        if self._processor is None:
            raise RuntimeError("preview decode owner is not ready")
        future: Future = Future()
        self._jobs.put(_DecodeJob(req, future, self._jobs.qsize()))
        return future.result(timeout=30.0)

    def health(self) -> dict[str, Any]:
        return {
            "ok": self._startup_error is None and self._processor is not None and not self._closed.is_set(),
            "decode_owner_ready": self._processor is not None,
            "decode_owner_thread_id": self._worker_thread_id,
            "queue_depth": self._jobs.qsize(),
            "error": f"{type(self._startup_error).__name__}: {self._startup_error}" if self._startup_error else None,
        }

    def stop(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._jobs.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        self._worker_thread_id = threading.get_ident()
        try:
            processor = self._processor_factory()
            warm = getattr(processor, "warm", None)
            if callable(warm):
                warm()
            self._processor = processor
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
        finally:
            self._ready.set()

        if self._startup_error is not None:
            return

        while not self._closed.is_set():
            job = self._jobs.get()
            if job is None:
                return
            try:
                queue_wait_ms = (time.perf_counter() - job.enqueued_at) * 1000.0
                processor = self._processor
                if processor is None:
                    raise RuntimeError("preview decode processor disappeared")
                # Single-flight by virtue of running on this dedicated thread:
                # while we're inside processor.process() no other decode runs.
                # StreamingPreviewSessionManager owns its own cadence policy.
                result = processor.process(job.req)
                result.metadata = {
                    **dict(result.metadata or {}),
                    "preview_owner_queue_depth": job.queue_depth_at_enqueue,
                    "broker_queue_wait_ms": queue_wait_ms,
                    "decode_owner_thread_id": self._worker_thread_id,
                    "request_audio_ms": job.req.audio_duration_ms,
                }
                if not job.future.done():
                    job.future.set_result(result)
            except BaseException as exc:  # noqa: BLE001
                if not job.future.done():
                    job.future.set_exception(exc)


def build_http_server(service: StreamingPreviewService, *, host: str, port: int) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == '/healthz':
                self._send_json(service.handle_health())
                return
            if self.path == '/warm':
                try:
                    self._send_json(service.handle_warm())
                except Exception as exc:  # noqa: BLE001
                    self._send_json({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}, status=500)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != '/preview_stream':
                self.send_error(404)
                return
            content_length = int(self.headers.get('Content-Length', '0'))
            body = self.rfile.read(content_length)
            try:
                payload = service.handle_preview_stream(body, {k.lower(): v for k, v in self.headers.items()})
            except Exception as exc:  # noqa: BLE001
                self._send_json({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}, status=500)
                return
            self._send_json(payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), _Handler)


class _ServerRunner:
    def __init__(self, httpd: ThreadingHTTPServer) -> None:
        self.httpd = httpd
        self.thread = threading.Thread(target=httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Juno v2 local streaming preview service')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8795)
    parser.add_argument(
        '--backend',
        default='mlx_whisper',
        choices=('mlx_whisper', 'mlx_whisper_streaming', 'faster_whisper'),
        help='Preview backend hosted by this isolated service process.',
    )
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--hf-repo-id', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--compute-type', default='int8')
    parser.add_argument('--sample-rate-hz', type=int, default=16000)
    parser.add_argument('--max-session-idle-seconds', type=float, default=30.0)
    parser.add_argument('--decode-cadence-ms', type=float, default=500.0,
                        help='Minimum wall-clock interval between Whisper decode calls. '
                             '500 ms is tight enough that LocalAgreement-2 produces its '
                             'first commit ~1.0 s after speech starts.')
    parser.add_argument('--min-decode-audio-ms', type=float, default=250.0,
                        help='Smallest audio buffer Whisper will be run on (below: skip). '
                             '250 ms is the floor below which Whisper-large-v3-turbo '
                             'reliably emits empty or stock-phrase output.')
    parser.add_argument('--force-trim-max-seconds', type=float, default=25.0,
                        help='Buffer hard cap. Beyond this we force-trim regardless of segment ends.')
    parser.add_argument('--force-trim-carry-seconds', type=float, default=5.0,
                        help='How much audio we keep after a force-trim.')
    parser.add_argument('--eager-load', action='store_true',
                        help='Load the preview model before /healthz returns ready. Default is lazy.')
    parser.add_argument('--vad-enabled', type=int, default=1,
                        help='1 (default) gates audio via Silero VAD using the official '
                             'VADIterator streaming API (threshold=0.5, min_silence_700ms, '
                             'speech_pad_400ms, 1s warmup). 0 bypasses VAD entirely. '
                             'arXiv:2501.11378 measured SileroVAD pre-filtering brings WER '
                             'on non-speech audio from 104.8% down to 8.0% — this is the '
                             'single biggest hallucination defense lever.')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    # Auto-enable HF Hub offline mode when the preview model is cached.
    # mlx_whisper.load_models() calls snapshot_download internally with
    # no local_files_only flag; without HF_HUB_OFFLINE=1 it would burn a
    # 10s timeout per file revalidating against the hub when the user
    # is offline. The preview service is a separate process from the
    # runtime service, so it needs its own check.
    from juno_v2.runtime.offline_mode import enable_offline_mode_if_cache_complete
    repo_ids: list[str] = []
    if args.hf_repo_id:
        repo_ids.append(str(args.hf_repo_id))
    elif args.model_path and "/" in str(args.model_path) and not Path(str(args.model_path)).exists():
        repo_ids.append(str(args.model_path))
    enable_offline_mode_if_cache_complete(repo_ids)

    service_metadata = _service_metadata(args)
    service = StreamingPreviewService(
        processor_factory=lambda: _build_processor(args),
        service_metadata=service_metadata,
        eager_load=bool(args.eager_load),
    )
    httpd = build_http_server(service, host=args.host, port=args.port)
    runner = _ServerRunner(httpd)
    runner.start()
    try:
        print(
            json.dumps(
                {
                    'ok': True,
                    'event': 'preview_service_started',
                    'host': args.host,
                    'port': args.port,
                    **service_metadata,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        runner.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        runner.stop()


def _service_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        'backend': str(args.backend),
        'model_path': str(args.model_path),
        'hf_repo_id': str(args.hf_repo_id) if args.hf_repo_id else None,
        'device': str(args.device),
        'compute_type': str(args.compute_type),
    }


def _build_processor(args: argparse.Namespace) -> StreamingPreviewSessionManager:
    backend = str(args.backend or 'mlx_whisper').strip().lower()
    if backend in {'mlx_whisper', 'mlx_whisper_streaming'}:
        decoder: StreamingPreviewDecoder = MlxWhisperStreamingDecoder(
            str(args.model_path),
            hf_repo_id=args.hf_repo_id,
        )
    else:
        whisper_model = _load_faster_whisper_model(
            Path(args.model_path), device=args.device, compute_type=args.compute_type
        )
        decoder = FasterWhisperStreamingDecoder(whisper_model)
    return StreamingPreviewSessionManager(
        decoder,
        sample_rate_hz=args.sample_rate_hz,
        decode_cadence_ms=float(args.decode_cadence_ms),
        min_decode_audio_ms=float(args.min_decode_audio_ms),
        force_trim_max_seconds=float(args.force_trim_max_seconds),
        force_trim_carry_seconds=float(args.force_trim_carry_seconds),
        max_session_idle_seconds=float(args.max_session_idle_seconds),
        vad_enabled=bool(int(args.vad_enabled)),
    )


def _load_faster_whisper_model(model_path: Path, *, device: str, compute_type: str):
    if not model_path.exists():
        raise FileNotFoundError(
            f'Streaming preview model path does not exist: {model_path}. '
            'Juno v2 requires local model artifacts and will not auto-download weights.'
        )
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('faster-whisper is required for the local streaming preview service') from exc
    return WhisperModel(str(model_path), device=device, compute_type=compute_type)


def _decode_wav_bytes(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError('Only 16-bit PCM WAV is supported')
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio


def _extract_sample_rate_hz(data: bytes) -> int | None:
    with wave.open(io.BytesIO(data), 'rb') as wf:
        return int(wf.getframerate())


def _decode_b64_json(value: str | None, *, default):
    if not value:
        return default
    try:
        decoded = base64.b64decode(value.encode('ascii'))
        return json.loads(decoded.decode('utf-8'))
    except Exception:
        return default


def _header_bool(value: str | None) -> bool:
    return (value or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _header_int(value: str | None) -> int:
    try:
        return int((value or '0').strip())
    except ValueError:
        return 0


if __name__ == '__main__':
    main()
