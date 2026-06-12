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


class StreamingLocalHttpJsonPreviewBackend(PreviewAsrBackend):
    """Streaming-native local HTTP path.

    Expects a local HTTP service with:
      GET  /healthz
      GET  /warm
      POST /preview_stream

    The backend keeps decoder state local to the service and uses utterance_id +
    decode_seq/reset flags to drive incremental decoding.
    """

    backend_name = 'streaming_local_http_json'

    def __init__(self, config: PreviewAsrConfig) -> None:
        self.config = config
        if not config.local_http_endpoint:
            raise ValueError('local_http_endpoint is required for StreamingLocalHttpJsonPreviewBackend')

    def warm(self) -> None:
        """Poll the service until the decoder is warm, it reports a startup
        error, or ``warm_deadline_sec`` expires.

        The first /warm on a cold service blocks while the model loads, so
        a single request bounded by ``local_http_timeout_sec`` routinely
        times out on a legitimate cold start — and treating that as fatal
        crash-loops the engine, killing the very load (or download) that
        was about to satisfy the warm. Per-request timeouts and transport
        errors here mean "not warm yet", not "failed"; only an explicit
        service-side startup error fails fast.
        """
        deadline = time.monotonic() + max(
            float(self.config.warm_deadline_sec),
            float(self.config.local_http_timeout_sec),
        )
        while True:
            payload = self._get_status('/warm')
            if payload.get('ok'):
                return
            error = payload.get('error')
            if error:
                raise RuntimeError(f'Local streaming preview ASR /warm failed: {error}')
            if time.monotonic() >= deadline:
                detail = 'service unreachable' if not payload else 'decoder still warming'
                raise RuntimeError(
                    f'Local streaming preview ASR /warm failed '
                    f'({detail} after {self.config.warm_deadline_sec:.0f}s)'
                )
            time.sleep(2.0)

    def healthcheck(self, *, strict: bool = False) -> bool:
        payload = self._get_status('/healthz', strict=strict)
        return bool(payload.get('ok', False))

    def _get_status(self, path: str, *, strict: bool = False) -> dict:
        health_url = self.config.local_http_endpoint.rstrip('/') + path
        req = request.Request(health_url, method='GET')
        try:
            with request.urlopen(req, timeout=self.config.local_http_timeout_sec) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode('utf-8'))
            ok = bool(payload.get('ok', False))
        except Exception:
            payload = {}
            ok = False
        if strict and not ok:
            raise RuntimeError(f'Local streaming preview ASR {path} failed')
        return payload

    def restart(self) -> None:
        # The local service owns decoder state; reset is expressed per utterance.
        return None

    def decode(self, req: PreviewDecodeRequest) -> PreviewDecodeResult:
        audio = np.asarray(req.audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        request_audio_ms = (int(audio.shape[0]) / max(int(req.sample_rate_hz or 16000), 1)) * 1000.0
        wav_bytes = encode_wav_bytes(audio, req.sample_rate_hz)
        started = time.perf_counter()
        http_req = request.Request(
            self.config.local_http_endpoint.rstrip('/') + '/preview_stream',
            data=wav_bytes,
            method='POST',
            headers={
                'Content-Type': 'audio/wav',
                'X-Juno-Language': req.language or self.config.language or '',
                'X-Juno-Allowed-Languages': base64.b64encode(json.dumps(req.allowed_languages).encode('utf-8')).decode('ascii'),
                'X-Juno-Language-Policy': req.language_policy or '',
                'X-Juno-Utterance-Id': req.utterance_id,
                'X-Juno-Is-Final': 'true' if req.is_final else 'false',
                'X-Juno-Decode-Seq': str(req.decode_seq),
                'X-Juno-Reset-Decoder-State': 'true' if req.reset_decoder_state else 'false',
                'X-Juno-Bias-Phrases': base64.b64encode(json.dumps(req.bias_phrases).encode('utf-8')).decode('ascii'),
                'X-Juno-Context': base64.b64encode(json.dumps(req.context_payload).encode('utf-8')).decode('ascii'),
            },
        )
        with request.urlopen(http_req, timeout=self.config.local_http_timeout_sec) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode('utf-8'))
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        return PreviewDecodeResult(
            utterance_id=req.utterance_id,
            text=str(payload.get('text', '')).strip(),
            start_ms=req.start_ms,
            end_ms=req.end_ms,
            audio_duration_ms=req.audio_duration_ms,
            is_final=req.is_final,
            backend_name=self.backend_name,
            language=payload.get('language', req.language or self.config.language),
            decode_ms=float(payload.get('decode_ms', roundtrip_ms)),
            metadata={
                'roundtrip_ms': roundtrip_ms,
                'raw': payload,
                'decode_seq': req.decode_seq,
                'request_audio_ms': request_audio_ms,
                # LocalAgreement-2 two-zone fields. These are the source of
                # truth for HUD rendering; ``text`` is the combined string for
                # legacy callers.
                'committed_text': str(payload.get('committed_text', '')),
                'tail_text': str(payload.get('tail_text', '')),
                'committed_word_count': payload.get('committed_word_count'),
                'tail_word_count': payload.get('tail_word_count'),
                'commit_events': payload.get('commit_events'),
                'segment_trim_events': payload.get('segment_trim_events'),
                'segment_trim_carry_seconds': payload.get('segment_trim_carry_seconds'),
                'force_trim_events': payload.get('force_trim_events'),
                'decode_attempts': payload.get('decode_attempts'),
                'decode_skipped_cadence': payload.get('decode_skipped_cadence'),
                'decode_skipped_too_short': payload.get('decode_skipped_too_short'),
                'decode_skipped_vad_silence': payload.get('decode_skipped_vad_silence'),
                'decode_silence_confirmations': payload.get('decode_silence_confirmations'),
                'decode_on_silence': payload.get('decode_on_silence'),
                'vad_admit_reason': payload.get('vad_admit_reason'),
                'decode_skip_reason': payload.get('decode_skip_reason'),
                'buffer_audio_ms': payload.get('buffer_audio_ms'),
                'buffer_start_t': payload.get('buffer_start_t'),
                'tail_suppressed_events': payload.get('tail_suppressed_events'),
                'tail_repeat_drop_events': payload.get('tail_repeat_drop_events'),
                'tail_repeat_drop_reason': payload.get('tail_repeat_drop_reason'),
                'committed_replay_agreement_drops': payload.get('committed_replay_agreement_drops'),
                'committed_replay_suppressed_events': payload.get('committed_replay_suppressed_events'),
                'committed_burst_budget_events': payload.get('committed_burst_budget_events'),
                'committed_burst_max_words': payload.get('committed_burst_max_words'),
                'tail_commit_quarantine_events': payload.get('tail_commit_quarantine_events'),
                'tail_display_suppress_reason': payload.get('tail_display_suppress_reason'),
                'tail_quarantine_reason': payload.get('tail_quarantine_reason'),
                'tail_suppress_reason': payload.get('tail_suppress_reason'),
                'committed_boh_strips': payload.get('committed_boh_strips'),
                'committed_boundary_letter_strips': payload.get('committed_boundary_letter_strips'),
                'tail_no_speech_prob': payload.get('tail_no_speech_prob'),
                'last_segment_no_speech_prob': payload.get('last_segment_no_speech_prob'),
                'preview_repair_terms': payload.get('preview_repair_terms'),
                'preview_repair_applied': payload.get('preview_repair_applied'),
                'preview_repairs': payload.get('preview_repairs'),
                'preview_orthography_applied': payload.get('preview_orthography_applied'),
                'preview_orthography_committed_changed': payload.get('preview_orthography_committed_changed'),
                'preview_orthography_tail_changed': payload.get('preview_orthography_tail_changed'),
                'avg_logprob': payload.get('avg_logprob'),
                'compression_ratio': payload.get('compression_ratio'),
                'tail_final_promotion_status': payload.get('tail_final_promotion_status'),
                'tail_final_promotion_reason': payload.get('tail_final_promotion_reason'),
                'tail_final_promotion_max_words': payload.get('tail_final_promotion_max_words'),
                'tail_final_promotion_events': payload.get('tail_final_promotion_events'),
                'tail_final_promotion_blocked_events': payload.get('tail_final_promotion_blocked_events'),
                'newly_committed_display_words': payload.get('newly_committed_display_words'),
                # P3.2: full committed text on final emit, so the lifecycle
                # artifact carries the actual HUD text without engine-log grep.
                'committed_text_final': payload.get('committed_text_final'),
                'decodes_since_last_commit': payload.get('decodes_since_last_commit'),
                'segments_dropped': payload.get('segments_dropped'),
                'dropped_segments_preview': payload.get('dropped_segments_preview'),
                'segment_count': payload.get('segment_count'),
                'preview_owner_queue_depth': payload.get('preview_owner_queue_depth'),
                'broker_queue_wait_ms': payload.get('broker_queue_wait_ms'),
                'mlx_whisper_decode_ms': payload.get('mlx_whisper_decode_ms'),
                'mlx_whisper_model_ref': payload.get('mlx_whisper_model_ref'),
                'mlx_whisper_model_source_type': payload.get('mlx_whisper_model_source_type'),
                'model_streaming_mode': payload.get('model_streaming_mode'),
                'model_is_native_streaming': payload.get('model_is_native_streaming'),
                'streaming_native': True,
            },
        )
