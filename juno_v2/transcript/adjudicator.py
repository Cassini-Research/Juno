from __future__ import annotations

import json
import os
import re
import sys
import time
import difflib
from dataclasses import dataclass, field, replace
from typing import Any  # noqa: F401 — used below for dict[str, Any]

from juno_v2.context.compiler import TranscriptAdjudicationPacket
from juno_v2.contracts.tracing import TraceKind
from juno_v2.contracts.writer import WriterMode, WriterTransformRequest
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.runtime.deployment import _env_bool, _env_int
from juno_v2.transcript.contracts import PatchOpType, PatchReason, TranscriptAdjudicationResult, TranscriptPatchOp
from juno_v2.transcript.patching import (
    diff_to_patch_ops,
    live_patch_is_safe,
    stable_prefix_chars,
    visible_text_hash,
)
from juno_v2.transcript.validators import (
    remove_instructional_exclusion_phrases,
    repair_low_signal_mid_sentence_capitalization,
    restore_explicit_final_word_tail,
    validate_adjudication_result,
)
from juno_v2.writer.backends.base import WriterBackend

_PATCH_OP_TYPES: dict[str, PatchOpType] = {
    "replace": "replace",
    "insert": "insert",
    "delete": "delete",
    "punctuate": "punctuate",
    "case": "case",
}
_PATCH_REASONS: dict[str, PatchReason] = {
    "asr_correction": "asr_correction",
    "memory_alias": "memory_alias",
    "user_replacement": "user_replacement",
    "screen_term": "screen_term",
    "file_or_symbol": "file_or_symbol",
    "self_correction": "self_correction",
    "spoken_punctuation": "spoken_punctuation",
    "itn": "itn",
    "capitalization": "capitalization",
    "spacing": "spacing",
}


@dataclass(slots=True)
class TranscriptAdjudicatorConfig:
    enabled: bool = True
    live_enabled: bool = field(default_factory=lambda: _live_corrector_enabled_default())
    max_live_ops: int = 8
    max_output_chars_multiplier: float = 1.6
    temperature: float = 0.0
    max_tokens_final: int = 1024
    max_tokens_live: int = 160
    final_chunking_enabled: bool = field(default_factory=lambda: _env_bool("JUNO_V2_FINAL_ADJUDICATION_CHUNKING", True))
    max_final_chunk_words: int = field(default_factory=lambda: _env_int("JUNO_V2_FINAL_ADJUDICATION_CHUNK_WORDS", 240))
    min_final_chunk_words: int = field(default_factory=lambda: _env_int("JUNO_V2_FINAL_ADJUDICATION_CHUNK_MIN_WORDS", 280))


def _live_corrector_enabled_default() -> bool:
    """New live-corrector gate with the old env var kept as a compatibility alias."""

    import os

    if os.getenv("JUNO_V2_LIVE_CORRECTOR_ENABLED") is not None:
        return _env_bool("JUNO_V2_LIVE_CORRECTOR_ENABLED", True)
    return _env_bool("JUNO_V2_LIVE_QWEN_ADJUDICATION", True)


class TranscriptAdjudicator:
    def __init__(
        self,
        backend: WriterBackend | None,
        recorder: TraceRecorder,
        config: TranscriptAdjudicatorConfig | None = None,
    ) -> None:
        self.backend = backend
        self.recorder = recorder
        self.config = config or TranscriptAdjudicatorConfig()

    def adjudicate(self, packet: TranscriptAdjudicationPacket) -> TranscriptAdjudicationResult:
        fallback = (packet.memory_candidate_text or packet.whisper_text or packet.raw_text or "").strip()
        base_hash = visible_text_hash(packet.base_visible_text or packet.live_preview_text or "")
        visible = packet.base_visible_text or packet.live_preview_text or ""
        stable_chars = (
            _live_stable_prefix_chars(visible)
            if packet.stage == "live"
            else stable_prefix_chars(visible)
        )
        live_preview_excerpt = (packet.live_preview_text or "").replace("\n", " ")[:80]
        whisper_excerpt = (packet.whisper_text or "").replace("\n", " ")[:80]
        print(
            f"[ADJ]         adjudicate_start utt={packet.utterance_id[:8]} "
            f"stage={packet.stage} live_preview={live_preview_excerpt!r} whisper={whisper_excerpt!r}",
            file=sys.stderr,
            flush=True,
        )

        if not self.config.enabled:
            print(f"[ADJ]         skipped utt={packet.utterance_id[:8]} reason=disabled", file=sys.stderr, flush=True)
            return self._rejected(packet, fallback, "disabled", base_hash=base_hash, stable_chars=stable_chars)
        if packet.stage == "live" and not self.config.live_enabled:
            print(f"[ADJ]         skipped utt={packet.utterance_id[:8]} reason=live_disabled", file=sys.stderr, flush=True)
            return self._rejected(packet, fallback, "live_disabled", base_hash=base_hash, stable_chars=stable_chars)
        if self.backend is None or not callable(getattr(self.backend, "rewrite", None)):
            print(f"[ADJ]         skipped utt={packet.utterance_id[:8]} reason=backend_unavailable", file=sys.stderr, flush=True)
            return self._rejected(packet, fallback, "backend_unavailable", base_hash=base_hash, stable_chars=stable_chars)

        if packet.stage == "final" and _should_chunk_final_adjudication(packet, self.config):
            return self._adjudicate_final_chunked(
                packet,
                fallback=fallback,
                base_hash=base_hash,
                stable_chars=stable_chars,
            )

        started = time.perf_counter()
        payload = packet.to_payload()
        payload["base_visible"]["hash"] = base_hash
        payload["base_visible"]["stable_prefix_chars"] = stable_chars
        live_window: tuple[int, int] | None = None
        source_text = packet.memory_candidate_text or packet.whisper_text or packet.raw_text
        if packet.stage == "live":
            payload, live_window, source_text = _window_live_payload(packet, payload, stable_chars)
        task_kind = "live_transcript_correction_v1" if packet.stage == "live" else "transcript_adjudication_v1"
        req = WriterTransformRequest(
            utterance_id=packet.utterance_id,
            instruction=(
                "Correct only the target live transcript window."
                if packet.stage == "live"
                else "Return strict JSON for transcript_adjudication_v1."
            ),
            source_text=source_text,
            mode=WriterMode.DEFAULT_SURFACE,
            context_payload={
                "task": task_kind,
                "schema_version": "transcript_adjudication_v1" if packet.stage != "live" else "live_transcript_correction_v1",
                "payload": payload,
            },
            metadata={
                "kind": task_kind,
                "stage": packet.stage,
                "max_tokens": self.config.max_tokens_live if packet.stage == "live" else self.config.max_tokens_final,
            },
        )

        try:
            result = self.backend.rewrite(req)
        except Exception as exc:  # noqa: BLE001
            self._record("oneshot_transcript_adjudication_rejected", {
                "utterance_id": packet.utterance_id,
                "reason": "backend_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(
                f"[ADJ]         adjudicate_done  utt={packet.utterance_id[:8]} "
                f"stage={packet.stage} rejected=True reason=backend_error "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return self._rejected(packet, fallback, "backend_error", base_hash=base_hash, stable_chars=stable_chars)

        decode_ms = float(getattr(result, "decode_ms", 0.0) or ((time.perf_counter() - started) * 1000.0))
        raw = str(getattr(result, "text", "") or "")
        obj = _strict_json_or_extract_once(raw)
        if packet.stage == "live" and obj is None:
            parsed = self._parse_live_text_result(
                packet,
                raw,
                fallback=fallback,
                base_hash=base_hash,
                stable_chars=stable_chars,
                backend_name=getattr(result, "backend_name", None),
                decode_ms=decode_ms,
                live_window=live_window,
            )
            if parsed.rejected:
                return parsed
            ok, reason = validate_adjudication_result(packet, parsed)
            if not ok:
                parsed.rejected = True
                parsed.rejected_reason = reason
                return parsed
            safe, safe_reason = live_patch_is_safe(
                packet.base_visible_text or packet.live_preview_text or "",
                list(parsed.ops),
                stable_prefix_chars=stable_chars,
            )
            if not safe:
                parsed.rejected = True
                parsed.rejected_reason = f"unsafe_live_patch:{safe_reason}"
                return parsed
            corrected_preview = (parsed.corrected_text or "").replace("\n", " ")[:120]
            print(
                f"[ADJ]         adjudicate_done  utt={packet.utterance_id[:8]} "
                f"stage=live rejected=False ops={len(parsed.ops)} "
                f"conf={parsed.confidence:.2f} decode_ms={decode_ms:.1f} "
                f"corrected={corrected_preview!r}",
                file=sys.stderr,
                flush=True,
            )
            return parsed
        if obj is None:
            return self._rejected(
                packet,
                fallback,
                "invalid_json",
                base_hash=base_hash,
                stable_chars=stable_chars,
                backend_name=getattr(result, "backend_name", None),
                decode_ms=decode_ms,
            )
        parsed = self._parse_result(
            packet,
            obj,
            fallback=fallback,
            base_hash=base_hash,
            stable_chars=stable_chars,
            backend_name=getattr(result, "backend_name", None),
            decode_ms=decode_ms,
            live_window=live_window,
        )
        if packet.stage == "final":
            repaired_text, capitalization_repairs = repair_low_signal_mid_sentence_capitalization(
                packet,
                parsed.corrected_text,
            )
            if capitalization_repairs and repaired_text != parsed.corrected_text:
                parsed.corrected_text = repaired_text
                parsed.metadata = dict(parsed.metadata or {})
                parsed.metadata["low_signal_capitalization_repairs"] = capitalization_repairs[:16]
            exclusion_text, exclusion_repairs = remove_instructional_exclusion_phrases(
                packet,
                parsed.corrected_text,
            )
            if exclusion_repairs and exclusion_text != parsed.corrected_text:
                parsed.corrected_text = exclusion_text
                parsed.metadata = dict(parsed.metadata or {})
                parsed.metadata["instructional_exclusion_repairs"] = exclusion_repairs[:8]
            tail_text, tail_restore = restore_explicit_final_word_tail(packet, parsed.corrected_text)
            if tail_restore and tail_text != parsed.corrected_text:
                parsed.corrected_text = tail_text
                parsed.metadata = dict(parsed.metadata or {})
                parsed.metadata["explicit_final_word_tail_restore"] = tail_restore
        ok, reason = validate_adjudication_result(packet, parsed)
        if not ok:
            parsed.rejected = True
            parsed.rejected_reason = reason
            print(
                f"[ADJ]         adjudicate_done  utt={packet.utterance_id[:8]} "
                f"stage={packet.stage} rejected=True reason={reason} "
                f"decode_ms={decode_ms:.1f}",
                file=sys.stderr,
                flush=True,
            )
            return parsed
        if packet.stage == "live":
            safe, safe_reason = live_patch_is_safe(
                packet.base_visible_text or packet.live_preview_text or "",
                list(parsed.ops),
                stable_prefix_chars=stable_chars,
            )
            if not safe:
                parsed.rejected = True
                parsed.rejected_reason = f"unsafe_live_patch:{safe_reason}"
                print(
                    f"[ADJ]         adjudicate_done  utt={packet.utterance_id[:8]} "
                    f"stage=live rejected=True reason=unsafe_live_patch:{safe_reason} "
                    f"decode_ms={decode_ms:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
                return parsed
        corrected_preview = (parsed.corrected_text or "").replace("\n", " ")[:120]
        print(
            f"[ADJ]         adjudicate_done  utt={packet.utterance_id[:8]} "
            f"stage={packet.stage} rejected=False ops={len(parsed.ops)} "
            f"conf={parsed.confidence:.2f} decode_ms={decode_ms:.1f} "
            f"corrected={corrected_preview!r}",
            file=sys.stderr,
            flush=True,
        )
        return parsed

    def _adjudicate_final_chunked(
        self,
        packet: TranscriptAdjudicationPacket,
        *,
        fallback: str,
        base_hash: str,
        stable_chars: int,
    ) -> TranscriptAdjudicationResult:
        source = _final_adjudication_source_text(packet)
        source_words = _word_count(source)
        ranges = _final_chunk_ranges(source, max_words=max(20, int(self.config.max_final_chunk_words or 70)))
        if len(ranges) <= 1:
            return self._rejected(packet, fallback, "chunking_not_applicable", base_hash=base_hash, stable_chars=stable_chars)

        self._record(
            "oneshot_transcript_adjudication_chunking_started",
            {
                "utterance_id": packet.utterance_id,
                "source_words": source_words,
                "source_chars": len(source),
                "chunk_count": len(ranges),
                "max_chunk_words": max(20, int(self.config.max_final_chunk_words or 70)),
            },
        )

        corrected_chunks: list[str] = []
        rejected_chunks: list[dict[str, Any]] = []
        backend_names: list[str] = []
        total_decode_ms = 0.0
        accepted = 0

        for idx, (char_start, char_end, word_start, word_end) in enumerate(ranges):
            chunk_source = source[char_start:char_end].strip()
            if not chunk_source:
                continue
            base_chunk = _slice_by_word_range(packet.base_visible_text or packet.live_preview_text, word_start, word_end)
            live_preview_chunk = _slice_by_word_range(packet.live_preview_text or packet.base_visible_text, word_start, word_end)
            whisper_chunk = _slice_by_word_range(packet.whisper_text, word_start, word_end) or chunk_source
            raw_chunk = _slice_by_word_range(packet.raw_text, word_start, word_end) or whisper_chunk or chunk_source
            memory_chunk = _slice_by_word_range(packet.memory_candidate_text, word_start, word_end) or chunk_source
            protected_terms = _protected_terms_for_chunk(packet.protected_terms, base_chunk, live_preview_chunk, whisper_chunk, memory_chunk, raw_chunk)
            chunk_packet = replace(
                packet,
                utterance_id=f"{packet.utterance_id}:chunk{idx + 1}",
                base_visible_text=base_chunk,
                live_preview_text=live_preview_chunk,
                whisper_text=whisper_chunk,
                memory_candidate_text=memory_chunk,
                raw_text=raw_chunk,
                protected_terms=protected_terms,
                metadata={
                    **dict(packet.metadata or {}),
                    "final_adjudication_chunk": {
                        "index": idx,
                        "count": len(ranges),
                        "source_word_start": word_start,
                        "source_word_end": word_end,
                        "source_char_start": char_start,
                        "source_char_end": char_end,
                    },
                },
            )
            result = self.adjudicate(chunk_packet)
            total_decode_ms += float(getattr(result, "decode_ms", 0.0) or 0.0)
            backend = str(getattr(result, "backend_name", "") or "")
            if backend and backend not in backend_names:
                backend_names.append(backend)
            text = str(getattr(result, "corrected_text", "") or "").strip()
            if getattr(result, "rejected", False):
                rejected_chunks.append(
                    {
                        "index": idx,
                        "reason": getattr(result, "rejected_reason", None),
                        "chars": len(chunk_source),
                    }
                )
                text = chunk_source
            else:
                accepted += 1
            corrected_chunks.append(text)

        corrected = _join_corrected_chunks(corrected_chunks)
        if not corrected:
            return self._rejected(packet, fallback, "empty_chunked_corrected_text", base_hash=base_hash, stable_chars=stable_chars, decode_ms=total_decode_ms)
        if accepted == 0:
            return self._rejected(
                packet,
                fallback,
                "all_chunks_rejected",
                base_hash=base_hash,
                stable_chars=stable_chars,
                backend_name=",".join(backend_names) if backend_names else None,
                decode_ms=total_decode_ms,
            )

        ops = tuple(
            diff_to_patch_ops(
                source,
                corrected,
                stable_prefix_chars=max(len(source), len(corrected)),
                reason="asr_correction",
                confidence=0.85,
            )
        )
        merged = TranscriptAdjudicationResult(
            utterance_id=packet.utterance_id,
            stage=packet.stage,
            corrected_text=corrected,
            ops=ops,
            confidence=0.85 if not rejected_chunks else 0.75,
            base_visible_revision=packet.base_visible_revision,
            base_text_hash=base_hash,
            base_visible_text=packet.base_visible_text or packet.live_preview_text or "",
            stable_prefix_chars=stable_chars,
            protected_terms_used=tuple(packet.protected_terms),
            backend_name=",".join(backend_names) if backend_names else getattr(self.backend, "backend_name", None),
            decode_ms=total_decode_ms,
            metadata={
                "chunked_final_adjudication": True,
                "chunk_count": len(ranges),
                "accepted_chunks": accepted,
                "rejected_chunks": rejected_chunks,
            },
        )
        ok, reason = validate_adjudication_result(
            packet,
            merged,
            allow_chunked_insertions=bool(merged.metadata.get("chunked_final_adjudication")),
        )
        if not ok:
            self._record(
                "oneshot_transcript_adjudication_chunking_rejected",
                {
                    "utterance_id": packet.utterance_id,
                    "reason": reason,
                    "chunk_count": len(ranges),
                    "accepted_chunks": accepted,
                    "rejected_chunks": rejected_chunks,
                    "decode_ms": total_decode_ms,
                },
            )
            return self._rejected(
                packet,
                fallback,
                f"chunked_validation_failed:{reason}",
                base_hash=base_hash,
                stable_chars=stable_chars,
                backend_name=merged.backend_name,
                decode_ms=total_decode_ms,
            )

        self._record(
            "oneshot_transcript_adjudication_chunking_done",
            {
                "utterance_id": packet.utterance_id,
                "chunk_count": len(ranges),
                "accepted_chunks": accepted,
                "rejected_chunks": rejected_chunks,
                "decode_ms": total_decode_ms,
                "corrected_chars": len(corrected),
            },
        )
        corrected_preview = corrected.replace("\n", " ")[:120]
        print(
            f"[ADJ]         adjudicate_done  utt={packet.utterance_id[:8]} "
            f"stage=final chunked=True rejected=False chunks={len(ranges)} "
            f"accepted={accepted} decode_ms={total_decode_ms:.1f} "
            f"corrected={corrected_preview!r}",
            file=sys.stderr,
            flush=True,
        )
        return merged

    def _parse_result(
        self,
        packet: TranscriptAdjudicationPacket,
        obj: dict[str, Any],
        *,
        fallback: str,
        base_hash: str,
        stable_chars: int,
        backend_name: str | None,
        decode_ms: float,
        live_window: tuple[int, int] | None = None,
    ) -> TranscriptAdjudicationResult:
        schema_version = str(obj.get("schema_version") or "").strip()
        if schema_version and schema_version != "transcript_adjudication_v1":
            return self._rejected(packet, fallback, "invalid_schema", base_hash=base_hash, stable_chars=stable_chars, backend_name=backend_name, decode_ms=decode_ms)
        corrected = str(obj.get("corrected_text") or "").strip()
        if not corrected:
            return self._rejected(packet, fallback, "empty_corrected_text", base_hash=base_hash, stable_chars=stable_chars, backend_name=backend_name, decode_ms=decode_ms)
        confidence = _clamp_float(obj.get("confidence"), default=0.0)
        ops = _parse_ops(obj.get("ops"))
        if packet.stage == "live" and live_window is not None:
            start, end = live_window
            base = packet.base_visible_text or packet.live_preview_text or ""
            window_text = base[start:end]
            corrected_window, reject_reason = _validate_live_window_correction(
                corrected,
                window_text=window_text,
                base=base,
                window_start=start,
            )
            if reject_reason is not None:
                return self._rejected(
                    packet,
                    fallback,
                    reject_reason,
                    base_hash=base_hash,
                    stable_chars=stable_chars,
                    backend_name=backend_name,
                    decode_ms=decode_ms,
                )
            corrected = _splice_window(base, start, end, corrected_window)
            ops = _shift_window_ops_if_local(ops, window_start=start, window_len=max(0, end - start))
            if not ops:
                ops = tuple(
                    _diff_window_to_patch_ops(
                        base,
                        start=start,
                        end=end,
                        corrected_window=corrected_window,
                        reason="asr_correction",
                        confidence=confidence or 0.8,
                    )
                )
        if not ops:
            if packet.stage == "live":
                ops = tuple(
                    diff_to_patch_ops(
                        packet.base_visible_text or packet.live_preview_text or "",
                        corrected,
                        stable_prefix_chars=stable_chars,
                        reason="asr_correction",
                        confidence=confidence or 0.8,
                    )
                )
            else:
                ops = tuple(
                    diff_to_patch_ops(
                        packet.memory_candidate_text or packet.whisper_text or packet.raw_text,
                        corrected,
                        stable_prefix_chars=max(len(packet.memory_candidate_text or packet.whisper_text or packet.raw_text), len(corrected)),
                        reason="asr_correction",
                        confidence=confidence or 0.8,
                    )
                )
        protected = obj.get("protected_terms_used")
        protected_terms_used = tuple(str(x).strip() for x in protected if str(x).strip()) if isinstance(protected, list) else ()
        resolution_metadata = {
            "notes": obj.get("notes"),
            "schema_inferred": not bool(schema_version),
        }
        for key in (
            "intent",
            "formatting_plan",
            "self_corrections",
            "terms_used",
            "uncertainty",
        ):
            value = obj.get(key)
            if value is not None:
                resolution_metadata[key] = value
        return TranscriptAdjudicationResult(
            utterance_id=packet.utterance_id,
            stage=packet.stage,
            corrected_text=corrected,
            ops=ops,
            confidence=confidence,
            base_visible_revision=packet.base_visible_revision,
            base_text_hash=base_hash,
            base_visible_text=packet.base_visible_text or packet.live_preview_text or "",
            stable_prefix_chars=stable_chars,
            protected_terms_used=protected_terms_used,
            backend_name=backend_name,
            decode_ms=decode_ms,
            metadata=resolution_metadata,
        )

    def _parse_live_text_result(
        self,
        packet: TranscriptAdjudicationPacket,
        output: str,
        *,
        fallback: str,
        base_hash: str,
        stable_chars: int,
        backend_name: str | None,
        decode_ms: float,
        live_window: tuple[int, int] | None,
    ) -> TranscriptAdjudicationResult:
        if _looks_like_live_prompt_echo(output):
            return self._rejected(
                packet,
                fallback,
                "live_prompt_echo",
                base_hash=base_hash,
                stable_chars=stable_chars,
                backend_name=backend_name,
                decode_ms=decode_ms,
            )
        corrected_window = _clean_live_text_output(output)
        if not corrected_window:
            return self._rejected(packet, fallback, "empty_corrected_text", base_hash=base_hash, stable_chars=stable_chars, backend_name=backend_name, decode_ms=decode_ms)
        if _looks_like_broken_structured_output(corrected_window):
            return self._rejected(packet, fallback, "invalid_json", base_hash=base_hash, stable_chars=stable_chars, backend_name=backend_name, decode_ms=decode_ms)

        base = packet.base_visible_text or packet.live_preview_text or ""
        if live_window is None:
            corrected = corrected_window
        else:
            start, end = live_window
            window_text = base[start:end]
            if _looks_like_full_live_transcript_output(
                output=corrected_window,
                base=base,
                window_text=window_text,
                window_start=start,
            ):
                return self._rejected(
                    packet,
                    fallback,
                    "live_output_not_window_scoped",
                    base_hash=base_hash,
                    stable_chars=stable_chars,
                    backend_name=backend_name,
                    decode_ms=decode_ms,
                )
            corrected_window, reject_reason = _validate_live_window_correction(
                corrected_window,
                window_text=window_text,
                base=base,
                window_start=start,
            )
            if reject_reason is not None:
                return self._rejected(
                    packet,
                    fallback,
                    reject_reason,
                    base_hash=base_hash,
                    stable_chars=stable_chars,
                    backend_name=backend_name,
                    decode_ms=decode_ms,
                )
            corrected = _splice_window(base, start, end, corrected_window)
            ops = tuple(
                _diff_window_to_patch_ops(
                    base,
                    start=start,
                    end=end,
                    corrected_window=corrected_window,
                    reason="asr_correction",
                    confidence=0.8,
                )
            )
        if live_window is None:
            ops = tuple(
                diff_to_patch_ops(
                    base,
                    corrected,
                    stable_prefix_chars=stable_chars,
                    reason="asr_correction",
                    confidence=0.8,
                )
            )
        return TranscriptAdjudicationResult(
            utterance_id=packet.utterance_id,
            stage=packet.stage,
            corrected_text=corrected,
            ops=ops,
            confidence=0.8,
            base_visible_revision=packet.base_visible_revision,
            base_text_hash=base_hash,
            base_visible_text=base,
            stable_prefix_chars=stable_chars,
            protected_terms_used=(),
            backend_name=backend_name,
            decode_ms=decode_ms,
            metadata={"live_text_only": True},
        )

    def _rejected(
        self,
        packet: TranscriptAdjudicationPacket,
        fallback: str,
        reason: str,
        *,
        base_hash: str,
        stable_chars: int,
        backend_name: str | None = None,
        decode_ms: float = 0.0,
    ) -> TranscriptAdjudicationResult:
        return TranscriptAdjudicationResult(
            utterance_id=packet.utterance_id,
            stage=packet.stage,
            corrected_text=fallback,
            ops=(),
            confidence=0.0,
            base_visible_revision=packet.base_visible_revision,
            base_text_hash=base_hash,
            base_visible_text=packet.base_visible_text or packet.live_preview_text or "",
            stable_prefix_chars=stable_chars,
            protected_terms_used=(),
            rejected=True,
            rejected_reason=reason,
            backend_name=backend_name,
            decode_ms=decode_ms,
        )

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self.recorder.record(TraceKind.SYSTEM, event, payload)
        except Exception:
            pass


def _should_chunk_final_adjudication(packet: TranscriptAdjudicationPacket, config: TranscriptAdjudicatorConfig) -> bool:
    if not bool(getattr(config, "final_chunking_enabled", True)):
        return False
    if packet.stage != "final":
        return False
    source = _final_adjudication_source_text(packet)
    if not source.strip():
        return False
    min_words = max(1, int(getattr(config, "min_final_chunk_words", 90) or 90))
    max_words = max(20, int(getattr(config, "max_final_chunk_words", 70) or 70))
    # Chunking only fires when the source is large enough to split into ≥2
    # chunks AFTER the min-tail-absorption floor. Below that threshold the
    # chunker would collapse to a single range and the chunked path would
    # have nothing to do — fall through to the single-pass adjudicate instead.
    splittable_threshold = max_words + _MIN_FINAL_CHUNK_WORDS
    return _word_count(source) >= max(min_words, max_words + 1, splittable_threshold)


def _final_adjudication_source_text(packet: TranscriptAdjudicationPacket) -> str:
    return (packet.memory_candidate_text or packet.whisper_text or packet.raw_text or "").strip()


def _word_count(text: str) -> int:
    return sum(1 for _ in _word_spans(text or ""))


# A final chunk shorter than this is not worth a Qwen call on its own — too
# little context, and Qwen will "rewrite" the fragment into something
# unrelated (observed in production: source "no" -> "emission"). When the
# tail of a long transcript would land below this threshold we absorb it
# into the preceding chunk, so every chunk still goes through the full final
# adjudication pass — we just never present Qwen with a fragment.
_MIN_FINAL_CHUNK_WORDS = 12


def _final_chunk_ranges(text: str, *, max_words: int) -> list[tuple[int, int, int, int]]:
    spans = list(_word_spans(text or ""))
    if not spans:
        return []
    max_words = max(20, int(max_words or 70))
    total = len(spans)
    ranges: list[tuple[int, int, int, int]] = []
    start_word = 0
    while start_word < total:
        end_word = min(total, start_word + max_words)
        if end_word < total:
            end_word = _prefer_sentence_boundary(text, spans, start_word, end_word, max_words=max_words)
        # If the remaining tail after this chunk would be a fragment shorter
        # than the minimum, absorb the tail into this chunk so no fragment
        # ever reaches Qwen.
        if 0 < total - end_word < _MIN_FINAL_CHUNK_WORDS:
            end_word = total
        char_start = spans[start_word][0]
        char_end = spans[end_word - 1][1]
        ranges.append((char_start, char_end, start_word, end_word))
        start_word = end_word
    return ranges


def _prefer_sentence_boundary(
    text: str,
    spans: list[tuple[int, int]],
    start_word: int,
    proposed_end_word: int,
    *,
    max_words: int,
) -> int:
    floor = start_word + max(20, int(max_words * 0.55))
    floor = min(floor, proposed_end_word)
    for word_idx in range(proposed_end_word - 1, floor - 1, -1):
        start, end = spans[word_idx]
        word = text[start:end].rstrip()
        if re.search(r"[.!?][\"')\]]?$", word):
            return word_idx + 1
        gap_end = spans[word_idx + 1][0] if word_idx + 1 < len(spans) else len(text)
        gap = text[end:gap_end]
        if re.search(r"[.!?][\"')\]]?\s+$", gap):
            return word_idx + 1
    return proposed_end_word


def _slice_by_word_range(text: str, start_word: int, end_word: int) -> str:
    value = text or ""
    spans = list(_word_spans(value))
    if not spans or start_word >= len(spans):
        return ""
    start = spans[max(0, start_word)][0]
    end = spans[min(len(spans), max(start_word + 1, end_word)) - 1][1]
    return value[start:end].strip()


def _join_corrected_chunks(chunks: list[str]) -> str:
    out = ""
    for chunk in chunks:
        text = (chunk or "").strip()
        if not text:
            continue
        if not out:
            out = text
            continue
        if text[:1] in {".", ",", "!", "?", ":", ";", ")", "]"}:
            out += text
        elif out.endswith(("-", "/", "\n")):
            out += text
        else:
            out += " " + text
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _protected_terms_for_chunk(protected_terms: tuple[str, ...], *evidence_parts: str) -> tuple[str, ...]:
    evidence = " ".join(str(part or "") for part in evidence_parts)
    out: list[str] = []
    for term in protected_terms or ():
        t = str(term or "").strip()
        if t and _term_present_loose(evidence, t):
            out.append(t)
    return tuple(out)


def _term_present_loose(text: str, term: str) -> bool:
    return _term_count_loose(text, term) > 0


def _term_count_loose(text: str, term: str) -> int:
    haystack = text or ""
    tokens = re.findall(r"[A-Za-z0-9]+", term or "")
    if not tokens:
        return 0
    separator = r"[\W_]*" if len(tokens) >= 2 and all(len(tok) == 1 and tok.isalpha() for tok in tokens) else r"[\W_]+"
    pattern = r"(?<![A-Za-z0-9])" + separator.join(re.escape(tok) for tok in tokens) + r"(?![A-Za-z0-9])"
    return len(re.findall(pattern, haystack, flags=re.IGNORECASE))


def _window_live_payload(
    packet: TranscriptAdjudicationPacket,
    payload: dict[str, Any],
    stable_chars: int,
    *,
    max_words: int = 25,
) -> tuple[dict[str, Any], tuple[int, int] | None, str]:
    base = packet.base_visible_text or packet.live_preview_text or packet.raw_text or ""
    if not base.strip():
        return payload, None, packet.memory_candidate_text or packet.whisper_text or packet.raw_text
    end = max(0, min(len(base), int(stable_chars or 0)))
    if end <= 0:
        end = len(base)
    start = _window_start_for_last_words(base[:end], max_words=max_words)
    window_text = base[start:end]
    trimmed = dict(payload)
    trimmed["asr"] = dict(payload.get("asr") or {})
    trimmed["context"] = dict(payload.get("context") or {})
    trimmed["base_visible"] = dict(payload.get("base_visible") or {})
    trimmed["output_schema"] = dict(payload.get("output_schema") or {})

    trimmed["base_visible"].update(
        {
            "text": window_text,
            "absolute_start_char": start,
            "absolute_end_char": end,
            "stable_prefix_chars": max(0, end - start),
            "corrected_text_scope": "target_window",
        }
    )
    trimmed["asr"]["live_preview_visible"] = window_text
    trimmed["asr"]["raw_text"] = _clip_to_window(packet.raw_text, window_text)
    trimmed["asr"]["memory_candidate"] = _clip_to_window(packet.memory_candidate_text, window_text)
    trimmed["asr"]["whisper_raw"] = _clip_to_window(packet.whisper_text, window_text)
    trimmed["terms"] = list(payload.get("terms") or [])[:16]
    trimmed["protected_terms"] = list(payload.get("protected_terms") or [])[:16]
    trimmed["context"] = {
        "app_name": packet.app_name,
        "app_category": packet.app_category,
        "window_title": packet.window_title,
        "focused_file_path": packet.focused_file_path,
        "symbol_under_cursor": packet.symbol_under_cursor,
    }
    trimmed["output_schema"]["corrected_text"] = "corrected target-window text only"
    trimmed["live_patch_contract"] = {
        "target_window_start_char": start,
        "target_window_end_char": end,
        "ops_may_use_window_relative_offsets": True,
        "return_noop_when_not_confident": True,
        "do_not_rewrite_unstable_tail": True,
    }
    return trimmed, (start, end), window_text


def _live_stable_prefix_chars(text: str) -> int:
    words = list(_word_spans(text or ""))
    if len(words) <= 2:
        return 0
    if len(words) <= 6:
        unstable = 1
    elif len(words) <= 12:
        unstable = 2
    else:
        unstable = 5
    cutoff_start, _ = words[-unstable]
    return cutoff_start


def _window_start_for_last_words(text: str, *, max_words: int) -> int:
    spans = list(_word_spans(text or ""))
    if len(spans) <= max_words:
        return 0
    return spans[-max_words][0]


def _word_spans(text: str):
    in_word = False
    start = 0
    for idx, ch in enumerate(text):
        if ch.isspace():
            if in_word:
                yield start, idx
                in_word = False
        elif not in_word:
            start = idx
            in_word = True
    if in_word:
        yield start, len(text)


def _clip_to_window(value: str, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    if len(text) <= max(len(fallback) + 80, 240):
        return text
    return fallback


def _shift_window_ops_if_local(
    ops: tuple[TranscriptPatchOp, ...],
    *,
    window_start: int,
    window_len: int,
) -> tuple[TranscriptPatchOp, ...]:
    if not ops or window_start <= 0:
        return ops
    if any(op.start_char < 0 or op.end_char > window_len for op in ops):
        return ops
    return tuple(
        TranscriptPatchOp(
            op=op.op,
            start_char=op.start_char + window_start,
            end_char=op.end_char + window_start,
            text=op.text,
            reason=op.reason,
            confidence=op.confidence,
            source_text=op.source_text,
            metadata=dict(op.metadata),
        )
        for op in ops
    )


def _diff_window_to_patch_ops(
    base: str,
    *,
    start: int,
    end: int,
    corrected_window: str,
    reason: str,
    confidence: float,
) -> list[TranscriptPatchOp]:
    window = (base or "")[max(0, start): max(0, end)]
    corrected = corrected_window or ""
    if window == corrected:
        return []
    word_ops = _diff_window_words_to_patch_ops(
        window,
        corrected,
        window_start=max(0, start),
        reason=reason,
        confidence=confidence,
    )
    if word_ops:
        return word_ops
    matcher = difflib.SequenceMatcher(a=window, b=corrected, autojunk=False)
    op_reason = _PATCH_REASONS.get(reason, "asr_correction")
    ops: list[TranscriptPatchOp] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            op: PatchOpType = "replace"
        elif tag == "insert":
            op = "insert"
        elif tag == "delete":
            op = "delete"
        else:
            continue
        abs_start = max(0, start) + i1
        abs_end = max(0, start) + i2
        ops.append(
            TranscriptPatchOp(
                op=op,
                start_char=abs_start,
                end_char=abs_end,
                text=corrected[j1:j2],
                reason=op_reason,
                confidence=max(0.0, min(1.0, float(confidence))),
                source_text=window[i1:i2] or None,
            )
        )
    return ops


def _diff_window_words_to_patch_ops(
    window: str,
    corrected: str,
    *,
    window_start: int,
    reason: str,
    confidence: float,
) -> list[TranscriptPatchOp]:
    base_words = [(s, e, window[s:e]) for s, e in _word_spans(window)]
    corrected_words = [(s, e, corrected[s:e]) for s, e in _word_spans(corrected)]
    if not base_words or not corrected_words:
        return []
    base_keys = [_fold_token(w) for _, _, w in base_words]
    corrected_keys = [_fold_token(w) for _, _, w in corrected_words]
    matcher = difflib.SequenceMatcher(a=base_keys, b=corrected_keys, autojunk=False)
    op_reason = _PATCH_REASONS.get(reason, "asr_correction")
    ops: list[TranscriptPatchOp] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Same folded tokens can still need punctuation/case repair; leave
            # those to the char fallback by returning no word ops.
            continue
        if tag == "insert":
            anchor = base_words[i1][0] if i1 < len(base_words) else len(window)
            replacement = " ".join(w for _, _, w in corrected_words[j1:j2])
            if not replacement:
                continue
            ops.append(
                TranscriptPatchOp(
                    op="insert",
                    start_char=window_start + anchor,
                    end_char=window_start + anchor,
                    text=replacement + (" " if i1 < len(base_words) else ""),
                    reason=op_reason,
                    confidence=max(0.0, min(1.0, float(confidence))),
                    source_text=None,
                )
            )
            continue
        start_char = base_words[i1][0]
        end_char = base_words[i2 - 1][1]
        replacement = " ".join(w for _, _, w in corrected_words[j1:j2])
        ops.append(
            TranscriptPatchOp(
                op="delete" if tag == "delete" else "replace",
                start_char=window_start + start_char,
                end_char=window_start + end_char,
                text=replacement,
                reason=op_reason,
                confidence=max(0.0, min(1.0, float(confidence))),
                source_text=window[start_char:end_char] or None,
            )
        )
    return ops


def _fold_token(token: str) -> str:
    return re.sub(r"\W+", "", (token or "").casefold(), flags=re.UNICODE)


def _splice_window(base: str, start: int, end: int, replacement: str) -> str:
    prefix = base[:start]
    suffix = base[end:]
    middle = replacement or ""
    if prefix and middle and not prefix[-1].isspace() and not middle[0].isspace():
        middle = " " + middle
    if suffix and middle and not middle[-1].isspace() and not suffix[0].isspace():
        middle = middle + " "
    return prefix + middle + suffix


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_ops(raw: Any) -> tuple[TranscriptPatchOp, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[TranscriptPatchOp] = []
    for item in raw[:32]:
        if not isinstance(item, dict):
            continue
        op = _PATCH_OP_TYPES.get(str(item.get("op") or "").strip().lower())
        if op is None:
            continue
        reason = _PATCH_REASONS.get(str(item.get("reason") or "asr_correction").strip(), "asr_correction")
        try:
            start = int(item.get("start_char", 0))
            end = int(item.get("end_char", start))
        except (TypeError, ValueError):
            continue
        out.append(
            TranscriptPatchOp(
                op=op,
                start_char=max(0, start),
                end_char=max(0, end),
                text=str(item.get("text") or ""),
                reason=reason,
                # Qwen frequently omits per-op confidence in its JSON output
                # even when it produces a clean correction. Defaulting to 0
                # caused every omitted-confidence op to be silently rejected
                # by the shell's >=0.72 threshold, so the HUD never showed
                # corrections it had actually computed. Trust the model's
                # patch when it didn't volunteer a confidence — it already
                # passed schema and stable-prefix safety checks.
                confidence=_clamp_float(item.get("confidence"), default=0.85),
                source_text=str(item.get("source_text")) if item.get("source_text") is not None else None,
                metadata=dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {},
            )
        )
    return tuple(out)


def _strict_json_or_extract_once(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # One extraction attempt only. The caller still sees invalid preamble as
    # rejection unless a complete JSON object can be sliced exactly.
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clean_live_text_output(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return ""
    if out.startswith("```"):
        out = out.strip("`").strip()
        if "\n" in out:
            first, rest = out.split("\n", 1)
            out = rest if first.strip().lower() in {"text", "plaintext", "txt", "json"} else out
        out = out.strip()
    for prefix in (
        "Corrected target window:",
        "Corrected window:",
        "Corrected text:",
        "Output:",
        "Here is the corrected text:",
        "Here's the corrected text:",
    ):
        if out.lower().startswith(prefix.lower()):
            out = out[len(prefix):].strip()
    return out.strip().strip('"').strip()


_LIVE_PROMPT_ECHO_PREFIXES = (
    "target window:",
    "target live transcript window:",
    "base visible:",
    "base_visible:",
    "input:",
    "instruction:",
)


_LIVE_STOCK_SILENCE_PHRASES = {
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "please subscribe",
    "subscribe",
    "bye",
    "goodbye",
    "okay",
    "ok",
}


def _looks_like_live_prompt_echo(text: str) -> bool:
    lowered = (text or "").strip().casefold()
    if not lowered:
        return False
    return any(lowered.startswith(prefix) for prefix in _LIVE_PROMPT_ECHO_PREFIXES)


def _validate_live_window_correction(
    corrected_window: str,
    *,
    window_text: str,
    base: str,
    window_start: int,
) -> tuple[str, str | None]:
    corrected = (corrected_window or "").strip()
    window = (window_text or "").strip()
    if not corrected:
        return corrected, "empty_corrected_text"
    if _looks_like_live_prompt_echo(corrected):
        return corrected, "live_prompt_echo"

    corrected_phrase = _fold_phrase(corrected)
    window_phrase = _fold_phrase(window)
    if corrected_phrase in _LIVE_STOCK_SILENCE_PHRASES and corrected_phrase not in window_phrase:
        return corrected, "live_stock_phrase_not_in_window"

    output_tokens = _scope_tokens(corrected)
    window_tokens = _scope_tokens(window)
    if not output_tokens or not window_tokens:
        return corrected, None

    if _tokens_start_with(output_tokens, window_tokens) and len(output_tokens) > len(window_tokens):
        return corrected, "live_output_added_continuation"
    if _tokens_start_with(window_tokens, output_tokens) and len(window_tokens) - len(output_tokens) >= 4:
        return corrected, "live_output_dropped_window_prefix"

    max_tokens = max(len(window_tokens) + 1, int(len(window_tokens) * 1.18) + 1)
    if len(output_tokens) > max_tokens:
        return corrected, "live_output_added_content"

    if len(window_tokens) >= 4:
        similarity = difflib.SequenceMatcher(
            a=window_tokens,
            b=output_tokens,
            autojunk=False,
        ).ratio()
        if similarity < 0.62:
            return corrected, "live_output_not_aligned_to_window"

    suffix = base[window_start + len(window_text) :]
    if suffix.strip():
        corrected = _strip_added_terminal_sentence_punctuation(corrected, window)
    return corrected, None


def _strip_added_terminal_sentence_punctuation(corrected: str, window_text: str) -> str:
    if not corrected:
        return corrected
    stripped = corrected.rstrip()
    if not stripped.endswith((".", "!", "?")):
        return corrected
    original = (window_text or "").rstrip()
    if original.endswith((".", "!", "?")):
        return corrected
    return stripped[:-1].rstrip()


def _scope_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").casefold(), flags=re.UNICODE)


def _tokens_start_with(tokens: list[str], prefix: list[str]) -> bool:
    return bool(prefix) and len(tokens) >= len(prefix) and tokens[: len(prefix)] == prefix


def _fold_phrase(text: str) -> str:
    return re.sub(r"[\s\.\!\?,;:'\"\-–—…]+", " ", (text or "").casefold()).strip()


def _looks_like_broken_structured_output(text: str) -> bool:
    stripped = (text or "").lstrip()
    if not stripped:
        return True
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    lowered = stripped[:160].lower()
    return "schema_version" in lowered or '"corrected_text"' in lowered or "'corrected_text'" in lowered


def _looks_like_full_live_transcript_output(
    *,
    output: str,
    base: str,
    window_text: str,
    window_start: int,
) -> bool:
    out = _fold_for_scope_check(output)
    window = _fold_for_scope_check(window_text)
    full = _fold_for_scope_check(base)
    if not out or not window:
        return True
    if out == window:
        return False
    if full and out == full:
        return True
    suffix = base[window_start + len(window_text) :].strip()
    if suffix:
        suffix_words = suffix.split()
        if len(suffix_words) >= 2:
            suffix_tail = _fold_for_scope_check(" ".join(suffix_words[-min(5, len(suffix_words)) :]))
            if suffix_tail and out.endswith(suffix_tail):
                return True
    if len(output) > max(len(window_text) + 80, int(len(window_text) * 1.8)):
        return True
    prefix = base[:window_start].strip()
    if prefix:
        base_head_words = base.split()[:4]
        if window_start > 0 and len(base_head_words) >= 2:
            head = _fold_for_scope_check(" ".join(base_head_words))
            if head and out.startswith(head):
                return True
        prefix_words = prefix.split()
        if len(prefix_words) >= 2:
            head = _fold_for_scope_check(" ".join(prefix_words[-4:]))
            if head and out.startswith(head):
                return True
    return False


def _fold_for_scope_check(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _clamp_float(value: Any, *, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        f = default
    return max(0.0, min(1.0, f))
