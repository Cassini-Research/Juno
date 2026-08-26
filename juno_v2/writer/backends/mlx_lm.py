from __future__ import annotations

import gc
import inspect
import json
import re
import sys
import time
from typing import Any

from juno_v2.contracts.writer import WriterTransformRequest, WriterTransformResult
from juno_v2.runtime.mlx_lock import mlx_decode_guard
from juno_v2.writer.backends.base import WriterBackend
from juno_v2.writer.config import WriterConfig

_ACTION_EXTRACTION_MAX_TOKENS = 1024
_V3_ACTION_EXTRACTION_MAX_TOKENS = _ACTION_EXTRACTION_MAX_TOKENS


class MlxLmWriterBackend(WriterBackend):
    backend_name = "mlx_lm"

    def __init__(self, config: WriterConfig) -> None:
        self.config = config
        self._loaded = False
        self._model = None
        self._tokenizer = None
        # Static-prefix KV caches keyed by prefix hash. Reserved for hot,
        # lean prefixes (the dictation editor) — KV memory is ~150KB/token
        # on Qwen3-4B, so each cached prefix must stay ~1k tokens.
        self._kv_caches: dict[str, dict[str, Any]] = {}
        self._split_prefix_memo: dict[str, str | None] = {}

    def _static_chat_prefix(self, system_prompt: str) -> str | None:
        """Rendered chat prefix (system + user-block opening) for caching."""
        key = str(hash(system_prompt))
        if key in self._split_prefix_memo:
            return self._split_prefix_memo[key]
        prefix: str | None = None
        try:
            if getattr(self._tokenizer, "chat_template", None) is not None:
                marker = "\x01JUNO_SPLIT\x01"
                rendered = _apply_chat_template_no_thinking(
                    self._tokenizer,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": marker},
                    ],
                )
                idx = rendered.find(marker)
                prefix = rendered[:idx] if idx > 0 else None
            else:
                prefix = system_prompt + "\n\n"
        except Exception:
            prefix = None
        self._split_prefix_memo[key] = prefix
        return prefix

    def _generate_with_prefix_cache(
        self,
        *,
        full_prompt: str,
        prefix_str: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        deadline_s: float | None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Generate using a reusable KV cache for ``prefix_str``.

        Returns ``None`` on any incompatibility (tokenizer boundary mismatch,
        non-trimmable cache, API drift) so the caller falls back to the
        plain path — caching must never be able to break production output.
        """
        try:
            import hashlib

            from mlx_lm import stream_generate  # type: ignore
            from mlx_lm.models.cache import (  # type: ignore
                can_trim_prompt_cache,
                make_prompt_cache,
                trim_prompt_cache,
            )
            from mlx_lm.sample_utils import make_sampler  # type: ignore

            tok = self._tokenizer
            full_ids = tok.encode(full_prompt)
            key = hashlib.sha256(prefix_str.encode("utf-8")).hexdigest()[:16]
            entry = self._kv_caches.get(key)
            if entry is None:
                prefix_ids = tok.encode(prefix_str)
                cache = make_prompt_cache(self._model)
                for _ in stream_generate(
                    self._model, tok, prompt=prefix_ids, max_tokens=1, prompt_cache=cache
                ):
                    pass
                if not can_trim_prompt_cache(cache):
                    return None
                offset = int(getattr(cache[0], "offset", 0))
                trim_prompt_cache(cache, max(0, offset - len(prefix_ids)))
                entry = {"ids": list(prefix_ids), "cache": cache}
                self._kv_caches[key] = entry
            prefix_ids = entry["ids"]
            if list(full_ids[: len(prefix_ids)]) != prefix_ids:
                return None
            suffix_ids = list(full_ids[len(prefix_ids):])
            if not suffix_ids:
                return None
            cache = entry["cache"]
            sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
            pieces: list[str] = []
            timed_out = False
            started = time.perf_counter()
            for resp in stream_generate(
                self._model,
                tok,
                prompt=suffix_ids,
                max_tokens=int(max_tokens),
                prompt_cache=cache,
                sampler=sampler,
            ):
                pieces.append(resp.text)
                if deadline_s is not None and (time.perf_counter() - started) > deadline_s:
                    timed_out = True
                    break
            # Rewind the cache to the static prefix for the next call.
            offset = int(getattr(cache[0], "offset", 0))
            trim_prompt_cache(cache, max(0, offset - len(prefix_ids)))
            return "".join(pieces), {
                "generation_api": "cached_stream",
                "prompt_cache_hit": True,
                "cached_prefix_tokens": len(prefix_ids),
                "suffix_tokens": len(suffix_ids),
                "timed_out": timed_out,
            }
        except Exception as exc:  # noqa: BLE001 — cache must never break output
            print(f"[WRITER]      prompt_cache_fallback err={exc}", file=sys.stderr, flush=True)
            self._kv_caches.clear()
            return None

    def warm(self) -> None:
        if self._loaded:
            return
        if not self.config.model_path:
            raise ValueError("model_path is required for MlxLmWriterBackend")
        try:
            from mlx_lm import load  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mlx-lm is required for the mlx_lm writer backend") from exc
        self._model, self._tokenizer = load(self.config.model_path)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import mlx.core as mx  # type: ignore
            mx.clear_cache()
        except Exception:
            pass
        gc.collect()

    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        self.warm()
        assert self._model is not None and self._tokenizer is not None
        try:
            from mlx_lm import generate  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("mlx-lm generate() is unavailable") from exc

        prompt = _build_writer_prompt(req)
        system_prompt = _system_prompt(req)
        if getattr(self._tokenizer, "chat_template", None) is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt = _apply_chat_template_no_thinking(self._tokenizer, messages)
        else:
            prompt = system_prompt + "\n\n" + prompt

        task = str((req.context_payload or {}).get("task") or req.metadata.get("kind") or "rewrite").strip()
        prompt_chars = len(prompt) if isinstance(prompt, str) else -1
        print(
            f"[WRITER]      rewrite_start utt={req.utterance_id[:8]} "
            f"task={task} prompt_chars={prompt_chars} "
            f"max_tokens={self.config.max_tokens} temp={self.config.temperature}",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        max_tokens = _max_tokens_for_request(req, self.config.max_tokens)
        deadline_ms = (req.metadata or {}).get("deadline_ms")
        deadline_s = (float(deadline_ms) / 1000.0) if deadline_ms else None
        cached_out: tuple[str, dict[str, Any]] | None = None
        # Issue #7: serialize MLX decode against preview/final lanes on
        # other threads. Without the guard, overlapping decodes from
        # different worker threads can crash with
        # ``RuntimeError: There is no Stream(gpu, N) in current thread``.
        with mlx_decode_guard():
            if (req.metadata or {}).get("cache_prefix"):
                prefix_str = self._static_chat_prefix(system_prompt)
                if prefix_str:
                    cached_out = self._generate_with_prefix_cache(
                        full_prompt=prompt,
                        prefix_str=prefix_str,
                        max_tokens=max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        deadline_s=deadline_s,
                    )
            if cached_out is not None:
                text, generation_meta = cached_out
            else:
                text, generation_meta = _generate_with_sampling(
                    generate,
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    deadline_s=deadline_s,
                )
        decode_ms = (time.perf_counter() - started) * 1000.0
        text_preview = (str(text or "")).replace("\n", " ")[:120]
        print(
            f"[WRITER]      rewrite_done  utt={req.utterance_id[:8]} "
            f"task={task} decode_ms={decode_ms:.1f} chars={len(str(text or ''))} "
            f"text={text_preview!r}",
            file=sys.stderr,
            flush=True,
        )
        return WriterTransformResult(
            utterance_id=req.utterance_id,
            text=_clean_writer_output(text, req=req),
            backend_name=self.backend_name,
            decode_ms=decode_ms,
            metadata={
                "writer_model_path": self.config.model_path,
                "max_tokens": max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "prompt_chars": prompt_chars,
                "output_chars": len(str(text or "")),
                "output_tokens_estimate": max(1, len(str(text or "")) // 4) if text else 0,
                **generation_meta,
            },
        )

    def extract_memory_candidates(
        self,
        *,
        text: str,
        kind: str,
        limit: int = 6,
    ) -> list[dict[str, Any]] | None:
        """Extract candidate memory entries from a transcript.

        ``kind`` is one of ``"vocab"`` | ``"snippet"`` | ``"replacement"``.
        Returns a list of ``{"term": str, "note": str}`` dicts (vocab) or
        the relevant shape for the other kinds. Returns ``None`` if the
        backend can't load mlx_lm, never raises.

        The macOS shell calls this from the Save Phrase / Add Snippet
        flows so the user sees real candidates instead of a regex
        approximation. We keep the prompt tight so cold-start dominates;
        decode itself is sub-second on Apple Silicon.
        """
        self.warm()
        assert self._model is not None and self._tokenizer is not None
        try:
            from mlx_lm import generate  # type: ignore
        except Exception:
            return None

        clipped = (text or "").strip()
        if not clipped:
            return []
        clipped = clipped[:1200]
        n = max(1, min(int(limit), 12))

        if kind == "vocab":
            sys = (
                "You extract vocabulary candidates a dictation app should learn. "
                "Pick proper nouns, brand names, jargon, acronyms, and unusual technical terms — "
                "anything an automatic speech recognizer might mis-transcribe. "
                "Skip generic English words and anything already in normal usage. "
                "Reply with one JSON object only, no markdown, key 'candidates' = array of objects, "
                "each with 'term' (the exact spelling the user wants) and 'note' (one short reason or "
                "the phonetic the user might say, max 8 words). Limit to "
                f"{n} entries. Empty array if nothing qualifies."
            )
        elif kind == "snippet":
            sys = (
                "You spot reusable phrasings the user might want to save as a snippet. "
                "Pick boilerplate openers/closers, recurring sentences, or phrases that look "
                "like they could be triggered by a short tag. Reply with one JSON object only, "
                "no markdown, key 'candidates' = array of objects with 'trigger' (a short slug, "
                "lowercase, no spaces) and 'body' (the snippet text). Limit to "
                f"{n}. Empty array if nothing qualifies."
            )
        elif kind == "replacement":
            sys = (
                "You spot text the user might want auto-replaced — abbreviations the speaker "
                "always means to expand, names the dictation tends to spell wrong, etc. "
                "Reply with one JSON object only, no markdown, key 'candidates' = array of "
                "objects with 'trigger' (what the user says) and 'replacement' (what should be "
                f"written). Limit to {n}. Empty array if nothing qualifies."
            )
        else:
            return []

        user = f"Transcript:\n{clipped}\n\nJSON:"
        if getattr(self._tokenizer, "chat_template", None) is not None:
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ]
            prompt = _apply_chat_template_no_thinking(self._tokenizer, messages)
        else:
            prompt = sys + "\n\n" + user

        try:
            with mlx_decode_guard():
                raw, _generation_meta = _generate_with_sampling(
                    generate,
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.2,
                    top_p=0.95,
                )
        except Exception:
            return None

        # Best-effort JSON extraction. The model occasionally wraps the
        # object in stray prose; we slice on the first '{' / last '}'.
        import json as _json
        s = (raw or "").strip()
        start = s.find("{")
        end = s.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            obj = _json.loads(s[start : end + 1])
        except Exception:
            return []
        cands = obj.get("candidates")
        if not isinstance(cands, list):
            return []
        # Normalize: keep only entries with the expected required key.
        out: list[dict[str, Any]] = []
        for entry in cands[:n]:
            if not isinstance(entry, dict):
                continue
            if kind == "vocab":
                term = str(entry.get("term") or "").strip()
                if not term or len(term) > 60:
                    continue
                out.append({"term": term, "note": str(entry.get("note") or "").strip()[:80]})
            elif kind == "snippet":
                trigger = str(entry.get("trigger") or "").strip()
                body = str(entry.get("body") or "").strip()
                if not trigger or not body:
                    continue
                out.append({"trigger": trigger[:32], "body": body[:280]})
            elif kind == "replacement":
                trigger = str(entry.get("trigger") or "").strip()
                rep = str(entry.get("replacement") or "").strip()
                if not trigger or not rep:
                    continue
                out.append({"trigger": trigger[:60], "replacement": rep[:120]})
        return _filter_memory_extraction_candidates(kind=kind, source_text=clipped, candidates=out, limit=n)

    def classify_dictation_vs_edit_selection(
        self, *, spoken: str, selection_excerpt: str
    ) -> dict[str, Any] | None:
        """Return ``{"intent": "edit"|"dictate", "confidence": float, "instruction": str}`` or None."""
        self.warm()
        assert self._model is not None and self._tokenizer is not None
        try:
            from mlx_lm import generate  # type: ignore
        except Exception:
            return None

        sys = (
            "You classify a single spoken utterance while the user has highlighted text in an editor. "
            "Reply with one JSON object only, no markdown, with these keys:\n"
            "  intent: \"edit\" or \"dictate\".\n"
            "  confidence: float between 0 and 1.\n"
            "  edit_instruction: REQUIRED when intent=\"edit\". A concise English rewrite directive "
            "(e.g. \"make this more formal\", \"convert to bullets\", \"fix grammar\"). Never leave "
            "this empty for intent=\"edit\".\n"
            "  dictation_text: REQUIRED when intent=\"dictate\". The exact words the user wants typed "
            "verbatim into the document.\n"
            "Use intent=edit when they are asking to change the highlight (tone, clarity, bullets, grammar). "
            "Use intent=dictate when they are dictating new words to insert (e.g. continuing a sentence). "
            "Populate only the field that matches the chosen intent; leave the other absent or empty."
        )
        user = (
            f"Highlighted excerpt (truncated):\n{selection_excerpt}\n\n"
            f"Spoken utterance:\n{spoken}\n\nJSON:"
        )
        if getattr(self._tokenizer, "chat_template", None) is not None:
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ]
            prompt = _apply_chat_template_no_thinking(self._tokenizer, messages)
        else:
            prompt = sys + "\n\n" + user

        started = time.perf_counter()
        with mlx_decode_guard():
            text, _generation_meta = _generate_with_sampling(
                generate,
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=min(96, int(self.config.max_tokens)),
                temperature=0.05,
                top_p=0.9,
            )
        decode_ms = (time.perf_counter() - started) * 1000.0
        raw = _clean_writer_output(text)
        parsed = _extract_json_object(raw)
        if not isinstance(parsed, dict):
            return None
        intent = str(parsed.get("intent") or "").strip().lower()
        if intent not in {"edit", "dictate"}:
            return None
        try:
            conf = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        # D4: read the intent-specific field, falling back to the legacy
        # overloaded `instruction` field for back-compat. Path A keeps the
        # outward-facing `instruction` key on the result (one call site,
        # well-defined semantics) but populates it from the field the
        # model was told to fill for the chosen intent.
        edit_instr = str(parsed.get("edit_instruction") or "").strip()
        dictation_text = str(parsed.get("dictation_text") or "").strip()
        legacy_instr = str(parsed.get("instruction") or "").strip()
        if intent == "edit":
            instr = edit_instr or legacy_instr
            if not instr:
                # Model violated the prompt by omitting edit_instruction.
                # Fall back to the spoken utterance rather than silently
                # dropping the edit — downstream gate logs this case.
                instr = spoken.strip()
        else:  # intent == "dictate"
            instr = dictation_text or legacy_instr or spoken.strip()
        return {
            "intent": intent,
            "confidence": max(0.0, min(1.0, conf)),
            "instruction": instr or spoken.strip(),
            "decode_ms": decode_ms,
        }


def _generate_with_sampling(
    generate: Any,
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    deadline_s: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Call mlx_lm.generate with the sampler API (mlx_lm >= 0.22.0)."""
    try:
        from mlx_lm.sample_utils import make_sampler  # type: ignore
    except Exception:
        kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "verbose": False,
        }
        return generate(model, tokenizer, prompt=prompt, **kwargs), {"generation_api": "legacy"}

    sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
    if deadline_s is not None:
        try:
            from mlx_lm import stream_generate  # type: ignore

            pieces: list[str] = []
            timed_out = False
            started = time.perf_counter()
            for resp in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=int(max_tokens),
                sampler=sampler,
            ):
                pieces.append(resp.text)
                if (time.perf_counter() - started) > deadline_s:
                    timed_out = True
                    break
            return "".join(pieces), {
                "generation_api": "stream_sampler",
                "timed_out": timed_out,
            }
        except Exception as exc:  # noqa: BLE001 — generation fallback must preserve output
            print(f"[WRITER]      stream_deadline_fallback err={exc}", file=sys.stderr, flush=True)

    kwargs = {
        "max_tokens": int(max_tokens),
        "verbose": False,
        "sampler": sampler,
    }
    meta: dict[str, Any] = {"generation_api": "sampler"}
    if deadline_s is not None:
        meta["deadline_unenforced"] = True
    return generate(model, tokenizer, prompt=prompt, **kwargs), meta


def _apply_chat_template_no_thinking(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    apply_chat_template = getattr(tokenizer, "apply_chat_template")
    kwargs: dict[str, Any] = {"add_generation_prompt": True}
    try:
        sig = inspect.signature(apply_chat_template)
        supports_tokenize = (
            "tokenize" in sig.parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
        )
        supports_enable_thinking = (
            "enable_thinking" in sig.parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
        )
    except (TypeError, ValueError):
        supports_tokenize = True
        supports_enable_thinking = True
    if supports_tokenize:
        # Keep the writer backend's generation contract string-in. Several
        # chat tokenizers default to token ids; MLX can generate from those,
        # but prompt observability and strict JSON behavior become opaque.
        kwargs["tokenize"] = False
    if supports_enable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return apply_chat_template(messages, **kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        if "enable_thinking" in fallback_kwargs:
            fallback_kwargs.pop("enable_thinking", None)
            try:
                return apply_chat_template(messages, **fallback_kwargs)
            except TypeError:
                pass
        if "tokenize" in fallback_kwargs:
            fallback_kwargs.pop("tokenize", None)
            return apply_chat_template(messages, **fallback_kwargs)
        raise


def _extract_json_object(text: str) -> dict[str, Any] | None:
    import json

    t = (text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _filter_memory_extraction_candidates(
    *,
    kind: str,
    source_text: str,
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Apply product memory policy to model-suggested candidates.

    The model is useful for spotting terms, but persistence/UI boundaries
    must enforce the same "worth biasing ASR" policy as the rest of memory.
    This keeps common words and invented snippet bodies out without adding
    utterance-specific rules.
    """

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    def append_candidate(normalized: dict[str, Any]) -> None:
        key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        out.append(normalized)

    for item in candidates:
        if not isinstance(item, dict):
            continue
        normalized: dict[str, Any] | None = None
        if kind == "vocab":
            term = str(item.get("term") or "").strip()
            note = str(item.get("note") or "").strip()
            for vocab_term in _vocab_candidate_terms(term, note=note, source_text=source_text):
                if _vocab_candidate_allowed(vocab_term, source_text=source_text):
                    append_candidate({"term": vocab_term, "note": note[:80]})
                    if len(out) >= max(1, int(limit)):
                        return out
            continue
        elif kind == "replacement":
            trigger = str(item.get("trigger") or "").strip()
            replacement = str(item.get("replacement") or "").strip()
            if _replacement_candidate_allowed(trigger, replacement, source_text=source_text):
                normalized = {"trigger": trigger[:60], "replacement": replacement[:120]}
        elif kind == "snippet":
            trigger = str(item.get("trigger") or "").strip()
            body = str(item.get("body") or "").strip()
            if _snippet_candidate_allowed(trigger, body, source_text=source_text):
                normalized = {"trigger": trigger[:32], "body": body[:280]}
        if normalized is None:
            continue
        append_candidate(normalized)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _vocab_candidate_terms(term: str, *, note: str = "", source_text: str) -> list[str]:
    value = _strip_vocab_relation_tail((term or "").strip())
    if not value:
        return []
    expanded = _strip_vocab_relation_tail(
        _expand_vocab_phrase_from_note(value, note=note, source_text=source_text)
        or _expand_source_vocab_phrase(value, source_text=source_text)
    )
    split = _split_composite_vocab_phrase(expanded, source_text=source_text)
    return split or [expanded]


def _strip_vocab_relation_tail(term: str) -> str:
    value = re.sub(r"\s+", " ", (term or "").strip())
    if not value:
        return ""
    value = re.sub(
        r"\s+(?:(?:is|are|was|were|may|might|can|could|should)\s+)?"
        r"(?:sound(?:s|ed)?|pronounced|spelled|means|called|known)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip(" ,.;:-")


def _expand_source_vocab_phrase(term: str, *, source_text: str) -> str:
    value = (term or "").strip()
    if not value or not source_text:
        return value
    # If the model returns only the first token of a spoken acronym sequence
    # ("MLX" from "MLX LM"), keep the phrase the user actually said.
    pattern = re.compile(
        rf"(?<![A-Za-z0-9])({re.escape(value)}(?:\s+[A-Z0-9]{{2,}})+)(?![A-Za-z0-9])"
    )
    match = pattern.search(source_text)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    expanded = _expand_source_descriptive_vocab_phrase(value, source_text=source_text)
    if expanded:
        return expanded
    return value


def _expand_source_descriptive_vocab_phrase(term: str, *, source_text: str) -> str:
    if not _explicit_teach_context(source_text):
        return ""
    value = (term or "").strip()
    if not value:
        return ""
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])")
    for match in pattern.finditer(source_text):
        tail = source_text[match.end():]
        descriptors = _leading_vocab_descriptors(tail)
        if descriptors:
            return re.sub(r"\s+", " ", " ".join([value, *descriptors])).strip()
    return ""


def _expand_vocab_phrase_from_note(term: str, *, note: str, source_text: str) -> str:
    if not _explicit_teach_context(source_text):
        return ""
    value = (term or "").strip()
    raw_note = (note or "").strip()
    if not value or not raw_note:
        return ""
    for clause in re.split(r"[,;()\n]|\bnot\b", raw_note, maxsplit=3, flags=re.IGNORECASE):
        clause = re.sub(r"\s+", " ", clause).strip(" .,:;-")
        if not clause:
            continue
        match = re.search(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", clause)
        if not match:
            continue
        descriptors = _leading_vocab_descriptors(clause[match.end():])
        if not descriptors:
            continue
        candidate = re.sub(r"\s+", " ", " ".join([value, *descriptors])).strip()
        if _candidate_grounded(candidate, source_text):
            return candidate
    return ""


def _leading_vocab_descriptors(text: str) -> list[str]:
    descriptors: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9._-]*", text or ""):
        if token.casefold() in _VOCAB_DESCRIPTOR_STOPWORDS:
            break
        if _looks_like_standalone_vocab_piece(token):
            break
        if not token[:1].islower():
            break
        descriptors.append(token)
        if len(descriptors) >= 3:
            break
    return descriptors


_VOCAB_DESCRIPTOR_STOPWORDS = {
    "and",
    "as",
    "is",
    "are",
    "be",
    "like",
    "means",
    "or",
    "should",
    "these",
    "this",
    "term",
    "terms",
    "word",
    "words",
}


def _split_composite_vocab_phrase(term: str, *, source_text: str) -> list[str]:
    value = re.sub(r"\s+", " ", (term or "").strip())
    if not value:
        return []
    if not _explicit_teach_context(source_text):
        return [value]
    pieces = value.split()
    if len(pieces) <= 2:
        return [value]
    # Qwen sometimes merges adjacent spoken names into one candidate. Split
    # runs of TitleCase personal/product-looking words while preserving
    # acronym phrases such as "MLX LM".
    if all(_looks_like_standalone_vocab_piece(piece) for piece in pieces):
        return pieces
    out: list[str] = []
    idx = 0
    while idx < len(pieces):
        piece = pieces[idx]
        if re.fullmatch(r"[A-Z0-9]{2,}", piece):
            run = [piece]
            idx += 1
            while idx < len(pieces) and re.fullmatch(r"[A-Z0-9]{2,}", pieces[idx]):
                run.append(pieces[idx])
                idx += 1
            out.append(" ".join(run))
            continue
        if _looks_like_standalone_vocab_piece(piece):
            out.append(piece)
        idx += 1
    if len(out) == 1 and len(pieces) > 1:
        return [value]
    return out or [value]


def _looks_like_standalone_vocab_piece(value: str) -> bool:
    piece = (value or "").strip()
    if not piece:
        return False
    if re.fullmatch(r"[A-Z0-9]{2,}(?:[._-][A-Z0-9]{2,})*", piece):
        return True
    if re.search(r"[a-z][A-Z]", piece):
        return True
    if any(ch.isalpha() for ch in piece) and any(ch.isdigit() for ch in piece):
        return True
    return bool(re.fullmatch(r"[A-Z][a-z]{2,}[A-Za-z]*", piece))


def _vocab_candidate_allowed(term: str, *, source_text: str) -> bool:
    value = (term or "").strip()
    if not value or not _candidate_grounded(value, source_text):
        return False
    if _memory_command_phrase(value):
        return False
    from juno_v2.memory.entity_policy import session_entity_allowed

    if session_entity_allowed(value):
        return True
    if _explicit_teach_context(source_text) and _has_named_or_identifier_piece(value):
        return True
    return False


def _replacement_candidate_allowed(trigger: str, replacement: str, *, source_text: str) -> bool:
    trig = (trigger or "").strip()
    repl = (replacement or "").strip()
    if not trig or not repl or not _candidate_grounded(trig, source_text):
        return False
    if _memory_command_phrase(trig):
        return False
    from juno_v2.memory.entity_policy import session_entity_allowed
    from juno_v2.memory.stores.corrections import is_safe_correction_pair

    if not is_safe_correction_pair(trig, repl):
        return False
    if session_entity_allowed(trig):
        return True
    if _looks_like_abbreviation_trigger(trig):
        return True
    if re.search(r"\b(?:replace|means|expand|spell|autocorrect)\b", source_text or "", re.IGNORECASE):
        return True
    return False


def _snippet_candidate_allowed(trigger: str, body: str, *, source_text: str) -> bool:
    trig = (trigger or "").strip()
    text = (body or "").strip()
    if not trig or not text:
        return False
    if len(text) > 280 or _memory_command_phrase(trig):
        return False
    # Snippet extraction must not invent reusable boilerplate. The body has
    # to be a phrase the user actually said unless the request explicitly
    # says it is defining a snippet.
    if _candidate_grounded(text, source_text):
        return True
    return bool(re.search(r"\b(?:save|create|add)\s+(?:this\s+)?snippet\b", source_text or "", re.IGNORECASE))


def _candidate_grounded(candidate: str, source_text: str) -> bool:
    from juno_v2.memory.entity_policy import term_present_in_text

    return term_present_in_text(candidate, source_text)


def _explicit_teach_context(source_text: str) -> bool:
    text = source_text or ""
    return bool(
        re.search(r"\b(?:teach|remember|learn)\s+(?:juno\b)?", text, re.IGNORECASE)
        or re.search(r"\badd\s+(?:these\s+)?(?:terms?|words?|names?)\b", text, re.IGNORECASE)
        or re.search(r"\b(?:should|must)\s+be\s+(?:remembered|learned)\b", text, re.IGNORECASE)
    )


def _memory_command_phrase(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?i)\s*(?:teach|remember|learn|add|save)\s+juno(?:\s+(?:this|these|term|terms|word|words))?\s*",
            value or "",
        )
    )


def _has_named_or_identifier_piece(value: str) -> bool:
    pieces = re.findall(r"[A-Za-z][A-Za-z0-9._-]*", value or "")
    if not pieces:
        return False
    for piece in pieces:
        if re.fullmatch(r"[A-Z0-9]{2,}(?:[._-][A-Z0-9]{2,})*", piece):
            return True
        if re.search(r"[a-z][A-Z]", piece):
            return True
        if any(ch.isalpha() for ch in piece) and any(ch.isdigit() for ch in piece):
            return True
        if re.fullmatch(r"[A-Z][a-z]{2,}[A-Za-z]*", piece):
            return True
    return False


def _looks_like_abbreviation_trigger(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", value or "")
    return 2 <= len(compact) <= 8 and (
        compact.isupper()
        or any(ch.isdigit() for ch in compact)
        or compact.casefold() == compact
    )


def _max_tokens_for_request(req: WriterTransformRequest, configured_max_tokens: int) -> int:
    context = req.context_payload or {}
    task = str(context.get("task") or req.metadata.get("kind") or "").strip()
    schema_version = str(context.get("schema_version") or "").strip()
    meta_override = req.metadata.get("max_tokens") if req.metadata else None
    if task == "voice_action_extraction":
        override = int(meta_override) if isinstance(meta_override, int) and meta_override > 0 else 0
        return max(int(configured_max_tokens), override, _ACTION_EXTRACTION_MAX_TOKENS)
    if task in {"turn_planning_v1", "turn_repair_v1", "transform_generation_v1"}:
        override = int(meta_override) if isinstance(meta_override, int) and meta_override > 0 else 0
        return max(int(configured_max_tokens), override, 1536)
    if isinstance(meta_override, int) and meta_override > 0:
        return int(meta_override)
    return int(configured_max_tokens)


def _system_prompt(req: WriterTransformRequest | None = None) -> str:
    task = ""
    if req is not None:
        task = str((req.context_payload or {}).get("task") or req.metadata.get("kind") or "").strip()
    if task == "dictation_edit_v1":
        from juno_v2.writer.dictation_editor import DICTATION_EDIT_SYSTEM_PROMPT

        return DICTATION_EDIT_SYSTEM_PROMPT
    if task == "voice_action_extraction":
        schema_version = "actions_intent_v2"
        if req is not None:
            schema_version = str((req.context_payload or {}).get("schema_version") or schema_version).strip()
        return (
            "You are a local voice-action classifier for Juno. "
            "You extract explicit Juno voice actions from a wake-gated utterance. "
            "Classify execution intent before extraction. "
            "Never execute praise, questions, examples, quoted commands, product descriptions, or ordinary dictation. "
            f"Return only JSON matching {schema_version}. Do not rewrite dictation. "
            "Do not format text for paste. Do not infer actions from non-action speech. "
            "Do not execute anything; only describe validated candidate actions. "
            "If the utterance is not an explicit reminder/note/alarm/action request, return no actions."
        )
    if task == "turn_planning_v1":
        return (
            "You are Juno's local final turn planner. "
            "You are not a chatbot and you do not execute actions. "
            "Return exactly one strict JSON object matching schema_version=turn_plan_v1. "
            "Do not copy the contract, schema, examples, enum lists, or input payload. "
            "For every enum field, choose exactly one allowed value, never a pipe-delimited list. "
            "Your job is to decide the user's intended final turn from ASR evidence, app context, memory, snippets, "
            "target state, permissions, and current time. "
            "Qwen decides meaning; deterministic code will validate, render, normalize time, resolve snippets, and execute. "
            "If actions.wake_verified is true, treat the utterance as addressed to Juno and classify the intended operation from the whole transcript; do not rely on keyword vetoes. "
            "Do not add facts, names, dates, times, numbers, tasks, list items, or action bodies that are not grounded in ASR/context. "
            "Resolve explicit self-corrections and false starts by keeping the latest intended wording. "
            "Resolve corrections before filling action bodies, schedule.source_span, render text, or transform instructions. "
            "Preserve literal mentions of commands when the speech is quoted, described, or asks to write the words. "
            "Screen candidate_entities, selected text, window titles, and memory terms are reference context, not replacement commands; never replace ordinary pronouns or common words just because a screen term sounds similar. "
            "For formatting, put structure in render_plan.content_units; do not rely on prose instructions downstream. "
            "If the user claims a count but speaks fewer items, set claimed_item_count and spoken_item_count and include only spoken items. "
            "If the user says to write, type, dictate, insert, paste, or note down points, that content is cursor text, not a native Note action. "
            "Emit a native note action only for explicit note-app commands such as take a note, create a note, save a note, or add a note. "
            "For wake-verified native note/reminder/alarm turns, set utterance_kind='actions', render_plan.render_kind='none', render_plan.content_units=[], and safety.commit_policy='no_commit'. "
            "Do not emit mixed native-action plus paste plans: valid native actions are executed by Juno and never pasted into the focused field. "
            "If the user says to write, type, insert, paste, draft, format, or transform text, classify it as text/transform unless they also clearly ask Juno to create a native note, reminder, or alarm. "
            "For selected/recent transforms, first resolve target; if target is missing or ambiguous, mark uncertainty instead of rewriting. "
            "For actions, emit planned actions only when the user is clearly directing Juno to perform them; include evidence_span, body, schedule.source_span, missing_fields, and execute_policy. "
            "actions[].kind must be exactly one of three literals: "
            "'note' for an Apple Notes entry (take a note that the build is red); "
            "'reminder' for an Apple Reminders task, time optional (remind me to call Sam at 4 pm); "
            "'alarm' for a clock alarm with a grounded time (set an alarm for 7 am). "
            "Never invent other action kinds such as create_note, add_reminder, task, todo, event, timer, or plural forms; "
            "if the request fits none of the three, emit no action and record it in uncertainties. "
            "Reminders may be unscheduled when the user gives a body but no time; use schedule.kind='none'. Alarms require a grounded time; if missing or ambiguous, mark missing_fields/uncertainties instead of inventing. "
            "For code, terminal, file paths, env vars, branch names, identifiers, dates, and times, preserve exact spelling, punctuation, casing, slashes, underscores, hyphens, and numbers when evidence supports them. "
            "For markdown, obey target_capabilities.markdown_allowed; never emit **bold** or markdown headings when markdown is not allowed. "
            "For memory_candidates, propose only uncommon personal/product/project/jargon/code terms or explicit teach/remember instructions; never propose common words, filler words, command words, or retracted variants. "
            "If uncertain, represent uncertainty in JSON. Do not explain outside JSON."
        )
    if task == "turn_repair_v1":
        return (
            "You repair a previously invalid Juno turn_plan_v1 JSON object. "
            "Return exactly one corrected strict JSON object. No prose, no markdown. "
            "Do not return the repair request, invalid object wrapper, allowed_values, schema, or explanation. "
            "Only fix schema/validation issues described in the payload; do not change grounded user meaning. "
            "For list formatting, never invent missing entries to satisfy a claimed count; keep only spoken source items, "
            "remove ungrounded numbered/bulleted units, and record claimed/spoken count mismatch in uncertainties. "
            "For every enum field, choose exactly one allowed value."
        )
    if task == "transform_generation_v1":
        return (
            "You transform selected or recent user text according to a typed Juno turn plan. "
            "Return exactly one JSON object with schema_version=transform_generation_v1 and transformed_text. "
            "Do not include the instruction, original text, markdown fences, or explanation. "
            "Preserve required names, numbers, dates, identifiers, and protected terms. "
            "Do not add facts or promises not present in source text or explicit instruction."
        )
    if task == "transcript_adjudication_v1":
        return (
            "You are Juno's local final speech resolver. "
            "Your job is to recover the exact text the user meant to say from complete ASR evidence, local context, "
            "memory, screen terms, app context, and explicit spoken correction cues. "
            "You are not a chatbot. You are not a deterministic parser. You are the semantic speech-resolution layer. "
            "Return exactly one JSON object matching schema_version=transcript_adjudication_v1. "
            "For final-stage adjudication, corrected_text must reflect the complete SPOKEN transcript in "
            "memory_candidate/whisper_raw/raw_text, not the live HUD snapshot. base_visible is only the live HUD "
            "snapshot and may be shorter, stale, or wrong: never truncate or shorten the final answer to match "
            "base_visible, and never drop later dictation that base_visible has not caught up to yet. "
            "Removing the speaker's own self-corrections and false starts is REQUIRED cleanup, not truncation: when the "
            "speaker revises themselves mid-utterance, keep only the corrected wording and drop both the retracted "
            "words and the correction cue. Do not invert the meaning of an explicit correction. "
            "Do not treat literal mentions of editing phrases as commands: preserve literal content when the surrounding "
            "speech is describing, quoting, or asking to write the phrase itself. "
            "Preserve explicitly requested closing text as dictated content, not evaluator metadata. "
            "For final-stage adjudication, set ops to [] so the caller computes the diff; do not spend output tokens listing per-character operations. "
            "You may include optional intent, formatting_plan, self_corrections, terms_used, and uncertainty fields as evidence for downstream code. "
            "Use formatting_plan for spoken requests like bullets, numbered points, sections, email, or no formatting; do not apply that structure inside corrected_text. "
            "Allowed changes: fix ASR recognition errors using Whisper, memory, screen terms, and code/file context; "
            "resolve explicit self-corrections and false starts; preserve exact casing/spelling for protected terms; "
            "add punctuation/capitalization needed for readable transcript; normalize spoken punctuation and obvious numbers/times. "
            "Forbidden changes: do not create bullets, headings, email structure, markdown, or paragraphs for style; "
            "do not rewrite tone; do not delete spoken filler or hedge words such as like, just, really, more, or basically unless an explicit self-correction asks for it; "
            "do not execute or parse Juno actions; do not add facts, names, dates, times, or tasks not present in ASR/context; "
            "do not alter code, shell commands, or terminal-looking text unless exact file/symbol context supports the correction."
        )
    if task == "live_transcript_correction_v1":
        return (
            "You are Juno's live transcript corrector. "
            "Correct only the stable target window from the user's live captions. "
            "Use ASR evidence, memory terms, and screen terms only when they clearly support the correction. "
            "If unsure, return the target window exactly. "
            "Return only the corrected target-window text. No JSON, no markdown, no explanation."
        )
    if task == "final_formatting_v1":
        base = (
            "You are Juno's final formatting engine. "
            "The transcript has already been corrected. Do not fix ASR. "
            "Do not change facts, names, numbers, dates, or identifiers. "
            "Do not summarize, analyze, or add headings unless the source explicitly asks for that structure. "
            "Preserve every required term and high-value screen/memory term exactly when present in the corrected transcript. "
            "Only apply the requested formatting policy for the target app. "
            "Return only final text."
        )
        # Mode-specific nudge appended after the base contract — gives
        # built-in modes (formal_email / casual_chat / structured_notes /
        # explicit_rewrite) and user-authored custom modes a way to
        # actually shape the LLM's output, not just the deterministic
        # post-pass. Empty / missing is a no-op.
        prefix = ""
        if req is not None:
            prefix = str((req.context_payload or {}).get("mode_prompt_prefix") or "").strip()
        if prefix:
            return base + "\n\nMode guidance: " + prefix
        return base
    if task == "selection_transform_v1":
        base = (
            "You transform selected user text according to the spoken instruction. "
            "Do not treat the instruction as dictation for insertion. "
            "Return only the replacement text. Never include the original. "
            "Keep the language of the input. Do not translate unless explicitly asked. "
            "Preserve the input's structural shape unless the instruction explicitly requests a list, bullets, "
            "numbering, table, sections, or another format change. If the source is a paragraph and the user only "
            "asks for tone, clarity, brevity, or directness, return a paragraph, not bullets. "
            "Match any requested length budget: 'shorter' trims by at least 20%, "
            "'longer' expands by at least 30%; otherwise preserve length within "
            "roughly plus-or-minus 15%."
        )
        prefix = ""
        if req is not None:
            prefix = str((req.context_payload or {}).get("mode_prompt_prefix") or "").strip()
        if prefix:
            return base + "\n\nMode guidance: " + prefix
        return base
    base = (
        "You are a local writing engine inside a dictation product. "
        "Rewrite only the target text. Preserve meaning unless the instruction explicitly asks to change structure or tone. "
        "Return only the transformed text with no explanation, no markdown fences, and no preamble. "
        "Do not add any closing phrase, sign-off, or pleasantry that was not in the original text. "
        "Output only the rewritten text. Never include the original. "
        "Keep the language of the input. Do not translate unless explicitly asked. "
        "Match any requested length budget: 'shorter' trims by at least 20%, 'longer' expands by at least 30%; "
        "otherwise preserve length within roughly plus-or-minus 15%. "
        "The user message may contain several labeled blocks (instruction, app/context, selected text, source text). "
        "The block to rewrite is the one labeled 'Source text:'. The others are reference material."
    )
    prefix = ""
    if req is not None:
        prefix = str((req.context_payload or {}).get("mode_prompt_prefix") or "").strip()
    if prefix:
        return base + "\n\nMode guidance: " + prefix
    return base


def _build_writer_prompt(req: WriterTransformRequest) -> str:
    context = req.context_payload or {}
    task = str(context.get("task") or req.metadata.get("kind") or "").strip()
    if task == "dictation_edit_v1":
        return str(context.get("payload_text") or req.source_text or "")
    if task == "transcript_adjudication_v1":
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else context
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if task in {"turn_planning_v1", "turn_repair_v1"}:
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else context
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if task == "transform_generation_v1":
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else context
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if task == "live_transcript_correction_v1":
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else context
        base_visible = payload.get("base_visible") if isinstance(payload, dict) else {}
        asr = payload.get("asr") if isinstance(payload, dict) else {}
        terms = payload.get("terms") if isinstance(payload, dict) else []
        protected_terms = payload.get("protected_terms") if isinstance(payload, dict) else []
        target = str(base_visible.get("text") if isinstance(base_visible, dict) else req.source_text or "").strip()
        memory_candidate = str(asr.get("memory_candidate") if isinstance(asr, dict) else "").strip()
        whisper_raw = str(asr.get("whisper_raw") if isinstance(asr, dict) else "").strip()
        term_names: list[str] = []
        if isinstance(terms, list):
            for item in terms[:12]:
                if isinstance(item, dict):
                    value = str(item.get("canonical") or item.get("text") or "").strip()
                else:
                    value = str(item or "").strip()
                if value and value.casefold() not in {x.casefold() for x in term_names}:
                    term_names.append(value)
        lines = [
            "Task: live transcript correction",
            "Return only the corrected target window. If no confident fix, repeat it exactly.",
            "Target window:",
            target,
        ]
        if memory_candidate and memory_candidate != target:
            lines.extend(["Memory-normalized candidate:", memory_candidate])
        if whisper_raw and whisper_raw != target:
            lines.extend(["ASR alternate:", whisper_raw])
        if term_names:
            lines.extend(["Relevant terms:", ", ".join(term_names)])
        if isinstance(protected_terms, list) and protected_terms:
            lines.extend(["Protected terms:", ", ".join(str(x) for x in protected_terms[:12])])
        lines.append("Corrected target window:")
        return "\n".join(lines)
    if task == "voice_action_extraction":
        lines = [
            "Task: voice_action_extraction",
            f"Schema version: {context.get('schema_version') or 'actions_intent_v2'}",
        ]
        now_iso = str(context.get("now_iso") or "").strip()
        if now_iso:
            lines.append(f"Current local time (now_iso): {now_iso}")
        allowed = context.get("allowed_action_kinds") or ["note", "reminder", "alarm"]
        if isinstance(allowed, list):
            lines.append("Allowed action kinds: " + ", ".join(str(k) for k in allowed))
        lines.extend(
            [
                f"Instruction: {req.instruction}",
                "Source text:",
                req.source_text,
                "Output JSON:",
            ]
        )
        return "\n".join(lines)
    if task == "final_formatting_v1":
        payload = {
            "task": "final_formatting_v1",
            "policy": context.get("policy"),
            "reference_only_target": {
                "app_name": context.get("app_name"),
                "app_category": context.get("app_category"),
                "window_title": context.get("window_title"),
                "mode_name": context.get("mode_name"),
            },
            "reference_only_context": {
                "focused_text_before": context.get("focused_text_before"),
                "focused_text_after": context.get("focused_text_after"),
                "selected_text_excerpt": context.get("selected_text_excerpt"),
                "style_card": context.get("style_card"),
                "writer_tone_addon": context.get("writer_tone_addon"),
                "required_preserved_terms": context.get("required_preserved_terms") or [],
                "candidate_entities": context.get("candidate_entities") or [],
                "recent_screen_terms": context.get("recent_screen_terms") or [],
                "formatting_contract": (
                    "Apply structure only. Preserve content units, names, dates, numbers, identifiers, "
                    "and required_preserved_terms exactly unless punctuation-only changes are needed. "
                    "The target/context fields are reference-only; never include app names, window titles, "
                    "font names, toolbar/status words, or screen terms unless they also appear in corrected_transcript."
                ),
            },
            "corrected_transcript": req.source_text,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    lines = [
        f"Instruction: {req.instruction}",
        f"Writer mode: {req.mode.value}",
    ]
    app_name = context.get("app_name")
    if app_name:
        lines.append(f"App: {app_name}")
    app_category = context.get("app_category")
    if app_category:
        lines.append(f"Surface category: {app_category}")
    window_title = str(context.get("window_title") or "").strip()
    if window_title:
        lines.append(f"Window title: {window_title[:160]}")
    target_kind = context.get("target_kind")
    if target_kind:
        lines.append(f"Target kind: {target_kind}")
    mode_policy = context.get("mode_policy") or {}
    if isinstance(mode_policy, dict) and mode_policy:
        policy_bits = []
        for key in ("mode_name", "base_mode", "writer_behavior", "cleanup_policy", "punctuation_policy"):
            value = mode_policy.get(key)
            if value:
                policy_bits.append(f"{key}={value}")
        if policy_bits:
            lines.append("Mode policy:")
            lines.append(", ".join(policy_bits)[:360])
    style_card = context.get("style_card") or {}
    if isinstance(style_card, dict) and style_card:
        style_name = str(style_card.get("name") or "").strip()
        parts = []
        for key in ("formality", "length_pref", "structure", "sign_off", "notes"):
            value = style_card.get(key)
            if value:
                parts.append(f"{key}={value}")
        if style_name or parts:
            lines.append("Style guidance:")
            lines.append(", ".join(filter(None, [style_name] + parts))[:240])
    selected = str(context.get("selected_text") or "").strip()
    if selected:
        lines.append("Selected text context:")
        lines.append(selected[:1200])
    relevant_terms: list[str] = []
    for key in ("candidate_entities", "recent_screen_terms"):
        value = context.get(key)
        if isinstance(value, list):
            for item in value:
                term = str(item or "").strip()
                if term and term.casefold() not in {x.casefold() for x in relevant_terms}:
                    relevant_terms.append(term)
    if relevant_terms:
        lines.append("Relevant screen/session terms:")
        lines.append(", ".join(relevant_terms[:32])[:900])
    field_excerpt = str(context.get("field_text_excerpt") or "").strip()
    if field_excerpt:
        lines.append("Nearby text:")
        lines.append(field_excerpt[:1200])
    before = str(context.get("focused_text_before") or "").strip()
    after = str(context.get("focused_text_after") or "").strip()
    if before:
        lines.append("Text before cursor:")
        lines.append(before[-400:])
    if after:
        lines.append("Text after cursor:")
        lines.append(after[:300])
    focused_file = str(context.get("focused_file_path") or "").strip()
    if focused_file:
        lines.append(f"Focused file/path: {focused_file[:220]}")
    symbol = str(context.get("symbol_under_cursor") or "").strip()
    if symbol:
        lines.append(f"Symbol under cursor: {symbol[:120]}")
    memory_packet = context.get("memory_packet") or {}
    if memory_packet:
        lines.append("Memory hints:")
        hints = []
        for key in ("lexicon_terms", "canonical_forms", "replacements", "corrections", "session_entities"):
            value = memory_packet.get(key)
            if value:
                hints.append(f"{key}: {value}")
        if hints:
            lines.extend(hints[:8])
    lines.append("Source text:")
    lines.append(req.source_text)
    lines.append("Output:")
    return "\n".join(lines)


def _clean_writer_output(text: Any, *, req: "WriterTransformRequest | None" = None) -> str:
    output = str(text or "").strip()
    task = ""
    if req is not None:
        task = str((req.context_payload or {}).get("task") or req.metadata.get("kind") or "").strip()
    expect_json = task in {
        "transcript_adjudication_v1",
        "voice_action_extraction",
        "turn_planning_v1",
        "turn_repair_v1",
        "transform_generation_v1",
    }
    output = _strip_thinking_artifacts(output, expect_json=expect_json)
    if output.startswith("```"):
        output = output.strip("`").strip()
        if "\n" in output:
            output = output.split("\n", 1)[1].strip()
    if task in {
        "transcript_adjudication_v1",
        "voice_action_extraction",
        "turn_planning_v1",
        "turn_repair_v1",
        "transform_generation_v1",
    }:
        return output
    for prefix in (
        "Output:",
        "Rewritten text:",
        "Transformed text:",
        "Here is",
        "Here's",
        "Sure,",
        "Cleaned text:",
        "Polished:",
        "Polished text:",
        "The polished text is:",
        "The rewritten text is:",
    ):
        if output.startswith(prefix):
            output = output[len(prefix):].strip()
    output = _strip_polite_model_suffix(output, req=req)
    return output


def _strip_thinking_artifacts(output: str, *, expect_json: bool) -> str:
    s = (output or "").strip()
    if not s:
        return ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()
    if "<think>" not in s.lower():
        return s.replace("</think>", "").strip()
    if expect_json:
        start = s.find("{")
        return s[start:].strip() if start >= 0 else ""
    idx = s.lower().find("<think>")
    return s[:idx].replace("</think>", "").strip()


# Phrases that only ever come from an AI assistant — never from a human writing
# something in a real app. Strip unconditionally.
_AI_VERBOSE_CLOSINGS = re.compile(
    r"(?:^|(?<=\.))\s*"
    r"(?:"
    r"hope\s+this\s+helps?[.!]*|"
    r"let\s+me\s+know\s+if\s+(?:you\s+(?:need|have)|there\s+(?:is|are))\b[^.!]{0,60}[.!]*|"
    r"feel\s+free\s+to\s+(?:ask|reach\s+out|let\s+me\s+know)\b[^.!]{0,60}[.!]*|"
    r"(?:please\s+)?let\s+me\s+know\s+if\s+you\s+have\s+any\s+(?:questions?|concerns?)[.!]*|"
    r"is\s+there\s+anything\s+(?:else\s+)?(?:i\s+can\s+help|you\s+(?:need|would\s+like))\b[^.!]{0,60}[.!]*|"
    r"happy\s+to\s+help[^.!]{0,40}[.!]*|"
    r"thank\s+you\s+for\s+(?:using|trying|choosing)\b[^.!]{0,60}[.!]*"
    r")"
    r"\s*$",
    re.IGNORECASE,
)

# Sign-off phrases that are legitimate in email/letter context but are artifacts
# in messaging, notes, code, etc.
_CONTEXTUAL_CLOSINGS = re.compile(
    r"(?:(?<=\.)|(?<=!)|(?<=\?)|\n)\s*"
    r"(?:"
    r"thank\s+you(?:\s+for\s+\w+)?[.!,]*|"
    r"thanks[.!,]*|"
    r"best\s+regards?[,.]?|"
    r"kind\s+regards?[,.]?|"
    r"warm\s+regards?[,.]?|"
    r"sincerely[,.]?|"
    r"regards?[,.]?|"
    r"cheers[.!,]*"
    r")"
    r"\s*$",
    re.IGNORECASE,
)


def _strip_polite_model_suffix(output: str, *, req: "WriterTransformRequest | None" = None) -> str:
    # Verbose AI phrases are never legitimate user content.
    output = _AI_VERBOSE_CLOSINGS.sub("", output).strip()

    # Sign-off phrases need context to decide.
    m = _CONTEXTUAL_CLOSINGS.search(output)
    if m is None:
        return output

    # If the source text already contained this phrase the user wrote it — keep it.
    # Normalize punctuation so "Thanks!" matches "thanks" in the source.
    source_lower = ((req.source_text if req else "") or "").lower()
    candidate = re.sub(r"[^\w\s]", "", m.group(0).strip().lower()).strip()
    if candidate and source_lower and candidate in source_lower:
        return output

    # Keep if the context is explicitly email/letter or the user asked for a sign-off.
    if _closing_is_contextually_appropriate(req):
        return output

    return output[: m.start()].strip()


def _closing_is_contextually_appropriate(req: "WriterTransformRequest | None") -> bool:
    """True when a sign-off belongs in the output (email, letter, explicit request)."""
    if req is None:
        return False
    from juno_v2.contracts.writer import WriterMode
    if req.mode == WriterMode.FORMAL_EMAIL:
        return True
    context = req.context_payload or {}
    if str(context.get("app_category") or "").strip().lower() == "email":
        return True
    instruction_lower = (req.instruction or "").lower()
    if any(kw in instruction_lower for kw in ("sign off", "sign-off", "signoff", "closing", "signature", "regards", "letter")):
        return True
    style_card = context.get("style_card") or {}
    if isinstance(style_card, dict) and style_card.get("sign_off"):
        return True
    return False
