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
        # Issue #7: serialize MLX decode against preview/final lanes on
        # other threads. Without the guard, overlapping decodes from
        # different worker threads can crash with
        # ``RuntimeError: There is no Stream(gpu, N) in current thread``.
        with mlx_decode_guard():
            text, generation_meta = _generate_with_sampling(
                generate,
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
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
        return out

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

    kwargs = {
        "max_tokens": int(max_tokens),
        "verbose": False,
        "sampler": make_sampler(temp=float(temperature), top_p=float(top_p)),
    }
    return generate(model, tokenizer, prompt=prompt, **kwargs), {"generation_api": "sampler"}


def _apply_chat_template_no_thinking(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    apply_chat_template = getattr(tokenizer, "apply_chat_template")
    kwargs: dict[str, Any] = {"add_generation_prompt": True}
    try:
        sig = inspect.signature(apply_chat_template)
        supports_enable_thinking = (
            "enable_thinking" in sig.parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
        )
    except (TypeError, ValueError):
        supports_enable_thinking = True
    if supports_enable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" not in kwargs:
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("enable_thinking", None)
        return apply_chat_template(messages, **fallback_kwargs)


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


def _max_tokens_for_request(req: WriterTransformRequest, configured_max_tokens: int) -> int:
    context = req.context_payload or {}
    task = str(context.get("task") or req.metadata.get("kind") or "").strip()
    schema_version = str(context.get("schema_version") or "").strip()
    meta_override = req.metadata.get("max_tokens") if req.metadata else None
    if task == "voice_action_extraction":
        override = int(meta_override) if isinstance(meta_override, int) and meta_override > 0 else 0
        return max(int(configured_max_tokens), override, _ACTION_EXTRACTION_MAX_TOKENS)
    if isinstance(meta_override, int) and meta_override > 0:
        return int(meta_override)
    return int(configured_max_tokens)


def _system_prompt(req: WriterTransformRequest | None = None) -> str:
    task = ""
    if req is not None:
        task = str((req.context_payload or {}).get("task") or req.metadata.get("kind") or "").strip()
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
            "words and the correction cue. For example, "
            "'set up a meeting at 4 pm, actually make that 5 pm' -> 'Set up a meeting at 5 pm.'; "
            "'send it to John, no, to Sarah' -> 'Send it to Sarah.'; "
            "'Japan, no actually Korea, is the customer meeting location' -> "
            "'Korea is the customer meeting location.'; "
            "'let's meet Tuesday, I mean Wednesday' -> \"Let's meet Wednesday.\" "
            "'no actually write LumaRay as one product word' means the user is correcting the instruction; "
            "do not invert it into 'do not write LumaRay'. "
            "'do not include the words scratch that in the final note' is an instruction for the current note; "
            "exclude that instruction and the phrase unless the user explicitly quotes it as literal content. "
            "Do not treat literal mentions as commands: if the user says phrases such as 'the words blank space', "
            "'text should stay as text', or describes a correction rule, preserve the literal content unless the "
            "surrounding speech clearly asks to apply it to the current utterance. "
            "Phrases such as 'at the end say the final word is complete' are dictated content, not evaluator metadata; "
            "preserve the requested final-word text in corrected_text. "
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
    if task == "transcript_adjudication_v1":
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
            "target": {
                "app_name": context.get("app_name"),
                "app_category": context.get("app_category"),
                "window_title": context.get("window_title"),
                "mode_name": context.get("mode_name"),
            },
            "context": {
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
                    "and required_preserved_terms exactly unless punctuation-only changes are needed."
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
    expect_json = task in {"transcript_adjudication_v1", "voice_action_extraction"}
    output = _strip_thinking_artifacts(output, expect_json=expect_json)
    if output.startswith("```"):
        output = output.strip("`").strip()
        if "\n" in output:
            output = output.split("\n", 1)[1].strip()
    if task in {"transcript_adjudication_v1", "voice_action_extraction"}:
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
