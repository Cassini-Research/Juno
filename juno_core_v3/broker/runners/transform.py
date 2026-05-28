"""Transform-class session runner.

A Transform session takes **selected text** (plus an optional instruction)
and produces replacement text. Unlike Insert, it does not touch the active
field directly — the surface (Mac overlay, workbench, ...) is responsible
for applying the replacement, because only the surface knows whether the
selection is still valid.

Two execution modes:

1. **Deterministic** — the hint is one of the first-class enum values
   (``polish``, ``bullets``, ``numbered``, ``upper``, ``lower``, ``title``).
   These run offline with no model required.
2. **Model-backed** — the hint is free-form text (``"make this friendlier"``,
   ``"summarize"``, ``"shorten"``). Requires a ``WriterBackend``. Without a
   backend we fall back to the closest deterministic option and mark the
   result as degraded so the surface can show a hint to the user.

The runner is pure Python and deliberately has *no* knowledge of audio,
VAD, insertion, or the engine session runner. That separation is what makes
Transform auditable per the North Star.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import uuid

from juno_v2.code_grammar.engine import CodeGrammarEngine, CodeGrammarMode, render_file_tag
from juno_v2.contracts.writer import WriterMode, WriterTransformRequest
from juno_v2.meeting_grammar.engine import (
    MeetingGrammarEngine,
    MeetingGrammarMode,
    detect_meeting_text,
)
from juno_v2.writer.backends.base import WriterBackend
from juno_v2.transforms.catalog import get_builtin
from juno_v2.writer.deterministic import (
    normalize_plain_dictation,
    render_bullets,
    render_lowercase,
    render_numbered,
    render_title_case,
    render_uppercase,
)

_code_grammar_engine = CodeGrammarEngine()
_meeting_grammar_engine = MeetingGrammarEngine()


class TransformHint(str, Enum):
    """First-class (deterministic) transforms."""

    POLISH = "polish"
    BULLETS = "bullets"
    NUMBERED = "numbered"
    UPPER = "upper"
    LOWER = "lower"
    TITLE = "title"
    # Code grammar transforms
    SNAKE = "snake"
    CAMEL = "camel"
    PASCAL = "pascal"
    KEBAB = "kebab"
    SCREAMING = "screaming"
    FILE_TAG = "file_tag"
    CODE_AUTO = "code_auto"
    # Meeting grammar transform
    MEETING_AUTO = "meeting_auto"


_DETERMINISTIC_KEYWORDS = {
    "polish": TransformHint.POLISH,
    "clean up": TransformHint.POLISH,
    "normalize": TransformHint.POLISH,
    "bullet points": TransformHint.BULLETS,
    "bullets": TransformHint.BULLETS,
    "bullet list": TransformHint.BULLETS,
    "numbered list": TransformHint.NUMBERED,
    "numbered": TransformHint.NUMBERED,
    "uppercase": TransformHint.UPPER,
    "upper": TransformHint.UPPER,
    "all caps": TransformHint.UPPER,
    "lowercase": TransformHint.LOWER,
    "lower": TransformHint.LOWER,
    "title case": TransformHint.TITLE,
    "title": TransformHint.TITLE,
    # Code grammar keywords
    "snake case": TransformHint.SNAKE,
    "snake_case": TransformHint.SNAKE,
    "camel case": TransformHint.CAMEL,
    "camelcase": TransformHint.CAMEL,
    "pascal case": TransformHint.PASCAL,
    "pascalcase": TransformHint.PASCAL,
    "kebab case": TransformHint.KEBAB,
    "kebab-case": TransformHint.KEBAB,
    "screaming snake": TransformHint.SCREAMING,
    "screaming snake case": TransformHint.SCREAMING,
    "screaming_snake": TransformHint.SCREAMING,
    "file tag": TransformHint.FILE_TAG,
    "at file": TransformHint.FILE_TAG,
    "code auto": TransformHint.CODE_AUTO,
    # Meeting grammar keywords
    "meeting": TransformHint.MEETING_AUTO,
    "meeting notes": TransformHint.MEETING_AUTO,
    "meeting auto": TransformHint.MEETING_AUTO,
    "format meeting": TransformHint.MEETING_AUTO,
}


def _match_deterministic(raw: str) -> TransformHint | None:
    """Try to map a free-form hint onto a deterministic transform."""
    lower = raw.strip().casefold()
    if not lower:
        return None
    if lower in _DETERMINISTIC_KEYWORDS:
        return _DETERMINISTIC_KEYWORDS[lower]
    # Contains-match, longest first, so "make it all caps" matches "all caps".
    for keyword in sorted(_DETERMINISTIC_KEYWORDS, key=len, reverse=True):
        if keyword in lower:
            return _DETERMINISTIC_KEYWORDS[keyword]
    return None


@dataclass(slots=True)
class TransformRequest:
    selected_text: str
    hint: str
    app_category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # When True the runner skips the writer backend regardless of hint.
    # Set by the broker on YELLOW-tier sessions to enforce degradation.
    force_deterministic: bool = False


@dataclass(slots=True)
class TransformResult:
    replacement_text: str
    used_backend: bool
    deterministic: bool
    hint_resolved: str
    degraded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replacement_text": self.replacement_text,
            "used_backend": self.used_backend,
            "deterministic": self.deterministic,
            "hint_resolved": self.hint_resolved,
            "degraded": self.degraded,
            "metadata": dict(self.metadata),
        }


@dataclass
class TransformRunner:
    """Execute a single Transform session.

    ``writer_backend`` is optional: if missing, model-only hints degrade to
    the nearest deterministic transform (with ``degraded=True`` on the
    result) rather than raising, because the surface should never be left
    with nothing to show.
    """

    writer_backend: WriterBackend | None = None

    def run(self, req: TransformRequest) -> TransformResult:
        raw_hint = (req.hint or "").strip()
        selected = req.selected_text or ""
        if not selected.strip():
            return TransformResult(
                replacement_text="",
                used_backend=False,
                deterministic=True,
                hint_resolved="noop_empty_selection",
                metadata={"reason": "empty selection"},
            )

        catalog_id = (req.metadata.get("transform_id") or "").strip()
        cat = get_builtin(catalog_id) if catalog_id else None
        if cat is not None:
            text = selected
            for pre in cat.deterministic_preprocessors:
                if pre == "bullets":
                    text = render_bullets(text)
                elif pre == "numbered":
                    text = render_numbered(text)
            instr = (cat.model_prompt_template or "").strip()
            custom_note = (req.metadata.get("custom_transform_instruction") or "").strip()
            if custom_note:
                instr = f"{instr}\n{custom_note}".strip() if instr else custom_note
            exec_meta = {
                "transform_id": cat.transform_id,
                "transform_source": req.metadata.get("transform_source", "builtin"),
                "target_class": req.metadata.get("target_class", "selected_text"),
                "catalog": cat.to_dict(),
            }
            if instr and self.writer_backend is not None and not req.force_deterministic:
                try:
                    write_req = WriterTransformRequest(
                        utterance_id=f"transform_{uuid.uuid4().hex[:12]}",
                        instruction=instr,
                        source_text=text,
                        mode=WriterMode.EXPLICIT_REWRITE,
                        context_payload={
                            "app_category": req.app_category,
                            **req.metadata,
                        },
                    )
                    result = self.writer_backend.rewrite(write_req)
                    rewritten = result.text if result is not None else None
                    return TransformResult(
                        replacement_text=rewritten if rewritten else text,
                        used_backend=True,
                        deterministic=False,
                        hint_resolved=cat.transform_id,
                        metadata={
                            **exec_meta,
                            "backend_name": getattr(result, "backend_name", None),
                            "decode_ms": getattr(result, "decode_ms", 0.0),
                            "post_processors": list(cat.post_processors),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    return TransformResult(
                        replacement_text=_degraded_fallback(text),
                        used_backend=False,
                        deterministic=True,
                        hint_resolved=cat.transform_id,
                        degraded=True,
                        metadata={**exec_meta, "backend_error": str(exc)},
                    )
            return TransformResult(
                replacement_text=text,
                used_backend=False,
                deterministic=True,
                hint_resolved=cat.transform_id,
                metadata=exec_meta,
            )

        # For code contexts with a polish hint, try CodeGrammarEngine AUTO mode
        # BEFORE regular deterministic matching: inline markers like
        # "snake case hello world" should produce "hello_world", not polished prose.
        #
        # Terminal surfaces stay explicit-only here. They often carry shell-like
        # text where aggressive code grammar inference is more harmful than
        # helpful, so terminal-safe mode avoids AUTO rewrites.
        if req.app_category == "code" and raw_hint.casefold() in ("", "polish"):
            cg_result = _code_grammar_engine.apply(selected, mode=CodeGrammarMode.AUTO)
            if cg_result.changed:
                return TransformResult(
                    replacement_text=cg_result.text,
                    used_backend=False,
                    deterministic=True,
                    hint_resolved=f"code_auto:{','.join(cg_result.rules_applied)}",
                    metadata={"code_grammar_rules": cg_result.rules_applied},
                )

        # Meeting surfaces (or any surface whose content *looks* like a
        # meeting transcript when the hint is blank/polish) get the
        # meeting overlay. Mirrors the code overlay: deterministic, no
        # model, only fires when it can actually change the text.
        # Content-based detection lets "docs" notes be promoted when
        # they contain speaker tags — the app classifier alone can't
        # tell a meeting note from a journal.
        if raw_hint.casefold() in ("", "polish"):
            should_try_meeting = req.app_category == "meeting" or (
                req.app_category in (None, "", "unknown", "docs", "email")
                and detect_meeting_text(selected)
            )
            if should_try_meeting:
                mg_result = _meeting_grammar_engine.apply(
                    selected, mode=MeetingGrammarMode.AUTO
                )
                if mg_result.changed:
                    return TransformResult(
                        replacement_text=mg_result.text,
                        used_backend=False,
                        deterministic=True,
                        hint_resolved=f"meeting_auto:{','.join(mg_result.rules_applied)}",
                        metadata={
                            "meeting_grammar_rules": mg_result.rules_applied,
                            "meeting_attendees": mg_result.attendees,
                            "meeting_action_items": mg_result.action_items,
                        },
                    )

        det = _match_deterministic(raw_hint)
        if det is not None:
            return TransformResult(
                replacement_text=_apply_deterministic(det, selected),
                used_backend=False,
                deterministic=True,
                hint_resolved=det.value,
            )

        # Free-form hint: try the writer backend (unless caller forced deterministic).
        if self.writer_backend is not None and not req.force_deterministic:
            try:
                write_req = WriterTransformRequest(
                    utterance_id=f"transform_{uuid.uuid4().hex[:12]}",
                    instruction=raw_hint,
                    source_text=selected,
                    mode=WriterMode.EXPLICIT_REWRITE,
                    context_payload={
                        "app_category": req.app_category,
                        **req.metadata,
                    },
                )
                result = self.writer_backend.rewrite(write_req)
                rewritten = result.text if result is not None else None
                return TransformResult(
                    replacement_text=rewritten if rewritten else selected,
                    used_backend=True,
                    deterministic=False,
                    hint_resolved=raw_hint,
                    metadata={
                        "backend_name": getattr(result, "backend_name", None),
                        "decode_ms": getattr(result, "decode_ms", 0.0),
                    },
                )
            except Exception as exc:  # noqa: BLE001 — surface needs a response
                return TransformResult(
                    replacement_text=_degraded_fallback(selected),
                    used_backend=False,
                    deterministic=True,
                    hint_resolved="polish_fallback",
                    degraded=True,
                    metadata={"backend_error": str(exc), "requested_hint": raw_hint},
                )

        # No backend, no deterministic match → polish so the user sees
        # *something* cleaner than the raw text, and mark degraded so the UI
        # can explain.
        return TransformResult(
            replacement_text=_degraded_fallback(selected),
            used_backend=False,
            deterministic=True,
            hint_resolved="polish_fallback",
            degraded=True,
            metadata={"requested_hint": raw_hint, "reason": "no writer backend"},
        )


def _apply_deterministic(hint: TransformHint, text: str) -> str:
    if hint is TransformHint.POLISH:
        return normalize_plain_dictation(text)
    if hint is TransformHint.BULLETS:
        return render_bullets(text)
    if hint is TransformHint.NUMBERED:
        return render_numbered(text)
    if hint is TransformHint.UPPER:
        return render_uppercase(text)
    if hint is TransformHint.LOWER:
        return render_lowercase(text)
    if hint is TransformHint.TITLE:
        return render_title_case(text)
    # Code grammar transforms
    if hint is TransformHint.SNAKE:
        return _code_grammar_engine.convert(text, mode=CodeGrammarMode.SNAKE)
    if hint is TransformHint.CAMEL:
        return _code_grammar_engine.convert(text, mode=CodeGrammarMode.CAMEL)
    if hint is TransformHint.PASCAL:
        return _code_grammar_engine.convert(text, mode=CodeGrammarMode.PASCAL)
    if hint is TransformHint.KEBAB:
        return _code_grammar_engine.convert(text, mode=CodeGrammarMode.KEBAB)
    if hint is TransformHint.SCREAMING:
        return _code_grammar_engine.convert(text, mode=CodeGrammarMode.SCREAMING)
    if hint is TransformHint.FILE_TAG:
        return render_file_tag(text)
    if hint is TransformHint.CODE_AUTO:
        result = _code_grammar_engine.apply(text, mode=CodeGrammarMode.AUTO)
        return result.text
    if hint is TransformHint.MEETING_AUTO:
        result = _meeting_grammar_engine.apply(text, mode=MeetingGrammarMode.AUTO)
        return result.text
    return text


def _degraded_fallback(text: str) -> str:
    """Safe default for unknown hints when no backend is available."""
    return normalize_plain_dictation(text)
