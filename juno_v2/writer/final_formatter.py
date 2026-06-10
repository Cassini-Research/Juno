from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable

from juno_v2.code_grammar import CodeGrammarEngine, CodeGrammarMode
from juno_v2.context.compiler import FormattingPacket
from juno_v2.contracts.writer import WriterMode, WriterTransformRequest, WriterTransformResult
from juno_v2.meeting_grammar import MeetingGrammarEngine, MeetingGrammarMode
from juno_v2.writer.backends.base import WriterBackend
from juno_v2.writer.deterministic import (
    normalize_dictation_orthography,
    run_pipeline,
)


class FinalFormatter:
    """Commit-time formatter for surfaces/modes that expect structure.

    Transcript correctness is handled before this class. This formatter can
    shape email/notes/explicit rewrite output, but it must not fix ASR or
    invent facts.
    """

    def __init__(
        self,
        *,
        backend: WriterBackend | None,
        acquire: Callable[[], object] | None = None,
        release: Callable[[], object] | None = None,
    ) -> None:
        self.backend = backend
        self.acquire = acquire
        self.release = release
        self.last_rejection: dict[str, Any] | None = None

    def format(self, packet: FormattingPacket) -> WriterTransformResult | None:
        self.last_rejection = None
        policy = (packet.final_formatting_policy or "minimal").strip()
        if policy in {"none", "minimal", "verbatim", "code_or_terminal"}:
            return None
        if policy == "messaging":
            text = _messaging_boundary(packet.corrected_text, packet=packet)
            if text == packet.corrected_text:
                return None
            return WriterTransformResult(
                utterance_id=packet.utterance_id,
                text=text,
                backend_name="deterministic_final_formatter",
                metadata={"task": "final_formatting_v1", "policy": policy, "deterministic": True},
            )
        if policy not in {"email", "structured_notes", "explicit_rewrite", "command_generate"}:
            return None
        if policy == "structured_notes":
            text = run_pipeline(packet.corrected_text, app_category=packet.app_category)
            if text and text != packet.corrected_text:
                required_terms = _required_terms_for_packet(packet)
                if _missing_required_terms(packet.corrected_text, text, required_terms):
                    self.last_rejection = _build_rejection(
                        "required_terms_dropped",
                        source=packet.corrected_text,
                        formatted=text,
                        backend="deterministic_final_formatter",
                        required_terms=required_terms,
                    )
                else:
                    self.last_rejection = None
                    return WriterTransformResult(
                        utterance_id=packet.utterance_id,
                        text=text,
                        backend_name="deterministic_final_formatter",
                        metadata={
                            "task": "final_formatting_v1",
                            "policy": policy,
                            "deterministic": True,
                            "strategy": "structured_notes_pipeline",
                        },
                    )
        if self.backend is None or not callable(getattr(self.backend, "rewrite", None)):
            if self.last_rejection is None:
                self.last_rejection = {"reason": "backend_unavailable"}
            return None

        required_terms = _required_terms_for_packet(packet)
        instruction = (
            "Apply only the requested commit-time formatting policy. "
            "The transcript is already corrected. Do not fix ASR. "
            "Do not change facts, names, numbers, dates, or identifiers. "
            "Return only final text."
        )
        req = WriterTransformRequest(
            utterance_id=packet.utterance_id,
            instruction=instruction,
            source_text=packet.corrected_text,
            mode=_writer_mode_for_packet(packet),
            context_payload={
                "task": "final_formatting_v1",
                "policy": policy,
                "app_name": packet.app_name,
                "app_category": packet.app_category,
                "window_title": packet.window_title,
                "mode_name": packet.mode_name,
                "focused_text_before": packet.focused_text_before,
                "focused_text_after": packet.focused_text_after,
                # final_formatting_v1 shapes corrected dictation only.
                # Selection transforms use selection_transform_v1; leaking
                # selected text here lets a missed transform command rewrite
                # the selection and paste it as plain dictation.
                "selected_text_excerpt": "",
                "style_card": packet.style_card,
                "writer_tone_addon": packet.writer_tone_addon,
                "candidate_entities": list((packet.metadata or {}).get("candidate_entities") or []),
                "recent_screen_terms": list((packet.metadata or {}).get("recent_screen_terms") or []),
                "required_preserved_terms": required_terms,
                # Mode-specific tone/structure nudge — built-in modes
                # ship one, custom modes let users author their own.
                # Read by mlx_lm._system_prompt for final_formatting_v1
                # only (action/adjudication paths intentionally exclude
                # it — they have purpose-built single-task prompts).
                "mode_prompt_prefix": packet.mode_prompt_prefix or "",
            },
            metadata={"kind": "final_formatting_v1", "policy": policy},
        )

        acquired = False
        if self.acquire is not None:
            self.acquire()
            acquired = True
        try:
            result = self.backend.rewrite(req)
        finally:
            if acquired and self.release is not None:
                self.release()
        text = (result.text or "").strip()
        context_terms = _context_terms_for_packet(packet)
        rejection_reason = _formatted_output_rejection_reason(
            packet.corrected_text,
            text,
            required_terms=required_terms,
            context_terms=context_terms,
        )
        if rejection_reason is not None:
            self.last_rejection = _build_rejection(
                rejection_reason,
                source=packet.corrected_text,
                formatted=text,
                backend=result.backend_name,
                required_terms=required_terms,
            )
            return None
        result.metadata = {**result.metadata, "task": "final_formatting_v1", "policy": policy}
        return result


def apply_commit_boundary_rules(text: str, *, app_category: str | None = None) -> str:
    cat = (app_category or "").strip().lower()
    if cat in {"code", "terminal"}:
        return text or ""
    return normalize_dictation_orthography(text)


def should_run_final_formatting(policy: str | None, *, app_category: str | None = None) -> bool:
    cat = (app_category or "").strip().lower()
    if cat in {"code", "terminal"}:
        return False
    return (policy or "").strip() in {"email", "structured_notes", "explicit_rewrite", "command_generate", "messaging"}


def _messaging_boundary(text: str, *, packet: FormattingPacket) -> str:
    out = apply_commit_boundary_rules(text, app_category=packet.app_category)
    style = packet.style_card or {}
    casual = str(style.get("formality") or "").strip().lower() in {"casual", "informal"}
    if casual and out.endswith("."):
        return out[:-1]
    return out


def _writer_mode_for_packet(packet: FormattingPacket) -> WriterMode:
    name = (packet.mode_name or "").strip()
    try:
        return WriterMode(name)
    except ValueError:
        if packet.final_formatting_policy == "email":
            return WriterMode.FORMAL_EMAIL
        if packet.final_formatting_policy == "structured_notes":
            return WriterMode.STRUCTURED_NOTES
        if packet.final_formatting_policy in {"explicit_rewrite", "command_generate"}:
            return WriterMode.EXPLICIT_REWRITE
        return WriterMode.DEFAULT_SURFACE


# ---------------------------------------------------------------------------
# Issue #12 — code_grammar / meeting_grammar auto-fire
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GrammarPostPassResult:
    """Outcome of :func:`apply_grammar_postpass`.

    ``applied`` is True iff one of the grammar engines ran AND produced
    a change. ``engine`` is ``"code_grammar"`` / ``"meeting_grammar"``
    or ``None``. ``rules_applied`` mirrors the underlying engine's
    rule list for trace observability.
    """

    text: str
    applied: bool
    engine: str | None = None
    rules_applied: list[str] = field(default_factory=list)


# Categories that trigger each engine. Kept tight on purpose: the
# audit plan flags this as opt-in by category, with the surface-preset
# table (issue #27) doing the per-app routing.
_CODE_CATEGORIES = frozenset({"code", "terminal"})
_MEETING_CATEGORIES = frozenset({"meeting"})


def apply_grammar_postpass(
    text: str,
    *,
    app_category: str | None,
    code_grammar_enabled: bool = True,
    meeting_grammar_enabled: bool = True,
) -> GrammarPostPassResult:
    """Auto-fire ``code_grammar`` / ``meeting_grammar`` based on
    ``app_category``.

    Runs *after* :func:`apply_commit_boundary_rules` so the no-touch
    code/terminal branch is preserved (text comes through unchanged
    from boundary rules and this post-pass shapes it).

    Returns a :class:`GrammarPostPassResult` so callers can stamp
    trace metadata. ``applied=False`` when the category doesn't
    qualify, the toggle is off, the text is empty, or the underlying
    engine produced no change.
    """
    if not text:
        return GrammarPostPassResult(text=text or "", applied=False)
    cat = (app_category or "").strip().lower()

    if cat in _CODE_CATEGORIES and code_grammar_enabled:
        engine = CodeGrammarEngine()
        result = engine.apply(text, mode=CodeGrammarMode.AUTO)
        if result.changed:
            return GrammarPostPassResult(
                text=result.text,
                applied=True,
                engine="code_grammar",
                rules_applied=list(result.rules_applied),
            )

    if cat in _MEETING_CATEGORIES and meeting_grammar_enabled:
        engine = MeetingGrammarEngine()
        result = engine.apply(text, mode=MeetingGrammarMode.AUTO)
        if result.changed:
            return GrammarPostPassResult(
                text=result.text,
                applied=True,
                engine="meeting_grammar",
                rules_applied=list(result.rules_applied),
            )

    return GrammarPostPassResult(text=text, applied=False)


def _formatted_output_rejection_reason(
    source: str,
    formatted: str,
    *,
    required_terms: list[str] | tuple[str, ...] | None = None,
    context_terms: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    if not formatted:
        return "empty_output"
    lowered = formatted.strip().casefold()
    if lowered.startswith(("sure", "here is", "here's", "the formatted text is")):
        return "assistant_preamble"
    if _contains_unsolicited_markdown_emphasis(source, formatted):
        return "unsupported_markdown_emphasis"
    if _context_only_terms_added(source, formatted, context_terms):
        return "context_terms_added"
    src_words = max(1, len((source or "").split()))
    out_words = len(formatted.split())
    if src_words >= 4:
        ratio = out_words / src_words
        if ratio < 0.35 or ratio > 2.2:
            return "word_count_ratio"
    if not _content_words_preserved(source, formatted):
        return "content_words_dropped"
    if _missing_required_terms(source, formatted, required_terms):
        return "required_terms_dropped"
    if _source_required_terms_dropped(source, formatted):
        return "source_required_terms_dropped"
    return None


_FORMATTER_CONTENT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "me", "my", "no", "not", "now", "of",
    "on", "or", "our", "she", "so", "that", "the", "then", "there",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "who", "why", "will", "with", "would", "you", "your",
})
_SOURCE_REQUIRED_TERM_STOPWORDS = _FORMATTER_CONTENT_STOPWORDS | frozenset({
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "hey", "hi", "hello", "note", "please",
})


def _contains_unsolicited_markdown_emphasis(source: str, formatted: str) -> bool:
    if "**" not in (formatted or ""):
        return False
    return "**" not in (source or "")


def _content_words_preserved(source: str, formatted: str) -> bool:
    ignored_source_tokens = _structural_formatting_instruction_tokens(source)
    source_tokens = [
        token
        for token in _formatter_content_tokens(source)
        if token not in ignored_source_tokens
    ]
    if len(source_tokens) < 8:
        return True
    output_tokens = set(_formatter_content_tokens(formatted))
    if not output_tokens:
        return False
    preserved = sum(1 for token in source_tokens if token in output_tokens)
    return preserved / max(1, len(source_tokens)) >= 0.62


def _formatter_content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for match in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or ""):
        token = match.group(0).casefold()
        if len(token) <= 2:
            continue
        if token in _FORMATTER_CONTENT_STOPWORDS:
            continue
        out.append(token)
    return out


def _build_rejection(
    reason: str,
    *,
    source: str,
    formatted: str,
    backend: str,
    required_terms: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    rejection: dict[str, Any] = {
        "reason": reason,
        "source_chars": len(source or ""),
        "output_chars": len(formatted or ""),
        "source_words": len((source or "").split()),
        "output_words": len((formatted or "").split()),
        "backend": backend,
    }
    missing = _missing_required_terms(source, formatted, required_terms)
    if missing:
        rejection["missing_required_terms"] = missing
    source_missing = _source_required_terms_dropped(source, formatted)
    if source_missing:
        rejection["missing_source_required_terms"] = source_missing
    return rejection


def _required_terms_for_packet(packet: FormattingPacket) -> list[str]:
    metadata = packet.metadata or {}
    terms: list[str] = []
    for key in ("required_preserved_terms", "candidate_entities", "recent_screen_terms"):
        value = metadata.get(key)
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            term = str(item or "").strip()
            if not term:
                continue
            if not _high_value_term(term):
                continue
            if not _term_present_in_text(term, packet.corrected_text):
                continue
            if term.casefold() in {x.casefold() for x in terms}:
                continue
            terms.append(term)
    return terms[:24]


def _source_required_terms_dropped(source: str, formatted: str) -> list[str]:
    missing: list[str] = []
    for term in _source_required_terms(source):
        if _term_present_in_text(term, formatted):
            continue
        missing.append(term)
    return missing


def _source_required_terms(source: str) -> list[str]:
    terms: list[str] = []
    structural_counts = _structural_instruction_count_terms(source)
    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9]{2,}|[A-Z]{2,}(?:\s+[A-Z]{2,})*|\d+(?:[.:/-]\d+)*)\b", source or ""):
        term = match.group(0).strip()
        if not term or term.casefold() in _SOURCE_REQUIRED_TERM_STOPWORDS:
            continue
        if term in structural_counts:
            continue
        if term.casefold() in {item.casefold() for item in terms}:
            continue
        terms.append(term)
    return terms[:24]


def _structural_instruction_count_terms(source: str) -> set[str]:
    text = source or ""
    counts: set[str] = set()
    count_nouns = r"(?:points?|items?|bullets?|bullet\s+points?|numbered\s+items?|steps?|todos?|tasks?|takeaways?)"
    lead_verbs = (
        r"(?:note\s+down|write\s+down|list|make|create|draft|outline|give\s+me|"
        r"turn\s+(?:this|that|it)\s+into)"
    )
    patterns = (
        rf"\b{lead_verbs}\s+(?P<count>\d+)\s+{count_nouns}\b",
        rf"\b(?P<count>\d+)\s+{count_nouns}\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            counts.add(match.group("count"))
    return counts


def _structural_formatting_instruction_tokens(source: str) -> set[str]:
    if not _looks_like_structural_formatting_request(source):
        return set()
    return {
        "bullet",
        "bullets",
        "create",
        "down",
        "draft",
        "eighth",
        "fifth",
        "first",
        "fourth",
        "give",
        "items",
        "list",
        "make",
        "ninth",
        "note",
        "numbered",
        "outline",
        "point",
        "points",
        "second",
        "seventh",
        "sixth",
        "steps",
        "takeaways",
        "tasks",
        "tenth",
        "third",
        "todos",
        "write",
    }


def _looks_like_structural_formatting_request(source: str) -> bool:
    text = source or ""
    if _structural_instruction_count_terms(text):
        return True
    return bool(
        re.search(
            r"\b(?:note\s+down|write\s+down|turn\s+(?:this|that|it)\s+into|"
            r"bullets?|numbered\s+list|checklist|first\b.+\bsecond\b)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _missing_required_terms(
    source: str,
    formatted: str,
    required_terms: list[str] | tuple[str, ...] | None,
) -> list[str]:
    missing: list[str] = []
    for term in required_terms or []:
        t = str(term or "").strip()
        if not t:
            continue
        if not _term_present_in_text(t, source):
            continue
        if _term_present_in_text(t, formatted):
            continue
        missing.append(t)
    return missing


def _context_terms_for_packet(packet: FormattingPacket) -> list[str]:
    metadata = packet.metadata or {}
    terms: list[str] = []

    def add(value: object) -> None:
        text = str(value or "").strip()
        if not text or len(text) > 120:
            return
        if text.casefold() in {item.casefold() for item in terms}:
            return
        if _term_present_in_text(text, packet.corrected_text):
            return
        if not _high_value_term(text):
            return
        terms.append(text)

    add(packet.app_name)
    add(packet.window_title)
    for key in ("candidate_entities", "recent_screen_terms", "required_preserved_terms"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            for item in value[:40]:
                add(item)
    return terms[:40]


def _context_only_terms_added(
    source: str,
    formatted: str,
    context_terms: list[str] | tuple[str, ...] | None,
) -> list[str]:
    added: list[str] = []
    for term in context_terms or []:
        t = str(term or "").strip()
        if not t:
            continue
        if _term_present_in_text(t, source):
            continue
        if not _term_present_in_text(t, formatted):
            continue
        added.append(t)
    return added


def _term_present_in_text(term: str, text: str) -> bool:
    term_units = _term_units(term)
    text_units = _term_units(text)
    if not term_units or not text_units:
        return False
    width = len(term_units)
    for idx in range(0, len(text_units) - width + 1):
        if text_units[idx : idx + width] == term_units:
            return True
    compact_term = "".join(term_units)
    if len(compact_term) >= 4 and compact_term in "".join(text_units):
        return True
    return False


def _term_units(text: str) -> list[str]:
    return [m.group(0).casefold() for m in re.finditer(r"[A-Za-z0-9]+", text or "")]


def _high_value_term(term: str) -> bool:
    units = _term_units(term)
    if not units:
        return False
    compact = "".join(units)
    if len(compact) < 3:
        return False
    if len(units) > 1:
        return True
    token = units[0]
    if token in _FORMATTER_CONTENT_STOPWORDS:
        return False
    return term[:1].isupper() or any(ch.isdigit() for ch in term) or any(ch.isupper() for ch in term[1:])
