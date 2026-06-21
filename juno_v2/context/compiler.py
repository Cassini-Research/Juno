from __future__ import annotations

import os
import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from juno_v2.context.provider import _context_candidate_allowed, _extract_candidates
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemoryServingPacket, MemorySnapshot
from juno_v2.contracts.modes import ModePolicy, ModeSelection
from juno_v2.memory.bias import screen_term_prompt_worthy
from juno_v2.memory.entity_policy import session_entity_allowed
from juno_v2.memory.ranking import rank_memory_for_context
from juno_v2.personalization.seed.models import SeedBiasAttachment
from juno_v2.runtime.deployment import _env_int

CompiledTermSource = Literal["memory", "replacement", "correction", "screen", "selection", "session", "file", "symbol", "snippet", "style"]


@dataclass(slots=True)
class CompiledTerm:
    text: str
    canonical: str | None
    spoken_forms: tuple[str, ...]
    source: CompiledTermSource
    priority: float
    protected: bool = False
    scope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "canonical": self.canonical,
            "spoken_forms": list(self.spoken_forms),
            "source": self.source,
            "priority": self.priority,
            "protected": self.protected,
            "scope": self.scope,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AsrBiasPacket:
    initial_prompt: str | None
    bias_phrases: tuple[str, ...]
    terms: tuple[CompiledTerm, ...]
    max_prompt_chars: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_prompt": self.initial_prompt,
            "bias_phrases": list(self.bias_phrases),
            "terms": [t.to_dict() for t in self.terms],
            "max_prompt_chars": self.max_prompt_chars,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class TranscriptAdjudicationPacket:
    stage: Literal["live", "final"]
    utterance_id: str
    base_visible_text: str
    base_visible_revision: int | None
    live_preview_text: str
    whisper_text: str
    memory_candidate_text: str
    raw_text: str
    context_terms: tuple[CompiledTerm, ...]
    protected_terms: tuple[str, ...]
    selected_text_excerpt: str
    focused_text_before: str
    focused_text_after: str
    field_text_excerpt: str
    app_name: str | None
    app_category: str | None
    window_title: str | None
    focused_file_path: str | None
    symbol_under_cursor: str | None
    mode_name: str
    transcript_policy: str
    final_formatting_policy: str
    no_touch: bool
    privacy_suppressed: bool
    language: str | None
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        final_stage = self.stage == "final"
        live_preview_visible = "" if final_stage else self.live_preview_text
        base_visible_text = "" if final_stage else self.base_visible_text
        return {
            "task": "transcript_adjudication_v1",
            "stage": self.stage,
            "mode": {
                "name": self.mode_name,
                "transcript_policy": self.transcript_policy,
                "final_formatting_policy": self.final_formatting_policy,
                "no_touch": self.no_touch,
            },
            "asr": {
                "live_preview_visible": live_preview_visible,
                "whisper_raw": self.whisper_text,
                "memory_candidate": self.memory_candidate_text,
                "raw_text": self.raw_text,
            },
            "evidence_contract": {
                "final_authoritative_sources": ["memory_candidate", "whisper_raw", "raw_text"],
                "live_preview_visible": (
                    "low_trust_ui_hint_not_final_content_evidence"
                    if self.stage == "final"
                    else "target_live_window_hint"
                ),
                "base_visible": (
                    "low_trust_ui_hint_not_final_content_evidence"
                    if self.stage == "final"
                    else "editable_live_window"
                ),
                "rule": (
                    "For final stage, do not add words, numbers, names, or phrases "
                    "that only appear in live_preview_visible or base_visible."
                ),
            },
            "speech_resolution_contract": {
                "role": (
                    "Resolve what the user meant to say from the complete ASR evidence, "
                    "local terms, app context, and explicit spoken correction cues."
                ),
                "corrected_text": (
                    "Content-preserving final transcript. Resolve self-corrections, "
                    "ASR mistakes, spoken punctuation, protected terms, dates, and casing. "
                    "Do not apply note/email/list formatting here."
                ),
                "intent": (
                    "Optional object describing dictation, structured_dictation, transform, "
                    "action, or mixed intent. This is evidence for downstream routing; "
                    "corrected_text remains a transcript."
                ),
                "formatting_plan": (
                    "Optional object describing requested structure such as bullets, numbered "
                    "lists, sections, email, or no_formatting. Do not directly format corrected_text."
                ),
                "self_corrections": (
                    "Optional array of resolved spoken corrections with evidence spans. "
                    "Literal mentions such as 'the words blank space' must be preserved."
                ),
                "uncertainty": "Optional array for low-confidence spans or needed confirmation.",
            },
            "context": {
                "app_name": self.app_name,
                "app_category": self.app_category,
                "window_title": self.window_title,
                "selected_text_excerpt": self.selected_text_excerpt,
                "focused_text_before": self.focused_text_before,
                "focused_text_after": self.focused_text_after,
                "field_text_excerpt": self.field_text_excerpt,
                "focused_file_path": self.focused_file_path,
                "symbol_under_cursor": self.symbol_under_cursor,
            },
            "terms": [t.to_dict() for t in self.context_terms],
            "protected_terms": list(self.protected_terms),
            "base_visible": {
                "text": base_visible_text,
                "revision": self.base_visible_revision,
                "hash": self.metadata.get("base_text_hash"),
                "stable_prefix_chars": self.metadata.get("stable_prefix_chars"),
            },
            "output_schema": {
                "schema_version": "transcript_adjudication_v1",
                "corrected_text": "string",
                "ops": "[] for final stage; array only for live target-window patches",
                "confidence": "number",
                "protected_terms_used": "array",
                "intent": "optional object",
                "formatting_plan": "optional object",
                "self_corrections": "optional array",
                "terms_used": "optional array",
                "uncertainty": "optional array",
            },
            "diagnostics": {
                "self_correction_cues": list(self.metadata.get("self_correction_cues") or []),
                "post_asr_context_enriched": bool(self.metadata.get("post_asr_context_enriched")),
                "post_asr_context_terms": list(self.metadata.get("post_asr_context_terms") or []),
            },
        }


@dataclass(slots=True)
class FormattingPacket:
    utterance_id: str
    corrected_text: str
    app_name: str | None
    app_category: str | None
    window_title: str | None
    mode_name: str
    final_formatting_policy: str
    style_card: dict[str, Any] | None
    focused_text_before: str
    focused_text_after: str
    selected_text_excerpt: str
    writer_tone_addon: str | None
    metadata: dict[str, Any]
    # The mode policy's prompt_prefix — a tone/structure nudge that is
    # appended to Qwen's system prompt for final_formatting_v1 calls so
    # built-in modes (formal_email, casual_chat, structured_notes,
    # explicit_rewrite) and user-authored custom modes actually shape
    # the LLM's behaviour, not just the deterministic post-pass.
    # Defaults to None for backward compat with constructors that
    # haven't been updated yet — _system_prompt treats None / "" as a
    # no-op.
    mode_prompt_prefix: str | None = None


@dataclass(slots=True)
class ActionPacket:
    utterance_id: str
    corrected_text: str
    raw_or_normalized_text_for_wake_gate: str
    now_iso: str | None
    context_terms: tuple[CompiledTerm, ...]
    metadata: dict[str, Any]


@dataclass(slots=True)
class SelectionTransformPacket:
    utterance_id: str
    instruction_text: str
    selected_text: str
    selected_word_count: int
    app_name: str | None
    app_category: str | None
    focused_text_before: str
    focused_text_after: str
    mode_name: str
    style_card: dict[str, Any] | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class CompiledContext:
    utterance_id: str
    context: TypedContextBundle
    memory_snapshot: MemorySnapshot
    mode_selection: ModeSelection
    mode_policy: ModePolicy
    memory_packet: MemoryServingPacket
    terms: tuple[CompiledTerm, ...]
    transcript_hint: str | None
    language: str | None
    stage: str
    metadata: dict[str, Any]

    def asr_bias_packet(self, max_prompt_chars: int | None = None) -> AsrBiasPacket:
        budget = int(max_prompt_chars or _env_int("JUNO_V2_ASR_BIAS_PROMPT_CHARS", 768))
        asr_terms = tuple(t for t in self.terms if _asr_bias_term_allowed(t))[:16]
        phrases = tuple(t.text for t in asr_terms)
        prefer = _pack_prefer_line(phrases, max_chars=budget)
        return AsrBiasPacket(
            initial_prompt=prefer or None,
            bias_phrases=phrases,
            terms=asr_terms,
            max_prompt_chars=budget,
            metadata={
                "term_count": len(self.terms),
                "asr_term_count": len(asr_terms),
                "memory_packet_counts": _memory_counts(self.memory_snapshot),
            },
        )

    def transcript_packet(
        self,
        *,
        stage: Literal["live", "final"],
        base_visible_text: str = "",
        base_visible_revision: int | None = None,
        live_preview_text: str = "",
        whisper_text: str = "",
        memory_candidate_text: str = "",
        raw_text: str = "",
    ) -> TranscriptAdjudicationPacket:
        live = stage == "live"
        evidence_text = _join_context_hints(
            base_visible_text if live else None,
            live_preview_text if live else None,
            whisper_text,
            memory_candidate_text,
            raw_text,
        )
        context_terms = (
            self.terms[:24]
            if live
            else _final_transcript_context_terms(self.terms, evidence_text=evidence_text)[:24]
        )
        protected_terms = (
            tuple(t.canonical or t.text for t in self.terms if t.protected)[:32]
            if live
            else _final_transcript_protected_terms(context_terms, evidence_text=evidence_text)[:16]
        )
        return TranscriptAdjudicationPacket(
            stage=stage,
            utterance_id=self.utterance_id,
            base_visible_text=base_visible_text or "",
            base_visible_revision=base_visible_revision,
            live_preview_text=live_preview_text or base_visible_text or "",
            whisper_text=whisper_text or "",
            memory_candidate_text=memory_candidate_text or whisper_text or raw_text or "",
            raw_text=raw_text or whisper_text or "",
            context_terms=context_terms,
            protected_terms=protected_terms,
            selected_text_excerpt=_words(self.context.selected_text, 1000) if not live else self.context.selected_text[:1200],
            focused_text_before=self.context.focused_text_before[-(400 if live else 1000):],
            focused_text_after=self.context.focused_text_after[: (250 if live else 600)],
            field_text_excerpt=self.context.field_text_excerpt[:1200],
            app_name=self.context.app_name,
            app_category=self.context.app_category,
            window_title=self.context.window_title,
            focused_file_path=self.context.focused_file_path,
            symbol_under_cursor=self.context.symbol_under_cursor,
            mode_name=self.mode_selection.effective_mode,
            transcript_policy=getattr(self.mode_policy, "transcript_correction_policy", "standard"),
            final_formatting_policy=getattr(self.mode_policy, "final_formatting_policy", "minimal"),
            no_touch=(self.context.app_category or "").strip().lower() in {"terminal", "code"},
            privacy_suppressed=str(self.metadata.get("suppression") or "none") not in {"", "none"},
            language=self.language,
            metadata=dict(self.metadata),
        )

    def formatting_packet(
        self,
        *,
        corrected_text: str,
        style_card: dict[str, Any] | None = None,
        writer_tone_addon: str | None = None,
    ) -> FormattingPacket:
        return FormattingPacket(
            utterance_id=self.utterance_id,
            corrected_text=corrected_text,
            app_name=self.context.app_name,
            app_category=self.context.app_category,
            window_title=self.context.window_title,
            mode_name=self.mode_selection.effective_mode,
            final_formatting_policy=getattr(self.mode_policy, "final_formatting_policy", "minimal"),
            style_card=style_card,
            focused_text_before=self.context.focused_text_before[-1000:],
            focused_text_after=self.context.focused_text_after[:600],
            selected_text_excerpt=self.context.selected_text[:1200],
            writer_tone_addon=writer_tone_addon,
            mode_prompt_prefix=getattr(self.mode_policy, "prompt_prefix", "") or None,
            metadata=dict(self.metadata),
        )

    def action_packet(
        self,
        *,
        corrected_text: str,
        raw_or_normalized_text_for_wake_gate: str,
        now_iso: str | None = None,
    ) -> ActionPacket:
        return ActionPacket(
            utterance_id=self.utterance_id,
            corrected_text=corrected_text,
            raw_or_normalized_text_for_wake_gate=raw_or_normalized_text_for_wake_gate,
            now_iso=now_iso,
            context_terms=self.terms[:24],
            metadata=dict(self.metadata),
        )

    def selection_transform_packet(
        self,
        *,
        instruction_text: str,
        selected_text: str,
        style_card: dict[str, Any] | None = None,
    ) -> SelectionTransformPacket:
        return SelectionTransformPacket(
            utterance_id=self.utterance_id,
            instruction_text=instruction_text,
            selected_text=_words(selected_text, 1000),
            selected_word_count=len((selected_text or "").split()),
            app_name=self.context.app_name,
            app_category=self.context.app_category,
            focused_text_before=self.context.focused_text_before[-1000:],
            focused_text_after=self.context.focused_text_after[:600],
            mode_name=self.mode_selection.effective_mode,
            style_card=style_card,
            metadata=dict(self.metadata),
        )


def compile_context(
    *,
    utterance_id: str,
    context: TypedContextBundle,
    memory_snapshot: MemorySnapshot,
    mode_selection: ModeSelection,
    mode_policy: ModePolicy,
    transcript_hint: str | None,
    session_terms: list[str] | None,
    language: str | None,
    stage: str,
    seed_attachment: SeedBiasAttachment | None = None,
    final_transcript_text: str | None = None,
) -> CompiledContext:
    ranking_hint = _join_context_hints(transcript_hint, final_transcript_text)
    memory_packet = rank_memory_for_context(
        memory_snapshot,
        context=context,
        mode_policy=mode_policy,
        effective_mode=mode_selection.effective_mode,
        transcript_hint=ranking_hint,
        session_terms=session_terms,
    )
    terms = _compile_terms(
        context=context,
        snapshot=memory_snapshot,
        memory_packet=memory_packet,
        transcript_hint=ranking_hint,
        session_terms=session_terms,
        seed_attachment=seed_attachment,
    )
    return CompiledContext(
        utterance_id=utterance_id,
        context=context,
        memory_snapshot=memory_snapshot,
        mode_selection=mode_selection,
        mode_policy=mode_policy,
        memory_packet=memory_packet,
        terms=tuple(terms),
        transcript_hint=transcript_hint,
        language=language,
        stage=stage,
        metadata={
            "stage": stage,
            "term_count": len(terms),
            "candidate_term_count": len(context.candidate_entities or []),
            "session_term_count": len(session_terms or []),
            "memory_packet_counts": _memory_counts(memory_snapshot),
            "seed_attached": seed_attachment is not None,
            "post_asr_context_enriched": bool(final_transcript_text and final_transcript_text.strip()),
            "post_asr_context_terms": [t.text for t in terms[:16]],
        },
    )


_COMMON = {
    "the", "and", "for", "that", "this", "with", "from", "you", "your",
    "have", "will", "just", "into", "about", "what", "when", "where",
    "who", "why", "how", "then", "there", "here", "they", "them", "our",
    "are", "was", "were", "been", "does", "did", "not", "yes", "but",
    "she", "him", "his", "her", "its", "can",
    "is", "it", "in", "on", "of", "to", "as", "at", "by", "or", "we", "he",
    "ai", "so", "while",
}


def _compile_terms(
    *,
    context: TypedContextBundle,
    snapshot: MemorySnapshot,
    memory_packet: MemoryServingPacket,
    transcript_hint: str | None,
    session_terms: list[str] | None,
    seed_attachment: SeedBiasAttachment | None,
) -> list[CompiledTerm]:
    seen: set[str] = set()
    out: list[CompiledTerm] = []

    def add(
        text: str,
        *,
        source: CompiledTermSource,
        priority: float,
        canonical: str | None = None,
        spoken_forms: tuple[str, ...] = (),
        protected: bool = False,
        scope: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        t = (text or "").strip()
        if not _term_allowed(t):
            return
        key = t.casefold()
        if key in seen:
            return
        seen.add(key)
        out.append(
            CompiledTerm(
                text=t,
                canonical=canonical,
                spoken_forms=spoken_forms,
                source=source,
                priority=priority,
                protected=protected,
                scope=scope,
                metadata=metadata or {},
            )
        )

    # Selection-derived terms get highest priority. Re-run the unified
    # extractor against only the selected_text so we can attribute them
    # to the right source — context.candidate_entities is a merged list
    # across all surfaces and would mis-attribute terms that happen to
    # appear elsewhere.
    if context.selected_text:
        for term in _extract_candidates([context.selected_text])[:8]:
            add(term, source="selection", priority=100.0, protected=True)
    if context.symbol_under_cursor:
        add(context.symbol_under_cursor, source="symbol", priority=96.0, protected=True)
    if context.focused_file_path:
        base = os.path.basename(context.focused_file_path)
        add(base, source="file", priority=92.0, protected=True, metadata={"path": context.focused_file_path})
        stem, _ext = os.path.splitext(base)
        add(stem, source="file", priority=88.0, protected=True, metadata={"path": context.focused_file_path})
    explicit_candidate_terms = {
        str(term or "").strip().casefold()
        for term in (context.metadata or {}).get("explicit_candidate_entities", [])
        if str(term or "").strip()
    }

    # Screen terms come from the same extractor that populated
    # context.candidate_entities at bundle-build time. Sharing the
    # extractor with the Whisper bias path (RecognitionBiasEngine) means
    # the LLM adjudicator and Whisper see the same set of "rare" screen
    # tokens — they used to diverge because this compile path ran its
    # own stricter uppercase-only regex.
    for term in context.candidate_entities[:16]:
        explicit = term.casefold() in explicit_candidate_terms
        if not _context_candidate_allowed(term):
            continue
        if not explicit and not screen_term_prompt_worthy(term):
            continue
        add(term, source="screen", priority=82.0, protected=explicit or _looks_identifier(term) or _looks_proper_noun(term))

    hint_tokens = _tokens(" ".join([transcript_hint or "", context.window_title or "", context.app_name or ""]))
    for term in _hint_matched_memory_subterms(snapshot, hint_tokens)[:8]:
        add(
            term,
            source="memory",
            priority=86.0,
            canonical=term,
            spoken_forms=tuple(sorted(h for h in hint_tokens if _hint_matches_memory_subterm(h, term)))[:4],
            protected=True,
            metadata={"source": "hint_matched_memory_subterm"},
        )
    for rule in snapshot.replacements:
        trigger = (rule.trigger or "").strip()
        replacement = (rule.replacement or "").strip()
        overlap = bool(_tokens(trigger) & hint_tokens)
        add(
            replacement,
            source="replacement",
            priority=78.0 + (10.0 if overlap else 0.0),
            canonical=replacement,
            spoken_forms=(trigger,),
            protected=True,
            scope=rule.scope,
        )
    for entry in snapshot.lexicon:
        add(
            entry.canonical_form,
            source="memory",
            priority=62.0 + min(20.0, float(entry.boost) * 4.0),
            canonical=entry.canonical_form,
            spoken_forms=tuple(x for x in [entry.term, *entry.aliases] if x),
            protected=_looks_identifier(entry.canonical_form) or _looks_proper_noun(entry.canonical_form) or bool(entry.boost >= 1.4),
        )
    for pair in snapshot.corrections:
        add(
            pair.corrected,
            source="correction",
            priority=55.0 + min(15.0, float(pair.count)),
            canonical=pair.corrected,
            spoken_forms=(pair.observed,),
        )
    for term in session_terms or []:
        if not session_entity_allowed(term):
            continue
        add(term, source="session", priority=45.0, protected=_looks_identifier(term) or _looks_proper_noun(term))
    for ent in snapshot.session_entities:
        if not session_entity_allowed(ent.value):
            continue
        add(ent.value, source="session", priority=42.0 + min(8.0, float(ent.count)), protected=_looks_identifier(ent.value) or _looks_proper_noun(ent.value))
    for term in memory_packet.lexicon_terms[:12]:
        add(term, source="memory", priority=50.0)

    out.sort(key=lambda t: (-t.priority, t.text.casefold()))
    return out[:64]


_FINAL_TRANSCRIPT_ALWAYS_HINT_SOURCES = frozenset({"screen", "selection", "file", "symbol"})
_FINAL_TRANSCRIPT_EVIDENCE_SOURCES = frozenset(
    {"memory", "replacement", "correction", "session", "snippet", "style"}
)
_FINAL_TRANSCRIPT_ALWAYS_PROTECT_SOURCES = frozenset({"selection", "file", "symbol"})


def _final_transcript_context_terms(
    terms: tuple[CompiledTerm, ...],
    *,
    evidence_text: str,
) -> tuple[CompiledTerm, ...]:
    out: list[CompiledTerm] = []
    seen: set[str] = set()
    for term in terms:
        if term.source in _FINAL_TRANSCRIPT_ALWAYS_HINT_SOURCES:
            allowed = True
        elif term.source in _FINAL_TRANSCRIPT_EVIDENCE_SOURCES:
            allowed = _compiled_term_present_in_text(term, evidence_text)
        else:
            allowed = _compiled_term_present_in_text(term, evidence_text)
        if not allowed:
            continue
        key = (term.canonical or term.text).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return tuple(out)


def _final_transcript_protected_terms(
    terms: tuple[CompiledTerm, ...],
    *,
    evidence_text: str,
) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term.protected:
            continue
        value = (term.canonical or term.text or "").strip()
        if not value:
            continue
        if term.source not in _FINAL_TRANSCRIPT_ALWAYS_PROTECT_SOURCES and not _compiled_term_present_in_text(term, evidence_text):
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return tuple(out)


def _compiled_term_present_in_text(term: CompiledTerm, text: str) -> bool:
    for value in (term.text, term.canonical, *term.spoken_forms):
        if _term_present_loose(text, value):
            return True
    return False


def _term_present_loose(text: str, term: str | None) -> bool:
    tokens = re.findall(r"[A-Za-z0-9]+", term or "")
    if not tokens:
        return False
    separator = r"[\W_]*" if len(tokens) >= 2 and all(len(tok) == 1 and tok.isalpha() for tok in tokens) else r"[\W_]+"
    pattern = r"(?<![A-Za-z0-9])" + separator.join(re.escape(tok) for tok in tokens) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, text or "", flags=re.IGNORECASE))


def _term_allowed(text: str) -> bool:
    if not text or len(text) < 2 or len(text) > 80:
        return False
    if text.casefold() in _COMMON:
        return False
    if len(text.split()) > 8:
        return False
    if text.count(".") > 4:
        return False
    return True


def _join_context_hints(*values: str | None) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        v = re.sub(r"\s+", " ", (value or "").strip())
        if not v:
            continue
        key = v.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return " ".join(out)


def _looks_identifier(text: str) -> bool:
    if not text:
        return False
    # camelCase / dotted.extension / snake_or_kebab — classic code shapes.
    if re.search(r"[A-Z][a-z]+[A-Z]|\.[A-Za-z0-9]{1,5}$|[_/-]", text):
        return True
    # Letter+digit mixed (alpha42, cosmos1, chrome120) — by construction
    # never a regular English word, so safe to treat as protected and
    # have Qwen preserve exact spelling.
    if any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
        return True
    return False


def _looks_proper_noun(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return False
    if t.casefold() in _COMMON:
        return False
    if _looks_like_glued_pronoun_i(t):
        return False
    return bool(re.match(r"^[A-Z][A-Za-z]{2,}$", t) or re.search(r"[a-z][A-Z]", t))


def _looks_named_phrase(text: str) -> bool:
    t = (text or "").strip()
    if _looks_identifier(t) or _looks_proper_noun(t):
        return True
    parts = re.findall(r"[A-Za-z][A-Za-z0-9]*", t)
    if not parts or len(parts) > 4:
        return False
    return all(part[:1].isupper() and part.casefold() not in _COMMON for part in parts)


def _asr_bias_term_allowed(term: CompiledTerm) -> bool:
    if term.source in {"snippet", "style"}:
        return False
    if term.source in {"selection", "symbol", "file"}:
        return True
    if term.source == "screen":
        return term.protected or _looks_named_phrase(term.text)
    if term.source in {"replacement", "correction"}:
        return term.protected or term.priority >= 60.0
    if term.source == "session":
        return term.protected or term.priority >= 48.0 or _looks_named_phrase(term.text)
    if term.source == "memory":
        return term.protected or term.priority >= 70.0 or (term.priority >= 60.0 and _looks_named_phrase(term.text))
    return False


def _looks_like_glued_pronoun_i(text: str) -> bool:
    return bool(re.match(r"^[a-z]{2,}I(?:m|d|ll|ve|re)?$", (text or "").strip()))


def _hint_matched_memory_subterms(snapshot: MemorySnapshot, hint_tokens: set[str]) -> list[str]:
    if not hint_tokens:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for entry in snapshot.lexicon:
        surfaces = [
            str(getattr(entry, "canonical_form", "") or ""),
            str(getattr(entry, "term", "") or ""),
            *(str(alias or "") for alias in getattr(entry, "aliases", ()) or ()),
        ]
        for surface in surfaces:
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", surface):
                if not _rare_memory_subterm_allowed(token):
                    continue
                if not any(_hint_matches_memory_subterm(hint, token) for hint in hint_tokens):
                    continue
                key = token.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(token)
                if len(out) >= 16:
                    return out
    return out


def _rare_memory_subterm_allowed(token: str) -> bool:
    t = (token or "").strip()
    if len(t) < 3 or len(t) > 32:
        return False
    if t.casefold() in _COMMON:
        return False
    folded = t.casefold()
    return (
        t.isupper()
        or any(ch.isdigit() for ch in t)
        or bool(re.search(r"[a-z][A-Z]", t))
        or bool(re.search(r"q(?!u)", folded))
    )


def _hint_matches_memory_subterm(hint: str, term: str) -> bool:
    h = (hint or "").strip().casefold()
    t = (term or "").strip().casefold()
    if len(h) < 3 or not t:
        return False
    if h == t:
        return True
    if not _rare_memory_subterm_allowed(h):
        return False
    if h[:1] != t[:1] or abs(len(h) - len(t)) > 2:
        return False
    if not _rare_memory_subterm_allowed(term):
        return False
    return difflib.SequenceMatcher(a=h, b=t, autojunk=False).ratio() >= 0.74


def _tokens(text: str) -> set[str]:
    return {p.strip(".,!?;:()[]{}<>\"'`").casefold() for p in (text or "").replace("_", " ").replace("-", " ").split() if len(p.strip()) >= 2}


def _words(text: str, limit: int) -> str:
    parts = (text or "").split()
    if len(parts) <= limit:
        return text or ""
    return " ".join(parts[:limit])




def _pack_prefer_line(phrases: tuple[str, ...], *, max_chars: int) -> str:
    prefix = "Prefer exact forms: "
    if not phrases or max_chars <= len(prefix):
        return ""
    used = len(prefix)
    out: list[str] = []
    for phrase in phrases:
        add = len(phrase) + (2 if out else 0)
        if used + add > max_chars:
            break
        out.append(phrase)
        used += add
    return prefix + ", ".join(out) if out else ""


def _memory_counts(snapshot: MemorySnapshot) -> dict[str, int]:
    return {
        "lexicon": len(snapshot.lexicon),
        "replacements": len(snapshot.replacements),
        "corrections": len(snapshot.corrections),
        "session_entities": len(snapshot.session_entities),
    }
