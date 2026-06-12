from __future__ import annotations

import os
import re
import typing
from dataclasses import dataclass, field

from juno_v2.contracts.context import RecognitionBiasPlan, TypedContextBundle
from juno_v2.contracts.memory import MemoryServingPacket, MemorySnapshot, NormalizationChange, TranscriptNormalization
from juno_v2.contracts.modes import ModePolicy
from juno_v2.memory.fold import fold_match_pattern
from juno_v2.memory.profile import build_personalization_profile
from juno_v2.memory.ranking import rank_memory_for_context
from juno_v2.memory.entity_policy import session_entity_allowed
from juno_v2.memory.store import _looks_like_hallucination
from juno_v2.memory.stores.corrections import is_safe_correction_pair
from juno_v2.memory.term_policy import learned_term_allowed
from juno_v2.runtime.deployment import _env_int

# Issue #28 — PII guard for ``recent_finals`` before they enter
# ``initial_prompt``. Whisper feeds the prompt back into decoding, so a
# credit-card number from a prior utterance could re-emerge in the next
# transcript. Mask digit runs of >=6 chars unconditionally.
_DIGIT_RUN_RE = re.compile(r"\d{6,}")
_INITIAL_PROMPT_MAX_WORDS = 140

# Conservative gate: avoid rewriting high-frequency English with seed variants.
_SEED_CANON_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "its",
        "let",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "she",
        "use",
    }
)

_LOW_SIGNAL_SESSION_ENTITY_WORDS = frozenset(
    {
        "a",
        "all",
        "an",
        "also",
        "and",
        "ai",
        "ah",
        "app",
        "applications",
        "are",
        "as",
        "at",
        "be",
        "been",
        "bunch",
        "bar",
        "buddy",
        "but",
        "by",
        "can",
        "clear",
        "clearly",
        "code",
        "context",
        "create",
        "customer",
        "deadline",
        "design",
        "decode",
        "did",
        "document",
        "do",
        "does",
        "edited",
        "every",
        "focus",
        "font",
        "formatting",
        "fresh",
        "for",
        "from",
        "get",
        "go",
        "he",
        "here",
        "hmm",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "last",
        "like",
        "me",
        "missing",
        "may",
        "mmm",
        "my",
        "no",
        "not",
        "owner",
        "regular",
        "status",
        "style",
        "task",
        "title",
        "one",
        "of",
        "ok",
        "okay",
        "on",
        "or",
        "our",
        "pull",
        "request",
        "pro",
        "see",
        "she",
        "similarly",
        "so",
        "start",
        "that",
        "thank",
        "the",
        "then",
        "there",
        "they",
        "things",
        "this",
        "today",
        "to",
        "tomorrow",
        "um",
        "umm",
        "was",
        "we",
        "were",
        "what",
        "whatever",
        "when",
        "where",
        "while",
        "which",
        "who",
        "why",
        "would",
        "with",
        "yes",
        "yesterday",
        "you",
    }
)

_GENERIC_SINGLE_CANONICALIZATION_ALIAS_WORDS = frozenset({"demo", "example"})


@dataclass(slots=True)
class RecognitionBiasEngine:
    max_bias_phrases: int = 24
    max_prompt_chars: int = field(default_factory=lambda: _env_int("JUNO_V2_ASR_BIAS_PROMPT_CHARS", 768))
    # Filter out multi-sentence artifacts from bias phrase candidates.
    max_bias_phrase_chars: int = 80

    def build_plan(
        self,
        *,
        utterance_id: str,
        snapshot: MemorySnapshot,
        context: TypedContextBundle,
        base_prompt: str | None = None,
        memory_packet: MemoryServingPacket | None = None,
        mode_policy: ModePolicy | None = None,
        effective_mode: str | None = None,
        seed_attachment: typing.Any | None = None,
        recent_finals: typing.Sequence[str] | None = None,
    ) -> RecognitionBiasPlan:
        if memory_packet is not None:
            packet = memory_packet
        else:
            packet = rank_memory_for_context(
                snapshot,
                context=context,
                mode_policy=mode_policy,
                effective_mode=effective_mode,
            )
        phrases = self._candidate_phrases(packet, context)
        if seed_attachment is not None:
            seed_phrases = self._seed_asr_phrases(seed_attachment, context=context)
            phrases = self._merge_seed_bias_phrases(phrases, seed_phrases)
        capped_phrases = _diversify_bias_phrases(
            phrases,
            screen_terms=list(getattr(context, "candidate_entities", None) or []),
            cap=self.max_bias_phrases,
        )
        prompt_parts: list[str] = []
        base_s = base_prompt.strip() if base_prompt else ''
        selection_fragment = ''
        if context.selected_text:
            selection_fragment = 'Selection context: ' + context.selected_text[:96]
        # Local Whisper backends (mlx/faster_whisper) only consume initial_prompt, not bias_phrases.
        # Greedy-pack ranked phrases into the prefer-exact segment so decode-time hints match the list.
        outer_without_prefer = [x for x in (base_s, selection_fragment) if x]
        outer_joined = ' | '.join(outer_without_prefer) if outer_without_prefer else ''
        prefer_budget = self.max_prompt_chars - len(outer_joined) - (3 if outer_without_prefer else 0)
        prefer_budget = max(0, prefer_budget)
        prefer_line, prefer_embedded = self._build_prefer_exact_forms_line(capped_phrases, max_chars=prefer_budget)
        if base_s:
            prompt_parts.append(base_s)
        if prefer_line:
            prompt_parts.append(prefer_line)
        if selection_fragment:
            prompt_parts.append(selection_fragment)
        # Issue #28 — append recent finals (PII-redacted) so Whisper has
        # session-scoped recency bias. Take newest-first so truncation
        # drops the oldest finals.
        recent_fragment = self._build_recent_finals_fragment(recent_finals)
        if recent_fragment:
            prompt_parts.append(recent_fragment)
        prompt = ' | '.join([p for p in prompt_parts if p]).strip() or None
        if prompt and len(prompt) > self.max_prompt_chars:
            prompt = prompt[: self.max_prompt_chars - 3].rstrip() + '...'
        # Word-budget cap — Whisper's prompt window is ~224 tokens; 140
        # words leaves headroom for the decoded text. Truncate the
        # *trailing* recent-finals tail first, then any other tail.
        if prompt:
            prompt = _truncate_to_word_budget(prompt, max_words=_INITIAL_PROMPT_MAX_WORDS)
        profile = build_personalization_profile(packet, effective_mode=effective_mode)
        meta: dict = {
            'candidate_count': len(phrases),
            'bias_phrases_total': len(capped_phrases),
            'prefer_forms_phrases_embedded': prefer_embedded,
            'memory_serving_packet': packet.to_dict(),
            'personalization_profile': profile.to_dict(),
            'effective_mode': effective_mode,
            'mode_policy': mode_policy.to_dict() if mode_policy is not None else None,
        }
        if seed_attachment is not None:
            meta['juno_seed_bias'] = dict(seed_attachment.metadata)
            if seed_attachment.structured_bundle is not None:
                meta['juno_seed_structured_bias'] = seed_attachment.structured_bundle.to_dict()
            meta['seed_canonicalization'] = list(seed_attachment.canonicalization_tuples)
        return RecognitionBiasPlan(
            utterance_id=utterance_id,
            context=context,
            bias_phrases=capped_phrases,
            initial_prompt=prompt,
            metadata=meta,
        )

    def _build_recent_finals_fragment(
        self, recent_finals: typing.Sequence[str] | None
    ) -> str:
        """Compose the trailing ``Recent: ...`` fragment from the last
        finalized transcripts.

        PII guard: digit runs of >=6 chars are masked before the value
        enters the prompt. Empty / whitespace-only entries after masking
        are dropped. Returns an empty string when there's nothing to add.
        """
        if not recent_finals:
            return ''
        cleaned: list[str] = []
        for line in recent_finals:
            if not line:
                continue
            masked = _DIGIT_RUN_RE.sub('[redacted]', str(line)).strip()
            if masked:
                cleaned.append(masked)
        if not cleaned:
            return ''
        # Newest at the right (callers pass oldest-first); join with "; "
        # so reading flows naturally. Cap at the last 3 entries — older
        # context offers diminishing recency benefit.
        tail = cleaned[-3:]
        return 'Recent: ' + '; '.join(tail)

    def build_serving_packet(self, *, snapshot: MemorySnapshot) -> MemoryServingPacket:
        lexicon = sorted(
            (item for item in snapshot.lexicon if not _is_low_signal_lexicon_pair(item.term, item.canonical_form)),
            key=lambda item: (-float(item.boost), item.canonical_form.casefold()),
        )
        corrections = sorted(
            (
                item
                for item in snapshot.corrections
                if is_safe_correction_pair(item.observed, item.corrected)
            ),
            key=lambda item: (-int(item.count), item.corrected.casefold()),
        )
        entities = sorted(
            (item for item in snapshot.session_entities if self.is_session_entity_candidate(item.value)),
            key=lambda item: (-int(item.count), item.value.casefold()),
        )
        replacements = sorted(snapshot.replacements, key=lambda item: (-len(item.trigger), item.trigger.casefold()))
        served_lexicon = lexicon[:10]
        lexicon_aliases: dict[str, list[str]] = {}
        for item in served_lexicon:
            aliases: list[str] = []
            seen_aliases: set[str] = {item.canonical_form.casefold()}
            for value in [item.term, *item.aliases]:
                alias = (value or "").strip()
                key = alias.casefold()
                if not alias or key in seen_aliases:
                    continue
                seen_aliases.add(key)
                aliases.append(alias)
            if aliases:
                lexicon_aliases[item.canonical_form] = aliases[:8]

        return MemoryServingPacket(
            lexicon_terms=[item.canonical_form for item in served_lexicon],
            replacements=[{'trigger': item.trigger, 'replacement': item.replacement, 'scope': item.scope} for item in replacements[:6]],
            corrections=[{'observed': item.observed, 'corrected': item.corrected, 'count': item.count} for item in corrections[:6]],
            session_entities=[item.value for item in entities[:8]],
            metadata={
                'lexicon_total': len(snapshot.lexicon),
                'replacement_total': len(snapshot.replacements),
                'correction_total': len(snapshot.corrections),
                'correction_served_total': len(corrections),
                'session_entity_total': len(snapshot.session_entities),
                'lexicon_aliases': lexicon_aliases,
            },
        )

    def normalize_transcript(self, text: str, *, snapshot: MemorySnapshot, plan: RecognitionBiasPlan, scope: str) -> TranscriptNormalization:
        current = (text or '').strip()
        raw = current
        applied: list[NormalizationChange] = []
        current = self._apply_corrections(current, snapshot, applied)
        current = self._apply_replacements(current, snapshot, applied, app_name=plan.context.app_name)
        current = self._apply_canonicalization(current, snapshot, plan, applied)
        current = self._apply_seed_canonicalization(current, plan, applied)
        return TranscriptNormalization(
            raw_text=raw,
            normalized_text=current,
            applied=applied,
            metadata={'scope': scope, 'bias_phrase_count': len(plan.bias_phrases)},
        )

    def extract_session_entities(self, text: str) -> list[str]:
        # Reuse the screen extractor so committed transcripts harvest
        # the same shapes (Title-case, snake/kebab, lowercase letter+
        # digit handles) that drive the live screen-bias prompt. Before
        # unification this used a regex that only matched uppercase-
        # initial tokens, so a learned "alpha42" would never seed the
        # memory store from prior commits.
        from juno_v2.context.provider import _extract_candidates
        out: list[str] = []
        for candidate in _extract_candidates([text or '']):
            if not self.is_session_entity_candidate(candidate):
                continue
            out.append(candidate)
            if len(out) >= 16:
                break
        return out

    def is_session_entity_candidate(self, value: str) -> bool:
        return _is_session_entity_candidate(value)

    def _build_prefer_exact_forms_line(self, phrases: list[str], *, max_chars: int) -> tuple[str, int]:
        """Return ``(\"Prefer exact forms: a, b, ...\", n)`` using the first *n* phrases that fit in *max_chars*."""
        prefix = 'Prefer exact forms: '
        if max_chars <= len(prefix) or not phrases:
            return ('', 0)
        out: list[str] = []
        used = len(prefix)
        for p in phrases:
            piece = p.strip()
            if not piece:
                continue
            add_len = len(piece) if not out else len(piece) + 2
            if used + add_len > max_chars:
                break
            out.append(piece)
            used += add_len
        if not out:
            return ('', 0)
        return (prefix + ', '.join(out), len(out))

    def _merge_seed_bias_phrases(self, phrases: list[str], seed_phrases: typing.Sequence[str]) -> list[str]:
        """Append context-relevant seed phrases without displacing live/user hints."""
        seen: set[str] = set()
        out: list[str] = []

        def extend(values: typing.Sequence[str]) -> None:
            for value in values:
                v = (value or '').strip()
                if not v or len(v) > self.max_bias_phrase_chars:
                    continue
                if _looks_like_hallucination(v):
                    continue
                k = v.casefold()
                if k in seen:
                    continue
                seen.add(k)
                out.append(v)

        extend(phrases)
        extend(seed_phrases)
        return out

    def _seed_asr_phrases(self, seed_attachment: typing.Any, *, context: TypedContextBundle) -> list[str]:
        structured = getattr(seed_attachment, "structured_bundle", None)
        if structured is None:
            return [
                str(phrase)
                for phrase in getattr(seed_attachment, "extra_bias_phrases", ()) or ()
                if self._seed_phrase_visible_in_context(str(phrase), context)
            ]
        out: list[str] = []
        for term in getattr(structured, "terms", ()) or ():
            pack_name = str(getattr(term, "pack_name", "") or "")
            tags = {str(tag) for tag in getattr(term, "tags", ()) or ()}
            canonical = str(getattr(term, "canonical", "") or "")
            bias_strings = [str(value) for value in getattr(term, "bias_strings", ()) or ()]
            if (
                pack_name in {"memory_lexicon", "memory_correction", "runtime_context", "personal_entities"}
                or "runtime_only" in tags
                or self._seed_phrase_visible_in_context(canonical, context)
                or any(self._seed_phrase_visible_in_context(value, context) for value in bias_strings)
            ):
                out.extend(bias_strings or [canonical])
        return out

    def _seed_phrase_visible_in_context(self, phrase: str, context: TypedContextBundle) -> bool:
        value = (phrase or "").strip()
        if not value:
            return False
        folded = value.casefold()
        evidence: list[str] = []
        for item in (
            context.app_name,
            context.window_title,
            context.symbol_under_cursor,
            context.focused_file_path,
            context.selected_text,
            context.field_text_excerpt,
        ):
            if item:
                evidence.append(str(item))
        evidence.extend(str(item) for item in context.candidate_entities[:48] if str(item).strip())
        haystack = "\n".join(evidence).casefold()
        return bool(haystack and folded in haystack)

    def _candidate_phrases(self, packet: MemoryServingPacket, context: TypedContextBundle) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        def add(value: str) -> None:
            v = (value or '').strip()
            if not v or len(v) > self.max_bias_phrase_chars:
                return
            if _looks_like_hallucination(v):
                return
            k = v.casefold()
            if k in seen:
                return
            seen.add(k)
            out.append(v)
        if context.selected_text:
            add(context.selected_text)
        if context.symbol_under_cursor:
            add(context.symbol_under_cursor)
        if context.focused_file_path:
            # Strip directory + extension so the bias phrase is
            # speakable. Keep the raw filename too (some users dictate
            # "open dot py" to mean "``__main__.py``" or similar).
            base = os.path.basename(context.focused_file_path)
            stem = os.path.splitext(base)[0]
            add(stem)
            add(base)
        for candidate in context.candidate_entities:
            if self.is_session_entity_candidate(candidate):
                add(candidate)
        for term in packet.lexicon_terms:
            if _is_low_signal_phrase(term):
                continue
            add(term)
        for pair in packet.corrections:
            add(str(pair.get('corrected', '')))
            add(str(pair.get('observed', '')))
        for rule in packet.replacements:
            add(str(rule.get('replacement', '')))
            add(str(rule.get('trigger', '')))
        for entity in packet.session_entities:
            if self.is_session_entity_candidate(entity):
                add(entity)
        return out

    def _apply_corrections(self, text: str, snapshot: MemorySnapshot, applied: list[NormalizationChange]) -> str:
        out = text
        for pair in sorted(snapshot.corrections, key=lambda item: len(item.observed), reverse=True):
            if not is_safe_correction_pair(pair.observed, pair.corrected):
                continue
            out = _replace_phrase(out, pair.observed, pair.corrected, applied, kind='correction', source='memory_correction')
        return out

    def _apply_replacements(self, text: str, snapshot: MemorySnapshot, applied: list[NormalizationChange], *, app_name: str | None) -> str:
        out = text
        app_scope = (app_name or '').strip().casefold()
        for rule in sorted(snapshot.replacements, key=lambda item: len(item.trigger), reverse=True):
            scope = (rule.scope or 'global').casefold()
            if scope not in {'global', '', app_scope} and not scope.startswith('app:'):
                continue
            if scope.startswith('app:') and scope[4:] != app_scope:
                continue
            out = _replace_phrase(out, rule.trigger, rule.replacement, applied, kind='replacement', source=scope or 'global', case_sensitive=rule.case_sensitive)
        return out

    def _apply_canonicalization(self, text: str, snapshot: MemorySnapshot, plan: RecognitionBiasPlan, applied: list[NormalizationChange]) -> str:
        out = text
        canonical_map: list[tuple[str, str, str]] = []
        for entry in snapshot.lexicon:
            if _is_low_signal_lexicon_pair(entry.term, entry.canonical_form):
                continue
            if self._lexicon_canonicalization_allowed(entry.canonical_form, entry.canonical_form, plan.context):
                canonical_map.append((entry.canonical_form, entry.canonical_form, 'lexicon_canonical'))
            if self._lexicon_canonicalization_allowed(entry.term, entry.canonical_form, plan.context):
                canonical_map.append((entry.term, entry.canonical_form, 'lexicon_term'))
            for alias in entry.aliases:
                if self._lexicon_canonicalization_allowed(alias, entry.canonical_form, plan.context):
                    canonical_map.append((alias, entry.canonical_form, 'lexicon_alias'))
        # Most context candidates are ASR bias hints, not deterministic rewrite
        # rules. Treating every screen/live candidate as a canonical form
        # capitalized ordinary dictation words ("go" -> "Go", "missing" ->
        # "Missing") whenever the live transcript had sentence-start casing.
        # Only terms that pass the low-signal entity filter are safe to
        # canonicalize here.
        for candidate in plan.context.candidate_entities:
            if _context_candidate_can_canonicalize(candidate) and self.is_session_entity_candidate(candidate):
                canonical_map.append((candidate, candidate, 'context_candidate'))
        if plan.context.selected_text:
            canonical_map.append((plan.context.selected_text, plan.context.selected_text, 'selected_text'))
        canonical_map.sort(key=lambda item: len(item[0]), reverse=True)
        seen_pairs: set[tuple[str, str]] = set()
        for trigger, replacement, source in canonical_map:
            key = (trigger.casefold(), replacement.casefold())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            out = _replace_phrase(out, trigger, replacement, applied, kind='canonicalization', source=source)
        return out

    def _lexicon_canonicalization_allowed(
        self,
        trigger: str,
        replacement: str,
        context: TypedContextBundle,
    ) -> bool:
        t = (trigger or "").strip()
        r = (replacement or "").strip()
        if not t or not r:
            return False
        trigger_tokens = re.findall(r"[A-Za-z0-9]+", t)
        replacement_tokens = re.findall(r"[A-Za-z0-9]+", r)
        if not trigger_tokens or not replacement_tokens:
            return False
        if len(trigger_tokens) == 1 and trigger_tokens[0].casefold() in (
            _LOW_SIGNAL_SESSION_ENTITY_WORDS | _GENERIC_SINGLE_CANONICALIZATION_ALIAS_WORDS
        ):
            return self._seed_phrase_visible_in_context(r, context)
        if len(trigger_tokens) >= 2 or len(replacement_tokens) <= 1:
            return True
        if any(ch.isdigit() for ch in t) or any(ch in t for ch in {"_", "-", ".", "/", "#"}):
            return True
        if re.search(r"[a-z][A-Z]|\b[A-Z]{2,}\b", t):
            return True
        return self._seed_phrase_visible_in_context(r, context)

    def _apply_seed_canonicalization(self, text: str, plan: RecognitionBiasPlan, applied: list[NormalizationChange]) -> str:
        rows = plan.metadata.get('seed_canonicalization') if isinstance(plan.metadata, dict) else None
        if not rows:
            return text
        out = text
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                continue
            trigger, replacement, source = str(row[0]), str(row[1]), str(row[2])
            if not self._seed_canonicalization_allowed(trigger, replacement, context=plan.context):
                continue
            out = _replace_phrase(out, trigger, replacement, applied, kind='seed_canonicalization', source=str(source))
        return out

    def _seed_canonicalization_allowed(
        self,
        variant: str,
        canonical: str,
        *,
        context: TypedContextBundle | None = None,
    ) -> bool:
        v = (variant or '').strip()
        c = (canonical or '').strip()
        if len(v) < 3 or len(c) < 2:
            return False
        if len(v) > 48:
            return False
        if v.casefold() == c.casefold():
            return False
        tokens = v.replace("'", '').split()
        if len(tokens) == 1 and tokens[0].casefold() in _SEED_CANON_STOPWORDS:
            return False
        canonical_tokens = re.findall(r"[A-Za-z0-9]+", c)
        if len(tokens) == 1 and len(canonical_tokens) > 1:
            token = tokens[0]
            if token.isalpha() and token.casefold() == token:
                return context is not None and self._seed_phrase_visible_in_context(c, context)
        return True


def _replace_phrase(
    text: str,
    trigger: str,
    replacement: str,
    applied: list[NormalizationChange],
    *,
    kind: str,
    source: str,
    case_sensitive: bool = False,
) -> str:
    trigger = (trigger or '').strip()
    replacement = (replacement or '').strip()
    if not trigger or not replacement:
        return text
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = _phrase_pattern(trigger)

    def repl(match: re.Match[str]) -> str:
        before = match.group(0)
        if before == replacement:
            return before
        applied.append(NormalizationChange(kind=kind, source=source, before=before, after=replacement))
        return replacement

    return re.sub(pattern, repl, text, flags=flags)


def _truncate_to_word_budget(text: str, *, max_words: int) -> str:
    """Trim *text* to at most *max_words* whitespace-delimited tokens.

    Used by :meth:`RecognitionBiasEngine.build_plan` to enforce the
    Whisper prompt word budget after assembly.
    """
    if not text:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def _is_session_entity_candidate(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    if not learned_term_allowed(v):
        return False
    if _looks_like_hallucination(v):
        return False
    return session_entity_allowed(v)


def is_biasable_runtime_context_candidate(value: str) -> bool:
    return _is_session_entity_candidate(value) and _context_candidate_can_canonicalize(value)


def is_low_signal_lexicon_pair(term: str, canonical: str) -> bool:
    return _is_low_signal_lexicon_pair(term, canonical)


def _is_low_signal_phrase(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return True
    alpha_tokens = re.findall(r"[A-Za-z]+", v)
    if len(alpha_tokens) != 1:
        return False
    return alpha_tokens[0].casefold() in _LOW_SIGNAL_SESSION_ENTITY_WORDS


def _is_low_signal_lexicon_pair(term: str, canonical: str) -> bool:
    t = (term or "").strip()
    c = (canonical or "").strip()
    if not t or not c:
        return True
    return (
        _is_low_signal_phrase(t)
        and _is_low_signal_phrase(c)
        and t.casefold() == c.casefold()
    )


def _context_candidate_can_canonicalize(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    alpha_tokens = re.findall(r"[A-Za-z]+", v)
    if not alpha_tokens:
        return False
    if len(alpha_tokens) >= 2:
        return True
    token = alpha_tokens[0]
    if token.isupper() and len(token) >= 2:
        return True
    if any(ch.isdigit() for ch in v):
        return True
    if any(ch in v for ch in {"_", "-", ".", "/", "#"}):
        return True
    # Plain Titlecase is allowed only after the caller's low-signal filter has
    # rejected ordinary sentence words like Go/Missing/Every. This preserves
    # real screen/title entities such as Project Alpha and names such as Avery.
    return token[:1].isupper()


def _phrase_pattern(phrase: str) -> str:
    """Build a regex that matches ``phrase`` in text, tolerant of
    whitespace/punctuation variants between alphanumeric chunks.

    Used by every memory normalisation pass (corrections, replacements,
    lexicon canonicalisation, seed canonicalisation) to apply stored
    triggers against ASR output. The flexibility is intentional: ASR
    routinely segments compounds differently than the user typed them
    (``"signoff"`` vs ``"sign off"``, ``"Q.B.R"`` vs ``"Q B R"``).
    """
    pattern = fold_match_pattern(phrase)
    if pattern is not None:
        return pattern
    # Trigger had no alphanumeric Latin content to anchor on — fall back
    # to the legacy escape, which preserves exact-match semantics for
    # unusual triggers (all-punctuation or non-Latin scripts).
    escaped = re.escape(phrase)
    if phrase[:1].isalnum() and phrase[-1:].isalnum():
        return rf'(?<!\w){escaped}(?!\w)'
    return escaped


# Single screen tokens must look like a name the user could plausibly say:
# a capitalized word or an acronym, letters only. Harvested screen text is
# full of OCR junk ("lTr", "Mwrk5pace"), UI chrome with digits ("Adi41"),
# and truncations — none of which belong in a Whisper prompt.
_SCREEN_TERM_TOKEN_RE = re.compile(r"^(?:[A-Z][a-zA-Z]{1,23}|[A-Z]{2,8})$")
_SCREEN_TERM_MAX_WORDS = 3


def _screen_term_prompt_worthy(text: str) -> bool:
    """Gate for screen-harvested terms entering the Whisper initial prompt.

    Production 2026-06-11: with WhatsApp Web on screen the prompt filled
    with "High, Back, Forward, Reload, Show, Update, Saved, lTr,
    Mwrk5pace, Adi41, …" — common UI words bias recognition away from
    the user's actual speech, junk tokens degrade decoding on quiet
    audio (replay: the same prompt turned a faint utterance into a
    "team team team…" repetition loop), and chat-contact names leak
    into a model prompt. Memory-lane phrases are policy-gated at learn
    time; screen terms get the equivalent gate here at serving time.
    """
    words = (text or "").split()
    if not words or len(words) > _SCREEN_TERM_MAX_WORDS:
        return False
    from juno_v2.memory.entity_policy import common_english_single_word

    if len(words) == 1:
        token = words[0]
        if not _SCREEN_TERM_TOKEN_RE.match(token):
            return False
        return not common_english_single_word(token)
    # Multi-word: every token must be word-shaped, and at least one must be
    # a non-common word — "Privacy Settings" is UI chrome, "Cassini
    # Research" is a name worth biasing toward.
    if not all(re.match(r"^[A-Za-z][A-Za-z'\-]{0,23}$", w) for w in words):
        return False
    return any(not common_english_single_word(w) for w in words)


def _diversify_bias_phrases(
    phrases: list[str],
    *,
    screen_terms: list,
    cap: int,
) -> list[str]:
    """Per-utterance bias selection: screen terms first, then memory terms
    with family de-duplication.

    A flat first-N cap let one pack family flood the prompt (ten "Claude *"
    variants) so distinct terms never biased recognition. Family key = first
    token; max 3 per family. Screen terms are what the user is looking at —
    they outrank everything, but only after the prompt-worthiness gate
    (see :func:`_screen_term_prompt_worthy`).
    """
    out: list[str] = []
    seen: set[str] = set()
    family_counts: dict[str, int] = {}

    def _take(term: str, *, family_limit: int) -> None:
        text = str(term).strip() if not isinstance(term, dict) else str(term.get("text") or "").strip()
        if not text or len(out) >= cap:
            return
        key = text.casefold()
        if key in seen:
            return
        family = key.split()[0]
        if family_counts.get(family, 0) >= family_limit:
            return
        seen.add(key)
        family_counts[family] = family_counts.get(family, 0) + 1
        out.append(text)

    for term in screen_terms[:16]:
        text = str(term).strip() if not isinstance(term, dict) else str(term.get("text") or "").strip()
        if not _screen_term_prompt_worthy(text):
            continue
        _take(text, family_limit=4)
    for phrase in phrases:
        _take(phrase, family_limit=3)
    return out
