from __future__ import annotations

import re

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemoryServingPacket, MemorySnapshot, ReplacementRule
from juno_v2.contracts.modes import ModePolicy
from juno_v2.memory.entity_policy import session_entity_allowed

_LOW_SIGNAL_SESSION_ENTITY_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "ai",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "context",
        "customer",
        "deadline",
        "did",
        "document",
        "do",
        "does",
        "edited",
        "focus",
        "font",
        "for",
        "from",
        "he",
        "here",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "owner",
        "our",
        "regular",
        "she",
        "status",
        "style",
        "task",
        "title",
        "so",
        "that",
        "the",
        "then",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "while",
        "who",
        "why",
        "with",
        "yes",
        "you",
    }
)


def rank_memory_for_context(
    snapshot: MemorySnapshot,
    *,
    context: TypedContextBundle,
    mode_policy: ModePolicy | None = None,
    effective_mode: str | None = None,
    transcript_hint: str | None = None,
    session_terms: list[str] | None = None,
) -> MemoryServingPacket:
    """Build a bounded, ranked serving packet (no raw store dumps)."""
    app_scope = (context.app_name or "").strip().casefold()
    mode_key = (effective_mode or (mode_policy.mode_name if mode_policy else "") or "").strip().casefold()
    hint_tokens = _token_set(
        " ".join(
            part
            for part in [
                transcript_hint or "",
                context.selected_text or "",
                context.focused_text_before or "",
                context.focused_text_after or "",
                context.window_title or "",
                " ".join(session_terms or []),
            ]
            if part
        )
    )

    def _rep_score(rule: ReplacementRule) -> tuple[float, int]:
        sc = (rule.scope or "global").casefold()
        bonus = 0.0
        if sc in {"global", ""}:
            bonus += 1.0
        if sc == app_scope and app_scope:
            bonus += 8.0
        if sc.startswith("app:") and app_scope and sc[4:] == app_scope:
            bonus += 10.0
        if mode_key and mode_key in sc:
            bonus += 6.0
        if hint_tokens and (_token_set(rule.trigger) & hint_tokens):
            bonus += 12.0
        return (-bonus, -len(rule.trigger))

    def _term_score(value: str, base: float = 0.0) -> tuple[float, str]:
        bonus = float(base)
        toks = _token_set(value)
        if hint_tokens and toks & hint_tokens:
            bonus += 20.0
        return (-bonus, value.casefold())

    lexicon = sorted(
        (item for item in snapshot.lexicon if not _low_signal_lexicon_pair(item.term, item.canonical_form)),
        key=lambda item: _term_score(item.canonical_form, float(item.boost)),
    )
    def _edit_distance_le1(a: str, b: str) -> bool:
        if abs(len(a) - len(b)) > 1:
            return False
        if a == b:
            return True
        if len(a) == len(b):
            return sum(1 for x, y in zip(a, b) if x != y) == 1
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        for i in range(len(longer)):
            if shorter == longer[:i] + longer[i + 1:]:
                return True
        return False

    def _correction_admissible(item) -> bool:
        # Same rule as replacements: a correction pair serves only when its
        # observed or corrected form is plausibly present in this turn.
        tokens = _token_set(f"{item.observed} {item.corrected}")
        if not tokens or not hint_tokens:
            return False
        if tokens & hint_tokens:
            return True
        return any(
            len(t) >= 4 and any(_edit_distance_le1(t, h) for h in hint_tokens)
            for t in tokens
        )

    corrections = sorted(
        (item for item in snapshot.corrections if _correction_admissible(item)),
        key=lambda item: _term_score(f"{item.observed} {item.corrected}", float(item.count)),
    )
    entities = sorted(
        (item for item in snapshot.session_entities if _session_entity_allowed(item.value)),
        key=lambda item: _term_score(item.value, float(item.count)),
    )
    def _replacement_admissible(rule: ReplacementRule) -> bool:
        """Serve a replacement only when its trigger is plausibly present.

        Unconditional serving let a seeded "launch code" rule inject
        LAUNCH-CODE-991 into a selected-text rewrite that never mentioned it
        (production 2026-06-11). Triggers must appear in the spoken hint /
        selection / focused text / window title / session terms, or be a
        single-edit near miss of a hint token.
        """
        trigger_tokens = _token_set(rule.trigger)
        if not trigger_tokens or not hint_tokens:
            return False
        if trigger_tokens & hint_tokens:
            return True
        for trig in trigger_tokens:
            if len(trig) >= 4 and any(_edit_distance_le1(trig, hint) for hint in hint_tokens):
                return True
        return False

    replacements = sorted(
        (rule for rule in snapshot.replacements if _replacement_admissible(rule)),
        key=lambda r: _rep_score(r),
    )
    snippets = sorted(
        (
            item
            for item in list(getattr(snapshot, "snippets", []) or [])
            if isinstance(item, dict) and _snippet_allowed(item, app_scope=app_scope, hint_tokens=hint_tokens)
        ),
        key=lambda item: _snippet_score(item, app_scope=app_scope, hint_tokens=hint_tokens),
    )

    served_lexicon = lexicon[:12]
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
        replacements=[
            {'trigger': item.trigger, 'replacement': item.replacement, 'scope': item.scope}
            for item in replacements[:8]
        ],
        corrections=[
            {'observed': item.observed, 'corrected': item.corrected, 'count': item.count}
            for item in corrections[:8]
        ],
        session_entities=[item.value for item in entities[:10]],
        snippets=[
            {
                'trigger': str(item.get('trigger') or '').strip(),
                'scope': str(item.get('scope') or 'global').strip() or 'global',
                'body_preview': str(item.get('body') or '')[:500],
                'body_chars': len(str(item.get('body') or '')),
                'case_sensitive': bool(item.get('case_sensitive', False)),
            }
            for item in snippets[:8]
        ],
        metadata={
            'lexicon_total': len(snapshot.lexicon),
            'replacement_total': len(snapshot.replacements),
            'correction_total': len(snapshot.corrections),
            'session_entity_total': len(snapshot.session_entities),
            'snippet_total': len(list(getattr(snapshot, "snippets", []) or [])),
            'ranking': {
                'app_scope': app_scope or None,
                'mode': mode_key or None,
                'hint_token_count': len(hint_tokens),
                'session_term_count': len(session_terms or []),
            },
            'lexicon_aliases': lexicon_aliases,
        },
    )


def _session_entity_allowed(value: str) -> bool:
    return session_entity_allowed(value)


def _snippet_allowed(item: dict, *, app_scope: str, hint_tokens: set[str]) -> bool:
    trigger = str(item.get("trigger") or "").strip()
    body = str(item.get("body") or "")
    if not trigger or not body:
        return False
    scope = str(item.get("scope") or "global").strip().casefold() or "global"
    if scope in {"global", app_scope, ""}:
        return True
    if scope.startswith("app:") and app_scope and scope[4:] == app_scope:
        return True
    return bool(_token_set(trigger) & hint_tokens)


def _snippet_score(item: dict, *, app_scope: str, hint_tokens: set[str]) -> tuple[float, int, str]:
    scope = str(item.get("scope") or "global").strip().casefold() or "global"
    trigger = str(item.get("trigger") or "").strip()
    bonus = 0.0
    if scope == "global":
        bonus += 1.0
    if scope == app_scope and app_scope:
        bonus += 8.0
    if scope.startswith("app:") and app_scope and scope[4:] == app_scope:
        bonus += 10.0
    if _token_set(trigger) & hint_tokens:
        bonus += 14.0
    return (-bonus, -len(trigger), trigger.casefold())


def _low_signal_phrase(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return True
    alpha_tokens = re.findall(r"[A-Za-z]+", v)
    if len(alpha_tokens) != 1:
        return False
    return alpha_tokens[0].casefold() in _LOW_SIGNAL_SESSION_ENTITY_WORDS


def _low_signal_lexicon_pair(term: str, canonical: str) -> bool:
    t = (term or "").strip()
    c = (canonical or "").strip()
    if not t or not c:
        return True
    return _low_signal_phrase(t) and _low_signal_phrase(c) and t.casefold() == c.casefold()


def _token_set(text: str | None) -> set[str]:
    out: set[str] = set()
    for raw in (text or "").replace("_", " ").replace("-", " ").split():
        token = raw.strip(".,!?;:()[]{}<>\"'`").casefold()
        if len(token) >= 2:
            out.add(token)
    return out
