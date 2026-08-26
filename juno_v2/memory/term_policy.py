from __future__ import annotations

import unicodedata


MIN_LEARNED_TERM_CHARS = 3

# Issue #79 — a learned *term* is a name/identifier, not a sentence. Anything
# past this many whitespace tokens is prose that must not enter the lexicon
# (and therefore the Whisper ``initial_prompt`` "Prefer exact forms" list).
MAX_LEARNED_TERM_TOKENS = 6

# Sentence-final punctuation. ``.`` is included but guarded by
# ``_token_is_abbreviation_like`` so terms such as "Node.js", "v1.2", "e.g."
# and "Acme Corp." keep the punctuation that belongs to them.
_SENTENCE_TERMINATORS = ".?!…"


def meaningful_char_count(value: str | None) -> int:
    """Count alphanumeric characters in a user-learned memory term.

    The policy intentionally counts Unicode letters/digits instead of the
    latin-only fold key so legitimate non-English names are not rejected.
    """

    if not value:
        return 0
    normalized = unicodedata.normalize("NFKC", str(value))
    return sum(1 for ch in normalized if ch.isalnum())


def learned_term_allowed(value: str | None, *, min_chars: int = MIN_LEARNED_TERM_CHARS) -> bool:
    return meaningful_char_count(value) >= max(1, int(min_chars))


def _token_is_abbreviation_like(token: str) -> bool:
    """True when a token's trailing ``.`` belongs to the token itself.

    Covers the two shapes that matter in practice:

    * internal dots — "e.g.", "U.S.", "Node.js.", "v1.2."
    * short capitalised abbreviations — "Inc.", "Corp.", "Co.", "Jr."
    """

    core = token.rstrip(_SENTENCE_TERMINATORS)
    if not core:
        return True
    if "." in core:
        return True
    return len(core) <= 4 and core[:1].isupper()


def strip_terminal_sentence_punctuation(value: str | None) -> str:
    """Drop sentence-final ``.?!…`` from a bias phrase.

    Whisper mimics the punctuation style of its ``initial_prompt``; joining
    phrases with ``", "`` without stripping produced ``.,`` / ``?,`` runs that
    bled into raw ASR output (issue #79).

    Rule: only a *multi-token* phrase can carry sentence punctuation, and only
    when its last token is not abbreviation-like. Single-token phrases are
    returned untouched, so terms whose punctuation is part of the term
    ("Node.js", "v1.2", "C++", "e.g.") are never mangled.
    """

    text = (value or "").rstrip()
    if not text:
        return ""
    tokens = text.split()
    if len(tokens) < 2:
        return text
    if _token_is_abbreviation_like(tokens[-1]):
        return text
    stripped = text.rstrip(_SENTENCE_TERMINATORS).rstrip()
    return stripped or text


def learned_term_is_sentence_like(
    value: str | None, *, max_tokens: int = MAX_LEARNED_TERM_TOKENS
) -> bool:
    """True when a promotion candidate reads as a sentence rather than a term.

    Sentence-shaped candidates are rejected before they reach the lexicon, so
    the serving path never has to render prose as a "Prefer exact forms" entry.
    """

    text = (value or "").strip()
    if not text:
        return False
    tokens = text.split()
    if len(tokens) > max(1, int(max_tokens)):
        return True
    if strip_terminal_sentence_punctuation(text) != text:
        return True
    return any(
        token.endswith(tuple(_SENTENCE_TERMINATORS)) and not _token_is_abbreviation_like(token)
        for token in tokens[:-1]
    )


__all__ = [
    "MAX_LEARNED_TERM_TOKENS",
    "MIN_LEARNED_TERM_CHARS",
    "learned_term_allowed",
    "learned_term_is_sentence_like",
    "meaningful_char_count",
    "strip_terminal_sentence_punctuation",
]
