from __future__ import annotations

import unicodedata


MIN_LEARNED_TERM_CHARS = 3

# Issue #79 — a learned *term* is a name/identifier, not a sentence. Anything
# past this many whitespace tokens is prose that must not enter the lexicon
# (and therefore the Whisper ``initial_prompt`` "Prefer exact forms" list).
MAX_LEARNED_TERM_TOKENS = 6

# Sentence-final punctuation. ``?``, ``!`` and ``…`` never end an abbreviation,
# so they are stripped from any multi-token phrase. A trailing ``.`` is guarded
# by ``_token_is_abbreviation_like`` so "Node.js", "v1.2", "e.g." and
# "Acme Corp." keep the punctuation that belongs to them.
_SENTENCE_TERMINATORS = ".?!…"
_HARD_TERMINATORS = "?!…"

# Explicit abbreviation list for the trailing-``.`` guard. A length/case
# heuristic ("<=4 chars, uppercase initial") was tried first and was wrong: it
# also matched ordinary sentence-final words and names, so "Please review the
# PR.", "I asked Bob." and "Call the CEO." kept their periods. Only a closed
# set of abbreviations may keep a trailing dot; everything else is prose.
# Multi-dot forms ("e.g.", "i.e.", "U.S.") are covered by the internal-dot rule.
_ABBREVIATIONS = frozenset(
    {
        "inc",
        "corp",
        "co",
        "ltd",
        "llc",
        "jr",
        "sr",
        "st",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "vs",
        "etc",
        "no",
        "dept",
    }
)


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
    * listed abbreviations — "Inc.", "Corp.", "Co.", "Jr.", "St.", "etc."

    Only consulted for a trailing ``.``; ``?``/``!``/``…`` never end an
    abbreviation.
    """

    core = token.rstrip(_SENTENCE_TERMINATORS)
    if not core:
        return True
    if "." in core:
        return True
    return core.casefold() in _ABBREVIATIONS


def _token_ends_sentence(token: str) -> bool:
    """True when *token*'s trailing punctuation is sentence-final, not part of it."""

    if not token or token[-1] not in _SENTENCE_TERMINATORS:
        return False
    if token[-1] in _HARD_TERMINATORS:
        # A term may legitimately end in "!" ("Yahoo!"), but only as a
        # single-token phrase — callers gate on token count before asking.
        # Inside a multi-token phrase, "Yahoo! Mail" is therefore treated as
        # sentence-shaped; a rare, deliberate false reject that keeps the rule
        # from letting "Where is Tom?" through.
        return True
    return not _token_is_abbreviation_like(token)


def strip_terminal_sentence_punctuation(value: str | None) -> str:
    """Drop sentence-final ``.?!…`` from a bias phrase.

    Whisper mimics the punctuation style of its ``initial_prompt``; joining
    phrases with ``", "`` without stripping produced ``.,`` / ``?,`` runs that
    bled into raw ASR output (issue #79).

    Rule: only a *multi-token* phrase can carry sentence punctuation. Trailing
    ``?``/``!``/``…`` are always stripped from such a phrase; a trailing ``.``
    is stripped unless the final token is abbreviation-like. Single-token
    phrases are returned untouched, so terms whose punctuation is part of the
    term ("Node.js", "v1.2", "C++", "e.g.", "Yahoo!") are never mangled.
    """

    text = (value or "").rstrip()
    if not text:
        return ""
    if len(text.split()) < 2:
        return text
    stripped = text
    while stripped:
        tokens = stripped.split()
        if not tokens or not _token_ends_sentence(tokens[-1]):
            break
        stripped = stripped[:-1].rstrip()
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
    return any(_token_ends_sentence(token) for token in tokens[:-1])


__all__ = [
    "MAX_LEARNED_TERM_TOKENS",
    "MIN_LEARNED_TERM_CHARS",
    "learned_term_allowed",
    "learned_term_is_sentence_like",
    "meaningful_char_count",
    "strip_terminal_sentence_punctuation",
]
