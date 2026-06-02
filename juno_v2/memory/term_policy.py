from __future__ import annotations

import unicodedata


MIN_LEARNED_TERM_CHARS = 3


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


__all__ = ["MIN_LEARNED_TERM_CHARS", "learned_term_allowed", "meaningful_char_count"]
