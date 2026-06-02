"""Match-normalization for the memory layer.

Every memory comparison site — snippet expansion, replacement application,
vocabulary canonicalization, correction re-application, store dedup, server
conflict detection — compares a stored trigger against ASR output. The
shapes don't agree: the user types ``"signoff"`` but Whisper segments it
as ``"sign off"``; a vocabulary canonical written ``"Q.B.R"`` shows up in
the transcript as ``"Q B R"``; a replacement trigger entered as
``"my-email"`` is heard as ``"my email"``. Without normalization every
one of these is a silent miss.

The fold here is one structural character-class rule — not a list of
tuned cases:

    fold_key(s) = NFKD-decompose(s)
                → drop combining marks (diacritics, accents)
                → casefold
                → drop everything outside [a-z0-9]

Variants that collapse to the same key:

* whitespace runs (regular space, NBSP, em space, tab, newline)
* punctuation (hyphen, em-dash, slash, period, comma, paren, bracket, quote)
* case (upper, lower, title, camel)
* Unicode normalisation forms (NFC/NFD, fullwidth/halfwidth)
* combining diacritics (``café`` ↔ ``cafe``, ``naïve`` ↔ ``naive``)

We deliberately keep alphanumeric latin and digits. Non-latin scripts pass
through casefold and survive the [a-z0-9] filter as empty — which is the
correct behaviour: a CJK trigger should match itself byte-for-byte
through ``fold_match_pattern``'s fallback path rather than be flattened
to nothing.

``fold_match_pattern`` builds a regex that finds a trigger inside text
allowing the same whitespace/punctuation slack between alphanumeric
chunks. ``sign off`` matches ``signoff``, ``Sign-Off``, ``sign  off``, but
not ``signoffer`` (the ``[A-Za-z0-9]`` negative lookarounds preserve word
boundaries on the alphanumeric side).
"""
from __future__ import annotations

import re
import unicodedata

_NON_ALNUM_LATIN = re.compile(r"[^a-z0-9]+")
_ALNUM_LATIN_CHUNK = re.compile(r"[A-Za-z0-9]+")


def fold_key(s: str | None) -> str:
    """Collapse ``s`` to its memory-match canonical key.

    Returns the empty string when the input has no alphanumeric Latin
    content (e.g. all-punctuation, all-CJK, ``None``, or whitespace).
    Callers must treat ``""`` as "do not use this key for matching" —
    otherwise an empty key collides with every other empty key.
    """
    if not s:
        return ""
    decomposed = unicodedata.normalize("NFKD", s)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM_LATIN.sub("", stripped.casefold())


def fold_match_pattern(
    trigger: str | None,
    *,
    case_sensitive: bool = False,
) -> str | None:
    """Build a regex pattern that finds ``trigger`` inside text, tolerating
    whitespace/punctuation variants — including between every pair of
    alphanumeric characters of the trigger.

    The pattern is derived from :func:`fold_key`, not the raw trigger:
    we strip everything outside ``[a-z0-9]`` from the trigger and insert
    ``[\\W_]*`` between every surviving character. That way a stored
    ``"QBR"`` matches text ``"Q.B.R"``, ``"Q B R"``, or ``"qbr"``; a
    stored ``"sign off"`` matches ``"signoff"``, ``"sign-off"``, etc.

    Word boundaries are anchored on the alphanumeric side via
    ``(?<![A-Za-z0-9])`` / ``(?![A-Za-z0-9])`` rather than ``\\b``, so a
    short trigger like ``"qbr"`` does not fire inside ``"acquire"`` —
    the surrounding letters fail the lookaround. Internal gaps are
    matched by ``[\\W_]*`` which absorbs any Unicode non-word run.

    Returns ``None`` when ``trigger`` has no alphanumeric Latin content
    (all-punctuation, empty, or pure non-Latin). For pure non-Latin
    triggers we fall back to an escaped exact match with ``\\b`` so
    callers get something usable; for empty/punctuation triggers we
    refuse — there's nothing meaningful to anchor on and matching an
    empty key against every empty span would explode.
    """
    if not trigger:
        return None
    key = fold_key(trigger)
    if not key:
        # Empty fold-key splits two cases:
        #   * trigger is all-punctuation/whitespace — refuse, there's
        #     nothing alphanumeric to anchor on and matching every
        #     empty span would explode.
        #   * trigger contains Unicode letters/digits outside the
        #     [a-z0-9] window (CJK, Cyrillic, etc.) — escape literally
        #     so callers still have a working exact-match pattern.
        stripped = trigger.strip()
        if not stripped or not any(c.isalnum() for c in stripped):
            return None
        escaped = re.escape(stripped)
        return rf"(?<!\w){escaped}(?!\w)"
    # Per-character `[\W_]*` so the pattern matches any whitespace/
    # punctuation between fold-equivalent letters. Safe in practice
    # because `[\W_]*` matches only non-word runs (letters and digits
    # are kept), and the lookarounds anchor word boundaries on the
    # alphanumeric side. `q[\W_]*b[\W_]*r` matches "q-b-r" but not
    # "acquire" (the preceding "a" fails the lookbehind).
    middle = r"[\W_]*".join(re.escape(c) for c in key)
    return rf"(?<![A-Za-z0-9]){middle}(?![A-Za-z0-9])"


__all__ = ["fold_key", "fold_match_pattern"]
