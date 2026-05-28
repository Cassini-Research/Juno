from __future__ import annotations

import re

from juno_v2.writer.deterministic import normalize_plain_dictation


_MONTH_CANONICAL = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
}
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTH_CANONICAL) + r")\b", re.IGNORECASE)
_STANDALONE_LOWER_I_RE = re.compile(r"(?<![A-Za-z])i(?![A-Za-z])")
_STANDALONE_LETTER_RE = re.compile(r"(?<![A-Za-z'.])([bcdefghjklmnopqrstuvwxyz])(?![A-Za-z'.])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_COMMON_INTERNAL_WORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "go", "got", "had", "has",
    "have", "he", "her", "here", "him", "his", "how", "if", "in", "is",
    "it", "it's", "me", "my", "not", "now", "of", "on", "or", "our",
    "random", "see", "session", "she", "show", "so", "still", "that",
    "the", "then", "there", "they", "this", "to", "up", "was", "we",
    "we're", "were", "what", "when", "where", "while", "who", "why",
    "will", "with", "you", "your",
}


def normalize_preview_orthography(
    committed_text: str,
    tail_text: str,
) -> tuple[str, str, dict[str, object]]:
    """Normalize HUD display text without mutating LocalAgreement state.

    Committed text is stable enough for full sentence orthography. The tail is
    still volatile, so it only gets context-aware mechanical fixes; its first
    word is capitalized only at the beginning of the utterance or after a hard
    sentence boundary in committed text.
    """
    raw_committed = committed_text or ""
    raw_tail = tail_text or ""
    committed = _normalize_inline_tail(raw_committed, capitalize_start=True) if raw_committed.strip() else ""
    tail_capitalize = _tail_starts_sentence(committed)
    tail = _normalize_inline_tail(raw_tail, capitalize_start=tail_capitalize)
    applied = int(committed != raw_committed) + int(tail != raw_tail)
    return committed, tail, {
        "preview_orthography_applied": applied,
        "preview_orthography_committed_changed": committed != raw_committed,
        "preview_orthography_tail_changed": tail != raw_tail,
    }


def _tail_starts_sentence(committed_text: str) -> bool:
    stripped = (committed_text or "").rstrip()
    return not stripped or stripped.endswith((".", "!", "?", "\n"))


def _normalize_inline_tail(text: str, *, capitalize_start: bool) -> str:
    current = normalize_plain_dictation(text)
    if not current:
        return current
    current = _STANDALONE_LOWER_I_RE.sub("I", current)
    current = _MONTH_RE.sub(lambda m: _MONTH_CANONICAL[m.group(1).casefold()], current)
    current = _STANDALONE_LETTER_RE.sub(lambda m: m.group(1).upper(), current)
    current = _lower_unexpected_internal_capitals(
        current,
        start_at_sentence=capitalize_start,
    )
    if capitalize_start:
        current = _capitalize_sentence_starts(current)
    else:
        current = _capitalize_after_sentence_boundaries(current)
    return current.strip()


def _lower_unexpected_internal_capitals(text: str, *, start_at_sentence: bool) -> str:
    out: list[str] = []
    pos = 0
    sentence_start = start_at_sentence
    for match in _WORD_RE.finditer(text):
        between = text[pos : match.start()]
        out.append(between)
        sentence_start = _advance_sentence_state(sentence_start, between)
        token = match.group(0)
        replacement = token
        if (
            not sentence_start
            and token[:1].isupper()
            and not token.isupper()
            and token.casefold() in _COMMON_INTERNAL_WORDS
        ):
            replacement = token.casefold()
        out.append(replacement)
        sentence_start = False
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def _advance_sentence_state(sentence_start: bool, text: str) -> bool:
    state = sentence_start
    for ch in text:
        if ch == ".":
            state = True
        elif ch in "!?\n":
            state = True
        elif not ch.isspace() and ch not in "\"'([{":
            state = False
    return state


def _capitalize_sentence_starts(text: str) -> str:
    out: list[str] = []
    capitalize_next = True
    for idx, ch in enumerate(text):
        if capitalize_next and ch.isalpha():
            out.append(ch.upper())
            capitalize_next = False
            continue
        out.append(ch)
        if ch == "." and idx + 1 < len(text) and not text[idx + 1].isspace():
            continue
        if ch in ".!?\n":
            capitalize_next = True
        elif not ch.isspace() and ch not in "\"'([{":
            capitalize_next = False
    return "".join(out)


def _capitalize_after_sentence_boundaries(text: str) -> str:
    out: list[str] = []
    capitalize_next = False
    for idx, ch in enumerate(text):
        if capitalize_next and ch.isalpha():
            out.append(ch.upper())
            capitalize_next = False
            continue
        out.append(ch)
        if ch == "." and idx + 1 < len(text) and not text[idx + 1].isspace():
            continue
        if ch in ".!?\n":
            capitalize_next = True
        elif not ch.isspace() and ch not in "\"'([{":
            capitalize_next = False
    return "".join(out)
