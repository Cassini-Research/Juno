from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

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
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
_WORD_LIST_PATHS = (
    Path("/usr/share/dict/words"),
    Path("/usr/share/dict/web2"),
)
_GRAMMATICAL_INTERNAL_WORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "go", "got", "had", "has",
    "have", "he", "her", "here", "him", "his", "how", "if", "in", "is",
    "it", "it's", "make", "maybe", "me", "my", "need", "not", "now", "of",
    "on", "or", "our", "see", "she", "show", "so", "still", "that", "the",
    "then", "there", "they", "this", "to", "uh", "um", "up", "was", "we",
    "we're", "were", "what", "when", "where", "while", "who", "why", "will",
    "with", "you", "your",
}
_ORDINARY_WORD_SUFFIXES = ("ed", "er", "est", "ing", "ly")


_JOIN_PERIOD_RE = re.compile(r"(?<!\.)(?<![A-Z])(?<!\be\.g)(?<!\bi\.e)(?<!\betc)\.( +)(?=[a-z])")
_DISPLAY_NEWLINE_CUE_RE = re.compile(
    r"(?:^|(?<= ))(?P<cue>new +(?:line|paragraph))\b[ .,]*",
    re.IGNORECASE,
)
_NEWLINE_CUE_DETERMINERS = {"the", "a", "an", "this", "that", "each", "every", "my", "your", "our"}


def _smooth_window_joins(committed: str) -> str:
    """Display-only cleanup of rolling-window seams in the live HUD.

    Each preview window punctuates independently, so committed text shows
    "how we think. ask, interrupt ourselves. and move…" — a stray period and
    lowercase restart at every window join. The final paste re-decodes the
    full audio, so this never touches the pasted text; it only keeps the
    HUD readable. Spoken newline cues render as actual breaks for the same
    reason (the user said "new line", the HUD showed the words).
    """
    if not committed:
        return committed
    out = _JOIN_PERIOD_RE.sub(r"\1", committed)

    def _cue_repl(match: re.Match[str]) -> str:
        before = out[: match.start()].rstrip()
        prev_word = before.rsplit(" ", 1)[-1].rstrip(",.;:").casefold() if before else ""
        if prev_word in _NEWLINE_CUE_DETERMINERS:
            return match.group(0)
        return "\n\n" if "paragraph" in match.group("cue").casefold() else "\n"

    out = _DISPLAY_NEWLINE_CUE_RE.sub(_cue_repl, out)
    return out


def normalize_preview_orthography(
    committed_text: str,
    tail_text: str,
    *,
    protected_terms: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, str, dict[str, object]]:
    """Normalize HUD display text without mutating LocalAgreement state.

    Preview punctuation is not stable enough to drive sentence recasing. Keep
    live display orthography mechanical: fix standalone tokens and capitalize
    only the first visible word while preserving final-stage punctuation cleanup
    for the final transcript path.
    """
    raw_committed = committed_text or ""
    raw_tail = tail_text or ""
    protected = _protected_word_norms(protected_terms)
    committed = (
        _normalize_inline_tail(
            raw_committed,
            capitalize_start=True,
            trust_sentence_boundaries=False,
            protected_word_norms=protected,
        )
        if raw_committed.strip()
        else ""
    )
    tail_capitalize = _tail_starts_sentence(committed)
    tail = _normalize_inline_tail(
        raw_tail,
        capitalize_start=tail_capitalize,
        trust_sentence_boundaries=False,
        protected_word_norms=protected,
    )
    applied = int(committed != raw_committed) + int(tail != raw_tail)
    return committed, tail, {
        "preview_orthography_applied": applied,
        "preview_orthography_committed_changed": committed != raw_committed,
        "preview_orthography_tail_changed": tail != raw_tail,
    }


def _tail_starts_sentence(committed_text: str) -> bool:
    stripped = (committed_text or "").rstrip()
    return not stripped or stripped.endswith((".", "!", "?", "\n"))


def _normalize_inline_tail(
    text: str,
    *,
    capitalize_start: bool,
    trust_sentence_boundaries: bool = True,
    protected_word_norms: set[str] | None = None,
) -> str:
    current = normalize_plain_dictation(text)
    if not current:
        return current
    current = _STANDALONE_LOWER_I_RE.sub("I", current)
    current = _MONTH_RE.sub(lambda m: _MONTH_CANONICAL[m.group(1).casefold()], current)
    current = _STANDALONE_LETTER_RE.sub(lambda m: m.group(1).upper(), current)
    current = _lower_unexpected_internal_capitals(
        current,
        start_at_sentence=capitalize_start,
        trust_sentence_boundaries=trust_sentence_boundaries,
        protected_word_norms=protected_word_norms or set(),
    )
    if capitalize_start:
        current = (
            _capitalize_sentence_starts(current)
            if trust_sentence_boundaries
            else _capitalize_first_alpha(current)
        )
    elif trust_sentence_boundaries:
        current = _capitalize_after_sentence_boundaries(current)
    return current.strip()


def _lower_unexpected_internal_capitals(
    text: str,
    *,
    start_at_sentence: bool,
    trust_sentence_boundaries: bool = True,
    protected_word_norms: set[str] | None = None,
) -> str:
    out: list[str] = []
    pos = 0
    sentence_start = start_at_sentence
    protected = protected_word_norms or set()
    for match in _WORD_RE.finditer(text):
        between = text[pos : match.start()]
        out.append(between)
        if trust_sentence_boundaries:
            sentence_start = _advance_sentence_state(sentence_start, between)
        elif pos > 0:
            sentence_start = False
        token = match.group(0)
        replacement = token
        false_boundary = _looks_like_preview_false_boundary(between)
        if _should_lower_internal_titlecase(
            token,
            sentence_start=sentence_start,
            false_boundary=false_boundary,
            protected_word_norms=protected,
        ):
            replacement = token.casefold()
        out.append(replacement)
        sentence_start = False
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def _should_lower_internal_titlecase(
    token: str,
    *,
    sentence_start: bool,
    false_boundary: bool,
    protected_word_norms: set[str],
) -> bool:
    if sentence_start or not token[:1].isupper() or token.isupper():
        return False
    norm = _word_norm(token)
    if not norm or norm in protected_word_norms:
        return False
    if norm in _GRAMMATICAL_INTERNAL_WORDS:
        return True
    if len(norm) > 4 and norm.endswith(_ORDINARY_WORD_SUFFIXES):
        return True
    # Preview ASR often inserts a period or ellipsis at a pause, then Titlecases
    # the next ordinary English token. Since live preview sentence boundaries are
    # not trusted, lower those ordinary words while leaving unknown names alone.
    if false_boundary and _looks_like_ordinary_english_word(norm):
        return True
    return False


def _looks_like_preview_false_boundary(text_between_words: str) -> bool:
    return any(ch in text_between_words for ch in ".!?")


def _looks_like_ordinary_english_word(norm: str) -> bool:
    if norm in _english_words():
        return True
    return len(norm) > 4 and norm.endswith(_ORDINARY_WORD_SUFFIXES)


def _protected_word_norms(terms: list[str] | tuple[str, ...] | None) -> set[str]:
    out: set[str] = set()
    for term in terms or ():
        for token in _TOKEN_RE.findall(str(term or "")):
            norm = _word_norm(token)
            if norm:
                out.add(norm)
    return out


def _word_norm(token: str) -> str:
    return "".join(ch.lower() for ch in token if ch.isalnum())


@lru_cache(maxsize=1)
def _english_words() -> frozenset[str]:
    for path in _WORD_LIST_PATHS:
        try:
            if path.exists():
                return frozenset(
                    line.strip().casefold()
                    for line in path.read_text(errors="ignore").splitlines()
                    if line.strip()
                )
        except OSError:
            continue
    return frozenset()


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


def _capitalize_first_alpha(text: str) -> str:
    out: list[str] = []
    changed = False
    for ch in text:
        if not changed and ch.isalpha():
            out.append(ch.upper())
            changed = True
            continue
        out.append(ch)
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
