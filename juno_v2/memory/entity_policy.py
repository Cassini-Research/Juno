from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from juno_v2.memory.ai_dictionary import is_ai_glossary_term
from juno_v2.memory.term_policy import learned_term_allowed


_FALLBACK_COMMON_SINGLE_WORDS = frozenset(
    """
    able about above action actually after again against ago ahead all almost
    already also always am among an and another any anyone anything app are
    around as ask asked at away back bad basically be because been before being
    best better between big bold both bring build bunch but by call can case
    check clear clearly code coming common context continuous could couple
    create current customer day deadline deep default demo did different do
    does doing done dont down each earlier easily edit edited else end enough
    especially even ever every everything exactly example fail facing few final
    finally find first fix flow focus font for format formatting from get gets
    getting give go goes going good got great had happen happening hard has
    have he hello here how hud i idea if in include information inside install
    is issue issues it its just keep kind know last later learn learning less
    lets like line list local lot made make many may me mean means mention
    middle might model more most much my need needs never new next no not note
    now of off old on once one only open or other our out over owner paragraph
    pass pause pasted plan please point previous proper pull put putting random
    read really regular request rerun reset review right run same see seems set
    should show since so some something source start started status still style
    take task tell term terms text than that the their them then there these
    thing things think this those through time title to today tomorrow too use
    used user very was way we well were what when where which while who why will
    with without word words work works would write writing wrong yes yet you
    your
    """.split()
)

_COMMON_WORDS_PATH = os.environ.get("JUNO_V2_COMMON_WORDS_PATH", "/usr/share/dict/words")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_RARE_SINGLE_WORD_ALLOWLIST = frozenset(
    {
        "codex",
        "juno",
    }
)


def term_present_in_text(term: str, text: str) -> bool:
    term_units = [m.group(0).casefold() for m in _WORD_RE.finditer(term or "")]
    text_units = [m.group(0).casefold() for m in _WORD_RE.finditer(text or "")]
    if not term_units or not text_units:
        return False
    width = len(term_units)
    for idx in range(0, len(text_units) - width + 1):
        if text_units[idx : idx + width] == term_units:
            return True
    compact_term = "".join(term_units)
    return len(compact_term) >= 4 and compact_term in "".join(text_units)


def session_entity_allowed(value: str) -> bool:
    """Return True only for terms worth feeding back into future ASR.

    Session entities are automatic memory, not user-authored vocabulary. The
    policy is intentionally stricter than lexicon learning: ordinary English
    single words are cheap for ASR to hear and expensive when fed back as bias.
    """

    v = (value or "").strip()
    if not v or not learned_term_allowed(v):
        return False
    if _looks_like_lowercase_slug(v):
        return False
    units = _word_units(v)
    if not units:
        return False
    if len(units) == 1:
        return _single_token_entity_allowed(v, units[0])
    return _multi_token_entity_allowed(v, units)


def commit_session_entity_allowed(
    value: str,
    *,
    committed_text: str,
    spoken_evidence_text: str,
    context_text: str = "",
    context_backed: bool = False,
) -> bool:
    if not session_entity_allowed(value) and not _context_backed_common_proper_allowed(
        value,
        committed_text=committed_text,
        context_text=context_text,
        context_backed=context_backed,
    ):
        return False
    if not term_present_in_text(value, committed_text):
        return False
    if term_present_in_text(value, spoken_evidence_text):
        return True
    if context_backed and term_present_in_text(value, context_text):
        return True
    return bool(context_text and term_present_in_text(value, context_text))


def _context_backed_common_proper_allowed(
    value: str,
    *,
    committed_text: str,
    context_text: str,
    context_backed: bool,
) -> bool:
    """Allow common-looking proper names only when current screen context proves them.

    Automatic session memory must not learn ordinary words from transcripts. But
    screen context is stronger evidence: if WhatsApp/Slack shows "Tara" or
    "Noah" and the committed output also contains that exact titlecase token, it
    is useful short-lived context even though `/usr/share/dict/words` marks the
    name as common English.
    """

    if not context_backed:
        return False
    v = (value or "").strip()
    if not v or not learned_term_allowed(v):
        return False
    units = _word_units(v)
    if len(units) != 1:
        return False
    token = units[0]
    if len(token) < 4:
        return False
    if not re.fullmatch(r"[A-Z][a-z]{3,}[A-Za-z]*", v):
        return False
    if not common_english_single_word(token):
        return False
    return term_present_in_text(v, committed_text) and term_present_in_text(v, context_text)


def common_english_single_word(value: str) -> bool:
    units = _word_units(value)
    if len(units) != 1:
        return False
    word = units[0].casefold()
    if not word:
        return False
    if word in _FALLBACK_COMMON_SINGLE_WORDS:
        return True
    common = _system_common_words()
    if word in common:
        return True
    return any(base in _FALLBACK_COMMON_SINGLE_WORDS or base in common for base in _common_bases(word))


def _single_token_entity_allowed(original: str, token: str) -> bool:
    if token.casefold() in _RARE_SINGLE_WORD_ALLOWLIST:
        return True
    if is_ai_glossary_term(original):
        return True
    if _looks_like_acronym(original):
        return True
    if _looks_like_mixed_case_identifier(original):
        return True
    if any(ch.isalpha() for ch in original) and any(ch.isdigit() for ch in original):
        return True
    if common_english_single_word(token):
        return False
    if re.fullmatch(r"[A-Z][a-z]{2,}[A-Za-z]*", original):
        return True
    return False


def _multi_token_entity_allowed(original: str, units: list[str]) -> bool:
    pieces = re.findall(r"[A-Za-z][A-Za-z0-9]*", original)
    if not pieces:
        return False
    if any(is_ai_glossary_term(piece) for piece in pieces):
        return True
    if any(_single_token_has_identifier_shape(piece) for piece in pieces):
        return True
    if all(piece[:1].isupper() for piece in pieces):
        return any(piece.casefold() not in _FALLBACK_COMMON_SINGLE_WORDS for piece in pieces)
    return False


def _single_token_has_identifier_shape(value: str) -> bool:
    return (
        _looks_like_acronym(value)
        or _looks_like_mixed_case_identifier(value)
        or (any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value))
    )


def _looks_like_acronym(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{2,}(?:[._-][A-Z0-9]{2,})*", value or ""))


def _looks_like_mixed_case_identifier(value: str) -> bool:
    return bool(re.search(r"[a-z][A-Z]", value or ""))


def _looks_like_lowercase_slug(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)+", value or ""))


def _word_units(value: str) -> list[str]:
    return [m.group(0) for m in _WORD_RE.finditer(value or "")]


def _common_bases(word: str) -> tuple[str, ...]:
    out: list[str] = []
    if len(word) > 4 and word.endswith("ies"):
        out.append(word[:-3] + "y")
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        out.extend((stem, stem + "e"))
        if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
            out.append(stem[:-1])
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        out.extend((stem, stem + "e"))
        if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
            out.append(stem[:-1])
    if len(word) > 4 and word.endswith("ly"):
        out.append(word[:-2])
    if len(word) > 4 and word.endswith("er"):
        out.append(word[:-2])
    if len(word) > 5 and word.endswith("est"):
        out.append(word[:-3])
    if len(word) > 3 and word.endswith("s"):
        out.append(word[:-1])
    return tuple(dict.fromkeys(base for base in out if len(base) >= 3))


@lru_cache(maxsize=1)
def _system_common_words() -> frozenset[str]:
    path = Path(_COMMON_WORDS_PATH)
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return frozenset()
    words: set[str] = set()
    for line in raw.splitlines():
        word = line.strip().casefold()
        if re.fullmatch(r"[a-z]{3,}", word):
            words.add(word)
    return frozenset(words)


__all__ = [
    "commit_session_entity_allowed",
    "common_english_single_word",
    "session_entity_allowed",
    "term_present_in_text",
]
