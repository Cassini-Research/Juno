from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "go", "have", "he",
    "her", "here", "hey", "him", "his", "how", "i", "if", "in", "is",
    "it", "its", "me", "my", "no", "not", "of", "on", "or", "our",
    "random", "she", "show", "so", "that", "the", "then", "there",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "while", "who", "why", "with", "you", "your",
}

_TRUSTED_LOWERCASE_SINGLE_SOURCES = {
    "candidate_entity",
    "memory_lexicon",
    "memory_correction",
    "memory_replacement",
}

_COMMON_SINGLE_WORD_REPAIR_BLOCKLIST = {
    "live",
    "look",
    "love",
    "make",
    "move",
    "read",
    "show",
    "take",
    "tech",
    "text",
    "word",
    "work",
}


@dataclass(frozen=True, slots=True)
class PreviewPersonalizationTerm:
    text: str
    source: str = "unknown"

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(_tokens(self.text))

    @property
    def norm(self) -> str:
        return "".join(self.tokens)

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "source": self.source}


def collect_preview_personalization_terms(
    context_payload: dict[str, Any] | None,
    bias_phrases: list[str] | tuple[str, ...] | None = None,
) -> list[PreviewPersonalizationTerm]:
    payload = context_payload if isinstance(context_payload, dict) else {}
    out: list[PreviewPersonalizationTerm] = []
    seen: set[str] = set()

    def add(value: Any, source: str) -> None:
        term = _clean_term(str(value or ""))
        if not term:
            return
        item = PreviewPersonalizationTerm(term, source=source)
        if not _term_allowed_for_preview_repair(item):
            return
        key = item.norm.casefold()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(item)

    structured = payload.get("preview_personalization_terms")
    if isinstance(structured, list):
        for row in structured[:64]:
            if isinstance(row, dict):
                add(row.get("text"), str(row.get("source") or "preview_context"))
            else:
                add(row, "preview_context")

    for value in payload.get("candidate_entities") or []:
        add(value, "candidate_entity")
    for value in payload.get("recent_screen_terms") or []:
        add(value, "recent_screen_term")

    return out[:48]


def preview_personalization_terms_from_plan(plan: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(value: Any, source: str) -> None:
        term = PreviewPersonalizationTerm(_clean_term(str(value or "")), source=source)
        if not _term_allowed_for_preview_repair(term):
            return
        key = term.norm.casefold()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(term.to_dict())

    context = getattr(plan, "context", None)
    for value in getattr(context, "candidate_entities", []) or []:
        add(value, "candidate_entity")
    for value in (getattr(context, "metadata", {}) or {}).get("recent_screen_terms", []) or []:
        add(value, "recent_screen_term")
    if getattr(context, "selected_text", ""):
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'_-]{2,}", str(context.selected_text))[:12]:
            add(token, "selection")

    metadata = getattr(plan, "metadata", {}) or {}
    packet = metadata.get("memory_serving_packet") if isinstance(metadata, dict) else None
    if isinstance(packet, dict):
        for value in packet.get("lexicon_terms") or []:
            add(value, "memory_lexicon")
        for row in packet.get("corrections") or []:
            if isinstance(row, dict):
                add(row.get("corrected"), "memory_correction")
        for row in packet.get("replacements") or []:
            if isinstance(row, dict):
                add(row.get("replacement"), "memory_replacement")

    return out[:48]


def repair_preview_word_dicts(
    word_dicts: list[dict],
    *,
    context_payload: dict[str, Any] | None,
    bias_phrases: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    terms = collect_preview_personalization_terms(context_payload, bias_phrases)
    if not terms or not word_dicts:
        return list(word_dicts or []), {
            "preview_repair_terms": len(terms),
            "preview_repair_applied": 0,
            "preview_repairs": [],
        }

    repaired = [dict(w) for w in word_dicts]
    applied: list[dict[str, Any]] = []
    occupied: set[int] = set()
    terms_by_len = sorted(terms, key=lambda t: (-len(t.tokens), -len(t.norm), t.text.casefold()))

    for term in terms_by_len:
        term_tokens = term.tokens
        if not term_tokens:
            continue
        for width in _repair_widths(term):
            if width > 4:
                continue
            repaired_this_term = False
            for start in range(0, len(repaired) - width + 1):
                idxs = set(range(start, start + width))
                if occupied & idxs:
                    continue
                span_words = repaired[start : start + width]
                span_text = " ".join(str(w.get("word") or w.get("text") or "") for w in span_words).strip()
                replacement = _embedded_term_replacement(span_text, term) if width == 1 else None
                if replacement is not None:
                    _replace_span(repaired, start, width, replacement)
                    occupied |= idxs
                    applied.append(
                        {
                            "raw": span_text,
                            "replacement": replacement,
                            "term": term.text,
                            "source": term.source,
                            "ratio": round(_similarity(_norm(span_text), term.norm), 4),
                            "start_index": start,
                            "width": width,
                        }
                    )
                    repaired_this_term = True
                    break
                if not _span_can_repair(span_text, term):
                    continue
                _replace_span(repaired, start, width, term.text)
                occupied |= idxs
                applied.append(
                    {
                        "raw": span_text,
                        "replacement": term.text,
                        "source": term.source,
                        "ratio": round(_similarity(_norm(span_text), term.norm), 4),
                        "start_index": start,
                        "width": width,
                    }
                )
                repaired_this_term = True
                break
            if repaired_this_term:
                break

    return repaired, {
        "preview_repair_terms": len(terms),
        "preview_repair_applied": len(applied),
        "preview_repairs": applied[:8],
    }


def _replace_span(words: list[dict], start: int, width: int, replacement: str) -> None:
    first = words[start]
    last = words[start + width - 1]
    text = str(first.get("word") or first.get("text") or "")
    prefix = re.match(r"^\W+", text)
    suffix = re.search(r"\W+$", str(last.get("word") or last.get("text") or ""))
    replacement_parts = replacement.split()
    if len(replacement_parts) == width and width > 1:
        for offset, part in enumerate(replacement_parts):
            word = part
            if offset == 0 and prefix:
                word = f"{prefix.group(0)}{word}"
            if offset == width - 1 and suffix:
                word = f"{word}{suffix.group(0)}"
            words[start + offset]["word"] = word
            words[start + offset]["text"] = word
        return
    words[start]["word"] = f"{prefix.group(0) if prefix else ''}{replacement}{suffix.group(0) if suffix else ''}"
    for idx in range(start + 1, start + width):
        words[idx]["word"] = ""
        words[idx]["text"] = ""


def _span_can_repair(span_text: str, term: PreviewPersonalizationTerm) -> bool:
    span_tokens = _tokens(span_text)
    if not span_tokens:
        return False
    span_norm = "".join(span_tokens)
    term_norm = term.norm
    if span_norm.casefold() == term_norm.casefold():
        # Exact acoustic match: canonicalize casing for durable named terms.
        if term.source == "session_entity":
            return False
        return _term_has_case_value(term.text)
    if len(term_norm) < 4 or len(span_norm) < 4:
        return False
    if len(span_tokens) == 1 and span_tokens[0].casefold() in _STOPWORDS:
        return False
    if _single_lowercase_term(term) and _blocked_single_word_span(span_tokens):
        return False
    if _span_matches_explicit_alias(span_norm, term):
        return True
    ratio = _similarity(span_norm.casefold(), term_norm.casefold())
    if len(term.tokens) == 1 and len(span_tokens) > 1:
        if term.source == "session_entity":
            return False
        if _single_lowercase_term(term):
            return False
        if ratio < 0.72:
            return False
    if (
        span_norm.casefold().startswith("a")
        and len(span_norm) == len(term_norm) + 1
        and span_norm[1:2].casefold() == term_norm[:1].casefold()
        and _edit_distance_at_most(span_norm[1:].casefold(), term_norm.casefold(), 1)
    ):
        return True
    if span_norm[:1].casefold() != term_norm[:1].casefold():
        return False
    if (
        max(len(span_norm), len(term_norm)) <= 7
        and abs(len(span_norm) - len(term_norm)) <= 1
        and _edit_distance_at_most(span_norm.casefold(), term_norm.casefold(), 1)
    ):
        return True
    span_soundex = _soundex(span_norm)
    term_soundex = _soundex(term_norm)
    if span_soundex and term_soundex and span_soundex == term_soundex:
        if _single_lowercase_term(term) and _blocked_single_word_span(span_tokens):
            return False
        if len(term.tokens) > 1 and (len(span_tokens) != len(term.tokens) or ratio < 0.78):
            return False
        if abs(len(span_norm) - len(term_norm)) <= 2 or ratio >= 0.62:
            return True
    threshold = 0.88 if len(term_norm) <= 6 else 0.82
    if len(term.tokens) > 1:
        threshold = 0.84
    return ratio >= threshold


def _embedded_term_replacement(span_text: str, term: PreviewPersonalizationTerm) -> str | None:
    if len(term.tokens) < 2:
        return None
    raw = (span_text or "").strip()
    if not raw or re.search(r"\s", raw):
        return None
    norm_chars: list[str] = []
    raw_indexes: list[int] = []
    for idx, ch in enumerate(raw):
        if ch.isalnum():
            norm_chars.append(ch.casefold())
            raw_indexes.append(idx)
    norm = "".join(norm_chars)
    term_norm = term.norm.casefold()
    if len(term_norm) < 6:
        return None
    start = norm.find(term_norm)
    if start < 0:
        return None
    end = start + len(term_norm)
    if start == 0 and end == len(norm):
        return None
    raw_start = raw_indexes[start]
    raw_end = raw_indexes[end - 1] + 1
    prefix = raw[:raw_start]
    suffix = raw[raw_end:]
    if not prefix and not suffix:
        return None
    parts = [part for part in (prefix, term.text, suffix) if part]
    return " ".join(parts)


def _term_allowed_for_preview_repair(term: PreviewPersonalizationTerm) -> bool:
    text = term.text.strip()
    toks = term.tokens
    if not text or not toks or len(text) > 80 or len(toks) > 4:
        return False
    if len(term.norm) < 3:
        return False
    if len(toks) == 1 and toks[0].casefold() in _STOPWORDS:
        return False
    if re.fullmatch(r"[A-Z]", text.strip()):
        return False
    return (
        _term_has_case_value(text)
        or any(ch.isdigit() for ch in text)
        or len(toks) > 1
        or _trusted_lowercase_single_term(term)
    )


def _trusted_lowercase_single_term(term: PreviewPersonalizationTerm) -> bool:
    return _single_lowercase_term(term) and term.source in _TRUSTED_LOWERCASE_SINGLE_SOURCES


def _single_lowercase_term(term: PreviewPersonalizationTerm) -> bool:
    toks = term.tokens
    text = term.text.strip()
    return len(toks) == 1 and text.casefold() == text and toks[0] not in _STOPWORDS


def _repair_widths(term: PreviewPersonalizationTerm) -> tuple[int, ...]:
    base = len(term.tokens)
    if base <= 0:
        return ()
    widths: list[int] = []
    if base == 1:
        widths.extend((3, 2, 1))
    else:
        widths.extend((base, base - 1))
    return tuple(width for width in dict.fromkeys(widths) if 1 <= width <= 4)


def _span_matches_explicit_alias(span_norm: str, term: PreviewPersonalizationTerm) -> bool:
    aliases = _explicit_preview_phrase_aliases(term)
    if not aliases:
        return False
    key = span_norm.casefold()
    if key in aliases:
        return True
    if term.norm.casefold() == "pytest" and key.startswith("pythonm"):
        command_tail = key.removeprefix("pythonm")
        return any(
            command_tail == alias
            or re.fullmatch(rf"{re.escape(alias)}(?:testsv?(?:2|two))?", command_tail) is not None
            for alias in aliases
        )
    return False


def _explicit_preview_phrase_aliases(term: PreviewPersonalizationTerm) -> set[str]:
    if term.norm.casefold() == "pytest":
        return {"mpitest", "mpidest", "pidest", "pydest", "pitest", "pytests"}
    return set()


def _blocked_single_word_span(span_tokens: list[str]) -> bool:
    return len(span_tokens) == 1 and span_tokens[0].casefold() in _COMMON_SINGLE_WORD_REPAIR_BLOCKLIST


def _term_has_case_value(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if re.search(r"[a-z][A-Z]|[_/-]", clean):
        return True
    if re.search(r"\b[A-Z]{2,}\b", clean):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9']*", clean)
    if not words:
        return False
    return any(w[:1].isupper() and w.casefold() not in _STOPWORDS for w in words)


def _clean_term(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    return value.strip(" ,.!?;:\"'`")


def _tokens(value: str) -> list[str]:
    return [m.group(0).casefold() for m in re.finditer(r"[A-Za-z0-9]+", value or "")]


def _norm(value: str) -> str:
    return "".join(_tokens(value))


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def _soundex(value: str) -> str:
    letters = re.findall(r"[A-Za-z]", value or "")
    if not letters:
        return ""
    first = letters[0].upper()
    codes = {
        **dict.fromkeys("BFPVbfpv", "1"),
        **dict.fromkeys("CGJKQSXZcgjkqsxz", "2"),
        **dict.fromkeys("DTdt", "3"),
        **dict.fromkeys("Ll", "4"),
        **dict.fromkeys("MNmn", "5"),
        **dict.fromkeys("Rr", "6"),
    }
    out: list[str] = []
    previous = codes.get(letters[0], "")
    for ch in letters[1:]:
        code = codes.get(ch, "")
        if code and code != previous:
            out.append(code)
        previous = code
    return (first + "".join(out) + "000")[:4]


def _edit_distance_at_most(a: str, b: str, limit: int) -> bool:
    if abs(len(a) - len(b)) > limit:
        return False
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return False
        previous = current
    return previous[-1] <= limit
