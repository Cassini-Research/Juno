from __future__ import annotations

import difflib
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from juno_v2.context.compiler import TranscriptAdjudicationPacket
    from juno_v2.transcript.contracts import TranscriptAdjudicationResult


_ASSISTANT_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure\b|here\s+is\b|here's\b|the\s+corrected\s+text\s+is\b|corrected\s+text\s*:)",
    re.IGNORECASE,
)
_MARKDOWN_FENCE_RE = re.compile(r"```")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S")


def validate_adjudication_result(
    packet: TranscriptAdjudicationPacket,
    result: TranscriptAdjudicationResult,
    *,
    allow_chunked_insertions: bool = False,
) -> tuple[bool, str]:
    text = str(getattr(result, "corrected_text", "") or "")
    checks = [
        no_assistant_artifacts,
        lambda t: protected_terms_preserved(packet, t),
        lambda t: evidence_proper_terms_not_mutated(packet, t),
        lambda t: numbers_dates_safe(packet, t),
        lambda t: semantic_drift_safe(packet, t),
        lambda t: source_words_preserved(packet, t),
        lambda t: low_signal_mid_sentence_capitalization_safe(packet, t),
        lambda t: no_formatting_in_transcript_stage(packet, t),
    ]
    if not allow_chunked_insertions:
        checks.insert(4, lambda _t: unsupported_insertions_safe(packet, result))
    for check in checks:
        ok, reason = check(text)
        if not ok:
            return False, reason
    return True, "ok"


def protected_terms_preserved(packet: TranscriptAdjudicationPacket, corrected_text: str) -> tuple[bool, str]:
    terms = tuple(getattr(packet, "protected_terms", ()) or ())
    if not terms:
        return True, "ok"
    raw_context = _authoritative_transcript_text(packet)
    for term in terms:
        t = str(term or "").strip()
        if not t:
            continue
        if _looks_like_glued_pronoun_i(t):
            continue
        # Only require preservation when the term was part of the evidence or
        # the adjudicator claims it used it.
        if (
            _term_present_loose(raw_context, t)
            and not _term_present_loose(corrected_text, t)
            and not _near_protected_alias_present(corrected_text, t, terms)
        ):
            return False, f"protected_term_dropped:{t}"
        authoritative_sources = (
            getattr(packet, "memory_candidate_text", ""),
            getattr(packet, "whisper_text", ""),
            getattr(packet, "raw_text", ""),
        )
        required_count = max(_term_count_loose(str(source or ""), t) for source in authoritative_sources)
        observed_count = _term_count_loose(corrected_text, t) + _near_protected_alias_count(corrected_text, t, terms)
        if required_count > 1 and observed_count < required_count:
            return False, f"protected_term_count_dropped:{t}"
    return True, "ok"


def _near_protected_alias_present(text: str, term: str, protected_terms: tuple[Any, ...]) -> bool:
    return _near_protected_alias_count(text, term, protected_terms) > 0


def _near_protected_alias_count(text: str, term: str, protected_terms: tuple[Any, ...]) -> int:
    folded = term.casefold()
    count = 0
    for other in protected_terms:
        alias = str(other or "").strip()
        if not alias or alias.casefold() == folded:
            continue
        if not _near_spelling(folded, alias.casefold()):
            continue
        count += _term_count_loose(text, alias)
    return count


_LOW_SIGNAL_PROPER_TERMS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "he", "here", "hey",
    "how", "i", "if", "in", "is", "it", "its", "me", "my", "no",
    "not", "now", "of", "on", "or", "our", "she", "that", "the",
    "then", "there", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "who", "why", "with", "yes", "you",
}


def evidence_proper_terms_not_mutated(packet: TranscriptAdjudicationPacket, corrected_text: str) -> tuple[bool, str]:
    evidence_terms = _proper_terms_from_packet(packet)
    if not evidence_terms:
        return True, "ok"
    folded_out = (corrected_text or "").casefold()
    output_terms = _extract_proper_like_terms(corrected_text)
    output_folded = {t.casefold() for t in output_terms}
    protected_folded = {str(t or "").strip().casefold() for t in (getattr(packet, "protected_terms", ()) or ()) if str(t or "").strip()}
    for term in evidence_terms:
        folded = term.casefold()
        if folded in folded_out:
            continue
        for out in output_terms:
            out_folded = out.casefold()
            if out_folded == folded:
                break
            if out_folded in output_folded and _near_spelling(folded, out_folded):
                if out_folded in protected_folded:
                    break
                return False, f"proper_noun_mutated:{term}->{out}"
    return True, "ok"


def numbers_dates_safe(packet: TranscriptAdjudicationPacket, corrected_text: str) -> tuple[bool, str]:
    source = _authorized_transcript_and_terms_text(packet)
    src_nums = _numeric_tokens(source)
    out_nums = _numeric_tokens(corrected_text)
    if out_nums and not out_nums <= src_nums:
        return False, "new_number_without_evidence"
    return True, "ok"


def _term_present_loose(text: str, term: str) -> bool:
    """Return True when ``term`` appears with equivalent token boundaries.

    Context extraction can protect phrases such as ``May 18 2026`` while the
    final correction writes normal punctuation, e.g. ``May 18, 2026``. Treat
    punctuation between alphanumeric tokens as formatting, not as a dropped
    protected term. Single-token names still require a normal word-boundary
    match, so this does not turn arbitrary substrings into protected hits.
    """
    haystack = text or ""
    tokens = re.findall(r"[A-Za-z0-9]+", term or "")
    if not tokens:
        return False
    separator = r"[\W_]*" if len(tokens) >= 2 and all(len(tok) == 1 and tok.isalpha() for tok in tokens) else r"[\W_]+"
    pattern = r"(?<![A-Za-z0-9])" + separator.join(re.escape(tok) for tok in tokens) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, haystack, flags=re.IGNORECASE))


def _term_count_loose(text: str, term: str) -> int:
    haystack = text or ""
    tokens = re.findall(r"[A-Za-z0-9]+", term or "")
    if not tokens:
        return 0
    separator = r"[\W_]*" if len(tokens) >= 2 and all(len(tok) == 1 and tok.isalpha() for tok in tokens) else r"[\W_]+"
    pattern = r"(?<![A-Za-z0-9])" + separator.join(re.escape(tok) for tok in tokens) + r"(?![A-Za-z0-9])"
    return len(re.findall(pattern, haystack, flags=re.IGNORECASE))


def _proper_terms_from_packet(packet: TranscriptAdjudicationPacket) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        term = (value or "").strip()
        if not _proper_term_allowed(term):
            return
        key = term.casefold()
        if key in seen:
            return
        seen.add(key)
        out.append(term)

    for term in getattr(packet, "protected_terms", ()) or ():
        add(str(term or ""))

    for item in getattr(packet, "context_terms", ()) or ():
        try:
            add(str(getattr(item, "canonical", None) or getattr(item, "text", "") or ""))
        except Exception:
            continue

    context_values = [
        getattr(packet, "selected_text_excerpt", ""),
        getattr(packet, "focused_text_before", ""),
        getattr(packet, "focused_text_after", ""),
        getattr(packet, "field_text_excerpt", ""),
        getattr(packet, "window_title", ""),
        getattr(packet, "focused_file_path", ""),
        getattr(packet, "symbol_under_cursor", ""),
    ]
    if getattr(packet, "stage", "") == "live":
        context_values.extend((
            getattr(packet, "base_visible_text", ""),
            getattr(packet, "live_preview_text", ""),
        ))

    for value in context_values:
        for term in _extract_proper_like_terms(str(value or "")):
            add(term)

    return tuple(out[:32])


def _extract_proper_like_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"(?<!\w)(?:[A-Z][A-Za-z]{2,}|[A-Za-z]+[A-Z][A-Za-z]*)(?!\w)", text or ""):
        term = match.group(0)
        if _looks_like_glued_pronoun_i(term):
            continue
        terms.append(term)
    return terms


def _proper_term_allowed(term: str) -> bool:
    if len(term) < 3 or len(term) > 48:
        return False
    if term.casefold() in _LOW_SIGNAL_PROPER_TERMS:
        return False
    if _looks_like_glued_pronoun_i(term):
        return False
    if not any(ch.isalpha() for ch in term):
        return False
    return True


def _looks_like_glued_pronoun_i(term: str) -> bool:
    token = (term or "").strip()
    # SFSpeech can glue a normal lowercase word to the pronoun I at a pause,
    # producing artifacts like "workI" or "thingsI". Those are boundary errors,
    # not names or identifiers, and must not become protected terms.
    return bool(re.match(r"^[a-z]{2,}I(?:m|d|ll|ve|re)?$", token))


def _near_spelling(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 3:
        return False
    if not a or not b or a[:1] != b[:1]:
        return False
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio() >= 0.76


def unsupported_insertions_safe(packet: TranscriptAdjudicationPacket, result: TranscriptAdjudicationResult) -> tuple[bool, str]:
    source = _authorized_transcript_and_terms_text(packet)
    for op in getattr(result, "ops", ()) or ():
        if getattr(op, "op", None) != "insert":
            continue
        inserted = str(getattr(op, "text", "") or "").strip()
        words = re.findall(r"[A-Za-z][A-Za-z']*", inserted)
        if len(words) < 3:
            continue
        if _phrase_present_loose(source, inserted):
            continue
        return False, f"unsupported_inserted_phrase:{_short_reason(inserted)}"
    return _inserted_token_spans_safe(
        packet,
        str(getattr(result, "corrected_text", "") or ""),
    )


def _inserted_token_spans_safe(packet: TranscriptAdjudicationPacket, corrected_text: str) -> tuple[bool, str]:
    source_text = _primary_authoritative_transcript_text(packet)
    authorized_text = _authorized_transcript_and_terms_text(packet)
    source_tokens = _token_values(source_text)
    output_tokens = _token_values(corrected_text)
    if not source_tokens or not output_tokens:
        return True, "ok"

    matcher = difflib.SequenceMatcher(a=source_tokens, b=output_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or j1 >= j2:
            continue
        out_span = output_tokens[j1:j2]
        src_span = source_tokens[i1:i2]
        inserted_like = tag == "insert" or len(out_span) >= len(src_span) + 3
        if not inserted_like:
            continue
        duplicate = _duplicated_ngram(source_tokens, output_tokens, out_span)
        if duplicate:
            return False, f"duplicated_output_phrase:{_short_reason(duplicate)}"
        phrase = " ".join(out_span)
        if _phrase_tokens_present(authorized_text, out_span):
            continue
        if len(out_span) < 3 and not any(_meaningful_inserted_token(tok) for tok in out_span):
            continue
        return False, f"unsupported_output_phrase:{_short_reason(phrase)}"
    return True, "ok"


def _phrase_present_loose(text: str, phrase: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9]+", phrase or "")
    if not tokens:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"[\W_]+".join(re.escape(tok) for tok in tokens) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, text or "", flags=re.IGNORECASE))


def _authoritative_transcript_text(packet: TranscriptAdjudicationPacket) -> str:
    if getattr(packet, "stage", "") == "live":
        values = (
            getattr(packet, "base_visible_text", ""),
            getattr(packet, "live_preview_text", ""),
            getattr(packet, "memory_candidate_text", ""),
            getattr(packet, "whisper_text", ""),
            getattr(packet, "raw_text", ""),
        )
    else:
        # Final-stage preview is a UI hint only. It must never authorize words
        # that final ASR or memory-normalized transcript evidence did not carry.
        values = (
            getattr(packet, "memory_candidate_text", ""),
            getattr(packet, "whisper_text", ""),
            getattr(packet, "raw_text", ""),
        )
    return " ".join(str(x or "") for x in values if str(x or "").strip())


def _primary_authoritative_transcript_text(packet: TranscriptAdjudicationPacket) -> str:
    if getattr(packet, "stage", "") == "live":
        return str(
            getattr(packet, "base_visible_text", "")
            or getattr(packet, "live_preview_text", "")
            or getattr(packet, "memory_candidate_text", "")
            or getattr(packet, "whisper_text", "")
            or getattr(packet, "raw_text", "")
            or ""
        )
    return str(
        getattr(packet, "memory_candidate_text", "")
        or getattr(packet, "whisper_text", "")
        or getattr(packet, "raw_text", "")
        or ""
    )


def _authorized_transcript_and_terms_text(packet: TranscriptAdjudicationPacket) -> str:
    values: list[str] = [_authoritative_transcript_text(packet)]
    for term in getattr(packet, "protected_terms", ()) or ():
        if str(term or "").strip():
            values.append(str(term))
    for item in getattr(packet, "context_terms", ()) or ():
        try:
            for value in (
                getattr(item, "text", None),
                getattr(item, "canonical", None),
                *(getattr(item, "spoken_forms", ()) or ()),
            ):
                if str(value or "").strip():
                    values.append(str(value))
        except Exception:
            continue
    return " ".join(values)


def _numeric_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for match in re.finditer(r"(?<![A-Za-z0-9])\d+(?:[:.]\d+)?(?![A-Za-z0-9])", text or ""):
        raw = match.group(0)
        folded = raw.replace(".", ":")
        out.add(folded)
        for part in re.split(r"[:.]", raw):
            if part:
                out.add(part)
    return out


def _token_values(text: str) -> list[str]:
    return [m.group(0).casefold() for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or "")]


def _phrase_tokens_present(text: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    haystack = _token_values(text)
    needle = tuple(tokens)
    width = len(needle)
    return any(tuple(haystack[idx : idx + width]) == needle for idx in range(0, max(0, len(haystack) - width + 1)))


def _duplicated_ngram(source_tokens: list[str], output_tokens: list[str], inserted_tokens: list[str]) -> str | None:
    max_width = min(6, len(inserted_tokens))
    for width in range(max_width, 2, -1):
        for idx in range(0, len(inserted_tokens) - width + 1):
            ngram = tuple(inserted_tokens[idx : idx + width])
            if not _meaningful_ngram(ngram):
                continue
            source_count = _ngram_count(source_tokens, ngram)
            if source_count <= 0:
                continue
            if _ngram_count(output_tokens, ngram) > source_count:
                return " ".join(ngram)
    return None


def _meaningful_ngram(tokens: tuple[str, ...]) -> bool:
    return any(len(tok) >= 3 and tok not in _DRIFT_STOPWORDS for tok in tokens)


def _meaningful_inserted_token(token: str) -> bool:
    folded = (token or "").casefold()
    if len(folded) < 3:
        return False
    return folded not in _LOW_SIGNAL_INSERTION_ALLOWLIST


_LOW_SIGNAL_INSERTION_ALLOWLIST = frozenset({"a", "an", "the", "and", "or", "to", "of", "in", "on", "for"})


def _drift_word_tokens(text: str) -> list[str]:
    return [tok.casefold() for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text or "")]


def _is_leading_prefix(source: str, output: str) -> bool:
    """Return True when output is a strict leading word-prefix of source."""
    src_tokens = _drift_word_tokens(source)
    out_tokens = _drift_word_tokens(output)
    if not out_tokens or len(out_tokens) >= len(src_tokens):
        return False
    return src_tokens[: len(out_tokens)] == out_tokens


def _ngram_count(tokens: list[str], ngram: tuple[str, ...]) -> int:
    width = len(ngram)
    if width <= 0:
        return 0
    return sum(1 for idx in range(0, len(tokens) - width + 1) if tuple(tokens[idx : idx + width]) == ngram)


def _short_reason(text: str, *, limit: int = 32) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def semantic_drift_safe(packet: TranscriptAdjudicationPacket, corrected_text: str) -> tuple[bool, str]:
    source = str(getattr(packet, "memory_candidate_text", "") or getattr(packet, "whisper_text", "") or "")
    src_words = max(1, len(source.split()))
    out_words = len((corrected_text or "").split())
    if out_words == 0:
        return False, "empty_output"
    if src_words >= 4:
        ratio = out_words / src_words
        if ratio > 1.8:
            return False, "large_unexplained_size_change"
        if ratio < 0.45 and _is_leading_prefix(source, corrected_text):
            # A large shrink is only unexplained when it has the shape of a
            # HUD/base_visible truncation: the output is a leading prefix of
            # the source and the tail was cut. Spoken self-corrections shrink
            # by dropping earlier retracted words while keeping the corrected
            # tail, so they fall through to the content-overlap invariant.
            return False, "large_unexplained_size_change"
    # General content-word overlap invariant. Applies at all source lengths.
    # Three cases:
    #   1. Source and output both have content words → require overlap.
    #   2. Source has content words, output dropped them all → reject.
    #   3. Source has only stopwords ("no", "yes", "ok"), output invented
    #      content words → reject. The output must mirror the source's shape.
    src_content = _content_words(source)
    out_content = _content_words(corrected_text)
    if src_content and out_content:
        overlap = src_content & out_content
        min_required = max(1, int(round(min(len(src_content), len(out_content)) * 0.3)))
        if len(overlap) < min_required:
            return False, "content_word_overlap_below_floor"
    elif src_content and not out_content:
        return False, "content_words_dropped_entirely"
    elif not src_content and out_content:
        return False, "content_words_invented_from_stopword_source"
    return True, "ok"


_SELF_CORRECTION_RE = re.compile(
    r"\b(?:scratch|delete|remove|strike|ignore|replace|change)\s+that\b"
    r"|\bmake\s+that\b"
    r"|\bi\s+mean\b"
    r"|\bactually\s+(?:make|change|set|use|send|call|put|switch|move)\b"
    r"|\bno,?\s+(?:actually\s+)?(?:to|make|change|set|use|send|call|put|switch|move)\b",
    re.IGNORECASE,
)

_DROPPABLE_DISFLUENCIES = {
    "uh",
    "um",
    "er",
    "erm",
}


def source_words_preserved(packet: TranscriptAdjudicationPacket, corrected_text: str) -> tuple[bool, str]:
    if getattr(packet, "stage", "") != "final":
        return True, "ok"
    source = _primary_authoritative_transcript_text(packet)
    if not source.strip():
        return True, "ok"
    if _SELF_CORRECTION_RE.search(source) and not _is_leading_prefix(source, corrected_text):
        return True, "ok"
    source_tokens = _token_values(source)
    output_tokens = _token_values(corrected_text)
    if not source_tokens or not output_tokens:
        return True, "ok"
    matcher = difflib.SequenceMatcher(a=source_tokens, b=output_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        src_span = source_tokens[i1:i2]
        out_span = output_tokens[j1:j2]
        if tag == "insert":
            continue
        if src_span and all(tok in _DROPPABLE_DISFLUENCIES for tok in src_span):
            continue
        if tag == "replace" and out_span:
            if _collapsed_letter_acronym_equivalent(src_span, out_span):
                continue
            if len(out_span) >= len(src_span):
                continue
            if len(src_span) - len(out_span) <= 1 and _mostly_near_substitution(src_span, out_span):
                continue
        dropped = [tok for tok in src_span if tok not in set(out_span)]
        if dropped:
            return False, f"source_words_dropped:{_short_reason(' '.join(dropped))}"
    return True, "ok"


def _collapsed_letter_acronym_equivalent(source_tokens: list[str], output_tokens: list[str]) -> bool:
    if not source_tokens or not output_tokens:
        return False
    source_joined = "".join(source_tokens)
    output_joined = "".join(output_tokens)
    if len(source_joined) < 3 or source_joined != output_joined:
        return False
    source_is_letters = len(source_tokens) >= 2 and all(len(tok) == 1 and tok.isalpha() for tok in source_tokens)
    output_is_letters = len(output_tokens) >= 2 and all(len(tok) == 1 and tok.isalpha() for tok in output_tokens)
    return (source_is_letters and len(output_tokens) == 1) or (output_is_letters and len(source_tokens) == 1)


def _mostly_near_substitution(source_tokens: list[str], output_tokens: list[str]) -> bool:
    if not source_tokens or not output_tokens:
        return False
    matcher = difflib.SequenceMatcher(a=source_tokens, b=output_tokens, autojunk=False)
    return matcher.ratio() >= 0.72


def low_signal_mid_sentence_capitalization_safe(packet: TranscriptAdjudicationPacket, corrected_text: str) -> tuple[bool, str]:
    if getattr(packet, "stage", "") != "final":
        return True, "ok"
    source = _primary_authoritative_transcript_text(packet)
    source_fold = (source or "").casefold()
    protected = {
        str(t or "").strip().casefold()
        for t in (getattr(packet, "protected_terms", ()) or ())
        if str(t or "").strip()
    }
    for match in re.finditer(r"\b[A-Z][a-z]{1,}\b", corrected_text or ""):
        token = match.group(0)
        folded = token.casefold()
        if folded not in _LOW_SIGNAL_MID_SENTENCE_CAP_WORDS:
            continue
        if folded in protected:
            continue
        if folded not in source_fold:
            continue
        if _titlecase_common_word_allowed(corrected_text, match.start(), match.end(), folded):
            continue
        return False, f"low_signal_mid_sentence_capitalization:{token}"
    return True, "ok"


def _titlecase_common_word_allowed(text: str, start: int, end: int, folded: str) -> bool:
    prefix = text[:start].rstrip()
    if not prefix:
        return True
    if prefix.endswith((".", "!", "?", "\n")):
        return True
    suffix = text[end:]
    next_word = re.match(r"^\W*([A-Za-z0-9]+)", suffix)
    prev_word_match = re.search(r"([A-Za-z0-9]+)\W*$", prefix)
    prev = prev_word_match.group(1).casefold() if prev_word_match else ""
    nxt = next_word.group(1).casefold() if next_word else ""
    if folded == "may":
        if nxt.isdigit() or prev in {"in", "on", "by", "before", "after", "since", "until"}:
            return True
    return False


# Tokens of length <= 2 and the common-stopword set below are dropped before
# computing source/output overlap. This is deliberately a small, stable set
# (articles, pronouns, common verbs) — anything more aggressive would mask
# legitimate Qwen drift.
_DRIFT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have",
        "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "just", "me", "my", "no", "not", "now",
        "of", "on", "or", "our", "ours", "she", "so", "some", "such", "than",
        "that", "the", "their", "theirs", "them", "then", "there", "these",
        "they", "this", "those", "to", "too", "us", "was", "we", "were",
        "what", "when", "where", "which", "while", "who", "whom", "why",
        "will", "with", "would", "yes", "you", "your", "yours",
    }
)

_LOW_SIGNAL_MID_SENTENCE_CAP_WORDS = _DRIFT_STOPWORDS | frozenset(
    {
        "all",
        "also",
        "basically",
        "clearly",
        "every",
        "final",
        "first",
        "last",
        "like",
        "many",
        "may",
        "more",
        "one",
        "pro",
        "really",
        "show",
        "some",
        "thank",
        "whatever",
    }
)


def _content_words(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text or "")
    out: set[str] = set()
    for tok in tokens:
        if len(tok) < 3:
            continue
        folded = tok.casefold()
        if folded in _DRIFT_STOPWORDS:
            continue
        out.add(folded)
    return out


def no_assistant_artifacts(corrected_text: str) -> tuple[bool, str]:
    text = corrected_text or ""
    if not text.strip():
        return False, "empty_output"
    if _ASSISTANT_PREAMBLE_RE.search(text):
        return False, "assistant_preamble"
    if _MARKDOWN_FENCE_RE.search(text):
        return False, "markdown_fence"
    return True, "ok"


def no_formatting_in_transcript_stage(packet: TranscriptAdjudicationPacket, corrected_text: str) -> tuple[bool, str]:
    policy = str(getattr(packet, "transcript_policy", "") or "")
    if policy == "none":
        return True, "ok"
    if _HEADING_RE.search(corrected_text or ""):
        return False, "transcript_heading_formatting"
    if getattr(packet, "stage", "") == "live" and _BULLET_RE.search(corrected_text or ""):
        return False, "live_structural_formatting"
    return True, "ok"
