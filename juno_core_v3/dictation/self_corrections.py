"""Deterministic application of unambiguous spoken retakes.

The model lanes (adjudicator / writer) are told about self-correction cues
and may apply them with full meaning-level judgment — but those lanes fall
back or pass through on a large share of real utterances, which shipped
literal "Scratch that" into pasted text (production 2026-06-10).

This module applies only the *unambiguous* subset deterministically: a
retake, where the phrase spoken after the marker clearly re-speaks the
phrase spoken right before it ("after the first. Scratch that after the
last install." / "for 3pm scratch that 4pm"). Anything below the
similarity bar is left untouched, because "scratch that" can be literal
content ("the scratch-that wasn't caught properly").
"""
from __future__ import annotations

import difflib
import re
from typing import Any

# Spoken mid-utterance edit markers. Kept in one place; the dictation
# pipeline imports this for cue *detection* and this module uses it for
# retake *application*.
MID_UTTERANCE_EDIT_MARKER_RE = re.compile(
    r"(?<!\w)(?:(?:scratch|delete|remove)\s+that|no\s+(?:actually|wait)|nah\s+(?:actually|wait)|sorry\s+(?:actually|rather)|actually\s+no)(?!\w)[\s,.;:!?-]*",
    re.IGNORECASE,
)

_SENTENCE_FINAL_RE = re.compile(r"[.!?][\"')\]]*$")
_NUMERIC_PHRASE_RE = re.compile(r"[\d][\d:. ]*(?:am|pm)?")
_MAX_RETAKE_TOKENS = 12

_WEEKDAY_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
_DATE_FILLER_WORDS = {"on", "next", "this", "coming", "the", "of"}
_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}
# ASR renders the spoken cue "scratch that" as "scratched at" when it sits
# between two clock expressions ("at 3pm scratched at 4.15pm"). Only that
# temporal sandwich is rewritten — prose like "the paint scratched at the
# edge" never matches.
_SCRATCHED_AT_TEMPORAL_RE = re.compile(
    r"(?P<prep>\bat\s+)?(?P<old>\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)?)"
    r"[,\s]+scratched\s+at\s+"
    r"(?P<new>\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)?)",
    re.IGNORECASE,
)


def _slot_class(key: str) -> str | None:
    tokens = [t for t in key.split() if t not in _DATE_FILLER_WORDS]
    if not tokens or len(tokens) > 3:
        return None
    if all(t in _WEEKDAY_WORDS for t in tokens):
        return "weekday"
    if all(t in _MONTH_WORDS or re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", t) for t in tokens) and any(
        t in _MONTH_WORDS or re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)", t) for t in tokens
    ):
        return "date"
    return None


def _norm_key(tokens: list[str]) -> str:
    out = []
    for tok in tokens:
        cleaned = re.sub(r"[^\w']+", "", tok).casefold()
        if cleaned:
            out.append(cleaned)
    return " ".join(out)


def _first_norm(tokens: list[str]) -> str:
    key = _norm_key(tokens[:1])
    return key


def apply_unambiguous_retakes(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Apply high-confidence retakes; return (new_text, applied_records).

    Conservative by construction: a marker is applied only when the
    after-phrase resembles the before-phrase (shared opening token or
    strong sequence similarity). Markers that do not look like retakes are
    preserved verbatim for the model lanes / the reader.
    """
    source = str(text or "")
    if not source.strip():
        return source, []

    applied: list[dict[str, Any]] = []

    def _scratched_at(match: re.Match[str]) -> str:
        applied.append(
            {
                "marker": "scratched at",
                "removed": match.group("old").strip(),
                "kept_preview": match.group("new").strip(),
                "ratio": 1.0,
            }
        )
        prep = match.group("prep") or ""
        return f"{prep}{match.group('new')}"

    source = _SCRATCHED_AT_TEMPORAL_RE.sub(_scratched_at, source)

    # Right-to-left so earlier char offsets stay valid after each splice.
    matches = list(MID_UTTERANCE_EDIT_MARKER_RE.finditer(source))
    for match in reversed(matches):
        tokens = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", source)]
        marker_start, marker_end = match.start(), match.end()
        before = [t for t in tokens if t[1] <= marker_start]
        after = [t for t in tokens if t[0] >= marker_end]
        if not before or not after:
            continue

        # After-phrase: capped at the first sentence end or the next marker.
        next_marker = MID_UTTERANCE_EDIT_MARKER_RE.search(source, marker_end)
        after_phrase: list[tuple[int, int, str]] = []
        for tok in after[:_MAX_RETAKE_TOKENS]:
            if next_marker is not None and tok[1] > next_marker.start():
                break
            after_phrase.append(tok)
            # Sentence or clause boundary ends the retake phrase ("…actually
            # 7, to call mom" — only "7" replaces, the rest continues).
            if _SENTENCE_FINAL_RE.search(tok[2]) or tok[2].endswith((",", ";")):
                break
        if not after_phrase:
            continue
        after_tokens = [t[2] for t in after_phrase]
        after_key = _norm_key(after_tokens)
        if not after_key:
            continue

        best: tuple[float, int] | None = None  # (ratio, span_len)
        # Same-slot replacements ("Friday → Monday", "the 5th → the 12th")
        # share no characters; match by slot class with token-count symmetry
        # so articles/prepositions stay balanced on both sides.
        after_slots: dict[int, str] = {}
        for prefix_len in range(1, min(4, len(after_tokens)) + 1):
            slot = _slot_class(_norm_key(after_tokens[:prefix_len]))
            if slot:
                after_slots[prefix_len] = slot
        slot_pick: tuple[int, int] | None = None
        if after_slots:
            for span_len in range(1, min(4, len(before)) + 1):
                cand_slot = _slot_class(_norm_key([t[2] for t in before[-span_len:]]))
                if cand_slot is None:
                    continue
                for prefix_len, slot in sorted(after_slots.items()):
                    if slot != cand_slot:
                        continue
                    if span_len == prefix_len:
                        slot_pick = (span_len, prefix_len)
                        break
                    if slot_pick is None:
                        slot_pick = (span_len, prefix_len)
                if slot_pick is not None and slot_pick[0] == slot_pick[1]:
                    break
        if slot_pick is not None:
            best = (1.0, slot_pick[0])
        target_len = len(after_tokens)
        for span_len in range(max(1, target_len - 2), min(len(before), target_len + 2) + 1):
            cand_tokens = [t[2] for t in before[-span_len:]]
            cand_key = _norm_key(cand_tokens)
            if not cand_key:
                continue
            ratio = difflib.SequenceMatcher(a=cand_key, b=after_key, autojunk=False).ratio()
            shares_opening = _first_norm(cand_tokens) == _first_norm(after_tokens)
            # Short same-opener retakes ("to Bob → to Alice", "3pm → 4pm")
            # carry little overlap mass, so the opener match substitutes for
            # raw similarity when both phrases are brief.
            short_retake = (
                shares_opening
                and len(cand_tokens) <= 4
                and len(after_tokens) <= 4
            )
            # A number replacing a number at a correction marker is the
            # canonical spoken time fix ("at 6, no actually 7").
            numeric_retake = bool(
                _NUMERIC_PHRASE_RE.fullmatch(cand_key)
                and _NUMERIC_PHRASE_RE.fullmatch(after_key)
            )
            if ratio >= 0.6 or (shares_opening and ratio >= 0.45) or short_retake or numeric_retake:
                if best is None or ratio > best[0]:
                    best = (ratio, span_len)
        if best is None:
            continue

        span_len = best[1]
        removed_start = before[-span_len][0]
        removed_text = source[removed_start:marker_end]
        source = source[:removed_start] + source[marker_end:]
        applied.append(
            {
                "marker": match.group(0).strip(" ,.;:!?-"),
                "removed": removed_text.strip()[:160],
                "kept_preview": " ".join(after_tokens)[:120],
                "ratio": round(best[0], 3),
            }
        )

    applied.reverse()
    return source, applied
