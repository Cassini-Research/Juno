"""Action grammar parser.

Deterministic, regex-driven extraction of action verbs from a normalized
ASR transcript. The grammar lives in three layers:

1. A wake-word strip ("juno" or "hey juno"). If no wake-word is present,
   ``parse_actions`` returns ``None``
   and the caller falls back to today's INSERT/TRANSFORM pipeline.
2. A verb-prefix scan that locates every ``NOTE_VERB`` and ``REMIND_VERB``
   occurrence in the post-wake text. Each verb anchors an action.
3. Per-action body extraction: text between consecutive verb anchors becomes
   the body of the earlier action, with leading connectives ("to", "that",
   "about", ":") and trailing separators ("and", ",", "also") stripped.

For reminders, the parser additionally extracts a trailing ``WHEN`` clause
("tomorrow at 9am", "in 2 hours", "on May 15"). The clause text is removed
from the body and handed to :func:`juno_core_v3.actions.timeparse.parse_when`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from juno_core_v3.actions.contracts import Action, ActionKind, ParsedTime
from juno_core_v3.actions.timeparse import parse_when

# ---------------------------------------------------------------------------
# Wake-word grammar
# ---------------------------------------------------------------------------

# Matches the only supported leading wake phrases: "Juno" and "Hey Juno".
# Anchored to the start so mid-sentence product mentions remain dictation.
_WAKE_RE = re.compile(
    r"^\s*(?:hey[\s,.!?:;\-—]+)?juno\b[\s,.!?:;\-—]*",
    re.IGNORECASE,
)


def strip_wake(transcript: str) -> str | None:
    """Public alias for :func:`_strip_wake`.

    The pipeline hook needs to know whether the wake phrase is present
    *independently* of whether any action verb matched, so the LLM
    fallback can be invoked on wake-prefixed utterances that the
    deterministic grammar declines.
    """

    return _strip_wake(transcript)


def _strip_wake(transcript: str) -> str | None:
    """Remove the wake-word prefix. Return ``None`` when absent.

    The match is anchored at the start of the transcript (after stripping
    leading whitespace/punctuation) so dictation that *contains* "juno"
    later in the sentence is left untouched.
    """

    if not transcript:
        return None
    text = transcript.lstrip(" \t\n.,!?:;-—\"'(")
    m = _WAKE_RE.match(text)
    if m is None:
        return None
    post_wake = text[m.end():].strip()
    if _looks_like_product_version_mention(post_wake):
        return None
    return post_wake


def _looks_like_product_version_mention(post_wake: str) -> bool:
    return re.match(r"^(?:v\s*2|v\s*two)\b", post_wake or "", re.IGNORECASE) is not None


# ---------------------------------------------------------------------------
# Verb grammar
# ---------------------------------------------------------------------------

_NOTE_VERBS = (
    "add a note",
    "add note",
    "take a note",
    "take note",
    "make a note",
    "save this note",
    "save a note",
    "save this",
    "note this",
    "note that",
    "jot this down",
    "jot down",
    "write this down",
    "write that down",
)

_REMIND_VERBS = (
    "set a reminder",
    "remind me",
    "remember to",
    "remember that",
    "reminder",
)

# Alarm verbs are separate from reminders so the executor can route them
# to the EventKit-event sink (Calendar alert) rather than the Reminders
# sink. "Set a timer for" is intentionally aliased to alarm — macOS has
# no first-class voice timers, and a one-shot calendar alert at +N
# minutes is the closest sane equivalent.
_ALARM_VERBS = (
    "set an alarm",
    "set alarm",
    "wake me up",
    "wake me",
    "set a timer for",
    "set a timer",
    "ping me at",
    "ping me",
    "alarm at",
    "alarm",
)

_VERB_TABLE: tuple[tuple[str, ActionKind], ...] = tuple(
    [(v, ActionKind.NOTE) for v in _NOTE_VERBS]
    + [(v, ActionKind.REMINDER) for v in _REMIND_VERBS]
    + [(v, ActionKind.ALARM) for v in _ALARM_VERBS]
)

# Build a single alternation, longest-first so "save this note" wins over
# "save this".
_VERB_PATTERN = "|".join(
    re.escape(v) for v, _ in sorted(_VERB_TABLE, key=lambda kv: -len(kv[0]))
)
_VERB_RE = re.compile(rf"\b(?:{_VERB_PATTERN})\b", re.IGNORECASE)
_VERB_LOOKUP: dict[str, ActionKind] = {v.lower(): k for v, k in _VERB_TABLE}

@dataclass(frozen=True, slots=True)
class _VerbHit:
    start: int
    end: int
    kind: ActionKind
    text: str


def _find_verbs(text: str) -> list[_VerbHit]:
    hits: list[_VerbHit] = []
    for m in _VERB_RE.finditer(text):
        kind = _VERB_LOOKUP[m.group(0).lower()]
        hits.append(_VerbHit(m.start(), m.end(), kind, m.group(0)))
    return hits


# ---------------------------------------------------------------------------
# Body cleanup
# ---------------------------------------------------------------------------

_LEADING_CONNECTIVE_RE = re.compile(
    r"^[\s,.:;\-—]*(?:is(?:\s+to)?|to|that|about|of|on|saying(?:\s+that)?)\b[\s,.:;\-—]*",
    re.IGNORECASE,
)
_LEADING_PUNCT_RE = re.compile(r"^[\s,.:;\-—]+")
_TRAILING_SEP_RE = re.compile(
    r"[\s,.:;\-—]*\b(?:and|also|then|plus|next)(?:\s+please)?\b[\s,.:;\-—]*$",
    re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[\s,.:;\-—]+$")


def _strip_body(body: str, *, has_next_action: bool) -> str:
    body = _LEADING_PUNCT_RE.sub("", body)
    body = _LEADING_CONNECTIVE_RE.sub("", body)
    if has_next_action:
        body = _TRAILING_SEP_RE.sub("", body)
    body = _TRAILING_PUNCT_RE.sub("", body)
    return body.strip()


# ---------------------------------------------------------------------------
# Time clause extraction (reminder-only)
# ---------------------------------------------------------------------------

# Ordered list of trailing-clause patterns. Each must match at end-of-string
# and capture the entire time clause in group "when". Order matters: longer
# / more specific patterns first.
_WHEN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # "at 6 p.m. today", "6pm today", "at 11pm tomorrow"
        r"\b(?P<when>(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)\s+(?:today|tomorrow|tonight))\s*$",
        # "tomorrow at 9am", "today at noon", "tonight at 8"
        r"\b(?P<when>(?:tomorrow|today|tonight|this\s+(?:morning|afternoon|evening|night))(?:\s+at\s+[\w:.\s]+?)?)\s*$",
        # "next Tuesday at 3pm", "next week", "next month"
        r"\b(?P<when>next\s+\w+(?:\s+at\s+[\w:.\s]+?)?)\s*$",
        # "on May 15", "on Friday", "on the 15th"
        r"\b(?P<when>on\s+(?:the\s+)?[A-Za-z0-9]+(?:\s+\d+(?:st|nd|rd|th)?)?(?:\s+at\s+[\w:.\s]+?)?)\s*$",
        # "in 2 hours", "in 30 minutes", "in a week"
        r"\b(?P<when>in\s+(?:a|an|\d+)\s+\w+)\s*$",
        # "at 9am", "at 17:30", "at noon"
        r"\b(?P<when>at\s+[\w:.\s]+?)\s*$",
        # bare "tomorrow", "tonight", etc.
        r"\b(?P<when>tomorrow|tonight|today)\s*$",
    )
)

_LEADING_WHEN_CONNECTOR_RE = re.compile(r"\s+\b(?:to|for)\b\s+", re.IGNORECASE)
_DURATION_ONLY_RE = re.compile(
    r"^(?:a|an|\d+)\s+(?:second|seconds|minute|minutes|hour|hours|day|days|week|weeks)\b",
    re.IGNORECASE,
)
_ALARM_TIME_ONLY_RE = re.compile(
    r"^(?:"
    r"(?:at\s+)?(?:\d{1,2}(?::\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?|noon|midnight)"
    r"(?:\s+(?:today|tomorrow|tonight))?"
    r"|(?:today|tomorrow|tonight)(?:\s+at\s+[\w:.\s]+)?"
    r"|in\s+(?:a|an|\d+)\s+\w+"
    r"|(?:a|an|\d+)\s+(?:second|seconds|minute|minutes|hour|hours|day|days|week|weeks)"
    r")\s*$",
    re.IGNORECASE,
)
_TIME_SELF_CORRECTION_RE = re.compile(
    r"\b(?:at\s+)?(?:\d{1,2}(?::\d{2})?|\w+)\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?"
    r"\s*(?:no\s+wait|no\s+no|actually|my\s+bad|make\s+it)\s+"
    r"(?P<latest>(?:at\s+)?(?:\d{1,2}(?::\d{2})?|\w+)\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?)\b",
    re.IGNORECASE,
)


def _time_clause_candidates(
    body: str,
    *,
    kind: ActionKind,
    verb_text: str,
) -> list[str]:
    """Return plausible time-clause candidates for reminder/alarm bodies.

    Some verb aliases intentionally absorb the leading preposition
    (``alarm at``, ``set a timer for``). These helpers reconstruct the
    missing shape so ``parse_when`` sees the clause the user actually
    spoke.
    """

    cleaned = body.strip()
    if not cleaned:
        return []

    candidates: list[str] = []
    lowered = cleaned.lower()

    verb_lower = verb_text.lower()
    if kind == ActionKind.ALARM:
        if verb_lower.endswith(" at") and not lowered.startswith("at "):
            candidates.append(f"at {cleaned}")
        if _DURATION_ONLY_RE.match(cleaned) and not lowered.startswith("in "):
            candidates.append(f"in {cleaned}")
    if lowered.startswith("for "):
        stripped = cleaned[4:].strip()
        if stripped:
            candidates.append(stripped)
    candidates.append(cleaned)

    # Preserve order, remove duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _extract_leading_when(
    body: str,
    *,
    kind: ActionKind,
    verb_text: str,
    now: datetime | None = None,
) -> tuple[str, ParsedTime | None]:
    """Parse front-loaded time clauses like ``at 5pm to leave``.

    Reminder and alarm phrasings frequently put the time clause before the
    action body. The advertised Juno examples do this as well, so the
    parser needs to support both:

    - ``remind me at 5pm to leave for the store``
    - ``wake me up in 25 minutes``
    - ``alarm at 3:30 to leave for the airport``
    """

    for candidate in _time_clause_candidates(body, kind=kind, verb_text=verb_text):
        match = _LEADING_WHEN_CONNECTOR_RE.search(candidate)
        if match is not None:
            clause = candidate[: match.start()].strip(" \t,.;:—-")
            remainder = candidate[match.end() :].strip(" \t,.;:—-")
            parsed = parse_when(clause, now=now)
            if parsed is not None:
                return remainder, parsed

        # Whole-candidate parse: only useful for alarms where an empty
        # body is acceptable (the executor labels them "Alarm"). For
        # reminders we must NOT swallow the body — dateparser is
        # generous enough to match "call sam tomorrow at 9am" as a
        # whole, which would erase the body and drop the action.
        if kind == ActionKind.ALARM and _ALARM_TIME_ONLY_RE.match(candidate):
            parsed = parse_when(candidate, now=now)
            if parsed is not None:
                return "", parsed

    return body, None


def _extract_when(body: str, now: datetime | None = None) -> tuple[str, ParsedTime | None]:
    """Pull a trailing time clause off ``body`` if present.

    Returns ``(remaining_body, parsed_time_or_None)``. If a clause is matched
    syntactically but the time parser fails, the clause is left attached to
    the body and ``None`` is returned — Phase 7 (LLM fallback) can rescue it
    later, and in the meantime the user sees the full text in the reminder.
    """

    def _latest_time(match: re.Match[str]) -> str:
        latest = match.group("latest").strip()
        return latest if latest.lower().startswith("at ") else f"at {latest}"

    body = _TIME_SELF_CORRECTION_RE.sub(_latest_time, body)
    for pattern in _WHEN_PATTERNS:
        m = pattern.search(body)
        if m is None:
            continue
        clause = m.group("when").strip()
        parsed = parse_when(clause, now=now)
        if parsed is None:
            continue
        remaining = body[: m.start()].rstrip(" \t,.;:—-")
        return remaining, parsed
    return body, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_actions(
    transcript: str,
    *,
    now: datetime | None = None,
) -> list[Action] | None:
    """Parse a transcript into a list of :class:`Action`.

    Returns ``None`` when the transcript is not addressed to Juno (no
    wake-word) or when no action verb is present in the post-wake text.
    Returns an empty list only in the unlikely case that every detected
    verb produced an empty body — callers should treat that as ``None``.
    """

    if not transcript or not transcript.strip():
        return None

    body_text = _strip_wake(transcript)
    if body_text is None:
        return None

    hits = _find_verbs(body_text)
    hits.sort(key=lambda hit: (hit.start, hit.end))
    if not hits:
        return None

    actions: list[Action] = []
    for idx, hit in enumerate(hits):
        next_start = hits[idx + 1].start if idx + 1 < len(hits) else len(body_text)
        body = body_text[hit.end : next_start]
        cleaned = _strip_body(body, has_next_action=idx + 1 < len(hits))

        when: ParsedTime | None = None
        if hit.kind in (ActionKind.REMINDER, ActionKind.ALARM) and cleaned:
            cleaned, when = _extract_leading_when(
                cleaned,
                kind=hit.kind,
                verb_text=hit.text,
                now=now,
            )
            if when is None:
                cleaned, when = _extract_when(cleaned, now=now)

        if hit.kind == ActionKind.ALARM and when is not None and not cleaned:
            cleaned = "Alarm"

        if not cleaned:
            continue

        raw_span = body_text[hit.start : next_start].strip(" \t,.;:—-")
        actions.append(
            Action(
                kind=hit.kind,
                body=cleaned,
                raw_span=raw_span,
                when=when,
            )
        )

    return actions or None
