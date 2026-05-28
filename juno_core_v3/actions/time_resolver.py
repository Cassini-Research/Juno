"""Time clause resolver.

``dateparser`` gives us a usable timestamp for most natural-language clauses,
but on its own it fails three categories that Siri famously botches and that
we want to nail:

1. **Missing time** — "remind me in 4 days", "remind me Monday".
   dateparser may return midnight or "now's hour"; users almost never want
   either. We default to **9:00 AM local** and flag the result as inferred so
   the HUD can show an "edit time?" affordance without blocking.

2. **Past time-of-day** — "remind me at 5pm" when it's already 6 PM.
   Without intervention this would fire instantly (or never). We **roll
   forward by one day** and surface the inference; "9 PM tonight" → "9 PM
   tomorrow", labeled clearly so the user can correct in the chip.

3. **Past explicit date** — "remind me on March 1" after March 1.
   Treated as user error, kept verbatim, but flagged ``needs_confirmation``
   so the HUD warns before silently scheduling something in the past.

The resolver does **not** call the network or an LLM — keep it fast and
deterministic. ``timeparse.parse_when`` orchestrates resolver + optional LLM
fallback (Phase 7); see that module for the call sequence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any

from juno_core_v3.actions.contracts import ParsedTime

# Default time-of-day for date-only clauses ("Monday", "in 4 days").
# 9 AM is the strongest heuristic across every consumer reminder app I checked
# (Apple Reminders, Google Tasks, Todoist, Things). Configurable later via
# Settings → Voice Actions if users push back; not exposed yet to keep the
# surface area small.
_DEFAULT_HOUR = 9
_DEFAULT_MINUTE = 0

# Word-time defaults. The user's mental model is fuzzy — these are what most
# people mean colloquially and what the HUD chip will surface so they can
# correct before it fires.
_WORD_TIME_DEFAULTS: dict[str, tuple[int, int]] = {
    "morning": (9, 0),
    "noon": (12, 0),
    "afternoon": (14, 0),
    "evening": (18, 0),
    "tonight": (20, 0),
    "night": (20, 0),
    "midnight": (0, 0),
    "end of day": (17, 0),
    "eod": (17, 0),
    "cob": (17, 0),
    "close of business": (17, 0),
}

# Tokens that signal an explicit time-of-day in the user's clause. If none
# match, the clause is "date-only" and we default the hour. Order doesn't
# matter — alternation is matched left-to-right inside ``re.search``.
_TIME_TOKEN_RE = re.compile(
    r"""
    \b(?:
        \d{1,2}(?:(?::|\.)\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?) # "5pm", "10.30 am"
      | \d{1,2}:\d{2}                               # "17:30", "9:00"
      | (?:o['’]?clock)                             # "five o'clock"
      | morning | afternoon | evening | tonight | night
      | noon | midday | midnight
      | (?:end\s+of\s+day) | eod | cob
      | (?:close\s+of\s+business)
      | in\s+\d+\s+(?:minute|min|hour|hr|second|sec)s?  # "in 30 minutes" — precise relative
      | in\s+(?:half\s+an|half|a\s+quarter(?:\s+of\s+an)?|an?)\s+(?:hour|hr)  # "in half an hour"
    )\b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# Tokens that signal an explicit calendar date in the user's clause. When the
# parsed timestamp ends up in the past AND the clause has no explicit date
# (only a bare time-of-day), we roll forward a day. When it has an explicit
# date, we keep it but flag ``needs_confirmation``.
_DATE_TOKEN_RE = re.compile(
    r"""
    \b(?:
        today | tomorrow | tonight | yesterday | weekend
      | mon(?:day)? | tue(?:sday)? | wed(?:nesday)? | thu(?:rsday)?
      | fri(?:day)? | sat(?:urday)? | sun(?:day)?
      | jan(?:uary)? | feb(?:ruary)? | mar(?:ch)? | apr(?:il)? | may
      | jun(?:e)? | jul(?:y)? | aug(?:ust)? | sep(?:tember)? | sept
      | oct(?:ober)? | nov(?:ember)? | dec(?:ember)?
      | next | last | this | upcoming | coming | following
      | in\s+\d+\s+(?:minute|hour|day|week|month|year)s?
      | \d{1,2}(?:st|nd|rd|th)                 # ordinal day "14th", "1st"
      | \d{1,2}/\d{1,2}                        # "3/15"
      | \d{4}-\d{2}-\d{2}                      # ISO date
    )\b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class _ClauseShape:
    has_time_token: bool
    has_date_token: bool
    word_time_default: tuple[int, int] | None  # (hour, minute) for "morning" etc


def _shape_for(clause: str) -> _ClauseShape:
    text = clause.lower()
    word_default: tuple[int, int] | None = None
    for word, hm in _WORD_TIME_DEFAULTS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            word_default = hm
            break
    return _ClauseShape(
        has_time_token=bool(_TIME_TOKEN_RE.search(clause)),
        has_date_token=bool(_DATE_TOKEN_RE.search(clause)),
        word_time_default=word_default,
    )


def _now_aware(now: datetime | None) -> datetime:
    # Default to the *local* timezone, not UTC. The downstream Swift HUD
    # renders ISO strings as local wall-clock; if we anchor defaults in UTC
    # the user sees their 9 AM default land at 1 PM (in UTC+4) or 5 AM
    # (in UTC-4), depending on offset. Local-tz keeps "9 AM" meaning 9 AM.
    if now is None:
        return datetime.now().astimezone()
    if now.tzinfo is None:
        return now.astimezone()
    return now


def _ensure_aware(dt: datetime, ref_tz: Any) -> datetime:
    if dt.tzinfo is None:
        # ``ref_tz`` is the resolver's "now" tz, which is local by default
        # (see ``_now_aware``). Falling back to UTC here would re-introduce
        # the same offset bug for parsers that return naive datetimes.
        return dt.replace(tzinfo=ref_tz) if ref_tz is not None else dt.astimezone()
    return dt


def _format_local(dt: datetime) -> str:
    """Compact human time for inference notes — "Tue 9:00 AM"."""

    return dt.strftime("%a %-I:%M %p").strip()


def resolve_time(
    *,
    clause: str,
    parsed: datetime | None,
    parser_confidence: float,
    parser_source: str,
    now: datetime | None = None,
) -> ParsedTime | None:
    """Apply default-fill and roll-forward rules to a raw parser result.

    ``parsed`` is the datetime the underlying parser (typically dateparser)
    returned, or ``None`` if it failed. ``clause`` is the original user text
    so we can inspect it for time/date tokens.

    Returns ``None`` only if both ``parsed`` is ``None`` *and* the clause has
    no salvageable shape (no word-time and no date token). Otherwise we
    construct a best-effort :class:`ParsedTime` with appropriate flags.
    """

    if not clause or not clause.strip():
        return None
    cleaned = clause.strip().rstrip(".,;:!?")
    shape = _shape_for(cleaned)
    now_dt = _now_aware(now)

    # Branch A: parser failed, but we have enough shape to construct something.
    if parsed is None:
        # If clause has neither date nor time signals, give up — caller can
        # fall back to LLM (Phase 7) or treat as a no-when reminder.
        if not shape.has_date_token and not shape.word_time_default:
            return None
        # Date-only: tomorrow at 9 AM (PREFER_DATES_FROM=future intent).
        base_date = now_dt.date() + timedelta(days=1)
        hour, minute = shape.word_time_default or (_DEFAULT_HOUR, _DEFAULT_MINUTE)
        candidate = datetime.combine(base_date, time(hour, minute), tzinfo=now_dt.tzinfo)
        note = (
            "no time specified — defaulted to 9:00 AM"
            if shape.word_time_default is None
            else f"interpreted as {_format_local(candidate)}"
        )
        return ParsedTime(
            iso=candidate.isoformat(),
            confidence=0.55,
            source="default",
            inferred=True,
            inference_note=note,
            needs_confirmation=True,
        )

    parsed_aware = _ensure_aware(parsed, now_dt.tzinfo)
    iso = parsed_aware.isoformat()
    inferred = False
    inference_note: str | None = None
    needs_confirmation = False
    confidence = max(0.0, min(1.0, parser_confidence))

    # Rule 1: date-only clause — override hour to default. dateparser will
    # often have used "now's hour" or 00:00, neither of which matches user
    # intent for "remind me Monday".
    if not shape.has_time_token:
        defaulted = parsed_aware.replace(
            hour=_DEFAULT_HOUR, minute=_DEFAULT_MINUTE, second=0, microsecond=0
        )
        # Push to next day if defaulted moment has already passed today.
        if defaulted <= now_dt and defaulted.date() == now_dt.date():
            defaulted = defaulted + timedelta(days=1)
        parsed_aware = defaulted
        iso = parsed_aware.isoformat()
        inferred = True
        inference_note = "no time specified — defaulted to 9:00 AM"
        confidence = min(confidence, 0.7)

    # Rule 2: past time-of-day with no explicit date → roll forward 1 day.
    # "at 5pm" when now is 6pm should fire tomorrow at 5pm, not instantly.
    elif parsed_aware <= now_dt and not shape.has_date_token:
        rolled = parsed_aware + timedelta(days=1)
        parsed_aware = rolled
        iso = parsed_aware.isoformat()
        inferred = True
        inference_note = (
            f"already passed today — rolled to {_format_local(parsed_aware)}"
        )
        confidence = min(confidence, 0.65)

    # Rule 3: past timestamp with explicit date → keep, flag for confirm.
    elif parsed_aware <= now_dt and shape.has_date_token:
        needs_confirmation = True
        inference_note = "this time is in the past — confirm before scheduling"
        confidence = min(confidence, 0.5)

    return ParsedTime(
        iso=iso,
        confidence=confidence,
        source=parser_source,
        inferred=inferred,
        inference_note=inference_note,
        needs_confirmation=needs_confirmation,
    )


__all__ = ["resolve_time"]
