"""Natural-language time clause parser.

Wraps the third-party ``dateparser`` library with project-specific defaults
(future-leaning interpretation, configurable "now"). The raw parser result
is then handed to :func:`time_resolver.resolve_time` which applies
default-fill (9 AM for date-only clauses) and roll-forward (tomorrow when
the requested hour has already passed). See ``time_resolver.py`` for the
full ruleset and motivation.

Phase 7 — **LLM fallback** — is wired here as an optional callable that the
shell can register at runtime via :func:`set_llm_fallback`. When dateparser
declines to produce a result we'll route the clause to the LLM, which is
expected to return ``(iso, confidence)`` or ``None``. The fallback runs
inside a try/except so a misbehaving LLM can never break dictation.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, time, timedelta
from typing import Any, Callable

from juno_core_v3.actions.contracts import ParsedTime
from juno_core_v3.actions.time_resolver import resolve_time

logger = logging.getLogger(__name__)

_RELATIVE_DAY_RE = re.compile(
    r"""
    ^(?P<day>today|tomorrow|day\s+after\s+tomorrow|tonight)
    (?:\s+(?:at\s+)?
        (?P<clock>\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?)
    )?
    (?:\s+(?:in\s+the\s+)?(?P<period>morning|afternoon|evening|night))?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
_CLOCK_RELATIVE_DAY_RE = re.compile(
    r"""
    ^(?P<clock>\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?)
    \s+
    (?P<day>day\s+after\s+tomorrow|tomorrow|tonight|today)
    (?:\s+(?:in\s+the\s+)?(?P<period>morning|afternoon|evening|night))?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
_CLOCK_RE = re.compile(
    r"^(?P<hour>\d{1,2})(?:(?::|\.)(?P<minute>\d{2}))?\s*(?P<ampm>a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?$",
    re.IGNORECASE,
)
_BARE_CLOCK_RE = re.compile(
    r"^(?:at\s+)?(?P<clock>\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?)$",
    re.IGNORECASE,
)

# Speech-shaped patterns dateparser declines on. Each is paired with a
# resolver function that returns a ``datetime`` (or None). Order matters:
# more specific patterns must come first so we don't shadow them with a
# loose weekday match. All patterns expect a fully-cleaned, lower-cased,
# trimmed clause and anchor with ^/$ so they don't fire mid-sentence —
# salvage handles in-sentence extraction separately.
_WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

_PERIOD_HOURS = {
    "morning": time(9, 0),
    "afternoon": time(14, 0),
    "evening": time(18, 0),
    "night": time(20, 0),
    "noon": time(12, 0),
    "midnight": time(0, 0),
}

# "(this|next|coming|upcoming|on)? <weekday> (at TIME)? ((in the )?<period>)?"
# dateparser stumbles on "next thursday", "this friday", "monday morning".
_WEEKDAY_RE = re.compile(
    r"""
    ^
    (?:(?P<mod>this|next|coming|upcoming|on|on\s+the|the)\s+)?
    (?P<weekday>mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday|s)?|thu(?:r(?:s(?:day)?)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)
    (?:\s+(?:at\s+)?(?P<clock>\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?))?
    (?:\s+(?:in\s+the\s+)?(?P<period>morning|afternoon|evening|night|noon))?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "tonight" / "tonight at 8" / "tonight at 8pm"
_TONIGHT_RE = re.compile(
    r"""
    ^tonight
    (?:\s+(?:at\s+)?(?P<clock>\d{1,2}(?:(?::|\.)\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)?))?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "in half an hour", "in an hour", "in a quarter (of an) hour", "half an hour from now"
_FRACTIONAL_DURATION_RE = re.compile(
    r"""
    ^
    (?:in\s+)?
    (?P<qty>
        half\s+an
      | half
      | a\s+quarter\s+of\s+an
      | a\s+quarter
      | an
      | a
    )
    \s+
    (?P<unit>hour|hr)
    (?:\s+from\s+now)?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "this weekend" / "next weekend"
_WEEKEND_RE = re.compile(
    r"^(?P<mod>this|next|coming|upcoming)\s+weekend$",
    re.IGNORECASE,
)

# "next week" / "this week" — week-level
_WEEK_RE = re.compile(
    r"^(?P<mod>this|next|coming|upcoming|following)\s+week$",
    re.IGNORECASE,
)

# "first/start/beginning of (next|this) month", "end of (this|next) month"
_MONTH_BOUNDARY_RE = re.compile(
    r"""
    ^(?P<edge>first|start|beginning|end|last\s+day)
    \s+of\s+
    (?:the\s+)?
    (?P<mod>this|next|the\s+coming|coming|upcoming)?
    \s*month
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "next month" / "this month" — bare. Less common but Siri-supported.
_BARE_MONTH_RE = re.compile(
    r"^(?P<mod>this|next|coming|upcoming|following)\s+month$",
    re.IGNORECASE,
)

# "on the 14th", "the 14th", "14th" — bare ordinal day with no month.
# We resolve to the nearest future occurrence in current or next month.
_BARE_ORDINAL_DAY_RE = re.compile(
    r"^(?:on\s+the\s+|the\s+|on\s+)?(?P<day>\d{1,2})(?:st|nd|rd|th)$",
    re.IGNORECASE,
)

try:  # pragma: no cover - import resolved at runtime
    import dateparser as _dateparser  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - missing optional dep
    _dateparser = None


# Optional LLM fallback. Returns ``(iso_string, confidence)`` or ``None`` /
# raises. Registered by the shell at startup so the core stays free of LLM
# imports. Default is a no-op so unit tests remain hermetic.
LlmFallback = Callable[[str, datetime | None], tuple[str, float] | None]
_llm_fallback: LlmFallback | None = None


def set_llm_fallback(fn: LlmFallback | None) -> None:
    """Register (or clear) the LLM time-parse fallback.

    The callable is invoked only when dateparser returns ``None`` and the
    clause contains some date-like signal — never on plain dictation. Pass
    ``None`` to disable.
    """

    global _llm_fallback
    _llm_fallback = fn


def _settings(now: datetime | None) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "PREFER_DATES_FROM": "future",
        "RETURN_AS_TIMEZONE_AWARE": True,
    }
    if now is not None:
        settings["RELATIVE_BASE"] = now
    return settings


def _reference_now(now: datetime | None) -> datetime:
    """Return the reference moment in the *user's* timezone.

    When ``now`` is provided we keep it in its own tz — converting to the
    system tz here used to silently cross date boundaries (a user at
    22:45 UTC+4 would land at 00:15 UTC+5:30, shifting "tomorrow" by one
    day). When ``now`` is naïve we attach the system tz; when it's
    omitted we use the system tz directly.
    """

    if now is not None:
        if now.tzinfo is None:
            return now.astimezone()
        return now
    return datetime.now().astimezone()


def _parse_clock(raw: str, *, period: str | None = None) -> time | None:
    match = _CLOCK_RE.match(raw.strip())
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or "0")
    if hour > 23 or minute > 59:
        return None
    ampm = re.sub(r"[^apm]", "", (match.group("ampm") or "").lower())
    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm.startswith("p") and hour != 12:
            hour += 12
        elif ampm.startswith("a") and hour == 12:
            hour = 0
    elif period:
        bucket = period.strip().lower()
        if bucket in {"afternoon", "evening", "night"} and 1 <= hour < 12:
            hour += 12
        elif bucket == "morning" and hour == 12:
            hour = 0
    return time(hour, minute)


def _weekday_idx(name: str) -> int | None:
    return _WEEKDAYS.get(name.strip().lower())


def _next_weekday(ref: datetime, target_idx: int, *, modifier: str | None) -> datetime:
    """Return the date of the next occurrence of *target_idx* (Mon=0)
    relative to *ref*.

    Modifier semantics, calibrated to common spoken usage (Siri/Google):

    - ``"next"`` → calendar week's weekday (today + 1..14, the *upcoming*
      occurrence; matches what most people mean by "next monday" even
      though dictionaries split hairs). If today *is* that weekday, jump
      a full week to avoid scheduling something an hour from now.
    - ``"this"`` / ``"coming"`` / ``"upcoming"`` → nearest future
      occurrence in 1..7 days. Same-day weekday → 7 days out.
    - ``"on"`` / unmodified → 0..6 days; same-day allowed (the resolver
      will roll forward later if the time of day has passed).
    """

    delta = (target_idx - ref.weekday()) % 7
    mod = (modifier or "").strip().lower().replace("on the", "on")
    if mod == "next":
        # "next monday" — at least 1 day away, prefer the closer one.
        # Most people actually mean the next occurrence; bump only when
        # today *is* that weekday.
        if delta == 0:
            delta = 7
    elif mod in {"this", "coming", "upcoming"}:
        if delta == 0:
            delta = 7
    # "on monday" / bare "monday" → 0 is fine; resolver handles roll-fwd.
    return ref + timedelta(days=delta)


def _try_weekday(clause: str, now: datetime | None) -> datetime | None:
    match = _WEEKDAY_RE.match(clause)
    if match is None:
        return None
    idx = _weekday_idx(match.group("weekday"))
    if idx is None:
        return None
    ref = _reference_now(now)
    target_date = _next_weekday(ref, idx, modifier=match.group("mod"))
    raw_clock = match.group("clock")
    period = match.group("period")
    if raw_clock:
        parsed_time = _parse_clock(raw_clock, period=period)
        if parsed_time is None:
            return None
    elif period:
        parsed_time = _PERIOD_HOURS.get(period.strip().lower())
        if parsed_time is None:
            return None
    else:
        # No time-of-day → midnight; resolver applies the 9 AM default
        # and flags inferred=True. Returning the bare date keeps the
        # explicit-date branch active so the resolver does not roll
        # forward on a past hour-of-day it never had.
        parsed_time = time(0, 0)
    return datetime.combine(target_date.date(), parsed_time, tzinfo=ref.tzinfo)


def _try_tonight(clause: str, now: datetime | None) -> datetime | None:
    match = _TONIGHT_RE.match(clause)
    if match is None:
        return None
    ref = _reference_now(now)
    raw_clock = match.group("clock")
    if raw_clock:
        parsed_time = _parse_clock(raw_clock, period="evening")
        if parsed_time is None:
            return None
    else:
        parsed_time = time(20, 0)  # canonical "tonight"
    return datetime.combine(ref.date(), parsed_time, tzinfo=ref.tzinfo)


def _try_fractional_duration(clause: str, now: datetime | None) -> datetime | None:
    match = _FRACTIONAL_DURATION_RE.match(clause)
    if match is None:
        return None
    qty = re.sub(r"\s+", " ", match.group("qty").strip().lower())
    minutes = {
        "half an": 30,
        "half": 30,
        "a quarter": 15,
        "a quarter of an": 15,
        "a": 60,
        "an": 60,
    }.get(qty)
    if minutes is None:
        return None
    ref = _reference_now(now)
    return ref + timedelta(minutes=minutes)


def _try_weekend(clause: str, now: datetime | None) -> datetime | None:
    match = _WEEKEND_RE.match(clause)
    if match is None:
        return None
    ref = _reference_now(now)
    # Saturday is the weekend anchor.
    target = _next_weekday(ref, 5, modifier="this")
    if match.group("mod").strip().lower() == "next":
        target = target + timedelta(days=7)
    return datetime.combine(target.date(), time(0, 0), tzinfo=ref.tzinfo)


def _try_week(clause: str, now: datetime | None) -> datetime | None:
    match = _WEEK_RE.match(clause)
    if match is None:
        return None
    ref = _reference_now(now)
    mod = match.group("mod").strip().lower()
    days = 7 if mod in {"next", "following"} else 1
    return datetime.combine((ref + timedelta(days=days)).date(), time(0, 0), tzinfo=ref.tzinfo)


def _add_months(d: datetime, months: int) -> datetime:
    month_idx = d.month - 1 + months
    year = d.year + month_idx // 12
    month = month_idx % 12 + 1
    # Clamp day to last valid day of target month.
    day = min(d.day, _last_day_of_month(year, month))
    return d.replace(year=year, month=month, day=day)


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    next_first = datetime(year, month + 1, 1)
    return (next_first - timedelta(days=1)).day


def _try_month_boundary(clause: str, now: datetime | None) -> datetime | None:
    match = _MONTH_BOUNDARY_RE.match(clause)
    if match is None:
        return None
    ref = _reference_now(now)
    edge = re.sub(r"\s+", " ", match.group("edge").strip().lower())
    mod = (match.group("mod") or "").strip().lower().replace("the coming", "next")
    months_ahead = 1 if mod in {"next", "coming", "upcoming"} else 0
    target = _add_months(ref, months_ahead)
    if edge in {"first", "start", "beginning"}:
        day = 1
    else:  # end / last day
        day = _last_day_of_month(target.year, target.month)
    candidate = target.replace(day=day, hour=0, minute=0, second=0, microsecond=0)
    # If "this month" but the chosen edge has already passed, bump to next.
    if months_ahead == 0 and candidate.date() < ref.date():
        bumped = _add_months(ref, 1)
        if edge in {"first", "start", "beginning"}:
            candidate = bumped.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            candidate = bumped.replace(
                day=_last_day_of_month(bumped.year, bumped.month),
                hour=0, minute=0, second=0, microsecond=0,
            )
    return candidate


def _try_bare_month(clause: str, now: datetime | None) -> datetime | None:
    match = _BARE_MONTH_RE.match(clause)
    if match is None:
        return None
    ref = _reference_now(now)
    mod = match.group("mod").strip().lower()
    months_ahead = 1 if mod in {"next", "coming", "upcoming", "following"} else 0
    target = _add_months(ref, months_ahead)
    return target.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _try_bare_ordinal_day(clause: str, now: datetime | None) -> datetime | None:
    match = _BARE_ORDINAL_DAY_RE.match(clause)
    if match is None:
        return None
    day = int(match.group("day"))
    if not (1 <= day <= 31):
        return None
    ref = _reference_now(now)
    # Try this month first; if day < today or invalid, jump to next month.
    try:
        candidate = ref.replace(day=day, hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        candidate = None
    if candidate is None or candidate.date() < ref.date():
        nxt = _add_months(ref, 1)
        last = _last_day_of_month(nxt.year, nxt.month)
        if day > last:
            return None  # e.g. "the 31st" when next month has 30 days
        candidate = nxt.replace(day=day, hour=0, minute=0, second=0, microsecond=0)
    return candidate


def _try_project_common(clause: str, now: datetime | None) -> datetime | None:
    """Parse high-value action phrases dateparser can miss.

    The production path has dateparser installed, but live action extraction
    needs a deterministic fallback for speech-shaped clauses such as
    "day after tomorrow 11 a.m. in the morning". Without this, the resolver's
    shape-only path sees "tomorrow" and schedules the wrong day/time.

    Order is deliberate: most-specific anchors first (today/tomorrow,
    tonight, fractional durations) before week/month-level patterns and
    finally bare weekdays. Bare clock times are the last fallback so a
    leading "5pm" with no date context still resolves to today.
    """

    cleaned = clause.strip().lower()
    if not cleaned:
        return None

    # 1. today / tomorrow / day after tomorrow / tonight (full clause)
    for handler in (
        _try_clock_relative_day,
        _try_relative_day,
        _try_tonight,
        _try_fractional_duration,
        _try_month_boundary,
        _try_weekend,
        _try_week,
        _try_bare_month,
        _try_weekday,
        _try_bare_ordinal_day,
        _try_bare_clock,
    ):
        out = handler(cleaned, now)
        if out is not None:
            return out
    return None


def _try_clock_relative_day(clause: str, now: datetime | None) -> datetime | None:
    match = _CLOCK_RELATIVE_DAY_RE.match(clause)
    if match is None:
        return None
    day = re.sub(r"\s+", " ", match.group("day").lower())
    offset = {
        "today": 0,
        "tomorrow": 1,
        "day after tomorrow": 2,
        "tonight": 0,
    }.get(day)
    if offset is None:
        return None
    period = match.group("period")
    period_hint = period or ("evening" if day == "tonight" else None)
    parsed_time = _parse_clock(match.group("clock"), period=period_hint)
    if parsed_time is None:
        return None
    ref = _reference_now(now)
    return datetime.combine(
        ref.date() + timedelta(days=offset),
        parsed_time,
        tzinfo=ref.tzinfo,
    )


def _try_relative_day(clause: str, now: datetime | None) -> datetime | None:
    match = _RELATIVE_DAY_RE.match(clause)
    if match is None:
        return None
    day = re.sub(r"\s+", " ", match.group("day").lower())
    offset = {
        "today": 0,
        "tomorrow": 1,
        "day after tomorrow": 2,
        "tonight": 0,
    }.get(day)
    if offset is None:
        return None
    ref = _reference_now(now)
    raw_clock = match.group("clock")
    period = match.group("period")
    if raw_clock:
        # "tonight at 8" should disambiguate to PM.
        period_hint = period or ("evening" if day == "tonight" else None)
        parsed_time = _parse_clock(raw_clock, period=period_hint)
        if parsed_time is None:
            return None
    elif period:
        parsed_time = _PERIOD_HOURS.get(period.strip().lower())
        if parsed_time is None:
            return None
    elif day == "tonight":
        parsed_time = time(20, 0)
    else:
        return None
    return datetime.combine(
        ref.date() + timedelta(days=offset),
        parsed_time,
        tzinfo=ref.tzinfo,
    )


def _try_bare_clock(clause: str, now: datetime | None) -> datetime | None:
    bare = _BARE_CLOCK_RE.match(clause)
    if bare is None:
        return None
    parsed_time = _parse_clock(bare.group("clock"))
    if parsed_time is None:
        return None
    ref = _reference_now(now)
    return datetime.combine(ref.date(), parsed_time, tzinfo=ref.tzinfo)


def _try_dateparser(clause: str, now: datetime | None) -> datetime | None:
    if _dateparser is None:
        return None
    try:
        parsed = _dateparser.parse(clause, settings=_settings(now))
    except Exception:  # noqa: BLE001 - third-party can raise anything
        logger.exception("dateparser raised for clause=%r", clause)
        return None
    return _anchor_to_reference_tz(parsed, now)


def _anchor_to_reference_tz(parsed: datetime | None, now: datetime | None) -> datetime | None:
    """Reinterpret dateparser's wall clock into the caller's timezone.

    dateparser computes wall-clock fields against ``RELATIVE_BASE`` (the
    broker ``now``, in the user's tz) but attaches the SYSTEM tz to the
    result. When the two differ, "Friday at 2pm" came back as 14:00 in the
    machine's zone — the right wall clock anchored to the wrong zone, i.e.
    an alarm firing hours off whenever broker time and system time diverge
    (the deterministic tier-0/tier-2 lanes already anchor via
    ``_reference_now``). The wall clock is what the speaker meant, so we
    reinterpret rather than convert. Only do this when the parsed tz equals
    the system default — an explicit zone in the clause ("2pm UTC") is
    dateparser telling us something, and we keep it.
    """
    if parsed is None or now is None or now.tzinfo is None:
        return parsed
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=now.tzinfo)
    system_offset = datetime.now().astimezone().utcoffset()
    if parsed.utcoffset() == now.utcoffset():
        return parsed
    if parsed.utcoffset() == system_offset:
        return parsed.replace(tzinfo=now.tzinfo)
    return parsed


# Salvage patterns: substrings the user almost certainly intended as the
# trigger. Each is tried left-to-right inside the original clause when the
# whole-clause parse failed. The order is "most-specific to least-specific"
# so a phrase like "next Thursday at 5pm" matches before bare "Thursday".
_SALVAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Month + day (with optional ordinal & year): "Dec 14", "December 14th",
    # "14th of December", "14 dec 2026", "on the 14th of December".
    re.compile(
        r"""
        \b(?:on\s+(?:the\s+)?)?
        (?:
            (?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?
                (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|
                   jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|
                   nov(?:ember)?|dec(?:ember)?)
                (?:\s+\d{4})?)
          |
            (?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|
                  jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|
                  nov(?:ember)?|dec(?:ember)?)
                \s+\d{1,2}(?:st|nd|rd|th)?
                (?:,?\s+\d{4})?)
        )
        (?:\s+(?:at\s+)?\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|am|pm)?)?
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    # ISO date with optional time.
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"),
    # MM/DD or M/D with optional /YY[YY] and time.
    re.compile(
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?(?:\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)?)?\b",
        re.IGNORECASE,
    ),
    # Modifier + weekday (with optional time/period).
    re.compile(
        r"""
        \b(?:this|next|coming|upcoming|on)\s+
        (?:mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)
        (?:day)?
        (?:\s+(?:at\s+)?\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)?)?
        (?:\s+(?:in\s+the\s+)?(?:morning|afternoon|evening|night))?
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    # Clock-prefixed day anchor: "7am tomorrow", "5 pm tonight".
    re.compile(
        r"""
        \b\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)
        \s+(?:day\s+after\s+tomorrow|tomorrow|tonight|today)
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    # "tomorrow", "tonight", "today", "day after tomorrow" + optional time.
    re.compile(
        r"""
        \b(?:day\s+after\s+tomorrow|tomorrow|tonight|today)
        (?:\s+(?:at\s+)?\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)?)?
        (?:\s+(?:in\s+the\s+)?(?:morning|afternoon|evening|night))?
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    # "in N (minute|hour|day|week|month|year)s"
    re.compile(
        r"\bin\s+\d+\s+(?:minute|min|hour|hr|day|week|month|year)s?\b",
        re.IGNORECASE,
    ),
    # "in half an hour", "in an hour", "in a quarter (of an) hour"
    re.compile(
        r"\bin\s+(?:half|a\s+quarter(?:\s+of\s+an)?|an?)\s+(?:hour|hr)\b",
        re.IGNORECASE,
    ),
    # Bare weekday (last resort — easy to over-match, so kept narrow).
    # Optional clock and/or period after the weekday so "monday at 9am" /
    # "monday morning" is captured as a single span rather than just "monday".
    re.compile(
        r"""
        \b(?:on\s+)?
        (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)
        (?:\s+(?:at\s+)?\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)?)?
        (?:\s+(?:in\s+the\s+)?(?:morning|afternoon|evening|night))?
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    # Bare ordinal day-of-month: "on the 14th".
    re.compile(r"\bon\s+the\s+\d{1,2}(?:st|nd|rd|th)\b", re.IGNORECASE),
)


def _salvage_subphrase(clause: str, now: datetime | None) -> tuple[datetime, str] | None:
    """Extract a parsable date/time sub-phrase from a noisy clause.

    Used only after dateparser declines the whole sentence. Each pattern
    is tried in order; for each match we re-run dateparser on the
    extracted span (and the project-common patterns), returning the first
    success along with the extracted span itself. The caller passes that
    span to ``resolve_time`` as the *clause* so shape detection (time
    token vs. date token) reflects the salvaged phrase, not the noise
    around it.
    """

    cleaned = clause.strip()
    if not cleaned:
        return None
    for pat in _SALVAGE_PATTERNS:
        for m in pat.finditer(cleaned):
            candidate = m.group(0).strip().rstrip(".,;:!?")
            if not candidate or _span_is_full_clause(candidate, cleaned):
                continue
            parsed = _try_dateparser(candidate, now)
            if parsed is None:
                parsed = _try_project_common(candidate, now)
            if parsed is not None:
                return parsed, candidate
    return None


def _span_is_full_clause(span: str, full: str) -> bool:
    """Avoid infinite recursion when the salvage match equals the input."""

    return span.strip().lower() == full.strip().lower()


def _try_llm(clause: str, now: datetime | None) -> tuple[datetime | None, float]:
    """Run the registered LLM fallback. Always swallows exceptions."""

    if _llm_fallback is None:
        return None, 0.0
    try:
        out = _llm_fallback(clause, now)
    except Exception:  # noqa: BLE001 - protect dictation flow
        logger.exception("LLM time-parse fallback raised for clause=%r", clause)
        return None, 0.0
    if out is None:
        return None, 0.0
    iso, conf = out
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        logger.warning("LLM returned invalid ISO timestamp: %r", iso)
        return None, 0.0
    return dt, max(0.0, min(1.0, float(conf)))


def parse_when(clause: str, *, now: datetime | None = None) -> ParsedTime | None:
    """Parse a natural-language time clause into a :class:`ParsedTime`.

    Returns ``None`` only when neither dateparser nor the LLM fallback nor
    the resolver's shape-only path can produce *any* sensible answer. In
    every other case the result carries enough flags (``inferred``,
    ``needs_confirmation``, ``inference_note``) for the HUD to communicate
    uncertainty to the user.
    """

    if not clause or not clause.strip():
        return None

    cleaned = clause.strip().rstrip(".,;:!?")

    # Spoken dotted clocks ("11.30 pm") are colon clocks. dateparser
    # handles most shapes, but "at 11.30 pm tomorrow" specifically makes
    # it drop the clock and return tomorrow-at-now — an actively wrong
    # answer at full confidence (production 2026-06-11: an alarm spoken
    # as "at 11.30 pm tomorrow" was created for the wrong time). Only
    # rewrite when a meridiem follows, so decimals ("3.5 hours") and
    # version-like numbers are untouched.
    cleaned = re.sub(
        r"\b(\d{1,2})\.(\d{2})(?=\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)\b)",
        r"\1:\2",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Tier 0: pre-empt dateparser for clauses where it produces *wrong*
    # answers (not None — actively misleading). Bare ordinal days like
    # "on the 5th" past today get returned by dateparser as "same date,
    # next year" instead of "next month, same ordinal", which is what
    # every spoken assistant does. ``_try_bare_ordinal_day`` knows the
    # right answer.
    if _BARE_ORDINAL_DAY_RE.match(cleaned.lower()):
        preempted = _try_bare_ordinal_day(cleaned.lower(), now)
        if preempted is not None:
            return resolve_time(
                clause=cleaned,
                parsed=preempted,
                parser_confidence=0.85,
                parser_source="dateparser",
                now=now,
            )

    if _CLOCK_RELATIVE_DAY_RE.match(cleaned.lower()):
        preempted = _try_clock_relative_day(cleaned.lower(), now)
        if preempted is not None:
            return resolve_time(
                clause=cleaned,
                parsed=preempted,
                parser_confidence=0.85,
                parser_source="dateparser",
                now=now,
            )

    # Day-first relative clauses with a clock ("tomorrow at 9.15am") must
    # also be deterministic: dateparser accepts the day but silently drops
    # dotted clocks, returning the CURRENT time on that day — an actively
    # wrong answer (a 21:42 reminder shipped for a 9:15 request,
    # production 2026-06-11).
    day_first = _RELATIVE_DAY_RE.match(cleaned.lower())
    if day_first is not None and day_first.group("clock"):
        preempted = _try_relative_day(cleaned.lower(), now)
        if preempted is not None:
            return resolve_time(
                clause=cleaned,
                parsed=preempted,
                parser_confidence=0.85,
                # Contract allows a fixed source set; this is the same
                # preemption pattern the clock-first branch uses.
                parser_source="dateparser",
                now=now,
            )

    # Tier 1: dateparser.
    parsed = _try_dateparser(cleaned, now)
    if parsed is not None:
        return resolve_time(
            clause=cleaned,
            parsed=parsed,
            parser_confidence=0.85,
            parser_source="dateparser",
            now=now,
        )

    # Tier 2: project deterministic fallbacks for common action clauses.
    parsed = _try_project_common(cleaned, now)
    if parsed is not None:
        return resolve_time(
            clause=cleaned,
            parsed=parsed,
            parser_confidence=0.85,
            parser_source="dateparser",
            now=now,
        )

    # Tier 2b: salvage. If the clause is a full sentence with a date or
    # time phrase buried inside ("remind me on 14th of December to ..."),
    # extract the most-likely sub-phrase and parse just that. Confidence
    # is dropped because the user may have intended a different one of
    # several candidates — the HUD will surface needs_confirmation.
    salvaged = _salvage_subphrase(cleaned, now)
    if salvaged is not None:
        salvaged_dt, salvaged_clause = salvaged
        pt = resolve_time(
            clause=salvaged_clause,
            parsed=salvaged_dt,
            parser_confidence=0.7,
            parser_source="salvage",
            now=now,
        )
        if pt is not None:
            return ParsedTime(
                iso=pt.iso,
                confidence=min(pt.confidence, 0.7),
                source="salvage",
                inferred=True,
                inference_note=(
                    pt.inference_note
                    or f"extracted '{salvaged_clause}' from a longer sentence — confirm before scheduling"
                ),
                needs_confirmation=True,
            )

    # Tier 3: LLM fallback (optional, may be disabled).
    llm_dt, llm_conf = _try_llm(cleaned, now)
    if llm_dt is not None:
        # LLM confidence is multiplied by 0.9 — even a confident LLM is less
        # reliable than dateparser for trivially-parseable clauses, and these
        # are the cases dateparser already declined on.
        return resolve_time(
            clause=cleaned,
            parsed=llm_dt,
            parser_confidence=llm_conf * 0.9,
            parser_source="llm",
            now=now,
        )

    # Tier 4: resolver shape-only path. If the user said something like
    # "tomorrow" with no parser available, we can still infer 9 AM next day.
    return resolve_time(
        clause=cleaned,
        parsed=None,
        parser_confidence=0.0,
        parser_source="default",
        now=now,
    )


__all__ = ["parse_when", "set_llm_fallback", "LlmFallback"]
