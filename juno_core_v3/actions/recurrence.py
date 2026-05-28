"""Recurrence utilities.

Expand a :class:`SeriesRule` into the first N concrete instants. Used
by:

- The HUD chip "Daily for 10 days at 4 PM (next: tomorrow)" preview.
- The Swift sink's ``EKRecurrenceRule`` payload — Apple's API only
  accepts the rule itself, not pre-expanded instances, but we still
  expand for trace records and the actions index seed.

Pure-Python, no third-party deps. Day-arithmetic uses ``timedelta``
which is already imported across the codebase. Month arithmetic is
done manually because ``timedelta(days=30)`` doesn't preserve
day-of-month semantics across February / 31-day months.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

from juno_core_v3.actions.contracts import SeriesRule


# Soft cap on previewed expansions so a runaway open-ended series
# doesn't generate unbounded work.
_MAX_PREVIEW = 30


@dataclass(frozen=True, slots=True)
class ExpandedOccurrence:
    """One concrete instant produced by expanding a :class:`SeriesRule`."""

    iso: str
    occurrence_index: int  # 0-based
    excluded: bool = False  # filtered by exclude_dates_iso


def expand_series(
    rule: SeriesRule,
    *,
    limit: int | None = None,
    horizon_days: int = 1825,  # ~5 years; enough for yearly birthdays + safety
) -> list[ExpandedOccurrence]:
    """Return up to ``limit`` (default :data:`_MAX_PREVIEW`) occurrences.

    Anchored at ``rule.first_occurrence_iso``. For open-ended rules
    (no ``count`` and no ``until_iso``), we cap at ``_MAX_PREVIEW`` and
    do not exceed ``horizon_days`` past the anchor.

    Returned occurrences honor ``exclude_dates_iso`` by setting
    ``excluded=True`` on those rows; callers filter as needed.
    """
    cap = limit if limit is not None else _MAX_PREVIEW
    if cap <= 0 or not rule.first_occurrence_iso:
        return []

    start = _parse_iso(rule.first_occurrence_iso)
    if start is None:
        return []

    until = _parse_iso(rule.until_iso) if rule.until_iso else None
    horizon = start + timedelta(days=horizon_days)

    excluded = {
        _date_only(_parse_iso(s)) for s in rule.exclude_dates_iso if _parse_iso(s) is not None
    }

    out: list[ExpandedOccurrence] = []
    iterator = _iterate(rule, start)
    for idx, dt in enumerate(iterator):
        if idx >= cap:
            break
        if rule.count is not None and idx >= rule.count:
            break
        if until is not None and dt > until:
            break
        if dt > horizon:
            break
        is_excluded = _date_only(dt) in excluded
        out.append(
            ExpandedOccurrence(
                iso=_format_iso(dt),
                occurrence_index=idx,
                excluded=is_excluded,
            )
        )
    return out


def next_occurrences_summary(rule: SeriesRule, *, limit: int = 5) -> list[str]:
    """ISO strings for the next ``limit`` non-excluded occurrences. HUD chip aid."""
    out: list[str] = []
    for occ in expand_series(rule, limit=limit + len(rule.exclude_dates_iso)):
        if occ.excluded:
            continue
        out.append(occ.iso)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Internal iterators
# ---------------------------------------------------------------------------


def _iterate(rule: SeriesRule, start: datetime) -> Iterator[datetime]:
    """Yield successive occurrence datetimes for the rule, anchored at start."""
    if rule.freq == "DAILY":
        yield from _iter_daily(start, rule.interval)
        return
    if rule.freq == "WEEKLY":
        if rule.by_day:
            yield from _iter_weekly_byday(start, rule.interval, rule.by_day)
        else:
            yield from _iter_daily(start, rule.interval * 7)
        return
    if rule.freq == "MONTHLY":
        if rule.by_month_day:
            yield from _iter_monthly_bymonthday(start, rule.interval, rule.by_month_day)
        else:
            yield from _iter_monthly_anchor(start, rule.interval)
        return
    if rule.freq == "YEARLY":
        yield from _iter_yearly(start, rule.interval, rule.by_month, rule.by_month_day)
        return
    return


def _iter_daily(start: datetime, interval: int) -> Iterator[datetime]:
    cur = start
    while True:
        yield cur
        cur = cur + timedelta(days=interval)


_BYDAY_TO_WEEKDAY = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}


def _iter_weekly_byday(
    start: datetime, interval: int, by_day: tuple[str, ...]
) -> Iterator[datetime]:
    """Yield occurrences for a weekly-with-by_day rule.

    Strategy: walk the week of the anchor, emit any matching days >= the
    anchor's date. Then jump ``interval`` weeks and emit all matching
    days of the new week. Repeat.
    """
    weekdays_wanted = sorted({_BYDAY_TO_WEEKDAY[d] for d in by_day if d in _BYDAY_TO_WEEKDAY})
    if not weekdays_wanted:
        return
    week_start = start - timedelta(days=start.weekday())  # back to Monday
    week_offset = 0
    while True:
        for wd in weekdays_wanted:
            cand = week_start + timedelta(days=wd, weeks=week_offset)
            cand = cand.replace(
                hour=start.hour,
                minute=start.minute,
                second=start.second,
                microsecond=start.microsecond,
                tzinfo=start.tzinfo,
            )
            if cand < start:
                continue
            yield cand
        week_offset += interval


def _iter_monthly_bymonthday(
    start: datetime, interval: int, by_month_day: tuple[int, ...]
) -> Iterator[datetime]:
    """Yield occurrences for monthly-by-day-of-month."""
    days_wanted = sorted(set(by_month_day))
    if not days_wanted:
        return
    year, month = start.year, start.month
    while True:
        for d in days_wanted:
            last_day = calendar.monthrange(year, month)[1]
            actual_day = min(d, last_day)
            cand = start.replace(year=year, month=month, day=actual_day)
            if cand < start:
                continue
            yield cand
        month += interval
        while month > 12:
            month -= 12
            year += 1


def _iter_monthly_anchor(start: datetime, interval: int) -> Iterator[datetime]:
    """Monthly recurrence anchored on the start day-of-month."""
    year, month, day = start.year, start.month, start.day
    cur = start
    while True:
        yield cur
        month += interval
        while month > 12:
            month -= 12
            year += 1
        last_day = calendar.monthrange(year, month)[1]
        actual = min(day, last_day)
        cur = start.replace(year=year, month=month, day=actual)


def _iter_yearly(
    start: datetime,
    interval: int,
    by_month: tuple[int, ...],
    by_month_day: tuple[int, ...],
) -> Iterator[datetime]:
    """Yearly recurrence. by_month[*] x by_month_day[*] cross product if both set."""
    months = list(by_month) if by_month else [start.month]
    days = list(by_month_day) if by_month_day else [start.day]
    year = start.year
    while True:
        for m in sorted(set(months)):
            for d in sorted(set(days)):
                last_day = calendar.monthrange(year, m)[1]
                actual = min(d, last_day)
                cand = start.replace(year=year, month=m, day=actual)
                if cand < start:
                    continue
                yield cand
        year += interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # ``fromisoformat`` accepts "2026-05-07T09:00:00+07:00" and
        # "2026-05-07" since Python 3.11. Older shapes coerce via Z.
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _format_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _date_only(dt: datetime | None) -> tuple[int, int, int] | None:
    if dt is None:
        return None
    return (dt.year, dt.month, dt.day)


__all__ = ["ExpandedOccurrence", "expand_series", "next_occurrences_summary"]
