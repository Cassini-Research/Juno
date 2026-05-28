"""Resolve vague spoken time buckets to concrete defaults."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def resolve_vague(bucket: str, *, now: datetime, tz: str) -> datetime:
    """Map a vague bucket to a concrete datetime in ``tz``.

    The defaults intentionally match the Juno actions rehaul plan. The
    returned datetime is timezone-aware and ready to serialize as ISO-8601.
    """

    zone = _zone(tz, now)
    local_now = _aware(now).astimezone(zone)
    key = (bucket or "").strip().lower()

    if key == "later":
        return _snap_to_next_five(local_now + timedelta(hours=2))
    if key == "soon":
        return _snap_to_next_five(local_now + timedelta(minutes=30))
    if key == "tonight":
        return _today_or_tomorrow(local_now, time(20, 0), zone)
    if key == "morning":
        return _today_or_tomorrow(local_now, time(8, 0), zone)
    if key == "afternoon":
        return _today_or_tomorrow(local_now, time(14, 0), zone)
    if key == "evening":
        return _today_or_tomorrow(local_now, time(19, 0), zone)
    if key == "weekend":
        days_until_saturday = (5 - local_now.weekday()) % 7
        if days_until_saturday == 0 and local_now.time() >= time(9, 0):
            days_until_saturday = 7
        target_date = (local_now + timedelta(days=days_until_saturday)).date()
        return datetime.combine(target_date, time(9, 0), tzinfo=zone)
    if key == "next_week":
        days_until_monday = (7 - local_now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        target_date = (local_now + timedelta(days=days_until_monday)).date()
        return datetime.combine(target_date, time(9, 0), tzinfo=zone)
    raise ValueError(f"unknown vague bucket: {bucket}")


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _zone(tz: str, now: datetime) -> timezone | ZoneInfo:
    name = (tz or "").strip()
    if name:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            pass
    current = now.tzinfo
    if current is not None:
        return current
    return timezone.utc


def _today_or_tomorrow(now: datetime, target_time: time, zone: timezone | ZoneInfo) -> datetime:
    candidate = datetime.combine(now.date(), target_time, tzinfo=zone)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return candidate


def _snap_to_next_five(value: datetime) -> datetime:
    discard = value.minute % 5
    if discard == 0 and value.second == 0 and value.microsecond == 0:
        return value
    delta = 5 - discard if discard else 0
    snapped = value + timedelta(minutes=delta)
    return snapped.replace(second=0, microsecond=0)


__all__ = ["resolve_vague"]
