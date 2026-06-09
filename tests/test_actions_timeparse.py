"""Unit tests for juno_core_v3/actions/timeparse.py parse_when.

Every test passes an explicit ``now`` so results never depend on the wall
clock. ``NOW`` is Tuesday 2026-06-09 10:00 UTC. Clauses handled by the
project's deterministic handlers (tier 0 / tier 2) inherit ``now``'s
timezone, so those assert exact instants. Clauses that dateparser resolves
against the *system* timezone (bare clock times, absolute month/day dates)
assert wall-clock fields only, keeping the tests machine-independent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from juno_core_v3.actions.timeparse import parse_when

UTC = timezone.utc
NOW = datetime(2026, 6, 9, 10, 0, tzinfo=UTC)  # Tuesday


def _dt(pt) -> datetime:
    return datetime.fromisoformat(pt.iso)


# ---------------------------------------------------------------------------
# Relative-day clauses
# ---------------------------------------------------------------------------


def test_tomorrow_at_5pm() -> None:
    pt = parse_when("tomorrow at 5pm", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 10)
    assert (dt.hour, dt.minute) == (17, 0)
    assert pt.confidence == 0.85
    assert pt.source == "dateparser"
    assert pt.inferred is False
    assert pt.needs_confirmation is False


def test_day_after_tomorrow_11am() -> None:
    pt = parse_when("day after tomorrow 11am", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 11)
    assert (dt.hour, dt.minute) == (11, 0)
    assert pt.inferred is False


def test_clock_before_relative_day_uses_now_tz_exactly() -> None:
    # "5 pm tomorrow" hits the deterministic clock+relative-day pre-empt,
    # which anchors in now's timezone — assert the exact instant.
    pt = parse_when("5pm tomorrow", now=NOW)
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 10, 17, 0, tzinfo=UTC)


def test_tonight_defaults_to_8pm() -> None:
    pt = parse_when("tonight", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 9)
    assert (dt.hour, dt.minute) == (20, 0)


def test_tonight_at_8_disambiguates_to_pm() -> None:
    pt = parse_when("tonight at 8", now=NOW)
    assert pt is not None
    assert _dt(pt).hour == 20


# ---------------------------------------------------------------------------
# Weekday clauses
# ---------------------------------------------------------------------------


def test_next_monday_is_upcoming_monday_with_9am_default() -> None:
    # NOW is Tuesday June 9; the upcoming Monday is June 15. Date-only, so
    # the resolver fills 9 AM and flags the inference.
    pt = parse_when("next monday", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 15)
    assert dt.weekday() == 0
    assert (dt.hour, dt.minute) == (9, 0)
    assert pt.inferred is True
    assert pt.confidence == 0.7
    assert "9:00 AM" in (pt.inference_note or "")


def test_next_thursday_at_5pm() -> None:
    pt = parse_when("next thursday at 5pm", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert dt.weekday() == 3
    assert dt > datetime(2026, 6, 9, 10, 0, tzinfo=UTC).astimezone(dt.tzinfo)
    assert (dt.hour, dt.minute) == (17, 0)
    assert pt.inferred is False


def test_monday_morning_uses_period_default() -> None:
    pt = parse_when("monday morning", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert dt.weekday() == 0
    assert (dt.year, dt.month, dt.day) == (2026, 6, 15)
    assert (dt.hour, dt.minute) == (9, 0)
    # "morning" counts as a time token, so no 9 AM default-fill inference.
    assert pt.inferred is False


# ---------------------------------------------------------------------------
# Bare ordinal days — future-leaning
# ---------------------------------------------------------------------------


def test_bare_ordinal_later_this_month_stays_this_month() -> None:
    pt = parse_when("on the 14th", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 14)
    assert (dt.hour, dt.minute) == (9, 0)  # date-only → default fill
    assert pt.inferred is True


def test_bare_ordinal_already_passed_jumps_to_next_month() -> None:
    # The 5th has passed on June 9 — future-leaning means July 5, not
    # dateparser's "same date next year".
    pt = parse_when("on the 5th", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 7, 5)


def test_bare_ordinal_today_rolls_to_next_day_when_9am_passed() -> None:
    # "the 9th" is today; the 9 AM default has already passed at 10:00, so
    # the resolver pushes one day forward.
    pt = parse_when("the 9th", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 10)
    assert (dt.hour, dt.minute) == (9, 0)


def test_bare_ordinal_31st_skips_to_month_that_has_it() -> None:
    # June has only 30 days; "the 31st" resolves into July.
    pt = parse_when("on the 31st", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 7, 31)


# ---------------------------------------------------------------------------
# Clock parsing
# ---------------------------------------------------------------------------


def test_bare_clock_pm_resolves_to_17_hours() -> None:
    pt = parse_when("at 5pm", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.hour, dt.minute) == (17, 0)
    # Future-leaning: never schedules into the past.
    assert dt > NOW
    assert dt - NOW < timedelta(days=2)


def test_clock_with_minutes_and_dotted_meridiem() -> None:
    pt = parse_when("10.30 a.m. tomorrow", now=NOW)
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 10, 10, 30, tzinfo=UTC)


def test_twelve_am_is_midnight_and_twelve_pm_is_noon() -> None:
    am = parse_when("12am tomorrow", now=NOW)
    pm = parse_when("12pm tomorrow", now=NOW)
    assert am is not None and pm is not None
    assert _dt(am) == datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
    assert _dt(pm) == datetime(2026, 6, 10, 12, 0, tzinfo=UTC)


def test_24h_clock() -> None:
    pt = parse_when("17:30 tomorrow", now=NOW)
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 10, 17, 30, tzinfo=UTC)


def test_in_half_an_hour_is_relative_to_now() -> None:
    pt = parse_when("in half an hour", now=NOW)
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 9, 10, 30, tzinfo=UTC)


def test_in_2_hours_is_relative_to_now() -> None:
    pt = parse_when("in 2 hours", now=NOW)
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 9, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Week / weekend / month-boundary clauses
# ---------------------------------------------------------------------------


def test_next_weekend_lands_on_the_following_saturday() -> None:
    pt = parse_when("next weekend", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 20)
    assert dt.weekday() == 5
    assert (dt.hour, dt.minute) == (9, 0)  # date-only default fill


def test_next_week_lands_seven_days_out_with_default_hour() -> None:
    pt = parse_when("next week", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 16)
    assert (dt.hour, dt.minute) == (9, 0)
    assert pt.inferred is True


def test_end_of_next_month() -> None:
    pt = parse_when("end of next month", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 7, 31)


# ---------------------------------------------------------------------------
# Salvage subphrases
# ---------------------------------------------------------------------------


def test_salvage_extracts_month_day_from_noisy_sentence() -> None:
    pt = parse_when("remind me on the 14th of December to buy gifts", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.month, dt.day) == (12, 14)
    assert dt.hour == 9  # date-only → default fill
    assert pt.source == "salvage"
    assert pt.inferred is True
    assert pt.needs_confirmation is True
    assert pt.confidence <= 0.7


def test_salvage_extracts_relative_day_from_noisy_sentence() -> None:
    pt = parse_when("please could you do the thing tomorrow at 3pm thanks", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 10)
    assert (dt.hour, dt.minute) == (15, 0)
    assert pt.source == "salvage"
    assert pt.needs_confirmation is True


def test_salvage_note_mentions_extracted_span() -> None:
    pt = parse_when("remember the meeting next friday at 2pm with the team", now=NOW)
    assert pt is not None
    assert pt.source == "salvage"
    dt = _dt(pt)
    assert dt.weekday() == 4
    assert dt.hour == 14
    assert pt.inference_note  # always carries an explanation for the HUD


# ---------------------------------------------------------------------------
# Failure / degenerate inputs
# ---------------------------------------------------------------------------


def test_no_time_signal_returns_none() -> None:
    assert parse_when("buy milk and eggs", now=NOW) is None


def test_empty_and_whitespace_return_none() -> None:
    assert parse_when("", now=NOW) is None
    assert parse_when("   ", now=NOW) is None


def test_trailing_punctuation_is_tolerated() -> None:
    pt = parse_when("tomorrow at 5pm.", now=NOW)
    assert pt is not None
    dt = _dt(pt)
    assert (dt.year, dt.month, dt.day) == (2026, 6, 10)
    assert dt.hour == 17
