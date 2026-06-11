"""Unit tests for vague-time bucket resolution and recurrence expansion.

Covers juno_core_v3/actions/vague_time.py ``resolve_vague`` and
juno_core_v3/actions/recurrence.py ``expand_series`` /
``next_occurrences_summary``. All datetimes are explicit and aware.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from juno_core_v3.actions.contracts import SeriesRule
from juno_core_v3.actions.recurrence import expand_series, next_occurrences_summary
from juno_core_v3.actions.vague_time import resolve_vague

UTC = timezone.utc
NOW = datetime(2026, 6, 9, 10, 0, tzinfo=UTC)  # Tuesday


# ---------------------------------------------------------------------------
# resolve_vague — buckets
# ---------------------------------------------------------------------------


def test_tonight_is_8pm_today_when_still_ahead() -> None:
    assert resolve_vague("tonight", now=NOW, tz="UTC") == datetime(
        2026, 6, 9, 20, 0, tzinfo=UTC
    )


def test_tonight_rolls_to_tomorrow_when_8pm_passed() -> None:
    late = datetime(2026, 6, 9, 21, 0, tzinfo=UTC)
    assert resolve_vague("tonight", now=late, tz="UTC") == datetime(
        2026, 6, 10, 20, 0, tzinfo=UTC
    )


def test_morning_is_8am_rolling_forward_when_passed() -> None:
    # 10:00 > 08:00, so "morning" means tomorrow morning.
    assert resolve_vague("morning", now=NOW, tz="UTC") == datetime(
        2026, 6, 10, 8, 0, tzinfo=UTC
    )
    early = datetime(2026, 6, 9, 6, 0, tzinfo=UTC)
    assert resolve_vague("morning", now=early, tz="UTC") == datetime(
        2026, 6, 9, 8, 0, tzinfo=UTC
    )


def test_afternoon_is_2pm_today_when_ahead() -> None:
    assert resolve_vague("afternoon", now=NOW, tz="UTC") == datetime(
        2026, 6, 9, 14, 0, tzinfo=UTC
    )


def test_evening_is_7pm_today_when_ahead() -> None:
    assert resolve_vague("evening", now=NOW, tz="UTC") == datetime(
        2026, 6, 9, 19, 0, tzinfo=UTC
    )


def test_weekend_is_saturday_9am() -> None:
    assert resolve_vague("weekend", now=NOW, tz="UTC") == datetime(
        2026, 6, 13, 9, 0, tzinfo=UTC
    )


def test_weekend_on_saturday_before_9am_stays_today() -> None:
    sat_early = datetime(2026, 6, 13, 8, 0, tzinfo=UTC)
    assert resolve_vague("weekend", now=sat_early, tz="UTC") == datetime(
        2026, 6, 13, 9, 0, tzinfo=UTC
    )


def test_weekend_on_saturday_after_9am_jumps_a_week() -> None:
    sat_late = datetime(2026, 6, 13, 10, 0, tzinfo=UTC)
    assert resolve_vague("weekend", now=sat_late, tz="UTC") == datetime(
        2026, 6, 20, 9, 0, tzinfo=UTC
    )


def test_next_week_is_monday_9am() -> None:
    assert resolve_vague("next_week", now=NOW, tz="UTC") == datetime(
        2026, 6, 15, 9, 0, tzinfo=UTC
    )


def test_next_week_on_monday_jumps_a_full_week() -> None:
    monday = datetime(2026, 6, 15, 10, 0, tzinfo=UTC)
    assert resolve_vague("next_week", now=monday, tz="UTC") == datetime(
        2026, 6, 22, 9, 0, tzinfo=UTC
    )


def test_soon_is_30_minutes_snapped_to_next_five() -> None:
    # 10:02 + 30min = 10:32 → snapped up to 10:35.
    now = datetime(2026, 6, 9, 10, 2, tzinfo=UTC)
    assert resolve_vague("soon", now=now, tz="UTC") == datetime(
        2026, 6, 9, 10, 35, tzinfo=UTC
    )


def test_soon_already_on_five_minute_boundary_is_unchanged() -> None:
    assert resolve_vague("soon", now=NOW, tz="UTC") == datetime(
        2026, 6, 9, 10, 30, tzinfo=UTC
    )


def test_later_is_two_hours_snapped() -> None:
    now = datetime(2026, 6, 9, 10, 3, tzinfo=UTC)
    assert resolve_vague("later", now=now, tz="UTC") == datetime(
        2026, 6, 9, 12, 5, tzinfo=UTC
    )


def test_explicit_zone_converts_before_bucketing() -> None:
    # 10:00 UTC is 15:30 in Kolkata; "tonight" lands at 20:00 local.
    out = resolve_vague("tonight", now=NOW, tz="Asia/Kolkata")
    assert out == datetime(2026, 6, 9, 20, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert out.utcoffset() is not None


def test_naive_now_is_treated_as_utc() -> None:
    naive = datetime(2026, 6, 9, 10, 0)
    assert resolve_vague("soon", now=naive, tz="UTC") == datetime(
        2026, 6, 9, 10, 30, tzinfo=UTC
    )


def test_unknown_zone_falls_back_to_now_tzinfo() -> None:
    out = resolve_vague("soon", now=NOW, tz="Not/AZone")
    assert out == datetime(2026, 6, 9, 10, 30, tzinfo=UTC)


def test_unknown_bucket_raises_value_error() -> None:
    with pytest.raises(ValueError):
        resolve_vague("whenever", now=NOW, tz="UTC")


# ---------------------------------------------------------------------------
# expand_series
# ---------------------------------------------------------------------------


def test_daily_with_interval_and_count() -> None:
    rule = SeriesRule(
        freq="DAILY",
        interval=2,
        first_occurrence_iso="2026-06-09T09:00:00+00:00",
        count=3,
    )
    occ = expand_series(rule)
    assert [o.iso for o in occ] == [
        "2026-06-09T09:00:00+00:00",
        "2026-06-11T09:00:00+00:00",
        "2026-06-13T09:00:00+00:00",
    ]
    assert [o.occurrence_index for o in occ] == [0, 1, 2]
    assert all(not o.excluded for o in occ)


def test_weekly_by_day_mo_fr_from_tuesday_anchor() -> None:
    # Anchored on Tuesday June 9: the Monday of the anchor week has passed,
    # so the series starts Friday June 12 and alternates MO/FR thereafter,
    # preserving the anchor's time-of-day.
    rule = SeriesRule(
        freq="WEEKLY",
        by_day=("MO", "FR"),
        first_occurrence_iso="2026-06-09T09:00:00+00:00",
    )
    occ = expand_series(rule, limit=4)
    assert [o.iso for o in occ] == [
        "2026-06-12T09:00:00+00:00",
        "2026-06-15T09:00:00+00:00",
        "2026-06-19T09:00:00+00:00",
        "2026-06-22T09:00:00+00:00",
    ]


def test_monthly_by_month_day_31_clamps_short_months() -> None:
    # ``_iter_monthly_bymonthday`` deliberately clamps to the last valid day
    # (min(d, last_day)) rather than skipping short months entirely.
    rule = SeriesRule(
        freq="MONTHLY",
        by_month_day=(31,),
        first_occurrence_iso="2026-01-31T09:00:00+00:00",
    )
    occ = expand_series(rule, limit=4)
    assert [o.iso for o in occ] == [
        "2026-01-31T09:00:00+00:00",
        "2026-02-28T09:00:00+00:00",  # 2026 is not a leap year
        "2026-03-31T09:00:00+00:00",
        "2026-04-30T09:00:00+00:00",
    ]


def test_until_iso_bounds_the_series() -> None:
    rule = SeriesRule(
        freq="DAILY",
        first_occurrence_iso="2026-06-09T09:00:00+00:00",
        until_iso="2026-06-12T09:00:00+00:00",
    )
    occ = expand_series(rule)
    assert [o.iso for o in occ] == [
        "2026-06-09T09:00:00+00:00",
        "2026-06-10T09:00:00+00:00",
        "2026-06-11T09:00:00+00:00",
        "2026-06-12T09:00:00+00:00",
    ]


def test_count_bounds_the_series_even_with_higher_limit() -> None:
    rule = SeriesRule(
        freq="DAILY",
        first_occurrence_iso="2026-06-09T09:00:00+00:00",
        count=2,
    )
    assert len(expand_series(rule, limit=10)) == 2


def test_open_ended_series_caps_at_preview_max() -> None:
    rule = SeriesRule(freq="DAILY", first_occurrence_iso="2026-06-09T09:00:00+00:00")
    assert len(expand_series(rule)) == 30


def test_exclude_dates_flags_rows_without_dropping_them() -> None:
    rule = SeriesRule(
        freq="DAILY",
        first_occurrence_iso="2026-06-09T09:00:00+00:00",
        exclude_dates_iso=("2026-06-11",),
    )
    occ = expand_series(rule, limit=4)
    assert [(o.iso, o.excluded) for o in occ] == [
        ("2026-06-09T09:00:00+00:00", False),
        ("2026-06-10T09:00:00+00:00", False),
        ("2026-06-11T09:00:00+00:00", True),
        ("2026-06-12T09:00:00+00:00", False),
    ]


def test_zero_limit_or_missing_anchor_yields_nothing() -> None:
    rule = SeriesRule(freq="DAILY", first_occurrence_iso="2026-06-09T09:00:00+00:00")
    assert expand_series(rule, limit=0) == []
    assert expand_series(SeriesRule(freq="DAILY")) == []
    assert expand_series(SeriesRule(freq="DAILY", first_occurrence_iso="garbage")) == []


# ---------------------------------------------------------------------------
# next_occurrences_summary
# ---------------------------------------------------------------------------


def test_summary_skips_excluded_dates() -> None:
    rule = SeriesRule(
        freq="DAILY",
        first_occurrence_iso="2026-06-09T09:00:00+00:00",
        exclude_dates_iso=("2026-06-10",),
    )
    assert next_occurrences_summary(rule, limit=3) == [
        "2026-06-09T09:00:00+00:00",
        "2026-06-11T09:00:00+00:00",
        "2026-06-12T09:00:00+00:00",
    ]


def test_summary_respects_limit() -> None:
    rule = SeriesRule(freq="DAILY", first_occurrence_iso="2026-06-09T09:00:00+00:00")
    out = next_occurrences_summary(rule, limit=5)
    assert len(out) == 5
    assert out[0] == "2026-06-09T09:00:00+00:00"
