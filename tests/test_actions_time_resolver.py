"""Unit tests for juno_core_v3/actions/time_resolver.py resolve_time.

``resolve_time`` is called with fully-controlled ``parsed`` datetimes and an
explicit aware ``now`` so every assertion is deterministic and timezone-safe.
NOW is Tuesday 2026-06-09 10:00 UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone

from juno_core_v3.actions.time_resolver import resolve_time

UTC = timezone.utc
NOW = datetime(2026, 6, 9, 10, 0, tzinfo=UTC)


def _dt(pt) -> datetime:
    return datetime.fromisoformat(pt.iso)


# ---------------------------------------------------------------------------
# Rule 1: date-only clauses get the 9 AM default fill
# ---------------------------------------------------------------------------


def test_date_only_clause_defaults_to_9am() -> None:
    pt = resolve_time(
        clause="monday",
        parsed=datetime(2026, 6, 15, 0, 0, tzinfo=UTC),
        parser_confidence=0.85,
        parser_source="dateparser",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
    assert pt.inferred is True
    assert pt.inference_note == "no time specified — defaulted to 9:00 AM"
    assert pt.needs_confirmation is False


def test_date_only_clause_overrides_parser_supplied_hour() -> None:
    # dateparser often returns "now's hour" for "in 4 days"; the resolver
    # must replace it with 9 AM.
    pt = resolve_time(
        clause="in 4 days",
        parsed=datetime(2026, 6, 13, 10, 0, tzinfo=UTC),
        parser_confidence=0.85,
        parser_source="dateparser",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 13, 9, 0, tzinfo=UTC)
    assert pt.inferred is True


def test_date_only_today_pushes_to_tomorrow_when_9am_passed() -> None:
    # "today" with the 9 AM default already in the past (now is 10:00) must
    # not schedule into the past — push to the next day.
    pt = resolve_time(
        clause="today",
        parsed=datetime(2026, 6, 9, 0, 0, tzinfo=UTC),
        parser_confidence=0.85,
        parser_source="dateparser",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 10, 9, 0, tzinfo=UTC)


def test_date_only_caps_confidence_at_0_7() -> None:
    pt = resolve_time(
        clause="next friday",
        parsed=datetime(2026, 6, 12, 0, 0, tzinfo=UTC),
        parser_confidence=0.85,
        parser_source="dateparser",
        now=NOW,
    )
    assert pt is not None
    assert pt.confidence == 0.7


def test_date_only_keeps_lower_parser_confidence() -> None:
    pt = resolve_time(
        clause="next friday",
        parsed=datetime(2026, 6, 12, 0, 0, tzinfo=UTC),
        parser_confidence=0.4,
        parser_source="llm",
        now=NOW,
    )
    assert pt is not None
    assert pt.confidence == 0.4
    assert pt.source == "llm"


# ---------------------------------------------------------------------------
# Rule 2: past time-of-day without explicit date rolls forward one day
# ---------------------------------------------------------------------------


def test_past_bare_time_rolls_forward_one_day() -> None:
    # "at 9am" when it's already 10:00 → tomorrow 9 AM.
    pt = resolve_time(
        clause="at 9am",
        parsed=datetime(2026, 6, 9, 9, 0, tzinfo=UTC),
        parser_confidence=0.85,
        parser_source="dateparser",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 10, 9, 0, tzinfo=UTC)
    assert pt.inferred is True
    assert "already passed today" in (pt.inference_note or "")
    assert pt.confidence == 0.65
    assert pt.needs_confirmation is False


def test_future_bare_time_is_kept_verbatim() -> None:
    pt = resolve_time(
        clause="at 5pm",
        parsed=datetime(2026, 6, 9, 17, 0, tzinfo=UTC),
        parser_confidence=0.85,
        parser_source="dateparser",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 9, 17, 0, tzinfo=UTC)
    assert pt.inferred is False
    assert pt.inference_note is None
    assert pt.confidence == 0.85


# ---------------------------------------------------------------------------
# Rule 3: past timestamp with explicit date → keep, flag for confirmation
# ---------------------------------------------------------------------------


def test_past_explicit_date_kept_but_flagged() -> None:
    pt = resolve_time(
        clause="march 1 at 5pm",
        parsed=datetime(2026, 3, 1, 17, 0, tzinfo=UTC),
        parser_confidence=0.85,
        parser_source="dateparser",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 3, 1, 17, 0, tzinfo=UTC)  # verbatim
    assert pt.needs_confirmation is True
    assert pt.inferred is False
    assert "in the past" in (pt.inference_note or "")
    assert pt.confidence == 0.5


# ---------------------------------------------------------------------------
# Branch A: parser failed (parsed=None)
# ---------------------------------------------------------------------------


def test_none_parsed_with_no_shape_returns_none() -> None:
    assert (
        resolve_time(
            clause="buy milk",
            parsed=None,
            parser_confidence=0.0,
            parser_source="default",
            now=NOW,
        )
        is None
    )


def test_empty_clause_returns_none() -> None:
    assert (
        resolve_time(
            clause="   ",
            parsed=None,
            parser_confidence=0.0,
            parser_source="default",
            now=NOW,
        )
        is None
    )


def test_none_parsed_with_date_token_defaults_tomorrow_9am() -> None:
    pt = resolve_time(
        clause="tomorrow",
        parsed=None,
        parser_confidence=0.0,
        parser_source="default",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 10, 9, 0, tzinfo=UTC)
    assert pt.source == "default"
    assert pt.confidence == 0.55
    assert pt.inferred is True
    assert pt.needs_confirmation is True


def test_none_parsed_with_word_time_uses_word_default() -> None:
    pt = resolve_time(
        clause="in the evening",
        parsed=None,
        parser_confidence=0.0,
        parser_source="default",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 10, 18, 0, tzinfo=UTC)
    assert pt.confidence == 0.55
    assert pt.needs_confirmation is True


# ---------------------------------------------------------------------------
# Confidence scaling / clamping
# ---------------------------------------------------------------------------


def test_out_of_range_parser_confidence_is_clamped() -> None:
    high = resolve_time(
        clause="tomorrow at 5pm",
        parsed=datetime(2026, 6, 10, 17, 0, tzinfo=UTC),
        parser_confidence=1.5,
        parser_source="dateparser",
        now=NOW,
    )
    low = resolve_time(
        clause="tomorrow at 5pm",
        parsed=datetime(2026, 6, 10, 17, 0, tzinfo=UTC),
        parser_confidence=-0.5,
        parser_source="dateparser",
        now=NOW,
    )
    assert high is not None and low is not None
    assert high.confidence == 1.0
    assert low.confidence == 0.0


def test_parser_confidence_passes_through_when_no_inference() -> None:
    pt = resolve_time(
        clause="tomorrow at 5pm",
        parsed=datetime(2026, 6, 10, 17, 0, tzinfo=UTC),
        parser_confidence=0.42,
        parser_source="llm",
        now=NOW,
    )
    assert pt is not None
    assert pt.confidence == 0.42


def test_naive_parsed_datetime_anchored_in_now_tz() -> None:
    # Parsers may return naive datetimes; the resolver attaches now's tz so
    # the instant survives serialization.
    pt = resolve_time(
        clause="tomorrow at 5pm",
        parsed=datetime(2026, 6, 10, 17, 0),
        parser_confidence=0.85,
        parser_source="dateparser",
        now=NOW,
    )
    assert pt is not None
    assert _dt(pt) == datetime(2026, 6, 10, 17, 0, tzinfo=UTC)
