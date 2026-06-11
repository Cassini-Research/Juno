"""Unit tests for action contracts and the reference resolver.

Covers juno_core_v3/actions/contracts.py (to_dict/from_dict round-trips and
constructor validation) and juno_core_v3/actions/reference_resolver.py
``resolve_target`` driven through a small in-memory fake of the actions
index. All datetimes are explicit and timezone-aware.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from juno_core_v3.actions.contracts import (
    Action,
    ActionKind,
    ActionOperation,
    ParsedTime,
    Schedule,
    SeriesRule,
    VagueSchedule,
)
from juno_core_v3.actions.reference_resolver import resolve_target

UTC = timezone.utc
NOW = datetime(2026, 6, 9, 10, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# ParsedTime
# ---------------------------------------------------------------------------


def test_parsed_time_minimal_dict_omits_default_flags() -> None:
    pt = ParsedTime(iso="2026-06-10T09:00:00+00:00", confidence=0.85)
    assert pt.to_dict() == {
        "iso": "2026-06-10T09:00:00+00:00",
        "confidence": 0.85,
        "source": "dateparser",
    }


def test_parsed_time_round_trip_with_all_flags() -> None:
    pt = ParsedTime(
        iso="2026-06-10T09:00:00+00:00",
        confidence=0.55,
        source="default",
        inferred=True,
        inference_note="no time specified — defaulted to 9:00 AM",
        needs_confirmation=True,
    )
    assert ParsedTime.from_dict(pt.to_dict()) == pt


def test_parsed_time_from_dict_fills_defaults() -> None:
    pt = ParsedTime.from_dict({"iso": "2026-06-10T09:00:00+00:00"})
    assert pt.confidence == 1.0
    assert pt.source == "dateparser"
    assert pt.inferred is False
    assert pt.inference_note is None
    assert pt.needs_confirmation is False


def test_parsed_time_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        ParsedTime(iso="2026-06-10T09:00:00+00:00", confidence=1.5)
    with pytest.raises(ValueError):
        ParsedTime(iso="2026-06-10T09:00:00+00:00", confidence=-0.1)


def test_parsed_time_unknown_source_raises() -> None:
    with pytest.raises(ValueError):
        ParsedTime(iso="2026-06-10T09:00:00+00:00", source="oracle")


# ---------------------------------------------------------------------------
# SeriesRule
# ---------------------------------------------------------------------------


def test_series_rule_round_trip_full() -> None:
    rule = SeriesRule(
        freq="WEEKLY",
        interval=2,
        by_day=("MO", "FR"),
        by_month_day=(1, 15),
        by_month=(6,),
        count=10,
        until_iso="2026-12-31T00:00:00+00:00",
        first_occurrence_iso="2026-06-12T09:00:00+00:00",
        tz="Asia/Kolkata",
        exclude_dates_iso=("2026-06-19",),
    )
    assert SeriesRule.from_dict(rule.to_dict()) == rule


def test_series_rule_minimal_dict_only_carries_freq_and_interval() -> None:
    assert SeriesRule(freq="DAILY").to_dict() == {"freq": "DAILY", "interval": 1}


def test_series_rule_from_dict_normalizes_case() -> None:
    rule = SeriesRule.from_dict({"freq": "weekly", "by_day": ["mo", "fr"]})
    assert rule.freq == "WEEKLY"
    assert rule.by_day == ("MO", "FR")


def test_series_rule_validation_errors() -> None:
    with pytest.raises(ValueError):
        SeriesRule(freq="HOURLY")
    with pytest.raises(ValueError):
        SeriesRule(freq="DAILY", interval=0)
    with pytest.raises(ValueError):
        SeriesRule(freq="WEEKLY", by_day=("MO", "XX"))
    with pytest.raises(ValueError):
        SeriesRule(freq="DAILY", count=0)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def _instant() -> ParsedTime:
    return ParsedTime(iso="2026-06-10T09:00:00+00:00", confidence=0.85)


def test_schedule_instant_round_trip_and_primary_iso() -> None:
    sched = Schedule(kind="instant", instant=_instant())
    assert Schedule.from_dict(sched.to_dict()) == sched
    assert sched.primary_iso == "2026-06-10T09:00:00+00:00"


def test_schedule_series_round_trip_and_primary_iso() -> None:
    rule = SeriesRule(freq="DAILY", first_occurrence_iso="2026-06-10T09:00:00+00:00")
    sched = Schedule(kind="series", series=rule)
    assert Schedule.from_dict(sched.to_dict()) == sched
    assert sched.primary_iso == "2026-06-10T09:00:00+00:00"


def test_schedule_vague_round_trip_and_primary_iso() -> None:
    vague = VagueSchedule(bucket="tonight", default_iso="2026-06-09T20:00:00+00:00")
    sched = Schedule(kind="vague", vague=vague)
    assert Schedule.from_dict(sched.to_dict()) == sched
    assert sched.primary_iso == "2026-06-09T20:00:00+00:00"
    assert sched.vague.needs_confirmation is True


def test_schedule_kind_must_match_variant() -> None:
    with pytest.raises(ValueError):
        Schedule(kind="series", instant=_instant())
    with pytest.raises(ValueError):
        Schedule(kind="bogus", instant=_instant())


def test_schedule_requires_exactly_one_variant() -> None:
    with pytest.raises(ValueError):
        Schedule(kind="instant")
    with pytest.raises(ValueError):
        Schedule(
            kind="instant",
            instant=_instant(),
            series=SeriesRule(freq="DAILY"),
        )


def test_vague_schedule_unknown_bucket_raises() -> None:
    with pytest.raises(ValueError):
        VagueSchedule(bucket="someday", default_iso="2026-06-09T20:00:00+00:00")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


def test_action_minimal_wire_shape_is_v2_compatible() -> None:
    action = Action(kind=ActionKind.NOTE, body="buy milk", raw_span="take a note buy milk")
    assert action.to_dict() == {
        "kind": "note",
        "body": "buy milk",
        "raw_span": "take a note buy milk",
        "when": None,
    }


def test_action_round_trip_full() -> None:
    action = Action(
        kind=ActionKind.REMINDER,
        body="call sam",
        raw_span="remind me to call sam tomorrow",
        when=_instant(),
        schedule=Schedule(kind="instant", instant=_instant()),
        operation=ActionOperation.UPDATE,
        target={"ref_kind": "by_id", "id": "j-1"},
        container={"list_name": "Errands"},
        juno_id="j-1",
        sink_id="ek-9",
        snooze_offset_seconds=600,
        relative_offset_seconds=-300,
        link_id="L1",
        links_to="L0",
        needs_confirmation=True,
    )
    data = action.to_dict()
    assert data["operation"] == "update"
    assert data["needs_confirmation"] is True
    assert Action.from_dict(data) == action


def test_action_from_dict_defaults_raw_span_to_body() -> None:
    action = Action.from_dict({"kind": "note", "body": "buy milk", "when": None})
    assert action.raw_span == "buy milk"
    assert action.operation is ActionOperation.CREATE
    assert action.when is None


def test_action_round_trip_with_when_only() -> None:
    action = Action(
        kind=ActionKind.ALARM,
        body="Alarm",
        raw_span="set an alarm at 7am",
        when=ParsedTime(iso="2026-06-10T07:00:00+00:00", confidence=0.85),
    )
    assert Action.from_dict(action.to_dict()) == action


# ---------------------------------------------------------------------------
# reference_resolver.resolve_target — fake actions index
# ---------------------------------------------------------------------------


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _row(
    juno_id: str,
    body: str,
    *,
    kind: str = "reminder",
    session: str = "sess-1",
    modified: datetime | None = None,
    status: str = "active",
    deleted_at: int | None = None,
) -> dict[str, Any]:
    modified = modified or (NOW - timedelta(minutes=1))
    return {
        "juno_id": juno_id,
        "body_normalized": body,
        "sink_kind": kind,
        "kind": kind,
        "status": status,
        "deleted_at": deleted_at,
        "last_seen_session": session,
        "last_modified_at": _ms(modified),
        "created_at": _ms(modified),
    }


class FakeIndex:
    """Tiny stand-in for the actions index, recent-first like the real one."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def get(self, juno_id: str) -> dict[str, Any] | None:
        for row in self.rows:
            if row["juno_id"] == juno_id:
                return row
        return None

    def find(
        self,
        *,
        body_substr: str | None = None,
        date_range: tuple[str, str] | None = None,
        list_name: str | None = None,
        kind: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        out = []
        for row in self.rows:
            if kind is not None and row.get("kind") != kind:
                continue
            if body_substr is not None and body_substr.lower() not in row["body_normalized"]:
                continue
            out.append(row)
        out.sort(key=lambda r: r["last_modified_at"], reverse=True)
        return out[:limit]

    def last_touched(self, *, kind: str | None = None) -> dict[str, Any] | None:
        rows = self.find(kind=kind, limit=1)
        return rows[0] if rows else None


# -- by_id ------------------------------------------------------------------


def test_by_id_resolves_with_full_confidence() -> None:
    row = _row("j-1", "call sam")
    out = resolve_target(
        {"ref_kind": "by_id", "id": "j-1"},
        index=FakeIndex([row]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is not None
    assert out.juno_id == "j-1"
    assert out.confidence == 1.0
    assert out.resolved_via == "by_id"
    assert out.candidates == [row]


def test_by_id_rejects_deleted_and_inactive_rows() -> None:
    deleted = _row("j-1", "call sam", deleted_at=_ms(NOW))
    completed = _row("j-2", "call sam", status="completed")
    index = FakeIndex([deleted, completed])
    assert (
        resolve_target({"ref_kind": "by_id", "id": "j-1"}, index=index, session_id="s", now=NOW)
        is None
    )
    assert (
        resolve_target({"ref_kind": "by_id", "id": "j-2"}, index=index, session_id="s", now=NOW)
        is None
    )


def test_by_id_missing_or_unknown_id_returns_none() -> None:
    index = FakeIndex([_row("j-1", "call sam")])
    assert resolve_target({"ref_kind": "by_id"}, index=index, session_id="s", now=NOW) is None
    assert (
        resolve_target({"ref_kind": "by_id", "id": "nope"}, index=index, session_id="s", now=NOW)
        is None
    )


# -- by_pronoun -------------------------------------------------------------


def test_by_pronoun_single_recent_row_high_confidence() -> None:
    row = _row("j-1", "call sam", modified=NOW - timedelta(minutes=2))
    out = resolve_target(
        {"ref_kind": "by_pronoun", "pronoun": "that"},
        index=FakeIndex([row]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is not None
    assert out.juno_id == "j-1"
    assert out.confidence == 0.95
    assert out.resolved_via == "by_pronoun"


def test_by_pronoun_multiple_recent_rows_low_confidence_picks_most_recent() -> None:
    older = _row("j-1", "call sam", modified=NOW - timedelta(minutes=4))
    newer = _row("j-2", "buy milk", modified=NOW - timedelta(minutes=1))
    out = resolve_target(
        {"ref_kind": "by_pronoun", "pronoun": "it"},
        index=FakeIndex([older, newer]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is not None
    assert out.juno_id == "j-2"
    assert out.confidence == 0.6
    assert len(out.candidates) == 2


def test_by_pronoun_ignores_rows_outside_5_minute_window() -> None:
    stale = _row("j-1", "call sam", modified=NOW - timedelta(minutes=10))
    out = resolve_target(
        {"ref_kind": "by_pronoun", "pronoun": "that"},
        index=FakeIndex([stale]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is None


def test_by_pronoun_ignores_rows_from_other_sessions() -> None:
    other = _row("j-1", "call sam", session="sess-OTHER")
    out = resolve_target(
        {"ref_kind": "by_pronoun", "pronoun": "that"},
        index=FakeIndex([other]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is None


def test_by_pronoun_unknown_pronoun_returns_none() -> None:
    row = _row("j-1", "call sam")
    out = resolve_target(
        {"ref_kind": "by_pronoun", "pronoun": "whatever"},
        index=FakeIndex([row]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is None


# -- by_description ----------------------------------------------------------


def test_by_description_literal_substring_match() -> None:
    rows = [_row("j-1", "gym session at four"), _row("j-2", "buy milk")]
    out = resolve_target(
        {"ref_kind": "by_description", "description": "gym session"},
        index=FakeIndex(rows),
        session_id="sess-1",
        now=NOW,
    )
    assert out is not None
    assert out.juno_id == "j-1"
    assert out.confidence == 0.9
    assert out.resolved_via == "by_description"


def test_by_description_token_overlap_ranking_beats_filler_words() -> None:
    # "the gym one" never literally appears in any body; ranking by token
    # overlap (stopwords removed) must still pick the gym row.
    gym = _row("j-1", "gym session", modified=NOW - timedelta(minutes=3))
    milk = _row("j-2", "buy milk", modified=NOW - timedelta(minutes=1))
    out = resolve_target(
        {"ref_kind": "by_description", "description": "the gym one"},
        index=FakeIndex([gym, milk]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is not None
    assert out.juno_id == "j-1"
    # Single-token overlap is treated as weak: confidence is capped at 0.65.
    assert out.confidence == 0.65


def test_by_description_zero_overlap_returns_none_instead_of_guessing() -> None:
    rows = [_row("j-1", "gym session"), _row("j-2", "buy milk")]
    out = resolve_target(
        {"ref_kind": "by_description", "description": "the dentist appointment"},
        index=FakeIndex(rows),
        session_id="sess-1",
        now=NOW,
    )
    assert out is None


def test_by_description_kind_filter_restricts_candidates() -> None:
    note = _row("j-1", "gym session", kind="note")
    reminder = _row("j-2", "gym session tonight", kind="reminder")
    out = resolve_target(
        {"ref_kind": "by_description", "description": "gym session", "kind": "note"},
        index=FakeIndex([note, reminder]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is not None
    assert out.juno_id == "j-1"
    assert out.confidence == 0.9  # single candidate after the kind filter


def test_by_description_empty_description_returns_none() -> None:
    out = resolve_target(
        {"ref_kind": "by_description", "description": "  "},
        index=FakeIndex([_row("j-1", "gym session")]),
        session_id="sess-1",
        now=NOW,
    )
    assert out is None


# -- misc -------------------------------------------------------------------


def test_unknown_ref_kind_or_non_dict_target_returns_none() -> None:
    index = FakeIndex([_row("j-1", "call sam")])
    assert (
        resolve_target({"ref_kind": "by_magic"}, index=index, session_id="s", now=NOW)
        is None
    )
    assert resolve_target("j-1", index=index, session_id="s", now=NOW) is None  # type: ignore[arg-type]
