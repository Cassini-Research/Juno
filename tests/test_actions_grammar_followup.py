"""Unit tests for the action grammar and the followup correction grammar.

Covers juno_core_v3/actions/grammar.py (``strip_wake``, ``parse_actions``)
and juno_core_v3/actions/followup.py (``parse_followup``,
``followup_action_for_row``). All time-dependent calls pass an explicit
``now`` (Tuesday 2026-06-09 10:00 UTC).
"""

from __future__ import annotations

from datetime import datetime, timezone

from juno_core_v3.actions.contracts import ActionKind, ActionOperation
from juno_core_v3.actions.followup import (
    FollowupIntent,
    followup_action_for_row,
    parse_followup,
)
from juno_core_v3.actions.grammar import parse_actions, strip_wake

UTC = timezone.utc
NOW = datetime(2026, 6, 9, 10, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# strip_wake
# ---------------------------------------------------------------------------


def test_strip_wake_plain_juno() -> None:
    assert strip_wake("Juno take a note buy milk") == "take a note buy milk"


def test_strip_wake_hey_juno_with_punctuation() -> None:
    assert strip_wake("Hey Juno, remind me to stretch") == "remind me to stretch"


def test_strip_wake_leading_punctuation_tolerated() -> None:
    assert strip_wake("... juno note this idea") == "note this idea"


def test_strip_wake_bare_wake_word_yields_empty_string() -> None:
    # Wake word present but nothing after it: empty string, not None, so
    # callers can distinguish "addressed Juno" from "not addressed".
    assert strip_wake("juno") == ""


def test_strip_wake_mid_sentence_mention_is_not_wake() -> None:
    assert strip_wake("I told juno to take a note") is None


def test_strip_wake_prefix_word_is_not_wake() -> None:
    assert strip_wake("junotown is lovely this time of year") is None


def test_strip_wake_rejects_product_version_mentions() -> None:
    assert strip_wake("juno v2 demo") is None
    assert strip_wake("Juno v 2 is shipping") is None
    assert strip_wake("juno v two demo") is None


def test_strip_wake_empty_input() -> None:
    assert strip_wake("") is None


# ---------------------------------------------------------------------------
# parse_actions — gating
# ---------------------------------------------------------------------------


def test_no_wake_word_returns_none() -> None:
    assert parse_actions("take a note buy milk", now=NOW) is None


def test_wake_word_without_verb_returns_none() -> None:
    assert parse_actions("juno hello there how are you", now=NOW) is None


def test_verb_with_empty_body_returns_none() -> None:
    assert parse_actions("juno take a note", now=NOW) is None


def test_empty_transcript_returns_none() -> None:
    assert parse_actions("", now=NOW) is None
    assert parse_actions("   ", now=NOW) is None


# ---------------------------------------------------------------------------
# parse_actions — NOTE extraction
# ---------------------------------------------------------------------------


def test_note_extraction_keeps_body_verbatim() -> None:
    actions = parse_actions("juno take a note buy milk and eggs", now=NOW)
    assert actions is not None and len(actions) == 1
    (note,) = actions
    assert note.kind is ActionKind.NOTE
    assert note.body == "buy milk and eggs"
    assert note.when is None
    assert note.raw_span == "take a note buy milk and eggs"


def test_note_leading_connective_is_stripped() -> None:
    actions = parse_actions("juno note that the wifi password changed", now=NOW)
    assert actions is not None and len(actions) == 1
    assert actions[0].body == "the wifi password changed"


# ---------------------------------------------------------------------------
# parse_actions — REMINDER with when clauses
# ---------------------------------------------------------------------------


def test_reminder_trailing_when_clause_extracted() -> None:
    actions = parse_actions("juno remind me to call sam tomorrow at 9am", now=NOW)
    assert actions is not None and len(actions) == 1
    (rem,) = actions
    assert rem.kind is ActionKind.REMINDER
    assert rem.body == "call sam"
    assert rem.when is not None
    assert datetime.fromisoformat(rem.when.iso) == datetime(
        2026, 6, 10, 9, 0, tzinfo=UTC
    )


def test_reminder_bare_tomorrow_gets_9am_default() -> None:
    actions = parse_actions("juno remind me to water the plants tomorrow", now=NOW)
    assert actions is not None and len(actions) == 1
    (rem,) = actions
    assert rem.body == "water the plants"
    assert rem.when is not None
    assert datetime.fromisoformat(rem.when.iso) == datetime(
        2026, 6, 10, 9, 0, tzinfo=UTC
    )
    assert rem.when.inferred is True


def test_reminder_leading_when_clause_extracted() -> None:
    # Front-loaded time clause: "remind me at 5pm to leave for the store".
    actions = parse_actions(
        "juno remind me at 5pm to leave for the store", now=NOW
    )
    assert actions is not None and len(actions) == 1
    (rem,) = actions
    assert rem.body == "leave for the store"
    assert rem.when is not None
    # Bare clocks resolve in the system tz; assert wall-clock fields only.
    when_dt = datetime.fromisoformat(rem.when.iso)
    assert (when_dt.hour, when_dt.minute) == (17, 0)


def test_reminder_without_when_clause_has_no_when() -> None:
    actions = parse_actions("juno remind me to renew the passport", now=NOW)
    assert actions is not None and len(actions) == 1
    (rem,) = actions
    assert rem.body == "renew the passport"
    assert rem.when is None


# ---------------------------------------------------------------------------
# parse_actions — compound utterances
# ---------------------------------------------------------------------------


def test_compound_note_then_reminder() -> None:
    actions = parse_actions(
        "juno take a note buy milk and remind me to call sam tomorrow", now=NOW
    )
    assert actions is not None and len(actions) == 2
    note, rem = actions
    assert note.kind is ActionKind.NOTE
    assert note.body == "buy milk"  # trailing "and" separator stripped
    assert rem.kind is ActionKind.REMINDER
    assert rem.body == "call sam"
    assert rem.when is not None
    assert datetime.fromisoformat(rem.when.iso) == datetime(
        2026, 6, 10, 9, 0, tzinfo=UTC
    )


def test_compound_two_reminders_each_get_their_own_when() -> None:
    actions = parse_actions(
        "juno remind me to stretch tomorrow at 8am and remind me to "
        "submit the report tomorrow at 5pm",
        now=NOW,
    )
    assert actions is not None and len(actions) == 2
    first, second = actions
    assert first.body == "stretch"
    assert datetime.fromisoformat(first.when.iso) == datetime(
        2026, 6, 10, 8, 0, tzinfo=UTC
    )
    assert second.body == "submit the report"
    assert datetime.fromisoformat(second.when.iso) == datetime(
        2026, 6, 10, 17, 0, tzinfo=UTC
    )


def test_alarm_with_time_only_body_is_labeled_alarm() -> None:
    actions = parse_actions("juno set an alarm at 7am tomorrow", now=NOW)
    assert actions is not None and len(actions) == 1
    (alarm,) = actions
    assert alarm.kind is ActionKind.ALARM
    assert alarm.body == "Alarm"
    assert alarm.when is not None
    assert datetime.fromisoformat(alarm.when.iso) == datetime(
        2026, 6, 10, 7, 0, tzinfo=UTC
    )


# ---------------------------------------------------------------------------
# parse_followup — intent classification
# ---------------------------------------------------------------------------


def test_followup_confirmations() -> None:
    for text in ("Yes", "yeah", "yep", "Looks good.", "that's right"):
        intent = parse_followup(text)
        assert intent is not None, text
        assert intent.kind == "confirm"


def test_followup_cancellations() -> None:
    for text in ("cancel", "Cancel that", "never mind", "nevermind", "undo that", "delete that"):
        intent = parse_followup(text)
        assert intent is not None, text
        assert intent.kind == "cancel"


def test_followup_recurrence_corrections() -> None:
    daily = parse_followup("make it daily")
    assert daily == FollowupIntent(kind="correct", field="recurrence", new_value="daily")
    every_day = parse_followup("every day")
    assert every_day is not None and every_day.new_value == "daily"
    weekday = parse_followup("every weekday")
    assert weekday == FollowupIntent(
        kind="correct", field="recurrence", new_value="weekday"
    )


def test_followup_time_corrections() -> None:
    assert parse_followup("no I meant 6pm") == FollowupIntent(
        kind="correct", field="time", new_value="6pm"
    )
    # "X not Y" keeps only the corrected value.
    assert parse_followup("i meant 7am not 7pm") == FollowupIntent(
        kind="correct", field="time", new_value="7am"
    )
    assert parse_followup("change it to 8pm") == FollowupIntent(
        kind="correct", field="time", new_value="8pm"
    )
    assert parse_followup("move to tomorrow") == FollowupIntent(
        kind="correct", field="time", new_value="tomorrow"
    )


def test_followup_unrelated_text_returns_none() -> None:
    assert parse_followup("what's the weather like") is None
    assert parse_followup("") is None


# ---------------------------------------------------------------------------
# followup_action_for_row
# ---------------------------------------------------------------------------

_PREVIOUS_ROW = {
    "sink_kind": "reminder",
    "juno_id": "j-1",
    "sink_id": "ek-9",
    "body_normalized": "call sam",
    "due_iso": "2026-06-10T09:00:00+00:00",
}


def test_cancel_builds_delete_action_targeting_row_by_id() -> None:
    action = followup_action_for_row(
        FollowupIntent(kind="cancel"), _PREVIOUS_ROW, now=NOW
    )
    assert action is not None
    assert action.kind is ActionKind.REMINDER
    assert action.operation is ActionOperation.DELETE
    assert action.body == "call sam"
    assert action.juno_id == "j-1"
    assert action.sink_id == "ek-9"
    assert action.target == {
        "ref_kind": "by_id",
        "id": "j-1",
        "resolved_via": "by_id",
        "confidence": 1.0,
    }


def test_time_correction_builds_update_with_instant_schedule() -> None:
    action = followup_action_for_row(
        FollowupIntent(kind="correct", field="time", new_value="6pm"),
        _PREVIOUS_ROW,
        now=NOW,
    )
    assert action is not None
    assert action.operation is ActionOperation.UPDATE
    assert action.when is not None
    when_dt = datetime.fromisoformat(action.when.iso)
    assert (when_dt.hour, when_dt.minute) == (18, 0)
    assert action.schedule is not None
    assert action.schedule.kind == "instant"
    assert action.schedule.instant is action.when
    assert action.juno_id == "j-1"


def test_recurrence_correction_daily_anchors_on_previous_due() -> None:
    action = followup_action_for_row(
        FollowupIntent(kind="correct", field="recurrence", new_value="daily"),
        _PREVIOUS_ROW,
        now=NOW,
    )
    assert action is not None
    assert action.operation is ActionOperation.UPDATE
    assert action.schedule is not None and action.schedule.kind == "series"
    rule = action.schedule.series
    assert rule.freq == "DAILY"
    assert rule.interval == 1
    assert rule.first_occurrence_iso == "2026-06-10T09:00:00+00:00"


def test_recurrence_correction_weekday_uses_mo_through_fr() -> None:
    action = followup_action_for_row(
        FollowupIntent(kind="correct", field="recurrence", new_value="weekday"),
        _PREVIOUS_ROW,
        now=NOW,
    )
    assert action is not None
    rule = action.schedule.series
    assert rule.freq == "WEEKLY"
    assert rule.by_day == ("MO", "TU", "WE", "TH", "FR")
    assert rule.first_occurrence_iso == "2026-06-10T09:00:00+00:00"


def test_recurrence_correction_without_due_falls_back_to_now() -> None:
    row = dict(_PREVIOUS_ROW, due_iso="")
    action = followup_action_for_row(
        FollowupIntent(kind="correct", field="recurrence", new_value="daily"),
        row,
        now=NOW,
    )
    assert action is not None
    assert action.schedule.series.first_occurrence_iso == NOW.isoformat()


def test_confirm_intent_builds_no_action() -> None:
    assert (
        followup_action_for_row(FollowupIntent(kind="confirm"), _PREVIOUS_ROW, now=NOW)
        is None
    )


def test_missing_juno_id_or_unknown_kind_builds_no_action() -> None:
    assert (
        followup_action_for_row(
            FollowupIntent(kind="cancel"), {"sink_kind": "reminder"}, now=NOW
        )
        is None
    )
    assert (
        followup_action_for_row(
            FollowupIntent(kind="cancel"),
            {"sink_kind": "task", "juno_id": "x"},
            now=NOW,
        )
        is None
    )


def test_unparseable_time_correction_builds_no_action() -> None:
    action = followup_action_for_row(
        FollowupIntent(kind="correct", field="time", new_value="gibberish value"),
        _PREVIOUS_ROW,
        now=NOW,
    )
    assert action is None
