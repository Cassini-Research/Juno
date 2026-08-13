from __future__ import annotations

from datetime import datetime, timedelta

from juno_v2.workbench import stats


def test_stats_summary_includes_seven_thirty_and_all_time_periods(monkeypatch, tmp_path):
    local_tz = datetime.now().astimezone().tzinfo
    now = datetime(2026, 7, 28, 12, 0, tzinfo=local_tz)

    def entry(*, days_ago: int, transcript: str, app: str) -> dict:
        timestamp = now - timedelta(days=days_ago)
        return {
            "ts_unix_ms": int(timestamp.timestamp() * 1000),
            "transcript": transcript,
            "context": {"app_name": app},
        }

    entries = [
        entry(days_ago=0, transcript="one two three four", app="Notes"),
        entry(days_ago=1, transcript="five six", app="Slack"),
        entry(days_ago=10, transcript="seven eight nine", app="Notes"),
        entry(days_ago=40, transcript="ten eleven twelve thirteen fourteen", app="Mail"),
        {"ts_unix_ms": 0, "transcript": "ignored"},
        entry(days_ago=2, transcript="", app="Notes"),
    ]
    monkeypatch.setattr(stats.time, "time", lambda: now.timestamp())
    monkeypatch.setattr(stats, "read_persistent_history", lambda *_args, **_kwargs: entries)

    payload = stats.compute_stats_summary(tmp_path)

    assert payload["words_today"] == 4
    assert payload["words_week"] == 6
    assert payload["words_by_day"][-2:] == [2, 4]

    periods = {period["id"]: period for period in payload["periods"]}
    assert set(periods) == {"7d", "30d", "all"}
    assert periods["7d"]["total_words"] == 6
    assert periods["7d"]["dictations"] == 2
    assert periods["30d"]["total_words"] == 9
    assert periods["30d"]["dictations"] == 3
    assert periods["all"]["total_words"] == 14
    assert periods["all"]["dictations"] == 4
    assert sum(periods["all"]["words_by_bucket"]) == 14
    assert periods["all"]["top_apps"][0] == {
        "name": "Notes",
        "words": 7,
    }


def test_stats_periods_return_zero_filled_buckets_without_history(monkeypatch, tmp_path):
    local_tz = datetime.now().astimezone().tzinfo
    now = datetime(2026, 7, 28, 12, 0, tzinfo=local_tz)
    monkeypatch.setattr(stats.time, "time", lambda: now.timestamp())
    monkeypatch.setattr(stats, "read_persistent_history", lambda *_args, **_kwargs: [])

    payload = stats.compute_stats_summary(tmp_path)
    periods = {period["id"]: period for period in payload["periods"]}

    assert len(periods["7d"]["words_by_bucket"]) == 7
    assert len(periods["30d"]["words_by_bucket"]) == 15
    assert periods["all"]["words_by_bucket"] == [0]
    assert all(value == 0 for period in periods.values() for value in period["words_by_bucket"])
