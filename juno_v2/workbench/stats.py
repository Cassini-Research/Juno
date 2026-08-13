from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import TypedDict

from juno_v2.observability.history_store import read_persistent_history


class StatsAppPayload(TypedDict):
    name: str
    words: int


class StatsPeriodPayload(TypedDict):
    id: str
    total_words: int
    dictations: int
    time_saved_s: int
    bucket_start_dates: list[str]
    bucket_end_dates: list[str]
    words_by_bucket: list[int]
    top_apps: list[StatsAppPayload]


class StatsSummaryPayload(TypedDict):
    ok: bool
    words_today: int
    words_week: int
    apps_today: int
    time_saved_s: int
    time_saved_min: int
    computed_at_unix_ms: int
    words_by_day: list[int]
    apps_today_top: list[str]
    top_app_today: str | None
    periods: list[StatsPeriodPayload]


def _day_key(ts_unix_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_unix_ms / 1000.0).astimezone()
    return dt.strftime("%Y-%m-%d")


def _last_seven_day_keys(now_ms: int) -> list[str]:
    today = datetime.fromtimestamp(now_ms / 1000.0).astimezone().date()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]


@dataclass(frozen=True, slots=True)
class _UsageRecord:
    day: date
    words: int
    app_name: str | None


def _period_payload(
    *,
    period_id: str,
    records: list[_UsageRecord],
    start_day: date,
    end_day: date,
    max_buckets: int,
) -> StatsPeriodPayload:
    included = [record for record in records if start_day <= record.day <= end_day]
    total_words = sum(record.words for record in included)

    app_words: dict[str, int] = defaultdict(int)
    for record in included:
        if record.app_name:
            app_words[record.app_name] += record.words

    top_apps: list[StatsAppPayload] = [
        {
            "name": app,
            "words": app_words[app],
        }
        for app in sorted(
            app_words,
            key=lambda name: (-app_words[name], name.casefold()),
        )[:5]
    ]

    span_days = max(1, (end_day - start_day).days + 1)
    bucket_days = max(1, ceil(span_days / max(1, max_buckets)))
    bucket_count = ceil(span_days / bucket_days)
    bucket_starts = [
        start_day + timedelta(days=index * bucket_days)
        for index in range(bucket_count)
    ]
    bucket_ends = [
        min(end_day, bucket_start + timedelta(days=bucket_days - 1))
        for bucket_start in bucket_starts
    ]
    words_by_bucket = [0] * bucket_count
    for record in included:
        index = min((record.day - start_day).days // bucket_days, bucket_count - 1)
        words_by_bucket[index] += record.words

    return {
        "id": period_id,
        "total_words": total_words,
        "dictations": len(included),
        "time_saved_s": int(round((total_words / 133.0) * 60.0)),
        "bucket_start_dates": [value.isoformat() for value in bucket_starts],
        "bucket_end_dates": [value.isoformat() for value in bucket_ends],
        "words_by_bucket": words_by_bucket,
        "top_apps": top_apps,
    }


@dataclass(slots=True)
class StatsSummary:
    words_today: int
    words_week: int
    apps_today: int
    time_saved_s: int
    time_saved_min: int
    computed_at_unix_ms: int
    # New (additive, optional fields for older shells):
    words_by_day: list[int] = field(default_factory=list)
    apps_today_top: list[str] = field(default_factory=list)
    top_app_today: str | None = None
    periods: list[StatsPeriodPayload] = field(default_factory=list)

    def to_dict(self) -> StatsSummaryPayload:
        return {
            "ok": True,
            "words_today": self.words_today,
            "words_week": self.words_week,
            "apps_today": self.apps_today,
            "time_saved_s": self.time_saved_s,
            "time_saved_min": self.time_saved_min,
            "computed_at_unix_ms": self.computed_at_unix_ms,
            "words_by_day": self.words_by_day,
            "apps_today_top": self.apps_today_top,
            "top_app_today": self.top_app_today,
            "periods": self.periods,
        }


class StatsCache:
    def __init__(self) -> None:
        self._cached: tuple[int, StatsSummaryPayload] | None = None

    def get_or_compute(self, *, log_dir: Path, ttl_s: int = 60) -> StatsSummaryPayload:
        now = int(time.time())
        if self._cached is not None:
            ts, payload = self._cached
            if now - ts <= ttl_s:
                return payload
        payload = compute_stats_summary(log_dir)
        self._cached = (now, payload)
        return payload


def compute_stats_summary(log_dir: Path) -> StatsSummaryPayload:
    entries = read_persistent_history(log_dir, limit=50_000)
    now_ms = int(time.time() * 1000)
    today = _day_key(now_ms)

    week_cutoff_ms = now_ms - (7 * 86400 * 1000)
    words_today = 0
    words_week = 0
    words_by_day_map: dict[str, int] = defaultdict(int)
    apps_today_count: dict[str, int] = defaultdict(int)
    usage_records: list[_UsageRecord] = []
    for e in entries:
        try:
            ts = int(e.get("ts_unix_ms") or 0)
        except (TypeError, ValueError):
            ts = 0
        if ts <= 0:
            continue
        transcript = str(e.get("transcript") or "")
        words = len([tok for tok in transcript.split() if tok])
        if ts >= week_cutoff_ms:
            words_week += words
            words_by_day_map[_day_key(ts)] += words
        if _day_key(ts) == today:
            words_today += words
            ctx = e.get("context") or {}
            app = ctx.get("app_name") if isinstance(ctx, dict) else None
            if isinstance(app, str) and app.strip():
                apps_today_count[app.strip()] += 1
        if words > 0:
            ctx = e.get("context") or {}
            raw_app = ctx.get("app_name") if isinstance(ctx, dict) else None
            app_name = raw_app.strip() if isinstance(raw_app, str) and raw_app.strip() else None
            usage_records.append(
                _UsageRecord(
                    day=datetime.fromtimestamp(ts / 1000.0).astimezone().date(),
                    words=words,
                    app_name=app_name,
                )
            )

    # Last 7 days, oldest → today, missing days → 0.
    seven_keys = _last_seven_day_keys(now_ms)
    words_by_day = [words_by_day_map.get(k, 0) for k in seven_keys]

    # Apps today, sorted by frequency (desc), name asc as tiebreaker.
    apps_today_top = sorted(
        apps_today_count.keys(),
        key=lambda name: (-apps_today_count[name], name),
    )
    top_app_today = apps_today_top[0] if apps_today_top else None

    # 133 WPM avg typing speed, as per plan. Keep the raw seconds so the
    # shell can show value even on short sessions; keep minutes for older
    # clients that only know the legacy field.
    time_saved_s = int(round((words_today / 133.0) * 60.0))
    time_saved_min = time_saved_s // 60
    today_date = datetime.fromtimestamp(now_ms / 1000.0).astimezone().date()
    first_usage_day = min(
        (record.day for record in usage_records),
        default=today_date,
    )
    periods = [
        _period_payload(
            period_id="7d",
            records=usage_records,
            start_day=today_date - timedelta(days=6),
            end_day=today_date,
            max_buckets=7,
        ),
        _period_payload(
            period_id="30d",
            records=usage_records,
            start_day=today_date - timedelta(days=29),
            end_day=today_date,
            max_buckets=15,
        ),
        _period_payload(
            period_id="all",
            records=usage_records,
            start_day=first_usage_day,
            end_day=today_date,
            max_buckets=12,
        ),
    ]
    return StatsSummary(
        words_today=words_today,
        words_week=words_week,
        apps_today=len(apps_today_count),
        time_saved_s=time_saved_s,
        time_saved_min=time_saved_min,
        computed_at_unix_ms=now_ms,
        words_by_day=words_by_day,
        apps_today_top=apps_today_top,
        top_app_today=top_app_today,
        periods=periods,
    ).to_dict()


__all__ = ["compute_stats_summary", "StatsCache"]
