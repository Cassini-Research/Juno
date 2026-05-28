from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

from juno_v2.observability.history_store import read_persistent_history


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


def _day_key(ts_unix_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_unix_ms / 1000.0).astimezone()
    return dt.strftime("%Y-%m-%d")


def _last_seven_day_keys(now_ms: int) -> list[str]:
    today = datetime.fromtimestamp(now_ms / 1000.0).astimezone().date()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]


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
    ).to_dict()


__all__ = ["compute_stats_summary", "StatsCache"]
