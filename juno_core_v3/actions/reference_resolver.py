"""Resolve v3 action target blocks against the Juno actions index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol


class _ActionsIndexLike(Protocol):
    def get(self, juno_id: str) -> dict[str, Any] | None: ...

    def find(
        self,
        *,
        body_substr: str | None = None,
        date_range: tuple[str, str] | None = None,
        list_name: str | None = None,
        kind: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]: ...

    def last_touched(self, *, kind: str | None = None) -> dict[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    juno_id: str
    confidence: float
    candidates: list[dict[str, Any]]
    resolved_via: str


def resolve_target(
    target: dict[str, Any],
    *,
    index: _ActionsIndexLike,
    session_id: str,
    now: datetime,
) -> ResolvedTarget | None:
    if not isinstance(target, dict):
        return None
    ref_kind = str(target.get("ref_kind") or "").strip().lower()
    if ref_kind == "by_id":
        return _resolve_by_id(target, index=index)
    if ref_kind == "by_pronoun":
        return _resolve_by_pronoun(target, index=index, session_id=session_id, now=now)
    if ref_kind == "by_description":
        return _resolve_by_description(target, index=index, now=now)
    if ref_kind == "by_query":
        return _resolve_by_query(target, index=index, now=now)
    return None


def _resolve_by_id(
    target: dict[str, Any],
    *,
    index: _ActionsIndexLike,
) -> ResolvedTarget | None:
    raw_id = target.get("id") or target.get("juno_id") or target.get("junoId")
    juno_id = _clean(raw_id)
    if not juno_id:
        return None
    row = index.get(juno_id)
    if row is None or not _is_resolvable(row):
        return None
    return ResolvedTarget(
        juno_id=str(row["juno_id"]),
        confidence=1.0,
        candidates=[row],
        resolved_via="by_id",
    )


def _resolve_by_pronoun(
    target: dict[str, Any],
    *,
    index: _ActionsIndexLike,
    session_id: str,
    now: datetime,
) -> ResolvedTarget | None:
    pronoun = _clean(target.get("pronoun"))
    if pronoun and pronoun not in {"this", "that", "it", "the_last_one"}:
        return None
    kind = _kind_filter(target)
    rows = [
        row
        for row in index.find(kind=kind, limit=25)
        if _is_resolvable(row)
        and _clean(row.get("last_seen_session")) == session_id
        and _within_recent_window(row, now=now, minutes=5)
    ]
    if not rows:
        return None
    candidates = rows[:5]
    confidence = 0.95 if len(candidates) == 1 else 0.6
    return ResolvedTarget(
        juno_id=str(candidates[0]["juno_id"]),
        confidence=confidence,
        candidates=candidates,
        resolved_via="by_pronoun",
    )


def _resolve_by_description(
    target: dict[str, Any],
    *,
    index: _ActionsIndexLike,
    now: datetime,
) -> ResolvedTarget | None:
    description = _clean(target.get("description"))
    if not description:
        return None
    kind = _kind_filter(target)
    date_range = _date_range_filter(target, now=now)
    list_name = _list_filter(target)

    # Pass 1: literal substring on the full description. This is the
    # cheap, high-precision case ("hiring for hardware" matches a body
    # already containing those words verbatim).
    rows = [
        row
        for row in index.find(
            body_substr=description,
            date_range=date_range,
            list_name=list_name,
            kind=kind,
            limit=25,
        )
        if _is_resolvable(row)
    ]

    if not rows:
        # Pass 2: descriptions usually carry filler words ("the 4 pm
        # hiring one") that defeat a literal LIKE on the full string.
        # Pull a wider candidate set of recent active rows in the same
        # kind / date range / list and let token-overlap ranking pick
        # the best match.
        rows = [
            row
            for row in index.find(
                date_range=date_range,
                list_name=list_name,
                kind=kind,
                limit=50,
            )
            if _is_resolvable(row)
        ]
        if not rows:
            return None

    candidates = _rank_by_description(rows, description)[:5]
    if not candidates:
        return None
    top = candidates[0]
    top_score = _description_score(top, description)
    # Reject the fallback set when nothing token-overlaps the description
    # at all — better to ask the user than mutate a random unrelated row.
    if top_score == 0:
        return None
    confidence = 0.9 if len(candidates) == 1 else 0.7
    if top_score < 2:
        # Single token in common but not exact / contains. Drop confidence
        # so the validator's ambiguity gate kicks in if there are ties.
        confidence = min(confidence, 0.65)
    return ResolvedTarget(
        juno_id=str(top["juno_id"]),
        confidence=confidence,
        candidates=candidates,
        resolved_via="by_description",
    )


def _resolve_by_query(
    target: dict[str, Any],
    *,
    index: _ActionsIndexLike,
    now: datetime,
) -> ResolvedTarget | None:
    filt = _filter(target)
    text_match = (
        _clean(filt.get("text_match"))
        or _clean(filt.get("body_substr"))
        or _clean(filt.get("description"))
        or _clean(target.get("description"))
    )
    rows = [
        row
        for row in index.find(
            body_substr=text_match,
            date_range=_date_range_filter(target, now=now),
            list_name=_list_filter(target),
            kind=_kind_filter(target),
            limit=25,
        )
        if _is_resolvable(row)
    ]
    if not rows:
        return None
    candidates = rows[:5]
    confidence = 0.85 if len(candidates) == 1 else 0.65
    return ResolvedTarget(
        juno_id=str(candidates[0]["juno_id"]),
        confidence=confidence,
        candidates=candidates,
        resolved_via="by_query",
    )


def _filter(target: dict[str, Any]) -> dict[str, Any]:
    raw = target.get("filter")
    return raw if isinstance(raw, dict) else {}


def _kind_filter(target: dict[str, Any]) -> str | None:
    filt = _filter(target)
    raw = (
        filt.get("kind")
        or filt.get("sink_kind")
        or target.get("kind")
        or target.get("sink_kind")
    )
    value = _clean(raw)
    return value if value in {"note", "reminder", "alarm"} else None


def _list_filter(target: dict[str, Any]) -> str | None:
    filt = _filter(target)
    raw = filt.get("list_name") or filt.get("listName") or target.get("list_name")
    return _clean(raw)


def _date_range_filter(target: dict[str, Any], *, now: datetime | None = None) -> tuple[str, str] | None:
    filt = _filter(target)
    raw = filt.get("date_range") or filt.get("dateRange")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        start = _clean(raw[0])
        end = _clean(raw[1])
        if start and end:
            return (start, end)
    if isinstance(raw, dict):
        start = _clean(raw.get("start_iso") or raw.get("start") or raw.get("from"))
        end = _clean(raw.get("end_iso") or raw.get("end") or raw.get("to"))
        if start and end:
            return (start, end)
    start = _clean(filt.get("start_iso") or filt.get("from_iso"))
    end = _clean(filt.get("end_iso") or filt.get("to_iso"))
    if start and end:
        return (start, end)
    when_text = _clean(
        filt.get("when_text")
        or filt.get("time_text")
        or target.get("when_text")
        or target.get("time_text")
    )
    if when_text and now is not None:
        try:
            from juno_core_v3.actions.timeparse import parse_when

            parsed = parse_when(when_text, now=now)
        except Exception:
            parsed = None
        if parsed is not None:
            try:
                dt = datetime.fromisoformat(parsed.iso)
            except ValueError:
                return None
            start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(days=1, microseconds=-1)
            return (start_dt.isoformat(), end_dt.isoformat())
    return None


_DESCRIPTION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "those",
        "these",
        "my",
        "your",
        "our",
        "his",
        "her",
        "their",
        "its",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "with",
        "and",
        "or",
        "from",
        "by",
        "about",
        "around",
        "one",
        "ones",
        "thing",
        "things",
        "reminder",
        "reminders",
        "alarm",
        "alarms",
        "note",
        "notes",
        "list",
        "lists",
    }
)


def _description_tokens(text: str) -> set[str]:
    """Tokens to compare across description ↔ body, minus filler words.

    Reminders, alarms, notes, and pronouns ("the", "this", "one") are
    pruned because they're carried by every reference and would
    artificially inflate similarity. Tokens of length 1–2 are also
    dropped because numbers like "the 4" are too ambiguous on their
    own — the ranking still rewards rows that match longer body words.
    """
    normalized = _normalize(text)
    return {tok for tok in normalized.split() if len(tok) > 2 and tok not in _DESCRIPTION_STOPWORDS}


def _description_score(row: dict[str, Any], description: str) -> int:
    """Higher = better match. Exact = 100, contains = 50, then token overlap."""
    body = str(row.get("body_normalized") or "")
    needle = _normalize(description)
    if not needle:
        return 0
    if body == needle:
        return 100
    if needle in body:
        return 50
    body_tokens = _description_tokens(body)
    needle_tokens = _description_tokens(description)
    if not body_tokens or not needle_tokens:
        return 0
    return len(body_tokens & needle_tokens)


def _rank_by_description(rows: list[dict[str, Any]], description: str) -> list[dict[str, Any]]:
    """Rank candidates by description match.

    Two-tier scoring lets us resolve "the gym one" → body "gym session"
    even when the user's reference only shares one meaningful token
    with the stored body. Without token overlap (the original
    contains-only ranking), a description with filler words like
    "move the 4 pm hiring reminder" silently failed every lookup
    because no body literally contained that whole phrase.
    """
    def score(row: dict[str, Any]) -> tuple[int, int]:
        modified = _int_ms(row.get("last_modified_at"))
        return (_description_score(row, description), modified)

    return sorted(rows, key=score, reverse=True)


def _is_resolvable(row: dict[str, Any]) -> bool:
    if row.get("deleted_at") is not None:
        return False
    return str(row.get("status") or "active").lower() == "active"


def _within_recent_window(row: dict[str, Any], *, now: datetime, minutes: int) -> bool:
    modified_ms = _int_ms(row.get("last_modified_at") or row.get("created_at"))
    if modified_ms <= 0:
        return False
    now_utc = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    cutoff_ms = int((now_utc - timedelta(minutes=minutes)).timestamp() * 1000)
    return modified_ms >= cutoff_ms


def _int_ms(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize(text: str) -> str:
    lowered = (text or "").lower()
    stripped = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip()


__all__ = ["ResolvedTarget", "resolve_target"]
