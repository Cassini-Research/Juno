"""Action contracts.

The wire/runtime shape for a parsed action and its execution result. Kept
small and immutable; serialization helpers (``to_dict`` / ``from_dict``) are
the canonical bridge to JSON wire formats and SQLite storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    NOTE = "note"
    REMINDER = "reminder"
    ALARM = "alarm"


class ActionOperation(str, Enum):
    """Operation an Action performs against its sink.

    Phase 1 of the rehaul ships ``CREATE`` only. Phase 2 wires the others.
    Validators added in Phase 1 already accept the wider enum so v3
    envelopes parse cleanly even before the dispatcher knows what to do
    with non-create operations — those get rejected with a stable
    ``rejected_reason`` until Phase 2 lands.
    """

    CREATE = "create"
    UPDATE = "update"
    COMPLETE = "complete"
    SNOOZE = "snooze"
    DELETE = "delete"
    QUERY = "query"
    APPEND_TO = "append_to"
    REMOVE_FROM = "remove_from"


class ActionStatus(str, Enum):
    OK = "ok"
    PERMISSION_DENIED = "permission_denied"
    SINK_ERROR = "sink_error"
    TIME_PARSE_FAILED = "time_parse_failed"


_TIME_SOURCES = ("dateparser", "llm", "user_edit", "default", "salvage")
_VALID_FREQ = frozenset({"DAILY", "WEEKLY", "MONTHLY", "YEARLY"})
_VALID_BYDAY = frozenset({"MO", "TU", "WE", "TH", "FR", "SA", "SU"})
_VALID_VAGUE_BUCKETS = frozenset(
    {"later", "soon", "tonight", "morning", "afternoon", "evening", "weekend", "next_week"}
)


@dataclass(frozen=True, slots=True)
class ParsedTime:
    """A parsed datetime expression.

    ``iso`` is an ISO-8601 string with timezone offset. ``confidence`` is in
    ``[0, 1]``. ``source`` records which parser produced the value so the
    HUD/history can reflect ambiguity (e.g. amber chip for low-confidence LLM).

    The resolver layer (``time_resolver.py``) may set:

    - ``inferred=True`` when we filled in missing information (e.g. user said
      "in 4 days" with no time, defaulted to 9 AM; or "at 5pm" rolled to
      tomorrow because 5 PM had already passed today).
    - ``inference_note`` is a short human-readable explanation for the chip.
    - ``needs_confirmation=True`` for genuinely ambiguous cases the user should
      eyeball before the reminder fires (e.g. an explicit past timestamp).
    """

    iso: str
    confidence: float = 1.0
    source: str = "dateparser"
    inferred: bool = False
    inference_note: str | None = None
    needs_confirmation: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        if self.source not in _TIME_SOURCES:
            raise ValueError(f"unknown time source: {self.source}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "iso": self.iso,
            "confidence": self.confidence,
            "source": self.source,
        }
        # Only emit the optional flags when set so plain dictation utterances
        # round-trip byte-identically and downstream consumers (Swift, SQLite)
        # ignore unfamiliar keys gracefully.
        if self.inferred:
            out["inferred"] = True
        if self.inference_note:
            out["inference_note"] = self.inference_note
        if self.needs_confirmation:
            out["needs_confirmation"] = True
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedTime:
        return cls(
            iso=str(data["iso"]),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "dateparser")),
            inferred=bool(data.get("inferred", False)),
            inference_note=(
                str(data["inference_note"])
                if data.get("inference_note") is not None
                else None
            ),
            needs_confirmation=bool(data.get("needs_confirmation", False)),
        )


@dataclass(frozen=True, slots=True)
class SeriesRule:
    """Recurrence rule, ICS-shaped so it maps directly to ``EKRecurrenceRule``.

    The shape mirrors RFC-5545's RRULE so we never have to invent a new
    vocabulary. ``first_occurrence_iso`` anchors the time-of-day for the
    series — Apple's ``EKRecurrenceRule`` does *not* carry a start time,
    that lives on the parent reminder/event's due date. We carry it here
    so the resolver can fill the parent due date when the model only
    emitted the rule.
    """

    freq: str  # one of "DAILY", "WEEKLY", "MONTHLY", "YEARLY"
    interval: int = 1
    by_day: tuple[str, ...] = ()  # subset of {MO,TU,WE,TH,FR,SA,SU}
    by_month_day: tuple[int, ...] = ()
    by_month: tuple[int, ...] = ()
    count: int | None = None  # bounded count
    until_iso: str | None = None  # bounded by date
    first_occurrence_iso: str = ""  # anchors time-of-day
    tz: str = ""
    exclude_dates_iso: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.freq not in _VALID_FREQ:
            raise ValueError(f"unknown freq: {self.freq}")
        if self.interval < 1:
            raise ValueError(f"interval must be >= 1, got {self.interval}")
        for d in self.by_day:
            if d not in _VALID_BYDAY:
                raise ValueError(f"unknown by_day: {d}")
        if self.count is not None and self.count < 1:
            raise ValueError(f"count must be >= 1 if set, got {self.count}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"freq": self.freq, "interval": self.interval}
        if self.by_day:
            out["by_day"] = list(self.by_day)
        if self.by_month_day:
            out["by_month_day"] = list(self.by_month_day)
        if self.by_month:
            out["by_month"] = list(self.by_month)
        if self.count is not None:
            out["count"] = self.count
        if self.until_iso:
            out["until_iso"] = self.until_iso
        if self.first_occurrence_iso:
            out["first_occurrence_iso"] = self.first_occurrence_iso
        if self.tz:
            out["tz"] = self.tz
        if self.exclude_dates_iso:
            out["exclude_dates_iso"] = list(self.exclude_dates_iso)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeriesRule:
        return cls(
            freq=str(data["freq"]).upper(),
            interval=int(data.get("interval", 1)),
            by_day=tuple(str(x).upper() for x in data.get("by_day") or ()),
            by_month_day=tuple(int(x) for x in data.get("by_month_day") or ()),
            by_month=tuple(int(x) for x in data.get("by_month") or ()),
            count=int(data["count"]) if data.get("count") is not None else None,
            until_iso=data.get("until_iso"),
            first_occurrence_iso=str(data.get("first_occurrence_iso") or ""),
            tz=str(data.get("tz") or ""),
            exclude_dates_iso=tuple(str(x) for x in data.get("exclude_dates_iso") or ()),
        )


@dataclass(frozen=True, slots=True)
class VagueSchedule:
    """A vague schedule the user pronounced ('later', 'tonight', 'this weekend').

    The model emits the bucket; a Python resolver picks ``default_iso``;
    the HUD shows an amber chip with the resolved time and an invitation
    to tap-to-edit. ``needs_confirmation`` is always true.
    """

    bucket: str  # see _VALID_VAGUE_BUCKETS
    default_iso: str
    tz: str = ""
    needs_confirmation: bool = True

    def __post_init__(self) -> None:
        if self.bucket not in _VALID_VAGUE_BUCKETS:
            raise ValueError(f"unknown vague bucket: {self.bucket}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "default_iso": self.default_iso,
            "tz": self.tz,
            "needs_confirmation": self.needs_confirmation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VagueSchedule:
        return cls(
            bucket=str(data["bucket"]),
            default_iso=str(data["default_iso"]),
            tz=str(data.get("tz") or ""),
            needs_confirmation=bool(data.get("needs_confirmation", True)),
        )


@dataclass(frozen=True, slots=True)
class Schedule:
    """Discriminated union over instant / series / vague.

    Exactly one of ``instant`` / ``series`` / ``vague`` is non-None and
    must match ``kind``. The constructor enforces this.
    """

    kind: str  # "instant" | "series" | "vague"
    instant: ParsedTime | None = None
    series: SeriesRule | None = None
    vague: VagueSchedule | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"instant", "series", "vague"}:
            raise ValueError(f"unknown schedule kind: {self.kind}")
        non_none = [
            x for x in (self.instant, self.series, self.vague) if x is not None
        ]
        if len(non_none) != 1:
            raise ValueError(
                f"Schedule must carry exactly one variant; got {len(non_none)}"
            )
        # Variant must match kind.
        if self.kind == "instant" and self.instant is None:
            raise ValueError("kind=instant but instant is None")
        if self.kind == "series" and self.series is None:
            raise ValueError("kind=series but series is None")
        if self.kind == "vague" and self.vague is None:
            raise ValueError("kind=vague but vague is None")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.instant is not None:
            out["instant"] = self.instant.to_dict()
        if self.series is not None:
            out["series"] = self.series.to_dict()
        if self.vague is not None:
            out["vague"] = self.vague.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schedule:
        kind = str(data["kind"])
        instant_data = data.get("instant")
        series_data = data.get("series")
        vague_data = data.get("vague")
        return cls(
            kind=kind,
            instant=ParsedTime.from_dict(instant_data) if instant_data else None,
            series=SeriesRule.from_dict(series_data) if series_data else None,
            vague=VagueSchedule.from_dict(vague_data) if vague_data else None,
        )

    @property
    def primary_iso(self) -> str | None:
        """Best-effort single instant for back-compat with code paths that
        only know how to render a single due date (e.g. older HUD code).
        """
        if self.instant is not None:
            return self.instant.iso
        if self.series is not None and self.series.first_occurrence_iso:
            return self.series.first_occurrence_iso
        if self.vague is not None:
            return self.vague.default_iso
        return None


@dataclass(frozen=True, slots=True)
class Action:
    """A parsed action ready for execution.

    ``body`` is the raw extracted span before the writer runs on it; the
    runner is responsible for invoking the writer mode and replacing this
    with the cleaned-up version on the corresponding :class:`ActionResult`.

    ``when`` is the legacy single-instant due date — kept for back-compat
    with v2 envelopes and the existing Swift sink code paths. New
    extractors emit ``schedule`` (a discriminated union over
    instant/series/vague); the resolver populates ``when`` from
    ``schedule.instant`` when present so older consumers keep working.

    ``operation`` is ``CREATE`` for v2 envelopes; v3 envelopes can ride
    other operations (update/delete/etc.) once Phase 2 lands.
    """

    kind: ActionKind
    body: str
    raw_span: str
    when: ParsedTime | None = None
    schedule: Schedule | None = None
    operation: ActionOperation = ActionOperation.CREATE
    target: dict[str, Any] | None = None
    container: dict[str, Any] | None = None
    juno_id: str | None = None
    sink_id: str | None = None
    snooze_offset_seconds: int | None = None
    relative_offset_seconds: int | None = None
    link_id: str | None = None
    links_to: str | None = None
    needs_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "body": self.body,
            "raw_span": self.raw_span,
            "when": None if self.when is None else self.when.to_dict(),
        }
        # Only emit new fields when set so v2-only consumers (Swift on
        # older builds) see a byte-identical wire shape until they
        # opt-in to the new fields.
        if self.schedule is not None:
            out["schedule"] = self.schedule.to_dict()
        if self.operation is not ActionOperation.CREATE:
            out["operation"] = self.operation.value
        if self.target is not None:
            out["target"] = dict(self.target)
        if self.container is not None:
            out["container"] = dict(self.container)
        if self.juno_id:
            out["juno_id"] = self.juno_id
        if self.sink_id:
            out["sink_id"] = self.sink_id
        if self.snooze_offset_seconds is not None:
            out["snooze_offset_seconds"] = self.snooze_offset_seconds
        if self.relative_offset_seconds is not None:
            out["relative_offset_seconds"] = self.relative_offset_seconds
        if self.link_id:
            out["link_id"] = self.link_id
        if self.links_to:
            out["links_to"] = self.links_to
        if self.needs_confirmation:
            out["needs_confirmation"] = True
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        when_data = data.get("when")
        schedule_data = data.get("schedule")
        op_raw = data.get("operation")
        operation = ActionOperation(op_raw) if op_raw else ActionOperation.CREATE
        target_data = data.get("target") if isinstance(data.get("target"), dict) else None
        container_data = data.get("container") if isinstance(data.get("container"), dict) else None
        return cls(
            kind=ActionKind(data["kind"]),
            body=str(data["body"]),
            raw_span=str(data.get("raw_span", data["body"])),
            when=None if when_data is None else ParsedTime.from_dict(when_data),
            schedule=None if schedule_data is None else Schedule.from_dict(schedule_data),
            operation=operation,
            target=target_data,
            container=container_data,
            juno_id=data.get("juno_id"),
            sink_id=data.get("sink_id"),
            snooze_offset_seconds=(
                int(data["snooze_offset_seconds"])
                if data.get("snooze_offset_seconds") is not None
                else None
            ),
            relative_offset_seconds=(
                int(data["relative_offset_seconds"])
                if data.get("relative_offset_seconds") is not None
                else None
            ),
            link_id=data.get("link_id"),
            links_to=data.get("links_to"),
            needs_confirmation=bool(data.get("needs_confirmation", False)),
        )


@dataclass(frozen=True, slots=True)
class ActionResult:
    """The outcome of executing a single :class:`Action` against its sink."""

    kind: ActionKind
    status: ActionStatus
    body_preview: str
    sink_id: str | None = None
    sink_url: str | None = None
    when_iso: str | None = None
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "body_preview": self.body_preview,
            "sink_id": self.sink_id,
            "sink_url": self.sink_url,
            "when_iso": self.when_iso,
            "error": self.error,
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionResult:
        return cls(
            kind=ActionKind(data["kind"]),
            status=ActionStatus(data["status"]),
            body_preview=str(data.get("body_preview", "")),
            sink_id=data.get("sink_id"),
            sink_url=data.get("sink_url"),
            when_iso=data.get("when_iso"),
            error=data.get("error"),
            extras=dict(data.get("extras") or {}),
        )
