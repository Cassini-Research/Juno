from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from juno_v2.contracts.context import TypedContextBundle


class ContextFieldKey(str, Enum):
    """Stable keys for provenance and budget accounting."""

    SELECTED_TEXT = "selected_text"
    FOCUSED_TEXT_BEFORE = "focused_text_before"
    FOCUSED_TEXT_AFTER = "focused_text_after"
    CLIPBOARD_TEXT = "clipboard_text"
    FIELD_TEXT_EXCERPT = "field_text_excerpt"
    APP_NAME = "app_name"
    WINDOW_TITLE = "window_title"


class FieldProvenance(str, Enum):
    """Where a field value originated (context plane contract, Phase 1 minimum)."""

    UNKNOWN = "unknown"
    STATIC = "static"
    WORKBENCH_SYNC = "workbench_sync"
    DESKTOP_PROVIDER = "desktop_provider"
    TEST_FIXTURE = "test_fixture"


@dataclass(slots=True)
class ContextPacketBudgets:
    """Hard caps to stop context drift (North Star: budgets)."""

    max_total_chars: int = 48_000
    max_selected_chars: int = 16_000
    max_clipboard_chars: int = 8_000
    max_around_chars: int = 12_000  # before + after combined cap in enforce
    max_field_excerpt_chars: int = 8_000


def _truncate(s: str, max_len: int) -> tuple[str, bool]:
    if len(s) <= max_len:
        return s, False
    return s[:max_len], True


@dataclass(slots=True)
class ContextPacket:
    """Context plane payload with per-field provenance (Phase 1)."""

    selected_text: str = ""
    focused_text_before: str = ""
    focused_text_after: str = ""
    clipboard_text: str = ""
    field_text_excerpt: str = ""
    app_name: str | None = None
    window_title: str | None = None
    candidate_entities: list[str] = field(default_factory=list)
    provenance: dict[str, FieldProvenance] = field(default_factory=dict)
    truncation_applied: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def enforce_budgets(self, budgets: ContextPacketBudgets | None = None) -> "ContextPacket":
        """Return a new packet with fields truncated per budgets; records truncation flags."""
        b = budgets or ContextPacketBudgets()
        prov = dict(self.provenance)
        trunc = dict(self.truncation_applied)

        sel, t1 = _truncate(self.selected_text, b.max_selected_chars)
        if t1:
            trunc[ContextFieldKey.SELECTED_TEXT.value] = True
        clip, t2 = _truncate(self.clipboard_text, b.max_clipboard_chars)
        if t2:
            trunc[ContextFieldKey.CLIPBOARD_TEXT.value] = True

        half_around = max(1, b.max_around_chars // 2)
        fb, t3a = _truncate(self.focused_text_before, half_around)
        fa, t3b = _truncate(self.focused_text_after, half_around)
        if t3a:
            trunc[ContextFieldKey.FOCUSED_TEXT_BEFORE.value] = True
        if t3b:
            trunc[ContextFieldKey.FOCUSED_TEXT_AFTER.value] = True

        excerpt, t4 = _truncate(self.field_text_excerpt, b.max_field_excerpt_chars)
        if t4:
            trunc[ContextFieldKey.FIELD_TEXT_EXCERPT.value] = True

        out = ContextPacket(
            selected_text=sel,
            focused_text_before=fb,
            focused_text_after=fa,
            clipboard_text=clip,
            field_text_excerpt=excerpt,
            app_name=self.app_name,
            window_title=self.window_title,
            candidate_entities=list(self.candidate_entities),
            provenance=prov,
            truncation_applied=trunc,
            metadata=dict(self.metadata),
        )
        total = (
            len(out.selected_text)
            + len(out.focused_text_before)
            + len(out.focused_text_after)
            + len(out.clipboard_text)
            + len(out.field_text_excerpt)
            + len(out.app_name or "")
            + len(out.window_title or "")
        )
        if total > b.max_total_chars:
            # Shrink largest text fields deterministically: clear clipboard then trim selected
            out.metadata = dict(out.metadata)
            out.metadata["budget_total_chars"] = total
            out.metadata["budget_exceeded"] = True
            out.clipboard_text = ""
            out.selected_text, _ = _truncate(out.selected_text, max(0, b.max_total_chars // 4))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_text": self.selected_text,
            "focused_text_before": self.focused_text_before,
            "focused_text_after": self.focused_text_after,
            "clipboard_text": self.clipboard_text,
            "field_text_excerpt": self.field_text_excerpt,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "candidate_entities": list(self.candidate_entities),
            "provenance": {k: v.value for k, v in self.provenance.items()},
            "truncation_applied": dict(self.truncation_applied),
            "metadata": dict(self.metadata),
        }


def build_context_packet_from_typed_bundle(
    bundle: "TypedContextBundle",
    *,
    default_provenance: FieldProvenance = FieldProvenance.UNKNOWN,
) -> ContextPacket:
    """Map existing v2 ``TypedContextBundle`` into a v3 ``ContextPacket`` with default provenance."""
    prov = {
        ContextFieldKey.SELECTED_TEXT.value: default_provenance,
        ContextFieldKey.FOCUSED_TEXT_BEFORE.value: default_provenance,
        ContextFieldKey.FOCUSED_TEXT_AFTER.value: default_provenance,
        ContextFieldKey.CLIPBOARD_TEXT.value: default_provenance,
        ContextFieldKey.FIELD_TEXT_EXCERPT.value: default_provenance,
        ContextFieldKey.APP_NAME.value: default_provenance,
        ContextFieldKey.WINDOW_TITLE.value: default_provenance,
    }
    return ContextPacket(
        selected_text=bundle.selected_text,
        focused_text_before=bundle.focused_text_before,
        focused_text_after=bundle.focused_text_after,
        clipboard_text=bundle.clipboard_text,
        field_text_excerpt=bundle.field_text_excerpt,
        app_name=bundle.app_name,
        window_title=bundle.window_title,
        candidate_entities=list(bundle.candidate_entities),
        provenance=prov,
        metadata={"source": "typed_context_bundle"},
    )
