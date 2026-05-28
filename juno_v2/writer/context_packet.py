from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from juno_core_v3.context.plane import ContextPlane, ContextPlaneConfig
from juno_v2.contracts.context import TypedContextBundle


@dataclass(slots=True)
class WriterContextPacket:
    """Bounded writer-facing context derived from ContextPlane + runtime bundle."""

    before_text_window: str
    after_text_window: str
    selection_text: str
    surface_category: str | None
    app_name: str | None
    focused_file_path: str | None
    symbol_under_cursor: str | None
    nearby_terms: tuple[str, ...]
    effective_mode: str | None
    style_card: dict[str, Any] | None
    active_transform: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["nearby_terms"] = list(self.nearby_terms)
        return d


def build_writer_context_packet(
    bundle: TypedContextBundle,
    *,
    effective_mode: str | None,
    max_window: int = 256,
    active_transform: str | None = None,
    style_card: dict[str, Any] | None = None,
) -> WriterContextPacket:
    plane = ContextPlane(ContextPlaneConfig())
    pkt = plane.build_from_typed_bundle(bundle, surface_id="writer_packet")
    before = (pkt.focused_text_before or "")[-max_window:]
    after = (pkt.focused_text_after or "")[:max_window]
    sel = pkt.selected_text or ""
    surf = bundle.app_category or pkt.metadata.get("app_category")
    terms: list[str] = []
    for ent in (bundle.candidate_entities or [])[:12]:
        if ent and ent not in terms:
            terms.append(ent)
    return WriterContextPacket(
        before_text_window=before,
        after_text_window=after,
        selection_text=sel,
        surface_category=str(surf) if surf else None,
        app_name=pkt.app_name or bundle.app_name,
        focused_file_path=bundle.focused_file_path,
        symbol_under_cursor=bundle.symbol_under_cursor,
        nearby_terms=tuple(terms),
        effective_mode=effective_mode,
        style_card=style_card,
        active_transform=active_transform,
    )
