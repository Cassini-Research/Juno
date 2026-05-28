from __future__ import annotations

from dataclasses import dataclass

from juno_v2.contracts.modes import ModePolicy, ModeSelection
from juno_v2.contracts.writer import WriterMode


@dataclass(slots=True)
class WriterState:
    mode: WriterMode = WriterMode.DEFAULT_SURFACE
    structure_mode: str | None = None
    structure_item_index: int = 1
    mode_policy: ModePolicy | None = None
    mode_selection: ModeSelection | None = None
    writer_tone_addon: str | None = None
