from __future__ import annotations

from juno_v2.contracts.commands import CommandTargetClass
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.workbench import ClientSelection


def resolve_command_target(
    *,
    text: str,
    context: TypedContextBundle,
    anchor_selection: ClientSelection | None,
    active_partial: str | None,
    command_shape: bool,
) -> tuple[CommandTargetClass, ClientSelection | None, str | None]:
    """Priority: selection > explicit span (quoted) > active utterance > recent > none."""
    if anchor_selection is not None and anchor_selection.start != anchor_selection.end and context.selected_text.strip():
        return CommandTargetClass.SELECTED_TEXT, anchor_selection, context.selected_text.strip()
    quoted = _extract_quoted_span(text)
    if quoted:
        return CommandTargetClass.EXPLICIT_SPAN, None, quoted
    if command_shape and active_partial and active_partial.strip():
        return CommandTargetClass.ACTIVE_UTTERANCE, None, active_partial.strip()
    recent = str(context.metadata.get("last_committed_text") or "").strip()
    if recent:
        start = context.metadata.get("last_committed_start")
        end = context.metadata.get("last_committed_end")
        if start is not None and end is not None:
            return CommandTargetClass.RECENT_COMMIT, ClientSelection(start=int(start), end=int(end)), recent
    if command_shape:
        return CommandTargetClass.NONE, None, None
    return CommandTargetClass.NONE, None, None


def _extract_quoted_span(text: str) -> str | None:
    for quote in ('"', "'", "\u201c", "\u201d"):
        if text.count(quote) >= 2:
            parts = text.split(quote)
            if len(parts) >= 3:
                inner = parts[1].strip()
                if inner:
                    return inner
    return None
