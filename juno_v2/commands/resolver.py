from __future__ import annotations

import re

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
    focused_tail = focused_text_before_tail(context.focused_text_before)
    if command_shape and focused_tail is not None:
        return (
            CommandTargetClass.FOCUSED_TEXT,
            ClientSelection(start=focused_tail["start"], end=focused_tail["end"]),
            focused_tail["text"],
        )
    if command_shape:
        return CommandTargetClass.NONE, None, None
    return CommandTargetClass.NONE, None, None


def focused_text_before_tail(text: str) -> dict | None:
    """Last paragraph of the focused field's pre-caret text.

    The shell replaces this target by deleting ``delete_chars`` characters
    back from the caret, so the count runs from the paragraph start to the
    caret INCLUDING any trailing whitespace — counting only the stripped
    paragraph chops the paragraph mid-word when the caret sits after a
    newline. ``trailing_text`` is that whitespace, re-appended to the
    transformed output so the caret's blank-line state survives the rewrite.
    Offsets are relative to the (possibly clipped) snapshot window whose end
    is the caret.
    """
    raw = str(text or "")
    stripped = raw.rstrip()
    if not stripped:
        return None
    trailing_text = raw[len(stripped):]
    end = len(stripped)
    breaks = list(re.finditer(r"\n\s*\n", stripped))
    start = breaks[-1].end() if breaks else 0
    while start < end and stripped[start].isspace():
        start += 1
    candidate = stripped[start:end].strip()
    if len(candidate) > 1000:
        candidate = candidate[-1000:].lstrip()
        start = end - len(candidate)
    if not candidate:
        return None
    return {
        "start": start,
        "end": end,
        "text": candidate,
        "delete_chars": len(raw) - start,
        "trailing_text": trailing_text,
    }


def _extract_quoted_span(text: str) -> str | None:
    for quote in ('"', "'", "\u201c", "\u201d"):
        if text.count(quote) >= 2:
            parts = text.split(quote)
            if len(parts) >= 3:
                inner = parts[1].strip()
                if inner:
                    return inner
    return None
