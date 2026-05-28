"""Derive IDE context (focused file path + symbol under cursor).

``TypedContextBundle`` now carries ``focused_file_path`` and
``symbol_under_cursor`` (P2 item 11). This module centralises how we
*derive* those two fields from the raw signals we already have:

* ``focused_file_path`` comes from ``kAXDocumentAttribute`` when the
  Swift ``juno-capability`` helper could read it
  (``focused_document_path`` in the helper payload), with a
  window-title heuristic fallback for apps that don't expose
  ``AXDocument`` directly (VS Code / Cursor, JetBrains, Sublime,
  Atom — all of which encode the file path into the window title in
  predictable ways).

* ``symbol_under_cursor`` is reconstructed from the caret-adjacent
  text the capability helper already captures
  (``focused_text_before`` / ``focused_text_after``). We extract the
  identifier that straddles the caret — this catches both "cursor
  sits inside ``fooBar``" and "cursor is just after ``fooBar(`` in
  a fresh line".

Everything in this module is pure, synchronous, and unit-testable.
Surfaces that can provide stronger signals (e.g. a VS Code
extension that exposes the LSP symbol at the caret over a local
socket) should set these fields directly on the bundle; the helpers
here are the portable fallback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

# Identifier characters we treat as part of a "symbol". Deliberately
# permissive so code-switching users (python ``_``, C-style ``$``,
# TypeScript generic ``<T>``) are all covered without language
# detection. The caret-adjacent text has already been clipped and
# redacted by the capability helper so there's no privacy risk to
# reading a few more characters.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_IDENT_RE_START = re.compile(r"^[A-Za-z0-9_]*")


# Known window-title patterns that encode the file path. Ordered
# from most specific to least. Each entry is a compiled regex
# whose first group is the path component.
#
# We only trust patterns that are file-path shaped (contain a path
# separator OR a recognisable extension), so stray window titles
# like "Untitled" don't masquerade as files.
_WINDOW_TITLE_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # VS Code / Cursor / VSCodium: "filename.py - folder - Visual Studio Code"
    # The first fragment before " - " is the base filename (sometimes
    # with a leading "● " dirty marker).
    re.compile(r"^\s*[●•]?\s*(?P<name>[^\s/][^/]*\.[A-Za-z0-9]+)\s+[-—–]\s"),
    # JetBrains IDEs: "filename.py - ProjectName - [~/path/to/project] - IDE"
    # Same prefix shape; covered by the rule above.
    # Xcode: "filename.swift — ProjectName"
    re.compile(r"^\s*(?P<name>[^\s/][^/]*\.[A-Za-z0-9]+)\s+[—–]\s"),
    # Sublime / TextMate / Atom: "filename.py — ~/path/to/project"
    re.compile(r"^\s*(?P<name>[^\s/][^/]*\.[A-Za-z0-9]+)\s+[—–]\s"),
    # Apple TextEdit / Pages (document-based): just the filename.
    re.compile(r"^\s*(?P<name>[^\s/][^/]*\.[A-Za-z0-9]+)\s*$"),
)


def derive_focused_file_path(
    helper_payload: Mapping[str, object] | None,
    window_title: str | None,
) -> str | None:
    """Return the document path for the focused window.

    Preference order:
      1. ``focused_document_path`` reported by the capability helper
         (authoritative; comes from ``AXDocument``).
      2. Window-title heuristic; only accepts matches that look like
         a filename with an extension, to avoid matching "Untitled".
    Returns ``None`` when we don't have a trustworthy answer.
    """
    if helper_payload:
        raw = helper_payload.get("focused_document_path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    if window_title:
        for pattern in _WINDOW_TITLE_FILE_PATTERNS:
            m = pattern.search(window_title)
            if m:
                name = m.group("name").strip()
                # Reject candidates that obviously aren't files
                # (extension too long = likely a word suffix).
                ext = Path(name).suffix
                if 2 <= len(ext) <= 8:
                    return name
    return None


def derive_symbol_under_cursor(
    focused_text_before: str | None,
    focused_text_after: str | None,
) -> str | None:
    """Return the identifier straddling the caret, if any.

    We stitch the tail of ``focused_text_before`` and the head of
    ``focused_text_after`` at the caret and extract a single
    identifier-shaped token that spans the join. When the caret sits
    on whitespace, punctuation, or outside any identifier we return
    ``None`` — better to emit no hint than a wrong one.
    """
    before = focused_text_before or ""
    after = focused_text_after or ""
    tail = _IDENT_RE.search(before)
    head = _IDENT_RE_START.match(after)
    left = tail.group(0) if tail else ""
    right = head.group(0) if head else ""
    symbol = left + right
    if not symbol:
        return None
    # Reject pure-numeric tokens ("42", "0x1f") — bias engine can't
    # use those and they add noise.
    if symbol[0].isdigit():
        return None
    return symbol


__all__ = ["derive_focused_file_path", "derive_symbol_under_cursor"]
