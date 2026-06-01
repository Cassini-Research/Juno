"""Pure-function helper for Issue #24: diff a pasted segment vs. the
post-edit field text.

The correction observer (``shells/macos/Sources/JunoTextMonitor/main.swift``)
emits the entire AX value of a focused field after the user edits it. The
broker previously treated the whole field as the "corrected" text, which
mixed in unrelated surrounding content (greeting, signature, prior
paragraphs the user never touched). For short pastes that landed in long
fields, this would either:

* slip past the 120-char safety filter and pollute the corrections store
  with whole-field rows, or
* fail the safety filter and silently drop the genuine edit.

This helper computes a *segment* diff: locate the pasted text inside the
observed field, then expand the located range to capture nearby edits,
and return ``(pasted_segment, corrected_segment)``. When the pasted text
isn't present at all (the user retyped), return ``None`` — fabricating a
correction from unrelated context is the bug we're fixing.
"""
from __future__ import annotations

import re

# Number of characters of context to include on each side of the pasted
# range. Keeps the corrected segment short enough to satisfy
# ``MAX_CORRECTION_TEXT_CHARS`` (120) while capturing typical fixes
# (one-word edits, punctuation tweaks at the boundary).
_CONTEXT_RADIUS = 16


def _fuzzy_locate(pasted: str, field: str) -> tuple[int, int] | None:
    """Find the best-matching window in ``field`` for ``pasted``.

    Slides a window of size ``len(pasted)`` (and ±2 chars for slack) and
    returns ``(start, end)`` of the highest-overlap window when the
    overlap ratio meets a 0.7 minimum. Returns ``None`` when no window
    is close enough — the user almost certainly typed something
    unrelated.

    Implementation note: this is O(field_len * pasted_len) but capped:
    we only accept fields up to ~10 KB (the AX value cap on the Swift
    side), and pasted is bounded by ``MAX_CORRECTION_TEXT_CHARS`` — so
    worst-case ~1.2M char ops, sub-millisecond on cpython.
    """
    if len(pasted) < 3 or len(field) < len(pasted):
        return None
    best_score = 0
    best_start = -1
    best_len = len(pasted)
    # Try window lengths from len(pasted)-2 to len(pasted)+2 to absorb
    # one or two char insertions/deletions at the boundary.
    for delta in (0, 1, -1, 2, -2):
        win_len = len(pasted) + delta
        if win_len <= 0 or win_len > len(field):
            continue
        for start in range(0, len(field) - win_len + 1):
            window = field[start:start + win_len]
            # Character-position overlap with pasted (up to min length).
            limit = min(len(pasted), win_len)
            score = sum(
                1 for i in range(limit) if pasted[i] == window[i]
            )
            if score > best_score:
                best_score = score
                best_start = start
                best_len = win_len
    if best_start < 0:
        return None
    threshold = max(2, int(len(pasted) * 0.7))
    if best_score < threshold:
        return None
    return best_start, best_start + best_len


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces for fuzzy locate.

    We do NOT modify the returned segments — this is only used to find
    the paste range when the AX layer adds/removes whitespace at the
    boundary. The returned ``corrected_segment`` is sliced from the
    original ``observed`` text, preserving the user's exact edit.
    """
    return re.sub(r"\s+", " ", text).strip()


def diff_pasted_segment(
    *, expected: str, observed: str
) -> tuple[str, str] | None:
    """Return ``(pasted_segment, corrected_segment)`` or ``None``.

    ``expected``: the text Juno pasted (what the user originally said).
    ``observed``: the full text of the focused field after the user
    edited it.

    Returns ``None`` when:
    * either input is empty,
    * the observed field equals the pasted text verbatim (nothing to
      learn — no edit happened), or
    * the pasted text cannot be located in the observed field, even
      after whitespace normalization (the user retyped from scratch and
      the original paste no longer appears — recording a correction
      here would mean fabricating a relationship between unrelated text).

    Otherwise, returns ``(pasted_segment, corrected_segment)`` where
    ``corrected_segment`` is a windowed slice of ``observed`` containing
    the user's edit + ``_CONTEXT_RADIUS`` chars of surrounding context.
    """
    pasted = (expected or "").strip()
    field = (observed or "").strip()
    if not pasted or not field:
        return None
    if pasted == field:
        return None

    # First pass: try to locate the paste verbatim in the field.
    idx = field.find(pasted)
    if idx >= 0:
        # The paste is still present unchanged. The user edited some
        # *other* part of the field (e.g. typed before/after the paste).
        # That isn't a correction of what Juno produced — drop it.
        return None

    # Second pass: whitespace-normalized locate. Find the pasted text in
    # the normalized form, then map back to a character range in the
    # original. We need the *approximate* range of where the paste
    # landed so we can window the diff.
    norm_pasted = _normalize_whitespace(pasted)
    norm_field = _normalize_whitespace(field)
    if norm_pasted and norm_pasted in norm_field:
        # The paste survived modulo whitespace edits; locate by the
        # first token (or first 8 chars) and use len(pasted) as the span.
        first_chunk = pasted.split()[0] if pasted.split() else pasted[:8]
        idx = field.find(first_chunk)
        if idx < 0:
            return None
        located_end = idx + len(pasted)
    else:
        # Try a fuzzy locate: slide a window of length len(pasted) ± 2
        # across the field and pick the position with the highest
        # character-by-character match ratio. Only accept if the best
        # window matches at least ~70% of pasted (i.e. an edit distance
        # below ~30%, the typical "user changed one word" case).
        located = _fuzzy_locate(pasted, field)
        if located is None:
            return None
        idx, located_end = located

    # Build a context window: pasted range ± _CONTEXT_RADIUS, clipped to
    # field bounds.
    window_start = max(0, idx - _CONTEXT_RADIUS)
    window_end = min(len(field), located_end + _CONTEXT_RADIUS)
    corrected_segment = field[window_start:window_end].strip()

    # If the windowed slice is identical to the paste, nothing changed.
    if corrected_segment == pasted:
        return None

    # Cap to MAX_CORRECTION_TEXT_CHARS; truncate from the start so the
    # tail (where edits often happen) survives.
    from juno_v2.memory.stores.corrections import MAX_CORRECTION_TEXT_CHARS

    if len(corrected_segment) > MAX_CORRECTION_TEXT_CHARS:
        corrected_segment = corrected_segment[-MAX_CORRECTION_TEXT_CHARS:]
    pasted_out = pasted
    if len(pasted_out) > MAX_CORRECTION_TEXT_CHARS:
        pasted_out = pasted_out[:MAX_CORRECTION_TEXT_CHARS]

    return pasted_out, corrected_segment
