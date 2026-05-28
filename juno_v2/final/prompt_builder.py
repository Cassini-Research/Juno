"""Compose Whisper ``initial_prompt`` from personalization signals.

Inputs:
    * ``lexicon_terms`` — top-K user vocabulary (already ranked by the
      personalization layer).
    * ``recent_finals`` — list of recently finalized transcripts (newest
      last). PII guard strips digit runs >=6 chars before the value
      enters the prompt.
    * ``asr_addon`` — surface-preset add-on string (per issue #27).

Output is truncated conservatively to <=140 words (~220 tokens), well
below Whisper's ~224-token prompt budget. When all inputs are empty the
function returns ``None`` so callers can fall back to whatever
``initial_prompt`` they already have.

Issue #28 — vocab biasing was wired all the way to ``mlx_whisper.transcribe``
but ``initial_prompt`` was never populated from user signals. This module
closes that gap.
"""
from __future__ import annotations

import re
from typing import Sequence

# Mask runs of 6+ digits before they enter the initial prompt.
# Whisper's prompt is fed back into decoding, so a credit-card number
# leaking into one utterance's prompt could surface in the next
# utterance's transcript.
_DIGIT_RUN_RE = re.compile(r"\d{6,}")

_MAX_WORDS = 140


def _mask_digit_runs(text: str) -> str:
    return _DIGIT_RUN_RE.sub("[redacted]", text or "")


def _truncate_to_word_budget(text: str, *, max_words: int = _MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def build_whisper_initial_prompt(
    *,
    lexicon_terms: Sequence[str],
    recent_finals: Sequence[str],
    asr_addon: str | None,
    max_words: int = _MAX_WORDS,
) -> str | None:
    """Compose a Whisper ``initial_prompt``.

    Layout::

        Vocabulary: term1, term2, ...

        <asr_addon>

        <recent_final_1>; <recent_final_2>; <recent_final_3>

    Truncation favours the *front* of the prompt (vocab + addon) and
    drops earliest recent_finals first, so the most-recent context
    survives the word cap.
    """
    vocab_clean: list[str] = []
    seen: set[str] = set()
    for term in lexicon_terms or []:
        t = (term or "").strip()
        if not t:
            continue
        k = t.casefold()
        if k in seen:
            continue
        seen.add(k)
        vocab_clean.append(t)

    addon_clean = (asr_addon or "").strip() or None

    # PII guard: mask digit runs in recent finals before they enter the
    # prompt. Empty finals after masking are dropped.
    finals_clean: list[str] = []
    for line in recent_finals or []:
        if not line:
            continue
        masked = _mask_digit_runs(str(line)).strip()
        if masked:
            finals_clean.append(masked)

    if not vocab_clean and not addon_clean and not finals_clean:
        return None

    head_parts: list[str] = []
    if vocab_clean:
        head_parts.append("Vocabulary: " + ", ".join(vocab_clean) + ".")
    if addon_clean:
        head_parts.append(addon_clean)
    head = "\n\n".join(head_parts)
    head_words = len(head.split())

    # Build recent-finals tail, taking from the right so most-recent
    # surfaces first.
    tail_budget = max(0, max_words - head_words)
    tail = ""
    if finals_clean and tail_budget > 0:
        # Walk newest -> oldest, accumulate until budget is exhausted.
        kept_reversed: list[str] = []
        used = 0
        for line in reversed(finals_clean):
            wc = len(line.split())
            # +1 for the joining "; "
            if used + wc + (1 if kept_reversed else 0) > tail_budget:
                # Try truncating the line itself if nothing has been kept.
                if not kept_reversed:
                    truncated = " ".join(line.split()[:tail_budget])
                    if truncated:
                        kept_reversed.append(truncated)
                break
            kept_reversed.append(line)
            used += wc + (1 if len(kept_reversed) > 1 else 0)
        # Restore chronological order (oldest first), so trailing context
        # in the prompt reads naturally.
        kept = list(reversed(kept_reversed))
        if kept:
            tail = "; ".join(kept)

    if tail:
        prompt = head + ("\n\n" if head else "") + tail
    else:
        prompt = head

    # Final hard cap (defensive — `head` itself could exceed the budget
    # if vocab is enormous).
    prompt = _truncate_to_word_budget(prompt, max_words=max_words)
    return prompt or None
