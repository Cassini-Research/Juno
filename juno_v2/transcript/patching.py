from __future__ import annotations

import difflib
import hashlib

from juno_v2.transcript.contracts import PatchOpType, PatchReason, TranscriptPatchOp

_PATCH_REASONS: dict[str, PatchReason] = {
    "asr_correction": "asr_correction",
    "memory_alias": "memory_alias",
    "user_replacement": "user_replacement",
    "screen_term": "screen_term",
    "file_or_symbol": "file_or_symbol",
    "self_correction": "self_correction",
    "spoken_punctuation": "spoken_punctuation",
    "itn": "itn",
    "capitalization": "capitalization",
    "spacing": "spacing",
}


def visible_text_hash(text: str) -> str:
    normalized = " ".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def stable_prefix_chars(text: str, *, last_n_words_unstable: int = 5) -> int:
    """Return the char offset before the live-unstable tail starts."""

    s = text or ""
    if not s.strip():
        return 0
    words = list(_word_spans(s))
    if len(words) <= last_n_words_unstable:
        return 0
    cutoff_start, _cutoff_end = words[-last_n_words_unstable]
    # Prefer a sentence boundary before the tail when available.
    prior = s[:cutoff_start]
    boundary = max(prior.rfind("."), prior.rfind("!"), prior.rfind("?"))
    if boundary >= 0 and cutoff_start - boundary <= 80:
        return boundary + 1
    return cutoff_start


def diff_to_patch_ops(
    base: str,
    corrected: str,
    *,
    stable_prefix_chars: int,
    reason: str,
    confidence: float,
) -> list[TranscriptPatchOp]:
    """Build char-offset patch ops over the stable prefix.

    The first implementation intentionally uses ``SequenceMatcher``. It is
    deterministic, easy to validate, and keeps the contract ready for a later
    token-aware aligner without changing callers.
    """

    base_prefix = (base or "")[: max(0, stable_prefix_chars)]
    corrected_prefix = (corrected or "")[: max(0, min(len(corrected or ""), stable_prefix_chars))]
    ops: list[TranscriptPatchOp] = []
    matcher = difflib.SequenceMatcher(a=base_prefix, b=corrected_prefix, autojunk=False)
    op_reason = _coerce_reason(reason)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        replacement = corrected_prefix[j1:j2]
        if tag == "replace":
            op: PatchOpType = "replace"
        elif tag == "insert":
            op = "insert"
        elif tag == "delete":
            op = "delete"
        else:
            continue
        ops.append(
            TranscriptPatchOp(
                op=op,
                start_char=i1,
                end_char=i2,
                text=replacement,
                reason=op_reason,
                confidence=max(0.0, min(1.0, float(confidence))),
                source_text=base_prefix[i1:i2] or None,
            )
        )
    return ops


def live_patch_is_safe(
    base: str,
    ops: list[TranscriptPatchOp],
    *,
    stable_prefix_chars: int,
) -> tuple[bool, str]:
    if not ops:
        return True, "empty"
    stable = max(0, min(len(base or ""), int(stable_prefix_chars)))
    if len(ops) > 8:
        return False, "too_many_ops"

    changed = 0
    for op in ops:
        if op.start_char > stable or op.end_char > stable:
            return False, "touches_unstable_tail"
        if op.start_char < 0 or op.end_char < op.start_char:
            return False, "invalid_range"
        deleted = op.end_char - op.start_char
        inserted = len(op.text or "")
        if deleted > 80:
            return False, "delete_too_large"
        if inserted > 80:
            return False, "insert_too_large"
        changed += max(deleted, inserted)

    if stable > 0:
        max_changed = max(16 if stable < 40 else 0, int(stable * 0.30))
        if changed > max_changed:
            return False, "changes_too_much"
    return True, "ok"


def _word_spans(text: str):
    in_word = False
    start = 0
    for idx, ch in enumerate(text):
        if ch.isspace():
            if in_word:
                yield start, idx
                in_word = False
        elif not in_word:
            start = idx
            in_word = True
    if in_word:
        yield start, len(text)


def _coerce_reason(reason: str) -> PatchReason:
    return _PATCH_REASONS.get(reason, "asr_correction")
