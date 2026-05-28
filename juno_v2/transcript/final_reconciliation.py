"""Final-stage transcript reconciliation — preview→final stable-prefix patch.

Audit Issue #1 (P0). Pre-fix the macOS HUD does a hard cut from the live
preview text to the committed final text at speech-end. The user's
mental model is "Whisper corrects the live preview in place at the divergent
spans" — that's what this module enables.

The function below takes the preview snapshot the shell was showing when the
final lane completed and the final transcript Whisper produced, and returns a
``TranscriptAdjudicationResult`` tagged ``stage="final"``. Its ``to_dict()``
shape matches the ``transcript_patch_v1`` envelope the shell already parses
for live adjudication patches (`HUDTranscriptStore.applyPatchEnvelope`); only
the ``stage`` field differs.

Algorithm — stable-prefix suffix replacement (Approach A from the audit fix
plan, preferred over token-level diff for v1):

1. Walk the two strings character-by-character until they diverge. The number
   of matched characters is ``stable_prefix_chars``.
2. Emit a single ``replace`` op covering ``[stable_prefix_chars, len(preview)]``
   in the preview, with the corresponding suffix of the final text as the
   replacement. When the preview was empty we emit ``insert`` instead so the
   shell-side semantics ("draw new text, don't diff-replace empty range") stay
   crisp.
3. When the strings are identical we still produce an envelope with empty
   ops — the shell can short-circuit on it without losing the audit-trail
   record that final reconciliation ran.

The contract is intentionally narrow: this function does not consult Qwen or
the writer service, does not bias on context, and never rejects a patch.
Final-stage reconciliation is a deterministic mechanical diff of two strings
the engine already trusts.
"""
from __future__ import annotations

from juno_v2.transcript.contracts import (
    TranscriptAdjudicationResult,
    TranscriptPatchOp,
)
from juno_v2.transcript.patching import visible_text_hash


def _common_prefix_chars(a: str, b: str) -> int:
    """Length of the longest common prefix of ``a`` / ``b`` that ends on a
    word boundary.

    Word-boundary alignment is the difference between "hello wrold" /
    "hello world" yielding ``stable_prefix_chars=6`` (the whole "hello ") vs
    7 (the partial "hello w"). The HUD treats ``stable_prefix_chars`` as the
    boundary up to which spans are immutable — partial-word boundaries
    confuse the span renderer and produce visually torn replacements ("hello
    w" + "orld" instead of "hello " + "world"). Aligning to the trailing
    whitespace before divergence keeps span identities word-aligned, which
    matches what the user described as "fix-in-place at the divergent
    spans."

    The strings agree everywhere — return the full length.
    """

    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    if i == len(a) or i == len(b):
        # One string is a prefix of the other — the matched region ends at a
        # natural boundary (end-of-string for the shorter side). No need to
        # back up; the divergence is purely a length difference.
        return i
    # Otherwise the divergence is mid-word. Back up to the most recent
    # whitespace so the stable prefix ends on a word boundary, keeping span
    # identities word-aligned in the HUD ("hello " + "world", not "hello w"
    # + "orld"). If the divergence is in the very first word, the safe
    # stable prefix is empty.
    j = i
    while j > 0 and not a[j - 1].isspace():
        j -= 1
    return j


def build_final_patch_envelope(
    *,
    preview_text: str,
    final_text: str,
    utterance_id: str,
) -> TranscriptAdjudicationResult:
    """Build a final-stage ``transcript_patch_v1`` envelope for the HUD.

    Parameters
    ----------
    preview_text:
        The live preview text the shell was showing when the final lane
        completed. Used as ``base_visible_text`` so the HUD can reconcile
        against drift the same way it does for live patches.
    final_text:
        The Whisper-final transcript. Becomes ``corrected_text`` and the
        target of the suffix-replace op.
    utterance_id:
        The utterance the patch belongs to. The HUD ignores envelopes whose
        utterance has already been committed.
    """

    preview = preview_text or ""
    final = final_text or ""

    stable_chars = _common_prefix_chars(preview, final)

    ops: tuple[TranscriptPatchOp, ...]
    if preview == final:
        # Identity case — emit no ops. ``stable_prefix_chars`` reaches the end
        # of the field so any downstream HUD invariant ("ops touch only the
        # stable prefix") trivially holds.
        ops = ()
    elif not preview:
        # Empty preview — single insert at offset 0. ``end_char == start_char``
        # because we're inserting, not replacing a range.
        ops = (
            TranscriptPatchOp(
                op="insert",
                start_char=0,
                end_char=0,
                text=final,
                reason="asr_correction",
                confidence=1.0,
                source_text=None,
            ),
        )
    else:
        # General case — single suffix-replace covering the divergent tail.
        # When ``stable_chars == len(preview)`` (final is a strict prefix-
        # superset) ``end_char == start_char`` and the op effectively inserts
        # the new tail; we still call it ``replace`` so callers don't have to
        # branch on op kind.
        ops = (
            TranscriptPatchOp(
                op="replace",
                start_char=stable_chars,
                end_char=len(preview),
                text=final[stable_chars:],
                reason="asr_correction",
                confidence=1.0,
                source_text=preview[stable_chars:] or None,
            ),
        )

    return TranscriptAdjudicationResult(
        utterance_id=utterance_id,
        stage="final",
        corrected_text=final,
        ops=ops,
        confidence=1.0,
        base_visible_revision=None,
        base_text_hash=visible_text_hash(preview),
        base_visible_text=preview,
        # For final reconciliation the entire field is "stable" — Whisper has
        # decided. Reporting ``stable_prefix_chars`` as the common-prefix
        # length keeps the field consistent with live-stage semantics (where
        # it marks the boundary between immutable and live-unstable text).
        stable_prefix_chars=stable_chars,
        protected_terms_used=(),
        backend_name="final_reconciliation",
        decode_ms=0.0,
        metadata={"source": "stable_prefix_suffix_replace"},
    )
