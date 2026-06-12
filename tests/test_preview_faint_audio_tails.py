"""Faint-microphone tail handling in the streaming preview.

Production 2026-06-11 (AirPods-in-pocket session): the mic level dropped
mid-session. The utterance "…testing Juno preview and let me see what are
the issues in the preview" decoded correctly in the FINAL lane, but the
live preview (a) committed a truncated dangling "w" from a trimmed buffer
and froze there, and (b) quarantined the trailing words as low-signal,
then erased them when an empty silence-window decode fed an empty
hypothesis into LocalAgreement — so the words never reached the display
even at utterance end.

Three behaviors are pinned here:

1. An empty decode on a silence tick (or at final) must PRESERVE a pending
   tail rather than erase it — silence is not evidence against words whose
   audio has been trimmed from the buffer.
2. At TRUE utterance end, a tail quarantined only for faint-audio reasons
   (low signal / post-speech grace / silence-window decode) is promoted
   when it passes the content guards. Content-suspicious tails (loops,
   blocklist) stay blocked.
3. No emit may end committed text on a dangling unsupported letter,
   whatever path produced it.
"""
from __future__ import annotations

import numpy as np

from juno_v2.contracts.preview import PreviewDecodeRequest
from juno_v2.preview.streaming_core import (
    StreamingPreviewSessionManager,
    WhisperDecodeOutput,
)


class _ScriptedDecoder:
    """Returns scripted (word, start, end) decodes, last script repeats."""

    def __init__(self, *scripts: list[tuple[str, float, float]]) -> None:
        self._scripts = list(scripts)
        self.calls = 0

    def warm(self) -> None:
        return None

    def decode(self, **kwargs: object) -> WhisperDecodeOutput:
        script = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1
        word_dicts = [{"word": w, "start": s, "end": e} for w, s, e in script]
        return WhisperDecodeOutput(
            text=" ".join(w for w, _, _ in script),
            language="en",
            decode_ms=1.0,
            word_dicts=word_dicts,
            segment_ends=[],
            metadata={"last_segment_no_speech_prob": 0.0},
        )


def _req(
    samples: int,
    *,
    is_final: bool = False,
    amplitude: float = 0.05,
) -> PreviewDecodeRequest:
    return PreviewDecodeRequest(
        utterance_id="utt",
        audio=np.ones(samples, dtype=np.float32) * amplitude,
        sample_rate_hz=16000,
        start_ms=0.0,
        end_ms=samples / 16.0,
        is_final=is_final,
        decode_seq=0,
        reset_decoder_state=False,
        context_payload={
            "preview_display_orthography": False,
            "preserve_state_on_decode_seq_zero": True,
        },
    )


def _manager(decoder: _ScriptedDecoder) -> StreamingPreviewSessionManager:
    return StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
        commit_draft_horizon_ms=0.0,
    )


_LEAD = [("hello", 0.0, 0.2), ("world", 0.2, 0.45)]
_WITH_TAIL = _LEAD + [
    ("in", 0.8, 0.95),
    ("the", 0.95, 1.1),
    ("preview", 1.1, 1.5),
]


def test_empty_silence_decode_preserves_pending_tail() -> None:
    decoder = _ScriptedDecoder(_LEAD, _LEAD, _WITH_TAIL, [])
    manager = _manager(decoder)

    manager.process(_req(8000))   # tail = hello world
    manager.process(_req(8000))   # agreement commits hello world
    manager.process(_req(16000))  # tail = in the preview
    state = manager._states["utt"]
    assert [w.text for w in state.hypothesis.tail] == ["in", "the", "preview"]

    # Empty chunk → silence tick → empty scripted decode over the buffer.
    manager.process(_req(0))

    assert [w.text for w in state.hypothesis.tail] == ["in", "the", "preview"], (
        "empty silence decode must not erase the pending tail"
    )
    assert state.silence_empty_decode_tail_preserved_events >= 1


def test_faint_tail_promoted_at_true_final() -> None:
    decoder = _ScriptedDecoder(_LEAD, _LEAD, _WITH_TAIL, [])
    manager = _manager(decoder)

    manager.process(_req(8000))
    manager.process(_req(8000))
    # Faint chunk: rms below the tail low-signal floor quarantines the tail.
    manager.process(_req(16000, amplitude=0.002))

    result = manager.process(_req(1600, is_final=True, amplitude=0.002))

    assert "in the preview" in result.committed_text, (
        f"faint-audio tail must promote at utterance end: {result.committed_text!r}"
    )
    promo = result.metadata.get("tail_final_promotion_status")
    assert promo in ("promoted", "agreement_committed"), promo


def test_content_suspicious_tail_still_blocked_at_final() -> None:
    loop_tail = _LEAD + [
        ("ansa", 0.8, 0.9),
        ("ansa", 0.9, 1.0),
        ("ansa", 1.0, 1.1),
        ("ansa", 1.1, 1.2),
        ("ansa", 1.2, 1.3),
    ]
    decoder = _ScriptedDecoder(_LEAD, _LEAD, loop_tail, [])
    manager = _manager(decoder)

    manager.process(_req(8000))
    manager.process(_req(8000))
    manager.process(_req(16000, amplitude=0.002))

    result = manager.process(_req(1600, is_final=True, amplitude=0.002))

    assert "ansa" not in result.committed_text, (
        f"loop tail must never promote: {result.committed_text!r}"
    )


def test_dangling_boundary_letter_never_emitted() -> None:
    truncated = [
        ("let", 0.0, 0.2),
        ("me", 0.2, 0.35),
        ("see", 0.35, 0.6),
        ("w", 0.62, 0.7),  # buffer trimmed mid-"what"
    ]
    decoder = _ScriptedDecoder(truncated, truncated, truncated)
    manager = _manager(decoder)

    for samples in (12000, 1600, 1600):
        result = manager.process(_req(samples))
        committed = result.committed_text
        assert not committed.rstrip().endswith(" w"), (
            f"dangling truncated letter emitted: {committed!r}"
        )
        assert committed.rstrip() != "w"
