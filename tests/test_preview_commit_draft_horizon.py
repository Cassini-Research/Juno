"""Draft-horizon guard for live-lane commits.

Production 2026-06-10 (U5): the user said "…the moment it listens to Hey
Juno, it stops working" — the rolling window ended mid-word, Whisper
completed the truncated audio as "hey Juno I don't know.", the hallucination
stayed stable for consecutive decodes of the nearly-identical buffer, and
LocalAgreement committed it into the HUD where it could never be repaired.

The guard demotes agreed words whose timestamps end inside the last N ms of
buffered audio (the truncated-decode zone) back to the tail; one more
growing-audio decode either confirms them or corrects them.
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


def _req(samples: int, *, is_final: bool = False) -> PreviewDecodeRequest:
    return PreviewDecodeRequest(
        utterance_id="utt",
        audio=np.ones(samples, dtype=np.float32) * 0.01,
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


_HALLUCINATED = [
    ("listens", 0.0, 0.2),
    ("to", 0.2, 0.4),
    ("hey", 0.4, 0.6),
    ("juno", 0.6, 0.8),
    # Whisper's stable completion of cut-off speech — must never commit.
    ("i", 1.1, 1.25),
    ("don't", 1.25, 1.45),
    ("know.", 1.45, 1.55),
]
_CORRECTED = [
    ("listens", 0.0, 0.2),
    ("to", 0.2, 0.4),
    ("hey", 0.4, 0.6),
    ("juno", 0.6, 0.8),
    ("it", 1.1, 1.3),
    ("stops", 1.3, 1.55),
    ("working", 1.55, 1.85),
]


def test_truncated_zone_hallucination_never_commits() -> None:
    decoder = _ScriptedDecoder(_HALLUCINATED, _HALLUCINATED, _HALLUCINATED, _CORRECTED, _CORRECTED)
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
        commit_draft_horizon_ms=600.0,
    )

    # Buffer grows 1.5s → 1.6s → 1.7s while the hallucination stays stable,
    # then the corrected decode arrives with more audio.
    steps = [24000, 1600, 1600, 8000, 4800]
    for samples in steps:
        result = manager.process(_req(samples))
        committed = result.committed_text
        assert "don't" not in committed, f"hallucination committed: {committed!r}"
        assert "know" not in committed, f"hallucination committed: {committed!r}"

    state = manager._states["utt"]
    assert state.commit_draft_horizon_demotions >= 1
    final_committed = state.hypothesis.committed_text()
    assert final_committed.startswith("listens to hey juno")
    assert "it stops working" in final_committed
    assert "don't" not in final_committed


def test_true_stop_still_promotes_tail_through_horizon() -> None:
    script = [
        ("ship", 0.0, 0.2),
        ("the", 0.2, 0.4),
        ("build", 0.4, 0.62),
    ]
    decoder = _ScriptedDecoder(script, script)
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
        commit_draft_horizon_ms=600.0,
    )

    manager.process(_req(11200))  # 0.7s buffer — "build" ends inside horizon
    final = manager.process(_req(1600, is_final=True))
    assert "build" in final.committed_text


def test_horizon_disabled_by_default_for_direct_construction(monkeypatch) -> None:
    monkeypatch.delenv("JUNO_V2_PREVIEW_COMMIT_DRAFT_HORIZON_MS", raising=False)
    decoder = _ScriptedDecoder(_HALLUCINATED, _HALLUCINATED)
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
    )
    assert manager.commit_draft_horizon_ms == 0.0


def test_horizon_env_enables_for_production_launcher(monkeypatch) -> None:
    monkeypatch.setenv("JUNO_V2_PREVIEW_COMMIT_DRAFT_HORIZON_MS", "600")
    decoder = _ScriptedDecoder(_HALLUCINATED, _HALLUCINATED)
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
    )
    assert manager.commit_draft_horizon_ms == 600.0
