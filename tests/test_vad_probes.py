"""Tests for juno_v2.vad.probes.

WebRTC and Silero VADs are statistical components: assertions are majority
votes over many frames, never per-frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from juno_v2.contracts.audio import AudioFrame
from juno_v2.contracts.speech import VadDecision
from juno_v2.vad.probes import (
    DualVadPolicy,
    EnergyVadProbe,
    SileroVadProbe,
    WebRtcVadProbe,
)

from tests.audio_fixtures import (
    require_say,
    silence,
    sine_tone,
    slice_into_frames,
    tts_speech_array,
)

TTS_SENTENCE = "hello world this is a dictation test"


def _speech_frames() -> list[AudioFrame]:
    require_say()
    return slice_into_frames(tts_speech_array(TTS_SENTENCE), source="tts")


def _silence_frames(duration_s: float = 1.0) -> list[AudioFrame]:
    return slice_into_frames(silence(duration_s), source="silence")


def _voiced(frames: list[AudioFrame], rms_floor: float = 0.02) -> list[AudioFrame]:
    """Restrict to frames with clear acoustic energy (skips TTS lead/tail gaps)."""
    voiced = [
        f
        for f in frames
        if float(np.sqrt(np.mean(np.square(f.samples)))) >= rms_floor
    ]
    assert len(voiced) >= 10, "TTS fixture produced too few energetic frames"
    return voiced


# ------------------------------------------------------------------ #
# EnergyVadProbe
# ------------------------------------------------------------------ #

def test_energy_probe_silence_is_not_speech() -> None:
    probe = EnergyVadProbe()
    for frame in _silence_frames(0.2):
        decision, rms = probe.detect(frame)
        assert decision is False
        assert rms == pytest.approx(0.0)


def test_energy_probe_tone_is_speech() -> None:
    probe = EnergyVadProbe(threshold=0.015)
    frames = slice_into_frames(sine_tone(duration_s=0.2, amplitude=0.2))
    for frame in frames:
        decision, rms = probe.detect(frame)
        assert decision is True
        assert rms == pytest.approx(0.2 / np.sqrt(2.0), rel=0.05)


def test_energy_probe_threshold_boundary() -> None:
    threshold = 0.05
    probe = EnergyVadProbe(threshold=threshold)
    clock_frames = slice_into_frames(np.full(320, threshold, dtype=np.float32))
    at_threshold = clock_frames[0]  # constant signal -> rms == value
    decision, rms = probe.detect(at_threshold)
    assert rms == pytest.approx(threshold)
    assert decision is True  # rms >= threshold is inclusive

    below = slice_into_frames(np.full(320, threshold * 0.98, dtype=np.float32))[0]
    decision, rms = probe.detect(below)
    assert decision is False
    assert rms < threshold


# ------------------------------------------------------------------ #
# WebRtcVadProbe (statistical: majority assertions)
# ------------------------------------------------------------------ #

def test_webrtc_probe_uses_real_vad() -> None:
    probe = WebRtcVadProbe()
    assert probe._vad is not None, "webrtcvad should be installed in this env"


def test_webrtc_probe_speech_frames_mostly_true() -> None:
    probe = WebRtcVadProbe(aggressiveness=2)
    voiced = _voiced(_speech_frames())
    hits = sum(1 for f in voiced if probe.detect(f)[0])
    assert hits / len(voiced) >= 0.6, f"only {hits}/{len(voiced)} voiced frames flagged"


def test_webrtc_probe_silence_frames_mostly_false() -> None:
    probe = WebRtcVadProbe(aggressiveness=2)
    frames = _silence_frames(1.0)
    hits = sum(1 for f in frames if probe.detect(f)[0])
    assert hits / len(frames) <= 0.2, f"{hits}/{len(frames)} silent frames flagged"


# ------------------------------------------------------------------ #
# SileroVadProbe (statistical: majority assertions)
# ------------------------------------------------------------------ #

def test_silero_probe_loaded() -> None:
    probe = SileroVadProbe()
    assert probe._model is not None, "silero_vad should be installed in this env"


def test_silero_probe_speech_scores_high_on_voiced_frames() -> None:
    # Threshold 0.30 matches the production "standard" speech profile: the
    # probe zero-pads each 20 ms frame (320 samples) to silero's 512-sample
    # window, which dilutes per-frame scores, so the deployed threshold is
    # deliberately below silero's nominal 0.5.
    probe = SileroVadProbe(threshold=0.30)
    frames = _speech_frames()
    voiced_indices = {
        f.index
        for f in _voiced(frames)
    }
    # Feed all frames in order (the model is stateful) but assert only on
    # the energetic ones.
    decisions: list[tuple[bool, float | None]] = []
    for frame in frames:
        decided, score = probe.detect(frame)
        if frame.index in voiced_indices:
            assert score is not None
            decisions.append((decided, score))
    speech_votes = sum(1 for decided, _ in decisions if decided)
    assert speech_votes / len(decisions) >= 0.5, (
        f"only {speech_votes}/{len(decisions)} voiced frames scored as speech"
    )
    mean_voiced = float(np.mean([s for _, s in decisions]))
    assert mean_voiced >= 0.25, f"mean voiced silero score too low: {mean_voiced}"


def test_silero_probe_silence_scores_low() -> None:
    probe = SileroVadProbe(threshold=0.5)
    frames = _silence_frames(1.0)
    scores = []
    for frame in frames:
        decided, score = probe.detect(frame)
        assert score is not None
        scores.append((decided, score))
    speech_votes = sum(1 for decided, _ in scores if decided)
    assert speech_votes / len(scores) <= 0.2
    assert float(np.mean([s for _, s in scores])) < 0.3


# ------------------------------------------------------------------ #
# DualVadPolicy.decide combinations (deterministic fake probes)
# ------------------------------------------------------------------ #

@dataclass
class _FixedProbe:
    value: bool
    score: float | None = None
    name: str = "fixed"

    def detect(self, frame: AudioFrame) -> tuple[bool, float | None]:
        return self.value, self.score


def _policy(
    webrtc: bool, silero: bool, energy: bool, *, require_silero_for_start: bool = True
) -> DualVadPolicy:
    return DualVadPolicy(
        webrtc=_FixedProbe(webrtc),
        silero=_FixedProbe(silero, score=0.9 if silero else 0.05),
        energy=_FixedProbe(energy, score=0.1 if energy else 0.0001),
        require_silero_for_start=require_silero_for_start,
    )


def _frame() -> AudioFrame:
    return slice_into_frames(sine_tone(duration_s=0.02, amplitude=0.1))[0]


def test_dual_policy_all_speech() -> None:
    decision = _policy(True, True, True).decide(_frame())
    assert isinstance(decision, VadDecision)
    assert decision.decision is True
    assert decision.is_silent is False
    assert decision.webrtc_speech and decision.silero_speech and decision.energy_speech
    assert decision.silero_score == pytest.approx(0.9)
    assert decision.metadata["policy"] == "dual_vad"


def test_dual_policy_all_silent() -> None:
    decision = _policy(False, False, False).decide(_frame())
    assert decision.decision is False
    assert decision.is_silent is True


def test_dual_policy_webrtc_alone_is_not_speech_when_silero_required() -> None:
    decision = _policy(True, False, True).decide(_frame())
    assert decision.decision is False
    # webrtc positive blocks the silence counter.
    assert decision.is_silent is False


def test_dual_policy_silero_plus_energy_is_speech() -> None:
    decision = _policy(False, True, True).decide(_frame())
    assert decision.decision is True
    assert decision.is_silent is False


def test_dual_policy_silero_alone_is_ambiguous() -> None:
    # Silero positive without webrtc or energy corroboration: not speech,
    # but not silent either — the ambiguous middle ground.
    decision = _policy(False, True, False).decide(_frame())
    assert decision.decision is False
    assert decision.is_silent is False


def test_dual_policy_energy_alone_does_not_block_silence() -> None:
    # Loud ambient noise: both speech-trained detectors negative -> silent.
    decision = _policy(False, False, True).decide(_frame())
    assert decision.decision is False
    assert decision.is_silent is True


def test_dual_policy_energy_alone_starts_speech_without_silero_requirement() -> None:
    decision = _policy(
        False, False, True, require_silero_for_start=False
    ).decide(_frame())
    assert decision.decision is True
    assert decision.is_silent is True  # silence gate is independent of decision


def test_dual_policy_frame_metadata_propagation() -> None:
    frame = _frame()
    decision = _policy(True, True, True).decide(frame)
    assert decision.frame_index == frame.index
    assert decision.start_ms == frame.start_ms
    assert decision.end_ms == frame.end_ms
    assert decision.energy_rms == pytest.approx(0.1)


# ------------------------------------------------------------------ #
# DualVadPolicy with real probes on real signals (sanity, majority votes)
# ------------------------------------------------------------------ #

def test_dual_policy_real_probes_speech_vs_silence() -> None:
    policy = DualVadPolicy(
        webrtc=WebRtcVadProbe(aggressiveness=2),
        silero=SileroVadProbe(threshold=0.3),
        energy=EnergyVadProbe(threshold=0.03),
        require_silero_for_start=True,
    )
    speech_frames = _speech_frames()
    voiced_indices = {f.index for f in _voiced(speech_frames)}
    speech_hits = 0
    voiced_count = 0
    for frame in speech_frames:
        decision = policy.decide(frame)
        if frame.index in voiced_indices:
            voiced_count += 1
            speech_hits += int(decision.decision)
    assert speech_hits / voiced_count >= 0.5

    silent_votes = 0
    silence_frames = _silence_frames(1.0)
    for frame in silence_frames:
        decision = policy.decide(frame)
        silent_votes += int(decision.is_silent)
    assert silent_votes / len(silence_frames) >= 0.8
