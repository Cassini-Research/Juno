"""Tests for the speech state machine, speech config helpers, and the
utterance audio buffer."""

from __future__ import annotations

import numpy as np
import pytest

from juno_v2.asr.utterance_buffer import UtteranceAudioBuffer, stability_delta_chars
from juno_v2.contracts.speech import SpeechEventKind, SpeechPhase, VadDecision
from juno_v2.speech.config import (
    SpeechStateConfig,
    get_speech_front_end_profile,
    pause_trigger_frames_for_seconds,
    preprocess_audio_frame,
    speech_state_config_for_profile,
)
from juno_v2.speech.state_machine import SpeechStateMachine

from tests.audio_fixtures import sine_tone, slice_into_frames

FRAME_MS = 20


def _decision(index: int, kind: str) -> VadDecision:
    """Build a synthetic VadDecision.

    kind: "speech" (decision True), "silent" (strict silence), or
    "ambiguous" (neither — e.g. energy-only noise).
    """
    speech = kind == "speech"
    silent = kind == "silent"
    return VadDecision(
        frame_index=index,
        start_ms=index * FRAME_MS,
        end_ms=(index + 1) * FRAME_MS,
        webrtc_speech=speech,
        silero_speech=speech,
        energy_speech=speech or kind == "ambiguous",
        energy_rms=0.1 if speech else 0.005,
        decision=speech,
        is_silent=silent,
    )


def _frames(count: int) -> list:
    return slice_into_frames(sine_tone(duration_s=count * FRAME_MS / 1000.0, amplitude=0.1))


def _drive(machine: SpeechStateMachine, kinds: list[str]) -> list:
    frames = _frames(len(kinds))
    events = []
    for frame, kind in zip(frames, kinds):
        events.extend(machine.process(frame, _decision(frame.index, kind)))
    return events


def _config(start: int = 3, pause: int = 2, end: int = 5) -> SpeechStateConfig:
    return SpeechStateConfig(
        start_trigger_frames=start,
        pause_trigger_frames=pause,
        end_trigger_frames=end,
    )


def _kinds(events: list) -> list[SpeechEventKind]:
    return [e.kind for e in events if e.kind != SpeechEventKind.FRAME_PROCESSED]


# ------------------------------------------------------------------ #
# SpeechStateMachine
# ------------------------------------------------------------------ #

def test_normal_start_emits_started_then_confirmed() -> None:
    machine = SpeechStateMachine(config=_config(start=3))
    events = _drive(machine, ["speech"] * 4)
    kinds = _kinds(events)
    assert kinds == [SpeechEventKind.SPEECH_STARTED, SpeechEventKind.SPEECH_CONFIRMED]
    assert machine.phase == SpeechPhase.IN_SPEECH
    assert machine.current_utterance_id is not None
    assert machine.utterance_count == 1
    started = next(e for e in events if e.kind == SpeechEventKind.SPEECH_STARTED)
    confirmed = next(e for e in events if e.kind == SpeechEventKind.SPEECH_CONFIRMED)
    assert started.utterance_id == confirmed.utterance_id
    assert started.frame_index == 0
    assert confirmed.frame_index == 2  # third consecutive speech frame
    assert confirmed.reason == "start_trigger_reached"


def test_weak_start_is_aborted() -> None:
    machine = SpeechStateMachine(config=_config(start=3, pause=2))
    events = _drive(machine, ["speech", "silent", "silent"])
    kinds = _kinds(events)
    assert kinds == [SpeechEventKind.SPEECH_STARTED, SpeechEventKind.SESSION_ABORTED]
    aborted = next(e for e in events if e.kind == SpeechEventKind.SESSION_ABORTED)
    assert aborted.reason == "weak_start_rejected"
    assert machine.phase == SpeechPhase.SILENT
    assert machine.current_utterance_id is None
    # An aborted weak start still consumed an utterance id.
    assert machine.utterance_count == 1


def test_mid_speech_pause_and_resume() -> None:
    machine = SpeechStateMachine(config=_config(start=2, pause=3, end=10))
    events = _drive(
        machine,
        ["speech"] * 3 + ["silent"] * 3 + ["speech"],
    )
    kinds = _kinds(events)
    assert kinds == [
        SpeechEventKind.SPEECH_STARTED,
        SpeechEventKind.SPEECH_CONFIRMED,
        SpeechEventKind.SPEECH_PAUSED,
        SpeechEventKind.SPEECH_RESUMED,
    ]
    assert machine.phase == SpeechPhase.IN_SPEECH
    # Same utterance survives the pause.
    utterance_ids = {e.utterance_id for e in events if e.utterance_id}
    assert len(utterance_ids) == 1


def test_end_of_utterance_after_sustained_silence() -> None:
    machine = SpeechStateMachine(config=_config(start=2, pause=2, end=6))
    events = _drive(machine, ["speech"] * 3 + ["silent"] * 7)
    kinds = _kinds(events)
    assert kinds == [
        SpeechEventKind.SPEECH_STARTED,
        SpeechEventKind.SPEECH_CONFIRMED,
        SpeechEventKind.SPEECH_PAUSED,
        SpeechEventKind.SPEECH_ENDED,
    ]
    ended = next(e for e in events if e.kind == SpeechEventKind.SPEECH_ENDED)
    assert ended.reason == "end_trigger_reached"
    assert ended.utterance_id is not None
    # Machine resets for the next utterance.
    assert machine.phase == SpeechPhase.SILENT
    assert machine.current_utterance_id is None


def test_two_utterances_get_distinct_ids() -> None:
    machine = SpeechStateMachine(config=_config(start=2, pause=2, end=4))
    first = _drive(machine, ["speech"] * 3 + ["silent"] * 5)
    second = _drive(machine, ["speech"] * 3 + ["silent"] * 5)
    first_id = next(e.utterance_id for e in first if e.kind == SpeechEventKind.SPEECH_ENDED)
    second_id = next(e.utterance_id for e in second if e.kind == SpeechEventKind.SPEECH_ENDED)
    assert first_id != second_id
    assert machine.utterance_count == 2


def test_ambiguous_frames_do_not_advance_counters() -> None:
    machine = SpeechStateMachine(config=_config(start=2, pause=3, end=6))
    _drive(machine, ["speech"] * 3)
    assert machine.phase == SpeechPhase.IN_SPEECH
    # Ambiguous frames (decision False, is_silent False): no pause is ever
    # triggered, no counter moves.
    events = _drive(machine, ["ambiguous"] * 20)
    assert _kinds(events) == []
    assert machine.phase == SpeechPhase.IN_SPEECH
    assert machine.silence_run_frames == 0
    # Ambiguous frames count as neither speech nor silence.
    assert machine.speech_frame_count == 3
    assert machine.silence_frame_count == 0


def test_ambiguous_frames_interrupt_silence_run_without_resuming() -> None:
    machine = SpeechStateMachine(config=_config(start=2, pause=4, end=20))
    _drive(machine, ["speech"] * 3)
    _drive(machine, ["silent"] * 3)  # below pause trigger
    assert machine.phase == SpeechPhase.IN_SPEECH
    assert machine.silence_run_frames == 3
    _drive(machine, ["ambiguous"] * 2)  # holds state, counters frozen
    assert machine.silence_run_frames == 3
    assert machine.phase == SpeechPhase.IN_SPEECH
    events = _drive(machine, ["silent"] * 1)  # 4th strict-silence frame
    assert _kinds(events) == [SpeechEventKind.SPEECH_PAUSED]


def test_summary_counts() -> None:
    machine = SpeechStateMachine(config=_config(start=2, pause=2, end=4))
    _drive(machine, ["speech"] * 3 + ["ambiguous"] * 2 + ["silent"] * 5)
    summary = machine.summary("session-1", total_audio_ms=200.0)
    assert summary.total_frames == 10
    assert summary.speech_frame_count == 3
    assert summary.silence_frame_count == 5
    assert summary.utterance_count == 1
    assert summary.completed is True


# ------------------------------------------------------------------ #
# speech config helpers
# ------------------------------------------------------------------ #

def test_speech_state_config_for_standard_profile() -> None:
    config = speech_state_config_for_profile("standard")
    assert config.profile_name == "standard"
    assert config.start_trigger_frames == 3
    assert config.pause_trigger_frames == 8
    assert config.end_trigger_frames == 50
    assert config.pre_roll_frames == 8
    assert config.ring_buffer_ms == 3000
    assert config.input_gain_db == 0.0
    assert config.ring_buffer_frames() == 150  # 3000 ms / 20 ms


def test_speech_state_config_for_soft_speech_profile() -> None:
    config = speech_state_config_for_profile("soft_speech")
    assert config.profile_name == "soft_speech"
    assert config.start_trigger_frames == 2
    assert config.pause_trigger_frames == 12
    assert config.end_trigger_frames == 60
    assert config.input_gain_db == 6.0
    assert config.ring_buffer_ms == 4000
    # soft_speech must stay strictly more patient than standard.
    standard = speech_state_config_for_profile("standard")
    assert config.end_trigger_frames > standard.end_trigger_frames


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="Unknown speech profile"):
        speech_state_config_for_profile("does_not_exist")
    with pytest.raises(ValueError):
        get_speech_front_end_profile("nope")


def test_profile_name_default_and_blank_fall_back_to_standard() -> None:
    assert get_speech_front_end_profile(None).name == "standard"
    assert get_speech_front_end_profile("  ").name == "standard"


def test_preprocess_audio_frame_zero_gain_is_identity() -> None:
    config = speech_state_config_for_profile("standard")
    frame = slice_into_frames(sine_tone(duration_s=0.02, amplitude=0.1))[0]
    assert preprocess_audio_frame(frame, config) is frame


def test_preprocess_audio_frame_applies_gain_and_clips() -> None:
    config = speech_state_config_for_profile("soft_speech")  # +6 dB
    samples = np.full(320, 0.1, dtype=np.float32)
    frame = slice_into_frames(samples)[0]
    boosted = preprocess_audio_frame(frame, config)
    assert boosted is not frame
    expected = 0.1 * (10 ** (6.0 / 20.0))  # ~0.1995
    assert boosted.samples[0] == pytest.approx(expected, rel=1e-4)
    assert boosted.metadata["input_gain_db"] == 6.0
    assert boosted.index == frame.index
    assert boosted.start_ms == frame.start_ms

    # Loud input clips at 1.0 instead of overflowing.
    loud = slice_into_frames(np.full(320, 0.9, dtype=np.float32))[0]
    clipped = preprocess_audio_frame(loud, config)
    assert float(np.max(clipped.samples)) == pytest.approx(1.0)


def test_pause_trigger_frames_for_seconds() -> None:
    assert pause_trigger_frames_for_seconds(None) is None
    assert pause_trigger_frames_for_seconds(1.0) == 50
    assert pause_trigger_frames_for_seconds(0.3) == 15
    assert pause_trigger_frames_for_seconds(1.0, frame_ms=10) == 100


def test_pause_trigger_frames_clamping() -> None:
    # Below the 0.05 s floor.
    low = pause_trigger_frames_for_seconds(0.0)
    assert low == pause_trigger_frames_for_seconds(-100.0)
    assert low is not None and low >= 1
    assert low == max(1, int(round(0.05 * 1000.0 / 20.0)))
    # Above the 5.0 s ceiling.
    assert pause_trigger_frames_for_seconds(999.0) == 250  # 5.0 s / 20 ms


# ------------------------------------------------------------------ #
# UtteranceAudioBuffer
# ------------------------------------------------------------------ #

def test_utterance_buffer_enforces_frame_ordering() -> None:
    buf = UtteranceAudioBuffer(utterance_id="utt-1")
    frames = _frames(4)
    buf.append(frames[0])
    buf.append(frames[2])
    buf.append(frames[1])  # out of order -> ignored
    buf.append(frames[2])  # duplicate -> ignored
    buf.append(frames[3])
    assert [f.index for f in buf.frames] == [0, 2, 3]


def test_utterance_buffer_seed_and_audio_concat() -> None:
    buf = UtteranceAudioBuffer(utterance_id="utt-2")
    frames = _frames(5)
    buf.seed(frames)
    audio = buf.audio()
    assert audio.dtype == np.float32
    assert audio.shape[0] == 5 * 320
    expected = np.concatenate([f.samples for f in frames])
    assert np.array_equal(audio, expected)
    assert buf.start_ms == pytest.approx(0.0)
    assert buf.end_ms == pytest.approx(100.0)
    assert buf.duration_ms == pytest.approx(100.0)


def test_utterance_buffer_empty_audio() -> None:
    buf = UtteranceAudioBuffer(utterance_id="utt-empty")
    assert buf.audio().shape == (0,)
    assert buf.duration_ms == 0.0
    assert buf.start_ms == 0.0 and buf.end_ms == 0.0


def test_should_decode_partial_timing() -> None:
    buf = UtteranceAudioBuffer(utterance_id="utt-3")
    frames = _frames(40)  # 800 ms total available
    # 200 ms buffered: below the min-audio floor.
    buf.seed(frames[:10])
    assert buf.should_decode_partial(interval_ms=100, min_audio_ms=300) is False
    # 400 ms buffered: above floor and 400 ms since last decode (never).
    buf.seed(frames[10:20])
    assert buf.should_decode_partial(interval_ms=100, min_audio_ms=300) is True
    buf.mark_partial_decode()
    assert buf.partial_decode_seq == 1
    assert buf.should_decode_partial(interval_ms=100, min_audio_ms=300) is False
    # Two more frames (40 ms): still inside the 100 ms interval.
    buf.seed(frames[20:22])
    assert buf.should_decode_partial(interval_ms=100, min_audio_ms=300) is False
    # Five total new frames (100 ms): interval reached again.
    buf.seed(frames[22:25])
    assert buf.should_decode_partial(interval_ms=100, min_audio_ms=300) is True


def test_update_last_emitted_text_stability_delta() -> None:
    buf = UtteranceAudioBuffer(utterance_id="utt-4")
    assert buf.last_emitted_text == ""
    assert buf.update_last_emitted_text("hello") == 5
    assert buf.update_last_emitted_text("hello") == 0
    assert buf.update_last_emitted_text("hello world") == 6
    # Divergence after the common prefix counts both tails.
    assert buf.update_last_emitted_text("hello there") == 10
    assert buf.last_emitted_text == "hello there"


def test_stability_delta_chars_directly() -> None:
    assert stability_delta_chars("", "") == 0
    assert stability_delta_chars("", "abc") == 3
    assert stability_delta_chars("abc", "") == 3
    assert stability_delta_chars("abc", "abd") == 2
    assert stability_delta_chars("same", "same") == 0
