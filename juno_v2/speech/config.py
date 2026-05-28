from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from juno_v2.contracts.audio import AudioFrame
from juno_v2.vad.probes import DualVadPolicy, EnergyVadProbe, SileroVadProbe, WebRtcVadProbe


@dataclass(frozen=True, slots=True)
class SpeechFrontEndProfile:
    name: str
    description: str
    input_gain_db: float
    start_trigger_frames: int
    pause_trigger_frames: int
    end_trigger_frames: int
    pre_roll_frames: int
    ring_buffer_ms: int
    webrtc_aggressiveness: int
    webrtc_fallback_threshold: float
    silero_threshold: float
    silero_fallback_threshold: float
    energy_threshold: float
    require_silero_for_start: bool


DEFAULT_SPEECH_PROFILE = "standard"


SPEECH_FRONTEND_PROFILES: dict[str, SpeechFrontEndProfile] = {
    "standard": SpeechFrontEndProfile(
        name="standard",
        description="Balanced default for normal dictation in a reasonably quiet environment.",
        input_gain_db=0.0,
        start_trigger_frames=3,
        pause_trigger_frames=8,
        end_trigger_frames=50,
        pre_roll_frames=8,
        ring_buffer_ms=3000,
        webrtc_aggressiveness=2,
        webrtc_fallback_threshold=0.012,
        silero_threshold=0.30,
        silero_fallback_threshold=0.014,
        energy_threshold=0.030,
        require_silero_for_start=True,
        # end_trigger_frames=50 → 1000 ms of continuous silence to end an
        # utterance. Up from the original 30 (600 ms) because real users
        # take 600-900 ms breaths between clauses; the shorter window
        # was splitting compound sentences mid-thought and dropping the
        # tail clauses to weak-start rejections. 1000 ms is the
        # commonly-cited "thinking pause" upper bound — long enough for
        # natural breathing, short enough that intentional end-of-turn
        # latency stays sub-second.
    ),
    "soft_speech": SpeechFrontEndProfile(
        name="soft_speech",
        description="Lower-threshold front end for slow, soft, or whisper-like speech in a controlled environment.",
        input_gain_db=6.0,
        start_trigger_frames=2,
        pause_trigger_frames=12,
        # 60 frames = 1200 ms. Bumped from 42 alongside standard's
        # bump from 30 → 50 (1000 ms) so soft_speech stays strictly
        # more patient about thinking pauses than standard, which is
        # the whole point of the profile.
        end_trigger_frames=60,
        pre_roll_frames=12,
        ring_buffer_ms=4000,
        webrtc_aggressiveness=1,
        webrtc_fallback_threshold=0.008,
        silero_threshold=0.38,
        silero_fallback_threshold=0.010,
        energy_threshold=0.008,
        require_silero_for_start=False,
    ),
}


@dataclass(slots=True)
class SpeechStateConfig:
    sample_rate_hz: int = 16000
    frame_ms: int = 20
    ring_buffer_ms: int = 3000
    start_trigger_frames: int = 3
    pause_trigger_frames: int = 8
    end_trigger_frames: int = 30
    pre_roll_frames: int = 8
    profile_name: str = DEFAULT_SPEECH_PROFILE
    input_gain_db: float = 0.0

    def ring_buffer_frames(self) -> int:
        return max(1, self.ring_buffer_ms // self.frame_ms)


def pause_trigger_frames_for_seconds(
    seconds: float | None,
    *,
    frame_ms: int = 20,
) -> int | None:
    """Convert a user-facing pause-sensitivity value (in seconds) into the
    integer ``pause_trigger_frames`` count the speech state machine consumes.

    Returns ``None`` when ``seconds`` is ``None`` so callers can use the
    sentinel to mean "leave the profile-driven default intact" — important
    for backwards compatibility with engines that never received the field
    over the wire.

    The macOS slider clamps to roughly 0.3–3.0s today (see
    ``JunoUserDefaults.pauseSensitivitySeconds``). We clamp again here so a
    bogus value from the broker contract can never produce a non-positive
    frame count that the state machine would treat as "fire on every frame".
    """
    if seconds is None:
        return None
    safe = max(0.05, min(5.0, float(seconds)))
    frames = int(round(safe * 1000.0 / float(frame_ms)))
    return max(1, frames)


def get_speech_front_end_profile(profile_name: str | None = None) -> SpeechFrontEndProfile:
    resolved = (profile_name or DEFAULT_SPEECH_PROFILE).strip() or DEFAULT_SPEECH_PROFILE
    try:
        return SPEECH_FRONTEND_PROFILES[resolved]
    except KeyError as exc:
        raise ValueError(f"Unknown speech profile: {profile_name}") from exc


def speech_state_config_for_profile(
    profile_name: str = DEFAULT_SPEECH_PROFILE,
    *,
    sample_rate_hz: int = 16000,
    frame_ms: int = 20,
) -> SpeechStateConfig:
    profile = get_speech_front_end_profile(profile_name)
    return SpeechStateConfig(
        sample_rate_hz=sample_rate_hz,
        frame_ms=frame_ms,
        ring_buffer_ms=profile.ring_buffer_ms,
        start_trigger_frames=profile.start_trigger_frames,
        pause_trigger_frames=profile.pause_trigger_frames,
        end_trigger_frames=profile.end_trigger_frames,
        pre_roll_frames=profile.pre_roll_frames,
        profile_name=profile.name,
        input_gain_db=profile.input_gain_db,
    )


def dual_vad_policy_for_profile(profile_name: str = DEFAULT_SPEECH_PROFILE) -> DualVadPolicy:
    profile = get_speech_front_end_profile(profile_name)
    return DualVadPolicy(
        webrtc=WebRtcVadProbe(
            aggressiveness=profile.webrtc_aggressiveness,
            fallback_threshold=profile.webrtc_fallback_threshold,
        ),
        silero=SileroVadProbe(
            threshold=profile.silero_threshold,
            fallback_threshold=profile.silero_fallback_threshold,
        ),
        energy=EnergyVadProbe(threshold=profile.energy_threshold),
        require_silero_for_start=profile.require_silero_for_start,
    )


def default_dual_vad_policy() -> DualVadPolicy:
    return dual_vad_policy_for_profile(DEFAULT_SPEECH_PROFILE)


def preprocess_audio_frame(frame: AudioFrame, state_config: SpeechStateConfig) -> AudioFrame:
    gain_db = getattr(state_config, "input_gain_db", 0.0)
    if abs(gain_db) < 1e-6:
        return frame
    linear = float(10 ** (gain_db / 20.0))
    boosted = np.clip(frame.samples.astype(np.float32) * linear, -1.0, 1.0)
    metadata = dict(frame.metadata)
    metadata["input_gain_db"] = gain_db
    return AudioFrame(
        index=frame.index,
        sample_rate_hz=frame.sample_rate_hz,
        start_sample=frame.start_sample,
        end_sample=frame.end_sample,
        start_ms=frame.start_ms,
        end_ms=frame.end_ms,
        samples=boosted,
        source=frame.source,
        metadata=metadata,
    )
