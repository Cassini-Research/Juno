from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from juno_v2.contracts.audio import AudioFrame


class VadProbe(Protocol):
    name: str

    def detect(self, frame: AudioFrame) -> tuple[bool, float | None]:
        ...


class _WebRtcVadLike(Protocol):
    def is_speech(self, pcm: bytes, sample_rate_hz: int) -> bool:
        ...


class _TensorLike(Protocol):
    def float(self) -> "_TensorLike":
        ...


class _TorchLike(Protocol):
    def from_numpy(self, array: npt.NDArray[np.float32]) -> _TensorLike:
        ...


class _ScoreLike(Protocol):
    def item(self) -> float:
        ...


class _SileroModelLike(Protocol):
    def __call__(self, tensor: _TensorLike, sample_rate_hz: int) -> _ScoreLike:
        ...


@dataclass(slots=True)
class EnergyVadProbe:
    threshold: float = 0.015
    name: str = "energy"

    def detect(self, frame: AudioFrame) -> tuple[bool, float | None]:
        rms = float(np.sqrt(np.mean(np.square(frame.samples), dtype=np.float64)))
        return rms >= self.threshold, rms


@dataclass(slots=True)
class WebRtcVadProbe:
    aggressiveness: int = 2
    fallback_threshold: float = 0.012
    name: str = "webrtc"
    _fallback: EnergyVadProbe = field(init=False, repr=False)
    _vad: _WebRtcVadLike | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._fallback = EnergyVadProbe(threshold=self.fallback_threshold)
        try:
            import webrtcvad  # type: ignore

            self._vad = webrtcvad.Vad(self.aggressiveness)
        except Exception:
            self._vad = None

    def detect(self, frame: AudioFrame) -> tuple[bool, float | None]:
        if self._vad is None:
            return self._fallback.detect(frame)
        pcm = np.clip(frame.samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
        is_speech = bool(self._vad.is_speech(pcm16, frame.sample_rate_hz))
        return is_speech, None


@dataclass(slots=True)
class SileroVadProbe:
    threshold: float = 0.5
    fallback_threshold: float = 0.014
    sample_rate_hz: int = 16000
    name: str = "silero"
    _fallback: EnergyVadProbe = field(init=False, repr=False)
    _torch: _TorchLike | None = field(init=False, repr=False, default=None)
    _model: _SileroModelLike | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._fallback = EnergyVadProbe(threshold=self.fallback_threshold)
        try:
            import torch  # type: ignore
            from silero_vad import load_silero_vad  # type: ignore

            self._torch = torch
            self._model = load_silero_vad()
        except Exception:
            self._torch = None
            self._model = None

    def _prepare_input(self, frame: AudioFrame) -> tuple[_TensorLike, int] | None:
        assert self._torch is not None
        samples = np.asarray(frame.samples, dtype=np.float32).reshape(-1)
        sample_rate_hz = frame.sample_rate_hz
        if sample_rate_hz != 16000 and sample_rate_hz % 16000 == 0:
            step = sample_rate_hz // 16000
            samples = samples[::step]
            sample_rate_hz = 16000
        if sample_rate_hz == 16000:
            target_samples = 512
        elif sample_rate_hz == 8000:
            target_samples = 256
        else:
            return None
        if samples.shape[0] < target_samples:
            samples = np.pad(samples, (0, target_samples - samples.shape[0]))
        elif samples.shape[0] > target_samples:
            samples = samples[-target_samples:]
        return self._torch.from_numpy(samples).float(), sample_rate_hz

    def detect(self, frame: AudioFrame) -> tuple[bool, float | None]:
        if self._model is None or self._torch is None:
            return self._fallback.detect(frame)
        prepared = self._prepare_input(frame)
        if prepared is None:
            return self._fallback.detect(frame)
        tensor, sample_rate_hz = prepared
        try:
            score = float(self._model(tensor, sample_rate_hz).item())
        except Exception as exc:
            message = str(exc).lower()
            if "too short" in message or "provided number of samples" in message:
                return self._fallback.detect(frame)
            raise
        return score >= self.threshold, score


@dataclass(slots=True)
class DualVadPolicy:
    """Combine fast gate, confidence gate, and energy fallback into one decision."""

    webrtc: WebRtcVadProbe
    silero: SileroVadProbe
    energy: EnergyVadProbe
    require_silero_for_start: bool = True

    def decide(self, frame: AudioFrame):
        from juno_v2.contracts.speech import VadDecision

        webrtc_speech, _ = self.webrtc.detect(frame)
        silero_speech, silero_score = self.silero.detect(frame)
        energy_speech, rms = self.energy.detect(frame)

        if self.require_silero_for_start:
            decision = bool((webrtc_speech and silero_speech) or (silero_speech and energy_speech))
        else:
            decision = bool((webrtc_speech and silero_speech) or energy_speech)

        # Option A silence gate: both speech-trained detectors must agree
        # the frame is not speech. Energy alone being quiet is not enough,
        # and energy alone being loud (ambient noise) is not enough to
        # block the silence counter.
        is_silent = (not silero_speech) and (not webrtc_speech)

        return VadDecision(
            frame_index=frame.index,
            start_ms=frame.start_ms,
            end_ms=frame.end_ms,
            webrtc_speech=webrtc_speech,
            silero_speech=silero_speech,
            energy_speech=energy_speech,
            energy_rms=float(rms or 0.0),
            decision=decision,
            is_silent=is_silent,
            silero_score=silero_score,
            metadata={
                "policy": "dual_vad",
                "require_silero_for_start": self.require_silero_for_start,
                "silence_gate": "silero_and_webrtc_negative",
            },
        )
