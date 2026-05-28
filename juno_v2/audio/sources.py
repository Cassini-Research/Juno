from __future__ import annotations

import queue
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

import numpy as np

from juno_v2.audio.clock import AudioFrameClock
from juno_v2.contracts.audio import AudioFrame


class AudioSource(ABC):
    """Abstract audio frame source."""

    @abstractmethod
    def frames(self) -> Iterable[AudioFrame]:
        raise NotImplementedError


@dataclass(slots=True)
class ReplayWavAudioSource(AudioSource):
    path: Path
    sample_rate_hz: int = 16000
    frame_ms: int = 20
    source_name: str = "wav_replay"

    def frames(self) -> Generator[AudioFrame, None, None]:
        path = Path(self.path)
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            if width != 2:
                raise ValueError(f"expected 16-bit PCM wav, got sample width={width}")
            if sample_rate != self.sample_rate_hz:
                raise ValueError(
                    f"expected sample_rate_hz={self.sample_rate_hz}, got {sample_rate}; WAV replay requires matching input rate"
                )
            clock = AudioFrameClock(sample_rate_hz=sample_rate, frame_ms=self.frame_ms, source=self.source_name)
            frame_bytes = clock.frame_samples * channels * width
            index = 0
            while True:
                chunk = wf.readframes(clock.frame_samples)
                if not chunk:
                    break
                if len(chunk) < frame_bytes:
                    chunk += b"\x00" * (frame_bytes - len(chunk))
                pcm = np.frombuffer(chunk, dtype=np.int16)
                if channels > 1:
                    pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
                samples = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                yield clock.make_frame(index=index, samples=samples)
                index += 1


@dataclass(slots=True)
class LiveMicrophoneAudioSource(AudioSource):
    sample_rate_hz: int = 16000
    frame_ms: int = 20
    channels: int = 1
    block_queue_maxsize: int = 128
    source_name: str = "microphone"

    def frames(self) -> Generator[AudioFrame, None, None]:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(
                "sounddevice is required for live microphone capture; install project dependencies before running live mode"
            ) from exc

        clock = AudioFrameClock(sample_rate_hz=self.sample_rate_hz, frame_ms=self.frame_ms, source=self.source_name)
        q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self.block_queue_maxsize)

        def callback(indata, frames, time_info, status) -> None:  # pragma: no cover - callback depends on live runtime
            del frames, time_info
            if status:
                pass
            data = np.copy(indata[:, 0]).astype(np.float32)
            try:
                q.put_nowait(data)
            except queue.Full:
                q.get_nowait()
                q.put_nowait(data)

        stream = sd.InputStream(
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            dtype="float32",
            blocksize=clock.frame_samples,
            callback=callback,
        )
        with stream:
            index = 0
            while True:
                block = q.get()
                if block is None:
                    break
                if block.shape[0] != clock.frame_samples:
                    if block.shape[0] < clock.frame_samples:
                        block = np.pad(block, (0, clock.frame_samples - block.shape[0]))
                    else:
                        block = block[: clock.frame_samples]
                yield clock.make_frame(index=index, samples=block)
                index += 1
