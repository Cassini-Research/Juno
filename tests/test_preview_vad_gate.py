import numpy as np

from juno_v2.preview.vad_gate import VadGate


def test_vad_warmup_admits_chunk_that_crosses_warmup_boundary() -> None:
    gate = VadGate(
        sample_rate_hz=16_000,
        warmup_seconds=1.0,
        silero_enabled=False,
        energy_fallback_rms=1.0,
    )

    admit, reason = gate.should_admit(np.zeros(3_200, dtype=np.float32))
    assert admit is True
    assert reason == "warmup"

    admit, reason = gate.should_admit(np.zeros(14_400, dtype=np.float32))
    assert admit is True
    assert reason == "warmup"

    admit, reason = gate.should_admit(np.zeros(3_200, dtype=np.float32))
    assert admit is False
    assert reason == "energy_fallback_silence"
