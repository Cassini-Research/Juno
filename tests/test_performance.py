"""Performance regression tests for hot text/audio paths.

These are not micro-benchmarks: budgets are set 10-50x above typical
timings (measured on Apple Silicon) so they only fail on order-of-magnitude
regressions (e.g. accidental quadratic behavior, pathological regex
backtracking), not on machine load. Each test uses the best (minimum)
of several runs, which is robust against transient CPU contention.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from juno_v2.context.app_classifier import classify_app_category
from juno_v2.context.redaction import ContextRedactor
from juno_v2.contracts.audio import AudioFrame
from juno_v2.contracts.speech import VadDecision
from juno_v2.itn.engine import ITNEngine, ITNProfile
from juno_v2.memory.fold import fold_key
from juno_v2.memory.hallucination import looks_like_hallucination
from juno_v2.speech.config import speech_state_config_for_profile
from juno_v2.speech.state_machine import SpeechStateMachine
from juno_v2.transcript.patching import diff_to_patch_ops
from juno_v2.writer.deterministic import normalize_plain_dictation


def best_of(runs: int, fn: Callable[[], object]) -> float:
    """Return the fastest wall-clock time of `runs` invocations, in seconds."""
    fn()  # warmup (imports, regex compilation, caches)
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _assert_budget(elapsed: float, budget_s: float, label: str) -> None:
    assert elapsed < budget_s, (
        f"{label} took {elapsed * 1000:.1f}ms (budget {budget_s * 1000:.0f}ms); "
        "an order-of-magnitude regression was introduced in this path"
    )


TRANSCRIPT_60_WORDS = (
    "okay so remind me tomorrow at five thirty pm to send the quarterly report "
    "comma and the budget is twenty five thousand dollars period also email "
    "support at example dot com about the api endpoint in main dot py before "
    "the meeting on april nineteenth twenty twenty six at nine am because we "
    "need the numbers for the review next monday morning"
)


def test_itn_full_profile_per_utterance_budget() -> None:
    engine = ITNEngine()
    elapsed = best_of(5, lambda: engine.run(TRANSCRIPT_60_WORDS, profile=ITNProfile.FULL))
    _assert_budget(elapsed, 0.10, "ITN FULL profile on a 60-word utterance")


def test_itn_scales_linearly_on_long_dictation() -> None:
    # ~1500 words: a long continuous dictation session.
    engine = ITNEngine()
    long_text = " ".join([TRANSCRIPT_60_WORDS] * 25)
    elapsed = best_of(3, lambda: engine.run(long_text, profile=ITNProfile.FULL))
    _assert_budget(elapsed, 1.0, "ITN FULL profile on a 1500-word dictation")


def test_normalize_plain_dictation_long_text() -> None:
    long_text = " ".join([TRANSCRIPT_60_WORDS] * 30)
    elapsed = best_of(5, lambda: normalize_plain_dictation(long_text))
    _assert_budget(elapsed, 0.25, "normalize_plain_dictation on ~1800 words")


def test_hallucination_detector_on_adversarial_input() -> None:
    # Highly repetitive text is the worst case for repetition scanning.
    repetitive = "thank you for watching " * 200
    normal = TRANSCRIPT_60_WORDS
    elapsed = best_of(
        5,
        lambda: (looks_like_hallucination(repetitive), looks_like_hallucination(normal)),
    )
    _assert_budget(elapsed, 0.10, "hallucination detection on 200x repeated phrase")


def test_redactor_on_10kb_mixed_text() -> None:
    redactor = ContextRedactor()
    chunk = (
        "Contact alice@example.com or visit https://internal.example.com/path?q=1 "
        "card 4532015112830366 code 123456 token sk-abcdef1234567890 plain words here "
    )
    text = chunk * (10_000 // len(chunk) + 1)
    elapsed = best_of(5, lambda: redactor.redact(text))
    _assert_budget(elapsed, 0.30, "ContextRedactor.redact on ~10KB")


def test_fold_key_vocabulary_scan() -> None:
    terms = [f"Term-{i} Café_{i} sign-off" for i in range(1000)]
    elapsed = best_of(5, lambda: [fold_key(t) for t in terms])
    _assert_budget(elapsed, 0.50, "fold_key over 1000 vocabulary terms")


def test_app_classifier_call_rate() -> None:
    apps = [
        ("Visual Studio Code", "main.py — juno", "com.microsoft.vscode"),
        ("Mail", "Re: budget", "com.apple.mail"),
        ("Terminal", "zsh", "com.apple.Terminal"),
        ("Slack", "#general", "com.tinyspeck.slackmacgap"),
        (None, None, None),
    ]
    elapsed = best_of(
        5,
        lambda: [
            classify_app_category(name, title, app_bundle_id=bundle)
            for _ in range(200)
            for (name, title, bundle) in apps
        ],
    )
    _assert_budget(elapsed, 0.30, "1000 app classification calls")


def test_diff_to_patch_ops_on_long_transcript() -> None:
    base = " ".join([TRANSCRIPT_60_WORDS] * 10)
    # A realistic adjudication: a few scattered word-level fixes.
    corrected = base.replace("comma", ",").replace("period", ".").replace(
        "twenty five thousand dollars", "$25,000"
    )
    elapsed = best_of(
        5,
        lambda: diff_to_patch_ops(
            base,
            corrected,
            stable_prefix_chars=len(base),
            reason="perf",
            confidence=0.9,
        ),
    )
    _assert_budget(elapsed, 0.40, "diff_to_patch_ops on ~600-word transcript")


def test_speech_state_machine_realtime_factor() -> None:
    """30 seconds of 20ms frames must process far faster than real time.

    The live engine calls process() on every frame between mic callback
    deadlines, so anything near RTF 1.0 would mean dropped audio.
    """
    config = speech_state_config_for_profile("standard")
    samples = np.zeros(320, dtype=np.float32)

    def run() -> None:
        machine = SpeechStateMachine(config=config)
        for i in range(1500):  # 30s at 20ms frames
            # Alternate speech/silence bursts to exercise all transitions.
            speech = (i // 100) % 2 == 0
            frame = AudioFrame(
                index=i,
                sample_rate_hz=16000,
                start_sample=i * 320,
                end_sample=(i + 1) * 320,
                start_ms=i * 20.0,
                end_ms=(i + 1) * 20.0,
                samples=samples,
            )
            decision = VadDecision(
                frame_index=i,
                start_ms=i * 20.0,
                end_ms=(i + 1) * 20.0,
                webrtc_speech=speech,
                silero_speech=speech,
                energy_speech=speech,
                energy_rms=0.1 if speech else 0.0,
                decision=speech,
                is_silent=not speech,
            )
            machine.process(frame, decision)

    elapsed = best_of(3, run)
    # Budget = 3s for 30s of audio → RTF 0.1, ~50x above typical.
    _assert_budget(elapsed, 3.0, "SpeechStateMachine over 30s of frames")
