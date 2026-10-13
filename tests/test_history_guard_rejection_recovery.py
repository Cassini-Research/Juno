"""History recovery hints for takes the engine's own guards rejected.

A guard rejection is a healthy engine deciding a take held no usable speech.
The shell reports it with the engine's code, and the row keeps no transcript,
so History must not offer to copy or re-insert one.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from juno_v2.workbench.server import WorkbenchApp

SILENCE_CODES = ("low_signal_audio", "silence_hallucination")
REJECTED_TRANSCRIPT_CODE = "hallucinated_transcript"
GUARD_CODES = (*SILENCE_CODES, REJECTED_TRANSCRIPT_CODE)


class _SilentRecorder:
    def record(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _app(log_dir: Path) -> WorkbenchApp:
    app = object.__new__(WorkbenchApp)
    app.config = SimpleNamespace(log_dir=log_dir, runtime_dir=None)
    app.session_id = "synthetic-session"
    app.recorder = _SilentRecorder()
    app._record_utterance_lifecycle = lambda *_args, **_kwargs: None
    return app


def _rejected_take_payload(code: str) -> dict[str, Any]:
    """The insertion/committed body the shell posts for a rejected take."""
    return {
        "utterance_id": f"synthetic-{code}",
        "transcript": "",
        "transcript_len": 0,
        "target_pid": 0,
        "ok": False,
        "paste_kind": "none",
        "paste_attempted": False,
        "trigger_source": "hotkey",
        "paste_failure_reason": code,
        "failure_reason": code,
        "app_name": "Notes",
        "app_bundle_id": "com.apple.Notes",
    }


def _history_row(tmp_path: Path, code: str) -> dict[str, Any]:
    app = _app(tmp_path)
    assert app.broker_insertion_committed(_rejected_take_payload(code))["ok"] is True
    entries = app.broker_utterance_history()["entries"]
    assert len(entries) == 1
    return entries[0]


@pytest.mark.parametrize("code", GUARD_CODES)
def test_guard_rejection_row_keeps_engine_code_and_offers_no_transcript_actions(
    tmp_path: Path, code: str
) -> None:
    row = _history_row(tmp_path, code)

    assert row["failure_reason"] == code
    assert row["transcript"] == ""
    recovery = row["recovery"]
    assert recovery["failure_code"] == code
    assert recovery["has_text"] is False
    assert recovery["actions"] == []
    assert recovery["category"] == "capture"


@pytest.mark.parametrize("code", SILENCE_CODES)
def test_silence_rejection_matches_the_existing_empty_audio_recovery(
    tmp_path: Path, code: str
) -> None:
    app = _app(tmp_path)

    hints = app._derive_recovery_hints({"failure_reason": code}, audio_present=False)
    empty_audio = app._derive_recovery_hints({"failure_reason": "empty_audio"}, audio_present=False)

    assert hints["severity"] == "info"
    assert {k: v for k, v in hints.items() if k != "failure_code"} == {
        k: v for k, v in empty_audio.items() if k != "failure_code"
    }


def test_rejected_transcript_is_a_warning_not_a_silent_take(tmp_path: Path) -> None:
    app = _app(tmp_path)

    hints = app._derive_recovery_hints(
        {"failure_reason": REJECTED_TRANSCRIPT_CODE}, audio_present=False
    )

    assert hints["severity"] == "warning"
    assert hints["actions"] == []


@pytest.mark.parametrize("code", GUARD_CODES)
def test_guard_rejection_with_retained_audio_offers_only_audio_actions(
    tmp_path: Path, code: str
) -> None:
    app = _app(tmp_path)

    hints = app._derive_recovery_hints({"failure_reason": code}, audio_present=True)

    assert hints["actions"] == ["replay_audio", "rerun_in_mode"]


def test_real_transcribe_failure_is_not_reclassified_as_a_guard_rejection(tmp_path: Path) -> None:
    row = _history_row(tmp_path, "transcribe_failed")

    assert row["failure_reason"] == "transcribe_failed"
    assert row["recovery"]["failure_code"] == "transcribe_failed"
    assert row["recovery"]["severity"] == "warning"
    assert row["recovery"]["category"] != "capture"
