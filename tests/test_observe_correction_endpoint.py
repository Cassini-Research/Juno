"""Regression tests for ``WorkbenchApp.broker_observe_correction`` (issue #121).

The endpoint may only report, trace and count a correction as observed when
the memory store actually accepted it. A pair the store's safety filter
refuses must come back not-ok, must not emit ``correction_observed`` and must
not touch the utterance's ``correction_count`` in product history.

Every string below is a synthetic fixture, and the memory store, trace log and
history database all live under ``tmp_path``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from juno_v2.memory.store import JsonMemoryStore
from juno_v2.memory.stores.corrections import (
    MAX_CORRECTION_TEXT_CHARS,
    MAX_CORRECTION_WORDS,
    is_safe_correction_pair,
)
from juno_v2.observability.product_history import get_product_history_store
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.runtime.config import WorkbenchRuntimeConfig
from juno_v2.workbench.server import WorkbenchApp

# One word over the store's word cap, still well under its character cap, with
# a single token edited: the shape issue #121 was measured on.
_OVER_WORD_CAP = " ".join(f"token{i:02d}" for i in range(MAX_CORRECTION_WORDS + 1))
_REJECTED_PAIR = (_OVER_WORD_CAP, _OVER_WORD_CAP.replace("token06", "tokin06"))

# Pairs ``is_safe_correction_pair`` refuses, each under a different rule.
_REJECTED_PAIRS = [
    pytest.param(*_REJECTED_PAIR, id="over_word_cap"),
    pytest.param("zentrava", "zentriva " + "q" * MAX_CORRECTION_TEXT_CHARS, id="over_char_cap"),
    pytest.param("the launch plan", "The launch plan", id="case_only_edit"),
    pytest.param("launch plan", "launch plan needs three more reviewers today", id="fragment_expansion"),
    pytest.param("zen zen zen zen zen", "zentriva", id="hallucination_shape"),
]

# Pairs the store accepts.
_ACCEPTED_PAIR = ("zentrava", "zentriva")
_OTHER_ACCEPTED_PAIR = ("dravone", "dravona")

_REJECTED_RESULT = {"ok": False, "error": "correction_rejected"}


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkbenchApp:
    """A ``WorkbenchApp`` carrying only the collaborators the endpoint touches.

    ``WorkbenchApp.__init__`` wires up the whole dictation engine, so (as in
    ``test_visible_seed_replacements.py``) the instance is allocated bare and
    handed a real memory store, trace recorder and config rooted in
    ``tmp_path``.
    """
    # Without this override the history database resolves under ``log_dir``.
    monkeypatch.delenv("JUNO_PRODUCT_HISTORY_DIR", raising=False)
    workbench = object.__new__(WorkbenchApp)
    workbench.memory = JsonMemoryStore(tmp_path / "memory")
    workbench.recorder = TraceRecorder(
        session_id="observe-correction-test",
        log_dir=tmp_path / "traces",
    )
    workbench.config = WorkbenchRuntimeConfig(log_dir=tmp_path / "workbench")
    workbench._settings = {"learn_from_corrections": True}
    workbench._lock = threading.RLock()
    history_db = get_product_history_store(Path(workbench.config.log_dir)).db_path
    assert tmp_path.resolve() in history_db.parents
    return workbench


def _commit_utterance(app: WorkbenchApp, utterance_id: str, text: str) -> None:
    """Create the history row a successful paste leaves behind."""
    store = get_product_history_store(Path(app.config.log_dir))
    assert store.update_insertion_commit(
        utterance_id=utterance_id,
        committed_text=text,
        ok=True,
        paste_kind="insert",
    )


def _history_entry(app: WorkbenchApp, utterance_id: str) -> dict[str, Any]:
    """The utterance as the History API lists it."""
    entries = get_product_history_store(Path(app.config.log_dir)).list_entries(limit=50)
    return next(entry for entry in entries if entry["utterance_id"] == utterance_id)


def _trace_payloads(app: WorkbenchApp, name: str) -> list[dict[str, Any]]:
    """Payloads of the recorded trace events called ``name``, oldest first."""
    return [
        event["payload"]
        for event in reversed(app.recorder.recent_events())
        if event["name"] == name
    ]


def _stored_corrections(app: WorkbenchApp) -> list[tuple[str, str, int]]:
    return [
        (row["observed"], row["corrected"], row["count"])
        for row in app.memory.corrections.raw()
    ]


# --------------------------------------------------------------------- #
# Rejected by the store
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(("observed", "corrected"), _REJECTED_PAIRS)
def test_rejected_pair_returns_correction_rejected_and_stores_nothing(
    app: WorkbenchApp, observed: str, corrected: str
) -> None:
    assert not is_safe_correction_pair(observed, corrected)

    result = app.broker_observe_correction({"observed": observed, "corrected": corrected})

    assert result == _REJECTED_RESULT
    assert _stored_corrections(app) == []


def test_rejected_pair_is_not_traced_as_observed(app: WorkbenchApp) -> None:
    observed, corrected = _REJECTED_PAIR

    app.broker_observe_correction({"observed": observed, "corrected": corrected})

    assert _trace_payloads(app, "correction_observed") == []


def test_rejected_pair_does_not_increment_correction_count(app: WorkbenchApp) -> None:
    observed, corrected = _REJECTED_PAIR
    _commit_utterance(app, "utt-rejected-pair", observed)
    before = _history_entry(app, "utt-rejected-pair")
    assert before["correction_count"] == 0

    app.broker_observe_correction(
        {"utterance_id": "utt-rejected-pair", "observed": observed, "corrected": corrected}
    )

    # The whole row is untouched: no count, and no ``updated_at`` bump that
    # would re-sort the utterance to the top of History.
    assert _history_entry(app, "utt-rejected-pair") == before


def test_rejected_pair_leaves_a_text_free_rejection_trace(app: WorkbenchApp) -> None:
    observed, corrected = _REJECTED_PAIR

    app.broker_observe_correction(
        {"source": "synthetic_tight_pair", "observed": observed, "corrected": corrected}
    )

    assert _trace_payloads(app, "correction_rejected") == [
        {
            "source": "synthetic_tight_pair",
            "observed_chars": len(observed),
            "corrected_chars": len(corrected),
        }
    ]


def test_post_paste_monitor_observations_the_store_rejects_never_reach_history(
    app: WorkbenchApp,
) -> None:
    """The shell's post-paste monitor path: one paste, eight refused posts."""
    pasted = "the launch plan is ready"
    field_after_edit = "The launch plan is ready"
    _commit_utterance(app, "utt-monitor-rejected", pasted)
    payload = {
        "source": "mac_post_paste_monitor",
        "utterance_id": "utt-monitor-rejected",
        "observed": pasted,
        "corrected": field_after_edit,
        "app_bundle_id": "com.example.editor",
    }

    results = [app.broker_observe_correction(dict(payload)) for _ in range(8)]

    assert results == [_REJECTED_RESULT] * 8
    assert _stored_corrections(app) == []
    assert _trace_payloads(app, "correction_observed") == []
    assert _history_entry(app, "utt-monitor-rejected")["correction_count"] == 0


# --------------------------------------------------------------------- #
# Accepted by the store
# --------------------------------------------------------------------- #


def test_accepted_pair_is_stored_traced_counted_and_reported_ok(app: WorkbenchApp) -> None:
    observed, corrected = _ACCEPTED_PAIR
    assert is_safe_correction_pair(observed, corrected)
    _commit_utterance(app, "utt-accepted-pair", "ship the zentrava build")

    result = app.broker_observe_correction(
        {
            "source": "synthetic_tight_pair",
            "utterance_id": "utt-accepted-pair",
            "observed": observed,
            "corrected": corrected,
        }
    )

    assert result == {"ok": True, "observed": observed, "corrected": corrected}
    assert _stored_corrections(app) == [(observed, corrected, 1)]
    assert _trace_payloads(app, "correction_observed") == [
        {
            "source": "synthetic_tight_pair",
            "observed_chars": len(observed),
            "corrected_chars": len(corrected),
            "expected_chars": len(corrected),
        }
    ]
    assert _trace_payloads(app, "correction_rejected") == []
    assert _history_entry(app, "utt-accepted-pair")["correction_count"] == 1


def test_post_paste_monitor_observation_the_store_accepts_is_counted(app: WorkbenchApp) -> None:
    pasted = "send the zentrava report today"
    field_after_edit = "send the zentriva report today"
    _commit_utterance(app, "utt-monitor-accepted", pasted)

    result = app.broker_observe_correction(
        {
            "source": "mac_post_paste_monitor",
            "utterance_id": "utt-monitor-accepted",
            "observed": pasted,
            "corrected": field_after_edit,
            "app_bundle_id": "com.example.editor",
        }
    )

    assert result["ok"] is True
    assert "zentriva" in result["corrected"]
    assert _stored_corrections(app) == [(result["observed"], result["corrected"], 1)]
    assert [event["source"] for event in _trace_payloads(app, "correction_observed")] == [
        "mac_post_paste_monitor"
    ]
    assert _history_entry(app, "utt-monitor-accepted")["correction_count"] == 1


def test_repeat_of_a_stored_correction_counts_each_time_the_store_reinforces_it(
    app: WorkbenchApp,
) -> None:
    """A repeat is accepted by the store (it bumps the row's ``count``), so the
    endpoint traces and counts it once per reinforcement."""
    observed, corrected = _ACCEPTED_PAIR
    _commit_utterance(app, "utt-repeated-pair", "ship the zentrava build")
    payload = {"utterance_id": "utt-repeated-pair", "observed": observed, "corrected": corrected}

    results = [app.broker_observe_correction(dict(payload)) for _ in range(2)]

    assert [result["ok"] for result in results] == [True, True]
    assert _stored_corrections(app) == [(observed, corrected, 2)]
    assert len(_trace_payloads(app, "correction_observed")) == 2
    assert _history_entry(app, "utt-repeated-pair")["correction_count"] == 2


def test_correction_count_matches_what_the_store_accepted_for_the_utterance(
    app: WorkbenchApp,
) -> None:
    _commit_utterance(app, "utt-mixed-pairs", "ship the zentrava build to dravone")
    pairs = [_ACCEPTED_PAIR, _REJECTED_PAIR, _REJECTED_PAIR, _OTHER_ACCEPTED_PAIR]

    results = [
        app.broker_observe_correction(
            {"utterance_id": "utt-mixed-pairs", "observed": observed, "corrected": corrected}
        )
        for observed, corrected in pairs
    ]

    assert [result["ok"] for result in results] == [True, False, False, True]
    assert _stored_corrections(app) == [(*_ACCEPTED_PAIR, 1), (*_OTHER_ACCEPTED_PAIR, 1)]
    assert len(_trace_payloads(app, "correction_observed")) == 2
    assert len(_trace_payloads(app, "correction_rejected")) == 2
    assert _history_entry(app, "utt-mixed-pairs")["correction_count"] == 2
