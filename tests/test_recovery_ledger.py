from __future__ import annotations

import json

import pytest

from juno_core_v3.recovery.ledger import RecoveryEntry, RecoveryLedger


def make_entry(
    *,
    ts: int = 1_700_000_000_000,
    kind: str = "committed",
    utterance_id: str | None = "utt_1",
    text: str = "hello world",
    metadata: dict | None = None,
) -> RecoveryEntry:
    return RecoveryEntry(
        ts_unix_ms=ts,
        broker_session_id="bro_test",
        kind=kind,
        utterance_id=utterance_id,
        text=text,
        metadata=metadata if metadata is not None else {"source": "test"},
    )


# ---- RecoveryEntry ------------------------------------------------------


def test_entry_json_line_round_trip() -> None:
    entry = make_entry(metadata={"a": 1, "nested": {"b": [1, 2]}})
    line = entry.to_json_line()
    assert line.endswith("\n")
    parsed = RecoveryEntry.from_json_line(line)
    assert parsed == entry


def test_entry_round_trip_preserves_none_utterance_and_unicode() -> None:
    entry = make_entry(utterance_id=None, text="héllo — ünïcode ✓")
    parsed = RecoveryEntry.from_json_line(entry.to_json_line())
    assert parsed.utterance_id is None
    assert parsed.text == "héllo — ünïcode ✓"
    # ensure_ascii=False keeps the text human-inspectable on disk.
    assert "héllo" in entry.to_json_line()


def test_entry_from_json_line_tolerates_missing_optional_fields() -> None:
    line = json.dumps(
        {"ts_unix_ms": 5, "broker_session_id": "b", "kind": "committed"}
    )
    parsed = RecoveryEntry.from_json_line(line)
    assert parsed.ts_unix_ms == 5
    assert parsed.utterance_id is None
    assert parsed.text == ""
    assert parsed.metadata == {}


def test_entry_from_json_line_null_text_and_metadata() -> None:
    line = json.dumps(
        {
            "ts_unix_ms": "7",
            "broker_session_id": "b",
            "kind": "capture_note",
            "utterance_id": "u",
            "text": None,
            "metadata": None,
        }
    )
    parsed = RecoveryEntry.from_json_line(line)
    assert parsed.ts_unix_ms == 7
    assert parsed.text == ""
    assert parsed.metadata == {}


# ---- RecoveryLedger ------------------------------------------------------


def test_append_creates_session_dir_and_file(tmp_path) -> None:
    ledger = RecoveryLedger(recovery_root=tmp_path / "recovery", broker_session_id="bro_abc")
    assert not ledger.session_dir.exists()
    ledger.append(make_entry())
    assert ledger.session_dir == tmp_path / "recovery" / "bro_abc"
    assert ledger.session_dir.is_dir()
    assert ledger.path == ledger.session_dir / "ledger.jsonl"
    assert ledger.path.is_file()
    assert len(ledger.path.read_text(encoding="utf-8").splitlines()) == 1


def test_append_is_append_only(tmp_path) -> None:
    ledger = RecoveryLedger(recovery_root=tmp_path, broker_session_id="bro_abc")
    ledger.append(make_entry(ts=1, text="first"))
    ledger.append(make_entry(ts=2, text="second"))
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "first"
    assert json.loads(lines[1])["text"] == "second"


def test_iter_entries_empty_when_file_missing(tmp_path) -> None:
    ledger = RecoveryLedger(recovery_root=tmp_path, broker_session_id="bro_missing")
    assert list(ledger.iter_entries()) == []
    assert ledger.read_all() == []


def test_iter_entries_skips_blank_lines(tmp_path) -> None:
    ledger = RecoveryLedger(recovery_root=tmp_path, broker_session_id="bro_abc")
    ledger.session_dir.mkdir(parents=True)
    e1 = make_entry(ts=1, text="one")
    e2 = make_entry(ts=2, text="two")
    ledger.path.write_text(
        "\n" + e1.to_json_line() + "\n\n   \n" + e2.to_json_line() + "\n",
        encoding="utf-8",
    )
    entries = list(ledger.iter_entries())
    assert [e.text for e in entries] == ["one", "two"]


def test_read_all_preserves_append_order(tmp_path) -> None:
    ledger = RecoveryLedger(recovery_root=tmp_path, broker_session_id="bro_abc")
    for i in range(5):
        ledger.append(make_entry(ts=i, text=f"t{i}", utterance_id=f"u{i}"))
    entries = ledger.read_all()
    assert [e.text for e in entries] == ["t0", "t1", "t2", "t3", "t4"]
    assert [e.ts_unix_ms for e in entries] == [0, 1, 2, 3, 4]
    assert entries == [
        make_entry(ts=i, text=f"t{i}", utterance_id=f"u{i}") for i in range(5)
    ]


def test_two_sessions_do_not_share_ledgers(tmp_path) -> None:
    a = RecoveryLedger(recovery_root=tmp_path, broker_session_id="bro_a")
    b = RecoveryLedger(recovery_root=tmp_path, broker_session_id="bro_b")
    a.append(make_entry(text="from a"))
    assert [e.text for e in a.read_all()] == ["from a"]
    assert b.read_all() == []


# ---- RecoverySession (hermetic paths only) --------------------------------


@pytest.fixture()
def recovery_session(tmp_path):
    from juno_v2.observability.tracing import TraceRecorder

    from juno_core_v3.recovery.session import RecoverySession

    recorder = TraceRecorder("sess_test", tmp_path / "traces")
    session = RecoverySession(
        broker_session_id="bro_sess",
        recovery_root=tmp_path / "recovery",
        recorder=recorder,
    )
    return session


def test_session_ingest_committed_snapshot(recovery_session) -> None:
    recovery_session.ingest_from_workbench_snapshot(
        {
            "last_committed_text": "final text",
            "last_committed_utterance_id": "utt_9",
            "pending_commit": False,
        }
    )
    entries = recovery_session.ledger.read_all()
    assert len(entries) == 1
    assert entries[0].kind == "committed"
    assert entries[0].text == "final text"
    assert entries[0].utterance_id == "utt_9"


def test_session_ingest_staged_fallback_when_pending_without_commit(recovery_session) -> None:
    recovery_session.ingest_from_workbench_snapshot(
        {
            "last_committed_text": "   ",
            "final_candidate_text": "never lose me",
            "pending_commit": True,
        }
    )
    entries = recovery_session.ledger.read_all()
    assert len(entries) == 1
    assert entries[0].kind == "staged_fallback"
    assert entries[0].text == "never lose me"


def test_session_ingest_noop_when_nothing_recoverable(recovery_session) -> None:
    recovery_session.ingest_from_workbench_snapshot(
        {"last_committed_text": "", "final_candidate_text": "", "pending_commit": False}
    )
    assert recovery_session.ledger.read_all() == []


def test_session_paste_last_prefers_committed_over_staged(recovery_session) -> None:
    assert recovery_session.paste_last_transcript() is None
    recovery_session.ingest_from_workbench_snapshot(
        {"last_committed_text": "", "final_candidate_text": "staged one", "pending_commit": True}
    )
    assert recovery_session.paste_last_transcript() == "staged one"
    recovery_session.ingest_from_workbench_snapshot(
        {"last_committed_text": "committed one", "pending_commit": False}
    )
    assert recovery_session.paste_last_transcript() == "committed one"


def test_session_retry_without_committed_text_raises(recovery_session) -> None:
    with pytest.raises(ValueError, match="no committed transcript to retry"):
        recovery_session.retry_last_commit_append(None)


def test_session_history_shape(recovery_session) -> None:
    recovery_session.ingest_from_workbench_snapshot(
        {"last_committed_text": "x" * 300, "last_committed_utterance_id": "utt_1"}
    )
    (item,) = recovery_session.history()
    assert item["kind"] == "committed"
    assert item["utterance_id"] == "utt_1"
    assert item["text_length"] == 300
    assert item["text_preview"] == "x" * 200
