from __future__ import annotations

import io
import math
import sqlite3
import struct
import wave
from types import SimpleNamespace

from juno_core_v3.broker.runners import InsertRequest, InsertRunner
from juno_core_v3.dictation.pipeline import OneShotDictationPipeline
from juno_core_v3.dictation.transcriber import TranscribeResult
from juno_v2.observability.history_store import read_persistent_history
from juno_v2.observability.product_history import ProductHistoryStore, _SCHEMA
from juno_v2.observability.reliability_provenance import test_provenance_fields as provenance_fields
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.runtime.uds_dispatch import dispatch_broker_http_like
from juno_v2.transcript.adjudicator import TranscriptAdjudicatorConfig


def _loud_wav_bytes() -> bytes:
    sample_rate = 16_000
    frames = [
        struct.pack("<h", int(12_000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
        for i in range(sample_rate // 2)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(frames))
    return buf.getvalue()


class _FakeTranscriber:
    backend_name = "fake_asr"

    def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
        return TranscribeResult(
            transcript="The frozen reference stays attributable.",
            language="en",
            backend_name=self.backend_name,
            audio_duration_ms=500.0,
            decode_ms=1.0,
            model_path="fake",
        )


def test_test_provenance_is_sparse_bounded_and_content_free() -> None:
    assert provenance_fields() == {}
    assert provenance_fields(
        test_run_id=" biweekly.2026-08-31 ",
        test_case_id="cafe:long-03",
    ) == {
        "test_run_id": "biweekly.2026-08-31",
        "test_case_id": "cafe:long-03",
    }
    assert provenance_fields(
        test_run_id="this is free form text",
        test_case_id="x" * 129,
    ) == {}


def test_product_history_migrates_and_filters_test_provenance(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    legacy_schema = _SCHEMA.replace(
        "actions_json TEXT,\n    test_run_id TEXT,\n    test_case_id TEXT",
        "actions_json TEXT",
    )
    with sqlite3.connect(db_path) as con:
        con.executescript(legacy_schema)
        con.execute(
            "INSERT INTO utterances (utterance_id, created_at, updated_at, transcript) VALUES (?, ?, ?, ?)",
            ("legacy", 1, 1, "old row"),
        )
        con.commit()

    store = ProductHistoryStore(db_path)
    store.init_schema()
    with sqlite3.connect(db_path) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(utterances)")}
    assert {"test_run_id", "test_case_id"}.issubset(columns)
    legacy = store.list_entries()[0]
    assert legacy["utterance_id"] == "legacy"
    assert "test_run_id" not in legacy
    assert "test_case_id" not in legacy

    store.upsert_from_pipeline_record(
        {
            "utterance_id": "case-1",
            "ts_unix_ms": 2,
            "transcript": "new row",
            "test_run_id": "run-17",
            "test_case_id": "clean:01",
        }
    )
    matching = store.list_entries(test_run_id="run-17", test_case_id="clean:01")
    assert [entry["utterance_id"] for entry in matching] == ["case-1"]
    assert matching[0]["test_run_id"] == "run-17"
    assert matching[0]["test_case_id"] == "clean:01"
    assert store.list_entries(test_run_id="different") == []
    assert store.list_entries(test_run_id="invalid free form filter") == []


def test_pipeline_persists_provenance_in_decision_history_and_result(tmp_path) -> None:
    recorder = TraceRecorder("evidence-ledger", tmp_path)
    pipeline = OneShotDictationPipeline(
        transcriber=_FakeTranscriber(),
        recorder=recorder,
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-evidence-1",
        save_history=True,
        save_audio=False,
        test_run_id="run-17",
        test_case_id="clean:01",
    )

    assert result.ok
    assert result.metadata["test_run_id"] == "run-17"
    assert result.metadata["test_case_id"] == "clean:01"
    decision = next(
        event
        for event in recorder.recent_events()
        if event.get("name") == "transcript_decision"
    )
    assert decision["payload"]["test_run_id"] == "run-17"
    assert decision["payload"]["test_case_id"] == "clean:01"

    rows = read_persistent_history(
        tmp_path,
        test_run_id="run-17",
        test_case_id="clean:01",
    )
    assert len(rows) == 1
    assert rows[0]["utterance_id"] == "utt-evidence-1"
    assert rows[0]["test_run_id"] == "run-17"
    assert rows[0]["test_case_id"] == "clean:01"


def test_insert_runner_forwards_test_provenance() -> None:
    class _Pipeline:
        writer_service = None

        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def run(self, wav_bytes: bytes, **kwargs: object) -> SimpleNamespace:
            self.kwargs = kwargs
            return SimpleNamespace(ok=True)

    pipeline = _Pipeline()
    InsertRunner(pipeline=pipeline).run(  # type: ignore[arg-type]
        InsertRequest(
            wav_bytes=b"wav",
            test_run_id="run-17",
            test_case_id="clean:01",
        )
    )

    assert pipeline.kwargs["test_run_id"] == "run-17"
    assert pipeline.kwargs["test_case_id"] == "clean:01"


def test_uds_broker_forwards_test_provenance() -> None:
    class _App:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def broker_dictation_transcribe(self, wav_bytes: bytes, **kwargs: object) -> dict[str, object]:
            self.kwargs = kwargs
            return {"ok": True, "audio_bytes": len(wav_bytes)}

    app = _App()
    out = dispatch_broker_http_like(
        app,
        {
            "method": "POST",
            "path": "/api/broker/dictation/ingest_wav",
            "payload": {
                "utterance_id": "utt-evidence-uds",
                "test_run_id": "run-17",
                "test_case_id": "cafe:03",
            },
        },
        b"wav",
    )

    assert out == {"ok": True, "audio_bytes": 3}
    assert app.kwargs["test_run_id"] == "run-17"
    assert app.kwargs["test_case_id"] == "cafe:03"
