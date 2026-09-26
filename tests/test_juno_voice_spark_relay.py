from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time
import types
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "juno_voice_spark_relay.py"
)
self_corrections = types.ModuleType("self_corrections")
self_corrections.apply_unambiguous_retakes = lambda text: (text, [])
sys.modules["self_corrections"] = self_corrections
SPEC = importlib.util.spec_from_file_location("juno_voice_spark_relay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
relay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relay)


class FakeUpstream:
    def __init__(self) -> None:
        self.calls = 0
        self.called = threading.Event()

    def transcribe_pcm(self, pcm: bytes, sample_rate: int, language: str) -> str:
        self.calls += 1
        self.called.set()
        return "Hello from the Spark."


class FailingBlockingUpstream:
    def __init__(self) -> None:
        self.calls = 0
        self.called = threading.Event()
        self.release = threading.Event()

    def transcribe_pcm(self, pcm: bytes, sample_rate: int, language: str) -> str:
        self.calls += 1
        self.called.set()
        self.release.wait(timeout=1)
        raise relay.RelayError("temporary upstream failure")


class OrderedLiveSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = FakeUpstream()
        self.store = relay.SessionStore(
            self.upstream,
            max_sessions=2,
            ttl_seconds=60,
            preview_seconds=0.5,
        )
        self.session = self.store.create("en", 16_000)

    def tearDown(self) -> None:
        self.store.stopping.set()
        self.store.executor.shutdown(wait=True, cancel_futures=True)

    def test_ordered_chunk_is_idempotent(self) -> None:
        pcm = b"\x01\x00" * 8_000
        first = self.store.append_ordered(
            self.session,
            sequence=0,
            pcm=pcm,
            is_final=False,
        )
        duplicate = self.store.append_ordered(
            self.session,
            sequence=0,
            pcm=pcm,
            is_final=False,
        )

        self.assertEqual(first, duplicate)
        self.assertEqual(len(self.session.pcm), len(pcm))
        self.assertEqual(self.session.next_sequence, 1)

    def test_out_of_order_chunk_is_rejected(self) -> None:
        with self.assertRaisesRegex(relay.RelayError, "out of order"):
            self.store.append_ordered(
                self.session,
                sequence=2,
                pcm=b"\0\0",
                is_final=False,
            )

    def test_empty_final_yields_to_the_authoritative_final_lane(self) -> None:
        self.store.append_ordered(
            self.session,
            sequence=0,
            pcm=b"\x01\x00" * 100,
            is_final=False,
        )
        response = self.store.append_ordered(
            self.session,
            sequence=1,
            pcm=b"",
            is_final=True,
        )

        self.assertTrue(response["is_final"])
        self.assertEqual(response["sequence"], 1)
        self.assertFalse(self.session.accepting_audio)
        self.assertFalse(self.upstream.called.wait(timeout=0.05))

    def test_failed_preview_waits_for_a_full_new_cadence_before_retry(self) -> None:
        self.store.stopping.set()
        self.store.executor.shutdown(wait=True, cancel_futures=True)
        upstream = FailingBlockingUpstream()
        store = relay.SessionStore(
            upstream,
            max_sessions=2,
            ttl_seconds=60,
            preview_seconds=0.5,
        )
        session = store.create("en", 16_000)
        try:
            # 0.5 seconds schedules the first inference.
            store.append(session, b"\x01\x00" * 8_000)
            self.assertTrue(upstream.called.wait(timeout=1))

            # A 0.25-second append during the failed request must not trigger
            # the old retry storm when that request returns.
            store.append(session, b"\x01\x00" * 4_000)
            upstream.release.set()
            time.sleep(0.1)
            self.assertEqual(upstream.calls, 1)

            # The next 0.25 seconds completes one new preview window.
            store.append(session, b"\x01\x00" * 4_000)
            deadline = time.monotonic() + 1
            while upstream.calls < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(upstream.calls, 2)
        finally:
            store.stopping.set()
            store.executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
