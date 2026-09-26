from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
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

    def test_empty_final_flushes_an_existing_utterance(self) -> None:
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
        self.assertTrue(self.upstream.called.wait(timeout=2))


if __name__ == "__main__":
    unittest.main()
