"""Commit-time promotion must learn user speech, not pipeline repairs.

Issue #80: a screen-derived protected term ("PASTF") rewrote a correctly heard
common word ("paste") in the near-miss reconciler, and the commit path then
promoted that label into session memory and the lexicon — treating the
pipeline's own guess as evidence the user said it. These tests pin the
provenance gate: a token is promotable only when the raw ASR (or a user
correction) actually produced it.
"""

from __future__ import annotations

import math
import struct
import tempfile
import wave

from juno_core_v3.dictation.pipeline import (
    OneShotDictationPipeline,
    UtteranceRecord,
    _commit_rewrite_is_repair_only,
    _entity_repair_introduced,
    _repair_introduced_surfaces,
)
from juno_core_v3.dictation.transcriber import TranscribeResult
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot


# --------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------- #


def _loud_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    frames = int(duration_s * sample_rate)
    samples = [
        int(12000 * math.sin(2 * math.pi * 220.0 * (i / sample_rate))) for i in range(frames)
    ]
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        with wave.open(handle.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        with open(handle.name, "rb") as fh:
            return fh.read()


class _Recorder:
    def __init__(self) -> None:
        self.log_dir = tempfile.mkdtemp(prefix="juno-commit-promotion-test-")
        self.events: list[tuple[object, str, dict]] = []

    def record(self, kind: object, name: str, payload: dict) -> None:
        self.events.append((kind, name, payload))

    def payloads(self, name: str) -> list[dict]:
        return [payload for _kind, event, payload in self.events if event == name]


class _FakeTranscriber:
    backend_name = "fake_asr"

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
        return TranscribeResult(
            transcript=self.transcript,
            language="en",
            backend_name="fake_asr",
            audio_duration_ms=1000.0,
            decode_ms=1.0,
            model_path="fake",
        )


class _FakeContextProvider:
    def __init__(self, bundle: TypedContextBundle) -> None:
        self.bundle = bundle

    def snapshot(self) -> TypedContextBundle:
        return self.bundle


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.entities: list[str] = []
        self.corrections: list[tuple[str, str]] = []

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(schema_version=1)

    def record_correction(self, raw: str, committed: str) -> bool:
        self.corrections.append((raw, committed))
        return True

    def upsert_session_entities(self, entities: list[str], source: str) -> None:
        self.entities.extend(entities)


class _FakeLearnedStore:
    def __init__(self) -> None:
        self.observations: list[str] = []
        self.acceptances: list[str] = []

    def increment_observation(self, token: str, *, from_suppressed_context: bool) -> None:
        self.observations.append(token)

    def increment_acceptance(self, token: str, *, from_suppressed_context: bool) -> None:
        self.acceptances.append(token)


class _FakePromotion:
    def __init__(self) -> None:
        self.correction_calls: list[dict] = []
        self.context_calls: list[dict] = []

    def maybe_promote_correction_to_lexicon(self, **kwargs: object) -> dict:
        self.correction_calls.append(dict(kwargs))
        return {"promoted": True}

    def maybe_promote_context_entity_to_lexicon(self, **kwargs: object) -> dict:
        self.context_calls.append(dict(kwargs))
        return {"promoted": True}


class _FakeSeedRuntime:
    def __init__(self) -> None:
        self.learned_store = _FakeLearnedStore()
        self.promotion = _FakePromotion()

    def build_seed_attachment(self, **kwargs: object) -> None:
        return None

    def observe_transcript_for_context_entities(self, *args: object, **kwargs: object) -> None:
        return None

    def context_plane_suppression_value(self, metadata: dict) -> None:
        return None

    def durable_memory_suppressed(self, *args: object, **kwargs: object) -> bool:
        return False


def _run_commit(
    *,
    utterance_id: str,
    transcript: str,
    context: TypedContextBundle,
) -> tuple[str, _FakeMemoryStore, _FakeSeedRuntime, _Recorder, dict]:
    memory = _FakeMemoryStore()
    seed = _FakeSeedRuntime()
    recorder = _Recorder()
    pipeline = OneShotDictationPipeline(
        transcriber=_FakeTranscriber(transcript),
        recorder=recorder,
        context_provider=_FakeContextProvider(context),
        memory_store=memory,  # type: ignore[arg-type]
        juno_seed_runtime=seed,  # type: ignore[arg-type]
        writer_enabled=False,
    )
    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id=utterance_id,
        save_history=False,
        save_audio=False,
    )
    assert result.ok
    learned = pipeline.record_insertion(
        utterance_id=utterance_id,
        committed_text=result.transcript,
    )
    return result.transcript, memory, seed, recorder, learned


# --------------------------------------------------------------------- #
# End-to-end: repair-introduced tokens must not be learned
# --------------------------------------------------------------------- #


def test_near_miss_repaired_token_is_not_promoted_at_commit() -> None:
    """Issue #80 repro: `PASTF` on screen rewrites a spoken `paste`.

    The rewrite itself is the near-miss reconciler's business (#68). What must
    not happen is memory learning `PASTF` off the back of it.
    """

    committed, memory, seed, recorder, learned = _run_commit(
        utterance_id="utt-repair-introduced",
        transcript="Click on paste.",
        context=TypedContextBundle(
            app_name="Xcode",
            app_category="code",
            window_title="PASTF panel",
            field_text_excerpt="PASTF",
            candidate_entities=["PASTF"],
        ),
    )

    # Precondition: the reconciler really did introduce the token, so the
    # promotion gate — not an absent rewrite — is what this test measures.
    assert "PASTF" in committed

    assert memory.entities == []
    assert memory.corrections == []
    assert seed.learned_store.observations == []
    assert seed.learned_store.acceptances == []
    assert seed.promotion.context_calls == []
    assert seed.promotion.correction_calls == []
    assert learned["learned"] is False
    assert learned["entity_count"] == 0

    payloads = recorder.payloads("oneshot_memory_updated_from_commit")
    assert payloads and "PASTF" in payloads[-1]["repair_introduced"]


def test_spoken_term_still_promotes_when_another_token_was_repaired() -> None:
    """A repair elsewhere in the utterance must not block honest learning."""

    committed, memory, seed, _recorder, learned = _run_commit(
        utterance_id="utt-mixed-provenance",
        transcript="LumaRay battery risk, then click on paste.",
        context=TypedContextBundle(
            app_name="Xcode",
            app_category="code",
            window_title="PASTF panel",
            field_text_excerpt="LumaRay PASTF",
            candidate_entities=["LumaRay", "PASTF"],
        ),
    )

    assert "PASTF" in committed
    assert "LumaRay" in committed

    assert "LumaRay" in memory.entities
    assert "PASTF" not in memory.entities
    assert seed.learned_store.observations == ["LumaRay"]
    assert seed.learned_store.acceptances == ["LumaRay"]
    assert [call["token"] for call in seed.promotion.context_calls] == ["LumaRay"]
    assert learned["learned"] is True


def test_commit_without_repairs_learns_normally() -> None:
    """No repair provenance means the gate is inert."""

    _committed, memory, seed, _recorder, learned = _run_commit(
        utterance_id="utt-no-repair",
        transcript="LumaRay battery risk is assigned.",
        context=TypedContextBundle(
            app_name="Notes",
            app_category="docs",
            candidate_entities=["LumaRay"],
        ),
    )

    assert "LumaRay" in memory.entities
    assert seed.learned_store.observations == ["LumaRay"]
    assert learned["learned"] is True


# --------------------------------------------------------------------- #
# record_insertion gate, driven directly through a seeded UtteranceRecord
# --------------------------------------------------------------------- #


def _pipeline_with_record(record: UtteranceRecord) -> tuple[
    OneShotDictationPipeline, _FakeMemoryStore, _FakeSeedRuntime
]:
    memory = _FakeMemoryStore()
    seed = _FakeSeedRuntime()
    pipeline = OneShotDictationPipeline(
        transcriber=_FakeTranscriber(""),
        recorder=_Recorder(),
        memory_store=memory,  # type: ignore[arg-type]
        juno_seed_runtime=seed,  # type: ignore[arg-type]
        writer_enabled=False,
    )
    pipeline.records.put(record)
    return pipeline, memory, seed


def test_lexicon_canonicalization_token_is_not_promoted() -> None:
    """Flat `NormalizationChange` provenance (`lexicon_canonical`) also gates."""

    record = UtteranceRecord(
        utterance_id="utt-lexicon-canonical",
        raw_text="Send the notes to noda.",
        normalized_text="Send the notes to NodaSync.",
        adjudicated_text="Send the notes to NodaSync.",
        writer_text="Send the notes to NodaSync.",
        plan=None,
        context=TypedContextBundle(app_name="Mail", app_category="mail"),
        literal_text="Send the notes to NodaSync.",
        raw_transcript="Send the notes to noda.",
        normalization_applied=(
            {
                "kind": "canonicalization",
                "source": "lexicon_canonical",
                "before": "noda",
                "after": "NodaSync",
            },
        ),
    )
    pipeline, memory, seed = _pipeline_with_record(record)

    learned = pipeline.record_insertion(
        utterance_id="utt-lexicon-canonical",
        committed_text="Send the notes to NodaSync.",
    )

    assert memory.entities == []
    assert memory.corrections == []
    assert seed.promotion.correction_calls == []
    assert learned["learned"] is False


def test_repair_target_the_user_also_said_stays_promotable() -> None:
    """A repair that fixes a second, garbled mention is not "introduced"."""

    raw = "NodaSync tracks the noda backlog."
    committed = "NodaSync tracks the NodaSync backlog."
    record = UtteranceRecord(
        utterance_id="utt-second-mention",
        raw_text=raw,
        normalized_text=committed,
        adjudicated_text=committed,
        writer_text=committed,
        plan=None,
        context=TypedContextBundle(app_name="Mail", app_category="mail"),
        literal_text=committed,
        raw_transcript=raw,
        normalization_applied=(
            {
                "rule": "protected_term_near_miss_reconciliation",
                "scope": "oneshot",
                "replacements": [
                    {"from": "noda", "to": "NodaSync", "source": "protected_term_near_miss"}
                ],
            },
        ),
    )
    pipeline, memory, seed = _pipeline_with_record(record)

    learned = pipeline.record_insertion(
        utterance_id="utt-second-mention",
        committed_text=committed,
    )

    assert "NodaSync" in memory.entities
    assert seed.learned_store.observations == ["NodaSync"]
    assert learned["learned"] is True


# --------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------- #


def test_repair_introduced_surfaces_reads_both_payload_shapes() -> None:
    applied = [
        {
            "rule": "protected_term_near_miss_reconciliation",
            "scope": "oneshot",
            "replacements": [
                {"from": "paste", "to": "PASTF", "source": "protected_term_near_miss"}
            ],
        },
        {
            "kind": "canonicalization",
            "source": "lexicon_alias",
            "before": "novadesc",
            "after": "NovaDesk",
        },
    ]

    surfaces = _repair_introduced_surfaces(applied, raw_transcript="Click on paste in novadesc.")

    assert set(surfaces) == {"PASTF", "NovaDesk"}


def test_repair_introduced_surfaces_ignore_non_repair_sources() -> None:
    applied = [
        {
            "kind": "correction",
            "source": "memory_correction",
            "before": "juneau",
            "after": "Juno",
        },
        {
            "rule": "self_correction_retakes",
            "scope": "oneshot",
            "replacements": [{"from": "nine", "to": "ten", "source": "retake"}],
        },
    ]

    assert _repair_introduced_surfaces(applied, raw_transcript="juneau at nine") == ()


def test_repair_introduced_surfaces_exempt_terms_present_in_raw_asr() -> None:
    applied = [
        {
            "rule": "protected_term_near_miss_reconciliation",
            "scope": "oneshot",
            "replacements": [
                {"from": "noda", "to": "NodaSync", "source": "protected_term_near_miss"}
            ],
        }
    ]

    assert (
        _repair_introduced_surfaces(
            applied, raw_transcript="NodaSync tracks the noda backlog."
        )
        == ()
    )


def test_entity_repair_introduced_matches_containing_phrases() -> None:
    assert _entity_repair_introduced("PASTF", ("PASTF",))
    assert _entity_repair_introduced("pastf", ("PASTF",))
    assert _entity_repair_introduced("PASTF Panel", ("PASTF",))
    assert not _entity_repair_introduced("LumaRay", ("PASTF",))
    assert not _entity_repair_introduced("PASTF", ())


def test_commit_rewrite_is_repair_only_requires_full_coverage() -> None:
    assert _commit_rewrite_is_repair_only(
        raw_transcript="Click on paste.",
        committed_text="Click on PASTF.",
        repair_introduced=("PASTF",),
    )
    # A word the user actually added means the pair is a real correction.
    assert not _commit_rewrite_is_repair_only(
        raw_transcript="Click on paste.",
        committed_text="Click on PASTF twice.",
        repair_introduced=("PASTF",),
    )
    assert not _commit_rewrite_is_repair_only(
        raw_transcript="Click on paste.",
        committed_text="Click on paste.",
        repair_introduced=("PASTF",),
    )
    assert not _commit_rewrite_is_repair_only(
        raw_transcript="Click on paste.",
        committed_text="Click on PASTF.",
        repair_introduced=(),
    )
