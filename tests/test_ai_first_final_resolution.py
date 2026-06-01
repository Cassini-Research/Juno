from __future__ import annotations

from dataclasses import replace
import json
import math
import struct
import tempfile
import wave

from juno_core_v3.dictation.pipeline import (
    OneShotDictationPipeline,
    _collect_self_correction_cues,
    _mode_policy_for_final_delivery,
    _reconcile_protected_term_near_misses,
    _reconcile_split_candidate_term,
)
from juno_core_v3.dictation.transcriber import TranscribeResult
from juno_v2.commands.grammar import parse_deterministic_command
from juno_v2.context.compiler import FormattingPacket, TranscriptAdjudicationPacket, compile_context
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import LexiconEntry, MemorySnapshot
from juno_v2.contracts.modes import ModeSelection, ModeSource
from juno_v2.contracts.workbench import ClientSelection, CommitMode
from juno_v2.contracts.writer import WriterActionKind, WriterMode, WriterTransformRequest, WriterTransformResult
from juno_v2.itn.rules import apply_spoken_punctuation
from juno_v2.memory.bias import RecognitionBiasEngine
from juno_v2.memory.ranking import rank_memory_for_context
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.preview.live_agreement import Word
from juno_v2.preview.personalization_repair import repair_preview_word_dicts
from juno_v2.preview.streaming_core import (
    WhisperDecodeOutput,
    _silence_agreement_commit_safe,
)
from juno_v2.transcript.contracts import TranscriptAdjudicationResult
from juno_v2.transcript.adjudicator import TranscriptAdjudicator
from juno_v2.transcript.validators import validate_adjudication_result
from juno_v2.writer.backends.mlx_lm import _build_writer_prompt, _system_prompt
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.deterministic import run_pipeline
from juno_v2.writer.final_formatter import FinalFormatter
from juno_v2.writer.service import WriterService
from juno_v2.workbench.server import _preview_candidates_from_session_context_tape


class _Recorder:
    def __init__(self) -> None:
        self.log_dir = tempfile.mkdtemp(prefix="juno-ai-first-test-")

    def record(self, *args: object, **kwargs: object) -> None:
        return None


def test_preview_repair_keeps_fuzzy_memory_terms_out_of_committed_hud() -> None:
    repaired, meta = repair_preview_word_dicts(
        [{"word": "karo", "start": 0.0, "end": 0.2}],
        context_payload={"candidate_entities": ["Korea"]},
    )

    assert repaired[0]["word"] == "karo"
    assert meta["preview_repair_applied"] == 0


def test_preview_repair_still_canonicalizes_exact_named_terms() -> None:
    repaired, meta = repair_preview_word_dicts(
        [{"word": "silviagamachi", "start": 0.0, "end": 0.2}],
        context_payload={"candidate_entities": ["SilviaGamachi"]},
    )

    assert repaired[0]["word"] == "SilviaGamachi"
    assert meta["preview_repair_applied"] == 1


def test_preview_candidates_include_session_context_tape_screen_terms() -> None:
    candidates = _preview_candidates_from_session_context_tape(
        {
            "snapshots": [
                {
                    "app_name": "Editor",
                    "window_title": "SilviaGamachi launch plan",
                    "selected_text": "SilviaGamachi owns the Project Atlas review.",
                    "focused_text_before": "",
                    "focused_text_after": "",
                }
            ]
        }
    )

    assert "SilviaGamachi" in candidates
    assert "Project Atlas" in candidates


def test_preview_repair_uses_explicit_memory_aliases_without_fuzzy_matching() -> None:
    repaired, meta = repair_preview_word_dicts(
        [{"word": "Niloufar", "start": 0.0, "end": 0.2}],
        context_payload={
            "preview_personalization_terms": [
                {
                    "text": "Nilofar",
                    "source": "seed_memory_lexicon",
                    "aliases": ["Nilofer", "Niloufar"],
                }
            ]
        },
    )

    assert repaired[0]["word"] == "Nilofar"
    assert meta["preview_repair_applied"] == 1


def test_preview_repair_allows_one_edit_only_against_explicit_aliases() -> None:
    repaired, meta = repair_preview_word_dicts(
        [{"word": "Nilefer", "start": 0.0, "end": 0.2}],
        context_payload={
            "preview_personalization_terms": [
                {
                    "text": "Nilofar",
                    "source": "seed_memory_lexicon",
                    "aliases": ["Nilofer", "Niloufar"],
                }
            ]
        },
    )

    assert repaired[0]["word"] == "Nilofar"
    assert meta["preview_repair_applied"] == 1


def test_preview_repair_allows_two_edits_for_long_explicit_aliases() -> None:
    repaired, meta = repair_preview_word_dicts(
        [{"word": "Nilefer", "start": 0.0, "end": 0.2}],
        context_payload={
            "preview_personalization_terms": [
                {
                    "text": "Nilofar",
                    "source": "seed_memory_lexicon",
                    "aliases": ["Nilofer"],
                }
            ]
        },
    )

    assert repaired[0]["word"] == "Nilofar"
    assert meta["preview_repair_applied"] == 1


def test_ranked_memory_packet_carries_aliases_for_preview_repair() -> None:
    packet = rank_memory_for_context(
        MemorySnapshot(
            schema_version=1,
            lexicon=[
                LexiconEntry(
                    term="Nilofar",
                    canonical_form="Nilofar",
                    aliases=["Nilofer", "Niloufar"],
                )
            ],
        ),
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        mode_policy=BUILTIN_MODES["default_surface"],
        effective_mode="default_surface",
        transcript_hint="Niloufar owner review",
    )
    repaired, meta = repair_preview_word_dicts(
        [{"word": "Niloufar", "start": 0.0, "end": 0.2}],
        context_payload={"preview_personalization_terms": [], "memory_serving_packet": packet.to_dict()},
    )

    assert packet.metadata["lexicon_aliases"]["Nilofar"] == ["Nilofer", "Niloufar"]
    assert repaired[0]["word"] == "Nilofar"
    assert meta["preview_repair_applied"] == 1


def test_final_formatting_prompt_carries_preservation_terms_to_qwen() -> None:
    req = WriterTransformRequest(
        utterance_id="utt-format",
        instruction="Apply only the requested commit-time formatting policy.",
        source_text="SilviaGamachi owns Project Atlas.",
        mode=WriterMode.DEFAULT_SURFACE,
        context_payload={
            "task": "final_formatting_v1",
            "policy": "structured_notes",
            "app_name": "Notes",
            "app_category": "docs",
            "window_title": "Project Atlas",
            "mode_name": "default_surface",
            "required_preserved_terms": ["SilviaGamachi", "Project Atlas"],
            "candidate_entities": ["SilviaGamachi"],
            "recent_screen_terms": ["Project Atlas"],
        },
        metadata={"kind": "final_formatting_v1"},
    )

    payload = json.loads(_build_writer_prompt(req))

    assert payload["context"]["required_preserved_terms"] == ["SilviaGamachi", "Project Atlas"]
    assert payload["context"]["candidate_entities"] == ["SilviaGamachi"]
    assert payload["context"]["recent_screen_terms"] == ["Project Atlas"]


def test_final_formatter_required_terms_are_sent_to_backend() -> None:
    captured: dict[str, object] = {}

    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            captured["prompt"] = json.loads(_build_writer_prompt(req))
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text="- SilviaGamachi owns Project Atlas.",
                backend_name="fake-qwen",
            )

    packet = FormattingPacket(
        utterance_id="utt-packet",
        corrected_text="SilviaGamachi owns Project Atlas.",
        app_name="Notes",
        app_category="docs",
        window_title="Project Atlas",
        mode_name="default_surface",
        final_formatting_policy="structured_notes",
        style_card=None,
        focused_text_before="",
        focused_text_after="",
        selected_text_excerpt="",
        writer_tone_addon=None,
        metadata={
            "candidate_entities": ["SilviaGamachi"],
            "recent_screen_terms": ["Project Atlas"],
        },
        mode_prompt_prefix="Structured notes: bullets when spoken.",
    )

    result = FinalFormatter(backend=Backend()).format(packet)

    assert result is not None
    prompt = captured["prompt"]
    assert isinstance(prompt, dict)
    assert "SilviaGamachi" in prompt["context"]["required_preserved_terms"]


def test_transcript_adjudicator_preserves_ai_first_resolution_metadata() -> None:
    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text=json.dumps(
                    {
                        "schema_version": "transcript_adjudication_v1",
                        "corrected_text": "Japan, no actually Korea, is the customer meeting location.",
                        "ops": [],
                        "confidence": 0.91,
                        "protected_terms_used": ["Korea"],
                        "intent": {"kind": "dictation"},
                        "formatting_plan": {"kind": "no_formatting"},
                        "self_corrections": [
                            {"cue": "no actually", "from": "Japan", "to": "Korea"}
                        ],
                        "terms_used": [{"term": "Korea", "source": "screen"}],
                    }
                ),
                backend_name="fake-qwen",
            )

    packet = TranscriptAdjudicationPacket(
        stage="final",
        utterance_id="utt-adj",
        base_visible_text="",
        base_visible_revision=None,
        live_preview_text="",
        whisper_text="Japan, no actually Korea, is the customer meeting location.",
        memory_candidate_text="Japan, no actually Korea, is the customer meeting location.",
        raw_text="Japan, no actually Korea, is the customer meeting location.",
        context_terms=(),
        protected_terms=("Korea",),
        selected_text_excerpt="",
        focused_text_before="",
        focused_text_after="",
        field_text_excerpt="",
        app_name="Notes",
        app_category="docs",
        window_title=None,
        focused_file_path=None,
        symbol_under_cursor=None,
        mode_name="default_surface",
        transcript_policy="standard",
        final_formatting_policy="minimal",
        no_touch=False,
        privacy_suppressed=False,
        language="en",
        metadata={},
    )

    result = TranscriptAdjudicator(backend=Backend(), recorder=_Recorder()).adjudicate(packet)

    assert not result.rejected
    assert result.metadata["intent"] == {"kind": "dictation"}
    assert result.metadata["formatting_plan"] == {"kind": "no_formatting"}
    assert result.metadata["self_corrections"][0]["to"] == "Korea"


def test_transcript_adjudicator_repairs_low_signal_mid_sentence_caps_before_validation() -> None:
    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text=json.dumps(
                    {
                        "schema_version": "transcript_adjudication_v1",
                        "corrected_text": "Do not invent docs. doggs, Thank you, or send.",
                        "ops": [],
                        "confidence": 0.91,
                        "protected_terms_used": [],
                    }
                ),
                backend_name="fake-qwen",
            )

    packet = _adjudication_packet(
        raw_text="Do not invent docs. doggs, Thank you, or send.",
    )

    result = TranscriptAdjudicator(backend=Backend(), recorder=_Recorder()).adjudicate(packet)

    assert not result.rejected
    assert "doggs, thank you" in result.corrected_text
    assert result.metadata["low_signal_capitalization_repairs"] == [{"from": "Thank", "to": "thank"}]


def test_transcript_validation_allows_duplicate_protected_term_cleanup() -> None:
    packet = _adjudication_packet(
        raw_text=(
            "Decision log includes keeping Qwen for final polish. "
            "Decision log includes keeping Qwen for final polish."
        ),
        protected_terms=("Qwen", "Decision log"),
    )
    result = _adjudication_result(
        "Decision log includes keeping Qwen for final polish.",
        protected_terms_used=("Qwen", "Decision log"),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert ok, reason


def test_transcript_validation_rejects_inverted_no_actually_cue() -> None:
    packet = _adjudication_packet(
        raw_text="No actually write LumaRay as one product word.",
        protected_terms=("LumaRay",),
    )
    result = _adjudication_result(
        "Do not write LumaRay as one product word.",
        protected_terms_used=("LumaRay",),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert not ok
    assert reason == "self_correction_cue_inverted:no_actually_write"


def test_transcript_validation_rejects_lost_final_tail_even_with_self_correction() -> None:
    packet = _adjudication_packet(
        raw_text=(
            "First section is risks scratch that make it open risks. "
            "At the end say the final word is complete."
        ),
        protected_terms=("open risks",),
    )
    result = _adjudication_result(
        "First section is open risks.",
        protected_terms_used=("open risks",),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert not ok
    assert reason.startswith("source_words_dropped:")


def test_transcript_validation_allows_no_actually_noun_correction() -> None:
    packet = _adjudication_packet(
        raw_text="Second section is decisions, no actually decision log.",
        protected_terms=("decision log",),
    )
    result = _adjudication_result(
        "Second section is decision log.",
        protected_terms_used=("decision log",),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert ok, reason


def test_transcript_validation_allows_repeated_question_hallucination_drop() -> None:
    packet = _adjudication_packet(
        raw_text="Question Question Decision log includes keeping Cwen useful.",
        protected_terms=("Decision log",),
    )
    result = _adjudication_result(
        "Decision log includes keeping Cwen useful.",
        protected_terms_used=("Decision log",),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert ok, reason


def test_transcript_adjudicator_restores_explicit_final_word_tail() -> None:
    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text=json.dumps(
                    {
                        "schema_version": "transcript_adjudication_v1",
                        "corrected_text": "First section is open risks.",
                        "ops": [],
                        "confidence": 0.91,
                        "protected_terms_used": ["open risks"],
                    }
                ),
                backend_name="fake-qwen",
            )

    packet = _adjudication_packet(
        raw_text="First section is risks scratch that make it open risks. At the end say the final word is complete.",
        protected_terms=("open risks",),
    )

    result = TranscriptAdjudicator(backend=Backend(), recorder=_Recorder()).adjudicate(packet)

    assert not result.rejected
    assert result.corrected_text.endswith("At the end say the final word is complete.")
    assert result.metadata["explicit_final_word_tail_restore"]["restored"] == (
        "At the end say the final word is complete."
    )


def test_transcript_adjudicator_removes_scratch_that_exclusion_instruction() -> None:
    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text=json.dumps(
                    {
                        "schema_version": "transcript_adjudication_v1",
                        "corrected_text": (
                            "Start a planning note. Add bullets under each section, but do not include "
                            "the words scratch that in the final note unless I explicitly say quote scratch that quote. "
                            "Open risks include tail words."
                        ),
                        "ops": [],
                        "confidence": 0.91,
                        "protected_terms_used": ["Open risks"],
                    }
                ),
                backend_name="fake-qwen",
            )

    packet = _adjudication_packet(
        raw_text=(
            "Start a planning note. Add bullets under each section, but do not include "
            "the words scratch that in the final note unless I explicitly say quote scratch that quote. "
            "Open risks include tail words."
        ),
        protected_terms=("Open risks",),
    )

    result = TranscriptAdjudicator(backend=Backend(), recorder=_Recorder()).adjudicate(packet)

    assert not result.rejected
    assert "scratch that" not in result.corrected_text.casefold()
    assert result.metadata["instructional_exclusion_repairs"]


def test_seed_canonicalization_does_not_expand_generic_alias_without_context() -> None:
    engine = RecognitionBiasEngine()

    assert not engine._seed_canonicalization_allowed(  # noqa: SLF001
        "docs",
        "Google Docs",
        context=TypedContextBundle(app_name="Editor", window_title="Debug note"),
    )
    assert engine._seed_canonicalization_allowed(  # noqa: SLF001
        "docs",
        "Google Docs",
        context=TypedContextBundle(app_name="Chrome", window_title="Google Docs - Launch plan"),
    )


def test_post_asr_memory_subterm_does_not_match_plain_prefix_word() -> None:
    compiled = compile_context(
        utterance_id="utt-open-risks",
        context=TypedContextBundle(app_name="Obsidian", app_category="docs"),
        memory_snapshot=MemorySnapshot(
            schema_version=1,
            lexicon=[LexiconEntry(term="OpenAI", canonical_form="OpenAI")],
        ),
        mode_selection=ModeSelection(
            effective_mode="default_surface",
            mode_source=ModeSource.AUTO,
            manual_mode_name=None,
            custom_mode_name=None,
            resolved_from_surface=None,
        ),
        mode_policy=BUILTIN_MODES["default_surface"],
        transcript_hint=None,
        final_transcript_text="Open risks include final transformation latency.",
        session_terms=None,
        language="en",
        stage="final",
    )

    hinted_terms = [
        term for term in compiled.terms if term.metadata.get("source") == "hint_matched_memory_subterm"
    ]

    assert hinted_terms == []


def test_spoken_structure_promotes_default_docs_to_structured_notes() -> None:
    policy = BUILTIN_MODES["default_surface"]
    context = TypedContextBundle(app_name="Notes", app_category="docs")

    promoted = _mode_policy_for_final_delivery(
        policy,
        context=context,
        raw_text="Start research notes. We need four sections. First battery risk. Second rollout. Third launch metric.",
        adjudicated_text="Start research notes. We need four sections. First battery risk. Second rollout. Third launch metric.",
        adjudication_result=None,
    )

    assert promoted.final_formatting_policy == "structured_notes"


def test_spoken_bullet_points_strip_ordinal_item_labels() -> None:
    rendered = run_pipeline(
        (
            "Create three bullets. First point is Passport. Second point is Charger. "
            "Third point is Korea, customer meeting location."
        ),
        app_category="docs",
    )

    assert rendered.strip() == (
        "Create three bullets:\n"
        "1. Passport.\n"
        "2. Charger.\n"
        "3. Korea, customer meeting location."
    )


def test_unpunctuated_spoken_bullets_render_from_real_audio_shape() -> None:
    rendered = run_pipeline(
        (
            "Create 3 bullets first point is Passport Second point is Charger "
            "Third point is Korea customer meeting location"
        ),
        app_category="docs",
    )

    assert rendered.strip() == (
        "Create 3 bullets:\n"
        "1. Passport.\n"
        "2. Charger.\n"
        "3. Korea customer meeting location."
    )


def test_explicit_rewrite_promotes_default_docs_without_selection_transform() -> None:
    policy = BUILTIN_MODES["default_surface"]
    context = TypedContextBundle(app_name="Notes", app_category="docs")

    promoted = _mode_policy_for_final_delivery(
        policy,
        context=context,
        raw_text="Write this as a concise status update. The launch is on track and the risk owner is Nilofar.",
        adjudicated_text="Write this as a concise status update. The launch is on track and the risk owner is Nilofar.",
        adjudication_result=None,
    )

    assert promoted.final_formatting_policy == "explicit_rewrite"


def test_spoken_structure_does_not_promote_terminal_or_explicit_no_formatting() -> None:
    policy = BUILTIN_MODES["default_surface"]

    class NoBulletsPlan:
        rejected = False
        metadata = {"formatting_plan": {"kind": "no_bullets", "instruction": "do not use bullet list"}}

    terminal = _mode_policy_for_final_delivery(
        policy,
        context=TypedContextBundle(app_name="Terminal", app_category="terminal"),
        raw_text="Do not turn this into bullets. First run pytest. Second run git diff.",
        adjudicated_text="Do not turn this into bullets. First run pytest. Second run git diff.",
        adjudication_result=None,
    )
    docs_no_format = _mode_policy_for_final_delivery(
        policy,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        raw_text="Do not turn this into bullets. First run pytest. Second run git diff.",
        adjudicated_text="Do not turn this into bullets. First run pytest. Second run git diff.",
        adjudication_result=None,
    )
    qwen_no_bullets = _mode_policy_for_final_delivery(
        policy,
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        raw_text="Keep this as a paragraph. First the launch note. Second the risk.",
        adjudicated_text="Keep this as a paragraph. First the launch note. Second the risk.",
        adjudication_result=NoBulletsPlan(),
    )

    assert terminal.final_formatting_policy == "minimal"
    assert docs_no_format.final_formatting_policy == "minimal"
    assert qwen_no_bullets.final_formatting_policy == "minimal"


def test_spoken_punctuation_preserves_literal_mentions_but_converts_commands() -> None:
    literal, literal_rules = apply_spoken_punctuation(
        "Not every pause is a full stop and the words full stop should stay as text."
    )
    converted, converted_rules = apply_spoken_punctuation("Hello comma world full stop")

    assert literal == "Not every pause is a full stop and the words full stop should stay as text."
    assert literal_rules == []
    assert converted == "Hello, world."
    assert converted_rules == ["spoken_punctuation"]


def test_self_correction_cues_are_evidence_not_text_rewrites() -> None:
    text = (
        "Blank space means the words blank space, and scratch that should remove "
        "the previous phrase when it is a command."
    )

    cues = _collect_self_correction_cues(text)

    assert cues
    assert "Blank space means the words blank space" in text
    assert cues[0]["marker"].casefold() == "scratch that"


def test_split_candidate_repair_does_not_absorb_function_words() -> None:
    unchanged, unchanged_replacements = _reconcile_split_candidate_term(
        "Project Atlas is on track.",
        "Atlas",
    )
    repaired, replacements = _reconcile_split_candidate_term(
        "First section is Luma Ray battery risk.",
        "LumaRay",
    )

    assert unchanged == "Project Atlas is on track."
    assert unchanged_replacements == []
    assert repaired == "First section is LumaRay battery risk."
    assert replacements == [{"from": "Luma Ray", "to": "LumaRay", "source": "explicit_candidate_split_phrase"}]


def test_protected_phrase_token_repair_handles_screen_name_near_miss() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="The visible screen says Sylvia Gamache.",
        protected_terms=("Silvia Gamache",),
    )

    assert repaired == "The visible screen says Silvia Gamache."
    assert replacements == [{"from": "Sylvia", "to": "Silvia", "source": "protected_term_near_miss"}]


def test_oneshot_salvages_qwen_self_correction_after_protected_phrase_repair() -> None:
    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript=(
                    "Start a note. The visible screen says Sylvia Gamache. "
                    "Japan, no actually Korea, is the customer meeting location."
                ),
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(
                app_name="Notes",
                app_category="docs",
                window_title="Silvia Gamache",
                candidate_entities=["Silvia Gamache"],
            )

    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text=json.dumps(
                    {
                        "schema_version": "transcript_adjudication_v1",
                        "corrected_text": (
                            "Start a note. The visible screen says Sylvia Gamache. "
                            "Korea is the customer meeting location."
                        ),
                        "ops": [],
                        "confidence": 0.91,
                        "protected_terms_used": ["Silvia Gamache"],
                        "self_corrections": [
                            {"cue": "no actually", "from": "Japan", "to": "Korea"}
                        ],
                    }
                ),
                backend_name="fake-qwen",
            )

    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=_Recorder(),
        context_provider=FakeContextProvider(),
        writer_service=WriterService(
            config=WriterConfig(enable_model_transforms=True),
            recorder=_Recorder(),
            backend=Backend(),
        ),
        writer_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-protected-salvage",
        window_title_hint="Silvia Gamache",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.transcript == (
        "Start a note. The visible screen says Silvia Gamache. "
        "Korea is the customer meeting location."
    )
    assert "Japan, no actually Korea" not in result.transcript
    assert any(item["rule"] == "adjudication_validation_repair" for item in result.normalization_applied)


def test_silence_agreement_allows_safe_final_word_not_generic_garbage() -> None:
    decoded = WhisperDecodeOutput(
        text="complete",
        language="en",
        decode_ms=1.0,
        word_dicts=[],
        segment_ends=[],
        metadata={"last_segment_no_speech_prob": 0.05},
    )

    assert _silence_agreement_commit_safe(
        [Word(start=0.1, end=0.3, text="complete")],
        decoded=decoded,
        previous_tail_reason="tail_silence_decode_quarantine",
        decode_on_silence=False,
    )
    assert not _silence_agreement_commit_safe(
        [Word(start=0.1, end=0.3, text="hello")],
        decoded=decoded,
        previous_tail_reason="tail_silence_decode_quarantine",
        decode_on_silence=False,
    )
    risky = WhisperDecodeOutput(
        text="complete",
        language="en",
        decode_ms=1.0,
        word_dicts=[],
        segment_ends=[],
        metadata={"last_segment_no_speech_prob": 0.55},
    )
    assert not _silence_agreement_commit_safe(
        [Word(start=0.1, end=0.3, text="complete")],
        decoded=risky,
        previous_tail_reason="tail_silence_decode_quarantine",
        decode_on_silence=False,
    )


def test_post_asr_context_enrichment_is_visible_in_qwen_packet() -> None:
    compiled = compile_context(
        utterance_id="utt-context",
        context=TypedContextBundle(app_name="Notes", app_category="docs"),
        memory_snapshot=MemorySnapshot(
            schema_version=1,
            lexicon=[LexiconEntry(term="Silvia Gamache", canonical_form="SilviaGamachi")],
        ),
        mode_selection=ModeSelection(
            effective_mode="default_surface",
            mode_source=ModeSource.AUTO,
            manual_mode_name=None,
            custom_mode_name=None,
            resolved_from_surface=None,
        ),
        mode_policy=BUILTIN_MODES["default_surface"],
        transcript_hint=None,
        final_transcript_text="The screen says Silvia Gamache next to Project Atlas.",
        session_terms=None,
        language="en",
        stage="final",
    )

    packet = compiled.transcript_packet(
        stage="final",
        whisper_text="The screen says Silvia Gamache next to Project Atlas.",
        memory_candidate_text="The screen says SilviaGamachi next to Project Atlas.",
        raw_text="The screen says Silvia Gamache next to Project Atlas.",
    )
    payload = packet.to_payload()

    assert payload["diagnostics"]["post_asr_context_enriched"] is True
    assert payload["speech_resolution_contract"]["role"].startswith("Resolve what the user meant")


def test_pipeline_passes_current_time_into_action_detection(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript="Hey Juno remind me to call Sam tomorrow at five.",
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    def fake_detect_actions_for_pipeline(**kwargs: object) -> None:
        captured["now"] = kwargs.get("now")
        packet = kwargs.get("context_packet")
        captured["now_iso"] = getattr(packet, "now_iso", None)
        return None

    monkeypatch.setattr(
        "juno_core_v3.dictation.pipeline.detect_actions_for_pipeline",
        fake_detect_actions_for_pipeline,
    )

    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=_Recorder(),
        writer_enabled=False,
    )
    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-action-now",
        save_history=False,
        save_audio=False,
    )

    assert result.is_action is True
    assert captured["now"] is not None
    assert isinstance(captured["now_iso"], str)
    assert "T" in captured["now_iso"]


def test_seed_memory_observes_context_terms_only_after_successful_commit() -> None:
    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript="LumaRay battery risk is assigned.",
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(
                app_name="Notes",
                app_category="docs",
                candidate_entities=["LumaRay", "Okay"],
            )

    class FakeMemoryStore:
        def __init__(self) -> None:
            self.entities: list[str] = []

        def snapshot(self) -> MemorySnapshot:
            return MemorySnapshot(schema_version=1)

        def record_correction(self, raw: str, committed: str) -> bool:
            return False

        def upsert_session_entities(self, entities: list[str], source: str) -> None:
            self.entities.extend(entities)

    class FakeLearnedStore:
        def __init__(self) -> None:
            self.observations: list[str] = []
            self.acceptances: list[str] = []

        def increment_observation(self, token: str, *, from_suppressed_context: bool) -> None:
            self.observations.append(token)

        def increment_acceptance(self, token: str, *, from_suppressed_context: bool) -> None:
            self.acceptances.append(token)

    class FakePromotion:
        def maybe_promote_correction_to_lexicon(self, **kwargs: object) -> dict[str, object]:
            return {}

        def maybe_promote_context_entity_to_lexicon(self, **kwargs: object) -> dict[str, object]:
            return {}

    class FakeSeedRuntime:
        def __init__(self) -> None:
            self.observe_calls = 0
            self.learned_store = FakeLearnedStore()
            self.promotion = FakePromotion()

        def build_seed_attachment(self, **kwargs: object) -> None:
            return None

        def observe_transcript_for_context_entities(self, *args: object, **kwargs: object) -> None:
            self.observe_calls += 1

        def context_plane_suppression_value(self, metadata: dict[str, object]) -> None:
            return None

        def durable_memory_suppressed(self, *args: object, **kwargs: object) -> bool:
            return False

    memory = FakeMemoryStore()
    seed = FakeSeedRuntime()
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=_Recorder(),
        context_provider=FakeContextProvider(),
        memory_store=memory,  # type: ignore[arg-type]
        juno_seed_runtime=seed,  # type: ignore[arg-type]
        writer_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-seed-after-commit",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert seed.observe_calls == 0
    assert seed.learned_store.observations == []

    learned = pipeline.record_insertion(
        utterance_id="utt-seed-after-commit",
        committed_text="LumaRay battery risk is assigned.",
    )

    assert learned["learned"] is True
    assert "LumaRay" in memory.entities
    assert "Okay" not in memory.entities
    assert seed.learned_store.observations == ["LumaRay"]
    assert seed.learned_store.acceptances == ["LumaRay"]


def test_voice_action_prompt_includes_now_iso() -> None:
    req = WriterTransformRequest(
        utterance_id="utt-action-prompt",
        instruction="Extract Juno voice actions from the transcript.",
        source_text="remind me to call Sam tomorrow at five",
        mode=WriterMode.DEFAULT_SURFACE,
        context_payload={
            "task": "voice_action_extraction",
            "schema_version": "actions_intent_v3",
            "now_iso": "2026-06-01T09:30:00+07:00",
            "allowed_action_kinds": ["note", "reminder", "alarm"],
        },
        metadata={"feature": "voice_actions_llm_fallback"},
    )

    prompt = _build_writer_prompt(req)

    assert "Current local time (now_iso): 2026-06-01T09:30:00+07:00" in prompt


def test_transcript_adjudication_prompt_includes_slot_correction_example() -> None:
    req = WriterTransformRequest(
        utterance_id="utt-adj-prompt",
        instruction="Resolve final transcript.",
        source_text="Japan, no actually Korea, is the customer meeting location.",
        mode=WriterMode.DEFAULT_SURFACE,
        context_payload={"task": "transcript_adjudication_v1"},
    )

    prompt = _system_prompt(req)

    assert "'Japan, no actually Korea, is the customer meeting location'" in prompt
    assert "'Korea is the customer meeting location.'" in prompt


def test_selected_transform_command_does_not_fall_into_final_formatting() -> None:
    captured: dict[str, object] = {}

    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            captured["request"] = req
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text="Shorter selected text.",
                backend_name="fake-qwen",
            )

    selected = "This is a long selected paragraph that should be rewritten."
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True),
        recorder=_Recorder(),
        backend=Backend(),
    )
    mode_policy = replace(BUILTIN_MODES["default_surface"], final_formatting_policy="structured_notes")

    result = service.process_transcript(
        utterance_id="utt-transform",
        final_text="make that shorter",
        raw_text="make that shorter",
        context=TypedContextBundle(
            selected_text=selected,
            app_name="Notes",
            app_category="docs",
        ),
        anchor_selection=ClientSelection(start=0, end=len(selected)),
        memory_store=None,
        memory_snapshot=None,
        memory_packet={},
        mode_policy=mode_policy,
        mode_selection=ModeSelection(
            effective_mode="default_surface",
            mode_source=ModeSource.AUTO,
            manual_mode_name=None,
            custom_mode_name=None,
            resolved_from_surface=None,
        ),
        partial_text="",
    )

    req = captured["request"]
    assert isinstance(req, WriterTransformRequest)
    assert req.context_payload.get("task") != "final_formatting_v1"
    assert req.source_text == selected
    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.commit_mode == CommitMode.REPLACE_SELECTION
    assert result.output_text == "Shorter selected text."


def test_recent_transform_command_uses_recent_clipboard_in_default_mode() -> None:
    captured: dict[str, object] = {}

    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            captured["request"] = req
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text="Clearer recent text.",
                backend_name="fake-qwen",
            )

    recent = "This is the recent Juno paste that should be rewritten."
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True),
        recorder=_Recorder(),
        backend=Backend(),
    )

    result = service.process_transcript(
        utterance_id="utt-recent-transform",
        final_text="make that text more clear",
        raw_text="make that text more clear",
        context=TypedContextBundle(
            app_name="Editor",
            app_category="unknown",
            recent_clipboard=[{"text": recent, "ts_unix_ms": 1, "redacted": False}],
        ),
        anchor_selection=None,
        memory_store=None,
        memory_snapshot=None,
        memory_packet={},
        mode_policy=BUILTIN_MODES["default_surface"],
        mode_selection=ModeSelection(
            effective_mode="default_surface",
            mode_source=ModeSource.AUTO,
            manual_mode_name=None,
            custom_mode_name=None,
            resolved_from_surface=None,
        ),
        partial_text="",
    )

    req = captured["request"]
    assert isinstance(req, WriterTransformRequest)
    assert req.source_text == recent
    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.commit_mode == CommitMode.REPLACE_SELECTION
    assert result.metadata["target"] == "recent_clipboard"
    assert result.metadata["target_text_chars"] == len(recent)
    assert result.output_text == "Clearer recent text."


def test_recent_transform_command_grammar_covers_natural_variants() -> None:
    clearer = parse_deterministic_command("make that text more clear")
    shorter = parse_deterministic_command("make that shorter and more direct")

    assert clearer is not None
    assert clearer.kind == "recent_edit"
    assert clearer.payload["instruction"] == "Improve clarity. Preserve meaning."
    assert shorter is not None
    assert shorter.kind == "recent_edit"
    assert shorter.payload["instruction"] == "Make the text more concise and direct. Preserve meaning."


def _loud_wav_bytes() -> bytes:
    import io

    sample_rate = 16_000
    frames = bytearray()
    for i in range(sample_rate):
        value = int(9000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
        frames.extend(struct.pack("<h", value))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return buf.getvalue()


def _adjudication_packet(
    *,
    raw_text: str,
    protected_terms: tuple[str, ...] = (),
) -> TranscriptAdjudicationPacket:
    return TranscriptAdjudicationPacket(
        stage="final",
        utterance_id="utt-validate",
        base_visible_text="",
        base_visible_revision=None,
        live_preview_text="",
        whisper_text=raw_text,
        memory_candidate_text=raw_text,
        raw_text=raw_text,
        context_terms=(),
        protected_terms=protected_terms,
        selected_text_excerpt="",
        focused_text_before="",
        focused_text_after="",
        field_text_excerpt="",
        app_name="Notes",
        app_category="docs",
        window_title=None,
        focused_file_path=None,
        symbol_under_cursor=None,
        mode_name="default_surface",
        transcript_policy="standard",
        final_formatting_policy="minimal",
        no_touch=False,
        privacy_suppressed=False,
        language="en",
        metadata={},
    )


def _adjudication_result(
    text: str,
    *,
    protected_terms_used: tuple[str, ...] = (),
) -> TranscriptAdjudicationResult:
    return TranscriptAdjudicationResult(
        utterance_id="utt-validate",
        stage="final",
        corrected_text=text,
        ops=(),
        confidence=0.98,
        base_visible_revision=None,
        base_text_hash=None,
        stable_prefix_chars=None,
        protected_terms_used=protected_terms_used,
    )
