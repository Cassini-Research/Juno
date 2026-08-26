from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import math
import struct
import tempfile
import wave

from juno_core_v3.dictation.pipeline import (
    OneShotDictationPipeline,
    _collect_self_correction_cues,
    _final_adjudication_fast_skip_reason,
    _mode_policy_for_final_delivery,
    _reconcile_explicit_candidate_term_confusions,
    _reconcile_proper_nouns_from_live_hint,
    _reconcile_protected_term_near_misses,
    _reconcile_split_candidate_term,
    _repair_terms_after_candidate_reconciliation,
    _unsafe_writer_surface_reason,
)
from juno_core_v3.dictation.transcriber import FinalBackendTranscriber, TranscribeResult
from juno_core_v3.actions.llm_extractor import validate_envelope
from juno_v2.commands.grammar import parse_deterministic_command
from juno_v2.context.compiler import FormattingPacket, TranscriptAdjudicationPacket, compile_context
from juno_v2.context.frozen_merge import merge_frozen_capability_into_bundle
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.final import FinalDecodeResult
from juno_v2.contracts.memory import LexiconEntry, MemorySnapshot, SessionEntity
from juno_v2.contracts.modes import ModeSelection, ModeSource
from juno_v2.contracts.workbench import ClientSelection, CommitMode
from juno_v2.contracts.writer import WriterActionKind, WriterIntentKind, WriterMode, WriterOutcome, WriterTransformRequest, WriterTransformResult
from juno_v2.final.config import FinalAsrConfig
from juno_v2.itn.rules import apply_spoken_punctuation
from juno_v2.memory.bias import RecognitionBiasEngine
from juno_v2.memory.ai_dictionary import AI_GLOSSARY
from juno_v2.memory.entity_policy import commit_session_entity_allowed, session_entity_allowed
from juno_v2.memory.ranking import rank_memory_for_context
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.memory.stores.corrections import is_safe_correction_pair
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.preview.live_agreement import Word
from juno_v2.preview.personalization_repair import repair_preview_word_dicts
from juno_v2.preview.streaming_core import (
    WhisperDecodeOutput,
    _silence_agreement_commit_safe,
)
from juno_v2.transcript.contracts import TranscriptAdjudicationResult
from juno_v2.transcript.adjudicator import TranscriptAdjudicator, TranscriptAdjudicatorConfig
from juno_v2.transcript.validators import validate_adjudication_result
from juno_v2.writer.backends.mlx_lm import _build_writer_prompt, _system_prompt
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.final_formatter import FinalFormatter
from juno_v2.writer.parser import WriterIntentParser
from juno_v2.writer.service import WriterService
from juno_v2.workbench.server import (
    _action_preview_display_text,
    _merge_action_preview_display_text,
    _preview_candidates_from_session_context_tape,
)
from juno_v2.context.provider import _extract_candidates


class _Recorder:
    def __init__(self) -> None:
        self.log_dir = tempfile.mkdtemp(prefix="juno-ai-first-test-")

    def record(self, *args: object, **kwargs: object) -> None:
        return None


def test_final_paste_guard_rejects_short_list_that_drops_opening_text() -> None:
    source = (
        "This matters. There are two things. First protect the opening text. "
        "Second keep the list safe."
    )
    surface = "- protect the opening text\n- keep the list safe"
    outcome = WriterOutcome(
        utterance_id="utt-list-loss-guard",
        action=WriterActionKind.PASS_THROUGH_COMMIT,
        output_text=surface,
        metadata={"structure": "bullets"},
    )

    assert _unsafe_writer_surface_reason(
        surface,
        fallback_text=source,
        raw_text=source,
        writer_outcome=outcome,
    ) == "list_content_omitted"


def test_oneshot_pipeline_restores_list_content_dropped_by_writer() -> None:
    source = (
        "This matters. There are two things. First protect the opening text. "
        "Second keep the list safe."
    )
    surface = "- protect the opening text\n- keep the list safe"

    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript=source,
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeWriter:
        config = WriterConfig(
            enable_model_transforms=False,
            enable_turn_planner=False,
            dictation_editor_enabled=False,
        )

        def plan_turn(self, **kwargs: object) -> None:
            return None

        def process_transcript(self, **kwargs: object) -> WriterOutcome:
            return WriterOutcome(
                utterance_id="utt-list-loss-pipeline",
                action=WriterActionKind.PASS_THROUGH_COMMIT,
                output_text=surface,
                metadata={"structure": "bullets"},
            )

    class RecordingRecorder(_Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[tuple[str, dict[str, object]]] = []

        def record(self, *args: object, **kwargs: object) -> None:
            if len(args) >= 3 and isinstance(args[1], str) and isinstance(args[2], dict):
                self.events.append((args[1], args[2]))

    recorder = RecordingRecorder()
    pipeline = OneShotDictationPipeline(
        transcriber=FakeTranscriber(),
        recorder=recorder,
        writer_service=FakeWriter(),  # type: ignore[arg-type]
        transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
        itn_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-list-loss-pipeline",
        save_history=False,
        save_audio=False,
    )

    assert result.ok
    assert result.transcript == source
    assert any(
        name == "oneshot_writer_surface_fallback"
        and payload.get("reason") == "list_content_omitted"
        for name, payload in recorder.events
    )


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


def test_preview_repair_allows_explicit_screen_phrase_single_token_confusion() -> None:
    repaired, meta = repair_preview_word_dicts(
        [
            {"word": "off", "start": 0.0, "end": 0.1},
            {"word": "callback", "start": 0.1, "end": 0.2},
            {"word": "flow", "start": 0.2, "end": 0.3},
        ],
        context_payload={"candidate_entities": ["auth callback flow"]},
    )

    assert [w["word"] for w in repaired] == ["auth", "callback", "flow"]
    assert meta["preview_repair_applied"] == 1


def test_preview_repair_does_not_rewrite_single_common_word_to_screen_term() -> None:
    repaired, meta = repair_preview_word_dicts(
        [{"word": "dogs", "start": 0.0, "end": 0.2}],
        context_payload={"candidate_entities": ["docs"]},
    )

    assert repaired[0]["word"] == "dogs"
    assert meta["preview_repair_applied"] == 0


def test_screen_candidate_extractor_filters_editor_chrome_terms() -> None:
    candidates = _extract_candidates(
        [
            (
                "Untitled paragraph style rgb 0 0 0 text colour Helvetica "
                "typeface Regular style 48 font size Edited document TextEdit "
                "Visible page mentions SilviaGamachi and Project Atlas."
            )
        ]
    )

    assert "SilviaGamachi" in candidates
    assert "Project" in candidates
    assert "Atlas" in candidates
    assert "Helvetica" not in candidates
    assert "Regular" not in candidates
    assert "Edited" not in candidates
    assert "TextEdit" not in candidates


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


def test_final_transcript_packet_keeps_screen_terms_but_filters_unrelated_memory() -> None:
    compiled = compile_context(
        utterance_id="utt-context-filter",
        context=TypedContextBundle(
            app_name="Chrome",
            app_category="email",
            window_title="Gmail - Compose",
            candidate_entities=["auth callback flow", "Jordan", "Chrome", "Gmail"],
            metadata={"explicit_candidate_entities": ["auth callback flow", "Jordan", "Chrome", "Gmail"]},
        ),
        memory_snapshot=MemorySnapshot(
            schema_version=1,
            lexicon=[
                LexiconEntry(term="Luma Ray", canonical_form="LumaRay", aliases=["Luma Ray"]),
                LexiconEntry(term="Discord", canonical_form="Discord"),
                LexiconEntry(term="BitsAndBytes", canonical_form="BitsAndBytes"),
            ],
        ),
        mode_selection=ModeSelection(
            effective_mode="default_surface",
            mode_source=ModeSource.AUTO,
            manual_mode_name=None,
            custom_mode_name=None,
            resolved_from_surface=None,
        ),
        mode_policy=BUILTIN_MODES["default_surface"],
        transcript_hint="",
        session_terms=[],
        language="en",
        stage="final_delivery",
        final_transcript_text="Ask the CLI to review the off callback flow for Luma Ray.",
    )
    packet = compiled.transcript_packet(
        stage="final",
        whisper_text="Ask the CLI to review the off callback flow for Luma Ray.",
        memory_candidate_text="Ask the CLI to review the off callback flow for LumaRay.",
        raw_text="Ask the CLI to review the off callback flow for Luma Ray.",
    )
    terms = {term.text for term in packet.context_terms}

    assert "auth callback flow" in terms
    assert "LumaRay" in terms
    assert "Discord" not in terms
    assert "BitsAndBytes" not in terms
    assert "LumaRay" in packet.protected_terms
    assert "Discord" not in packet.protected_terms
    assert "Chrome" not in packet.protected_terms


def test_ocr_junk_does_not_become_final_context_terms() -> None:
    compiled = compile_context(
        utterance_id="utt-ocr-junk",
        context=TypedContextBundle(
            app_name="NovaDesk",
            app_category="docs",
            window_title="NovaDesk",
            candidate_entities=["onboardin9", "Acces5ibility", "Rerninders", "JuDo", "NovaDesk"],
        ),
        memory_snapshot=MemorySnapshot(schema_version=1),
        mode_selection=ModeSelection(
            effective_mode="default_surface",
            mode_source=ModeSource.AUTO,
            manual_mode_name=None,
            custom_mode_name=None,
            resolved_from_surface=None,
        ),
        mode_policy=BUILTIN_MODES["default_surface"],
        transcript_hint="",
        session_terms=[],
        language="en",
        stage="final_delivery",
        final_transcript_text="I checked the onboarding flow and final paste.",
    )
    packet = compiled.transcript_packet(
        stage="final",
        whisper_text="I checked the onboarding flow and final paste.",
        memory_candidate_text="I checked the onboarding flow and final paste.",
        raw_text="I checked the onboarding flow and final paste.",
    )
    terms = {term.text for term in packet.context_terms}

    assert "NovaDesk" in terms
    for junk in ("onboardin9", "Acces5ibility", "Rerninders", "JuDo"):
        assert junk not in terms
        assert junk not in packet.protected_terms


def test_action_preview_display_collapses_wake_gated_action_fragments_only() -> None:
    display = _action_preview_display_text(
        "Hey Juno, take a note. Jordan wants the proposal by Friday morning.",
        "Remind me tomorrow at 9 to send the draft. Set an alarm for 3pm.",
    )
    normal = _action_preview_display_text(
        "Please take a note that Jordan wants the proposal.",
        "",
    )

    assert display == "Hey Juno: note, reminder, alarm"
    assert normal is None


def test_action_preview_display_merge_does_not_shrink_labels() -> None:
    assert _merge_action_preview_display_text(
        "Hey Juno: note, reminder",
        "Hey Juno: note",
    ) == "Hey Juno: note, reminder"
    assert _merge_action_preview_display_text(
        "Hey Juno: note",
        "Hey Juno: note, alarm",
    ) == "Hey Juno: note, alarm"


def test_preview_candidates_include_visible_field_excerpt_terms() -> None:
    candidates = _preview_candidates_from_session_context_tape(
        {
            "snapshots": [
                {
                    "app_name": "Editor",
                    "window_title": "Editor",
                    "selected_text": "",
                    "focused_text_before": "",
                    "focused_text_after": "",
                    "field_text_excerpt": "Visible page mentions Nilofar and SilviaGamachi for Project Atlas.",
                    "candidate_entities": ["Nilofar", "SilviaGamachi", "Project Atlas"],
                }
            ]
        }
    )

    assert "Nilofar" in candidates
    assert "SilviaGamachi" in candidates
    assert "Project Atlas" in candidates


def test_frozen_context_merges_visible_field_excerpt_terms() -> None:
    context = TypedContextBundle(app_name="Editor", window_title="Editor")

    changed = merge_frozen_capability_into_bundle(
        context,
        {
            "field_text_excerpt": "Visible page mentions Nilofar and SilviaGamachi.",
            "candidate_entities": ["Nilofar", "SilviaGamachi"],
        },
    )

    assert changed is True
    assert "Nilofar" in context.field_text_excerpt
    assert context.candidate_entities == ["Nilofar", "SilviaGamachi"]
    assert "explicit_candidate_entities" not in context.metadata


def test_frozen_context_filters_screen_ocr_junk_candidates() -> None:
    context = TypedContextBundle(app_name="NovaDesk", window_title="NovaDesk")

    changed = merge_frozen_capability_into_bundle(
        context,
        {
            "field_text_excerpt": "Visible page mentions NovaDesk and Cassini Research.",
            "candidate_entities": [
                "NovaDesk",
                "Cassini Research",
                "NOvaD",
                "SettiThJs",
                "CityXyoTer",
                "Acces5ibility",
                "bhlS.py",
                "atlons/Junty.app",
            ],
        },
    )

    assert changed is True
    assert context.candidate_entities == ["NovaDesk", "Cassini Research"]


def test_frozen_context_keeps_explicit_repair_terms_separate_from_screen_candidates() -> None:
    context = TypedContextBundle(app_name="Editor", window_title="Editor")

    changed = merge_frozen_capability_into_bundle(
        context,
        {
            "candidate_entities": ["Maia", "Nilofar"],
            "explicit_candidate_entities": ["Nilofar"],
        },
    )

    assert changed is True
    assert context.candidate_entities == ["Maia", "Nilofar"]
    assert context.metadata["explicit_candidate_entities"] == ["Nilofar"]


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

    assert payload["reference_only_context"]["required_preserved_terms"] == ["SilviaGamachi", "Project Atlas"]
    assert payload["reference_only_context"]["candidate_entities"] == ["SilviaGamachi"]
    assert payload["reference_only_context"]["recent_screen_terms"] == ["Project Atlas"]


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
    assert "SilviaGamachi" in prompt["reference_only_context"]["required_preserved_terms"]


def test_final_formatter_filters_ocr_junk_required_terms() -> None:
    captured: dict[str, object] = {}

    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            captured["prompt"] = json.loads(_build_writer_prompt(req))
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text="- NovaDesk final paste.",
                backend_name="fake-qwen",
            )

    packet = FormattingPacket(
        utterance_id="utt-packet-ocr",
        corrected_text="NovaDesk final paste.",
        app_name="NovaDesk",
        app_category="docs",
        window_title="NovaDesk",
        mode_name="structured_notes",
        final_formatting_policy="structured_notes",
        style_card=None,
        focused_text_before="",
        focused_text_after="",
        selected_text_excerpt="",
        writer_tone_addon=None,
        metadata={
            "candidate_entities": ["NovaDesk", "onboardin9", "Acces5ibility"],
            "recent_screen_terms": ["Rerninders", "JuDo"],
        },
        mode_prompt_prefix="",
    )

    result = FinalFormatter(backend=Backend()).format(packet)

    assert result is not None
    prompt = captured["prompt"]
    assert isinstance(prompt, dict)
    required = set(prompt["reference_only_context"]["required_preserved_terms"])
    assert "NovaDesk" in required
    for junk in ("onboardin9", "Acces5ibility", "Rerninders", "JuDo"):
        assert junk not in required


def test_final_formatter_rejects_context_only_metadata_leak() -> None:
    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text=(
                    "- Task: Review the off-callback flow\n"
                    "- Focus: Identify edge cases around expired sessions\n"
                    "- Document title: Untitled\n"
                    "- Font: Helvetica\n"
                    "- Status: Edited\n"
                    "- App: TextEdit"
                ),
                backend_name="fake-qwen",
            )

    packet = FormattingPacket(
        utterance_id="utt-leak",
        corrected_text="Review the off-callback flow and find edge cases around expired sessions.",
        app_name="TextEdit",
        app_category="docs",
        window_title="Untitled",
        mode_name="structured_notes",
        final_formatting_policy="structured_notes",
        style_card=None,
        focused_text_before="",
        focused_text_after="",
        selected_text_excerpt="",
        writer_tone_addon=None,
        metadata={"candidate_entities": ["Untitled", "Helvetica", "Regular", "Edited", "TextEdit"]},
    )

    formatter = FinalFormatter(backend=Backend())

    assert formatter.format(packet) is None
    assert formatter.last_rejection is not None
    assert formatter.last_rejection["reason"] == "context_terms_added"


def test_final_formatter_rejects_dropping_source_required_terms() -> None:
    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text="- Product name should be written in this sentence.",
                backend_name="fake-qwen",
            )

    packet = FormattingPacket(
        utterance_id="utt-source-term-drop",
        corrected_text="Juno should write the product name in this sentence.",
        app_name="Notes",
        app_category="docs",
        window_title=None,
        mode_name="structured_notes",
        final_formatting_policy="structured_notes",
        style_card=None,
        focused_text_before="",
        focused_text_after="",
        selected_text_excerpt="",
        writer_tone_addon=None,
        metadata={},
    )

    formatter = FinalFormatter(backend=Backend())

    assert formatter.format(packet) is None
    assert formatter.last_rejection is not None
    assert formatter.last_rejection["reason"] == "source_required_terms_dropped"
    assert formatter.last_rejection["missing_source_required_terms"] == ["Juno"]


def test_final_formatter_allows_structural_count_to_be_removed_from_list_output() -> None:
    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text="- Remove patches\n- Make Qwen plan\n- Validate spans",
                backend_name="fake-qwen",
            )

    packet = FormattingPacket(
        utterance_id="utt-structural-count",
        corrected_text="Note down 3 points. First remove patches. Second make Qwen plan. Third validate spans.",
        app_name="Notes",
        app_category="docs",
        window_title=None,
        mode_name="structured_notes",
        final_formatting_policy="structured_notes",
        style_card=None,
        focused_text_before="",
        focused_text_after="",
        selected_text_excerpt="",
        writer_tone_addon=None,
        metadata={},
    )

    formatter = FinalFormatter(backend=Backend())
    result = formatter.format(packet)

    assert result is not None
    assert result.text == "- Remove patches\n- Make Qwen plan\n- Validate spans"
    assert formatter.last_rejection is None


def test_memory_serving_filters_lowercase_kebab_session_artifacts() -> None:
    snapshot = MemorySnapshot(
        schema_version=1,
        session_entities=[
            SessionEntity(value="off-callback", count=3, source="oneshot_commit"),
            SessionEntity(value="Best", count=3, source="oneshot_commit"),
            SessionEntity(value="After", count=2, source="oneshot_commit"),
            SessionEntity(value="Putting", count=1, source="oneshot_commit"),
            SessionEntity(value="Nilofar", count=1, source="oneshot_commit"),
            SessionEntity(value="Nemotron", count=1, source="oneshot_commit"),
        ]
    )

    packet = RecognitionBiasEngine().build_serving_packet(snapshot=snapshot)
    ranked = rank_memory_for_context(snapshot, context=TypedContextBundle(app_name="Notes"))

    assert "off-callback" not in packet.session_entities
    assert "off-callback" not in ranked.session_entities
    assert "Best" not in packet.session_entities
    assert "Best" not in ranked.session_entities
    assert "After" not in packet.session_entities
    assert "Putting" not in packet.session_entities
    assert "Nilofar" in packet.session_entities
    assert "Nilofar" in ranked.session_entities
    assert "Nemotron" in packet.session_entities


def test_session_entity_policy_keeps_rare_terms_not_common_words() -> None:
    for term in ("Best", "After", "Some", "Putting", "Finally", "Check", "Now"):
        assert not session_entity_allowed(term)

    for term in ("Nilofar", "Ishida", "Nemotron", "Qwen", "ASR", "E2E", "LumaRay"):
        assert session_entity_allowed(term)


def test_context_backed_common_names_can_enter_session_memory() -> None:
    assert commit_session_entity_allowed(
        "Tara",
        committed_text="Tell Tara I am running late.",
        spoken_evidence_text="Tell Tara I am running late.",
        context_text="WhatsApp chat with Tara, Noah, and Riya",
        context_backed=True,
    )
    assert not commit_session_entity_allowed(
        "Tara",
        committed_text="Tell Tara I am running late.",
        spoken_evidence_text="Tell Tara I am running late.",
        context_text="",
        context_backed=False,
    )
    assert not commit_session_entity_allowed(
        "May",
        committed_text="May is busy tomorrow.",
        spoken_evidence_text="May is busy tomorrow.",
        context_text="May",
        context_backed=True,
    )


def test_correction_memory_rejects_case_only_common_word_changes() -> None:
    assert not is_safe_correction_pair("best", "Best")
    assert not is_safe_correction_pair("finally", "Finally")
    assert is_safe_correction_pair("qwen", "Qwen")


def test_transcript_adjudicator_preserves_ai_first_resolution_metadata() -> None:
    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text=json.dumps(
                    {
                        "schema_version": "transcript_adjudication_v1",
                        "corrected_text": "Paris, no actually Prague, is the customer meeting location.",
                        "ops": [],
                        "confidence": 0.91,
                        "protected_terms_used": ["Prague"],
                        "intent": {"kind": "dictation"},
                        "formatting_plan": {"kind": "no_formatting"},
                        "self_corrections": [
                            {"cue": "no actually", "from": "Paris", "to": "Prague"}
                        ],
                        "terms_used": [{"term": "Prague", "source": "screen"}],
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
        whisper_text="Paris, no actually Prague, is the customer meeting location.",
        memory_candidate_text="Paris, no actually Prague, is the customer meeting location.",
        raw_text="Paris, no actually Prague, is the customer meeting location.",
        context_terms=(),
        protected_terms=("Prague",),
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
    assert result.metadata["self_corrections"][0]["to"] == "Prague"


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
        raw_text="No actually write WidgetName as one word.",
        protected_terms=("WidgetName",),
    )
    result = _adjudication_result(
        "Do not write WidgetName as one word.",
        protected_terms_used=("WidgetName",),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert not ok
    assert reason == "self_correction_cue_inverted:no_actually_write"


def test_transcript_validation_rejects_lost_final_tail_even_with_self_correction() -> None:
    packet = _adjudication_packet(
        raw_text=(
            "First section is risks scratch that make it open risks. "
            "Closing sentence must remain."
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


def test_transcript_validation_allows_actually_prefix_cleanup_when_target_survives() -> None:
    packet = _adjudication_packet(
        raw_text="We'll send the proposal by Thursday, actually Friday morning.",
    )
    result = _adjudication_result(
        "We'll send the proposal by Friday morning.",
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert ok, reason


def test_transcript_validation_rejects_dropped_scratch_that_target() -> None:
    packet = _adjudication_packet(
        raw_text="The owner is Neil scratch that Nilofer.",
    )
    result = _adjudication_result(
        "The owner is Neil.",
        protected_terms_used=(),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert not ok
    assert reason.startswith("source_words_dropped:")


def test_transcript_validation_allows_scratch_that_target_replacement() -> None:
    packet = _adjudication_packet(
        raw_text="The owner is Neil scratch that Nilofer.",
        protected_terms=("Nilofer",),
    )
    result = _adjudication_result(
        "The owner is Nilofer.",
        protected_terms_used=("Nilofer",),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert ok, reason


def test_transcript_validation_rejects_dropped_no_actually_target() -> None:
    packet = _adjudication_packet(
        raw_text="The customer meeting is in Paris, no actually Prague.",
    )
    result = _adjudication_result(
        "The customer meeting is in Paris.",
        protected_terms_used=(),
    )

    ok, reason = validate_adjudication_result(packet, result)

    assert not ok
    assert reason.startswith("source_words_dropped:")


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


def test_live_hint_repair_does_not_rewrite_valid_word_to_context_app_name() -> None:
    text, replacements = _reconcile_proper_nouns_from_live_hint(
        live_hint="Hey Juno, add a reminder to go to disco with Ishita next week.",
        final_text="Hey Juno, add a reminder to go to Disco with Ishita next week.",
        protected_terms=("Discord",),
    )

    assert text == "Hey Juno, add a reminder to go to Disco with Ishita next week."
    assert replacements == []


def test_live_hint_repair_only_preserves_case_for_same_word() -> None:
    text, replacements = _reconcile_proper_nouns_from_live_hint(
        live_hint="Ishita owns the follow-up.",
        final_text="ISHITA owns the follow-up.",
        protected_terms=(),
    )

    assert text == "Ishita owns the follow-up."
    assert replacements == [{"from": "ISHITA", "to": "Ishita", "source": "live_hint_case"}]


def test_action_llm_body_prefers_spoken_evidence_over_model_word_substitution() -> None:
    source = "set an alarm for 5 pm tomorrow and add a reminder to go to Disco with Ishita next week."
    actions = validate_envelope(
        {
            "intent": "execute_action",
            "should_execute": True,
            "confidence": 0.94,
            "decision_evidence_span": source,
            "actions": [
                {
                    "kind": "reminder",
                    "body": "go to Discord with Ishita next week",
                    "evidence_span": "add a reminder to go to Disco with Ishita next week",
                    "when_text": "next week",
                    "confidence": 0.9,
                }
            ],
        },
        raw_span_fallback=source,
        now=datetime(2026, 6, 2, 12, 0, 0),
    )

    assert actions is not None
    assert actions[0].body == "go to Disco with Ishita"


def test_protected_phrase_token_repair_handles_screen_name_near_miss() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="The visible screen says Sylvia Gamache.",
        protected_terms=("Silvia Gamache",),
    )

    assert repaired == "The visible screen says Silvia Gamache."
    assert replacements == [{"from": "Sylvia", "to": "Silvia", "source": "protected_term_near_miss"}]


def test_explicit_candidate_repair_does_not_rewrite_common_words_to_screen_names() -> None:
    repaired, replacements = _reconcile_explicit_candidate_term_confusions(
        text="Please remind me to send the draft.",
        explicit_candidate_terms=("Maia",),
        protected_terms=("Maia",),
    )

    assert repaired == "Please remind me to send the draft."
    assert replacements == []


def test_static_ai_glossary_does_not_rewrite_plain_common_words_by_default() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="First alpha, second beta, third gamma.",
        protected_terms=(),
    )

    assert repaired == "First alpha, second beta, third gamma."
    assert replacements == []


def test_static_ai_glossary_does_not_include_person_names() -> None:
    assert "Marco" not in AI_GLOSSARY


def test_protected_context_can_repair_common_word_to_glossary_term() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="Use gamma for the local model layer.",
        protected_terms=("Gemma",),
    )

    assert repaired == "Use Gemma for the local model layer."
    assert replacements == [{"from": "gamma", "to": "Gemma", "source": "protected_term_near_miss"}]


def test_protected_context_does_not_rewrite_audit_to_screen_ocr_name() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="Run an end-to-end audit of the HUD punctuation path.",
        protected_terms=("Audii",),
    )

    assert repaired == "Run an end-to-end audit of the HUD punctuation path."
    assert replacements == []


def test_protected_context_preserves_phonetic_person_name_repair() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="Please send this to parish for review.",
        protected_terms=("Paresh",),
    )

    assert repaired == "Please send this to Paresh for review."
    assert replacements == [{"from": "parish", "to": "Paresh", "source": "protected_term_near_miss"}]


def test_protected_context_does_not_rewrite_name_to_ocr_lookalike() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="Ask Priya to check punctuation.",
        protected_terms=("Prlya",),
    )

    assert repaired == "Ask Priya to check punctuation."
    assert replacements == []


def test_screen_term_does_not_pluralize_common_word_in_user_speech() -> None:
    # Production 2026-06-11: Juno's own sidebar phrase made "Actions" a
    # repair target and "take a note, action items…" became "Actions
    # items…", which then broke turn-plan span grounding for the note body.
    # A common word and its own plural are the same word inflected, never a
    # near-miss.
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="Hey Juno, take a note, action items need to be finished",
        protected_terms=(
            "Juno Home History Actions Voice Commands Styles Dictionary",
            "Actions",
            "Juno",
        ),
    )

    assert repaired == "Hey Juno, take a note, action items need to be finished"
    assert replacements == []


def test_protected_context_does_not_rewrite_proper_term_to_common_word() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="Please send the Solara investor brief.",
        protected_terms=("Solar",),
    )

    assert repaired == "Please send the Solara investor brief."
    assert replacements == []


def test_static_ai_glossary_repairs_non_common_near_misses_without_app_gate() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="Use Quen for the local model layer.",
        protected_terms=(),
    )

    assert repaired == "Use Qwen for the local model layer."
    assert replacements == [{"from": "Quen", "to": "Qwen", "source": "protected_term_near_miss"}]


# --- Issue #68: OCR-corrupted screen terms must not rewrite correct words ----


def test_all_caps_screen_label_does_not_bypass_the_soundex_guard() -> None:
    # Issue #68 sub-defect 1. The soundex guard was skipped whenever
    # ``_single_token_has_identifier_shape(target)`` was true, and that is true
    # for ANY all-caps token. An OCR-mangled all-caps UI label therefore
    # rewrote a phonetically unrelated word purely on edit ratio.
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="I clicked on paste diagnostic thing",
        protected_terms=("PASTF",),
    )

    assert repaired == "I clicked on paste diagnostic thing"
    assert replacements == []


def test_token_that_is_itself_a_protected_term_is_never_rewritten() -> None:
    # Issue #68 sub-defect 2. Identity only disqualifies the one candidate the
    # token equals; the correctly spelled token was then matched against an
    # OCR-corrupted twin sitting in the same term list and rewritten into it.
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="we shipped Mixtral to production",
        protected_terms=("Mixtral", "Mixtrai"),
    )

    assert repaired == "we shipped Mixtral to production"
    assert replacements == []


def test_phrase_token_that_is_itself_a_protected_term_is_never_rewritten() -> None:
    # Same defect via the phrase-token expansion of a multi-word screen term:
    # "Gamache" comes from the phrase, and the corrupted standalone twin
    # "Gamashe" (same soundex) then rewrote the correctly spelled token.
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="ask Gamache about the rollout",
        protected_terms=("Silvia Gamache", "Gamashe"),
    )

    assert repaired == "ask Gamache about the rollout"
    assert replacements == []


def test_common_word_is_not_upgraded_to_longer_product_name() -> None:
    # Issue #68 sub-defect 3. There was a guard for observed-longer-and-prefix
    # -of-target but not the reverse, so an ordinary noun the user actually
    # said got a suffix appended from a screen term.
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="open the Widget of a single box",
        protected_terms=("WidgetX",),
    )

    assert repaired == "open the Widget of a single box"
    assert replacements == []


def test_self_truncated_proper_noun_is_still_repaired_without_eating_next_word() -> None:
    # The sub-defect 3 guard must stay narrow. The +/-1 length check already
    # blocks "Kube" -> "Kubernetes", so an unconditional reverse-prefix guard
    # only ever bit when observed is the target minus its final character --
    # which for a plain proper noun is a real ASR/self-truncation that must
    # still repair. Worse, blocking it in ``repl`` handed the token to
    # ``_reconcile_split_candidate_term``, which glued it to the FOLLOWING word
    # and deleted that word ("... Kubernete now" -> "... Kubernetes").
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="deploy to Kubernete now",
        protected_terms=("Kubernetes",),
    )

    assert repaired == "deploy to Kubernetes now"
    assert replacements == [
        {"from": "Kubernete", "to": "Kubernetes", "source": "protected_term_near_miss"}
    ]


def test_self_truncated_product_name_repair_preserves_following_function_word() -> None:
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="use Postgre so we can",
        protected_terms=("Postgres",),
    )

    assert repaired == "use Postgres so we can"
    assert replacements == [
        {"from": "Postgre", "to": "Postgres", "source": "protected_term_near_miss"}
    ]


def test_near_miss_repair_terms_exclude_what_the_candidate_pass_canonicalised() -> None:
    # Issue #68 sub-defect 4. The explicit-candidate pass canonicalises the
    # corrupted twin back to the real term, then the near-miss pass runs on its
    # output with the twin still on the repair list and reverts it — two
    # replacements, zero net effect.
    repair_terms = ("Mixtral", "Mixtrai")
    reconciled, candidate_replacements = _reconcile_explicit_candidate_term_confusions(
        text="we shipped Mixtrai to production",
        explicit_candidate_terms=("Mixtral",),
        protected_terms=repair_terms,
    )

    assert reconciled == "we shipped Mixtral to production"
    assert candidate_replacements == [
        {"from": "Mixtrai", "to": "Mixtral", "source": "explicit_candidate"}
    ]

    remaining_terms = _repair_terms_after_candidate_reconciliation(
        repair_terms, candidate_replacements
    )
    assert remaining_terms == ("Mixtral",)

    repaired, replacements = _reconcile_protected_term_near_misses(
        text=reconciled,
        protected_terms=remaining_terms,
    )

    assert repaired == "we shipped Mixtral to production"
    assert replacements == []

    # Belt and braces: even handed the unfiltered list, the near-miss pass must
    # leave the canonicalised token alone rather than revert it.
    unfiltered_repaired, unfiltered_replacements = _reconcile_protected_term_near_misses(
        text=reconciled,
        protected_terms=repair_terms,
    )

    assert unfiltered_repaired == "we shipped Mixtral to production"
    assert unfiltered_replacements == []


def test_repair_terms_after_candidate_reconciliation_is_a_no_op_without_edits() -> None:
    repair_terms = ("Mixtral", "Mixtrai")

    assert _repair_terms_after_candidate_reconciliation(repair_terms, []) == repair_terms
    assert (
        _repair_terms_after_candidate_reconciliation(
            repair_terms,
            [{"from": "Mixtral", "to": "Mixtral", "source": "explicit_candidate"}],
        )
        == repair_terms
    )


def test_genuine_misheard_product_name_is_still_repaired() -> None:
    # The issue #68 guards must not weaken real near-miss repair: a genuinely
    # misheard product name with matching soundex still gets fixed.
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="we shipped Mixxtral last week",
        protected_terms=("Mixtral",),
    )

    assert repaired == "we shipped Mixtral last week"
    assert replacements == [
        {"from": "Mixxtral", "to": "Mixtral", "source": "protected_term_near_miss"}
    ]


def test_near_miss_repair_still_fires_when_correct_term_appears_elsewhere() -> None:
    # Skipping exact-match tokens must be per-token, not a whole-text bail-out.
    repaired, replacements = _reconcile_protected_term_near_misses(
        text="Mixtral first, then Mixxtral again",
        protected_terms=("Mixtral",),
    )

    assert repaired == "Mixtral first, then Mixtral again"
    assert replacements == [
        {"from": "Mixxtral", "to": "Mixtral", "source": "protected_term_near_miss"}
    ]


def test_oneshot_salvages_qwen_self_correction_after_protected_phrase_repair() -> None:
    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript=(
                    "Start a note. The visible screen says Sylvia Gamache. "
                    "Paris, no actually Prague, is the customer meeting location."
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
                            "Prague is the customer meeting location."
                        ),
                        "ops": [],
                        "confidence": 0.91,
                        "protected_terms_used": ["Silvia Gamache"],
                        "self_corrections": [
                            {"cue": "no actually", "from": "Paris", "to": "Prague"}
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
        "Prague is the customer meeting location."
    )
    assert "Paris, no actually Prague" not in result.transcript
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
                candidate_entities=["LumaRay", "Okay", "Best", "After", "Some", "Putting"],
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
        committed_text="LumaRay battery risk is assigned. Best After Some Putting.",
    )

    assert learned["learned"] is True
    assert "LumaRay" in memory.entities
    assert "Okay" not in memory.entities
    assert "Best" not in memory.entities
    assert "After" not in memory.entities
    assert "Some" not in memory.entities
    assert "Putting" not in memory.entities
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


def test_wake_action_transcript_adjudication_still_runs_before_turn_planning() -> None:
    reason = _final_adjudication_fast_skip_reason(
        live_adjudication=False,
        transcript_hint=None,
        raw_text="Hey Juno, remind me tomorrow at 9 to send the draft.",
        normalized_text="Hey Juno, remind me tomorrow at 9 to send the draft.",
        normalization_applied=[],
        audio_duration_ms=3000.0,
    )
    ordinary = _final_adjudication_fast_skip_reason(
        live_adjudication=False,
        transcript_hint=None,
        raw_text="Remind me to send the draft, but write it as a note.",
        normalized_text="Remind me to send the draft, but write it as a note.",
        normalization_applied=[],
        audio_duration_ms=3000.0,
    )

    assert reason is None
    assert ordinary is None


def test_transcript_adjudication_prompt_omits_hardcoded_slot_examples() -> None:
    req = WriterTransformRequest(
        utterance_id="utt-adj-prompt",
        instruction="Resolve final transcript.",
        source_text="The speaker revised part of the sentence.",
        mode=WriterMode.DEFAULT_SURFACE,
        context_payload={"task": "transcript_adjudication_v1"},
    )

    prompt = _system_prompt(req)

    assert "Do not invert the meaning of an explicit correction." in prompt
    assert "Paris, no actually Prague" not in prompt
    assert "LumaRay" not in prompt


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


def test_writer_parser_does_not_misclassify_reply_dictation_as_transform() -> None:
    text = (
        "Reply to Jordan Quick recap from today, we'll send the Proposal by Friday morning, "
        "include pricing, rollout plan, and open questions, "
        "end with I'll keep it short."
    )

    intent = WriterIntentParser().parse(
        text,
        selection_present=False,
        active_mode=WriterMode.FORMAL_EMAIL,
        mode_policy=BUILTIN_MODES["formal_email"],
    )

    assert intent.kind == WriterIntentKind.DICTATE


def test_writer_parser_still_accepts_real_unscoped_transform_command() -> None:
    intent = WriterIntentParser().parse(
        "Make this shorter.",
        selection_present=False,
        active_mode=WriterMode.DEFAULT_SURFACE,
        mode_policy=BUILTIN_MODES["default_surface"],
    )

    assert intent.kind != WriterIntentKind.DICTATE


def test_selection_transform_prompt_preserves_paragraph_shape_without_structure_request() -> None:
    req = WriterTransformRequest(
        utterance_id="utt-selection-prompt",
        instruction="Make this shorter and more direct.",
        source_text="Customer interested in rollout plan, security notes, and open questions.",
        mode=WriterMode.DEFAULT_SURFACE,
        target_selection=ClientSelection(start=0, end=10),
        context_payload={"task": "selection_transform_v1"},
    )

    prompt = _system_prompt(req)

    assert "return a paragraph, not bullets" in prompt.lower()
    assert "explicitly requests a list" in prompt.lower()


def test_explicit_snippet_insert_commits_body_without_final_formatting() -> None:
    with tempfile.TemporaryDirectory(prefix="juno-snippet-test-") as tmp:
        memory = JsonMemoryStore(tmp)
        memory.snippets.add(
            trigger="customer follow up snippet",
            body="Customer Follow-Up\nContext:\nPain:\nNext step:\nOwner:\nDeadline:",
            scope="global",
        )
        service = WriterService(
            config=WriterConfig(enable_model_transforms=False),
            recorder=_Recorder(),
            backend=None,
        )
        mode_policy = replace(BUILTIN_MODES["default_surface"], final_formatting_policy="structured_notes")

        result = service.process_transcript(
            utterance_id="utt-snippet",
            final_text="Insert customer follow-up snippet.",
            raw_text="Insert customer follow-up snippet.",
            context=TypedContextBundle(app_name="Notes", app_category="docs"),
            anchor_selection=None,
            memory_store=memory,
            memory_snapshot=memory.snapshot(),
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

    assert result.action == WriterActionKind.PASS_THROUGH_COMMIT
    assert result.output_text == "Customer Follow-Up\nContext:\nPain:\nNext step:\nOwner:\nDeadline:"
    assert result.metadata["dictation_cleanup"]["pipeline"] == "snippet_direct_insert"
    assert result.metadata["punctuation_floor"] == {
        "changed": False,
        "rules_applied": [],
        "skip_reason": "snippet_expanded",
    }


def test_oneshot_response_exposes_snippet_expanded_metadata() -> None:
    spoken = "Insert customer follow-up snippet."

    class FakeTranscriber:
        backend_name = "fake_asr"

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            return TranscribeResult(
                transcript=spoken,
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=1.0,
                model_path="fake",
            )

    class FakeContextProvider:
        def snapshot(self) -> TypedContextBundle:
            return TypedContextBundle(app_name="Notes", app_category="docs")

    with tempfile.TemporaryDirectory(prefix="juno-snippet-oneshot-") as tmp:
        memory = JsonMemoryStore(tmp)
        memory.snippets.add(
            trigger="customer follow up snippet",
            body="Customer Follow-Up\nContext:\nPain:\nNext step:\nOwner:\nDeadline:",
            scope="global",
        )
        writer = WriterService(
            config=WriterConfig(enable_model_transforms=False),
            recorder=_Recorder(),
            backend=None,
        )
        pipeline = OneShotDictationPipeline(
            transcriber=FakeTranscriber(),
            recorder=_Recorder(),
            context_provider=FakeContextProvider(),
            memory_store=memory,
            writer_service=writer,
            transcript_adjudicator_config=TranscriptAdjudicatorConfig(enabled=False),
            itn_enabled=False,
        )
        result = pipeline.run(_loud_wav_bytes(), utterance_id="utt-snippet-meta", save_history=False, save_audio=False)

    payload = result.to_dict()
    writer_meta = payload["metadata"]["writer_outcome"]
    assert result.transcript == "Customer Follow-Up\nContext:\nPain:\nNext step:\nOwner:\nDeadline:"
    assert writer_meta["snippet_expanded"] is True
    assert payload["metadata"]["snippet_expanded"] is True
    assert writer_meta["dictation_cleanup"]["pipeline"] == "snippet_direct_insert"
    assert writer_meta["punctuation_floor"] == {
        "changed": False,
        "rules_applied": [],
        "skip_reason": "snippet_expanded",
    }


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


def test_recent_transform_command_uses_focused_text_before_without_commit_metadata() -> None:
    recent = "Permissions are reset. Actions are passing. We still need one production paste check."
    service = WriterService(
        config=WriterConfig(enable_model_transforms=False),
        recorder=_Recorder(),
        backend=None,
    )

    result = service.process_transcript(
        utterance_id="utt-focused-recent-bullets",
        final_text="Turn that into bullets.",
        raw_text="Turn that into bullets.",
        context=TypedContextBundle(
            app_name="Notes",
            app_category="docs",
            focused_text_before=recent,
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

    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.output_text == (
        "- Permissions are reset\n"
        "- Actions are passing\n"
        "- We still need one production paste check"
    )
    assert result.metadata["target"] == "focused_text_before"


def test_recent_model_transform_uses_focused_text_before_without_commit_metadata() -> None:
    captured: dict[str, object] = {}

    class Backend:
        def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
            captured["request"] = req
            return WriterTransformResult(
                utterance_id=req.utterance_id,
                text="Shorter launch check.",
                backend_name="fake-qwen",
            )

    recent = "Permissions are reset. Actions are passing. We still need one production paste check."
    service = WriterService(
        config=WriterConfig(enable_model_transforms=True),
        recorder=_Recorder(),
        backend=Backend(),
    )

    result = service.process_transcript(
        utterance_id="utt-focused-recent-shorter",
        final_text="Make that shorter.",
        raw_text="Make that shorter.",
        context=TypedContextBundle(
            app_name="Notes",
            app_category="docs",
            focused_text_before=recent,
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
    assert result.output_text == "Shorter launch check."
    assert result.metadata["target"] == "focused_text_before"


def test_focused_tail_replace_counts_chars_to_caret_and_keeps_trailing_whitespace() -> None:
    # The shell deletes target_text_chars back from the CARET. A caret two
    # newlines past the paragraph (user pressed Enter twice, then asked for
    # the rewrite) must count those newlines or the delete chops the
    # paragraph mid-word and strands the blank lines.
    recent = "Permissions are reset. Actions are passing. We still need one production paste check."
    service = WriterService(
        config=WriterConfig(enable_model_transforms=False),
        recorder=_Recorder(),
        backend=None,
    )

    result = service.process_transcript(
        utterance_id="utt-focused-trailing-ws",
        final_text="Turn that into bullets.",
        raw_text="Turn that into bullets.",
        context=TypedContextBundle(
            app_name="Notes",
            app_category="docs",
            focused_text_before="Intro para.\n\n" + recent + "\n\n",
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

    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.metadata["target"] == "focused_text_before"
    assert result.metadata["target_text_chars"] == len(recent) + 2
    # The caret's blank-line state survives the rewrite.
    assert result.output_text.endswith("\n\n")
    assert result.output_text.startswith("- Permissions are reset")


def test_delete_last_sentence_recent_ships_caret_anchored_replace_chars() -> None:
    # "Delete the last sentence" against the focused tail previously shipped
    # no target_text_chars at all — the shell deleted nothing and pasted a
    # duplicate of the shortened paragraph.
    recent = "Permissions are reset. Actions are passing. We still need one check."
    service = WriterService(
        config=WriterConfig(enable_model_transforms=False),
        recorder=_Recorder(),
        backend=None,
    )

    result = service.process_transcript(
        utterance_id="utt-focused-delete-sentence",
        final_text="Delete the last sentence.",
        raw_text="Delete the last sentence.",
        context=TypedContextBundle(
            app_name="Notes",
            app_category="docs",
            focused_text_before=recent + "\n",
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

    assert result.action == WriterActionKind.TRANSFORM_COMMIT
    assert result.metadata["target"] == "focused_text_before"
    assert result.metadata["target_text_chars"] == len(recent) + 1
    assert result.output_text == "Permissions are reset. Actions are passing.\n"


def test_deterministic_list_lanes_defer_to_editor_on_unresolved_correction_cue() -> None:
    # A surviving "scratch that" needs meaning-level judgment; the natural
    # bullet renderer must yield instead of shipping the cue inside a bullet.
    class _CapturingRecorder(_Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[tuple[str, dict]] = []

        def record(self, kind: object, name: str = "", payload: dict | None = None) -> None:
            self.events.append((str(name), dict(payload or {})))

    recorder = _CapturingRecorder()
    service = WriterService(
        config=WriterConfig(enable_model_transforms=False),
        recorder=recorder,
        backend=None,
    )

    def _run(text: str, uid: str):
        return service.process_transcript(
            utterance_id=uid,
            final_text=text,
            raw_text=text,
            context=TypedContextBundle(app_name="Notes", app_category="docs"),
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

    cue = _run(
        "I have three things to do today. First, email Sam scratch that email Sandra. "
        "Second, pay rent. Third, book flights.",
        "utt-list-cue",
    )
    assert not cue.output_text.startswith("- ")
    deferrals = [p for n, p in recorder.events if n == "deterministic_list_deferred_to_editor"]
    assert deferrals and deferrals[0]["lane"] == "natural_bullet_list"

    clean = _run(
        "I have three things to do today. First, email Sandra. Second, pay rent. "
        "Third, book flights.",
        "utt-list-clean",
    )
    assert clean.output_text == "- email Sandra\n- pay rent\n- book flights"


def test_recent_transform_command_grammar_covers_natural_variants() -> None:
    clearer = parse_deterministic_command("make that text more clear")
    shorter = parse_deterministic_command("make that shorter and more direct")

    assert clearer is not None
    assert clearer.kind == "recent_edit"
    assert clearer.payload["instruction"] == "Improve clarity. Preserve meaning."
    assert shorter is not None
    assert shorter.kind == "recent_edit"
    assert shorter.payload["instruction"] == "Make the text more concise and direct. Preserve meaning."


def test_final_asr_live_hint_audit_keeps_final_asr_on_by_default(monkeypatch) -> None:
    monkeypatch.delenv("JUNO_V2_SKIP_FINAL_ASR_ON_FINAL_PREVIEW_FLUSH", raising=False)

    class FakeTranscriber:
        backend_name = "fake_asr"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            self.calls += 1
            return TranscribeResult(
                transcript="final whisper transcript",
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=17.0,
                model_path="fake-whisper",
            )

    transcriber = FakeTranscriber()
    pipeline = OneShotDictationPipeline(
        transcriber=transcriber,
        recorder=_Recorder(),
        writer_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-final-asr-audit-default",
        transcript_hint="live preview transcript",
        shell_timeline={"final_preview_flush_received_ms": 123},
        save_history=False,
        save_audio=False,
        app_bundle_id="com.apple.Terminal",
    )

    audit = result.metadata["final_asr_live_hint_audit"]
    assert transcriber.calls == 1
    assert result.raw_transcript == "final whisper transcript"
    assert audit["hint_present"] is True
    assert audit["final_preview_flush_received"] is True
    assert audit["skip_eligible"] is True
    assert audit["skip_enabled"] is False
    assert audit["skip_used"] is False
    assert audit["backend"] == "fake_asr"


def test_final_asr_live_hint_skip_requires_explicit_env_and_final_preview_flush(monkeypatch) -> None:
    monkeypatch.setenv("JUNO_V2_SKIP_FINAL_ASR_ON_FINAL_PREVIEW_FLUSH", "1")

    class FakeTranscriber:
        backend_name = "fake_asr"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe_wav(self, *args: object, **kwargs: object) -> TranscribeResult:
            self.calls += 1
            return TranscribeResult(
                transcript="should not be used",
                language="en",
                backend_name="fake_asr",
                audio_duration_ms=1000.0,
                decode_ms=17.0,
                model_path="fake-whisper",
            )

    transcriber = FakeTranscriber()
    pipeline = OneShotDictationPipeline(
        transcriber=transcriber,
        recorder=_Recorder(),
        writer_enabled=False,
    )

    result = pipeline.run(
        _loud_wav_bytes(),
        utterance_id="utt-final-asr-audit-skip",
        transcript_hint="live preview transcript",
        shell_timeline={"final_preview_flush_received_ms": 123},
        save_history=False,
        save_audio=False,
        app_bundle_id="com.apple.Terminal",
    )

    audit = result.metadata["final_asr_live_hint_audit"]
    assert transcriber.calls == 0
    assert result.raw_transcript == "live preview transcript"
    assert result.backend_name == "live_transcript_hint_final"
    assert audit["skip_enabled"] is True
    assert audit["skip_used"] is True
    assert audit["decode_ms"] == 0.0


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


def test_auto_language_policy_reaches_backend_without_english_fallback() -> None:
    class CapturingBackend:
        backend_name = "fake_asr"
        config = FinalAsrConfig(model_path="fake", backend_name="fake_asr", language="en")

        def __init__(self) -> None:
            self.language_seen: str | None = "unset"
            self.policy_seen: str | None = None

        def warm(self) -> None:
            return None

        def decode(self, req: object) -> FinalDecodeResult:
            self.language_seen = getattr(req, "language", None)
            self.policy_seen = getattr(req, "language_policy", None)
            return FinalDecodeResult(
                utterance_id="utt-language",
                text="नमस्ते",
                start_ms=0.0,
                end_ms=1000.0,
                audio_duration_ms=1000.0,
                backend_name="fake_asr",
                language="hi",
            )

    backend = CapturingBackend()
    transcriber = FinalBackendTranscriber(backend=backend, language="en")

    result = transcriber.transcribe_wav(
        _loud_wav_bytes(),
        language=None,
        language_policy="auto_supported",
    )

    assert backend.language_seen is None
    assert backend.policy_seen == "auto_supported"
    assert result.language == "hi"


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


# --------------------------------------------------------------------- #
# Screen-term prompt hygiene
# --------------------------------------------------------------------- #


def test_screen_terms_filtered_before_whisper_prompt() -> None:
    from juno_v2.memory.bias import (
        _diversify_bias_phrases,
        is_biasable_runtime_context_candidate,
        screen_term_prompt_worthy,
    )

    # Production 2026-06-11: with WhatsApp Web on screen the Whisper prompt
    # filled with UI chrome, OCR junk, and chat-contact handles — and a
    # replay showed the junk prompt driving a faint utterance into a
    # "team team team…" repetition loop.
    screen_terms = [
        "WhatsApp",            # proper app name — keep
        "High",                # common English UI word — drop
        "Back",                # common English UI word — drop
        "Reload",              # common English UI word — drop
        "lTr",                 # OCR junk — drop
        "Mwrk5pace",           # OCR junk with digit — drop
        "Adi41",               # contact handle with digit — drop
        "onboardin9",          # OCR digit leak from onboarding text — drop
        "Acces5ibility",       # OCR digit leak from Accessibility text — drop
        "Rerninders",          # OCR near-miss of Reminders — drop
        "JuDo",                # OCR near-miss / odd mixed-case short token — drop
        "NOvaD",               # odd OCR camel-case fragment — drop
        "SettiThJs",           # odd OCR camel-case fragment — drop
        "CityXyoTer",          # three-part OCR camel fragment — drop
        "bhlS.py",             # OCR-shaped dotted identifier — drop
        "atlons/Junty.app",    # OCR-shaped path fragment — drop
        "FusionX Bookmarks New Tab Back Forward Reload Bookmark",  # run-on — drop
        "Cassini Research",    # name phrase — keep
        "NovaDesk",            # product-style name — keep
        "OpenAI",              # acronym-suffix product name — keep
        "VPN",                 # acronym — keep
        "juno_v2",             # technical identifier — runtime seed only
    ]
    out = _diversify_bias_phrases([], screen_terms=screen_terms, cap=24)

    assert "WhatsApp" in out
    assert "Cassini Research" in out
    assert "NovaDesk" in out
    assert "OpenAI" in out
    assert "VPN" in out
    for junk in (
        "High", "Back", "Reload", "lTr", "Mwrk5pace", "Adi41",
        "onboardin9", "Acces5ibility", "Rerninders", "JuDo", "NOvaD",
        "SettiThJs", "CityXyoTer", "bhlS.py", "atlons/Junty.app", "juno_v2",
    ):
        assert junk not in out, junk
    assert not any("Bookmarks New Tab" in t for t in out)
    for junk in (
        "onboardin9", "Acces5ibility", "Rerninders", "JuDo", "NOvaD",
        "SettiThJs", "CityXyoTer", "bhlS.py", "atlons/Junty.app",
    ):
        assert not screen_term_prompt_worthy(junk)
        assert not is_biasable_runtime_context_candidate(junk)
    assert is_biasable_runtime_context_candidate("Cassini Research")
    assert is_biasable_runtime_context_candidate("NovaDesk")
    assert is_biasable_runtime_context_candidate("OpenAI")
    assert is_biasable_runtime_context_candidate("juno_v2")


def test_common_ui_action_words_dropped_but_names_kept() -> None:
    # Regression: the single-word screen-term gate must reject common email /
    # chat / app action chrome (Send, Reply, Compose, ...) so it never floods
    # the Whisper bias prompt, while still admitting genuine names/acronyms
    # (which are also "common" words to the dictionary but worth biasing).
    from juno_v2.memory.bias import screen_term_prompt_worthy

    for ui_word in (
        "Send", "Reply", "Compose", "Archive", "Spam", "Snooze",
        "Filter", "Sort", "Move", "Print", "Share", "Delete", "Draft",
    ):
        assert not screen_term_prompt_worthy(ui_word), ui_word

    for name in ("Maia", "Maya", "River", "Grace", "Nilofar", "Cassini Research", "OpenAI"):
        assert screen_term_prompt_worthy(name), name


def test_screen_ocr_junk_filtered_from_preview_repair_terms() -> None:
    from juno_v2.preview.personalization_repair import collect_preview_personalization_terms

    terms = collect_preview_personalization_terms(
        {
            "candidate_entities": ["onboardin9", "Acces5ibility", "NovaDesk"],
            "recent_screen_terms": ["Rerninders", "JuDo", "Cassini Research"],
        }
    )
    texts = {term.text for term in terms}

    assert "NovaDesk" in texts
    assert "Cassini Research" in texts
    for junk in ("onboardin9", "Acces5ibility", "Rerninders", "JuDo"):
        assert junk not in texts


def test_memory_phrases_not_subject_to_screen_gate() -> None:
    from juno_v2.memory.bias import _diversify_bias_phrases

    # Memory-lane phrases are policy-gated at learn time; serving keeps them
    # even when they would fail the screen-term shape gate.
    out = _diversify_bias_phrases(
        ["o4-mini-high", "luma-mode"], screen_terms=[], cap=24
    )
    assert out == ["o4-mini-high", "luma-mode"]
