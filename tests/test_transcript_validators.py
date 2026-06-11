from __future__ import annotations

from juno_v2.context.compiler import TranscriptAdjudicationPacket
from juno_v2.transcript.contracts import TranscriptAdjudicationResult, TranscriptPatchOp
from juno_v2.transcript.validators import (
    no_assistant_artifacts,
    numbers_dates_safe,
    protected_terms_preserved,
    remove_instructional_exclusion_phrases,
    repair_low_signal_mid_sentence_capitalization,
    restore_explicit_final_word_tail,
    semantic_drift_safe,
    source_words_preserved,
    validate_adjudication_result,
)


def make_packet(**overrides: object) -> TranscriptAdjudicationPacket:
    base: dict[str, object] = dict(
        stage="final",
        utterance_id="utt-1",
        base_visible_text="",
        base_visible_revision=None,
        live_preview_text="",
        whisper_text="",
        memory_candidate_text="",
        raw_text="",
        context_terms=(),
        protected_terms=(),
        selected_text_excerpt="",
        focused_text_before="",
        focused_text_after="",
        field_text_excerpt="",
        app_name=None,
        app_category=None,
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
    base.update(overrides)
    return TranscriptAdjudicationPacket(**base)  # type: ignore[arg-type]


def make_result(
    corrected_text: str,
    *,
    stage: str = "final",
    ops: tuple[TranscriptPatchOp, ...] = (),
) -> TranscriptAdjudicationResult:
    return TranscriptAdjudicationResult(
        utterance_id="utt-1",
        stage=stage,  # type: ignore[arg-type]
        corrected_text=corrected_text,
        ops=ops,
        confidence=0.9,
        base_visible_revision=None,
        base_text_hash=None,
        stable_prefix_chars=None,
        protected_terms_used=(),
    )


# ---------------------------------------------------------------------------
# no_assistant_artifacts
# ---------------------------------------------------------------------------


def test_no_assistant_artifacts_accepts_plain_dictation() -> None:
    assert no_assistant_artifacts("please send the report tomorrow") == (True, "ok")


def test_no_assistant_artifacts_rejects_empty_output() -> None:
    assert no_assistant_artifacts("") == (False, "empty_output")
    assert no_assistant_artifacts("   \n ") == (False, "empty_output")


def test_no_assistant_artifacts_rejects_preambles() -> None:
    for text in (
        "Sure, here it is",
        "Here is the corrected text",
        "Here's the cleaned up version",
        "The corrected text is as follows",
        "Corrected text: please send it",
    ):
        ok, reason = no_assistant_artifacts(text)
        assert not ok and reason == "assistant_preamble", text


def test_no_assistant_artifacts_rejects_markdown_fences() -> None:
    ok, reason = no_assistant_artifacts("```\nsome code\n```")
    assert not ok and reason == "markdown_fence"


def test_no_assistant_artifacts_does_not_flag_mid_sentence_here_is() -> None:
    # The preamble pattern is anchored at the start of the output.
    assert no_assistant_artifacts("the package here is heavy") == (True, "ok")


# ---------------------------------------------------------------------------
# protected_terms_preserved
# ---------------------------------------------------------------------------


def test_protected_terms_no_terms_is_ok() -> None:
    packet = make_packet(memory_candidate_text="hello there")
    assert protected_terms_preserved(packet, "hello there") == (True, "ok")


def test_protected_term_dropped_from_output_is_rejected() -> None:
    packet = make_packet(
        memory_candidate_text="deploy the Kubernetes cluster",
        protected_terms=("Kubernetes",),
    )
    ok, reason = protected_terms_preserved(packet, "deploy the cluster")
    assert not ok and reason == "protected_term_dropped:Kubernetes"


def test_protected_term_preserved_case_insensitively() -> None:
    packet = make_packet(
        memory_candidate_text="deploy the kubernetes cluster",
        protected_terms=("Kubernetes",),
    )
    assert protected_terms_preserved(packet, "deploy the Kubernetes cluster") == (True, "ok")


def test_protected_term_absent_from_evidence_is_not_required() -> None:
    packet = make_packet(
        memory_candidate_text="send the weekly notes",
        protected_terms=("Kubernetes",),
    )
    assert protected_terms_preserved(packet, "send the weekly notes") == (True, "ok")


def test_protected_multitoken_term_tolerates_punctuation_formatting() -> None:
    # "May 18 2026" rewritten as "May 18, 2026" is formatting, not a drop.
    packet = make_packet(
        memory_candidate_text="the deadline is May 18 2026",
        protected_terms=("May 18 2026",),
    )
    assert protected_terms_preserved(packet, "The deadline is May 18, 2026.") == (True, "ok")


def test_protected_glued_pronoun_artifact_is_ignored() -> None:
    # SFSpeech artifacts like "workI" must not behave as protected terms.
    packet = make_packet(
        memory_candidate_text="after workI went home",
        protected_terms=("workI",),
    )
    assert protected_terms_preserved(packet, "after work I went home") == (True, "ok")


# ---------------------------------------------------------------------------
# numbers_dates_safe
# ---------------------------------------------------------------------------


def test_numbers_dates_safe_with_no_numbers() -> None:
    packet = make_packet(memory_candidate_text="no digits here")
    assert numbers_dates_safe(packet, "no digits here") == (True, "ok")


def test_numbers_present_in_source_are_allowed() -> None:
    packet = make_packet(memory_candidate_text="meet at 3:30 on the 14th")
    assert numbers_dates_safe(packet, "Meet at 3:30 on the 14th.") == (True, "ok")


def test_time_separator_normalization_is_not_a_new_number() -> None:
    packet = make_packet(memory_candidate_text="meet at 3.30")
    assert numbers_dates_safe(packet, "meet at 3:30") == (True, "ok")


def test_new_number_without_evidence_is_rejected() -> None:
    packet = make_packet(memory_candidate_text="meet at three")
    ok, reason = numbers_dates_safe(packet, "meet at 4")
    assert not ok and reason == "new_number_without_evidence"


def test_numbers_from_protected_terms_are_authorized() -> None:
    packet = make_packet(
        memory_candidate_text="ship it on the launch date",
        protected_terms=("May 18 2026",),
    )
    assert numbers_dates_safe(packet, "ship it on May 18 2026") == (True, "ok")


# ---------------------------------------------------------------------------
# semantic_drift_safe
# ---------------------------------------------------------------------------


def test_semantic_drift_identical_output_ok() -> None:
    packet = make_packet(memory_candidate_text="please send the quarterly report to finance")
    assert semantic_drift_safe(packet, "please send the quarterly report to finance") == (
        True,
        "ok",
    )


def test_semantic_drift_empty_output_rejected() -> None:
    packet = make_packet(memory_candidate_text="anything at all")
    assert semantic_drift_safe(packet, "") == (False, "empty_output")


def test_semantic_drift_large_expansion_rejected() -> None:
    packet = make_packet(memory_candidate_text="send the report to finance")
    bloated = "send the report to finance and also loop in everyone else we have ever met before"
    ok, reason = semantic_drift_safe(packet, bloated)
    assert not ok and reason == "large_unexplained_size_change"


def test_semantic_drift_truncation_shaped_shrink_rejected() -> None:
    source = "please send the quarterly report to the finance team today"
    packet = make_packet(memory_candidate_text=source)
    ok, reason = semantic_drift_safe(packet, "please send the quarterly")
    assert not ok and reason == "large_unexplained_size_change"


def test_semantic_drift_self_correction_shrink_allowed() -> None:
    # Shrinks that keep the corrected tail (not a leading prefix) fall through
    # to the overlap check and pass.
    source = "send it on tuesday scratch that send it on wednesday morning please"
    packet = make_packet(memory_candidate_text=source)
    assert semantic_drift_safe(packet, "send it on wednesday morning please") == (True, "ok")


def test_semantic_drift_content_overlap_floor_rejected() -> None:
    packet = make_packet(memory_candidate_text="the weather is nice today")
    ok, reason = semantic_drift_safe(packet, "send the report immediately")
    assert not ok and reason == "content_word_overlap_below_floor"


def test_semantic_drift_all_content_words_dropped_rejected() -> None:
    packet = make_packet(memory_candidate_text="deploy the staging build")
    ok, reason = semantic_drift_safe(packet, "that is it")
    assert not ok and reason == "content_words_dropped_entirely"


def test_semantic_drift_content_invented_from_stopword_source_rejected() -> None:
    packet = make_packet(memory_candidate_text="yes that is it")
    ok, reason = semantic_drift_safe(packet, "deploy the build")
    assert not ok and reason == "content_words_invented_from_stopword_source"


# ---------------------------------------------------------------------------
# source_words_preserved
# ---------------------------------------------------------------------------


def test_source_words_preserved_non_final_stage_is_skipped() -> None:
    packet = make_packet(stage="live", base_visible_text="keep all of these words")
    assert source_words_preserved(packet, "totally different") == (True, "ok")


def test_source_words_preserved_identical_output_ok() -> None:
    packet = make_packet(memory_candidate_text="keep every single word here")
    assert source_words_preserved(packet, "keep every single word here") == (True, "ok")


def test_source_words_dropped_is_rejected() -> None:
    packet = make_packet(memory_candidate_text="please send the quarterly report tomorrow")
    ok, reason = source_words_preserved(packet, "please send the report tomorrow")
    assert not ok and reason == "source_words_dropped:quarterly"


def test_disfluencies_may_be_dropped() -> None:
    packet = make_packet(memory_candidate_text="um please send it um now")
    assert source_words_preserved(packet, "please send it now") == (True, "ok")


def test_self_correction_cleanup_spans_may_be_dropped() -> None:
    packet = make_packet(
        memory_candidate_text="send it tuesday scratch that wednesday afternoon"
    )
    assert source_words_preserved(packet, "send it wednesday afternoon") == (True, "ok")


def test_equal_length_word_substitution_is_allowed() -> None:
    packet = make_packet(memory_candidate_text="i went their yesterday evening anyway")
    assert source_words_preserved(packet, "i went there yesterday evening anyway") == (True, "ok")


def test_spelled_letters_collapsed_to_acronym_allowed() -> None:
    packet = make_packet(memory_candidate_text="call the a p i endpoint now")
    assert source_words_preserved(packet, "call the api endpoint now") == (True, "ok")


# ---------------------------------------------------------------------------
# validate_adjudication_result end-to-end
# ---------------------------------------------------------------------------


def _happy_packet() -> TranscriptAdjudicationPacket:
    text = "please send the quarterly report to Sarah"
    return make_packet(
        memory_candidate_text=text,
        whisper_text=text,
        raw_text=text,
        protected_terms=("Sarah",),
    )


def test_validate_happy_path() -> None:
    packet = _happy_packet()
    result = make_result("please send the quarterly report to Sarah")
    assert validate_adjudication_result(packet, result) == (True, "ok")


def test_validate_rejects_empty_output() -> None:
    ok, reason = validate_adjudication_result(_happy_packet(), make_result(""))
    assert not ok and reason == "empty_output"


def test_validate_rejects_assistant_preamble() -> None:
    result = make_result("Sure, please send the quarterly report to Sarah")
    ok, reason = validate_adjudication_result(_happy_packet(), result)
    assert not ok and reason == "assistant_preamble"


def test_validate_rejects_markdown_fence() -> None:
    result = make_result("```\nplease send the quarterly report to Sarah\n```")
    ok, reason = validate_adjudication_result(_happy_packet(), result)
    assert not ok and reason == "markdown_fence"


def test_validate_rejects_dropped_protected_term() -> None:
    result = make_result("please send the quarterly report")
    ok, reason = validate_adjudication_result(_happy_packet(), result)
    assert not ok and reason == "protected_term_dropped:Sarah"


def test_validate_rejects_new_number() -> None:
    result = make_result("please send the quarterly report to Sarah at 5")
    ok, reason = validate_adjudication_result(_happy_packet(), result)
    assert not ok and reason == "new_number_without_evidence"


def test_validate_rejects_unsupported_inserted_phrase_op() -> None:
    op = TranscriptPatchOp(
        op="insert",
        start_char=0,
        end_char=0,
        text="entirely fabricated marketing copy",
        reason="asr_correction",
        confidence=0.9,
    )
    result = make_result("please send the quarterly report to Sarah", ops=(op,))
    ok, reason = validate_adjudication_result(_happy_packet(), result)
    assert not ok and reason.startswith("unsupported_inserted_phrase:")


def test_validate_rejects_semantic_drift() -> None:
    bloated = (
        "please send the quarterly report to Sarah and also forward a copy "
        "to everyone else on the entire team right away"
    )
    ok, reason = validate_adjudication_result(
        _happy_packet(), make_result(bloated), allow_chunked_insertions=True
    )
    assert not ok and reason == "large_unexplained_size_change"


def test_validate_rejects_dropped_source_words() -> None:
    result = make_result("please send the report to Sarah")
    ok, reason = validate_adjudication_result(
        _happy_packet(), result, allow_chunked_insertions=True
    )
    assert not ok and reason == "source_words_dropped:quarterly"


def test_validate_rejects_heading_formatting_in_transcript() -> None:
    result = make_result("# Report\nplease send the quarterly report to Sarah")
    ok, reason = validate_adjudication_result(_happy_packet(), result)
    assert not ok and reason == "transcript_heading_formatting"


def test_validate_allows_heading_when_policy_is_none() -> None:
    text = "please send the quarterly report to Sarah"
    packet = make_packet(
        memory_candidate_text=text,
        whisper_text=text,
        raw_text=text,
        protected_terms=("Sarah",),
        transcript_policy="none",
    )
    result = make_result(text)
    assert validate_adjudication_result(packet, result) == (True, "ok")


def test_validate_rejects_live_bullet_formatting() -> None:
    text = "please send the quarterly report"
    packet = make_packet(
        stage="live",
        base_visible_text=text,
        live_preview_text=text,
        memory_candidate_text=text,
        whisper_text=text,
        raw_text=text,
    )
    result = make_result("- please send\n- the quarterly report", stage="live")
    ok, reason = validate_adjudication_result(packet, result)
    assert not ok and reason == "live_structural_formatting"


def test_validate_rejects_low_signal_mid_sentence_capitalization() -> None:
    result = make_result("please send The quarterly report to Sarah")
    ok, reason = validate_adjudication_result(_happy_packet(), result)
    assert not ok and reason == "low_signal_mid_sentence_capitalization:The"


# ---------------------------------------------------------------------------
# repair_low_signal_mid_sentence_capitalization
# ---------------------------------------------------------------------------


def test_repair_lowercases_mid_sentence_stopword() -> None:
    packet = make_packet(memory_candidate_text="i think the plan works for now")
    repaired, repairs = repair_low_signal_mid_sentence_capitalization(
        packet, "I think The plan works for now."
    )
    assert repaired == "I think the plan works for now."
    assert repairs == [{"from": "The", "to": "the"}]


def test_repair_keeps_sentence_initial_capital() -> None:
    packet = make_packet(memory_candidate_text="the plan works. the team agrees")
    repaired, repairs = repair_low_signal_mid_sentence_capitalization(
        packet, "The plan works. The team agrees."
    )
    assert repaired == "The plan works. The team agrees."
    assert repairs == []


def test_repair_keeps_may_before_a_date_number() -> None:
    packet = make_packet(memory_candidate_text="the launch is due on may 18")
    repaired, repairs = repair_low_signal_mid_sentence_capitalization(
        packet, "The launch is due on May 18."
    )
    assert repaired == "The launch is due on May 18."
    assert repairs == []


def test_repair_keeps_protected_terms() -> None:
    packet = make_packet(
        memory_candidate_text="ask will about the rollout",
        protected_terms=("Will",),
    )
    repaired, repairs = repair_low_signal_mid_sentence_capitalization(
        packet, "Ask Will about the rollout."
    )
    assert repaired == "Ask Will about the rollout."
    assert repairs == []


def test_repair_skips_words_not_in_source() -> None:
    packet = make_packet(memory_candidate_text="plan works for now")
    text = "I think The plan works for now."  # "the" never appeared in source
    repaired, repairs = repair_low_signal_mid_sentence_capitalization(packet, text)
    assert repaired == text
    assert repairs == []


def test_repair_is_a_passthrough_for_live_stage() -> None:
    packet = make_packet(stage="live", memory_candidate_text="the plan")
    text = "we like The plan"
    assert repair_low_signal_mid_sentence_capitalization(packet, text) == (text, [])


# ---------------------------------------------------------------------------
# remove_instructional_exclusion_phrases
# ---------------------------------------------------------------------------


def test_remove_exclusion_phrase_with_bullets_preamble() -> None:
    packet = make_packet(memory_candidate_text="meeting notes")
    text = (
        "Add bullets under each section, but do not include the words scratch that "
        "in the final note unless I explicitly say quote scratch that quote. "
        "Here are the notes."
    )
    out, repairs = remove_instructional_exclusion_phrases(packet, text)
    assert out == "Here are the notes."
    assert len(repairs) == 1 and "removed" in repairs[0]


def test_remove_exclusion_phrase_standalone_variant() -> None:
    packet = make_packet(memory_candidate_text="meeting notes")
    text = (
        "Buy milk. Do not include the words scratch that in the final note unless "
        "I explicitly say quote scratch that quote. Buy eggs."
    )
    out, repairs = remove_instructional_exclusion_phrases(packet, text)
    assert out == "Buy milk. Buy eggs."
    assert len(repairs) == 1


def test_remove_exclusion_phrase_no_match_is_unchanged() -> None:
    packet = make_packet(memory_candidate_text="meeting notes")
    text = "Buy milk and eggs."
    assert remove_instructional_exclusion_phrases(packet, text) == (text, [])


def test_remove_exclusion_phrase_passthrough_for_live_stage() -> None:
    packet = make_packet(stage="live")
    text = (
        "do not include the words scratch that in the final note unless I "
        "explicitly say quote scratch that quote."
    )
    assert remove_instructional_exclusion_phrases(packet, text) == (text, [])


# ---------------------------------------------------------------------------
# restore_explicit_final_word_tail
# ---------------------------------------------------------------------------


def test_restore_final_word_tail_appends_missing_sentence() -> None:
    packet = make_packet(
        memory_candidate_text="make a note about apples. At the end say the final word is mango."
    )
    out, meta = restore_explicit_final_word_tail(packet, "Make a note about apples.")
    assert out == "Make a note about apples. At the end say the final word is mango."
    assert meta == {"restored": "At the end say the final word is mango."}


def test_restore_final_word_tail_noop_when_already_present() -> None:
    packet = make_packet(
        memory_candidate_text="at the end say the final word is mango."
    )
    text = "At the end say the final word is mango."
    assert restore_explicit_final_word_tail(packet, text) == (text, None)


def test_restore_final_word_tail_noop_without_cue() -> None:
    packet = make_packet(memory_candidate_text="make a note about apples")
    text = "Make a note about apples."
    assert restore_explicit_final_word_tail(packet, text) == (text, None)


def test_restore_final_word_tail_passthrough_for_live_stage() -> None:
    packet = make_packet(
        stage="live",
        memory_candidate_text="at the end say the final word is mango.",
    )
    assert restore_explicit_final_word_tail(packet, "notes") == ("notes", None)
