from __future__ import annotations

import wave
from pathlib import Path

import pytest

from juno_v2.reliability.longform_harness import (
    ManifestError,
    PlaybackPrerequisiteError,
    StageEvidence,
    build_manifest,
    execute_manifest,
    score_case,
    score_report,
)


def _wav(path: Path, seconds: float) -> Path:
    frames = int(16_000 * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * frames)
    return path


def _raw(path: Path, *, cohort: str = "normal") -> dict:
    return {
        "case_id": f"case-{cohort}",
        "cohort": cohort,
        "audio_path": str(path),
        "spoken_reference": "Please keep this complete sentence.",
        "expected_output": "Please keep this complete sentence.",
        "invariants": ["no_hallucinations", "no_silent_loss"],
        "target_stems": ["complete", "sentence"],
        "distractor_stems": ["completion"],
        "language": "en",
        "app": "TextEdit",
        "settings": {"preview": "small", "final": "medium", "memory": False},
    }


def _evidence(text: str = "Please keep this complete sentence.", *, run_id: str = "run", case_id: str = "case-normal", **overrides):
    data = {
        "run_id": run_id,
        "case_id": case_id,
        "live_emission_timeline": [{
            "seq": 1,
            "timestamp_ms": 1_000,
            "text": text,
            "committed_span": "",
            "draft_tail": text,
            "vad_state": "speech",
            "revision_operation": "replace",
        }],
        "terminal_live_text": text,
        "raw_final_asr": text,
        "normalization_memory": {"text": text, "memory_used": False},
        "editor_operations": [{"kind": "replace", "text_after": text}],
        "editor_output_text": text,
        "editor_operations_complete": True,
        "history_committed_text": text,
        "destination_text": text,
        "live_telemetry_complete": True,
        "destination_verified": True,
        "invariant_results": {"no_hallucinations": True, "no_silent_loss": True},
    }
    data.update(overrides)
    return data


def test_manifest_freezes_hashes_settings_and_longform_duration(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "normal.wav", 30.25)
    manifest = build_manifest([_raw(audio)], base_dir=tmp_path, run_id="review-test")
    case = manifest.cases[0]

    assert manifest.run_id == "review-test"
    assert case.duration_seconds >= 30
    assert len(case.audio_sha256) == 64
    assert len(case.spoken_reference_sha256) == 64
    assert len(case.expected_output_sha256) == 64
    assert len(case.settings_sha256) == 64
    assert case.run_id == manifest.run_id


def test_live_stages_use_spoken_reference_and_exact_surface_is_required(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "normal.wav", 30.0)
    raw = _raw(audio)
    raw["spoken_reference"] = "I said slash, not dash."
    raw["expected_output"] = "I said /, not dash."
    manifest = build_manifest([raw], run_id="run")
    case = manifest.cases[0]
    result = score_case(case, _evidence(text="I said slash, not dash."))

    assert result.status == "FAIL"
    assert result.first_divergent_stage == "normalization_memory"
    assert result.stages["live_emission"]["lexical_exact_match"] is True
    assert result.stages["live_emission"]["exact_surface_match"] is True
    assert result.stages["normalization_memory"]["lexical_exact_match"] is False

    casefold_only = score_case(case, _evidence(text="i said slash, not dash."))
    assert casefold_only.stages["live_emission"]["lexical_exact_match"] is True
    assert casefold_only.stages["live_emission"]["exact_surface_match"] is False
    assert casefold_only.status == "FAIL"


def test_manifest_rejects_short_clean_and_noise_cases(tmp_path: Path) -> None:
    short = _wav(tmp_path / "short.wav", 29.99)
    with pytest.raises(ManifestError, match="minimum"):
        build_manifest([_raw(short)], base_dir=tmp_path)

    cafe = _wav(tmp_path / "cafe.wav", 119.99)
    with pytest.raises(ManifestError, match="minimum"):
        build_manifest([_raw(cafe, cohort="cafe")], base_dir=tmp_path)


def test_manifest_rejects_overlong_noisy_case(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "cafe.wav", 180.01)
    with pytest.raises(ManifestError, match="maximum"):
        build_manifest([_raw(audio, cohort="cafe")], base_dir=tmp_path)


def test_manifest_requires_target_and_distractor_stems(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "normal.wav", 30.0)
    raw = _raw(audio)
    raw["distractor_stems"] = []
    with pytest.raises(ManifestError, match="distractor_stems"):
        build_manifest([raw], base_dir=tmp_path)


def test_not_observable_when_hud_is_missing_or_truncated(tmp_path: Path) -> None:
    case = build_manifest([_raw(_wav(tmp_path / "normal.wav", 30.0))], run_id="run").cases[0]

    missing = score_case(case, _evidence(live_emission_timeline=None, live_telemetry_complete=False))
    assert missing.status == "NOT_OBSERVABLE"
    assert "live_emission_telemetry_incomplete" in missing.observability_failures

    truncated = score_case(case, _evidence(live_telemetry_truncated=True))
    assert truncated.status == "NOT_OBSERVABLE"
    assert "live_emission_telemetry_truncated" in truncated.observability_failures


def test_seq_and_text_only_live_events_are_not_observable(tmp_path: Path) -> None:
    case = build_manifest([_raw(_wav(tmp_path / "normal.wav", 30.0))], run_id="run").cases[0]
    result = score_case(case, _evidence(live_emission_timeline=[{"seq": 1, "text": "Please keep this complete sentence."}]))

    assert result.status == "NOT_OBSERVABLE"
    assert "live_timeline_event_0_missing_timestamp" in result.observability_failures
    assert "live_timeline_event_0_missing_committed_span" in result.observability_failures
    assert "live_timeline_event_0_missing_draft_tail" in result.observability_failures
    assert "live_timeline_event_0_missing_vad_state" in result.observability_failures
    assert "live_timeline_event_0_missing_revision_operation" in result.observability_failures


def test_live_event_sequence_and_timestamp_must_be_monotonic(tmp_path: Path) -> None:
    case = build_manifest([_raw(_wav(tmp_path / "normal.wav", 30.0))], run_id="run").cases[0]
    timeline = [
        {"seq": 2, "timestamp_ms": 2_000, "text": "Please", "committed_span": "", "draft_tail": "Please", "vad_state": "speech", "revision_operation": "replace"},
        {"seq": 1, "timestamp_ms": 1_000, "text": "Please keep", "committed_span": "", "draft_tail": "Please keep", "vad_state": "speech", "revision_operation": "replace"},
    ]
    result = score_case(case, _evidence(live_emission_timeline=timeline))
    assert result.status == "NOT_OBSERVABLE"
    assert "live_timeline_event_1_non_monotonic_seq" in result.observability_failures
    assert "live_timeline_event_1_non_monotonic_timestamp" in result.observability_failures


def test_not_observable_when_destination_is_not_verified(tmp_path: Path) -> None:
    case = build_manifest([_raw(_wav(tmp_path / "normal.wav", 30.0))], run_id="run").cases[0]
    result = score_case(case, _evidence(destination_verified=False))
    assert result.status == "NOT_OBSERVABLE"
    assert "destination_unverified" in result.observability_failures


def test_zero_editor_operations_are_observed_only_with_explicit_output_and_complete_flag(tmp_path: Path) -> None:
    case = build_manifest([_raw(_wav(tmp_path / "normal.wav", 30.0))], run_id="run").cases[0]
    result = score_case(case, _evidence(editor_operations=[], editor_output_text="Please keep this complete sentence.", editor_operations_complete=True))
    assert result.stages["editor_operations"]["observed"] is True
    assert result.status == "PASS"

    incomplete = score_case(case, _evidence(editor_operations=[], editor_output_text="Please keep this complete sentence.", editor_operations_complete=False))
    assert incomplete.status == "NOT_OBSERVABLE"
    assert "editor_operations_incomplete" in incomplete.observability_failures


def test_provenance_is_bounded_and_must_match_the_frozen_case(tmp_path: Path) -> None:
    case = build_manifest([_raw(_wav(tmp_path / "normal.wav", 30.0))], run_id="run").cases[0]
    mismatch = score_case(case, _evidence(case_id="other-case"))
    assert mismatch.status == "NOT_OBSERVABLE"
    assert "provenance_mismatch" in mismatch.observability_failures

    with pytest.raises(ManifestError, match="1–128"):
        build_manifest([_raw(_wav(tmp_path / "bad.wav", 30.0))], run_id="contains spaces")


def test_hashes_are_rechecked_immediately_before_capture(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "normal.wav", 30.0)
    manifest = build_manifest([_raw(audio)], run_id="run")

    audio.write_bytes(audio.read_bytes() + b"changed")

    with pytest.raises(PlaybackPrerequisiteError, match="audio changed"):
        execute_manifest(manifest, capture=lambda case: _evidence(), allow_product_playback=True)


def test_live_and_prefix_regressions_and_other_diagnostics_are_reported(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "normal.wav", 30.0)
    raw = _raw(audio)
    raw["distractor_stems"] = ["background", "chippy"]
    raw["spoken_reference"] = "I said slash and keep the final tail."
    raw["expected_output"] = "I said / and keep the final tail."
    case = build_manifest([raw], run_id="run").cases[0]
    evidence = _evidence(
        text="I said slash and keep",
        terminal_live_text="I said slash and keep the final tail.",
        raw_final_asr="I said slash and keep the background tail.",
        normalization_memory={"text": "I said / and keep the final"},
        editor_output_text="I said / and keep the final tail.",
        history_committed_text="I said / and keep the final tail.",
        destination_text="I said / and keep the final tail.",
        live_emission_timeline=[
            {"seq": 1, "timestamp_ms": 1_000, "text": "I said slash and keep the final tail.", "committed_text": "I said slash", "draft_tail": "and keep the final tail.", "vad_state": "speech", "revision_operation": "replace"},
            {"seq": 2, "timestamp_ms": 2_000, "text": "I said slash and keep", "committed_text": "I said", "draft_tail": "and keep", "vad_state": "speech", "revision_operation": "replace"},
        ],
    )
    result = score_case(case, evidence)
    diagnostics = result.diagnostics
    assert diagnostics["correct_then_regressed_live_words"]
    assert diagnostics["committed_or_frozen_prefix_regressions"]
    assert diagnostics["hallucinated_or_extra_tokens"]
    assert diagnostics["distractor_or_background_leakage"]
    assert diagnostics["missing_tails"]
    assert diagnostics["semantic_replacement_spans"]


def test_scores_separate_stages_and_finds_first_divergence(tmp_path: Path) -> None:
    case = build_manifest([_raw(_wav(tmp_path / "normal.wav", 30.0))], run_id="run").cases[0]
    result = score_case(case, _evidence(raw_final_asr="Please keep this broken sentence."))
    assert result.status == "FAIL"
    assert result.first_divergent_stage == "raw_final_asr"
    assert result.stages["live_emission"]["exact_match"] is True
    assert result.stages["destination"]["exact_match"] is True
    assert result.stages["raw_final_asr"]["exact_match"] is False


def test_report_is_per_cohort_and_does_not_blend_observed_scores(tmp_path: Path) -> None:
    clean_audio = _wav(tmp_path / "normal.wav", 30.0)
    noisy_audio = _wav(tmp_path / "cafe.wav", 120.0)
    manifest = build_manifest([_raw(clean_audio), _raw(noisy_audio, cohort="cafe")], base_dir=tmp_path, run_id="run")
    report = score_report(manifest, {"case-normal": _evidence(), "case-cafe": _evidence(case_id="case-cafe", live_telemetry_complete=False)})

    assert set(report["cohorts"]) == {"normal", "cafe"}
    assert report["cohorts"]["normal"]["status_counts"]["PASS"] == 1
    assert report["cohorts"]["cafe"]["status_counts"]["NOT_OBSERVABLE"] == 1
    assert report["cohorts"]["cafe"]["mean_stage_scores"]["destination"] is None


def test_playback_requires_explicit_injected_prerequisites(tmp_path: Path) -> None:
    manifest = build_manifest([_raw(_wav(tmp_path / "normal.wav", 30.0))], run_id="run")
    with pytest.raises(PlaybackPrerequisiteError):
        execute_manifest(manifest)
    with pytest.raises(PlaybackPrerequisiteError):
        execute_manifest(manifest, capture=lambda case: _evidence(), allow_product_playback=False)
