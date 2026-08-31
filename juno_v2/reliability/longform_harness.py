"""Reference-owned long-form reliability harness.

This module deliberately does not launch Juno, mutate an editor, or write
product state.  It freezes a case manifest and scores an externally supplied
capture.  A caller that wants real playback must explicitly provide a capture
function and opt in; this keeps scheduled reviews read-only by default.

The capture contract is intentionally stage-shaped.  Mixing a final paste
with a live preview (or treating a short/truncated HUD sample as a full run)
was the source of several misleading reliability reports, so a run is
``NOT_OBSERVABLE`` until both the complete live emission timeline and the
destination verification are present.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import math
import re
import uuid
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


COHORT_MIN_DURATION_SECONDS: dict[str, float] = {
    "normal": 30.0,
    "structure": 30.0,
    "corrections": 30.0,
    "symbols": 30.0,
    "code_switch": 30.0,
    "cafe": 120.0,
    "background_speech": 120.0,
    "sustained_noise": 120.0,
}
COHORT_MAX_DURATION_SECONDS: dict[str, float | None] = {
    "normal": None,
    "structure": None,
    "corrections": None,
    "symbols": None,
    "code_switch": None,
    "cafe": 180.0,
    "background_speech": 180.0,
    "sustained_noise": 180.0,
}
STAGE_ORDER = (
    "live_emission",
    "terminal_live",
    "raw_final_asr",
    "normalization_memory",
    "editor_operations",
    "history_committed",
    "destination",
)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LIVE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp_ms", "ts_unix_ms", "time_ms", "timestamp", "ts"),
    "visible_text": ("visible_text", "text", "emitted_text", "live_text"),
    "committed_span": ("committed_span", "committed_text", "frozen_prefix"),
    "draft_tail": ("draft_tail", "draft_text", "draft"),
    "vad_state": ("vad_state", "vad", "voice_activity", "is_speech"),
    "revision_operation": ("revision_operation", "revision_op", "revision", "operation"),
}


class ManifestError(ValueError):
    """The frozen review manifest is incomplete or unsafe to execute."""


class PlaybackPrerequisiteError(RuntimeError):
    """Real product playback was requested without explicit prerequisites."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _text_tokens(value: str, *, casefold: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(value)
    return [token.casefold() for token in tokens] if casefold else tokens


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (OSError, wave.Error) as exc:
        raise ManifestError(f"{path}: audio must be a readable WAV file ({exc})") from exc
    if rate <= 0:
        raise ManifestError(f"{path}: WAV has invalid sample rate")
    return frames / rate


def _required_text(value: Any, field: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{case_id}: {field} must be a non-empty string")
    return value


def _validate_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ManifestError(f"{field} must be 1–128 ASCII letters, digits, '.', '_', ':', or '-'")
    return value


def _string_tuple(value: Any, field: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ManifestError(f"{case_id}: {field} must contain at least one item")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ManifestError(f"{case_id}: {field} must contain non-empty strings")
    return tuple(value)


@dataclasses.dataclass(frozen=True)
class LongformCase:
    """A human-owned case before it is frozen for playback."""

    case_id: str
    cohort: str
    audio_path: Path
    spoken_reference: str
    expected_output: str
    invariants: tuple[str, ...]
    target_stems: tuple[str, ...]
    distractor_stems: tuple[str, ...]
    language: str
    app: str
    settings: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, base_dir: Path = Path(".")) -> "LongformCase":
        case_id = _validate_id(raw.get("case_id"), "case_id")
        cohort = _required_text(raw.get("cohort"), "cohort", case_id)
        audio_value = _required_text(raw.get("audio_path"), "audio_path", case_id)
        path = Path(audio_value)
        if not path.is_absolute():
            path = base_dir / path
        invariants = _string_tuple(raw.get("invariants"), "invariants", case_id)
        target_stems = _string_tuple(raw.get("target_stems"), "target_stems", case_id)
        distractors = _string_tuple(raw.get("distractor_stems"), "distractor_stems", case_id)
        settings = raw.get("settings")
        if not isinstance(settings, Mapping) or not settings:
            raise ManifestError(f"{case_id}: settings must be a non-empty mapping")
        return cls(
            case_id=case_id,
            cohort=cohort,
            audio_path=path,
            spoken_reference=_required_text(raw.get("spoken_reference"), "spoken_reference", case_id),
            expected_output=_required_text(raw.get("expected_output"), "expected_output", case_id),
            invariants=invariants,
            target_stems=target_stems,
            distractor_stems=distractors,
            language=_required_text(raw.get("language"), "language", case_id),
            app=_required_text(raw.get("app"), "app", case_id),
            settings=dict(settings),
        )


@dataclasses.dataclass(frozen=True)
class FrozenCase:
    """Immutable case contract used for one review run."""

    case_id: str
    cohort: str
    audio_path: str
    audio_sha256: str
    duration_seconds: float
    spoken_reference: str
    expected_output: str
    invariants: tuple[str, ...]
    target_stems: tuple[str, ...]
    distractor_stems: tuple[str, ...]
    language: str
    app: str
    settings: Mapping[str, Any]
    spoken_reference_sha256: str
    expected_output_sha256: str
    settings_sha256: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "invariants": list(self.invariants),
            "target_stems": list(self.target_stems),
            "distractor_stems": list(self.distractor_stems),
            "settings": dict(self.settings),
        }


@dataclasses.dataclass(frozen=True)
class FrozenManifest:
    run_id: str
    cases: tuple[FrozenCase, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "manifest_sha256": self.manifest_sha256, "cases": [c.to_dict() for c in self.cases]}


def freeze_case(case: LongformCase, *, run_id: str) -> FrozenCase:
    _validate_id(case.case_id, "case_id")
    _validate_id(run_id, "run_id")
    if case.cohort not in COHORT_MIN_DURATION_SECONDS:
        raise ManifestError(f"{case.case_id}: unsupported cohort {case.cohort!r}")
    if not case.audio_path.is_file():
        raise ManifestError(f"{case.case_id}: audio does not exist: {case.audio_path}")
    duration = _wav_duration(case.audio_path)
    minimum = COHORT_MIN_DURATION_SECONDS[case.cohort]
    maximum = COHORT_MAX_DURATION_SECONDS[case.cohort]
    if duration < minimum:
        raise ManifestError(f"{case.case_id}: {case.cohort} audio is {duration:.2f}s; minimum is {minimum:.0f}s")
    if maximum is not None and duration > maximum:
        raise ManifestError(f"{case.case_id}: {case.cohort} audio is {duration:.2f}s; maximum is {maximum:.0f}s")
    raw_audio = case.audio_path.read_bytes()
    return FrozenCase(
        case_id=case.case_id,
        cohort=case.cohort,
        audio_path=str(case.audio_path.resolve()),
        audio_sha256=_sha256_bytes(raw_audio),
        duration_seconds=duration,
        spoken_reference=case.spoken_reference,
        expected_output=case.expected_output,
        invariants=case.invariants,
        target_stems=case.target_stems,
        distractor_stems=case.distractor_stems,
        language=case.language,
        app=case.app,
        settings=dict(case.settings),
        spoken_reference_sha256=_sha256_bytes(case.spoken_reference.encode()),
        expected_output_sha256=_sha256_bytes(case.expected_output.encode()),
        settings_sha256=_sha256_json(case.settings),
        run_id=run_id,
    )


def freeze_manifest(cases: Iterable[LongformCase], *, run_id: str | None = None) -> FrozenManifest:
    run_id = run_id or f"longform-{uuid.uuid4().hex}"
    _validate_id(run_id, "run_id")
    case_list = tuple(freeze_case(case, run_id=run_id) for case in cases)
    if not case_list:
        raise ManifestError("manifest must contain at least one case")
    ids = [case.case_id for case in case_list]
    if len(set(ids)) != len(ids):
        raise ManifestError("case_id values must be unique")
    payload = [case.to_dict() for case in case_list]
    return FrozenManifest(run_id=run_id, cases=case_list, manifest_sha256=_sha256_json(payload))


def build_manifest(raw_cases: Iterable[Mapping[str, Any]], *, base_dir: Path = Path("."), run_id: str | None = None) -> FrozenManifest:
    return freeze_manifest((LongformCase.from_dict(raw, base_dir=base_dir) for raw in raw_cases), run_id=run_id)


def load_manifest(path: Path, *, run_id: str | None = None) -> FrozenManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, Mapping) else None
    if not isinstance(raw_cases, list):
        raise ManifestError("manifest JSON must contain a cases list")
    return build_manifest(raw_cases, base_dir=path.parent, run_id=run_id or payload.get("run_id"))


@dataclasses.dataclass(frozen=True)
class StageEvidence:
    """Separate capture outputs; absent values are not silently inferred."""

    run_id: str | None = None
    case_id: str | None = None
    live_emission_timeline: tuple[Mapping[str, Any], ...] | None = None
    terminal_live_text: str | None = None
    raw_final_asr: str | None = None
    normalization_memory: Any = None
    editor_operations: tuple[Mapping[str, Any], ...] | None = None
    editor_output_text: str | None = None
    editor_operations_complete: bool = False
    history_committed_text: str | None = None
    destination_text: str | None = None
    live_telemetry_complete: bool = False
    live_telemetry_truncated: bool = False
    destination_verified: bool = False
    invariant_results: Mapping[str, bool] | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StageEvidence":
        timeline = raw.get("live_emission_timeline")
        operations = raw.get("editor_operations")
        return cls(
            run_id=raw.get("run_id"),
            case_id=raw.get("case_id"),
            live_emission_timeline=tuple(timeline) if isinstance(timeline, list) else timeline,
            terminal_live_text=raw.get("terminal_live_text"),
            raw_final_asr=raw.get("raw_final_asr"),
            normalization_memory=raw.get("normalization_memory"),
            editor_operations=tuple(operations) if isinstance(operations, list) else operations,
            editor_output_text=raw.get("editor_output_text"),
            editor_operations_complete=raw.get("editor_operations_complete", False) is True,
            history_committed_text=raw.get("history_committed_text"),
            destination_text=raw.get("destination_text"),
            live_telemetry_complete=raw.get("live_telemetry_complete", False) is True,
            live_telemetry_truncated=raw.get("live_telemetry_truncated", False) is True,
            destination_verified=raw.get("destination_verified", False) is True,
            invariant_results=raw.get("invariant_results"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "live_emission_timeline": list(self.live_emission_timeline) if self.live_emission_timeline is not None else None,
            "editor_operations": list(self.editor_operations) if self.editor_operations is not None else None,
        }


def _timeline_text(timeline: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not timeline:
        return None
    for event in reversed(timeline):
        for key in ("text", "visible_text", "emitted_text", "live_text"):
            value = event.get(key)
            if isinstance(value, str):
                return value
    return None


def _normalization_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "output", "normalized_text", "final_text"):
            if isinstance(value.get(key), str):
                return value[key]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in reversed(value):
            result = _normalization_text(item)
            if result is not None:
                return result
    return None


def _editor_text(operations: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not operations:
        return None
    for operation in reversed(operations):
        for key in ("result_text", "document_text", "text_after", "destination_text"):
            value = operation.get(key)
            if isinstance(value, str):
                return value
    return None


def _stage_values(evidence: StageEvidence) -> dict[str, str | None]:
    return {
        "live_emission": _timeline_text(evidence.live_emission_timeline),
        "terminal_live": evidence.terminal_live_text,
        "raw_final_asr": evidence.raw_final_asr,
        "normalization_memory": _normalization_text(evidence.normalization_memory),
        "editor_operations": evidence.editor_output_text if evidence.editor_output_text is not None else _editor_text(evidence.editor_operations),
        "history_committed": evidence.history_committed_text,
        "destination": evidence.destination_text,
    }


def _live_field(event: Mapping[str, Any], field: str) -> tuple[bool, Any]:
    for alias in _LIVE_FIELD_ALIASES[field]:
        if alias in event:
            return True, event[alias]
    return False, None


def _validate_live_timeline(timeline: Sequence[Mapping[str, Any]] | None) -> tuple[str, ...]:
    """Validate the complete per-emission observability contract.

    ``live_telemetry_complete`` is only an attestation from the capture
    adapter; it cannot waive this structural validation.  Every emission must
    identify when it happened, what was fully visible, the committed span,
    draft tail, VAD state, and the revision operation that produced it.
    """
    if timeline is None:
        return ("live_timeline_missing",)
    if not timeline:
        return ("live_timeline_empty",)
    failures: list[str] = []
    previous_seq: int | None = None
    previous_timestamp: float | None = None
    for index, event in enumerate(timeline):
        prefix = f"live_timeline_event_{index}"
        if not isinstance(event, Mapping):
            failures.append(f"{prefix}_not_mapping")
            continue
        sequence = event.get("seq")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            failures.append(f"{prefix}_invalid_seq")
        elif previous_seq is not None and sequence <= previous_seq:
            failures.append(f"{prefix}_non_monotonic_seq")
        if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence > 0:
            previous_seq = sequence

        has_timestamp, timestamp = _live_field(event, "timestamp")
        if not has_timestamp:
            failures.append(f"{prefix}_missing_timestamp")
        elif isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            failures.append(f"{prefix}_invalid_timestamp")
        elif previous_timestamp is not None and timestamp < previous_timestamp:
            failures.append(f"{prefix}_non_monotonic_timestamp")
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and math.isfinite(timestamp):
            previous_timestamp = float(timestamp)

        for field in ("visible_text", "committed_span", "draft_tail", "vad_state", "revision_operation"):
            present, value = _live_field(event, field)
            if not present:
                failures.append(f"{prefix}_missing_{field}")
            elif value is None:
                failures.append(f"{prefix}_invalid_{field}")
            elif field == "visible_text" and not isinstance(value, str):
                failures.append(f"{prefix}_invalid_{field}")
    return tuple(failures)


def _score_text(expected: str, actual: str | None) -> dict[str, Any]:
    if actual is None:
        return {
            "observed": False,
            "exact_match": False,
            "exact_surface_match": False,
            "lexical_exact_match": False,
            "score": None,
            "surface_score": None,
            "missing_tokens": [],
            "extra_tokens": [],
            "missing_tail": [],
            "semantic_replacement_spans": [],
        }
    expected_tokens = _text_tokens(expected)
    actual_tokens = _text_tokens(actual)
    expected_surface_tokens = _text_tokens(expected, casefold=False)
    actual_surface_tokens = _text_tokens(actual, casefold=False)
    matcher = difflib.SequenceMatcher(a=expected_tokens, b=actual_tokens, autojunk=False)
    missing: list[str] = []
    extra: list[str] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            missing.extend(expected_tokens[left_start:left_end])
        if tag in {"insert", "replace"}:
            extra.extend(actual_tokens[right_start:right_end])
    surface_matcher = difflib.SequenceMatcher(a=expected_surface_tokens, b=actual_surface_tokens, autojunk=False)
    replacement_spans: list[dict[str, Any]] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "replace":
            replacement_spans.append({
                "expected": expected_tokens[left_start:left_end],
                "actual": actual_tokens[right_start:right_end],
            })
    common_prefix = 0
    for expected_token, actual_token in zip(expected_tokens, actual_tokens):
        if expected_token != actual_token:
            break
        common_prefix += 1
    missing_tail = expected_tokens[common_prefix:] if len(actual_tokens) <= common_prefix else []
    return {
        "observed": True,
        "exact_match": expected == actual,
        "exact_surface_match": expected == actual,
        "lexical_exact_match": expected_tokens == actual_tokens,
        "score": round(matcher.ratio(), 4),
        "surface_score": round(surface_matcher.ratio(), 4),
        "missing_tokens": missing,
        "extra_tokens": extra,
        "missing_tail": missing_tail,
        "semantic_replacement_spans": replacement_spans,
    }


@dataclasses.dataclass(frozen=True)
class CaseResult:
    case_id: str
    cohort: str
    status: str
    first_divergent_stage: str | None
    stages: Mapping[str, Mapping[str, Any]]
    invariants: Mapping[str, bool | None]
    diagnostics: Mapping[str, Any]
    observability_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"observability_failures": list(self.observability_failures)}


def _stage_expected(case: FrozenCase) -> dict[str, str]:
    return {
        "live_emission": case.spoken_reference,
        "terminal_live": case.spoken_reference,
        "raw_final_asr": case.spoken_reference,
        "normalization_memory": case.expected_output,
        "editor_operations": case.expected_output,
        "history_committed": case.expected_output,
        "destination": case.expected_output,
    }


def _prefix_regressions(timeline: Sequence[Mapping[str, Any]] | None, *, expected: str) -> list[dict[str, Any]]:
    """Find words that were correct in one full emission and later regressed."""
    if not timeline:
        return []
    expected_tokens = _text_tokens(expected)
    snapshots: list[tuple[int | None, str, list[str]]] = []
    for event in timeline:
        text = next((event.get(key) for key in ("text", "visible_text", "emitted_text", "live_text") if isinstance(event.get(key), str)), None)
        if text is None:
            continue
        snapshots.append((event.get("seq"), text, _text_tokens(text)))
    regressions: list[dict[str, Any]] = []
    for prior_index, (prior_seq, prior_text, prior_tokens) in enumerate(snapshots):
        for token_index, expected_token in enumerate(expected_tokens):
            if token_index >= len(prior_tokens) or prior_tokens[token_index] != expected_token:
                continue
            for later_seq, later_text, later_tokens in snapshots[prior_index + 1 :]:
                if token_index >= len(later_tokens) or later_tokens[token_index] != expected_token:
                    regressions.append({
                        "token_index": token_index,
                        "expected": expected_token,
                        "prior_seq": prior_seq,
                        "prior_text": prior_text,
                        "later_seq": later_seq,
                        "later_text": later_text,
                    })
                    break
    return regressions


def _committed_prefix_regressions(timeline: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not timeline:
        return []
    snapshots: list[tuple[int | None, str, str]] = []
    for event in timeline:
        for key in ("committed_text", "frozen_prefix"):
            value = event.get(key)
            if isinstance(value, str):
                snapshots.append((event.get("seq"), key, value))
                break
    regressions: list[dict[str, Any]] = []
    for (prior_seq, key, prior), (later_seq, _, later) in zip(snapshots, snapshots[1:]):
        if prior and not later.startswith(prior):
            regressions.append({"field": key, "prior_seq": prior_seq, "later_seq": later_seq, "prior_text": prior, "later_text": later})
    return regressions


def _stem_hits(values: Mapping[str, str | None], stems: Sequence[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for stage, value in values.items():
        if not value:
            continue
        folded = value.casefold()
        for stem in stems:
            # Stems are lexical distractors, not arbitrary substrings.  Word
            # boundaries prevent a stem such as ``art`` from flagging
            # ``partial`` while still allowing multi-word stems.
            pattern = rf"(?<!\w){re.escape(stem.casefold())}(?!\w)"
            if re.search(pattern, folded):
                hits.append({"stage": stage, "stem": stem})
    return hits


def _diagnostics(case: FrozenCase, evidence: StageEvidence, values: Mapping[str, str | None], stages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing_tails = [{"stage": stage, "tokens": result["missing_tail"]} for stage, result in stages.items() if result["missing_tail"]]
    replacements = [{"stage": stage, "spans": result["semantic_replacement_spans"]} for stage, result in stages.items() if result["semantic_replacement_spans"]]
    hallucinated = [{"stage": stage, "tokens": result["extra_tokens"]} for stage, result in stages.items() if result["extra_tokens"]]
    return {
        "correct_then_regressed_live_words": _prefix_regressions(evidence.live_emission_timeline, expected=case.spoken_reference),
        "committed_or_frozen_prefix_regressions": _committed_prefix_regressions(evidence.live_emission_timeline),
        "hallucinated_or_extra_tokens": hallucinated,
        "distractor_or_background_leakage": _stem_hits(values, case.distractor_stems),
        "missing_tails": missing_tails,
        "semantic_replacement_spans": replacements,
    }


def score_case(case: FrozenCase, evidence: StageEvidence | Mapping[str, Any]) -> CaseResult:
    if isinstance(evidence, Mapping):
        evidence = StageEvidence.from_mapping(evidence)
    values = _stage_values(evidence)
    expected_by_stage = _stage_expected(case)
    stages = {stage: _score_text(expected_by_stage[stage], values[stage]) for stage in STAGE_ORDER}
    failures: list[str] = []
    if evidence.run_id is None or evidence.case_id is None:
        failures.append("provenance_missing")
    else:
        try:
            _validate_id(evidence.run_id, "run_id")
            _validate_id(evidence.case_id, "case_id")
        except ManifestError:
            failures.append("provenance_invalid")
        if evidence.run_id != case.run_id or evidence.case_id != case.case_id:
            failures.append("provenance_mismatch")
    if evidence.live_emission_timeline is None or not evidence.live_telemetry_complete:
        failures.append("live_emission_telemetry_incomplete")
    failures.extend(_validate_live_timeline(evidence.live_emission_timeline))
    if evidence.live_telemetry_truncated:
        failures.append("live_emission_telemetry_truncated")
    if evidence.destination_text is None or not evidence.destination_verified:
        failures.append("destination_unverified")
    # Every intermediate artifact is mandatory for an auditable E2E result.
    # In particular, a live+paste pair cannot prove that normalization or an
    # editor operation did not introduce the observed final text.
    for stage in STAGE_ORDER:
        if not stages[stage]["observed"]:
            failures.append(f"{stage}_missing")
    if not evidence.editor_operations_complete:
        failures.append("editor_operations_incomplete")
    if evidence.invariant_results is None:
        invariant_results: dict[str, bool | None] = {name: None for name in case.invariants}
        failures.append("invariants_missing")
    else:
        invariant_results = {name: evidence.invariant_results.get(name) for name in case.invariants}
        failures.extend(f"invariant_failed:{name}" for name, passed in invariant_results.items() if passed is not True)
    # A stage is divergent only when it was actually observed. Missing data is
    # an observability failure, never evidence that the stage was correct.
    first_divergence = next((stage for stage in STAGE_ORDER if stages[stage]["observed"] and not stages[stage]["exact_match"]), None)
    observability_failures = tuple(failure for failure in failures if not failure.startswith("invariant_failed:"))
    status = "NOT_OBSERVABLE" if observability_failures else ("PASS" if first_divergence is None and not failures else "FAIL")
    return CaseResult(case.case_id, case.cohort, status, first_divergence, stages, invariant_results, _diagnostics(case, evidence, values, stages), observability_failures)


def score_report(manifest: FrozenManifest, captures: Mapping[str, StageEvidence | Mapping[str, Any]]) -> dict[str, Any]:
    results = [score_case(case, captures[case.case_id]) if case.case_id in captures else CaseResult(case.case_id, case.cohort, "NOT_OBSERVABLE", None, {stage: _score_text(_stage_expected(case)[stage], None) for stage in STAGE_ORDER}, {name: None for name in case.invariants}, {}, ("capture_missing",)) for case in manifest.cases]
    cohorts: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        cohorts[result.cohort].append(result)

    def cohort_report(items: list[CaseResult]) -> dict[str, Any]:
        observed = [item for item in items if item.status != "NOT_OBSERVABLE"]
        return {
            "case_count": len(items),
            "status_counts": {status: sum(item.status == status for item in items) for status in ("PASS", "FAIL", "NOT_OBSERVABLE")},
            "mean_stage_scores": {stage: round(sum((item.stages[stage]["score"] or 0.0) for item in observed) / len(observed), 4) if observed else None for stage in STAGE_ORDER},
            "first_divergent_stage_counts": {stage: sum(item.first_divergent_stage == stage for item in items) for stage in STAGE_ORDER},
            "results": [item.to_dict() for item in items],
        }

    return {"run_id": manifest.run_id, "manifest_sha256": manifest.manifest_sha256, "cohorts": {cohort: cohort_report(items) for cohort, items in sorted(cohorts.items())}, "cases": [result.to_dict() for result in results]}


def _assert_frozen_case_unchanged(case: FrozenCase) -> None:
    """Fail closed if an input changed after the manifest was frozen."""
    path = Path(case.audio_path)
    if not path.is_file() or _sha256_bytes(path.read_bytes()) != case.audio_sha256:
        raise PlaybackPrerequisiteError(f"{case.case_id}: audio changed after manifest freeze")
    if _sha256_bytes(case.spoken_reference.encode()) != case.spoken_reference_sha256:
        raise PlaybackPrerequisiteError(f"{case.case_id}: spoken reference changed after manifest freeze")
    if _sha256_bytes(case.expected_output.encode()) != case.expected_output_sha256:
        raise PlaybackPrerequisiteError(f"{case.case_id}: expected output changed after manifest freeze")
    if _sha256_json(case.settings) != case.settings_sha256:
        raise PlaybackPrerequisiteError(f"{case.case_id}: settings changed after manifest freeze")


def execute_manifest(manifest: FrozenManifest, *, capture: Callable[[FrozenCase], StageEvidence | Mapping[str, Any]] | None = None, allow_product_playback: bool = False) -> dict[str, Any]:
    """Capture and score a frozen run.

    ``capture`` is injected by an explicitly configured product runner.  The
    harness itself has no permission to click, paste, or alter History.  This
    function raises instead of producing a misleading empty report when real
    playback prerequisites are absent.
    """
    if capture is None or not allow_product_playback:
        raise PlaybackPrerequisiteError("explicit capture callback and allow_product_playback=True are required")
    expected_manifest_hash = _sha256_json([case.to_dict() for case in manifest.cases])
    if expected_manifest_hash != manifest.manifest_sha256:
        raise PlaybackPrerequisiteError("frozen manifest contents changed before playback")
    captures: dict[str, StageEvidence | Mapping[str, Any]] = {}
    for case in manifest.cases:
        _assert_frozen_case_unchanged(case)
        captures[case.case_id] = capture(case)
    return score_report(manifest, captures)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a frozen Juno long-form reliability manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="freeze and validate; never play audio")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = load_manifest(args.manifest)
    payload = manifest.to_dict()
    if args.validate_only:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        raise PlaybackPrerequisiteError("playback is intentionally not implicit; inject a reviewed capture adapter")
    if args.output:
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
