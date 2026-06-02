from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

from juno_v2.final.eval import compute_quality_report


@dataclass(slots=True)
class BenchmarkCase:
    utterance_id: str
    reference_text: str
    language: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkSuiteReport:
    coverage: dict[str, bool] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float | None] = field(default_factory=dict)
    comparison: dict[str, float | str | None] = field(default_factory=dict)
    outstanding_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'coverage': dict(self.coverage),
            'counters': dict(self.counters),
            'rates': dict(self.rates),
            'quality': dict(self.quality),
            'comparison': dict(self.comparison),
            'outstanding_gaps': list(self.outstanding_gaps),
        }


def build_benchmark_suite_report(
    summary_payload: dict,
    *,
    cases: Iterable[BenchmarkCase] = (),
    comparison_summary_payload: dict | None = None,
) -> BenchmarkSuiteReport:
    metadata = summary_payload.get('metadata', {})
    runtime_truth = metadata.get('runtime_truth', {})
    utterance_records = metadata.get('utterance_records', []) or []
    memory_snapshot = metadata.get('memory_snapshot') or {}
    report = BenchmarkSuiteReport()
    cases = list(cases)
    reference_by_id = {c.utterance_id: c for c in cases}
    case_tags = {tag for case in cases for tag in case.tags}
    report.coverage = {
        'has_runtime_truth': bool(runtime_truth),
        'has_utterance_records': bool(utterance_records),
        'has_memory_snapshot': bool(memory_snapshot),
        'has_language_metadata': bool(metadata.get('requested_language_counts') or metadata.get('observed_language_counts')),
        'engine_mode_marked': bool(metadata.get('engine_mode')),
        'has_preview_quality_metrics': bool((runtime_truth.get('rates') or {}).get('preview_duplicate_rate') is not None),
        'has_curated_multilingual_cases': 'multilingual' in case_tags,
        'has_memory_reference_cases': 'memory' in case_tags,
        'has_baseline_vs_streaming_comparison': comparison_summary_payload is not None,
    }
    report.counters = {
        'utterance_count': int(summary_payload.get('utterance_count', 0)),
        'recorded_utterance_count': len(utterance_records),
        'memory_correction_pair_count': len((memory_snapshot or {}).get('corrections', []) or []),
        'memory_session_entity_count': len((memory_snapshot or {}).get('session_entities', []) or []),
        'reference_case_count': len(cases),
        'multilingual_case_count': sum('multilingual' in c.tags for c in cases),
        'memory_case_count': sum('memory' in c.tags for c in cases),
    }

    raw_changed = 0
    committed_changed = 0
    context_used = 0
    writer_actions = 0
    language_matches = 0
    references_scored = 0
    memory_helped = 0
    queue_metrics_present = 0
    worker_metrics_present = 0
    backpressure_samples = []
    service_samples = []
    wers = []
    cers = []

    for record in utterance_records:
        raw_text = (record.get('raw_text') or '').strip()
        final_text = (record.get('final_text') or '').strip()
        committed_text = (record.get('committed_text') or '').strip()
        if raw_text and final_text and raw_text != final_text:
            raw_changed += 1
        if final_text and committed_text and final_text != committed_text:
            committed_changed += 1
        if int(record.get('context_candidate_count', 0)) > 0 or int(record.get('bias_phrase_count', 0)) > 0:
            context_used += 1
        if record.get('writer_action') not in (None, '', 'noop', 'pass_through_commit'):
            writer_actions += 1
        requested = record.get('requested_language')
        observed = record.get('observed_language')
        if requested and observed and requested == observed:
            language_matches += 1
        queue_wait_ms = record.get('queue_wait_ms')
        worker_service_ms = record.get('worker_service_ms')
        if queue_wait_ms is not None:
            queue_metrics_present += 1
            backpressure_samples.append(float(queue_wait_ms))
        if worker_service_ms is not None:
            worker_metrics_present += 1
            service_samples.append(float(worker_service_ms))
        if raw_text and final_text and record.get('normalization_applied_count', 0) > 0:
            memory_helped += 1
        case = reference_by_id.get(record.get('utterance_id'))
        if case is not None and final_text:
            references_scored += 1
            quality = compute_quality_report(case.reference_text, final_text)
            wers.append(quality.word_error_rate)
            cers.append(quality.char_error_rate)

    denom = max(1, len(utterance_records))
    report.rates = {
        'normalization_change_rate': raw_changed / denom,
        'commit_rewrite_rate': committed_changed / denom,
        'context_used_rate': context_used / denom,
        'writer_action_rate': writer_actions / denom,
        'language_match_rate': language_matches / max(1, sum(1 for r in utterance_records if r.get('requested_language') and r.get('observed_language'))),
        'memory_helped_rate': memory_helped / denom,
        'queue_metric_coverage_rate': queue_metrics_present / denom,
        'worker_metric_coverage_rate': worker_metrics_present / denom,
        'avg_queue_wait_ms': mean(backpressure_samples) if backpressure_samples else 0.0,
        'avg_worker_service_ms': mean(service_samples) if service_samples else 0.0,
    }
    report.quality = {
        'reference_avg_wer': mean(wers) if wers else None,
        'reference_avg_cer': mean(cers) if cers else None,
        'reference_scored_count': references_scored,
    }
    report.comparison = _comparison_block(summary_payload, comparison_summary_payload)

    if not report.coverage['has_runtime_truth']:
        report.outstanding_gaps.append('runtime_truth_missing')
    if not report.coverage['has_utterance_records']:
        report.outstanding_gaps.append('utterance_records_missing')
    if report.coverage['has_utterance_records'] and queue_metrics_present == 0:
        report.outstanding_gaps.append('queue_wait_metrics_missing')
    if report.coverage['has_utterance_records'] and worker_metrics_present == 0:
        report.outstanding_gaps.append('worker_service_metrics_missing')
    if report.counters['reference_case_count'] == 0:
        report.outstanding_gaps.append('no_reference_cases_supplied')
    if not report.coverage['has_memory_snapshot']:
        report.outstanding_gaps.append('memory_snapshot_missing')
    if not report.coverage['engine_mode_marked']:
        report.outstanding_gaps.append('engine_mode_missing')
    if not report.coverage['has_preview_quality_metrics']:
        report.outstanding_gaps.append('preview_quality_metrics_missing')
    if not report.coverage['has_curated_multilingual_cases']:
        report.outstanding_gaps.append('curated_multilingual_cases_missing')
    if not report.coverage['has_memory_reference_cases']:
        report.outstanding_gaps.append('memory_reference_cases_missing')
    if comparison_summary_payload is None:
        report.outstanding_gaps.append('baseline_comparison_missing')
    return report


def _comparison_block(primary: dict, other: dict | None) -> dict[str, float | str | None]:
    primary_meta = primary.get('metadata', {})
    primary_truth = primary_meta.get('runtime_truth', {})
    if other is None:
        return {
            'primary_preview_backend': primary_meta.get('preview_backend'),
            'comparison_preview_backend': None,
            'primary_final_backend': primary_meta.get('final_backend'),
            'comparison_final_backend': None,
            'ttft_p50_delta_ms': None,
            'ttft_p95_delta_ms': None,
            'final_latency_p50_delta_ms': None,
            'final_latency_p95_delta_ms': None,
            'comparison_final_latency_p50_improvement_ms': None,
            'comparison_final_latency_p95_improvement_ms': None,
            'comparison_ttft_p50_improvement_ms': None,
            'comparison_ttft_p95_improvement_ms': None,
            'comparison_preview_duplicate_rate_improvement': None,
            'comparison_preview_regression_rate_improvement': None,
            'comparison_preview_churn_chars_per_emit_improvement': None,
            'comparison_preview_low_quality_suppression_rate_improvement': None,
        }
    other_meta = other.get('metadata', {})
    other_truth = other_meta.get('runtime_truth', {})
    ttft_p50_delta = _metric_delta(primary_truth, other_truth, 'ttft_ms', 'p50')
    ttft_p95_delta = _metric_delta(primary_truth, other_truth, 'ttft_ms', 'p95')
    duplicate_delta = _rate_delta(primary_truth, other_truth, 'preview_duplicate_rate')
    regression_delta = _rate_delta(primary_truth, other_truth, 'preview_regression_rate')
    churn_delta = _rate_delta(primary_truth, other_truth, 'preview_churn_chars_per_emit')
    suppression_delta = _rate_delta(primary_truth, other_truth, 'preview_low_quality_suppression_rate')
    return {
        'primary_preview_backend': primary_meta.get('preview_backend'),
        'comparison_preview_backend': other_meta.get('preview_backend'),
        'primary_final_backend': primary_meta.get('final_backend'),
        'comparison_final_backend': other_meta.get('final_backend'),
        'ttft_p50_delta_ms': ttft_p50_delta,
        'ttft_p95_delta_ms': ttft_p95_delta,
        'final_latency_p50_delta_ms': _metric_delta(primary_truth, other_truth, 'speech_end_to_final_ms', 'p50'),
        'final_latency_p95_delta_ms': _metric_delta(primary_truth, other_truth, 'speech_end_to_final_ms', 'p95'),
        'preview_emit_per_utterance_delta': _rate_delta(primary_truth, other_truth, 'preview_emit_per_utterance'),
        'comparison_ttft_p50_improvement_ms': ttft_p50_delta,
        'comparison_ttft_p95_improvement_ms': ttft_p95_delta,
        'comparison_preview_duplicate_rate_improvement': duplicate_delta,
        'comparison_preview_regression_rate_improvement': regression_delta,
        'comparison_preview_churn_chars_per_emit_improvement': churn_delta,
        'comparison_preview_low_quality_suppression_rate_improvement': suppression_delta,
        'comparison_final_latency_p50_improvement_ms': _metric_delta(primary_truth, other_truth, 'speech_end_to_final_ms', 'p50'),
        'comparison_final_latency_p95_improvement_ms': _metric_delta(primary_truth, other_truth, 'speech_end_to_final_ms', 'p95'),
    }


def _metric_delta(primary_truth: dict, other_truth: dict, metric: str, field: str) -> float | None:
    p = (((primary_truth or {}).get('latency') or {}).get(metric) or {}).get(field)
    o = (((other_truth or {}).get('latency') or {}).get(metric) or {}).get(field)
    if p is None or o is None:
        return None
    return float(p) - float(o)


def _rate_delta(primary_truth: dict, other_truth: dict, metric: str) -> float | None:
    p = (((primary_truth or {}).get('rates') or {}).get(metric)
         if 'rates' in (primary_truth or {}) else None)
    o = (((other_truth or {}).get('rates') or {}).get(metric)
         if 'rates' in (other_truth or {}) else None)
    if p is None or o is None:
        return None
    return float(p) - float(o)
