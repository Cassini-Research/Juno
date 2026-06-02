from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

from juno_v2.contracts.memory import MemorySnapshot


@dataclass(slots=True)
class DistributionSummary:
    count: int = 0
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None

    def to_dict(self) -> dict:
        return {
            'count': self.count,
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'p50': self.p50,
            'p95': self.p95,
            'p99': self.p99,
        }


@dataclass(slots=True)
class RuntimeTruthReport:
    latency: dict[str, DistributionSummary] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    language: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'latency': {k: v.to_dict() for k, v in self.latency.items()},
            'rates': dict(self.rates),
            'counters': dict(self.counters),
            'memory': dict(self.memory),
            'language': dict(self.language),
        }


def summarize_distribution(values: Iterable[float | None]) -> DistributionSummary:
    nums = sorted(float(v) for v in values if v is not None)
    if not nums:
        return DistributionSummary()
    return DistributionSummary(
        count=len(nums),
        min=nums[0],
        max=nums[-1],
        mean=mean(nums),
        p50=_percentile(nums, 0.50),
        p95=_percentile(nums, 0.95),
        p99=_percentile(nums, 0.99),
    )


def build_runtime_truth_report(
    *,
    utterance_metrics: list[dict],
    utterance_count: int,
    committed_count: int,
    conflict_count: int,
    preview_decode_count: int,
    final_decode_count: int,
    preview_emit_count: int,
    preview_duplicate_count: int = 0,
    preview_regression_count: int = 0,
    preview_churn_chars_total: int = 0,
    preview_low_quality_suppression_count: int = 0,
    preview_low_quality_emit_count: int = 0,
    normalization_change_count: int = 0,
    writer_action_count: int,
    writer_model_action_count: int,
    writer_deterministic_action_count: int,
    writer_noop_count: int,
    memory_snapshot: MemorySnapshot | None,
    memory_packet_summaries: list[dict] | None = None,
    requested_language_counts: dict[str, int],
    observed_language_counts: dict[str, int],
    language_policy_counts: dict[str, int],
    code_switch_utterance_count: int,
    language_mismatch_count: int,
) -> RuntimeTruthReport:
    report = RuntimeTruthReport()
    report.latency = {
        'ttft_ms': summarize_distribution(m.get('ttft_ms') for m in utterance_metrics),
        'speech_end_to_final_ms': summarize_distribution(m.get('speech_end_to_final_ms') for m in utterance_metrics),
        'speech_end_to_commit_ms': summarize_distribution(m.get('speech_end_to_commit_ms') for m in utterance_metrics),
        'final_decode_ms': summarize_distribution(m.get('final_decode_ms') for m in utterance_metrics),
        'first_preview_stability_delta_chars': summarize_distribution(m.get('first_preview_stability_delta_chars') for m in utterance_metrics),
        'first_preview_queue_wait_ms': summarize_distribution(m.get('first_preview_queue_wait_ms') for m in utterance_metrics),
        'first_preview_worker_service_ms': summarize_distribution(m.get('first_preview_worker_service_ms') for m in utterance_metrics),
        'final_queue_wait_ms': summarize_distribution(m.get('final_queue_wait_ms') for m in utterance_metrics),
        'final_worker_service_ms': summarize_distribution(m.get('final_worker_service_ms') for m in utterance_metrics),
        'writer_queue_wait_ms': summarize_distribution(m.get('writer_queue_wait_ms') for m in utterance_metrics),
        'writer_worker_service_ms': summarize_distribution(m.get('writer_worker_service_ms') for m in utterance_metrics),
    }
    report.counters = {
        'utterance_count': utterance_count,
        'committed_count': committed_count,
        'conflict_count': conflict_count,
        'preview_decode_count': preview_decode_count,
        'final_decode_count': final_decode_count,
        'preview_emit_count': preview_emit_count,
        'preview_duplicate_count': preview_duplicate_count,
        'preview_regression_count': preview_regression_count,
        'preview_churn_chars_total': preview_churn_chars_total,
        'preview_low_quality_suppression_count': preview_low_quality_suppression_count,
        'preview_low_quality_emit_count': preview_low_quality_emit_count,
        'normalization_change_count': normalization_change_count,
        'writer_action_count': writer_action_count,
        'writer_model_action_count': writer_model_action_count,
        'writer_deterministic_action_count': writer_deterministic_action_count,
        'writer_noop_count': writer_noop_count,
        'code_switch_utterance_count': code_switch_utterance_count,
        'language_mismatch_count': language_mismatch_count,
    }
    denom = max(1, utterance_count)
    report.rates = {
        'commit_rate': committed_count / denom,
        'conflict_rate': conflict_count / denom,
        'language_mismatch_rate': language_mismatch_count / denom,
        'code_switch_rate': code_switch_utterance_count / denom,
        'normalization_change_rate': normalization_change_count / max(1, final_decode_count + preview_emit_count),
        'preview_emit_per_utterance': preview_emit_count / denom,
        'preview_duplicate_rate': preview_duplicate_count / max(1, preview_emit_count),
        'preview_regression_rate': preview_regression_count / max(1, preview_emit_count),
        'preview_low_quality_suppression_rate': preview_low_quality_suppression_count / max(1, preview_decode_count),
        'preview_low_quality_emit_rate': preview_low_quality_emit_count / max(1, preview_emit_count),
        'preview_churn_chars_per_emit': preview_churn_chars_total / max(1, preview_emit_count),
        'preview_churn_chars_per_utterance': preview_churn_chars_total / denom,
        'writer_action_rate': writer_action_count / denom,
        'writer_model_action_rate': writer_model_action_count / denom,
        'writer_noop_rate': writer_noop_count / denom,
    }
    snapshot = memory_snapshot or MemorySnapshot(schema_version=1)
    packet_summaries = memory_packet_summaries or []
    report.memory = {
        'lexicon_entry_count': len(snapshot.lexicon),
        'replacement_rule_count': len(snapshot.replacements),
        'correction_pair_count': len(snapshot.corrections),
        'session_entity_count': len(snapshot.session_entities),
        'top_corrections': [item.to_dict() for item in sorted(snapshot.corrections, key=lambda x: (-x.count, x.corrected.casefold()))[:5]],
        'top_session_entities': [item.to_dict() for item in sorted(snapshot.session_entities, key=lambda x: (-x.count, x.value.casefold()))[:8]],
        'average_packet_bias_phrase_count': _mean_metric(packet_summaries, 'bias_phrase_count'),
        'average_packet_lexicon_terms': _mean_metric(packet_summaries, 'lexicon_terms'),
        'average_packet_replacements': _mean_metric(packet_summaries, 'replacements'),
        'average_packet_corrections': _mean_metric(packet_summaries, 'corrections'),
        'average_packet_session_entities': _mean_metric(packet_summaries, 'session_entities'),
    }
    report.language = {
        'requested_language_counts': dict(requested_language_counts),
        'observed_language_counts': dict(observed_language_counts),
        'language_policy_counts': dict(language_policy_counts),
    }
    return report


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError('Cannot compute percentile of empty sequence')
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    frac = pos - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


def _mean_metric(items: list[dict], key: str) -> float | None:
    vals = [float(item.get(key)) for item in items if item.get(key) is not None]
    if not vals:
        return None
    return mean(vals)
