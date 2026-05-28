from __future__ import annotations

from juno_v2.contracts.final import compute_quality_report
from juno_v2.contracts.language import MultilingualQualityReport
from juno_v2.language.normalize import LanguageAwareNormalizer, summarize_scripts


def compute_multilingual_quality_report(
    reference_text: str,
    hypothesis_text: str,
    *,
    requested_language: str | None = None,
    observed_language: str | None = None,
) -> MultilingualQualityReport:
    normalizer = LanguageAwareNormalizer()
    raw = compute_quality_report(reference_text, hypothesis_text)
    norm_ref = normalizer.normalize_for_eval(reference_text, language=requested_language or observed_language)
    norm_hyp = normalizer.normalize_for_eval(hypothesis_text, language=observed_language or requested_language)
    norm = compute_quality_report(norm_ref, norm_hyp)
    return MultilingualQualityReport(
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        requested_language=requested_language,
        observed_language=observed_language,
        word_error_rate=raw.word_error_rate,
        char_error_rate=raw.char_error_rate,
        normalized_word_error_rate=norm.word_error_rate,
        normalized_char_error_rate=norm.char_error_rate,
        reference_script_summary=summarize_scripts(reference_text),
        hypothesis_script_summary=summarize_scripts(hypothesis_text),
        metadata={
            'normalized_reference_text': norm_ref,
            'normalized_hypothesis_text': norm_hyp,
            'code_switch_reference': summarize_scripts(reference_text).code_switch_detected,
            'code_switch_hypothesis': summarize_scripts(hypothesis_text).code_switch_detected,
        },
    )
