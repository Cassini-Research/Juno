from __future__ import annotations

from dataclasses import dataclass, field

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.language import LanguageDecision
from juno_v2.language.catalog import (
    DEFAULT_SUPPORTED_LANGUAGES,
    LEGACY_POLICY_ALIASES,
    normalize_pair_languages,
    normalize_supported_languages,
)
from juno_v2.language.normalize import summarize_scripts


@dataclass(slots=True)
class LanguagePlannerConfig:
    policy_name: str = 'auto_supported'
    fixed_language: str | None = None
    supported_languages: list[str] = field(default_factory=lambda: list(DEFAULT_SUPPORTED_LANGUAGES))
    pair_languages: tuple[str, str] | None = None
    command_language: str = 'en'


@dataclass(slots=True)
class LanguagePlanner:
    config: LanguagePlannerConfig = field(default_factory=LanguagePlannerConfig)

    def plan_utterance(
        self,
        *,
        utterance_id: str,
        context: TypedContextBundle,
        configured_preview_language: str | None,
        configured_final_language: str | None,
    ) -> LanguageDecision:
        script_summary = summarize_scripts(' '.join([
            context.selected_text,
            context.focused_text_before,
            context.focused_text_after,
            context.window_title or '',
            context.app_name or '',
            context.clipboard_text or '',
        ]))
        policy_name, policy_pair = _resolve_policy(self.config.policy_name, self.config.pair_languages)
        supported = normalize_supported_languages(self.config.supported_languages)
        configured_language = _normalize_code(configured_final_language or configured_preview_language or self.config.fixed_language)
        if configured_language and policy_name == 'auto_supported':
            policy_name = 'fixed'

        # IMPORTANT: LanguageDecision.initial_prompt is deliberately left
        # empty (None) for every policy branch below. Whisper's
        # ``initial_prompt`` argument is designed for **vocabulary
        # biasing** (proper nouns, technical terms), not system
        # instructions. Passing a natural-language sentence like
        # "Transcribe in English. Preserve original words..." causes
        # the decoder to treat the prompt as previous audio context and
        # echo fragments of it back on short or ambiguous audio.
        #
        # The Phase 10 TTS accuracy harness surfaced this on its first
        # real run: 5 of 8 preview-lane samples returned the prompt
        # text verbatim (``"Transcribe in English. Preserve..."``)
        # instead of any transcription. Language selection is handled
        # separately via the ``language`` argument, so removing the
        # prompt is pure upside.
        #
        # If a future eval shows we need per-language vocabulary
        # hints (e.g. Hindi transliteration terms for pair:en,hi), the
        # place to inject them is ``RecognitionBiasEngine.build_plan``
        # via ``base_prompt``, NOT here. The bias engine knows how to
        # merge them with memory-derived phrases.

        if policy_name == 'fixed':
            request_language = configured_language or _normalize_code(self.config.fixed_language) or 'en'
            allowed = [request_language]
            return LanguageDecision(
                utterance_id=utterance_id,
                policy_name='fixed',
                request_language=request_language,
                allowed_languages=allowed,
                initial_prompt=None,
                command_language=self.config.command_language,
                script_summary=script_summary,
                metadata={'reason': 'fixed_language'},
            )

        if policy_name == 'pair':
            pair = policy_pair or normalize_pair_languages((_normalize_code('en'), _normalize_code('hi')))
            allowed = list(pair) if pair else list(supported[:2])
            return LanguageDecision(
                utterance_id=utterance_id,
                policy_name='pair',
                request_language=None,
                allowed_languages=allowed,
                pair_languages=list(allowed),
                initial_prompt=None,
                command_language=self.config.command_language,
                script_summary=script_summary,
                metadata={'reason': 'code_switch_pair', 'pair_languages': list(allowed)},
            )

        inferred = _infer_context_language(script_summary, supported)
        if inferred:
            return LanguageDecision(
                utterance_id=utterance_id,
                policy_name='auto_supported',
                request_language=inferred,
                allowed_languages=list(supported),
                initial_prompt=None,
                command_language=self.config.command_language,
                script_summary=script_summary,
                metadata={'reason': f'{inferred}_context_dominant'},
            )

        requested = configured_language if configured_language in supported else None
        return LanguageDecision(
            utterance_id=utterance_id,
            policy_name='auto_supported',
            request_language=requested,
            allowed_languages=list(supported),
            initial_prompt=None,
            command_language=self.config.command_language,
            script_summary=script_summary,
            metadata={'reason': 'supported_language_auto'},
        )


def _normalize_code(value: str | None) -> str | None:
    code = (value or '').strip().lower()
    return code or None


def _resolve_policy(policy_name: str | None, pair_languages: tuple[str, str] | None) -> tuple[str, tuple[str, str] | None]:
    raw = (policy_name or 'auto_supported').strip().lower()
    if raw in LEGACY_POLICY_ALIASES:
        normalized, preset = LEGACY_POLICY_ALIASES[raw]
        return normalized, normalize_pair_languages(pair_languages) or preset
    if raw.startswith('pair:'):
        parsed = tuple(item.strip().lower() for item in raw.split(':', 1)[1].split(',') if item.strip())
        return 'pair', normalize_pair_languages(parsed) or normalize_pair_languages(pair_languages)
    if raw not in {'fixed', 'auto_supported', 'pair'}:
        return 'auto_supported', normalize_pair_languages(pair_languages)
    return raw, normalize_pair_languages(pair_languages)


def _infer_context_language(summary, supported: list[str]) -> str | None:
    script_to_language = [
        ('devanagari', 'hi'),
        ('thai', 'th'),
        ('hangul', 'ko'),
        ('arabic', 'ar'),
        ('cyrillic', 'ru'),
    ]
    dominant = summary.dominant_script
    if dominant == 'han':
        if 'zh' in supported:
            return 'zh'
        if 'ja' in supported:
            return 'ja'
    if dominant == 'kana' and 'ja' in supported:
        return 'ja'
    for script_name, lang in script_to_language:
        if dominant == script_name and lang in supported:
            return lang
    if dominant == 'latin':
        for code in ('en', 'es', 'fr', 'de', 'pt', 'id'):
            if code in supported:
                return code
    return None
