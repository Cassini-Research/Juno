from __future__ import annotations

import re
import unicodedata

from juno_v2.contracts.language import ScriptSummary
from juno_v2.contracts.memory import NormalizationChange, TranscriptNormalization

_SPACE_RE = re.compile(r'\s+')
_SPACE_BEFORE_PUNCT_RE = re.compile(r'\s+([,.;:!?।،。！？：；])')
_MULTI_DASH_RE = re.compile(r'\s*[-–—]{2,}\s*')
_CJK_SPACE_RE = re.compile(r'\s*([。！？：；，、])\s*')


class LanguageAwareNormalizer:
    def normalize_transcript(
        self,
        text: str,
        *,
        requested_language: str | None,
        observed_language: str | None,
        policy_name: str | None,
        scope: str,
    ) -> TranscriptNormalization:
        raw = text or ''
        current = raw.strip()
        applied: list[NormalizationChange] = []
        current = _apply_cleanup(current, applied, requested_language=requested_language, observed_language=observed_language)
        summary = summarize_scripts(current)
        return TranscriptNormalization(
            raw_text=raw,
            normalized_text=current,
            applied=applied,
            metadata={
                'scope': scope,
                'requested_language': requested_language,
                'observed_language': observed_language,
                'policy_name': policy_name,
                'script_summary': summary.to_dict(),
            },
        )

    def normalize_for_eval(self, text: str, *, language: str | None = None) -> str:
        text = _SPACE_RE.sub(' ', (text or '').strip())
        text = _SPACE_BEFORE_PUNCT_RE.sub(r'\1', text)
        lang = (language or '').lower()
        if lang.startswith(('th', 'zh', 'ja', 'ko')):
            return text
        return text.lower()


def summarize_scripts(text: str) -> ScriptSummary:
    summary = ScriptSummary()
    for char in text or '':
        if not char.isalpha():
            continue
        code = ord(char)
        name = unicodedata.name(char, '')
        if 'DEVANAGARI' in name:
            summary.devanagari += 1
        elif 'THAI' in name:
            summary.thai += 1
        elif 'HIRAGANA' in name or 'KATAKANA' in name:
            summary.kana += 1
        elif 'HANGUL' in name:
            summary.hangul += 1
        elif 'ARABIC' in name:
            summary.arabic += 1
        elif 'CYRILLIC' in name:
            summary.cyrillic += 1
        elif _is_han(code):
            summary.han += 1
        elif 'LATIN' in name or ('A' <= char <= 'Z') or ('a' <= char <= 'z'):
            summary.latin += 1
        else:
            summary.other += 1
    return summary


def _is_han(code: int) -> bool:
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
    )


def _apply_cleanup(
    text: str,
    applied: list[NormalizationChange],
    *,
    requested_language: str | None,
    observed_language: str | None,
) -> str:
    out = text
    replacements = [
        (_SPACE_RE, ' ', 'whitespace_collapse'),
        (_SPACE_BEFORE_PUNCT_RE, r'\1', 'space_before_punctuation'),
        (_MULTI_DASH_RE, ' — ', 'dash_normalization'),
    ]
    for pattern, repl, source in replacements:
        new_out = pattern.sub(repl, out).strip()
        if new_out != out:
            applied.append(NormalizationChange(kind='language_normalization', source=source, before=out, after=new_out))
            out = new_out

    lang = (requested_language or observed_language or '').lower()
    if lang.startswith('hi'):
        new_out = re.sub(r'\s*।\s*', '। ', out).strip()
        if new_out != out:
            applied.append(NormalizationChange(kind='language_normalization', source='hindi_danda_spacing', before=out, after=new_out))
            out = new_out
    if lang.startswith('ar'):
        new_out = re.sub(r'\s*،\s*', '، ', out).strip()
        if new_out != out:
            applied.append(NormalizationChange(kind='language_normalization', source='arabic_comma_spacing', before=out, after=new_out))
            out = new_out
    if lang.startswith(('zh', 'ja')):
        new_out = _CJK_SPACE_RE.sub(r'\1', out)
        new_out = re.sub(r'\s+', ' ', new_out).strip()
        if new_out != out:
            applied.append(NormalizationChange(kind='language_normalization', source='cjk_punctuation_spacing', before=out, after=new_out))
            out = new_out
    return out
