from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class ScriptSummary:
    latin: int = 0
    devanagari: int = 0
    thai: int = 0
    han: int = 0
    kana: int = 0
    hangul: int = 0
    arabic: int = 0
    cyrillic: int = 0
    other: int = 0

    @property
    def total_letters(self) -> int:
        return self.latin + self.devanagari + self.thai + self.han + self.kana + self.hangul + self.arabic + self.cyrillic + self.other

    @property
    def dominant_script(self) -> str | None:
        counts = {
            'latin': self.latin,
            'devanagari': self.devanagari,
            'thai': self.thai,
            'han': self.han,
            'kana': self.kana,
            'hangul': self.hangul,
            'arabic': self.arabic,
            'cyrillic': self.cyrillic,
            'other': self.other,
        }
        best = max(counts.items(), key=lambda item: item[1])
        return best[0] if best[1] > 0 else None

    @property
    def code_switch_detected(self) -> bool:
        active = sum(int(value > 0) for value in (self.latin, self.devanagari, self.thai, self.han, self.kana, self.hangul, self.arabic, self.cyrillic, self.other))
        return active >= 2

    def to_dict(self) -> Dict[str, int | str | bool | None]:
        return {
            **asdict(self),
            'total_letters': self.total_letters,
            'dominant_script': self.dominant_script,
            'code_switch_detected': self.code_switch_detected,
        }


@dataclass(slots=True)
class LanguageDecision:
    utterance_id: str
    policy_name: str
    request_language: str | None = None
    allowed_languages: List[str] = field(default_factory=list)
    pair_languages: List[str] = field(default_factory=list)
    initial_prompt: str | None = None
    command_language: str = 'en'
    script_summary: ScriptSummary = field(default_factory=ScriptSummary)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'utterance_id': self.utterance_id,
            'policy_name': self.policy_name,
            'request_language': self.request_language,
            'allowed_languages': list(self.allowed_languages),
            'pair_languages': list(self.pair_languages),
            'initial_prompt': self.initial_prompt,
            'command_language': self.command_language,
            'script_summary': self.script_summary.to_dict(),
            'metadata': dict(self.metadata),
        }


@dataclass(slots=True)
class MultilingualQualityReport:
    reference_text: str
    hypothesis_text: str
    requested_language: str | None = None
    observed_language: str | None = None
    word_error_rate: float = 0.0
    char_error_rate: float = 0.0
    normalized_word_error_rate: float = 0.0
    normalized_char_error_rate: float = 0.0
    reference_script_summary: ScriptSummary = field(default_factory=ScriptSummary)
    hypothesis_script_summary: ScriptSummary = field(default_factory=ScriptSummary)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'reference_text': self.reference_text,
            'hypothesis_text': self.hypothesis_text,
            'requested_language': self.requested_language,
            'observed_language': self.observed_language,
            'word_error_rate': self.word_error_rate,
            'char_error_rate': self.char_error_rate,
            'normalized_word_error_rate': self.normalized_word_error_rate,
            'normalized_char_error_rate': self.normalized_char_error_rate,
            'reference_script_summary': self.reference_script_summary.to_dict(),
            'hypothesis_script_summary': self.hypothesis_script_summary.to_dict(),
            'metadata': dict(self.metadata),
        }
