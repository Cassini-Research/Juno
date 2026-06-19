from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_RAW_SURFACES = {"code", "terminal", "developer_tools"}
_NO_PUNCTUATION_MODES = {"verbatim", "command_mode"}
_CONTINUATION_FINAL_WORDS = {
    "a", "an", "and", "as", "at", "because", "but", "for", "from", "if",
    "in", "into", "of", "or", "so", "that", "the", "then", "to", "when",
    "where", "which", "while", "who", "with",
}
_AUX_QUESTION_STARTERS = {
    "am", "are", "can", "could", "did", "do", "does", "is", "should",
    "was", "were", "will", "would",
}
_WH_QUESTION_STARTERS = {"how", "what", "when", "where", "which", "who", "why"}
_QUESTION_PREFIXES = (
    "do you ",
    "does it ",
    "did you ",
    "can you ",
    "could you ",
    "would you ",
    "should we ",
    "are you ",
    "is it ",
    "was it ",
    "were you ",
)
_PERSONAL_SUBJECTS = {
    "i", "we", "you", "he", "she", "they", "there",
}
_DEMONSTRATIVE_SUBJECTS = {"it", "this", "that", "these", "those"}
_QUESTION_SUBJECTS = _PERSONAL_SUBJECTS | _DEMONSTRATIVE_SUBJECTS
_WH_NOUN_BRIDGES = {
    "day", "date", "kind", "name", "number", "one", "part", "place",
    "reason", "time", "type",
}


@dataclass(slots=True)
class PunctuationFloorResult:
    text: str
    changed: bool
    rules_applied: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "rules_applied": list(self.rules_applied),
            "skip_reason": self.skip_reason,
        }


def apply_final_punctuation_floor(
    text: str,
    *,
    app_category: str | None,
    writer_mode: str | None,
    punctuation_policy: str | None,
    final_formatting_policy: str | None,
    selected_text: str | None = None,
    selection_active: bool = False,
    wake_verified: bool = False,
    snippet_expanded: bool = False,
) -> PunctuationFloorResult:
    """Conservative final-paste punctuation for plain dictation.

    This is intentionally not a rewrite engine. It only adds an unambiguous
    terminal mark to normal prose after command/selection/action routing has
    already completed.
    """
    original = text or ""
    stripped = original.strip()
    if not stripped:
        return _skip(original, "empty")
    category = (app_category or "").strip().lower()
    if category in _RAW_SURFACES:
        return _skip(original, "raw_surface")
    mode = (writer_mode or "").strip().lower()
    if mode in _NO_PUNCTUATION_MODES:
        return _skip(original, "mode_no_punctuation")
    policy = (punctuation_policy or "standard").strip().lower()
    if policy in {"none", "literal_minimal", "verbatim"}:
        return _skip(original, "policy_no_punctuation")
    if wake_verified:
        return _skip(original, "wake_verified")
    if selection_active or (selected_text or "").strip():
        return _skip(original, "selection_present")
    if snippet_expanded:
        return _skip(original, "snippet_expanded")
    if "\n" in stripped or _looks_like_structured_text(stripped):
        return _skip(original, "structured_text")
    if stripped.endswith((".", "!", "?", ":", ";", ")", "]", "}", '"')):
        return _skip(original, "already_terminated")

    words = _words(stripped)
    if len(words) <= 3:
        return _skip(original, "short_utterance")
    if words[-1] in _CONTINUATION_FINAL_WORDS:
        return _skip(original, "continuation_tail")

    formatting = (final_formatting_policy or "minimal").strip().lower()
    if category == "messaging" or formatting == "messaging" or policy == "light":
        return _skip(original, "messaging_light")

    mark = "?" if _looks_like_question(words, stripped) else "."
    rule = "terminal_question" if mark == "?" else "terminal_period"
    return PunctuationFloorResult(text=original.rstrip() + mark, changed=True, rules_applied=[rule])


def _skip(text: str, reason: str) -> PunctuationFloorResult:
    return PunctuationFloorResult(text=text or "", changed=False, skip_reason=reason)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']*", (text or "").casefold())


def _looks_like_question(words: list[str], text: str) -> bool:
    if not words:
        return False
    lowered = re.sub(r"\s+", " ", (text or "").casefold()).strip()
    if any(lowered.startswith(prefix) for prefix in _QUESTION_PREFIXES):
        return True
    first = words[0]
    if first in _AUX_QUESTION_STARTERS:
        return _auxiliary_starts_question(words)
    if first not in _WH_QUESTION_STARTERS:
        return False
    return _wh_starts_question(words)


def _auxiliary_starts_question(words: list[str]) -> bool:
    """Return true for clear inverted questions, not imperative commands.

    The punctuation floor is a conservative last-mile helper. It should add a
    question mark for "do you..." or "does this work...", but not for commands
    such as "do this once..." where "do" is the main imperative verb.
    """
    if len(words) < 2:
        return False
    first, second = words[0], words[1]
    if first == "do":
        return second in _PERSONAL_SUBJECTS
    if first in {"does", "did"}:
        return second in _QUESTION_SUBJECTS
    if first == "am":
        return second == "i"
    return second in _QUESTION_SUBJECTS


def _wh_starts_question(words: list[str]) -> bool:
    """Detect clear wh-questions without tagging statement fragments.

    Examples that should stay statements: "why it ended...", "how this
    happened...", "what we need is...". Clear wh-questions still pass through:
    "why did it end", "how do we fix it", "what time is it".
    """
    if len(words) < 2:
        return False
    second = words[1]
    if second in _AUX_QUESTION_STARTERS:
        return True
    if len(words) >= 4 and second in _WH_NOUN_BRIDGES and words[2] in _AUX_QUESTION_STARTERS:
        return True
    return False


def _looks_like_structured_text(text: str) -> bool:
    return bool(
        re.search(r"^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s+", text, re.IGNORECASE)
        or re.search(r"\s(?:[-*•]|\d+[.)]|[a-z][.)])\s+", text, re.IGNORECASE)
    )
