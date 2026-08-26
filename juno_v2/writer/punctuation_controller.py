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
# A sentence boundary is a terminal mark followed by whitespace. Requiring the
# whitespace is what keeps decimals ("1.5 miles"), filenames ("config.json")
# and domains ("node.js") from being mistaken for the end of a sentence.
_SENTENCE_BOUNDARY = re.compile(r"[.?!…？！。]\s+")
# Dotted forms that do end in "." + space yet are not sentence ends.
_INITIALISM = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]$")
_ABBREVIATIONS = {
    "dr", "etc", "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs",
}
_WH_NOUN_BRIDGES = {
    "branch", "build", "command", "commit", "file", "model", "option", "pr",
    "release", "repo", "repository", "setting", "version",
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
    if stripped.endswith(
        (
            ".", "!", "?", ":", ";", ")", "]", "}", '"',
            # Unicode terminals: ellipsis and full-width / CJK marks. Without
            # these, text dictated/auto-formatted to end in "…" or "？" would
            # get a redundant ASCII "." or "?" appended.
            "…", "？", "！", "。", "：", "；", "”", "’",
        )
    ):
        return _skip(original, "already_terminated")

    words = _words(stripped)
    if len(words) <= 3:
        return _skip(original, "short_utterance")
    if words[-1] in _CONTINUATION_FINAL_WORDS:
        return _skip(original, "continuation_tail")

    formatting = (final_formatting_policy or "minimal").strip().lower()
    if category == "messaging" or formatting == "messaging" or policy == "light":
        return _skip(original, "messaging_light")

    # Only the trailing sentence decides the terminal mark. A buffer that opens
    # with "Can you ...?" and closes with an imperative must still end in ".".
    tail = _trailing_sentence(stripped)
    mark = "?" if _looks_like_question(_words(tail), tail) else "."
    rule = "terminal_question" if mark == "?" else "terminal_period"
    return PunctuationFloorResult(text=original.rstrip() + mark, changed=True, rules_applied=[rule])


def _skip(text: str, reason: str) -> PunctuationFloorResult:
    return PunctuationFloorResult(text=text or "", changed=False, skip_reason=reason)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']*", (text or "").casefold())


def _trailing_sentence(text: str) -> str:
    """Return the final sentence of a buffer.

    The question heuristics below all inspect the opening words of a clause, so
    for a multi-sentence buffer they must be given the final sentence rather
    than the whole buffer.

    Splitting is not harmless in either direction, so both are guarded. A split
    that fires too eagerly truncates the clause and loses a legitimate "?" --
    "Can you run 1.5 miles" must not be cut at the decimal point -- which is
    why a terminator only counts when whitespace (or end of text) follows it,
    and why the dotted forms that do pass that test but are not sentence ends
    (initialisms like "p.m." and abbreviations like "St.") are skipped over. A
    split that fires too late leaves the opening clause in view and is what
    this helper exists to prevent. Anything still ambiguous falls through to
    the whole buffer, which is the pre-existing behaviour.
    """
    body = (text or "").strip()
    for match in reversed(list(_SENTENCE_BOUNDARY.finditer(body))):
        if _ends_with_abbreviation(body[: match.start() + 1]):
            continue
        return body[match.end():].strip()
    return body


def _ends_with_abbreviation(head: str) -> bool:
    """Return true when the trailing "." of `head` abbreviates a word."""
    if not head.endswith("."):
        return False
    match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", head)
    if match is None:
        return False
    token = match.group(1)
    if _INITIALISM.fullmatch(token):
        return True
    return token.casefold() in _ABBREVIATIONS


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
    markers = re.findall(r"(?:^|\s)(?:[-*•]|\d+[.)]|[a-z][.)])\s+", text, re.IGNORECASE)
    return bool(
        re.search(r"^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s+", text, re.IGNORECASE)
        or len(markers) >= 2
    )
