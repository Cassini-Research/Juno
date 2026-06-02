from __future__ import annotations

import re

from juno_v2.contracts.commands import CommandPayload, DeterministicCommand, DeterministicCommandKind


_CommandPattern = tuple[re.Pattern[str], str, DeterministicCommandKind, CommandPayload]


_ACTIVE_PATTERNS: list[_CommandPattern] = [
    (re.compile(r"^scratch\s+that[.,!?]?$", re.I), "scratch_that", "discard_utterance", {}),
    (re.compile(r"^undo\s+that[.,!?]?$", re.I), "undo_that", "undo_last", {}),
    (re.compile(r"^delete\s+that[.,!?]?$", re.I), "delete_that", "discard_utterance", {}),
    (re.compile(r"^delete\s+last\s+word[.,!?]?$", re.I), "delete_last_word", "delete_words", {"words": 1}),
    (re.compile(r"^delete\s+last\s+two\s+words[.,!?]?$", re.I), "delete_last_two_words", "delete_words", {"words": 2}),
    (re.compile(r"^delete\s+last\s+sentence[.,!?]?$", re.I), "delete_last_sentence", "delete_sentence", {"scope": "utterance"}),
    (re.compile(r"^(?:(?:okay|ok|please)[,\s]+)*(?:go\s+to\s+)?(?:new\s+line|newline)[.,!?]?$", re.I), "new_line", "insert", {"text": "\n"}),
    (re.compile(r"^new\s+paragraph[.,!?]?$", re.I), "new_paragraph", "insert", {"text": "\n\n"}),
    (re.compile(r"^next\s+bullet[.,!?]?$", re.I), "next_bullet", "structure", {"mode": "bullet_item"}),
    (re.compile(r"^next\s+number[.,!?]?$", re.I), "next_number", "structure", {"mode": "numbered_item"}),
    (re.compile(r"^open\s+quote[.,!?]?$", re.I), "open_quote", "quote", {"text": "\u201c"}),
    (re.compile(r"^close\s+quote[.,!?]?$", re.I), "close_quote", "quote", {"text": "\u201d"}),
]

_RECENT_TARGET = r"(?:that(?:\s+text)?|the\s+last\s+(?:sentence|line|paragraph|answer|thing))"

_RECENT_PATTERNS: list[_CommandPattern] = [
    (re.compile(r"^fix\s+that[.,!?]?$", re.I), "fix_that", "recent_edit", {"instruction": "Fix grammar and clarity. Preserve meaning."}),
    (re.compile(r"^make\s+that\s+shorter[.,!?]?$", re.I), "make_shorter", "recent_edit", {"instruction": "Make the text more concise. Preserve meaning."}),
    (re.compile(rf"^make\s+{_RECENT_TARGET}\s+(?:more\s+)?(?:clear|clearer)[.,!?]?$", re.I), "make_clearer", "recent_edit", {"instruction": "Improve clarity. Preserve meaning."}),
    (re.compile(rf"^make\s+{_RECENT_TARGET}\s+(?:(?:more\s+)?(?:concise|direct|brief)|shorter)(?:\s+and\s+(?:more\s+)?(?:direct|concise|clear|clearer))?[.,!?]?$", re.I), "make_shorter_direct", "recent_edit", {"instruction": "Make the text more concise and direct. Preserve meaning."}),
    (re.compile(r"^make\s+that\s+longer[.,!?]?$", re.I), "make_longer", "recent_edit", {"instruction": "Expand with useful detail. Preserve meaning."}),
    (re.compile(r"^make\s+that\s+clearer[.,!?]?$", re.I), "make_clearer", "recent_edit", {"instruction": "Improve clarity. Preserve meaning."}),
    (re.compile(r"^make\s+that\s+more\s+formal[.,!?]?$", re.I), "make_formal", "recent_edit", {"instruction": "Rewrite in a formal tone. Preserve meaning."}),
    (re.compile(r"^make\s+that\s+more\s+casual[.,!?]?$", re.I), "make_casual", "recent_edit", {"instruction": "Rewrite in a casual tone. Preserve meaning."}),
    (re.compile(r"^turn\s+that\s+into\s+bullets[.,!?]?$", re.I), "bullets", "recent_edit", {"transform": "bullets"}),
    (re.compile(r"^turn\s+that\s+into\s+a\s+numbered\s+list[.,!?]?$", re.I), "numbered", "recent_edit", {"transform": "numbered"}),
    (re.compile(r"^translate\s+that\s+to\s+(.+?)[.,!?]?$", re.I), "translate", "translate", {}),
    (re.compile(r"^replace\s+(.+?)\s+with\s+(.+?)[.,!?]?$", re.I), "replace", "replace", {}),
    (re.compile(r"^delete\s+the\s+last\s+sentence[.,!?]?$", re.I), "del_last_sentence_doc", "delete_sentence", {"scope": "recent"}),
    (re.compile(r"^delete\s+the\s+last\s+paragraph[.,!?]?$", re.I), "del_last_para", "recent_edit", {"transform": "delete_paragraph"}),
]


def parse_deterministic_command(text: str) -> DeterministicCommand | None:
    t = (text or "").strip()
    if not t:
        return None
    for pat, name, kind, payload in _ACTIVE_PATTERNS:
        m = pat.match(t)
        if m:
            return DeterministicCommand(name=name, kind=kind, payload=dict(payload))
    for pat, name, kind, payload in _RECENT_PATTERNS:
        m = pat.match(t)
        if m:
            pl = dict(payload)
            if kind == "translate":
                pl["language"] = m.group(1).strip()
            if kind == "replace":
                pl["from"] = m.group(1).strip()
                pl["to"] = m.group(2).strip()
            return DeterministicCommand(name=name, kind=kind, payload=pl)
    return None
