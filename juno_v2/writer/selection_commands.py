"""Bounded recognition for spoken transforms of highlighted text.

This module is the shared deterministic floor for the parser, turn-planner
routing, and the selection intent gate.  It intentionally recognizes only
utterances whose complete normalized surface is an edit command.  That keeps
ordinary prose such as ``Make this shorter is what she said`` out of the
command lane while tolerating harmless ASR/politeness lead-ins such as
``to``, ``could you``, and ``I want you to``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SelectionCommandLane = Literal["deterministic", "model"]


@dataclass(frozen=True, slots=True)
class SelectionTransformCommand:
    transform_id: str
    lane: SelectionCommandLane
    instruction: str
    normalized_command: str
    transform_kind: str | None = None
    allows_list_output: bool = False
    content_preserving_list: bool = False


_TARGET = (
    r"(?:this(?:\s+text)?|it|the\s+(?:selected|highlighted)\s+text|"
    r"(?:selected|highlighted)\s+text|the\s+selection|selection)"
)
_COUNT = (
    r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty)"
)
_LANGUAGE = (
    r"(?:brazilian\s+portuguese|simplified\s+chinese|traditional\s+chinese|"
    r"latin\s+american\s+spanish|english|spanish|french|german|italian|"
    r"portuguese|dutch|polish|swedish|norwegian|danish|finnish|russian|"
    r"ukrainian|czech|slovak|hungarian|romanian|bulgarian|greek|turkish|"
    r"hebrew|arabic|persian|farsi|hindi|urdu|bengali|tamil|telugu|marathi|"
    r"gujarati|punjabi|kannada|malayalam|chinese|mandarin|cantonese|"
    r"japanese|korean|thai|vietnamese|indonesian|malay|tagalog|filipino|"
    r"swahili|afrikaans|amharic)"
)

_COMMAND_LEADINS = (
    re.compile(r"^(?:hey\s+)?juno[,\s]+", re.I),
    re.compile(r"^(?:please|hey|okay|ok|u+m+|uh+|a+h+)[,.\s]+", re.I),
    re.compile(r"^(?:can|could|would|will)\s+you(?:\s+please)?\s+", re.I),
    re.compile(
        r"^(?:i\s+(?:want|need)\s+(?:you\s+)?to|"
        r"i(?:'d|\s+would)\s+like\s+(?:you\s+)?to|"
        r"let(?:'s|s))\s+",
        re.I,
    ),
)
_EDIT_OPENERS = frozenset({
    "add", "bullet", "bulletize", "change", "clean", "clarify", "condense",
    "convert", "correct", "delete", "expand", "extract", "fix", "format",
    "improve", "insert", "make", "number", "paraphrase", "polish", "put",
    "remove", "rephrase", "rewrite", "shorten", "simplify", "summarise",
    "summarize", "tighten", "translate", "turn", "update",
})


def normalize_selection_command(text: str) -> str:
    """Normalize command framing without changing the transcript itself."""

    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return ""
    for _ in range(5):
        before = value
        for pattern in _COMMAND_LEADINS:
            value = pattern.sub("", value, count=1).strip()
        if value == before:
            break
    value = re.sub(r"(?:,\s*)?\bplease\b[.!?]*$", "", value, flags=re.I).strip()
    value = value.rstrip(" .!?").strip()

    # Whisper occasionally preserves the infinitive marker but drops the
    # preceding request (production: "to make this shorter").  Remove it only
    # before a strong edit opener; complete-command matching below still has
    # to succeed, so prose continuations remain dictation.
    leading_to = re.match(r"^to\s+(?P<rest>.+)$", value, flags=re.I)
    if leading_to is not None:
        rest = leading_to.group("rest").strip()
        first = re.sub(r"[^a-z]", "", rest.split(maxsplit=1)[0].casefold()) if rest else ""
        if first in _EDIT_OPENERS:
            value = rest
    return value


def _full(pattern: str, value: str) -> re.Match[str] | None:
    return re.fullmatch(pattern, value, flags=re.I)


def _model(
    transform_id: str,
    instruction: str,
    command: str,
    *,
    allows_list_output: bool = False,
) -> SelectionTransformCommand:
    return SelectionTransformCommand(
        transform_id=transform_id,
        lane="model",
        instruction=instruction,
        normalized_command=command,
        allows_list_output=allows_list_output,
    )


def recognize_selection_transform_command(text: str) -> SelectionTransformCommand | None:
    """Return a catalog transform only when the whole utterance is a command."""

    command = normalize_selection_command(text)
    if not command or len(command.split()) > 32:
        return None

    # Structure-only transforms stay deterministic. They preserve every
    # content unit and never ask a model to regenerate the selected text.
    if _full(
        rf"(?:(?:turn|convert|change|format|put|make|rewrite)\s+(?:{_TARGET}\s+)?"
        rf"(?:(?:into|as|in|to)\s+)?(?:a\s+)?(?:{_COUNT}\s+)?(?:bullet\s+points?|bullets?|"
        rf"bulleted\s+list|bullet\s+list|list)|bullet(?:ize)?\s+(?:{_TARGET}))",
        command,
    ):
        return SelectionTransformCommand(
            transform_id="bulletize",
            lane="deterministic",
            instruction="Convert the selected text to bullets without omitting content.",
            normalized_command=command,
            transform_kind="bullets",
            allows_list_output=True,
            content_preserving_list=True,
        )
    if _full(
        rf"(?:(?:turn|convert|change|format|put|make|rewrite)\s+(?:{_TARGET}\s+)?"
        rf"(?:(?:into|as|in|to)\s+)?(?:a\s+)?(?:numbered(?:\s+list)?|"
        rf"numbered\s+points?)|number\s+(?:{_TARGET}))",
        command,
    ):
        return SelectionTransformCommand(
            transform_id="numbered_list",
            lane="deterministic",
            instruction="Convert the selected text to a numbered list without omitting content.",
            normalized_command=command,
            transform_kind="numbered",
            allows_list_output=True,
            content_preserving_list=True,
        )

    for transform_id, transform_kind, pattern, instruction in (
        (
            "uppercase",
            "uppercase",
            rf"(?:(?:uppercase|capitalize)\s+{_TARGET}|make\s+{_TARGET}\s+uppercase)",
            "Convert the selected text to uppercase.",
        ),
        (
            "lowercase",
            "lowercase",
            rf"(?:lowercase\s+{_TARGET}|make\s+{_TARGET}\s+lowercase)",
            "Convert the selected text to lowercase.",
        ),
        (
            "title_case",
            "title_case",
            rf"(?:(?:title\s*case|titlecase)\s+{_TARGET}|make\s+{_TARGET}\s+(?:title\s*case|titlecase))",
            "Convert the selected text to title case.",
        ),
    ):
        if _full(pattern, command):
            return SelectionTransformCommand(
                transform_id=transform_id,
                lane="deterministic",
                instruction=instruction,
                normalized_command=command,
                transform_kind=transform_kind,
            )

    if _full(
        rf"(?:(?:fix|correct)\s+(?:(?:{_TARGET})(?:'s)?\s+)?(?:the\s+)?"
        rf"(?:grammar|spelling|punctuation|errors?|typos?)(?:\s+and\s+"
        rf"(?:grammar|spelling|punctuation|errors?|typos?))*|"
        rf"(?:fix|correct)\s+(?:the\s+)?(?:grammar|spelling|punctuation|errors?|typos?)"
        rf"\s+(?:in|of)\s+{_TARGET}|(?:fix|correct)\s+{_TARGET})",
        command,
    ):
        return _model("fix_grammar", "Fix grammar, spelling, and punctuation. Preserve meaning exactly.", command)
    if _full(rf"(?:shorten|condense|tighten)(?:\s+{_TARGET})?", command) or _full(
        rf"make(?:\s+{_TARGET})?\s+(?:(?:a\s+little|much)\s+)?(?:more\s+)?"
        rf"(?:concise|shorter|brief|direct)",
        command,
    ):
        return _model("make_shorter", "Make the selected text more concise while preserving meaning.", command)
    if _full(
        rf"(?:expand|elaborate(?:\s+on)?)(?:\s+{_TARGET})?(?:\s+with\s+(?:more\s+)?"
        rf"(?:detail|details|context))?",
        command,
    ) or _full(
        rf"make(?:\s+{_TARGET})?\s+(?:more\s+)?(?:longer|detailed|elaborate|comprehensive)",
        command,
    ):
        return _model("make_longer", "Expand the selected text with useful detail while preserving meaning.", command)
    if _full(rf"clarify(?:\s+{_TARGET})?", command) or _full(
        rf"(?:make\s+(?:{_TARGET}\s+)?(?:more\s+)?(?:clear|clearer|easier\s+to\s+understand)|"
        rf"improve(?:\s+{_TARGET})?\s+clarity)",
        command,
    ):
        return _model("make_clearer", "Improve clarity. Preserve meaning.", command)
    if _full(
        rf"(?:make|rewrite)(?:\s+{_TARGET})?\s+(?:more\s+)?"
        rf"(?:formal|professional|polished)",
        command,
    ):
        return _model("make_more_formal", "Rewrite in a formal, professional tone. Preserve meaning.", command)
    if _full(
        rf"(?:make|rewrite)(?:\s+{_TARGET})?\s+(?:more\s+)?"
        rf"(?:casual|informal|friendly|friendlier|conversational)",
        command,
    ):
        return _model("make_more_casual", "Rewrite in a casual, friendly tone. Preserve meaning.", command)

    summary_tail = (
        rf"(?:\s+(?:in|into|as)\s+{_COUNT}\s+(?:concise\s+)?"
        rf"(?:bullet\s+)?(?:points?|bullets?|sentences?))?"
    )
    if _full(
        rf"(?:summari[sz]e(?:\s+(?:{_TARGET}|this\s+whole\s+(?:chat|conversation|document)))?"
        rf"|(?:give\s+me|create)\s+a\s+summary(?:\s+of\s+(?:{_TARGET}))?){summary_tail}",
        command,
    ):
        return _model(
            "summarize",
            "Summarize the selected text into concise key points.",
            command,
            allows_list_output=True,
        )
    if _full(rf"simplify(?:\s+{_TARGET})?", command) or _full(
        rf"make(?:\s+{_TARGET})?\s+(?:simpler|easier\s+to\s+read)",
        command,
    ):
        return _model("simplify", "Simplify the selected text. Preserve meaning.", command)

    translated = _full(rf"translate(?:\s+{_TARGET})?\s+(?:to|into)\s+(?P<language>{_LANGUAGE})", command)
    if translated is not None:
        language = translated.group("language").strip()
        return _model(
            "translate_preserve_meaning",
            f"Translate faithfully to {language}. Preserve meaning and tone where possible.",
            command,
        )

    if _full(
        rf"(?:rewrite|turn|make|format)(?:\s+{_TARGET})?(?:\s+(?:into|as|for))?\s+(?:an?\s+)?email",
        command,
    ):
        return _model("email_rewrite", "Rewrite as a polished email.", command)
    if _full(
        rf"(?:rewrite|turn|make|format)(?:\s+{_TARGET})?(?:\s+(?:into|as|for))?\s+"
        rf"(?:an?\s+)?slack(?:\s+message)?",
        command,
    ):
        return _model("slack_rewrite", "Rewrite as a concise Slack message.", command)
    if _full(
        rf"(?:rewrite|turn|make|format)(?:\s+{_TARGET})?(?:\s+(?:into|as|for))?\s+"
        rf"(?:an?\s+)?(?:structured\s+)?notes?",
        command,
    ):
        return _model("notes_rewrite", "Rewrite as structured notes with bullets where helpful.", command, allows_list_output=True)
    if _full(
        rf"(?:rewrite|turn|make|format)(?:\s+{_TARGET})?(?:\s+(?:into|as|for))?\s+"
        rf"(?:an?\s+)?checklist",
        command,
    ):
        return _model("checklist_rewrite", "Rewrite as a short checklist.", command, allows_list_output=True)
    if _full(
        rf"(?:polish|improve|clean\s+up|rephrase|paraphrase|rewrite)(?:\s+{_TARGET})?"
        rf"|clean\s+{_TARGET}\s+up|polish\s+(?:the\s+)?wording",
        command,
    ):
        return _model("polish", "Polish wording and flow. Preserve meaning.", command)
    return None


def looks_like_selection_edit_command(text: str) -> bool:
    """Conservative risk signal used only to prevent selection overwrite."""

    command = normalize_selection_command(text)
    words = command.split()
    if not words or len(words) > 32:
        return False
    first = re.sub(r"[^a-z]", "", words[0].casefold())
    return first in _EDIT_OPENERS


def selection_command_allows_list_output(text: str, *, instruction: str = "") -> bool:
    command = recognize_selection_transform_command(text)
    if command is not None:
        return command.allows_list_output
    combined = f"{text}\n{instruction}"
    if re.search(
        r"\b(?:no|without)\s+(?:bullets?|lists?|numbering)\b|"
        r"\bdo\s+not\b.{0,48}\b(?:bullet|list|number)\b|"
        r"\b(?:as|in)\s+(?:a\s+)?(?:single\s+)?paragraph\b",
        combined,
        flags=re.I | re.S,
    ):
        return False
    return re.search(
        r"\b(?:summari[sz]e|key\s+points?|bullets?|bulleted|numbered\s+list|"
        r"checklist|structured\s+notes?)\b",
        combined,
        flags=re.I,
    ) is not None


def selection_command_requests_content_preserving_list(text: str, *, instruction: str = "") -> bool:
    command = recognize_selection_transform_command(text)
    if command is not None:
        return command.content_preserving_list
    combined = f"{text}\n{instruction}"
    if re.search(r"\b(?:summari[sz]e|extract|key\s+points?)\b", combined, flags=re.I):
        return False
    return re.search(
        r"\b(?:convert|turn|format|put|make|rewrite)\b.{0,80}"
        r"\b(?:bullets?|bullet\s+points?|bulleted\s+list|numbered\s+list)\b",
        combined,
        flags=re.I | re.S,
    ) is not None
