"""Meeting grammar engine — deterministic, no model, no cloud.

Voice-to-meeting-note transforms:

1. **Speaker tags.** Lines like ``Alice: hello team`` are promoted to
   ``**Alice:** hello team``. The speaker token is a single
   capitalised word or two capitalised words separated by a space.
   This keeps the heuristic tight: ``todo: ship it`` (lowercase) is
   *not* a speaker tag; ``TODO: ship it`` remains untouched too
   because the engine is case-sensitive — only Title-Case is treated
   as a name.
2. **Headings.** ``Action items`` / ``Action Items`` / ``Next step(s)``
   / ``Decisions`` on their own line become a bold heading.
3. **Bulleting.** Lines immediately *following* a detected heading are
   grouped under that heading as bullet items, until a blank line or
   another heading appears.
4. **Attendee extraction.** The first ``max_attendees`` distinct
   speaker names seen are surfaced on the result for downstream
   personalisation stores that key "session entities" by speaker.

All operations are pure / deterministic. No state is carried between
calls. Runs server-side before any writer-model rewrite so the model
sees clean, structured text instead of raw dictation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict


class MeetingGrammarMode(str, Enum):
    """Named meeting-grammar modes."""

    AUTO = "auto"          # detect speakers + headings
    SPEAKERS_ONLY = "speakers_only"
    HEADINGS_ONLY = "headings_only"


class MeetingGrammarResultPayload(TypedDict):
    text: str
    original_text: str
    mode: str
    changed: bool
    rules_applied: list[str]
    attendees: list[str]
    action_items: list[str]


@dataclass(slots=True)
class MeetingGrammarResult:
    """Output of :meth:`MeetingGrammarEngine.apply`."""

    text: str
    original_text: str
    mode: str
    changed: bool
    rules_applied: list[str] = field(default_factory=list)
    attendees: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)

    def to_dict(self) -> MeetingGrammarResultPayload:
        return {
            "text": self.text,
            "original_text": self.original_text,
            "mode": self.mode,
            "changed": self.changed,
            "rules_applied": list(self.rules_applied),
            "attendees": list(self.attendees),
            "action_items": list(self.action_items),
        }


# ---------------------------------------------------------------------------
# Detection heuristics
# ---------------------------------------------------------------------------

# Speaker: word(s) in Title Case, followed by ':'.
# - Single token: "Alice:"  — matches
# - Two tokens:   "Alice Chen:" — matches
# - Three+ tokens: not matched, keeps the heuristic conservative
# - Lowercase tokens ("todo: ship it") are rejected — the first letter
#   must be uppercase.
# - All-caps tokens ("TODO: ship it") are also rejected — we require at
#   least one lowercase character in each name token so SCREAMING_SNAKE
#   identifiers and acronyms don't get treated as speakers.
_SPEAKER_RE = re.compile(
    r"^\s*(?P<name>[A-Z][a-z][a-zA-Z'\-]*(?:\s+[A-Z][a-z][a-zA-Z'\-]*)?)\s*:\s*"
    r"(?P<body>\S.*)$"
)

# Headings. Case-insensitive, optional trailing punctuation.
_HEADING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*action\s*items?\s*[:\.]?\s*$", re.IGNORECASE), "Action items"),
    (re.compile(r"^\s*next\s*steps?\s*[:\.]?\s*$", re.IGNORECASE), "Next steps"),
    (re.compile(r"^\s*decisions?\s*[:\.]?\s*$", re.IGNORECASE), "Decisions"),
    (re.compile(r"^\s*agenda\s*[:\.]?\s*$", re.IGNORECASE), "Agenda"),
)

# Content-side detection for ``detect_meeting_text`` (a separate helper
# the transform runner / app classifier can consult when the surface
# category is ambiguous). Requires at least one speaker tag *or* a
# heading — a generic doc with none of these is not a meeting note.
# Detection hint-regex.
#
# The heading alternation uses an inline ``(?i:...)`` group so it stays
# case-insensitive ("Action Items" / "ACTION ITEMS" / "action items" all
# match). The speaker alternation keeps the stricter Title-Case rule
# from _SPEAKER_RE so SCREAMING_CASE identifiers (``TODO:``) don't
# trigger meeting-mode in the detector.
_MEETING_HINT_RE = re.compile(
    r"(?i:^\s*(action\s+items?|next\s+steps?|decisions?|agenda)\s*[:\.]?\s*$)"
    r"|(^\s*[A-Z][a-z][a-zA-Z'\-]*(?:\s+[A-Z][a-z][a-zA-Z'\-]*)?\s*:\s+\S)",
    re.MULTILINE,
)


def detect_meeting_text(text: str) -> bool:
    """Return True iff *text* looks like a meeting transcript.

    Used by the transform runner when the surface doesn't provide a
    strong app category (e.g. "docs" could be a personal journal or
    meeting notes). The rule is deliberately narrow: at least one
    speaker tag or explicit meeting heading must be present.
    """
    if not text:
        return False
    return _MEETING_HINT_RE.search(text) is not None


def _match_heading(line: str) -> str | None:
    for pat, canonical in _HEADING_PATTERNS:
        if pat.match(line):
            return canonical
    return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MeetingGrammarEngine:
    """Apply meeting-output grammar to a text string.

    ``max_attendees`` caps attendee extraction so a wildly long
    transcript doesn't produce an unbounded list.
    """

    def __init__(self, *, max_attendees: int = 32) -> None:
        self.max_attendees = max_attendees

    def apply(
        self,
        text: str,
        *,
        mode: MeetingGrammarMode | str = MeetingGrammarMode.AUTO,
    ) -> MeetingGrammarResult:
        if isinstance(mode, str):
            try:
                mode = MeetingGrammarMode(mode)
            except ValueError:
                mode = MeetingGrammarMode.AUTO

        raw = text or ""
        lines = raw.splitlines()
        out_lines: list[str] = []
        rules_applied: list[str] = []
        attendees: list[str] = []
        action_items: list[str] = []

        # Tracks the most recently emitted heading so lines following
        # it are collected as bullet items under it.
        current_heading: str | None = None

        def _remember_attendee(name: str) -> None:
            if name and name not in attendees and len(attendees) < self.max_attendees:
                attendees.append(name)

        for line in lines:
            stripped = line.strip()

            if not stripped:
                out_lines.append("")
                current_heading = None
                continue

            # Heading detection runs first so a heading that looks
            # like a speaker ("Decisions: we ship tomorrow") is
            # promoted as a heading-with-inline-content instead.
            if mode in (MeetingGrammarMode.AUTO, MeetingGrammarMode.HEADINGS_ONLY):
                heading_name = _match_heading(stripped)
                if heading_name is not None:
                    out_lines.append(f"**{heading_name}**")
                    current_heading = heading_name
                    if "headings" not in rules_applied:
                        rules_applied.append("headings")
                    continue

            if mode in (MeetingGrammarMode.AUTO, MeetingGrammarMode.SPEAKERS_ONLY):
                m = _SPEAKER_RE.match(stripped)
                if m is not None:
                    name = m.group("name")
                    body = m.group("body")
                    out_lines.append(f"**{name}:** {body}")
                    _remember_attendee(name)
                    if "speakers" not in rules_applied:
                        rules_applied.append("speakers")
                    # A speaker line terminates any heading-run.
                    current_heading = None
                    continue

            # Plain line. If we're in a heading run, convert it to a
            # bullet; otherwise keep it as-is.
            if current_heading is not None and mode in (
                MeetingGrammarMode.AUTO,
                MeetingGrammarMode.HEADINGS_ONLY,
            ):
                item_text = stripped.lstrip("-*• ").rstrip(".")
                if item_text:
                    out_lines.append(f"- {item_text}")
                    if current_heading == "Action items":
                        action_items.append(item_text)
                    if "bullets" not in rules_applied:
                        rules_applied.append("bullets")
                else:
                    out_lines.append(line)
                continue

            out_lines.append(line)

        rendered = "\n".join(out_lines).rstrip("\n")
        changed = rendered != raw
        # Preserve a trailing newline if the input had one — keeps
        # diffs stable in apps that require the final newline.
        if raw.endswith("\n") and not rendered.endswith("\n"):
            rendered = rendered + "\n"

        return MeetingGrammarResult(
            text=rendered,
            original_text=raw,
            mode=mode.value,
            changed=changed,
            rules_applied=rules_applied,
            attendees=attendees,
            action_items=action_items,
        )


__all__ = [
    "MeetingGrammarEngine",
    "MeetingGrammarMode",
    "MeetingGrammarResult",
    "detect_meeting_text",
]
