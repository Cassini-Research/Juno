"""Content-preservation guards for spoken list formatting.

List renderers may remove a proven announcement ("there are two things") and
ordinal syntax ("first is that"), but every other spoken word must survive.
This module is deliberately independent of the writer and turn-plan packages
so each delivery lane can apply the same final check without import cycles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_COUNT_RE = re.compile(
    r"\b(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty)\s+"
    r"(?:things?|points?|items?|steps?|reasons?|priorities|topics?|goals?|"
    r"tasks?|takeaways?|focus\s+areas?|bullets?|bullet\s+points?)\b",
    re.IGNORECASE,
)
_TRAILING_FIRST_MARKER_RE = re.compile(
    r"(?:^|[\s,.:;!?-]+)"
    r"(?:number\s+(?:one|1)|first(?:ly)?|one|1(?:st)?|a[.)]?)"
    r"(?:\s+(?:one|thing|point|item|step|reason|priority|topic|goal|task|"
    r"takeaway|focus\s+area))?"
    r"(?:\s+(?:is|are|was|were)(?:\s+(?:that|to))?|\s+(?:that|to))?"
    r"[\s,.:;!?-]*$",
    re.IGNORECASE,
)
_EXPLICIT_ANNOUNCEMENT_RE = re.compile(
    r"\b(?:start|begin)\s+(?:a\s+)?(?:bullet\s+list|bullets?|bulleted\s+list)\b"
    r"|\b(?:note\s+down|write\s+down|write|list|make|create|capture|give\s+me|"
    r"put\s+down)\b[^.!?;]{0,160}\b(?:things?|points?|items?|steps?|bullets?|"
    r"bullet\s+points?|checklist|numbered\s+list|list)\b[^.!?;]{0,60}$",
    re.IGNORECASE,
)
_ANNOUNCEMENT_HEAD_RES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:and\s+|then\s+)?(?:i\s+(?:think|believe|feel)\s+)?there\s+"
        r"(?:are|is)(?:\s+(?:only|about|exactly))?\s*$",
        r"\b(?:and\s+|then\s+)?here\s+(?:are|is)(?:\s+(?:only|about|exactly))?\s*$",
        r"\b(?:and\s+|then\s+)?(?:i|we|you|they)\s+(?:have|got|see)"
        r"(?:\s+(?:only|about|exactly))?\s*$",
        r"\b(?:and\s+|then\s+)?(?:i\s+(?:think|believe|feel)\s+)?"
        r"(?:i|we|you|they)\s+(?:need|want|have|plan)\s+to\s+"
        r"(?:focus\s+on|handle|cover|discuss|address|prioriti[sz]e|talk\s+about)\s*$",
        r"\b(?:and\s+|then\s+)?(?:i|we)\s+(?:am|are)\s+"
        r"(?:thinking|talking)\s+about\s*$",
        r"\b(?:and\s+|then\s+)?(?:focus\s+on|focused\s+on|cover|talk\s+about|"
        r"discuss|handle|prioriti[sz]e)\s*$",
    )
)
_ANNOUNCEMENT_TRAILER_RE = re.compile(
    r"^[\s,.:;!?-]*(?:to\s+(?:do|cover|discuss|address|handle|remember|consider)"
    r"(?:\s+(?:today|tonight|tomorrow|now|this\s+(?:week|month|morning|afternoon)))?)?"
    r"[\s,.:;!?-]*$",
    re.IGNORECASE,
)
_LIST_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s+(?:\[\s?\]\s+)?|\[\s?\]\s+|"
    r"\d{1,2}[.)]\s+|[A-Za-z][.)]\s+)(?P<body>.+?)\s*$"
)
_STRUCTURAL_GAP_WORDS = {
    "and",
    "also",
    "then",
    "plus",
    "next",
    "finally",
    "lastly",
    "number",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "first",
    "firstly",
    "second",
    "secondly",
    "third",
    "thirdly",
    "fourth",
    "fourthly",
    "fifth",
    "fifthly",
    "sixth",
    "sixthly",
    "seventh",
    "seventhly",
    "eighth",
    "eighthly",
    "ninth",
    "ninthly",
    "tenth",
    "tenthly",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
    "thing",
    "point",
    "item",
    "step",
    "reason",
    "priority",
    "topic",
    "goal",
    "task",
    "takeaway",
    "focus",
    "area",
    "is",
    "are",
    "was",
    "were",
    "that",
    "to",
}


@dataclass(frozen=True, slots=True)
class ListPrefixAnalysis:
    safe: bool
    substantive_prefix: str
    removable_start: int | None


@dataclass(frozen=True, slots=True)
class ListContentResult:
    text: str
    mode: str


def analyze_list_prefix(prefix: str) -> ListPrefixAnalysis:
    """Separate substantive leading text from proven list syntax.

    ``prefix`` is everything before the first rendered item's body. The
    returned ``removable_start`` is an offset into the original prefix; an
    editor DELETE that starts before it would remove substantive text.
    """

    raw = str(prefix or "")
    marker = _TRAILING_FIRST_MARKER_RE.search(raw)
    core_end = marker.start() if marker is not None else len(raw)
    core = raw[:core_end].rstrip(" \t\r\n,.:;!?-")
    if not _tokens(core):
        return ListPrefixAnalysis(True, "", core_end)

    explicit_matches = list(_EXPLICIT_ANNOUNCEMENT_RE.finditer(core))
    if explicit_matches:
        announcement = explicit_matches[-1]
        if not _tokens(core[announcement.end() :]):
            return ListPrefixAnalysis(
                True,
                raw[: announcement.start()].strip(" \t\r\n,;:-"),
                announcement.start(),
            )

    count_matches = list(_COUNT_RE.finditer(core))
    if count_matches:
        count = count_matches[-1]
        if _ANNOUNCEMENT_TRAILER_RE.fullmatch(core[count.end() :]) is not None:
            before_count = core[: count.start()]
            heads = [
                match
                for pattern in _ANNOUNCEMENT_HEAD_RES
                for match in pattern.finditer(before_count)
            ]
            if heads:
                # Prefer the widest proven announcement. Narrow nested matches
                # such as "focus on" must not leave "I think we need to" as
                # a false substantive heading.
                announcement = min(heads, key=lambda match: match.start())
                return ListPrefixAnalysis(
                    True,
                    raw[: announcement.start()].strip(" \t\r\n,;:-"),
                    announcement.start(),
                )

    # An ordinal-only list ("Context first alpha, second beta") has no
    # announcement to remove. Preserve the complete pre-marker text.
    if marker is not None:
        return ListPrefixAnalysis(
            True,
            core.strip(" \t\r\n,;:-"),
            marker.start(),
        )
    return ListPrefixAnalysis(False, raw.strip(), None)


def protect_list_render(source_text: str, rendered_text: str) -> ListContentResult:
    """Preserve all non-structural source words in a rendered list.

    A safely separable substantive prefix is placed above the list. If item
    alignment or any inter-item gap is uncertain, the complete corrected
    transcript wins over structural formatting.
    """

    source = str(source_text or "").strip()
    rendered = str(rendered_text or "").strip()
    if not source:
        return ListContentResult(rendered, "list_rendered")
    parsed = _parse_list_surface(rendered)
    if parsed is None:
        return ListContentResult(source, "complete_transcript_fallback")
    surface_prefix, bodies = parsed
    source_matches = list(_TOKEN_RE.finditer(source))
    located: list[tuple[int, int]] = []
    cursor = 0
    for body in bodies:
        body_tokens = _tokens(body)
        if not body_tokens:
            return ListContentResult(source, "complete_transcript_fallback")
        found = _find_token_run(source_matches, body_tokens, start=cursor)
        if found is None:
            return ListContentResult(source, "complete_transcript_fallback")
        first, last = found
        located.append((source_matches[first].start(), source_matches[last].end()))
        cursor = last + 1

    prefix = source[: located[0][0]]
    analysis = analyze_list_prefix(prefix)
    substantive = analysis.substantive_prefix if analysis.safe else prefix.strip()
    prefix_present = _tokens_are_ordered_subset(_tokens(substantive), _tokens(surface_prefix))

    for index in range(len(located) - 1):
        gap = source[located[index][1] : located[index + 1][0]]
        if not _gap_is_structural(gap):
            return ListContentResult(source, "complete_transcript_fallback")
    if _tokens(source[located[-1][1] :]):
        return ListContentResult(source, "complete_transcript_fallback")

    if substantive and not prefix_present:
        return ListContentResult(
            f"{substantive.rstrip()}\n{rendered}",
            "substantive_prefix_preserved",
        )
    return ListContentResult(rendered, "list_rendered")


def list_render_omits_substantive_text(source_text: str, rendered_text: str) -> bool:
    """True when the rendered list needs any preservation repair or fallback."""

    if _parse_list_surface(str(rendered_text or "").strip()) is None:
        return False
    protected = protect_list_render(source_text, rendered_text)
    return protected.text.strip() != str(rendered_text or "").strip()


def _parse_list_surface(text: str) -> tuple[str, list[str]] | None:
    prefix_lines: list[str] = []
    bodies: list[str] = []
    list_started = False
    for line in str(text or "").splitlines():
        match = _LIST_LINE_RE.match(line)
        if match is not None:
            list_started = True
            body = match.group("body").strip()
            if body:
                bodies.append(body)
            continue
        if not line.strip():
            continue
        if list_started:
            return None
        prefix_lines.append(line.strip())
    if not bodies:
        return None
    return "\n".join(prefix_lines), bodies


def _find_token_run(
    source_matches: list[re.Match[str]],
    needle: list[str],
    *,
    start: int,
) -> tuple[int, int] | None:
    haystack = [match.group(0).casefold() for match in source_matches]
    for index in range(start, len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index, index + len(needle) - 1
    return None


def _gap_is_structural(gap: str) -> bool:
    tokens = _tokens(gap)
    for token in tokens:
        if token in _STRUCTURAL_GAP_WORDS:
            continue
        if re.fullmatch(r"[a-z]", token):
            continue
        if re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", token):
            continue
        return False
    return True


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text or ""))]


def _tokens_are_ordered_subset(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return True
    position = 0
    for token in haystack:
        if token == needle[position]:
            position += 1
            if position == len(needle):
                return True
    return False
