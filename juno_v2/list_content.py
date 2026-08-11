"""Content-preservation guards for spoken list formatting.

List renderers may remove a proven announcement ("there are two things") and
ordinal syntax ("first is that"), but every other spoken word must survive.
This module is deliberately independent of the writer and turn-plan packages
so each delivery lane can apply the same final check without import cycles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypeAlias


SPOKEN_LIST_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
SPOKEN_LIST_ORDINALS = {
    "first": 1,
    "firstly": 1,
    "second": 2,
    "secondly": 2,
    "third": 3,
    "thirdly": 3,
    "fourth": 4,
    "fourthly": 4,
    "fifth": 5,
    "fifthly": 5,
    "sixth": 6,
    "sixthly": 6,
    "seventh": 7,
    "seventhly": 7,
    "eighth": 8,
    "eighthly": 8,
    "ninth": 9,
    "ninthly": 9,
    "tenth": 10,
    "tenthly": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}
SPOKEN_LIST_NUMBER_PATTERN = (
    r"(?:\d{1,2}|" + "|".join(SPOKEN_LIST_NUMBER_WORDS) + r")"
)
SPOKEN_LIST_ORDINAL_PATTERN = (
    r"(?:"
    + "|".join(sorted(SPOKEN_LIST_ORDINALS, key=len, reverse=True))
    + r"|\d{1,2}(?:st|nd|rd|th)?)"
)
SPOKEN_LIST_COUNT_NOUN_PATTERN = (
    r"(?:things?|points?|items?|steps?|reasons?|priorities|topics?|goals?|"
    r"tasks?|takeaways?|focus\s+areas?)"
)
SPOKEN_LIST_ITEM_LABELS = (
    ("thing",),
    ("point",),
    ("item",),
    ("step",),
    ("reason",),
    ("priority",),
    ("topic",),
    ("goal",),
    ("task",),
    ("takeaway",),
    ("focus", "area"),
)
SPOKEN_LIST_ITEM_LABEL_PATTERN = (
    r"(?:"
    + "|".join(r"\s+".join(re.escape(token) for token in label) for label in SPOKEN_LIST_ITEM_LABELS)
    + r")"
)
SPOKEN_LIST_CONNECTORS = ("and", "then", "plus")
SPOKEN_LIST_CONNECTOR_PATTERN = r"(?:" + "|".join(SPOKEN_LIST_CONNECTORS) + r")"
SPOKEN_LIST_SEQUENCE_MARKER_PATTERN = (
    rf"(?:number\s+{SPOKEN_LIST_NUMBER_PATTERN}|{SPOKEN_LIST_ORDINAL_PATTERN})"
)
SPOKEN_LIST_FIRST_MARKER_PATTERN = (
    r"(?:number\s+(?:one|1)|first(?:ly)?|1st|a(?:[.)]|(?=[,;:])))"
)

ListPreservationMode: TypeAlias = Literal[
    "list_rendered",
    "substantive_prefix_preserved",
    "complete_transcript_fallback",
]


# Python's Unicode-aware ``\w`` keeps multilingual dictation visible to the
# preservation guard. Excluding underscore keeps identifiers tokenized like
# ordinary words while retaining internal apostrophes.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_COUNT_RE = re.compile(
    rf"\b{SPOKEN_LIST_NUMBER_PATTERN}\s+"
    rf"(?:{SPOKEN_LIST_COUNT_NOUN_PATTERN}|bullets?|bullet\s+points?)\b",
    re.IGNORECASE,
)
_TRAILING_FIRST_MARKER_RE = re.compile(
    r"(?:^|[\s,.:;!?-]+)"
    rf"(?P<marker>{SPOKEN_LIST_FIRST_MARKER_PATTERN})"
    rf"(?:\s+{SPOKEN_LIST_ITEM_LABEL_PATTERN})?"
    r"(?:\s+(?:is|are|was|were)(?:\s+(?:that|to))?|\s+(?:that|to))?"
    r"[\s,.:;!?-]*$",
    re.IGNORECASE,
)
_EXPLICIT_ANNOUNCEMENT_RE = re.compile(
    r"\b(?:(?:please|(?:could|can|would)\s+you|i\s+(?:want|need)\s+you\s+to)\s+)?"
    r"(?:"
    r"(?:start|begin)\s+(?:a\s+)?(?:bullet\s+list|bullets?|bulleted\s+list)"
    r"|(?:note\s+down|write\s+down|write|list|make|create|capture|give\s+me|"
    r"put\s+down)\b[^.!?;]{0,160}\b"
    rf"(?:{SPOKEN_LIST_COUNT_NOUN_PATTERN}|bullets?|bullet\s+points?|"
    r"checklist|numbered\s+list|list)\b"
    r"(?:\s+to\s+(?:do|cover|discuss|address|handle|remember|consider)"
    r"(?:\s+(?:today|tonight|tomorrow|now|this\s+(?:week|month|morning|afternoon)))?"
    r")?"
    r")$",
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
_GAP_CONNECTORS = (
    ("and", "then"),
    ("and", "also"),
    ("and",),
    ("also",),
    ("then",),
    ("plus",),
)
_GAP_LINKS = {
    (),
    ("is",),
    ("is", "that"),
    ("is", "to"),
    ("are",),
    ("are", "that"),
    ("are", "to"),
    ("was",),
    ("was", "that"),
    ("was", "to"),
    ("were",),
    ("were", "that"),
    ("were", "to"),
    ("that",),
    ("to",),
}


@dataclass(frozen=True, slots=True)
class ListPrefixAnalysis:
    safe: bool
    substantive_prefix: str
    removable_start: int | None


@dataclass(frozen=True, slots=True)
class ListContentResult:
    text: str
    mode: ListPreservationMode


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

    explicit_matches = _explicit_announcement_matches(core)
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
                if _starts_clause(before_count, match.start())
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
            raw[: marker.start("marker")].strip(" \t\r\n,;:-"),
            marker.start("marker"),
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
    if not analysis.safe:
        return ListContentResult(source, "complete_transcript_fallback")
    substantive = analysis.substantive_prefix
    prefix_present = _tokens_are_ordered_subset(
        _tokens(substantive),
        _tokens(surface_prefix),
    )

    gaps = [
        source[located[index][1] : located[index + 1][0]]
        for index in range(len(located) - 1)
    ]
    gaps_are_structural = all(
        _gap_is_structural(gap, item_number=index + 2)
        for index, gap in enumerate(gaps)
    )
    if not gaps_are_structural and not (
        _prefix_has_explicit_list_command(prefix)
        and _gaps_are_unpunctuated_ordered_ordinals(gaps)
    ):
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


def _gap_is_structural(gap: str, *, item_number: int) -> bool:
    """Accept only a complete marker phrase for the next rendered item.

    A bag-of-words check cannot distinguish ``second priority is`` (structure)
    from ``priority. Second`` (omitted content). Parse the expected marker from
    the start of the gap and reject every unconsumed or ambiguous word.
    """

    tokens = tuple(_tokens(gap))
    if not tokens:
        return True
    marker_boundary_proven = re.match(r"\s*[.,;:!?-]", gap) is not None
    for connector in _GAP_CONNECTORS:
        if tokens[: len(connector)] == connector:
            tokens = tokens[len(connector) :]
            marker_boundary_proven = True
            break
    if re.fullmatch(r"\s*(?:\d{1,2}|[a-z])[.)]\s*", gap, re.IGNORECASE):
        marker_boundary_proven = True

    marker_length = _expected_marker_length(
        tokens,
        item_number=item_number,
        boundary_proven=marker_boundary_proven,
    )
    if marker_length is None:
        return False
    remainder = tokens[marker_length:]
    if remainder in _GAP_LINKS:
        return True

    for label in SPOKEN_LIST_ITEM_LABELS:
        if remainder[: len(label)] != label:
            continue
        # A bare noun can be real item content ("second priority ship"). A
        # following link makes the label structural ("second priority is").
        return remainder[len(label) :] in _GAP_LINKS - {()}
    return False


def _expected_marker_length(
    tokens: tuple[str, ...],
    *,
    item_number: int,
    boundary_proven: bool,
) -> int | None:
    if not boundary_proven:
        return None
    markers: set[tuple[str, ...]] = {
        ("next",),
        ("finally",),
        ("lastly",),
        (str(item_number),),
        (_numeric_ordinal(item_number),),
    }
    number_word = next(
        (
            word
            for word, value in SPOKEN_LIST_NUMBER_WORDS.items()
            if value == item_number
        ),
        None,
    )
    if number_word is not None:
        markers.add(("number", number_word))
    markers.add(("number", str(item_number)))
    markers.update(
        (word,)
        for word, value in SPOKEN_LIST_ORDINALS.items()
        if value == item_number
    )
    if 1 <= item_number <= 26:
        markers.add((chr(ord("a") + item_number - 1),))

    for marker in sorted(markers, key=len, reverse=True):
        if tokens[: len(marker)] == marker:
            return len(marker)
    return None


def _numeric_ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _gaps_are_unpunctuated_ordered_ordinals(gaps: list[str]) -> bool:
    if len(gaps) < 2:
        return False
    for index, gap in enumerate(gaps):
        if re.search(r"[.,;:!?-]", gap) is not None:
            return False
        tokens = tuple(_tokens(gap))
        expected = {
            (word,)
            for word, value in SPOKEN_LIST_ORDINALS.items()
            if value == index + 2
        }
        expected.add((_numeric_ordinal(index + 2),))
        if tokens not in expected:
            return False
    return True


def _prefix_has_explicit_list_command(prefix: str) -> bool:
    marker = _TRAILING_FIRST_MARKER_RE.search(prefix)
    core_end = marker.start() if marker is not None else len(prefix)
    core = prefix[:core_end].rstrip(" \t\r\n,.:;!?-")
    return bool(_explicit_announcement_matches(core))


def _explicit_announcement_matches(text: str) -> list[re.Match[str]]:
    return [
        match
        for match in _EXPLICIT_ANNOUNCEMENT_RE.finditer(text)
        if _starts_clause(text, match.start())
    ]


def _starts_clause(text: str, start: int) -> bool:
    leading = text[:start]
    return not _tokens(leading) or re.search(r"[.!?;:]\s*$", leading) is not None


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
