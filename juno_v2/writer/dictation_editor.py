"""Dictation editor — the AI lane for every dictation turn.

One cached-prefix model pass per utterance. The model never re-types the
transcript: it emits a compact, line-based edit script anchored to exact
source phrases (corrections, deletions of restarts/fillers, optional
structure). Application is deterministic with grounding guards, so output
cost is flat in utterance length and a malformed script can only ever fall
back to the deterministic floor — never corrupt the paste.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from juno_v2.list_content import analyze_list_prefix, protect_list_render
from juno_v2.memory.entity_policy import common_english_single_word

DICTATION_EDIT_TASK = "dictation_edit_v1"

# Static system prompt — this exact string is the KV-cached prefix. Keep it
# lean (~<1k tokens): every token here costs ~150KB of resident KV memory.
DICTATION_EDIT_SYSTEM_PROMPT = (
    "You are Juno's dictation editor. You receive a voice-dictated transcript "
    "and emit a minimal edit script. You never rewrite or re-type the "
    "transcript. You only emit the lines below, nothing else.\n"
    "\n"
    "Output lines (each optional except VERDICT):\n"
    "VERDICT: clean | edited\n"
    'EDIT: "<exact phrase from transcript>" => "<replacement>"\n'
    'DELETE: "<exact phrase from transcript>"\n'
    "STRUCT: numbered | bulleted | lettered\n"
    'TITLE: "<short title>"\n'
    'ITEM: "<exact phrase where an item starts>"\n'
    "\n"
    "Rules:\n"
    "- Phrases must be copied exactly from the transcript (8 words max).\n"
    "- EDIT fixes a misrecognized word/phrase using the context and the "
    "provided known terms (e.g. a name heard wrong, 'God' for 'got').\n"
    "- DELETE removes only false starts, abandoned rephrases, and filler "
    "sounds (per the filler policy given). Never delete a complete clause "
    "that carries meaning, emphasis, or opinion — keep the speaker's words "
    "even when redundant or blunt.\n"
    "- Spoken self-corrections: when the speaker replaces something "
    "('scratch that', 'no wait', 'actually'), DELETE the abandoned words "
    "and the correction marker, keep the replacement.\n"
    "- STRUCT only when the speaker clearly dictates a list (announced "
    "counts like 'four things', enumerations like 'first... second...', "
    "'a, b, c'). Emit one ITEM per spoken item, anchored at the phrase "
    "that starts the item's content (not the ordinal word). Never invent "
    "items that were not spoken, even if a count was announced.\n"
    "- When the transcript is already right, emit exactly: VERDICT: clean\n"
    "- Punctuation-only fixes: small EDITs are allowed (e.g. comma splice).\n"
    "\n"
    "Example 1:\n"
    "Transcript: I told him we got the budget approved, I mean the headcount approved, so we can hire.\n"
    "VERDICT: edited\n"
    'DELETE: "the budget approved, I mean"\n'
    "\n"
    "Example 2:\n"
    "Transcript: I am thinking about three things, a, get the deck done, b, ship the fix, c, email Sam.\n"
    "VERDICT: edited\n"
    "STRUCT: lettered\n"
    'ITEM: "get the deck done"\n'
    'ITEM: "ship the fix"\n'
    'ITEM: "email Sam"\n'
    'DELETE: "I am thinking about three things,"\n'
    "\n"
    "Example 3:\n"
    "Transcript: send the brief to Mira tonight\n"
    "VERDICT: clean\n"
)

_LINE_RE = re.compile(
    r"^(?P<key>VERDICT|EDIT|DELETE|STRUCT|TITLE|ITEM)\s*:\s*(?P<rest>.+?)\s*$",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r'"(.*?)"')
_MAX_PHRASE_WORDS = 12
_MAX_OPS = 24
# An edit script that changes more than this share of the text is more
# likely a model failure than a plausible set of dictation fixes.
_MAX_CHANGE_RATIO = 0.45
_ITEM_HEAD_RE = re.compile(
    r"^(?:and\s+|also\s+|then\s+)?(?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|next|finally|lastly)\b[\s,.:;-]*|[a-z][.)]\s+|\d{1,2}[.)]\s+)?",
    re.IGNORECASE,
)


@dataclass(slots=True)
class EditScript:
    verdict: str = "clean"
    edits: list[tuple[str, str]] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    struct: str | None = None
    title: str | None = None
    items: list[str] = field(default_factory=list)

    @property
    def has_ops(self) -> bool:
        return bool(self.edits or self.deletes or (self.struct and self.items))


def parse_edit_script(raw: str) -> EditScript | None:
    """Parse model output. Returns None when nothing usable was emitted."""
    if not raw or not raw.strip():
        return None
    script = EditScript()
    saw_verdict = False
    ops = 0
    for line in raw.splitlines():
        m = _LINE_RE.match(line.strip())
        if m is None:
            continue
        key = m.group("key").upper()
        rest = m.group("rest").strip()
        if key == "VERDICT":
            v = rest.strip().strip('"').lower()
            if v in {"clean", "edited"}:
                script.verdict = v
                saw_verdict = True
        elif key == "EDIT":
            quoted = _QUOTED_RE.findall(rest)
            if len(quoted) >= 2 and quoted[0].strip():
                script.edits.append((quoted[0], quoted[1]))
                ops += 1
        elif key == "DELETE":
            quoted = _QUOTED_RE.findall(rest)
            if quoted and quoted[0].strip():
                script.deletes.append(quoted[0])
                ops += 1
        elif key == "STRUCT":
            v = rest.strip().strip('"').lower()
            if v in {"numbered", "bulleted", "lettered"}:
                script.struct = v
        elif key == "TITLE":
            quoted = _QUOTED_RE.findall(rest)
            script.title = (quoted[0] if quoted else rest).strip()
        elif key == "ITEM":
            quoted = _QUOTED_RE.findall(rest)
            if quoted and quoted[0].strip():
                script.items.append(quoted[0])
                ops += 1
        if ops > _MAX_OPS:
            return None
    if not saw_verdict and not script.has_ops:
        return None
    return script


def _find_phrase(text: str, phrase: str, *, start: int = 0) -> tuple[int, int] | None:
    """Case-insensitive, word-boundary-aligned location of ``phrase``."""
    needle = re.sub(r"\s+", " ", phrase.strip())
    if not needle or len(needle.split()) > _MAX_PHRASE_WORDS:
        return None
    pattern = re.compile(
        r"(?<![\w])" + re.escape(needle).replace(r"\ ", r"\s+") + r"(?![\w])",
        re.IGNORECASE,
    )
    m = pattern.search(text, start)
    if m is None and needle[-1] in ".,;:!?":
        return _find_phrase(text, needle[:-1], start=start)
    return (m.start(), m.end()) if m else None


_DELETE_MARKER_RE = re.compile(
    r"\b(?:scratch\s+that|no\s+wait|no\s+no|i\s+mean|actually|rather|sorry|um+|uh+|erm)\b",
    re.IGNORECASE,
)


def _norm_tokens(text: str) -> list[str]:
    return [t for t in re.sub(r"[^\w\s']", " ", text.casefold()).split() if t]


def _delete_is_evidenced(
    text: str, start: int, end: int, *, first_item_start: int | None
) -> bool:
    """DELETE needs deterministic evidence that the span is not content.

    Allowed: spans containing a spoken correction marker or filler; tiny
    stumbles (≤2 tokens); restarts (the following words re-speak the span's
    opening); and list-announcement heads consumed by structure rendering.
    """
    span = text[start:end]
    tokens = _norm_tokens(span)
    if _DELETE_MARKER_RE.search(span):
        return True
    if first_item_start is not None and start < first_item_start:
        # A structural edit may remove only the proven list announcement and
        # ordinal syntax. The old broad ``start < first_item_start`` exception
        # also authorized deletion of an unrelated opening sentence.
        prefix = analyze_list_prefix(text[:first_item_start])
        if (
            prefix.safe
            and prefix.removable_start is not None
            and start >= prefix.removable_start
            and end <= first_item_start
        ):
            return True
        return False
    if len(tokens) <= 2:
        return True
    following = _norm_tokens(text[end : end + 160])
    if len(tokens) >= 2 and len(following) >= 2:
        # Restart evidence: the continuation re-speaks the abandoned
        # opening ("send it to the budget … send it to the headcount").
        for offset in range(0, min(4, len(following) - 1)):
            if following[offset : offset + 2] == tokens[:2]:
                return True
        # Replacement evidence: the words right after the span overlap its
        # tail strongly ("the budget approved" → "the headcount approved").
        if len(set(tokens[-3:]) & set(following[:6])) >= 2:
            return True
    return False


def apply_edit_script(
    source: str,
    script: EditScript,
    *,
    protected_terms: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Deterministically apply a parsed script. None ⇒ caller uses the floor."""
    text = source
    applied = {"edits": 0, "deletes": 0, "skipped": 0, "struct": None}

    spans: list[tuple[int, int, str]] = []
    for phrase, replacement in script.edits:
        loc = _find_phrase(text, phrase)
        if loc is None or len(replacement) > max(24, 3 * len(phrase)):
            applied["skipped"] += 1
            continue
        if _edit_drops_content_without_evidence(text, loc[0], loc[1], phrase, replacement):
            applied["skipped"] += 1
            continue
        if _case_only_edit_without_evidence(
            text,
            loc[0],
            phrase,
            replacement,
            protected_terms=protected_terms,
        ):
            applied["skipped"] += 1
            continue
        spans.append((loc[0], loc[1], replacement))
    first_item_start: int | None = None
    if script.struct and script.items:
        loc = _find_phrase(text, script.items[0])
        if loc is not None:
            first_item_start = loc[0]
    for phrase in script.deletes:
        loc = _find_phrase(text, phrase)
        if loc is None:
            applied["skipped"] += 1
            continue
        if not _delete_is_evidenced(text, loc[0], loc[1], first_item_start=first_item_start):
            # The model wanted to drop real content ("that is just not
            # acceptable behavior" — production over-delete). Without
            # deterministic restart/filler evidence, the speaker's words win.
            applied["skipped"] += 1
            continue
        spans.append((loc[0], loc[1], ""))

    # Right-to-left, drop overlaps.
    spans.sort(key=lambda s: s[0], reverse=True)
    last_start = len(text) + 1
    for start, end, repl in spans:
        if end > last_start:
            applied["skipped"] += 1
            continue
        text = text[:start] + repl + text[end:]
        last_start = start
        applied["edits" if repl else "deletes"] += 1
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text).strip(" ,")

    if script.struct and script.items:
        anchors: list[int] = []
        pos = 0
        for item in script.items:
            loc = _find_phrase(text, item, start=pos)
            if loc is None:
                continue
            anchors.append(loc[0])
            pos = loc[0] + 1
        if len(anchors) >= 2:
            struct_source = text
            raw_head = text[: anchors[0]]
            prefix = analyze_list_prefix(raw_head)
            head = (
                prefix.substantive_prefix
                if prefix.safe
                else raw_head.strip(" ,.;:-")
            )
            # A head that is just a spoken item marker ("a", "1", "first")
            # belongs to the first item, not to a heading line.
            if head and re.fullmatch(r"(?:[a-z]|\d{1,2}|first|firstly)[.)]?", head, re.IGNORECASE):
                head = ""
            chunks = [
                text[a:b].strip() for a, b in zip(anchors, anchors[1:] + [len(text)])
            ]
            lines: list[str] = []
            if head:
                lines.append(head)
            if script.title:
                lines.append(script.title.strip())
            for idx, chunk in enumerate(chunks):
                if idx < len(chunks) - 1:
                    # The spoken marker of the NEXT item ("…done, b") trails
                    # the current chunk — strip it.
                    chunk = re.sub(
                        r"[\s,;.]+(?:and\s+|also\s+|then\s+)?(?:[a-z]|\d{1,2})?[\s,.;:]*$",
                        "",
                        chunk,
                        flags=re.IGNORECASE,
                    )
                body = _ITEM_HEAD_RE.sub("", chunk, count=1).strip(" ,.;:-")
                if not body:
                    continue
                if script.struct == "numbered":
                    marker = f"{idx + 1}."
                elif script.struct == "lettered":
                    marker = f"{chr(ord('a') + idx)}."
                else:
                    marker = "-"
                lines.append(f"{marker} {body}")
            if len(lines) >= 2:
                protected = protect_list_render(struct_source, "\n".join(lines))
                text = protected.text
                applied["struct_content_preservation"] = protected.mode
                if protected.mode != "complete_transcript_fallback":
                    applied["struct"] = script.struct

    ratio = difflib.SequenceMatcher(a=source, b=text, autojunk=False).ratio()
    if applied["struct"] is None and ratio < (1.0 - _MAX_CHANGE_RATIO):
        return None
    if not text.strip():
        return None
    return text, applied


def _edit_drops_content_without_evidence(
    text: str,
    start: int,
    end: int,
    phrase: str,
    replacement: str,
) -> bool:
    """Reject model EDITs that silently compress meaningful spoken content."""
    old_tokens = _norm_tokens(phrase)
    new_tokens = _norm_tokens(replacement)
    if len(new_tokens) >= len(old_tokens):
        return False
    # Keep evidenced retakes/fillers available even when the model emits EDIT
    # rather than DELETE for the abandoned phrase.
    if _DELETE_MARKER_RE.search(phrase):
        return False
    if _edit_shrink_has_retake_evidence(text, start, end, old_tokens):
        return False
    # Collapsing a local stutter is content-preserving, even though it shrinks.
    if _collapse_adjacent_duplicate_tokens(old_tokens) == new_tokens:
        return False
    # Pure subset edits are the common failure mode: the model keeps a few
    # source words and silently drops the rest ("make a new" -> "new").
    if _tokens_are_ordered_subset(new_tokens, old_tokens):
        return True
    # If the replacement also introduces new words, allow small corrections but
    # reject broad compressions without deterministic retake evidence.
    if len(old_tokens) >= 4 and len(new_tokens) <= len(old_tokens) - 2:
        return True
    return False


def _tokens_are_ordered_subset(candidate: list[str], source: list[str]) -> bool:
    if not candidate:
        return bool(source)
    pos = 0
    for token in source:
        if pos < len(candidate) and candidate[pos] == token:
            pos += 1
    return pos == len(candidate)


def _collapse_adjacent_duplicate_tokens(tokens: list[str]) -> list[str]:
    collapsed: list[str] = []
    for token in tokens:
        if not collapsed or collapsed[-1] != token:
            collapsed.append(token)
    return collapsed


def _edit_shrink_has_retake_evidence(
    text: str,
    start: int,
    end: int,
    tokens: list[str],
) -> bool:
    if len(tokens) < 2:
        return False
    following = _norm_tokens(text[end : end + 160])
    if len(following) < 2:
        return False
    for offset in range(0, min(4, len(following) - 1)):
        if following[offset : offset + 2] == tokens[:2]:
            return True
    return len(set(tokens[-3:]) & set(following[:6])) >= 2


def _case_only_edit_without_evidence(
    source: str,
    start: int,
    phrase: str,
    replacement: str,
    *,
    protected_terms: list[str] | tuple[str, ...] | None,
) -> bool:
    """Reject model-only capitalization of ordinary words inside prose.

    Name/protected-term casing is useful, but screen/context junk can nudge the
    editor to capitalize ordinary words. Sentence-start casing is handled by
    the deterministic postpass, so mid-sentence common-word case edits need
    exact protected-term evidence.
    """
    old = (phrase or "").strip()
    new = (replacement or "").strip()
    if not old or not new or old == new:
        return False
    if _casefold_alnum(old) != _casefold_alnum(new):
        return False
    if _protected_term_exact_match(new, protected_terms):
        return False
    if _identifier_case_shape(new):
        return False
    if _at_sentence_start(source, start):
        return False
    tokens = _norm_tokens(old)
    if len(tokens) != 1:
        return False
    return common_english_single_word(tokens[0])


def _casefold_alnum(text: str) -> str:
    return re.sub(r"[^\w']+", "", (text or "").casefold())


def _protected_term_exact_match(
    replacement: str,
    protected_terms: list[str] | tuple[str, ...] | None,
) -> bool:
    target = _casefold_alnum(replacement)
    if not target:
        return False
    for term in protected_terms or ():
        if _casefold_alnum(str(term or "")) == target:
            return True
    return False


def _identifier_case_shape(text: str) -> bool:
    core = re.sub(r"[^A-Za-z0-9_./-]+", "", text or "")
    if not core:
        return False
    if any(ch.isdigit() for ch in core) and any(ch.isalpha() for ch in core):
        return True
    letters = [ch for ch in core if ch.isalpha()]
    return len(letters) >= 2 and all(ch.isupper() for ch in letters)


def _at_sentence_start(source: str, start: int) -> bool:
    before = (source or "")[: max(0, start)].rstrip()
    if not before:
        return True
    return before[-1] in ".!?\n"


_ABBREV_BEFORE_PERIOD_RE = re.compile(r"\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|st|no|inc|ltd)$", re.IGNORECASE)
_SENTENCE_START_RE = re.compile(r"([.!?][\"')\]]*\s+)([a-z])")


def capitalize_sentence_starts(text: str) -> str:
    """Deterministic sentence-casing for prose pastes ("places. more" →
    "places. More"). Abbreviation-aware; never touches anything but the
    first letter after terminal punctuation."""

    def _repl(match: re.Match[str]) -> str:
        sep, letter = match.group(1), match.group(2)
        before = text[: match.start()]
        if _ABBREV_BEFORE_PERIOD_RE.search(before[-8:]):
            return match.group(0)
        return sep + letter.upper()

    return _SENTENCE_START_RE.sub(_repl, text or "")


def build_editor_suffix(
    *,
    transcript: str,
    app_name: str | None,
    app_category: str | None,
    known_terms: list[str],
    filler_policy: str,
    style_hint: str | None,
) -> str:
    parts = [
        f"App: {app_name or 'unknown'} | Category: {app_category or 'unknown'} | "
        f"Filler policy: {filler_policy}",
    ]
    if known_terms:
        parts.append("Known terms: " + ", ".join(known_terms[:24]))
    if style_hint:
        parts.append(f"Style: {style_hint[:160]}")
    parts.append("Transcript:\n" + transcript)
    return "\n".join(parts)


_HESITATION_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s,;:.!?(\[]))(?:um+|uh+|hm+m*|mm-?hm+|mhm+|erm+|er|ah+|a{3,})(?=$|[\s,;:.!?)\]])",
    re.IGNORECASE,
)
_HESITATION_LITERAL_GUARD_RE = re.compile(r"(?:\bwords?\s+|[\"'“‘]\s*)$", re.IGNORECASE)
FILLER_STRIP_MODES = {"formal_email", "casual_chat", "structured_notes"}


def strip_hesitation_fillers(text: str) -> str:
    """Remove standalone spoken hesitations (um/uh/hmm/aaa…) from prose.

    Mode-gated by the caller (never verbatim/code/terminal). Literal guards:
    quoted fillers and "the word um" stay.
    """
    source = text or ""

    def _repl(match: re.Match[str]) -> str:
        if _HESITATION_LITERAL_GUARD_RE.search(source[: match.start()]):
            return match.group(0)
        return ""

    out = _HESITATION_RE.sub(_repl, source)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*\1+", r"\1", out)
    out = re.sub(r"^[\s,.;:]+", "", out)
    return out.strip()
