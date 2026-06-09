"""Deterministic writer primitives.

These are the transforms that never need a model — fast, offline,
testable. Layers, bottom up:

1. Casing / structure primitives (``render_bullets`` etc.)
2. Whitespace + punctuation normalisation (``normalize_plain_dictation``)
3. Newline policy. Spoken cues like "new line" or "new paragraph" become
   the corresponding whitespace, with app-category aware collapsing.
4. Snippet expansion. Expands user-defined triggers into their body text
   when the trigger appears as a standalone token.
5. App-category formatting. A thin presentation layer that knows a few
   categories (messaging / docs / email / code / terminal / unknown) and
   applies the minimum of each category's typographic norms.

Every layer is pure — no side effects, no I/O. The writer service composes
them.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Protocol

from juno_v2.memory.fold import fold_key, fold_match_pattern


# ---------------------------------------------------------------------- #
# Layer 1: legacy casing / structure primitives
# ---------------------------------------------------------------------- #


def render_bullets(text: str) -> str:
    items = _split_items(text)
    if not items:
        return text.strip()
    return "\n".join(f"- {item}" for item in items)


def render_numbered(text: str) -> str:
    items = _split_items(text)
    if not items:
        return text.strip()
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))


def render_uppercase(text: str) -> str:
    return (text or "").upper()


def render_lowercase(text: str) -> str:
    return (text or "").lower()


def render_title_case(text: str) -> str:
    return (text or "").title()


def _split_items(text: str) -> list[str]:
    compact = (text or "").replace("\r", " ").strip()
    if not compact:
        return []
    chunks: list[str] = []
    for raw in re.split(r"(?:\n+|[;•]|(?<=[.!?])\s+)", compact):
        item = raw.strip(" -•\t").rstrip(".!?")
        if item:
            chunks.append(item)
    # Comma fallback. When the strong separators above yield a single
    # chunk and the input contains list-shaped commas (≥1, every
    # resulting piece is non-empty), treat the commas as item
    # boundaries. ``render_bullets`` and ``render_numbered`` only run on
    # explicit user-triggered transforms, so being more liberal here
    # does not affect any auto-rendering path.
    if len(chunks) == 1 and "," in compact:
        comma_parts = [
            part.strip(" -•\t").rstrip(".!?")
            for part in compact.split(",")
        ]
        comma_parts = [p for p in comma_parts if p]
        if len(comma_parts) >= 2:
            return comma_parts
    return chunks


# ---------------------------------------------------------------------- #
# Layer 2: whitespace / punctuation normalisation (legacy)
# ---------------------------------------------------------------------- #


def normalize_plain_dictation(text: str) -> str:
    current = re.sub(r"\s+", " ", (text or "").strip())
    # Strip a stray space BEFORE punctuation ("hello , world" -> "hello, world").
    current = re.sub(r"\s+([,.;:!?])", r"\1", current)

    # Punctuation spacing rules. The historical single regex
    # `([,;:!?])(\S) -> \1 \2` was too aggressive and broke common
    # spoken/written tokens — most painfully:
    #   - "1,000"           -> "1, 000"            (comma in numbers)
    #   - "5:30 pm"         -> "5: 30 pm"          (time)
    #   - "1:1"             -> "1: 1"              (ratio / meeting)
    #   - "16:9"            -> "16: 9"             (aspect ratio)
    #   - "https://x.com"   -> "https: //x.com"    (URL scheme)
    #   - "no!!!"           -> "no! !!"            (multi-bang run)
    #   - "wait... what?"   -> "wait. .. what?"    (ellipsis)
    # The rules below are split per-character with negative lookaheads
    # for the cases each one needs to preserve. Existing
    # tests (test_normalize_fixes_punctuation_spacing,
    # test_normalize_adds_space_after_punctuation) still pass.

    # Comma: insert a space after, UNLESS followed by a digit
    # (preserves "1,000", "1,234,567", "5,600").
    current = re.sub(r",(?!\d)(\S)", r", \1", current)
    # Semicolon: always insert a space after. (Rare in dictation; no
    # known cases to preserve.)
    current = re.sub(r";(\S)", r"; \1", current)
    # Colon: insert a space after, UNLESS followed by a digit
    # (preserves "5:30", "1:1", "16:9") or "//" (preserves URL schemes
    # like "https://", "ws://").
    current = re.sub(r":(?!\d)(?!//)(\S)", r": \1", current)
    # Bang / question: insert a space after, UNLESS followed by another
    # bang or question (preserves "no!!!", "What?!", "?!?", "!?!").
    current = re.sub(r"([!?])(?![!?])(\S)", r"\1 \2", current)
    # Period: insert a space when followed by a non-space char, EXCEPT
    #   - when the right side looks like a 1-4 character file extension
    #     / short suffix terminated by a word boundary (auth.ts,
    #     file.json, 1.5, 1.5.2, github.com)
    #   - when the next char is another period (preserves "..." runs)
    # Longer tokens after a period are treated as a sentence boundary
    # ("hello.world" -> "hello. world").
    current = re.sub(r"\.(?![A-Za-z0-9]{1,4}(?:\W|$))(?!\.)(\S)", r". \1", current)
    return current.strip()


_MONTH_CANONICAL = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
}
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTH_CANONICAL) + r")\b", re.IGNORECASE)
_STANDALONE_LOWER_I_RE = re.compile(r"(?<![A-Za-z])i(?![A-Za-z])")
_STANDALONE_LETTER_RE = re.compile(r"(?<![A-Za-z'.])([bcdefghjklmnopqrstuvwxyz])(?![A-Za-z'.])")


def normalize_dictation_orthography(text: str) -> str:
    """Apply non-semantic orthography expected on normal dictation surfaces.

    This layer only fixes presentation that is independent of app context:
    sentence-leading capitalization, standalone "I", month-name casing, and
    spoken single-letter mentions. It deliberately stays out of code/terminal
    paths via the caller.
    """
    current = normalize_plain_dictation(text)
    if not current:
        return current
    current = _STANDALONE_LOWER_I_RE.sub("I", current)
    current = _MONTH_RE.sub(lambda m: _MONTH_CANONICAL[m.group(1).casefold()], current)
    current = _STANDALONE_LETTER_RE.sub(lambda m: m.group(1).upper(), current)
    current = _capitalize_sentence_starts(current)
    return current.strip()


def _capitalize_sentence_starts(text: str) -> str:
    out: list[str] = []
    capitalize_next = True
    chars = list(text)
    for idx, ch in enumerate(chars):
        if capitalize_next and ch.isalpha():
            out.append(ch.upper())
            capitalize_next = False
            continue
        out.append(ch)
        if ch == "." and idx + 1 < len(chars) and not chars[idx + 1].isspace():
            continue
        if ch in ".!?\n":
            capitalize_next = True
        elif not ch.isspace() and ch not in "\"'([{":
            capitalize_next = False
    return "".join(out)


# ---------------------------------------------------------------------- #
# Layer 3: newline policy
# ---------------------------------------------------------------------- #

# A cue preceded by a definite determiner/possessive is a mention of a
# paragraph or line ("the new paragraph is short"), not a spoken command.
# Indefinite articles are NOT guarded: "a new paragraph" is how the cue
# itself is often spoken ("insert a new paragraph"). Python re needs
# fixed-width lookbehinds, hence one per word.
_NOT_AFTER_DETERMINER = (
    r"(?<!\bthe )(?<!\bthis )(?<!\bthat )(?<!\beach )"
    r"(?<!\bevery )(?<!\bmy )(?<!\byour )(?<!\bour )(?<!\btheir )(?<!\bits )"
    r"(?<!\bhis )(?<!\bher )"
)

_NEWLINE_TOKEN_PATTERNS = [
    (re.compile(r"\b(?:okay|ok)[,\s]+go\s+to\s+(?:new\s+line|newline)\b\s*[.,!?]?", re.IGNORECASE), "\n"),
    (re.compile(r"\bgo\s+to\s+(?:new\s+line|newline)\b\s*[.,!?]?", re.IGNORECASE), "\n"),
    (re.compile(_NOT_AFTER_DETERMINER + r"\bnew\s+paragraph\b\s*[.,!?]?", re.IGNORECASE), "\n\n"),
    (re.compile(_NOT_AFTER_DETERMINER + r"\b(?:new\s+line|newline)\b\s*[.,!?]?", re.IGNORECASE), "\n"),
    (re.compile(_NOT_AFTER_DETERMINER + r"\bline\s+break\b\s*[.,!?]?", re.IGNORECASE), "\n"),
]


def apply_newline_policy(text: str) -> str:
    """Replace spoken newline cues with real newlines.

    We only fire on whole-word matches so a legitimate mention of
    ``"the new paragraph is short"`` is left alone because it isn't the
    standalone cue (context note: in practice users speak the cue in
    isolation between clauses, so this rarely false-triggers).

    Multiple runs of newlines are collapsed to at most two.
    """
    out = text or ""
    for pattern, sub in _NEWLINE_TOKEN_PATTERNS:
        out = pattern.sub(sub, out)
    # Drop whitespace hugging the inserted newlines.
    out = re.sub(r"[ \t]*\n[ \t]*", "\n", out)
    # Collapse runs of 3+ newlines to a paragraph break.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


# ---------------------------------------------------------------------- #
# Layer 4: snippet expansion
# ---------------------------------------------------------------------- #


class SnippetResolver(Protocol):
    def resolve(self, trigger: str, *, scope: str = "global") -> object | None: ...


_SNIPPET_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-_]*", re.UNICODE)


def expand_snippets(
    text: str,
    *,
    resolver: SnippetResolver | None,
    scope: str = "global",
    max_expansions: int = 32,
) -> str:
    """Expand snippet triggers into their body text.

    The matcher is fold-aware: a stored trigger of ``"signoff"`` fires on
    spoken ``"sign off"``, ``"Sign-Off"``, ``"signoff"``, and a stored
    trigger of ``"sign off"`` fires on ``"signoff"`` just the same. This
    is essential because ASR systematically segments compounds and
    punctuation differently than the user typed the trigger.

    Algorithm:
      1. Collect snippets via ``resolver.list()`` (preferred) or fall
         back to per-token ``resolve`` calls when ``list`` isn't
         available (custom Protocol adapters in tests).
      2. Index by ``fold_key(trigger)``; for clashes prefer the
         scope-specific entry over the global one.
      3. Tokenize the text into alphanumeric runs (anchoring word
         boundaries on the alphanumeric side, so ``"brb"`` won't fire
         inside ``"brbrain"``).
      4. Walk the token stream; for each starting position try windows
         of size N..1 (N = longest stored trigger in tokens). Each
         window's span is fold-keyed and looked up in the index.
      5. Longest match wins per position; replace the span with the
         body and advance past it. Plain text between matches is kept
         verbatim (no extra whitespace normalisation).
      6. Bail after ``max_expansions`` so a body that itself looks like
         a trigger can't blow the stack.
    """
    if not text or resolver is None:
        return text or ""

    list_fn = getattr(resolver, "list", None)
    snippets: list[object] = []
    if callable(list_fn):
        try:
            snippets = list(list_fn())
        except Exception:
            snippets = []

    if not snippets:
        return _expand_snippets_via_resolve_only(
            text, resolver=resolver, scope=scope, max_expansions=max_expansions
        )

    # Per (fold_key, case_sensitive) → snippet, scope-preferred.
    by_key: dict[tuple[str, bool], object] = {}
    for snip in snippets:
        trigger = (getattr(snip, "trigger", "") or "").strip()
        body = getattr(snip, "body", None)
        if not trigger or not body:
            continue
        case_sensitive = bool(getattr(snip, "case_sensitive", False))
        key = trigger if case_sensitive else fold_key(trigger)
        if not key:
            continue
        composite = (key, case_sensitive)
        existing = by_key.get(composite)
        if existing is None:
            by_key[composite] = snip
            continue
        existing_scope = getattr(existing, "scope", "global") or "global"
        new_scope = getattr(snip, "scope", "global") or "global"
        if existing_scope != scope and new_scope == scope:
            by_key[composite] = snip

    if not by_key:
        return text

    tokens = list(_SNIPPET_TOKEN_RE.finditer(text))
    if not tokens:
        return text

    # Window cap. A stored "signoff" (1 token) may need to span "sign off"
    # (2 tokens) in text — and vice versa — so the window cap can't be
    # the trigger's own token count. We probe up to the global maximum
    # of: the longest stored trigger's token count, doubled (to handle
    # ASR over-segmentation), with a hard floor of 4. Triggers longer
    # than 8 tokens are vanishingly rare; we cap there.
    max_trigger_tokens = 1
    for snip in by_key.values():
        n = len(_SNIPPET_TOKEN_RE.findall((getattr(snip, "trigger", "") or "")))
        if n > max_trigger_tokens:
            max_trigger_tokens = n
    max_window = min(max(max_trigger_tokens * 2, 4), 8)

    expansions = 0
    out: list[str] = []
    cursor = 0
    i = 0
    while i < len(tokens):
        matched = False
        # Greedy longest: try the biggest possible window first.
        max_try = min(max_window, len(tokens) - i)
        for window in range(max_try, 0, -1):
            if expansions >= max_expansions:
                break
            start_tok = tokens[i]
            end_tok = tokens[i + window - 1]
            span = text[start_tok.start(): end_tok.end()]
            span_fold = fold_key(span)
            # Probe both case-insensitive and case-sensitive indices.
            snippet = by_key.get((span_fold, False)) if span_fold else None
            if snippet is None:
                snippet = by_key.get((span, True))
            if snippet is None:
                continue
            case_sensitive = bool(getattr(snippet, "case_sensitive", False))
            stored_trigger = getattr(snippet, "trigger", "") or ""
            if case_sensitive and span != stored_trigger:
                continue
            body = getattr(snippet, "body", "") or ""
            if not body:
                continue
            out.append(text[cursor: start_tok.start()])
            out.append(body)
            cursor = end_tok.end()
            i += window
            expansions += 1
            matched = True
            break
        if not matched:
            i += 1
    out.append(text[cursor:])
    return "".join(out)


def _expand_snippets_via_resolve_only(
    text: str,
    *,
    resolver: SnippetResolver,
    scope: str,
    max_expansions: int,
) -> str:
    """Fallback path for resolvers that don't expose ``list()``.

    Uses the same token-window walk as the primary path, but asks
    ``resolver.resolve`` for each window's span (sized up to 3 tokens —
    we can't know the snippet store's longest trigger without
    ``list``). Standard SnippetStore implements ``list`` so this code
    runs only for test/mock resolvers.
    """
    tokens = list(_SNIPPET_TOKEN_RE.finditer(text))
    if not tokens:
        return text

    expansions = 0
    out: list[str] = []
    cursor = 0
    i = 0
    while i < len(tokens):
        matched = False
        max_try = min(3, len(tokens) - i)
        for window in range(max_try, 0, -1):
            if expansions >= max_expansions:
                break
            start_tok = tokens[i]
            end_tok = tokens[i + window - 1]
            span = text[start_tok.start(): end_tok.end()]
            snippet = resolver.resolve(span, scope=scope)
            if snippet is None:
                continue
            case_sensitive = bool(getattr(snippet, "case_sensitive", False))
            stored_trigger = getattr(snippet, "trigger", "") or ""
            if case_sensitive and span != stored_trigger:
                continue
            body = getattr(snippet, "body", "") or ""
            if not body:
                continue
            out.append(text[cursor: start_tok.start()])
            out.append(body)
            cursor = end_tok.end()
            i += window
            expansions += 1
            matched = True
            break
        if not matched:
            i += 1
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------- #
# Layer 5: app-category formatting
# ---------------------------------------------------------------------- #


class AppCategory(str, Enum):
    """Coarse presentation category of the active field.

    The broker's context plane decides which category applies; this enum is
    the writer-facing contract. ``UNKNOWN`` is always safe and is the
    fallback when the surface can't classify.
    """

    MESSAGING = "messaging"  # Slack, iMessage, Teams chat — punchy, no heavy formatting
    EMAIL = "email"          # Mail, Gmail — paragraphs allowed, prose-friendly
    DOCS = "docs"            # Notes, Docs, Pages — honours paragraph breaks
    CODE = "code"            # IDE / code chat — leave whitespace alone
    TERMINAL = "terminal"    # CLI — strip everything that breaks shell parsing
    FORMS = "forms"          # Web/native forms — single-line preferred
    MEETING = "meeting"      # meeting note surfaces — speaker tags + heading grouping
    UNKNOWN = "unknown"


_RAW_CATEGORIES = {AppCategory.CODE, AppCategory.TERMINAL}


def apply_app_formatting(text: str, *, category: AppCategory | str | None) -> str:
    """Apply presentation rules for the destination app category.

    The rules are intentionally minimal — the goal is to never make the
    text *worse* than the raw dictation, not to prescribe the user's
    writing style. Each category's policy:

    - ``CODE`` / ``TERMINAL``: no changes. We treat these as raw surfaces;
      any punctuation/whitespace massage would fight with what the user
      typed.
    - ``MESSAGING``: collapse double paragraph breaks to single newlines —
      chat apps render ``\\n\\n`` as an awkward empty line.
    - ``FORMS``: flatten newlines to single spaces, best-effort. A form
      field is usually single-line and pasted newlines either get stripped
      or look broken.
    - ``EMAIL`` / ``DOCS`` / ``UNKNOWN``: preserve paragraph structure;
      trim trailing whitespace per line.
    """
    if not text:
        return text or ""
    cat = _coerce_category(category)
    if cat in _RAW_CATEGORIES:
        return text

    if cat is AppCategory.MESSAGING:
        # Single newlines only; collapse paragraph breaks but keep line
        # breaks so bullet-style dictation survives.
        cleaned = re.sub(r"\n{2,}", "\n", text)
        return _strip_trailing_whitespace_per_line(cleaned)

    if cat is AppCategory.FORMS:
        return re.sub(r"\s*\n+\s*", " ", text).strip()

    # EMAIL / DOCS / UNKNOWN
    return _strip_trailing_whitespace_per_line(text)


def _coerce_category(category: AppCategory | str | None) -> AppCategory:
    if isinstance(category, AppCategory):
        return category
    if not category:
        return AppCategory.UNKNOWN
    try:
        return AppCategory(str(category).lower())
    except ValueError:
        return AppCategory.UNKNOWN


def _strip_trailing_whitespace_per_line(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + (
        "\n" if text.endswith("\n") else ""
    )


# ---------------------------------------------------------------------- #
# Layer 6: speech-correction patterns (backtrack and spelling chants)
# ---------------------------------------------------------------------- #
#
# When a speaker dictates "X, actually Y" they almost always mean "Y" —
# the X side is the false start they want replaced. Likewise when they
# spell out a word for a listener ("that's accommodate with two c's and
# two m's") the chant is meta-instruction, not part of the message.
# These patterns are very narrow on purpose: they only fire on shapes
# that are unambiguously corrections, never on legitimate uses of
# "actually" as an emphatic adverb.

# Meta-restart verb phrases. Speakers use these to abandon the prior
# thought and restart cleanly; the meta-sentence carries no content of
# its own and should be dropped entirely.
_META_RESTART_VERBS = (
    r"(?:"
    r"let\s+me\s+(?:rephrase|rewrite|restructure|make|put|say|try|reword|state|frame)\b"
    r"|"
    r"i\s+(?:mean|meant)\b"
    r")"
)

# Sentence-leading meta-restart: matches "Actually let me <verb> ..." and
# "Actually I mean/meant ..." from the start of the text or right after
# sentence punctuation. Captured group 1 is the leading boundary so the
# replacement preserves it.
_META_RESTART_PATTERN = re.compile(
    r"(?i)(^|[.!?]\s+)"
    r"Actually,?\s+"
    + _META_RESTART_VERBS +
    r"[^.!?]*[.!?]\s*"
)

# Trailing-clause meta-restart: matches "X actually. Let me <verb> ..."
# This shape arises when the speaker pauses ("X — actually, let me ...")
# and Whisper renders the dash + "actually" as a period before
# capitalising the next word. The combined effect is the same as the
# sentence-leading form: drop both the trailing "actually." and the
# meta-sentence that follows. The negative lookbehind prevents firing
# when "actually." is itself the leading word of a sentence (the
# leading form covers that already).
_TRAILING_ACTUALLY_META_PATTERN = re.compile(
    r"(?i)(?<![.!?])\s+actually\.\s+"
    + _META_RESTART_VERBS +
    r"[^.!?]*[.!?]\s*"
)

# Value correction: "<n1>[, ] [no ] actually [, ] <n2> [unit]" -> keep n2 (and unit).
# Both sides accept digits or spelled numbers up to thousand. The connector
# allows comma, period, or whitespace, and an optional "no" word ("X, no
# actually Y" — common in speech). Lookbehind/lookahead guards prevent
# misfiring inside decimals like "1.5 actually 2.0".
_NUMBER_WORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
)
_BACKTRACK_CONNECTOR = r"[,.\s]+(?:no[,.\s]+)?actually[,.\s]+"
_VALUE_CORRECTION_PATTERN = re.compile(
    r"(?<!\d\.)(?<!\d)\b(?:\d+|" + _NUMBER_WORD + r")\b"
    + _BACKTRACK_CONNECTOR +
    r"(\d+|" + _NUMBER_WORD + r")(?!\.\d)(?=\s|[,.;!?]|[ap]|$)"
    r"([,.\s]*(?:a\.?\s*m\.?|p\.?\s*m\.?|am|pm)|"
    r"\s+(?:o'?clock|hours?|minutes?|seconds?|days?|weeks?|months?|years?|dollars?|cents?))?",
    flags=re.IGNORECASE,
)

# Day/time-word correction: "today, no actually tomorrow morning" -> keep
# the second day phrase. A "day phrase" is one of {today, tomorrow,
# yesterday, tonight} optionally followed by a time-of-day word, OR the
# patterns "this morning / this evening / next week / tomorrow afternoon"
# etc. Longer alternatives come first so the regex engine prefers them
# over the bare-day form.
_DAY_TIME_PHRASE = (
    r"(?:"
    r"(?:today|tomorrow|yesterday|this|next|tonight)\s+"
    r"(?:morning|afternoon|evening|night|week|month|year)"
    r"|today|tomorrow|yesterday|tonight"
    r")"
)
_DAY_BACKTRACK_PATTERN = re.compile(
    r"(?i)\b" + _DAY_TIME_PHRASE
    + _BACKTRACK_CONNECTOR
    + r"(" + _DAY_TIME_PHRASE + r")\b"
)

# Name-redundancy: after lexicon canonicalization, the speaker's
# "<X>, actually his name is <Y>" with X == Y becomes a no-op. Drop the
# trailing assertion. Pronouns include neutral 'their' for completeness.
_NAME_REDUNDANCY_PATTERN = re.compile(
    r"(?i)(\b[A-Z][\w\-']*)"
    r"\s*,?\s+actually\s+(?:his|her|its|their)\s+name\s+is\s+"
    r"(\b[A-Z][\w\-']*)"
)

# Spelling-correction chant: "that's <word> with <n> <letter>'s and
# <n> <letter>'s" (and an optional 'and' connector to the next clause,
# which we eat so the trailing sentence doesn't start with a stray 'and').
_SPELLING_CHANT_PATTERN = re.compile(
    r"(?i)(?:^|[\s,;])"
    r"that'?s\s+\w+\s+with\s+"
    r"(?:\d+|" + _NUMBER_WORD + r")\s+[a-z]'?s"
    r"\s+and\s+"
    r"(?:\d+|" + _NUMBER_WORD + r")\s+[a-z]'?s"
    r"\b\s*"
    r"(?:and\s+|[.,;:]\s*)?"
)


def resolve_backtrack(text: str) -> str:
    """Resolve simple speech-correction patterns of the form 'X, actually Y'.

    Three narrow patterns, each with a clear semantic:

    1. Meta-restart: a sentence-leading "Actually let me ..." / "Actually
       I mean ..." is the speaker abandoning the prior thought; drop the
       whole meta-sentence.
    2. Value correction: a number/time replacement ("from 2 actually 3 PM"
       -> "from 3 PM").
    3. Name redundancy: after lexicon canonicalization the speaker's
       name-correction clause becomes a no-op ("Morgan actually his name
       is Morgan" -> "Morgan"); drop the trailing assertion.

    The function is a no-op when 'actually' is absent (cheap fast path).
    """
    if not text or "actually" not in text.lower():
        return text or ""

    out = text
    out = _META_RESTART_PATTERN.sub(lambda m: m.group(1), out)
    # Trailing "actually." + meta-restart sentence collapses to a single
    # period+space, closing the prior clause cleanly.
    out = _TRAILING_ACTUALLY_META_PATTERN.sub(". ", out)

    def _value_repl(match: re.Match[str]) -> str:
        n2, unit = match.group(1), match.group(2) or ""
        if unit:
            unit = re.sub(r"^[,.\s]+", " ", unit)
        return f"{n2}{unit}"
    out = _VALUE_CORRECTION_PATTERN.sub(_value_repl, out)
    out = _DAY_BACKTRACK_PATTERN.sub(lambda m: m.group(1), out)

    def _name_repl(match: re.Match[str]) -> str:
        n1, n2 = match.group(1), match.group(2)
        return n1 if n1.casefold() == n2.casefold() else match.group(0)
    out = _NAME_REDUNDANCY_PATTERN.sub(_name_repl, out)

    return out


def strip_correction_chants(text: str) -> str:
    """Drop spelling-correction chants the speaker dictates as asides.

    Pattern: "that's <word> with <n> <letter>'s and <n> <letter>'s [and|.]"
    The chant is meta-instruction the speaker addresses to a human
    listener; it's never part of the intended written message.
    """
    if not text or "that" not in text.lower():
        return text or ""
    return _SPELLING_CHANT_PATTERN.sub(" ", text)


# Filler-strip patterns. Each pattern targets a single, narrow shape that
# is unambiguously filler in the context the regex requires. The general
# rule: only strip a hedge ("like", "kinda", "sort of") when the
# surrounding context is itself a filler-shape (a verb of thinking
# followed by a determiner/pronoun starting a complete clause). This
# leaves legitimate uses untouched ("I think like a fox" — "a" is fine
# but the rule below requires the determiner to be the/we/it/this/etc,
# so "a fox" doesn't match).
_FILLER_PATTERNS = [
    # "I think|mean|guess [like|kinda|kind of|sort of] [article/pronoun]"
    # The mid-clause hedge before a complete clause is filler.
    (
        re.compile(
            r"(?i)\b(I\s+(?:think|mean|guess|feel))"
            r"\s+(?:like|kinda|kind\s+of|sort\s+of)\s+"
            r"(the|we|it|this|that|they|he|she|you|i|my|our|their|his|her)\b"
        ),
        r"\1 \2",
    ),
    # ", you know," parenthetical filler between commas — strip the
    # parenthetical entirely.
    (
        re.compile(r"(?i),\s*you\s+know,\s*"),
        ", ",
    ),
]


def strip_fillers(text: str) -> str:
    """Strip narrow, unambiguous filler shapes from dictated speech.

    Conservative: only fires on patterns that are filler in their
    context. Standalone "like" mid-sentence is left alone because
    distinguishing filler-"like" from comparison-"like" without
    deeper parsing is unsafe. See ``_FILLER_PATTERNS`` for the exact
    shapes covered.
    """
    if not text:
        return text or ""
    out = text
    for pattern, repl in _FILLER_PATTERNS:
        out = pattern.sub(repl, out)
    return out


# ---------------------------------------------------------------------- #
# Layer 6b: explicit two-point bullet rendering
# ---------------------------------------------------------------------- #
#
# The broader ordinal renderer intentionally requires 3+ items because
# "first / second" can be ordinary prose. "two bullet points" is different:
# the user has explicitly requested list structure, so render the next two
# ordinal items as bullets without involving the writer model.

_TWO_BULLET_POINTS_RE = re.compile(
    r"(?P<prefix>\s*)"
    r"\b(?:add|create|write|make|capture|note|put)?\s*"
    r"(?:two|2)\s+bullet\s+points?\s*[.:]?\s+"
    r"(?:first|one|1[.)])[,;:\s]+(?P<i1>[^.!?\n]+[.!?]?)\s+"
    r"(?:second|two|2[.)])[,;:\s]+(?P<i2>[^.!?\n]+?)"
    r"(?P<end>[.!?]|\s+(?=now\s+i\s+continue\b)|$)",
    flags=re.IGNORECASE,
)


def _format_two_bullet_points(match: re.Match[str]) -> str:
    prefix = match.groupdict().get("prefix") or ""
    if match.start() > 0 and "\n" not in prefix:
        prefix = "\n"
    items = []
    for key in ("i1", "i2"):
        item = _clean_spoken_ordinal_item(match.group(key) or "")
        if item:
            if not item.endswith((".", "?", "!")):
                item = f"{item}."
            items.append(item)
    if len(items) != 2:
        return match.group(0)
    trailing_break = bool((match.groupdict().get("end") or "").isspace())
    suffix = "\n" if trailing_break else ""
    return f"{prefix}- {items[0]}\n- {items[1]}{suffix}"


def render_two_bullet_points(text: str) -> str:
    """Render an explicit spoken two-item bullet request.

    This is intentionally narrow: it only fires when the transcript contains
    the words "two bullet points" or "2 bullet points" followed by two ordinal
    items. Plain "first X, second Y" prose is left untouched.
    """
    if not text:
        return text or ""
    return _TWO_BULLET_POINTS_RE.sub(_format_two_bullet_points, text)


# ---------------------------------------------------------------------- #
# Layer 7: list rendering from spoken ordinals
# ---------------------------------------------------------------------- #
#
# When a speaker enumerates 3+ items with ordinal cues — "first do X,
# second do Y, third do Z" — the writer renders the sequence as a
# numbered list. Three trigger families are supported, each with three
# anchor words:
#   ordinal:     first / second / third
#   ly-form:     firstly / secondly / thirdly
#   number-word: one / two / three
#
# Letters (a/b/c) are intentionally excluded for v1: the article 'a' is
# too common in prose to safely distinguish from the list-trigger 'a'
# without deeper parsing. They can be added later with stricter
# bracketing requirements if dictation cases demand it.
#
# Patterns operate on one sentence at a time (we sentence-split the
# input first) so a list's lead-in cannot accidentally span across the
# preceding sentence.

_LIST_TRIGGER_FAMILIES = [
    ["first", "second", "third"],
    ["firstly", "secondly", "thirdly"],
    ["one", "two", "three"],
]

_FOUR_ITEM_LIST_TRIGGER_FAMILIES = [
    ["first", "second", "third", "fourth"],
    ["firstly", "secondly", "thirdly", "fourthly"],
    ["one", "two", "three", "four"],
]

_SPOKEN_ORDINAL_ITEM_LABEL_RE = re.compile(
    r"^(?:(?:the\s+)?(?:point|item|bullet|section|task|step)\s+(?:is|was|:|-)\s+)",
    flags=re.IGNORECASE,
)


def _clean_spoken_ordinal_item(raw: str) -> str:
    """Normalize item text captured after an ordinal cue.

    In explicit list dictation, phrases such as "first point is Passport"
    use "point is" as spoken structure, not item content. The list renderer is
    the only caller, so stripping that label here does not affect prose.
    """
    item = re.sub(r"\s+", " ", (raw or "").strip()).rstrip(".,;:")
    if not item:
        return ""
    stripped = _SPOKEN_ORDINAL_ITEM_LABEL_RE.sub("", item).strip()
    item = stripped or item
    return item[0].upper() + item[1:] if item else ""


def _build_list_pattern(family: list[str]) -> re.Pattern[str]:
    """Match a 3-item list introduced by ordinal triggers from one family.

    The first ordinal anchors the list; the lead-in is whatever appears
    in the same sentence before it. Items are bounded by commas /
    semicolons; the optional 'and' connector before the final ordinal
    is consumed cleanly.
    """
    w0, w1, w2 = family[0], family[1], family[2]
    return re.compile(
        r"(?P<lead>.*?)"
        rf"\b{w0}\b[,\s]+(?P<i1>[^,;.\n]+?)\s*[,;]\s*(?:and\s+)?"
        rf"\b{w1}\b[,\s]+(?P<i2>[^,;.\n]+?)\s*[,;]\s*(?:and\s+)?"
        rf"\b{w2}\b[,\s]+(?P<i3>[^.!?\n]+?)\s*(?:[.!?](?:\s|$)|$)",
        flags=re.IGNORECASE,
    )


_LIST_PATTERNS = [_build_list_pattern(f) for f in _LIST_TRIGGER_FAMILIES]


def _format_list(match: re.Match[str]) -> str:
    """Render a captured 3-item list as a numbered list with an optional lead-in."""
    lead = (match.group("lead") or "").rstrip()
    # Strip a trailing colon, semicolon, or comma — we'll add a clean
    # colon when we render the lead-in line.
    while lead and lead[-1] in ",;:":
        lead = lead[:-1].rstrip()
    items = []
    for key in ("i1", "i2", "i3"):
        item = _clean_spoken_ordinal_item(match.group(key) or "")
        items.append(item)
    list_lines = [f"{idx + 1}. {item}." for idx, item in enumerate(items)]
    list_text = "\n".join(list_lines)
    if lead:
        return f"{lead}:\n{list_text}"
    return list_text


def render_list_from_ordinals(text: str) -> str:
    """Convert a 3-item ordinal-introduced list into a numbered list.

    Triggers come from one of three families:
        first / second / third
        firstly / secondly / thirdly
        one / two / three

    Output preserves the lead-in as a preceding line ending in a colon,
    then renders each item on its own numbered line with a capitalized
    first letter and trailing period:

        My top goals this week are:
        1. Finish the report.
        2. Send the presentation.
        3. Book investor calls.

    The function operates per-sentence — input is split on
    ``[.!?]\\s+`` boundaries first — so the lead-in cannot accidentally
    capture content from the preceding sentence.
    """
    if not text:
        return text or ""
    parts = re.split(r"(?<=[.!?])(\s+)", text)
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            for pattern in _LIST_PATTERNS:
                replaced = pattern.sub(_format_list, part)
                if replaced != part:
                    part = replaced
                    break
        rendered.append(part)
    return "".join(rendered)


# ---------------------------------------------------------------------- #
# Layer 7b: cross-sentence list rendering
# ---------------------------------------------------------------------- #
#
# When the speaker enumerates list items with a clear pause between
# them, Whisper renders each item as its own sentence (period-bounded)
# rather than as a comma-bounded clause. This pattern detects three
# consecutive sentences each starting with an ordinal trigger from the
# same family and renders them as a numbered list.
#
# Item content is allowed to include "<period><alphanumeric>" sequences
# so file extensions (auth.ts), decimals (1.5), and version numbers
# (v3.2) inside an item don't break the sentence-boundary detection.
# Real sentence boundaries are "<period><whitespace>" patterns; the
# regex distinguishes them by requiring the period to be followed by a
# word character to count it as item-internal.

_CROSS_SENTENCE_ITEM = r"(?:[^.!?\n]|\.\w)+"


def _build_cross_sentence_list_pattern(family: list[str]) -> re.Pattern[str]:
    """Match three consecutive sentences each starting with an ordinal trigger.

    The lead-in (a preceding sentence in the same paragraph) is
    optional. If present, it ends with sentence punctuation and is
    captured for inclusion as a colon-terminated header in the
    rendered output.
    """
    w0, w1, w2 = family[0], family[1], family[2]
    return re.compile(
        r"(?:(?P<lead>[^\n]*?[.!?])\s+)?"
        rf"\b{w0}\b[,;:\s]+(?P<i1>" + _CROSS_SENTENCE_ITEM + r")[.!?]\s*[\"'”]?\s+"
        rf"\b{w1}\b[,;:\s]+(?P<i2>" + _CROSS_SENTENCE_ITEM + r")[.!?]\s*[\"'”]?\s+"
        rf"\b{w2}\b[,;:\s]+(?P<i3>" + _CROSS_SENTENCE_ITEM + r")[.!?]\s*[\"'”]?\s*",
        flags=re.IGNORECASE,
    )


_CROSS_SENTENCE_LIST_PATTERNS = [
    _build_cross_sentence_list_pattern(f) for f in _LIST_TRIGGER_FAMILIES
]


def _build_four_item_cross_sentence_list_pattern(family: list[str]) -> re.Pattern[str]:
    """Match four consecutive ordinal-leading sentences before the 3-item rule."""

    w0, w1, w2, w3 = family[0], family[1], family[2], family[3]
    return re.compile(
        r"(?:(?P<lead>[^\n]*?[.!?])\s+)?"
        rf"\b{w0}\b[,;:\s]+(?P<i1>" + _CROSS_SENTENCE_ITEM + r")[.!?]\s*[\"'”]?\s+"
        rf"\b{w1}\b[,;:\s]+(?P<i2>" + _CROSS_SENTENCE_ITEM + r")[.!?]\s*[\"'”]?\s+"
        rf"\b{w2}\b[,;:\s]+(?P<i3>" + _CROSS_SENTENCE_ITEM + r")[.!?]\s*[\"'”]?\s+"
        rf"\b{w3}\b[,;:\s]+(?P<i4>" + _CROSS_SENTENCE_ITEM + r")[.!?]\s*[\"'”]?\s*",
        flags=re.IGNORECASE,
    )


_FOUR_ITEM_CROSS_SENTENCE_LIST_PATTERNS = [
    _build_four_item_cross_sentence_list_pattern(f) for f in _FOUR_ITEM_LIST_TRIGGER_FAMILIES
]

_THREE_SECTION_CONTINUOUS_RE = re.compile(
    r"(?P<lead>[^.!?\n]*?\b(?:3|three)\s+(?:bullets?|bullet\s+points?|sections?|items?|points?|things?)\b[^.!?\n]*?)\s+"
    r"(?:first|firstly|1st|one)(?:\s+is)?[,;:\s]+(?P<i1>.+?)\s+"
    r"(?:second|secondly|2nd|two)(?:\s+is)?[,;:\s]+(?P<i2>.+?)\s+"
    r"(?:third|thirdly|3rd|three)(?:\s+is)?[,;:\s]+(?P<i3>.+?)"
    r"(?=(?:\s+(?:a+h+|u+m+|the\s+final|final|end\s+with|text\s+should|words?\s+should)\b)|$)",
    flags=re.IGNORECASE,
)

_FOUR_SECTION_CONTINUOUS_RE = re.compile(
    r"(?P<lead>[^.!?\n]*?\b(?:4|four)\s+(?:sections?|items?|points?|things?)\b[^.!?\n]*?)\s+"
    r"(?:first|firstly|1st|one)(?:\s+is)?[,;:\s]+(?P<i1>.+?)\s+"
    r"(?:second|secondly|2nd|two)(?:\s+is)?[,;:\s]+(?P<i2>.+?)\s+"
    r"(?:third|thirdly|3rd|three)(?:\s+is)?[,;:\s]+(?P<i3>.+?)\s+"
    r"(?:fourth|fourthly|4th|four)(?:\s+is)?[,;:\s]+(?P<i4>.+?)"
    r"(?=(?:\s+(?:a+h+|u+m+|the\s+final|final|end\s+with|text\s+should|words?\s+should)\b)|$)",
    flags=re.IGNORECASE,
)


def _format_cross_sentence_list(match: re.Match[str]) -> str:
    """Render a captured cross-sentence list with its optional lead-in."""
    lead_raw = match.group("lead") or ""
    lead = lead_raw.rstrip()
    # Convert a trailing period on the lead-in into a colon — the
    # lead-in line introduces the list it precedes, so colon reads
    # cleaner than a period followed by numbered lines.
    if lead.endswith("."):
        lead = lead[:-1] + ":"
    items: list[str] = []
    for key in ("i1", "i2", "i3", "i4"):
        if key not in match.groupdict():
            continue
        item = _clean_spoken_ordinal_item(match.group(key) or "")
        if item:
            if item.count('"') % 2 == 1:
                item += '"'
        items.append(item)
    list_lines = [f"{idx + 1}. {item}." for idx, item in enumerate(items)]
    body = "\n".join(list_lines) + "\n"
    return f"{lead}\n{body}" if lead else body


def _format_continuous_ordinal_section_list(match: re.Match[str]) -> str:
    lead = (match.group("lead") or "").strip().rstrip(".,;:")
    items: list[str] = []
    for key in ("i1", "i2", "i3", "i4"):
        if key not in match.groupdict():
            continue
        item = _clean_spoken_ordinal_item(match.group(key) or "")
        items.append(item)
    body = "\n".join(f"{idx}. {item}." for idx, item in enumerate(items, start=1))
    return f"{lead}:\n{body}\n" if lead else f"{body}\n"


def render_list_from_ordinal_sentences(text: str) -> str:
    """Convert three consecutive ordinal-leading sentences into a numbered list.

    This is the cross-sentence companion to :func:`render_list_from_ordinals`.
    Where the within-sentence rule fires on comma-bounded items
    (``"are first A, second B, third C."``), this rule fires on
    period-bounded items
    (``"are A. First, ship X. Second, ship Y. Third, ship Z."``) — the
    same speaker intent, just rendered by Whisper with longer pauses
    between items.

    Trigger families and item-formatting rules match the within-sentence
    function exactly. Item content may contain ``"<period><alnum>"``
    fragments (file extensions, decimals) without being treated as
    sentence boundaries.
    """
    if not text:
        return text or ""
    out = _FOUR_SECTION_CONTINUOUS_RE.sub(_format_continuous_ordinal_section_list, text)
    out = _THREE_SECTION_CONTINUOUS_RE.sub(_format_continuous_ordinal_section_list, out)
    for pattern in _FOUR_ITEM_CROSS_SENTENCE_LIST_PATTERNS:
        out = pattern.sub(_format_cross_sentence_list, out)
    for pattern in _CROSS_SENTENCE_LIST_PATTERNS:
        out = pattern.sub(_format_cross_sentence_list, out)
    return out


# ---------------------------------------------------------------------- #
# Layer 7c: agenda-style "3 things" list repair
# ---------------------------------------------------------------------- #
#
# Whisper often drops the word "second" in agenda dictation while keeping
# the item shape, e.g. "we will talk about 3 things first is what is DeFi
# where is DeFi and the third and last is ...". This rule is deliberately
# scoped to an explicit "3 things" lead-in so ordinary prose with "first"
# and "third" does not get rewritten.

_THREE_THINGS_EXPLICIT_SECOND_RE = re.compile(
    r"(?P<prefix>\s*)(?P<lead>[^.!?\n]*?\b(?:3|three)\s+things?)\s+"
    r"(?:the\s+)?first(?:ly)?(?:\s+is)?\s+(?P<i1>.+?)\s+"
    r"(?:the\s+)?second(?:ly)?(?:\s+is)?\s+(?P<i2>.+?)\s+"
    r"(?:and\s+)?(?:the\s+)?third(?:ly)?(?:\s+and\s+last|\s+last)?(?:\s+is)?\s+"
    r"(?P<i3>[^.!?\n]+[.!?]?)",
    flags=re.IGNORECASE,
)

_THREE_THINGS_DROPPED_SECOND_RE = re.compile(
    r"(?P<prefix>\s*)(?P<lead>[^.!?\n]*?\b(?:3|three)\s+things?)\s+"
    r"(?:the\s+)?first(?:ly)?(?:\s+is)?\s+(?P<i1>.+?)\s+"
    r"(?P<i2>(?:what|where|why|how|when|who)\b.+?)\s+"
    r"(?:and\s+)?(?:the\s+)?third(?:ly)?(?:\s+and\s+last|\s+last)?(?:\s+is)?\s+"
    r"(?P<i3>[^.!?\n]+[.!?]?)",
    flags=re.IGNORECASE,
)

_THREE_THINGS_SENTENCE_RE = re.compile(
    r"(?P<prefix>\s*)(?P<lead>[^.!?\n]*?\b(?:3|three)\s+things?)\s*[.!?]\s+"
    r"(?:the\s+)?first(?:ly)?(?:\s+is)?[,;:\s]+(?P<i1>[^.!?\n]+[.!?]?)\s+"
    r"(?:the\s+)?second(?:ly)?(?:\s+is)?[,;:\s]+(?P<i2>[^.!?\n]+[.!?]?)\s+"
    r"(?:and\s+)?(?:the\s+)?third(?:ly)?(?:\s+and\s+last|\s+last)?"
    r"(?:\s+is)?[,;:\s]+(?P<i3>[^.!?\n]+[.!?]?)",
    flags=re.IGNORECASE,
)

_THREE_THINGS_NUMERIC_MARKER_RE = re.compile(
    r"(?P<prefix>\s*)(?P<lead>[^.!?\n]*?\b(?:3|three)\s+things?)\s*[.!?]\s+"
    r"(?:the\s+)?first(?:ly)?(?:\s+is)?[,;:\s]+(?P<i1>[^.!?\n]+[.!?]?)\s+"
    r"(?:2|two)[.)]\s+(?P<i2>.+?)\s+"
    r"(?:3|three)[.)]\s+(?P<i3>[^.!?\n]+[.!?]?)",
    flags=re.IGNORECASE,
)

_THREE_THINGS_COLLAPSED_FIRST_RE = re.compile(
    r"(?P<prefix>\s*)(?P<lead>[^.!?\n]*?\b(?:3|three)\s+things?)\s*[.!?]\s+"
    r"(?:the\s+)?first(?:ly)?(?:\s+is)?[,;:\s]+(?P<body>[^.!?\n]+[.!?]?)",
    flags=re.IGNORECASE,
)

_THREE_THINGS_EXPLICIT_MARKER_IN_BODY_RE = re.compile(
    r"\bsecond(?:ly)?\b|\bthird(?:ly)?\b|(?:^|\s)[23][.)]",
    flags=re.IGNORECASE,
)


def _clean_three_things_lead(lead: str) -> str:
    out = re.sub(r"\s+", " ", (lead or "").strip()).rstrip(".,;:")
    out = re.sub(r"(?i)^(?:hey\s+)?juno[,;:\s]+", "", out).strip()

    def split_topic(match: re.Match[str]) -> str:
        topic = match.group("topic").rstrip(".,;: ")
        agenda = match.group("agenda").strip()
        if agenda:
            agenda = agenda[0].upper() + agenda[1:]
        return f"{topic}. {agenda}"

    return re.sub(
        r"(?i)^(?P<topic>.+?)\s+(?P<agenda>we\s+(?:will|are\s+going\s+to)\s+(?:talk|discuss|cover)\b.*)$",
        split_topic,
        out,
        count=1,
    )


def _clean_three_things_item(item: str) -> str:
    out = re.sub(r"\s+", " ", (item or "").strip()).rstrip(",;:")
    if out:
        out = out[0].upper() + out[1:]
    return out


def _finish_three_things_item(item: str) -> str:
    if item.endswith((".", "?", "!")):
        return item
    return f"{item}."


def _format_three_things_list(match: re.Match[str]) -> str:
    prefix = match.groupdict().get("prefix") or ""
    lead = _clean_three_things_lead(match.group("lead") or "")
    items = [_clean_three_things_item(match.group(key) or "") for key in ("i1", "i2", "i3")]
    lines = [f"{idx + 1}. {_finish_three_things_item(item)}" for idx, item in enumerate(items) if item]
    if not lines:
        return match.group(0)
    body = "\n".join(lines)
    rendered = f"{lead}:\n{body}" if lead else body
    return f"{prefix}{rendered}"


def _split_collapsed_three_things_body(body: str) -> list[str] | None:
    compact = re.sub(r"\s+", " ", (body or "").strip()).rstrip(".!?")
    if _THREE_THINGS_EXPLICIT_MARKER_IN_BODY_RE.search(compact):
        return None
    compact = re.sub(r",\s+and\s+", ", ", compact, flags=re.IGNORECASE)
    parts = [part.strip(" ,;:") for part in compact.split(",") if part.strip(" ,;:")]
    if len(parts) < 3:
        return None
    item1 = parts[0]
    item2 = parts[1]
    tail = parts[2:]
    if len(tail) == 1:
        item3 = tail[0]
    elif len(tail) == 2:
        item3 = f"{tail[0]} and {tail[1]}"
    else:
        item3 = f"{', '.join(tail[:-1])}, and {tail[-1]}"
    return [item1, item2, item3]


def _format_collapsed_three_things_first(match: re.Match[str]) -> str:
    items = _split_collapsed_three_things_body(match.group("body") or "")
    if not items:
        return match.group(0)
    prefix = match.groupdict().get("prefix") or ""
    lead = _clean_three_things_lead(match.group("lead") or "")
    lines = [
        f"{idx + 1}. {_finish_three_things_item(_clean_three_things_item(item))}"
        for idx, item in enumerate(items)
    ]
    body = "\n".join(lines)
    rendered = f"{lead}:\n{body}" if lead else body
    return f"{prefix}{rendered}"


def render_three_things_agenda(text: str) -> str:
    """Render agenda-style "3 things" dictation as a numbered list."""
    if not text:
        return text or ""
    out = _THREE_THINGS_SENTENCE_RE.sub(_format_three_things_list, text)
    out = _THREE_THINGS_NUMERIC_MARKER_RE.sub(_format_three_things_list, out)
    out = _THREE_THINGS_COLLAPSED_FIRST_RE.sub(_format_collapsed_three_things_first, out)
    out = _THREE_THINGS_EXPLICIT_SECOND_RE.sub(_format_three_things_list, out)
    out = _THREE_THINGS_DROPPED_SECOND_RE.sub(_format_three_things_list, out)
    return out


# ---------------------------------------------------------------------- #
# Layer 7d: explicit numbered-marker repair
# ---------------------------------------------------------------------- #
#
# ASR/model output sometimes already includes numeric markers, but keeps
# them inline with the lead sentence: "Update: 1. Ship X. 2. Test Y.".
# Preserve the user's numbered wording and only repair marker boundaries
# when a clear 1/2/3 sequence is present.

_NUMBERED_LIST_SEQUENCE_RE = re.compile(
    r"(?s)(?<!\d)1[.)]\s+\S.+?(?<!\d)2[.)]\s+\S.+?(?<!\d)3[.)]\s+\S"
)
_NUMBERED_LIST_MARKER_SPACE_RE = re.compile(r"[ \t]+(?=(?:[1-9]|1\d|20)[.)]\s+\S)")


def normalize_explicit_numbered_markers(text: str) -> str:
    """Place explicit numeric list markers on their own lines."""
    if not text or not _NUMBERED_LIST_SEQUENCE_RE.search(text):
        return text or ""

    def _break_before_marker(match: re.Match[str]) -> str:
        if match.start() == 0 or text[match.start() - 1] == "\n":
            return match.group(0)
        return "\n"

    return _NUMBERED_LIST_MARKER_SPACE_RE.sub(_break_before_marker, text)


# ---------------------------------------------------------------------- #
# Pipeline
# ---------------------------------------------------------------------- #


def run_pipeline(
    text: str,
    *,
    snippet_resolver: SnippetResolver | None = None,
    app_category: AppCategory | str | None = None,
    enable_newline_policy: bool = True,
) -> str:
    """Convenience: normalise + newline policy + corrections + snippets + formatting.

    Order matters. We normalise per-line (``_normalize_preserving_newlines``)
    so the newline policy can still recognise "new paragraph" cues in the
    raw stream, then apply newline policy, then resolve speech corrections
    (backtrack + chant strip), then expand snippets, then apply the app
    category's typographic rules. Speech-correction resolution sits before
    snippet expansion so a snippet body can't accidentally introduce text
    that looks like a chant or backtrack pattern.
    """
    out = text or ""
    cat = _coerce_category(app_category)
    # Raw categories bypass every layer — we never massage code or shell
    # commands. This is an explicit design rule: the writer must not
    # "help" code because "help" here is usually "break".
    if cat in _RAW_CATEGORIES:
        return out
    out = _normalize_preserving_newlines(out)
    if enable_newline_policy:
        out = apply_newline_policy(out)
    # Strip fillers first so backtrack can operate on cleaner text (e.g.
    # "Um I think like X actually Y" -> "I think X actually Y" -> "Y").
    out = strip_fillers(out)
    out = resolve_backtrack(out)
    out = strip_correction_chants(out)
    # The strip steps may leave double spaces or stray punctuation gaps;
    # re-normalise per line so the output looks tidy.
    out = _normalize_preserving_newlines(out)
    # List rendering runs on the cleaned text so trigger words aren't
    # buried under filler/backtrack noise. Within-sentence rendering
    # runs first (it's the more specific shape), then cross-sentence
    # picks up any period-bounded ordinal sequences left over.
    out = render_two_bullet_points(out)
    out = render_three_things_agenda(out)
    out = render_list_from_ordinals(out)
    out = render_list_from_ordinal_sentences(out)
    out = normalize_explicit_numbered_markers(out)
    out = _normalize_preserving_newlines(out)
    if snippet_resolver is not None:
        scope = _category_to_scope(cat)
        out = expand_snippets(out, resolver=snippet_resolver, scope=scope)
    out = apply_app_formatting(out, category=cat)
    return out


def _normalize_preserving_newlines(text: str) -> str:
    """Same rules as ``normalize_plain_dictation`` but per-line.

    The legacy normalise collapses every run of whitespace (including
    newlines) to a single space, which breaks any pipeline that wants to
    use explicit paragraph breaks later. This variant applies the spacing
    rules inside each line so newlines survive.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned = [normalize_dictation_orthography(line) for line in lines]
    return "\n".join(cleaned)


def _category_to_scope(category: AppCategory) -> str:
    # Snippet scopes intentionally mirror app-category values so the store
    # doesn't need a second taxonomy.
    if category is AppCategory.UNKNOWN:
        return "global"
    return category.value


__all__ = [
    "AppCategory",
    "apply_app_formatting",
    "apply_newline_policy",
    "render_three_things_agenda",
    "render_two_bullet_points",
    "render_list_from_ordinal_sentences",
    "render_list_from_ordinals",
    "resolve_backtrack",
    "strip_correction_chants",
    "strip_fillers",
    "expand_snippets",
    "normalize_dictation_orthography",
    "normalize_plain_dictation",
    "normalize_explicit_numbered_markers",
    "render_bullets",
    "render_lowercase",
    "render_numbered",
    "render_title_case",
    "render_uppercase",
    "run_pipeline",
]
