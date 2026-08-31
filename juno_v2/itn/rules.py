"""Deterministic ITN rules — pure functions, no I/O, no ML.

Each rule is a function(text: str) -> tuple[str, list[str]] returning the
transformed text and a list of rule-ids that fired (for traceability).

Optional :class:`~juno_v2.itn.format_policy.ITNFormatPolicy` controls how
dates, clock times, and currency decimals are rendered. When omitted, policy
matches the historical Juno defaults (US-style prose dates, 12-hour clock,
period decimal separator).
"""
from __future__ import annotations

import re

from juno_v2.itn.format_policy import ITNFormatPolicy


# ---------------------------------------------------------------------- #
# Number word tables
# ---------------------------------------------------------------------- #

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}

_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_ORDINAL_TO_CARDINAL = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty-first": 21,
    "twenty-second": 22, "twenty-third": 23, "twenty-fourth": 24,
    "twenty-fifth": 25, "twenty-sixth": 26, "twenty-seventh": 27,
    "twenty-eighth": 28, "twenty-ninth": 29, "thirtieth": 30,
    "thirty-first": 31,
}

_ORDINAL_SUFFIX = {
    1: "st", 2: "nd", 3: "rd",
    **{n: "th" for n in range(4, 32) if n not in {1, 2, 3, 21, 22, 23, 31}},
    21: "st", 22: "nd", 23: "rd", 31: "st",
}

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_ABBREV = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

_MONTH_FULL = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def _decimal_sep(fmt: ITNFormatPolicy) -> str:
    return "," if fmt.currency_decimal == "comma" else "."


def _format_currency_amount(whole: int, cents: int | None, symbol: str, fmt: ITNFormatPolicy) -> str:
    """Format ``whole`` and optional sub-units (cents/pence) with ``symbol`` prefix."""
    sep = _decimal_sep(fmt)
    if cents is None:
        return f"{symbol}{whole}"
    frac = f"{cents:02d}"
    return f"{symbol}{whole}{sep}{frac}"


def _clock12_to_24(hour: int, is_pm: bool) -> int:
    if hour == 12:
        return 12 if is_pm else 0
    if is_pm:
        return hour + 12
    return hour


def _meridiem_is_pm(raw: str) -> bool:
    u = raw.upper().replace(".", "")
    return u.endswith("PM")


def _format_spoken_clock_time(hour: int, minute: int, meridiem: str | None, fmt: ITNFormatPolicy) -> str:
    if meridiem is None:
        return f"{hour:02d}:{minute:02d}"
    is_pm = _meridiem_is_pm(meridiem)
    if fmt.clock == "24h":
        h24 = _clock12_to_24(hour, is_pm)
        return f"{h24:02d}:{minute:02d}"
    mer = meridiem.upper().replace(".", "")
    if len(mer) > 2:
        mer = mer[:2]  # AM / PM
    return f"{hour}:{minute:02d} {mer}"


def _format_date_render(month_num: int, day: int, year: int | None, fmt: ITNFormatPolicy) -> str:
    if year is None:
        if fmt.date_style == "iso":
            return f"{month_num:02d}-{day:02d}"
        if fmt.date_style == "dmy_slash":
            return f"{day:02d}/{month_num:02d}"
        if fmt.date_style == "mdy_slash":
            return f"{month_num:02d}/{day:02d}"
        if fmt.date_style == "dmy_long":
            return f"{day} {_MONTH_FULL[month_num]}"
        month_abbrev = _MONTH_ABBREV[month_num]
        return f"{month_abbrev} {day}"
    if fmt.date_style == "iso":
        return f"{year:04d}-{month_num:02d}-{day:02d}"
    if fmt.date_style == "dmy_slash":
        return f"{day:02d}/{month_num:02d}/{year:04d}"
    if fmt.date_style == "mdy_slash":
        return f"{month_num:02d}/{day:02d}/{year:04d}"
    if fmt.date_style == "dmy_long":
        return f"{day} {_MONTH_FULL[month_num]} {year}"
    month_abbrev = _MONTH_ABBREV[month_num]
    return f"{month_abbrev} {day}, {year}"


# ---------------------------------------------------------------------- #
# Internal: parse a short number phrase into an integer
# ---------------------------------------------------------------------- #

def _parse_number_words(tokens: list[str]) -> int | None:
    """Parse a list of word tokens into an integer. Returns None if not parseable."""
    total = 0
    current = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i].lower().replace(",", "")
        if tok in _ONES:
            current += _ONES[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            if current == 0:
                current = 1
            current *= 100
        elif tok in ("thousand",):
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
        elif tok in ("million",):
            if current == 0:
                current = 1
            total += current * 1_000_000
            current = 0
        elif tok == "and":
            pass
        else:
            return None
        i += 1
    return total + current


# ---------------------------------------------------------------------- #
# Rule: inline number sequences ("twenty five" → "25", etc.)
# ---------------------------------------------------------------------- #

_NUMBER_WORD_PAT = re.compile(
    r"\b("
    + r"|".join(sorted(list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True))
    + r")(\s+("
    + r"|".join(sorted(list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True))
    + r"))*\b",
    re.IGNORECASE,
)


def _try_convert_number_phrase(match: re.Match) -> str:
    tokens = match.group(0).split()
    if not _is_compositional_inline_number(tokens):
        return match.group(0)
    val = _parse_number_words(tokens)
    if val is None:
        return match.group(0)
    return str(val)


def _is_compositional_inline_number(tokens: list[str]) -> bool:
    """Return whether adjacent basic number words form one English number.

    The inline matcher intentionally covers only zero-to-nineteen and tens;
    scaled forms are handled by the date/time/currency rules before this
    fallback.  Within that vocabulary, the only valid multi-token cardinal is
    a tens word followed by a non-zero unit (``twenty five``).  Treating an
    arbitrary run such as ``two three four`` as an additive expression silently
    changed enumerations into ``9``.
    """

    normalized = [token.casefold().replace(",", "") for token in tokens]
    if len(normalized) == 1:
        return normalized[0] in _ONES or normalized[0] in _TENS
    if len(normalized) != 2:
        return False
    first, second = normalized
    return first in _TENS and second in _ONES and 1 <= _ONES[second] <= 9


_NUMERIC_CONTEXT_BEFORE_RE = re.compile(
    r"(?:chapter|page|version|number|item|step|section|room|gate|level|line|"
    r"track|seat|grade|round|phase|part|option|figure|table|no\.?|#)\s*$",
    re.IGNORECASE,
)
_NUMERIC_CONTEXT_AFTER_RE = re.compile(
    r"^\s*(?:%|percent|am\b|pm\b|a\.m|p\.m|o'?clock|dollars?\b|bucks?\b|euros?\b|"
    r"pounds?\b|cents?\b|hundred\b|thousand\b|million\b|x\b|times\b|\d)",
    re.IGNORECASE,
)


def apply_numeric(text: str) -> tuple[str, list[str]]:
    """Convert spoken number words to digits — prose-aware.

    Multi-word numbers ("twenty five") and values ≥ 10 always convert.
    Standalone one–nine stay words in prose ("in this one also" must never
    become "in this 1 also") and convert only with numeric context around
    them ("chapter one", "one pm").
    """
    applied: list[str] = []
    out: list[str] = []
    last = 0
    for m in _NUMBER_WORD_PAT.finditer(text):
        phrase = m.group(0)
        tokens = phrase.split()
        val = _parse_number_words(tokens)
        repl = phrase
        if val is not None and _is_compositional_inline_number(tokens):
            convert = len(tokens) > 1 or val >= 10
            if not convert:
                convert = bool(_NUMERIC_CONTEXT_BEFORE_RE.search(text[: m.start()])) or bool(
                    _NUMERIC_CONTEXT_AFTER_RE.match(text[m.end():])
                )
            if convert:
                repl = str(val)
        out.append(text[last : m.start()])
        out.append(repl)
        last = m.end()
    out.append(text[last:])
    result = "".join(out)
    if result != text:
        applied.append("numeric_words_to_digits")
    return result, applied


# ---------------------------------------------------------------------- #
# Rule: ordinals in date contexts ("nineteenth" → "19th")
# ---------------------------------------------------------------------- #

def _ordinal_to_numeral(word: str) -> str | None:
    lo = word.lower()
    n = _ORDINAL_TO_CARDINAL.get(lo)
    if n is None:
        return None
    suffix = _ORDINAL_SUFFIX.get(n, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------------- #
# Rule: currency (spoken major units → symbolic amounts)
# ---------------------------------------------------------------------- #


def _spoken_number_group() -> str:
    """Regex fragment: one spoken integer (words), captured as group 1."""
    return (
        r"\b((?:"
        + r"|".join(sorted(list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True))
        + r")(?:\s+(?:"
        + r"|".join(sorted(list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True))
        + r"))*)"
    )


def _spoken_currency_amount_group() -> str:
    """Like :func:`_spoken_number_group` but allows ``hundred``, ``thousand``, ``million``, ``and``."""
    extra = ("hundred", "thousand", "million", "and")
    keys = sorted(set(_ONES) | set(_TENS) | set(extra), key=len, reverse=True)
    alts = "|".join(keys)
    return r"\b((?:" + alts + r")(?:\s+(?:" + alts + r"))*)\b"


def _optional_minor_unit_pattern(minor_label: str) -> str:
    lab = minor_label
    return (
        r"(?:\s+and\s+((?:"
        + r"|".join(sorted(list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True))
        + r")(?:\s+(?:"
        + r"|".join(sorted(list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True))
        + r"))*)\s+" + lab + r")?"
    )


_SPOKEN_CUR = _spoken_currency_amount_group()
_USD_PAT = re.compile(
    _SPOKEN_CUR + r"\s+dollar[s]?" + _optional_minor_unit_pattern(r"cent[s]?"),
    re.IGNORECASE,
)
_EUR_PAT = re.compile(
    _SPOKEN_CUR + r"\s+euro[s]?" + _optional_minor_unit_pattern(r"cent[s]?"),
    re.IGNORECASE,
)
_GBP_PAT = re.compile(
    _SPOKEN_CUR + r"\s+pound[s]?" + _optional_minor_unit_pattern(r"pence"),
    re.IGNORECASE,
)
_CHF_PAT = re.compile(
    _SPOKEN_CUR + r"\s+swiss\s+franc[s]?" + _optional_minor_unit_pattern(r"centimes"),
    re.IGNORECASE,
)
_JPY_PAT = re.compile(
    _SPOKEN_CUR + r"\s+yen\b",
    re.IGNORECASE,
)


def _parse_minor_amount(m: re.Match, minor_group_idx: int) -> int | None:
    raw = m.group(minor_group_idx)
    if not raw:
        return None
    return _parse_number_words(raw.split())


def _convert_usd(m: re.Match, fmt: ITNFormatPolicy) -> str:
    main = _parse_number_words(m.group(1).split())
    if main is None:
        return m.group(0)
    cents = _parse_minor_amount(m, 2)
    if cents is not None and (cents < 0 or cents > 99):
        return m.group(0)
    return _format_currency_amount(main, cents, "$", fmt)


def _convert_eur(m: re.Match, fmt: ITNFormatPolicy) -> str:
    main = _parse_number_words(m.group(1).split())
    if main is None:
        return m.group(0)
    cents = _parse_minor_amount(m, 2)
    if cents is not None and (cents < 0 or cents > 99):
        return m.group(0)
    return _format_currency_amount(main, cents, "€", fmt)


def _convert_gbp(m: re.Match, fmt: ITNFormatPolicy) -> str:
    main = _parse_number_words(m.group(1).split())
    if main is None:
        return m.group(0)
    pence = _parse_minor_amount(m, 2)
    if pence is not None and (pence < 0 or pence > 99):
        return m.group(0)
    return _format_currency_amount(main, pence, "£", fmt)


def _convert_chf(m: re.Match, fmt: ITNFormatPolicy) -> str:
    main = _parse_number_words(m.group(1).split())
    if main is None:
        return m.group(0)
    centimes = _parse_minor_amount(m, 2)
    if centimes is not None and (centimes < 0 or centimes > 99):
        return m.group(0)
    return _format_currency_amount(main, centimes, "CHF ", fmt)


def _convert_jpy(m: re.Match, fmt: ITNFormatPolicy) -> str:
    main = _parse_number_words(m.group(1).split())
    if main is None:
        return m.group(0)
    return _format_currency_amount(main, None, "¥", fmt)


def apply_currency(text: str, fmt: ITNFormatPolicy | None = None) -> tuple[str, list[str]]:
    pol = fmt or ITNFormatPolicy.default()
    applied: list[str] = []
    out = text
    before = out
    # Longest / most specific phrases first to avoid partial matches.
    out = _CHF_PAT.sub(lambda m: _convert_chf(m, pol), out)
    out = _USD_PAT.sub(lambda m: _convert_usd(m, pol), out)
    out = _EUR_PAT.sub(lambda m: _convert_eur(m, pol), out)
    out = _GBP_PAT.sub(lambda m: _convert_gbp(m, pol), out)
    out = _JPY_PAT.sub(lambda m: _convert_jpy(m, pol), out)
    if out != before:
        applied.append("currency")
    return out, applied


# ---------------------------------------------------------------------- #
# Rule: dates ("april nineteenth twenty twenty six" → "April 19, 2026")
# ---------------------------------------------------------------------- #

def _build_date_pattern() -> re.Pattern:
    month_alts = "|".join(sorted(_MONTH_NAMES.keys(), key=len, reverse=True))
    ordinal_alts = "|".join(sorted(_ORDINAL_TO_CARDINAL.keys(), key=len, reverse=True))
    ones_alts = "|".join(sorted(_ONES.keys(), key=len, reverse=True))
    tens_alts = "|".join(sorted(_TENS.keys(), key=len, reverse=True))
    num_alts = ones_alts + "|" + tens_alts
    # Each year half may itself be a tens+ones compound ("twenty six"), so
    # "twenty twenty six" splits as 20 / 26 → 2026, and "nineteen ninety
    # nine" as 19 / 99 → 1999.
    year_half = rf"(?:(?:{tens_alts})\s+(?:{ones_alts})|{num_alts})"
    return re.compile(
        rf"\b({month_alts})\s+({ordinal_alts}|\d{{1,2}})"
        rf"(?:\s+({year_half})\s+({year_half}))?\b",
        re.IGNORECASE,
    )


_DATE_PAT = _build_date_pattern()


def _convert_date(m: re.Match, fmt: ITNFormatPolicy) -> str:
    month_word = m.group(1).lower()
    month_num = _MONTH_NAMES.get(month_word)
    if month_num is None:
        return m.group(0)
    day_word = m.group(2)
    if day_word.isdigit():
        day = int(day_word)
    else:
        day = _ORDINAL_TO_CARDINAL.get(day_word.lower())
        if day is None:
            return m.group(0)
    year_part1 = m.group(3)
    year_part2 = m.group(4)
    year: int | None = None
    if year_part1 and year_part2:
        y1 = _parse_number_words(year_part1.split())
        y2 = _parse_number_words(year_part2.split())
        if y1 is not None and y2 is not None:
            year = y1 * 100 + y2
    return _format_date_render(month_num, day, year, fmt)


def apply_dates(text: str, fmt: ITNFormatPolicy | None = None) -> tuple[str, list[str]]:
    pol = fmt or ITNFormatPolicy.default()
    applied: list[str] = []
    out = _DATE_PAT.sub(lambda m: _convert_date(m, pol), text)
    if out != text:
        applied.append("dates")
    return out, applied


# ---------------------------------------------------------------------- #
# Rule: times ("three thirty pm" → "3:30 PM", "fourteen hundred" → "14:00")
# ---------------------------------------------------------------------- #

_ONES_PAT = "|".join(sorted(_ONES.keys(), key=len, reverse=True))
_TENS_PAT = "|".join(sorted(_TENS.keys(), key=len, reverse=True))
_NUM_PAT = _ONES_PAT + "|" + _TENS_PAT

# Dotted forms first, and `(?!\w)` instead of `\b`: after the trailing "."
# of "p.m." there is no word boundary, so `p\.m\.\b` can never match.
_MERIDIEM_PAT = r"a\.m\.|p\.m\.|am|pm"

_TIME_PAT = re.compile(
    rf"\b({_NUM_PAT})\s+({_NUM_PAT})\s+({_MERIDIEM_PAT})(?!\w)"
    rf"|\b({_NUM_PAT})\s+({_MERIDIEM_PAT})(?!\w)"
    rf"|\b(\d{{1,2}})\s+hundred\b",
    re.IGNORECASE,
)

# Word-form military time. A bare "fourteen hundred" is ambiguous (it is
# usually a quantity: "fourteen hundred people"), so word-form conversion
# requires explicit time context: an "at " prefix or an "hours" suffix.
_MILITARY_WORDS_PAT = re.compile(
    rf"\b(?:(at)\s+)?((?:{_NUM_PAT})(?:\s+(?:{_NUM_PAT}))?)\s+hundred(\s+hours)?\b",
    re.IGNORECASE,
)


def _convert_military_words(m: re.Match) -> str:
    at_prefix, hour_words, hours_suffix = m.group(1), m.group(2), m.group(3)
    if not at_prefix and not hours_suffix:
        return m.group(0)
    hour = _parse_number_words(hour_words.split())
    if hour is None or not 0 <= hour <= 23:
        return m.group(0)
    prefix = f"{at_prefix} " if at_prefix else ""
    return f"{prefix}{hour:02d}:00"


def _convert_time(m: re.Match, fmt: ITNFormatPolicy) -> str:
    if m.group(1) and m.group(2) and m.group(3):
        h = _parse_number_words(m.group(1).split())
        mins = _parse_number_words(m.group(2).split())
        meridiem = m.group(3)
        if h is not None and mins is not None:
            return _format_spoken_clock_time(h, mins, meridiem, fmt)
    elif m.group(4) and m.group(5):
        h = _parse_number_words(m.group(4).split())
        meridiem = m.group(5)
        if h is not None:
            return _format_spoken_clock_time(h, 0, meridiem, fmt)
    elif m.group(6):
        h = int(m.group(6))
        return f"{h:02d}:00"
    return m.group(0)


def apply_times(text: str, fmt: ITNFormatPolicy | None = None) -> tuple[str, list[str]]:
    pol = fmt or ITNFormatPolicy.default()
    applied: list[str] = []
    out = _TIME_PAT.sub(lambda m: _convert_time(m, pol), text)
    out = _MILITARY_WORDS_PAT.sub(_convert_military_words, out)
    if out != text:
        applied.append("times")
    return out, applied


# ---------------------------------------------------------------------- #
# Rule: email / URL ("support at example dot com" → "support@example.com")
# ---------------------------------------------------------------------- #

_EMAIL_PAT = re.compile(
    r"\b([\w.+-]+)\s+at\s+([\w.-]+)\s+dot\s+(\w+)\b",
    re.IGNORECASE,
)

_URL_DOT_PAT = re.compile(
    r"\b((?:www|http|https|ftp))\s+dot\s+([\w.-]+\s+dot\s+[\w]+|[\w.-]+)\b",
    re.IGNORECASE,
)

_COLON_SLASH_PAT = re.compile(
    r"\b(https?|ftp)\s+colon\s+(?:slash\s+slash|double\s+slash)\s+([\w./-]+)\b",
    re.IGNORECASE,
)


def _convert_email(m: re.Match) -> str:
    local = m.group(1)
    domain_parts = [m.group(2), m.group(3)]
    domain = ".".join(p.strip() for p in domain_parts)
    return f"{local}@{domain}"


def _convert_url_dot(m: re.Match) -> str:
    prefix = m.group(1).lower()
    rest = re.sub(r"\s+dot\s+", ".", m.group(2), flags=re.IGNORECASE)
    rest = re.sub(r"\s+", "", rest)
    if prefix in ("http", "https", "ftp"):
        return f"{prefix}://{rest}"
    return f"{prefix}.{rest}"


def _convert_colon_slash(m: re.Match) -> str:
    scheme = m.group(1).lower()
    rest = re.sub(r"\s+dot\s+", ".", m.group(2), flags=re.IGNORECASE)
    rest = re.sub(r"\s+", "", rest)
    return f"{scheme}://{rest}"


def apply_email_url(text: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    out = _COLON_SLASH_PAT.sub(_convert_colon_slash, text)
    out = _EMAIL_PAT.sub(_convert_email, out)
    out = _URL_DOT_PAT.sub(_convert_url_dot, out)
    if out != text:
        applied.append("email_url")
    return out, applied


# ---------------------------------------------------------------------- #
# Rule: code / file identifiers
# ("main dot ts" → "main.ts", "index underscore test" → "index_test")
# ---------------------------------------------------------------------- #

_CODE_DOT_PAT = re.compile(
    r"\b([\w-]+)\s+dot\s+([\w-]+)\b",
    re.IGNORECASE,
)

_CODE_UNDERSCORE_PAT = re.compile(
    r"\b([\w-]+)\s+underscore\s+([\w-]+)\b",
    re.IGNORECASE,
)

_CODE_DASH_PAT = re.compile(
    r"\b([\w_]+)\s+dash\s+([\w_]+)\b",
    re.IGNORECASE,
)

_CODE_SLASH_PAT = re.compile(
    r"\bdot\s+slash\s+([\w./-]+)\b",
    re.IGNORECASE,
)


def apply_code_identifiers(text: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    out = text
    new = _CODE_SLASH_PAT.sub(lambda m: f"./{m.group(1)}", out)
    new = _CODE_DOT_PAT.sub(lambda m: f"{m.group(1)}.{m.group(2)}", new)
    new = _CODE_UNDERSCORE_PAT.sub(lambda m: f"{m.group(1)}_{m.group(2)}", new)
    new = _CODE_DASH_PAT.sub(lambda m: f"{m.group(1)}-{m.group(2)}", new)
    if new != out:
        applied.append("code_identifiers")
    return new, applied


# ---------------------------------------------------------------------- #
# Rule: terminal operators ("pipe" → "|", "greater than" → ">", etc.)
# ---------------------------------------------------------------------- #

# Longest phrases first — a bare "pipe"/"ampersand" rule would otherwise
# consume the tail of "double pipe"/"double ampersand" and make the double
# forms unreachable.
_TERMINAL_OPS = [
    (re.compile(r"\bdouble\s+ampersand\b", re.IGNORECASE), "&&"),
    (re.compile(r"\bdouble\s+pipe\b", re.IGNORECASE), "||"),
    (re.compile(r"\bpipe\b", re.IGNORECASE), "|"),
    (re.compile(r"\bgreater\s+than\b", re.IGNORECASE), ">"),
    (re.compile(r"\bless\s+than\b", re.IGNORECASE), "<"),
    (re.compile(r"\bampersand\b", re.IGNORECASE), "&"),
]


def apply_terminal_ops(text: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    out = text
    for pat, sub in _TERMINAL_OPS:
        new = pat.sub(sub, out)
        if new != out:
            applied.append("terminal_ops")
            out = new
    return out, list(set(applied))


# ---------------------------------------------------------------------- #
# Rule: inline spoken punctuation
#   "hello comma world" → "hello, world"
#   "done period new line goodbye" → "done.\ngoodbye"
#   etc.
#
# Whole-utterance commands (e.g. a lone "new line") remain owned by
# `juno_v2/commands/grammar.py`; this rule only fires inline.
# ---------------------------------------------------------------------- #

# Order matters: longer phrases first, em dash before dash so "em dash"
# isn't eaten by a bare "dash" rule.
_SPOKEN_PUNCT_TABLE: list[tuple[str, str, str]] = [
    # phrase (regex body, expected to be \b-bounded), glyph, kind
    # kind controls spacing:
    #   "punct"    — preceded by no space, followed by single space
    #   "newline"  — replaced by literal newline(s), no surrounding spaces
    #   "open"     — followed by no space (next token attaches)
    #   "close"    — preceded by no space, followed by space
    #   "tight"    — no surrounding spaces (dash/hyphen)
    #   "loose"    — surrounding spaces (em dash)
    (r"new\s+paragraph", "\n\n", "newline"),
    (r"new\s*line|newline", "\n", "newline"),
    (r"full\s+stop", ".", "punct"),
    (r"question\s+mark", "?", "punct"),
    (r"exclamation\s+(?:point|mark)", "!", "punct"),
    (r"open\s+paren(?:thesis)?", "(", "open"),
    (r"close\s+paren(?:thesis)?", ")", "close"),
    (r"em\s+dash", "—", "loose"),
    (r"semicolon", ";", "punct"),
    (r"colon", ":", "punct"),
    (r"comma", ",", "punct"),
    (r"period", ".", "punct"),
    (r"hyphen", "-", "tight"),
    (r"dash", "-", "tight"),
]


def _build_spoken_punct_pattern() -> re.Pattern[str]:
    body = "|".join(f"(?:{phrase})" for phrase, _, _ in _SPOKEN_PUNCT_TABLE)
    return re.compile(rf"\b({body})\b", re.IGNORECASE)


_SPOKEN_PUNCT_PAT = _build_spoken_punct_pattern()


def _classify_spoken_punct(token: str) -> tuple[str, str]:
    """Return (glyph, kind) for a matched spoken-punctuation token."""
    norm = re.sub(r"\s+", " ", token.strip().lower())
    for phrase, glyph, kind in _SPOKEN_PUNCT_TABLE:
        if re.fullmatch(phrase, norm, re.IGNORECASE):
            return glyph, kind
    # Should not happen — pattern matched something we don't classify.
    return token, "punct"


def _spoken_punct_is_literal_mention(source: str, start: int, end: int) -> bool:
    before = (source or "")[max(0, start - 96):start].casefold()
    after = (source or "")[end:end + 64].casefold()
    # Determiner immediately before the cue ⇒ noun mention, not a spoken
    # command ("the new paragraph is short", "a comma goes here"). Inline
    # cues are spoken bare between content words; imperative phrasings like
    # "add a new paragraph" are writer commands and are not owned by this
    # inline rule either way.
    # "one" is deliberately absent: numerals commonly precede dictated
    # glyphs ("one em dash two" → "1 — 2"), unlike true determiners.
    if re.search(
        r"\b(?:the|a|an|this|that|these|those|each|every|any|no|my|your|our|"
        r"his|her|its|their|first|second|third|last|next|previous|another|same)\s+$",
        before,
    ):
        return True
    if re.search(
        r"\b(?:word|words|phrase|literal|text)\s+(?:called\s+|named\s+|as\s+)?$",
        before,
    ):
        return True
    if re.search(
        r"\b(?:say|says|said|mean|means|called|named|type|typed|write|written)\s+"
        r"(?:the\s+)?(?:word|words\s+)?$",
        before,
    ):
        return True
    if re.search(r"\b(?:is|are|was|were|means|mean|called|named)\s+(?:a|an|the)?\s*$", before):
        return True
    if re.search(r"\b(?:not|never|don't|dont|do\s+not)\b.{0,56}$", before) and re.match(
        r"\s+(?:as|in|inside|here|there|for|when|should|would|could|$)", after
    ):
        return True
    if re.match(r"\s+(?:as|in|inside)\s+(?:text|words?|this note)\b", after):
        return True
    return False


_SPOKEN_QUOTE_CUE_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?P<open>(?:open|begin|start)\s+quotes?)"
    r"|(?P<close>(?:close|end)\s+(?:of\s+)?quotes?|un\s*quote)"
    r"|(?P<bare>quotes?)"
    r")(?!\w)",
    re.IGNORECASE,
)
# A spoken quote pair spanning more than this many chars is more likely two
# unrelated mentions than one quotation.
_MAX_QUOTE_PAIR_SPAN = 240


def _spoken_quote_cue_is_literal(text: str, start: int, end: int) -> bool:
    """Noun/verb usage of bare "quote" — not a spoken quotation mark.

    Determiner or auxiliary before ("the quote", "I'll quote") and noun-ish
    continuations after ("quote from the article") read as content. The
    pairing requirement in the caller provides the second safety layer.
    """
    before = (text or "")[max(0, start - 48):start].casefold()
    after = (text or "")[end:end + 24].casefold()
    if re.search(
        r"\b(?:the|a|an|this|that|these|those|each|every|any|no|my|your|our|"
        r"his|her|its|their|one|another|same|to|will|would|can|could|cannot|"
        r"can't|won't|i'll|we'll|they'll|i|we|you|they)\s+$",
        before,
    ):
        return True
    if re.match(r"\s*(?:from|of|is|was|are|were|reads|says|in|on)\b", after):
        return True
    return False


def _apply_spoken_quote_pairs(text: str) -> tuple[str, list[str]]:
    """Convert paired spoken quote cues into straight quotes.

    "say quote we ship Friday unquote" → 'say "we ship Friday"'. Bare
    "quote" converts only when it pairs (an opener with a downstream
    closer); lone mentions ("I'll quote the answer", "the quote is long")
    stay literal. Explicit open/close forms still require a partner.
    """
    if not text:
        return text, []
    events: list[tuple[re.Match[str], str]] = []
    for m in _SPOKEN_QUOTE_CUE_RE.finditer(text):
        role = "open" if m.group("open") else ("close" if m.group("close") else "bare")
        # Bare "quote" gets a quote-specific literal guard. The generic
        # mention guard is wrong here: "He said quote we ship Friday
        # unquote" is the quotation idiom, yet the generic guard treats a
        # preceding "said" as a word-mention signal.
        if role == "bare" and _spoken_quote_cue_is_literal(text, m.start(), m.end()):
            continue
        events.append((m, role))
    if not events:
        return text, []

    pairs: list[tuple[re.Match[str], re.Match[str]]] = []
    pending: re.Match[str] | None = None
    for m, role in events:
        if role == "open":
            pending = pending or m
        elif role == "close":
            if pending is not None and (m.start() - pending.end()) <= _MAX_QUOTE_PAIR_SPAN:
                pairs.append((pending, m))
            pending = None
        else:  # bare — opener when nothing is pending, closer otherwise
            if pending is None:
                pending = m
            else:
                if (m.start() - pending.end()) <= _MAX_QUOTE_PAIR_SPAN:
                    pairs.append((pending, m))
                pending = None
    if not pairs:
        return text, []

    out = text
    for open_m, close_m in reversed(pairs):
        suffix = out[close_m.end():]
        if suffix and not suffix.startswith((" ", ",", ".", ";", ":", "!", "?", ")", "]", "\n")):
            suffix = " " + suffix
        out = out[: close_m.start()].rstrip() + '"' + suffix
        prefix = out[: open_m.start()].rstrip()
        body = out[open_m.end():].lstrip()
        out = (prefix + ' "' if prefix else '"') + body
    return out, ["spoken_quotes"]


def apply_spoken_punctuation(text: str) -> tuple[str, list[str]]:
    """Convert in-line spoken-punctuation tokens into glyphs.

    De-duplicates against an existing adjacent glyph (e.g. "hello, comma world"
    stays "hello, world" — no double comma). Whole-utterance commands like a
    lone "new line" are not owned by this rule.
    """
    applied: list[str] = []
    if not text:
        return text, applied
    text, quote_rules = _apply_spoken_quote_pairs(text)
    applied.extend(quote_rules)

    # First pass: substitute matches with sentinel-marked glyphs so we can
    # re-flow whitespace deterministically.
    OPEN, CLOSE = "\x00", "\x01"

    def _sub(m: re.Match) -> str:
        if _spoken_punct_is_literal_mention(text, m.start(), m.end()):
            return m.group(0)
        glyph, kind = _classify_spoken_punct(m.group(0))
        return f"{OPEN}{kind}:{glyph}{CLOSE}"

    interim = _SPOKEN_PUNCT_PAT.sub(_sub, text)
    if interim == text:
        return text, applied

    # Re-flow: unpack each sentinel and apply spacing rules; de-dup adjacent
    # equal glyphs.
    out: list[str] = []
    i = 0
    n = len(interim)
    sent_re = re.compile(rf"{OPEN}([^:]+):([^{CLOSE}]*){CLOSE}")
    while i < n:
        if interim[i] == OPEN:
            m = sent_re.match(interim, i)
            if not m:
                out.append(interim[i])
                i += 1
                continue
            kind = m.group(1)
            glyph = m.group(2)
            # Strip trailing whitespace already accumulated (so "hello ," → "hello,")
            if kind in ("punct", "close", "newline", "tight", "loose"):
                while out and out[-1] == " ":
                    out.pop()
            # ASR glues clause punctuation onto spoken cues ("…people, new
            # paragraph" / "wait, period"). A paragraph break must not leave a
            # dangling comma, and a spoken terminal mark replaces a comma the
            # ASR put in front of it.
            if kind == "newline" or (kind == "punct" and glyph not in {",", ";"}):
                while out and out[-1] in ",;":
                    out.pop()
                    while out and out[-1] == " ":
                        out.pop()
            # De-dup: if previous non-space char is the same glyph, skip insertion.
            tail = "".join(out)
            prev_nonspace = tail.rstrip()
            already = False
            if glyph == "\n" or glyph == "\n\n":
                # Don't double newlines mid-stream.
                if tail.endswith("\n\n") or (glyph == "\n" and tail.endswith("\n")):
                    already = True
            elif prev_nonspace.endswith(glyph):
                already = True
            if already:
                # Glyph already present; ensure standard spacing follows
                # (caller's leading whitespace was stripped).
                if kind in ("punct", "close", "loose"):
                    if not out or out[-1] != " ":
                        out.append(" ")
            if not already:
                if kind == "loose":
                    # surrounding spaces
                    if out and out[-1] != " ":
                        out.append(" ")
                    out.append(glyph)
                    out.append(" ")
                elif kind == "tight":
                    out.append(glyph)
                elif kind == "open":
                    out.append(glyph)
                elif kind == "close":
                    out.append(glyph)
                    out.append(" ")
                elif kind == "newline":
                    out.append(glyph)
                else:  # punct
                    out.append(glyph)
                    out.append(" ")
            # Skip following whitespace in input — every kind owns its own
            # trailing spacing. "open"/"tight" emit none, which is exactly
            # why the spoken gap must be consumed here: "open paren example"
            # attaches as "(example", "well hyphen known" as "well-known".
            j = m.end()
            while j < n and interim[j] == " ":
                j += 1
            if kind == "newline":
                # Consume punctuation the ASR attached to the spoken cue
                # ("New paragraph, text…" → the comma belongs to the cue,
                # not the new paragraph).
                while j < n and interim[j] in ".,;:!?":
                    j += 1
                    while j < n and interim[j] == " ":
                        j += 1
            i = j
            continue
        out.append(interim[i])
        i += 1

    result = "".join(out)
    # Final cleanup: collapse stray double spaces created by substitutions.
    result = re.sub(r"[ \t]{2,}", " ", result)
    # Trim trailing whitespace before newlines.
    result = re.sub(r"[ \t]+\n", "\n", result)
    # Trim trailing space at end of string.
    result = result.rstrip(" \t")
    if result != text:
        applied.append("spoken_punctuation")
    return result, applied
