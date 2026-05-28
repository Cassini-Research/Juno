"""Code grammar engine — deterministic, no model, no cloud.

Voice-to-code transforms:
1. Case styles: snake_case, camelCase, PascalCase, kebab-case, SCREAMING_SNAKE
2. File tagging: render @-prefixed file references for code-chat surfaces
3. Code-safe paths: dot-slash, tilde-slash shortcuts
4. Extension normalization: "dot ts" in code context → ".ts" (handled by ITN)
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class CodeGrammarMode(str, Enum):
    """Named case style and grammar modes."""

    SNAKE = "snake"              # hello_world
    CAMEL = "camel"              # helloWorld
    PASCAL = "pascal"            # HelloWorld
    KEBAB = "kebab"              # hello-world
    SCREAMING = "screaming"      # HELLO_WORLD
    FILE_TAG = "file_tag"        # @main.ts — code-chat file reference
    AUTO = "auto"                # detect from hint tokens in the text


@dataclass(slots=True)
class CodeGrammarResult:
    """Output of CodeGrammarEngine.apply()."""

    text: str
    original_text: str
    mode: str
    changed: bool
    rules_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, str | bool | list[str]]:
        return {
            "text": self.text,
            "original_text": self.original_text,
            "mode": self.mode,
            "changed": self.changed,
            "rules_applied": list(self.rules_applied),
        }


# ---------------------------------------------------------------------------
# Case conversion primitives
# ---------------------------------------------------------------------------

def _tokenize_phrase(phrase: str) -> list[str]:
    """Split a phrase into word tokens, stripping punctuation."""
    words = re.split(r"[\s_\-]+", phrase.strip())
    return [w for w in words if w]


def to_snake_case(phrase: str) -> str:
    """Convert a multi-word phrase to snake_case."""
    tokens = _tokenize_phrase(phrase)
    if not tokens:
        return phrase
    return "_".join(t.lower() for t in tokens)


def to_camel_case(phrase: str) -> str:
    """Convert a multi-word phrase to camelCase."""
    tokens = _tokenize_phrase(phrase)
    if not tokens:
        return phrase
    return tokens[0].lower() + "".join(t.capitalize() for t in tokens[1:])


def to_pascal_case(phrase: str) -> str:
    """Convert a multi-word phrase to PascalCase."""
    tokens = _tokenize_phrase(phrase)
    if not tokens:
        return phrase
    return "".join(t.capitalize() for t in tokens)


def to_kebab_case(phrase: str) -> str:
    """Convert a multi-word phrase to kebab-case."""
    tokens = _tokenize_phrase(phrase)
    if not tokens:
        return phrase
    return "-".join(t.lower() for t in tokens)


def to_screaming_snake(phrase: str) -> str:
    """Convert a multi-word phrase to SCREAMING_SNAKE_CASE."""
    tokens = _tokenize_phrase(phrase)
    if not tokens:
        return phrase
    return "_".join(t.upper() for t in tokens)


# ---------------------------------------------------------------------------
# File tagging
# ---------------------------------------------------------------------------

_FILE_EXT_ALIASES = {
    "ts": ".ts", "tsx": ".tsx", "js": ".js", "jsx": ".jsx",
    "py": ".py", "go": ".go", "rs": ".rs", "rb": ".rb",
    "swift": ".swift", "kt": ".kt", "java": ".java",
    "css": ".css", "html": ".html", "json": ".json",
    "yaml": ".yaml", "yml": ".yml", "toml": ".toml",
    "sh": ".sh", "md": ".md", "txt": ".txt",
    "c": ".c", "cpp": ".cpp", "h": ".h",
}


def _collapse_spoken_extension(text: str) -> str:
    out = text
    for ext in sorted(_FILE_EXT_ALIASES, key=len, reverse=True):
        spaced = r"\s+".join(re.escape(ch) for ch in ext)
        out = re.sub(rf"\bdot\s+{spaced}\b", f".{ext}", out, flags=re.IGNORECASE)
    return out


def normalize_spoken_filename(text: str) -> str:
    """Normalize simple spoken filename forms into code-style file references.

    Supported examples:
        "main dot ts" -> "main.ts"
        "main dot t s" -> "main.ts"
        "index underscore test dot p y" -> "index_test.py"

    This is intentionally narrow and deterministic. It only runs on explicit
    file-tag transforms and does not leak into generic punctuation paths.
    """
    raw = (text or "").strip().lstrip("@")
    if not raw:
        return raw
    out = re.sub(r"\s+", " ", raw)
    out = _collapse_spoken_extension(out)
    for pattern, replacement in (
        (r"\bunderscore\b", "_"),
        (r"\b(?:dash|hyphen)\b", "-"),
        (r"\bslash\b", "/"),
        (r"\bdot\b", "."),
    ):
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    out = re.sub(r"\s*([._/\-])\s*", r"\1", out)
    if any(marker in raw.casefold() for marker in (" dot ", " underscore ", " dash ", " hyphen ", " slash ")):
        out = re.sub(r"\s+", "", out)
    return out.strip()


def render_file_tag(filename: str) -> str:
    """Render a filename as an @-prefixed file reference.

    Examples:
        "main.ts"        → "@main.ts"
        "index_test.py"  → "@index_test.py"
        "app"            → "@app"
    """
    name = normalize_spoken_filename(filename)
    return f"@{name}"


def render_file_tag_from_parts(stem: str, extension: str | None = None) -> str:
    """Build an @-tagged file reference from a stem and optional extension.

    Examples:
        stem="main", ext="ts"  → "@main.ts"
        stem="index_test"      → "@index_test"
    """
    if extension:
        ext_dot = _FILE_EXT_ALIASES.get(extension.lower().lstrip("."), f".{extension}")
        return f"@{stem}{ext_dot}"
    return f"@{stem}"


# ---------------------------------------------------------------------------
# Voice-command detection (inline mode markers)
# ---------------------------------------------------------------------------

# Patterns that precede the phrase to convert, e.g. "snake case hello world"
_MODE_PREFIXES: list[tuple[re.Pattern, CodeGrammarMode]] = [
    (re.compile(r"\bsnake\s+case\b", re.IGNORECASE), CodeGrammarMode.SNAKE),
    (re.compile(r"\bcamel\s+case\b", re.IGNORECASE), CodeGrammarMode.CAMEL),
    (re.compile(r"\bpascal\s+case\b", re.IGNORECASE), CodeGrammarMode.PASCAL),
    (re.compile(r"\bkebab\s+case\b", re.IGNORECASE), CodeGrammarMode.KEBAB),
    (re.compile(r"\bscreaming\s+(?:snake\s+)?case\b", re.IGNORECASE), CodeGrammarMode.SCREAMING),
    (re.compile(r"\bat\s+file\b", re.IGNORECASE), CodeGrammarMode.FILE_TAG),
]

_CASE_FN: dict[CodeGrammarMode, Callable[[str], str]] = {
    CodeGrammarMode.SNAKE: to_snake_case,
    CodeGrammarMode.CAMEL: to_camel_case,
    CodeGrammarMode.PASCAL: to_pascal_case,
    CodeGrammarMode.KEBAB: to_kebab_case,
    CodeGrammarMode.SCREAMING: to_screaming_snake,
}


# ---------------------------------------------------------------------------
# CodeGrammarEngine
# ---------------------------------------------------------------------------

class CodeGrammarEngine:
    """Apply code-output grammar to a text string.

    Modes:
    - Explicit: caller passes `mode=CodeGrammarMode.SNAKE` etc.
    - AUTO: engine scans for voice-command prefix ("snake case <phrase>")
      and converts only the specified phrase.

    All operations are pure / deterministic.
    """

    def convert(self, phrase: str, *, mode: CodeGrammarMode | str) -> str:
        """Convert *phrase* to the target case style."""
        if isinstance(mode, str):
            try:
                mode = CodeGrammarMode(mode)
            except ValueError:
                return phrase

        fn = _CASE_FN.get(mode)
        if fn is None:
            return phrase
        return fn(phrase)

    def apply(self, text: str, *, mode: CodeGrammarMode | str = CodeGrammarMode.AUTO) -> CodeGrammarResult:
        """Apply grammar transforms to *text* using the specified *mode*.

        In AUTO mode, detect an inline voice-command prefix and convert only
        the phrase that follows it. Other text is left unchanged.
        """
        if isinstance(mode, str):
            try:
                mode = CodeGrammarMode(mode)
            except ValueError:
                mode = CodeGrammarMode.AUTO

        if mode == CodeGrammarMode.AUTO:
            return self._apply_auto(text)

        out = self.convert(text, mode=mode)
        changed = out != text
        return CodeGrammarResult(
            text=out,
            original_text=text,
            mode=mode.value,
            changed=changed,
            rules_applied=[mode.value] if changed else [],
        )

    def _apply_auto(self, text: str) -> CodeGrammarResult:
        """Scan *text* for inline mode markers and apply them.

        Pattern: "<mode prefix> <phrase>" where the phrase extends to the
        next sentence boundary or end of string.
        """
        out = text
        applied: list[str] = []

        for base_pat, mode in _MODE_PREFIXES:
            # Build a capturing pattern: marker + whitespace + phrase
            full_pat = re.compile(
                base_pat.pattern + r"\s+([\w\s]+?)(?=[,;.]|$)",
                re.IGNORECASE,
            )

            def _replace(m: re.Match, _mode: CodeGrammarMode = mode) -> str:
                phrase = m.group(1).strip()
                if not phrase:
                    return m.group(0)
                if _mode == CodeGrammarMode.FILE_TAG:
                    return render_file_tag(phrase)
                fn = _CASE_FN.get(_mode)
                if fn is None:
                    return m.group(0)
                return fn(phrase)

            new = full_pat.sub(_replace, out)
            if new != out:
                applied.append(mode.value)
                out = new

        return CodeGrammarResult(
            text=out,
            original_text=text,
            mode="auto",
            changed=(out != text),
            rules_applied=applied,
        )
