#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _join(*parts: str) -> str:
    return "".join(parts)


def _literal(*parts: str) -> re.Pattern[str]:
    return re.compile(re.escape(_join(*parts)), re.I)


def _word(*parts: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(_join(*parts)) + r"\b", re.I)


FORBIDDEN: dict[str, re.Pattern[str]] = {
    _join("ha", "rpy"): re.compile(_join("ha", "rpy") + "|" + _join("ha", "rpie"), re.I),
    _join("super", "whisper"): _literal("super", "whisper"),
    _join("ot", "ter"): _word("ot", "ter"),
    _join("gran", "ola"): _word("gran", "ola"),
    _join("fire", "flies"): _word("fire", "flies"),
    _join("clau", "de"): _word("clau", "de"),
    _join("co", "dex"): _word("co", "dex"),
    _join("chat", "gpt"): _word("chat", "gpt"),
    _join("gem", "ini"): _word("gem", "ini"),
    _join("co", "pilot"): _word("co", "pilot"),
    _join("wind", "surf"): _word("wind", "surf"),
    _join("tr", "ae"): _word("tr", "ae"),
    _join("anth", "ropic"): _word("anth", "ropic"),
    _join("open", "whispr"): _word("open", "whispr"),
    _join("dic", "tion"): _word("dic", "tion"),
    _join("chall", "enger"): _word("chall", "enger"),
    _join("gate", "keeper"): _word("gate", "keeper"),
    _join("hand", "over"): _word("hand", "over"),
    _join("product", "_eval"): _word("product", "_eval"),
    _join("audit", " flagged"): _literal("audit", " flagged"),
    _join("local", " mock"): _literal("local", " mock"),
    _join("agent", "_mistakes"): _literal("agent", "_mistakes"),
    _join("agent", "_process"): _literal("agent", "_process"),
    _join("cursor", "_tasks"): _literal("cursor", "_tasks"),
    _join("const", "ellation"): _word("const", "ellation"),
}

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".html",
    ".json",
    ".md",
    ".plist",
    ".py",
    ".sh",
    ".svg",
    ".swift",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".build",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _skip_path(path: Path) -> bool:
    if path == Path(__file__).resolve():
        return True
    rel_parts = path.relative_to(ROOT).parts
    if any(part in SKIP_DIRS for part in rel_parts):
        return True
    if any(part.startswith(".juno_v2_") for part in rel_parts):
        return True
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore", ".env.example"}:
        return True
    return False


def _allow_reason(path: Path, term: str) -> str | None:
    rel = _relative(path)
    if term == _join("ha", "rpy"):
        return None
    if rel.startswith("seed_data/packs/"):
        return "dictionary/domain vocabulary pack"
    if rel.startswith("juno_v2/memory/"):
        return "dictionary and memory feature code"
    lower = rel.lower()
    if any(token in lower for token in ("dictionary", "vocabulary", "pronunciation", "protected_term", "term_policy")):
        return "dictionary/vocabulary handling"
    if rel.startswith("juno_core_v3/dictation/") and term == _join("dic", "tion"):
        return "transcription API compatibility"
    if rel in {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}:
        return "legal notice"
    return None


def iter_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*") if path.is_file() and not _skip_path(path))


def main() -> int:
    failures: list[str] = []
    allowed: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for term, pattern in FORBIDDEN.items():
                if not pattern.search(line):
                    continue
                reason = _allow_reason(path, term)
                message = f"{_relative(path)}:{line_no}: {term}: {line.strip()}"
                if reason:
                    allowed.append(f"ALLOW {message} [{reason}]")
                else:
                    failures.append(message)
    for item in allowed:
        print(item)
    if failures:
        print("Forbidden public terms found:", file=sys.stderr)
        for item in failures:
            print(item, file=sys.stderr)
        return 1
    print("Public term audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
