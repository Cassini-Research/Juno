from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from threading import RLock

_RE_CC = re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14})\b")
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_RE_PASSWORD = re.compile(r"(?i)(?:password|passwd)\s*:\s*\S+")
_RE_REDACTED_MARKER = re.compile(r"^\[REDACTED \d+ chars\]$")


@dataclass(frozen=True, slots=True)
class ClipboardEntry:
    text: str
    ts_unix_ms: int
    redacted: bool = False


def _classify_for_storage(text: str) -> tuple[str, bool]:
    if _RE_REDACTED_MARKER.match(text):
        return text, True
    if "BEGIN PRIVATE KEY" in text or "BEGIN RSA PRIVATE KEY" in text:
        return f"[REDACTED {len(text)} chars]", True
    if (
        _RE_CC.search(text) is not None
        or _RE_SSN.search(text) is not None
        or _RE_PASSWORD.search(text) is not None
    ):
        return f"[REDACTED {len(text)} chars]", True
    return text, False


class ClipboardRingBuffer:
    """Thread-safe, bounded, redacted history of clipboard text."""

    def __init__(self, max_items: int = 20, max_chars: int = 4000) -> None:
        self._max_items = max_items
        self._max_chars = max_chars
        self._lock = RLock()
        self._deque: deque[ClipboardEntry] = deque()
        self._char_total = 0
        self._last_raw_push: str | None = None

    def push(self, text: str, *, ts_unix_ms: int | None = None) -> None:
        ts = int(time.time() * 1000) if ts_unix_ms is None else ts_unix_ms
        stored, redacted = _classify_for_storage(text)
        with self._lock:
            if self._last_raw_push == text:
                return
            self._last_raw_push = text
            entry = ClipboardEntry(text=stored, ts_unix_ms=ts, redacted=redacted)
            self._deque.append(entry)
            self._char_total += len(stored)
            self._evict_until_fits()

    def recent(self, limit: int = 5) -> list[ClipboardEntry]:
        with self._lock:
            items = list(self._deque)
        newest_first: list[ClipboardEntry] = []
        for i in range(len(items) - 1, -1, -1):
            newest_first.append(items[i])
            if len(newest_first) >= limit:
                break
        return newest_first

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()
            self._char_total = 0
            self._last_raw_push = None

    def _evict_until_fits(self) -> None:
        while len(self._deque) > self._max_items:
            self._pop_oldest()
        while self._char_total > self._max_chars and self._deque:
            self._pop_oldest()

    def _pop_oldest(self) -> None:
        oldest = self._deque.popleft()
        self._char_total -= len(oldest.text)
