from __future__ import annotations

import re

from juno_v2.contracts.memory import CorrectionPair
from juno_v2.memory.fold import fold_key
from juno_v2.memory.hallucination import looks_like_hallucination
from juno_v2.memory.stores._base import JsonFileStore

# Corrections longer than this are almost certainly hallucinated repetitions
# or garbage ASR output — storing them would contaminate future bias plans.
MAX_CORRECTION_TEXT_CHARS = 120
MAX_CORRECTION_WORDS = 12

# Hard cap on the number of correction rows persisted to ``corrections.json``.
# Beyond this, the store evicts rows ordered by ``(count ASC, list-position
# ASC)`` — i.e. lowest-frequency, oldest entries first — to bound disk size
# and write amplification (Issue #13).
MAX_CORRECTIONS = 1000


def is_safe_correction_pair(observed: str, corrected: str) -> bool:
    observed = (observed or "").strip()
    corrected = (corrected or "").strip()
    if not observed or not corrected or observed == corrected:
        return False
    if len(observed) > MAX_CORRECTION_TEXT_CHARS or len(corrected) > MAX_CORRECTION_TEXT_CHARS:
        return False
    if _word_count(observed) > MAX_CORRECTION_WORDS or _word_count(corrected) > MAX_CORRECTION_WORDS:
        return False
    if _contains_wake_phrase_only_on_one_side(observed, corrected):
        return False
    if _looks_like_fragment_expansion(observed, corrected):
        return False
    if _looks_like_case_or_punctuation_only_edit(observed, corrected) and _contains_low_signal_titlecase(corrected):
        return False
    if looks_like_hallucination(observed) or looks_like_hallucination(corrected):
        return False
    return True


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*", text))


def _collapsed(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _semantic_words(text: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*", text or "")).casefold()


def _looks_like_case_or_punctuation_only_edit(observed: str, corrected: str) -> bool:
    """Reject "corrections" that only change casing/punctuation.

    These rows are usually artifacts from test utterances or over-eager final
    formatting. Replaying them as memory corrections is how stale rows like
    "the live" -> "The live" contaminate future dictation.
    """
    return _semantic_words(observed) == _semantic_words(corrected)


_LOW_SIGNAL_TITLECASE_CORRECTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "clearly", "did", "do", "does", "every", "for", "from",
    "go", "he", "here", "how", "i", "if", "in", "is", "it", "its",
    "just", "like", "me", "missing", "my", "no", "not", "now", "of",
    "on", "or", "our", "see", "she", "that", "the", "then", "there",
    "they", "things", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "with", "would", "yes",
    "you",
}


def _contains_low_signal_titlecase(text: str) -> bool:
    for token in re.findall(r"\b[A-Z][a-z]{1,}\b", text or ""):
        if token.casefold() in _LOW_SIGNAL_TITLECASE_CORRECTION_WORDS:
            return True
    return False


def _contains_wake_phrase_only_on_one_side(observed: str, corrected: str) -> bool:
    observed_action = _looks_like_juno_action_wrapper(observed)
    corrected_action = _looks_like_juno_action_wrapper(corrected)
    return observed_action != corrected_action


def _looks_like_juno_action_wrapper(text: str) -> bool:
    """True for wake-word command wrappers, not product mentions like "Juno v2"."""
    return re.search(
        r"^\s*(?:hey|hello)?\s*juno\b\s*(?:,|\b)\s*"
        r"(?:take|add|create|make|remind|remember|note|set|schedule|alarm)\b",
        text or "",
        flags=re.IGNORECASE,
    ) is not None


def _enforce_cap(data: list[dict]) -> list[dict]:
    """Trim ``data`` to at most :data:`MAX_CORRECTIONS` rows.

    Eviction policy: keep rows with the highest ``count``; among ties,
    keep the most recently inserted (largest list index — newer rows are
    appended). Reads :data:`MAX_CORRECTIONS` at call time so tests can
    monkeypatch it without re-importing.
    """
    cap = MAX_CORRECTIONS
    if len(data) <= cap:
        return data
    indexed = list(enumerate(data))
    indexed.sort(
        key=lambda pair: (int(pair[1].get("count", 1)), pair[0]),
        reverse=True,
    )
    kept = indexed[:cap]
    kept.sort(key=lambda pair: pair[0])
    return [item for _, item in kept]


def _looks_like_fragment_expansion(observed: str, corrected: str) -> bool:
    observed_norm = _collapsed(observed)
    corrected_norm = _collapsed(corrected)
    if not observed_norm or not corrected_norm:
        return False

    observed_words = _word_count(observed)
    corrected_words = _word_count(corrected)
    word_delta = corrected_words - observed_words
    char_delta = len(corrected_norm) - len(observed_norm)

    if observed_norm in corrected_norm and (word_delta >= 3 or char_delta >= 20):
        return True
    if observed_words <= 4 and corrected_words >= max(8, observed_words * 2 + 2):
        return True
    return False


class CorrectionStore:
    """Observed-to-corrected pairs learned after commit.

    Stored as ``corrections.json``. Guards against ASR-hallucinated inputs
    via :func:`looks_like_hallucination` so the memory plane doesn't learn
    noise.
    """

    FILENAME = "corrections.json"

    def __init__(self, file_store: JsonFileStore) -> None:
        self._fs = file_store
        self._fs.ensure_default([])

    def list(self) -> list[CorrectionPair]:
        return [CorrectionPair(**item) for item in self._fs.read([])]

    def raw(self) -> list[dict]:
        return list(self._fs.read([]))

    def record(self, observed: str, corrected: str) -> bool:
        observed = (observed or "").strip()
        corrected = (corrected or "").strip()
        if not is_safe_correction_pair(observed, corrected):
            return False
        observed_key = fold_key(observed)
        corrected_key = fold_key(corrected)
        with self._fs.lock:
            data = self._fs.read([])
            for idx, item in enumerate(data):
                # Dedup via fold_key so an existing correction "u r" → "you are"
                # also matches a new attempt "U.R" → "You are" — same intent.
                if (
                    fold_key(str(item.get("observed", ""))) == observed_key
                    and fold_key(str(item.get("corrected", ""))) == corrected_key
                ):
                    data[idx] = {
                        **item,
                        "count": int(item.get("count", 1)) + 1,
                    }
                    self._fs.write(data)
                    return True
            data.append(
                CorrectionPair(observed=observed, corrected=corrected, count=1).to_dict()
            )
            data = _enforce_cap(data)
            self._fs.write(data)
            return True

    def remove(self, observed: str, corrected: str | None = None) -> bool:
        """Delete one or all correction rows whose ``observed`` matches.

        When ``corrected`` is provided, only the exact ``(observed,
        corrected)`` pair is removed (matched via :func:`fold_key`). When
        it is ``None``, every row with a matching ``observed`` fold-key
        is deleted. Returns ``True`` when at least one row was removed.
        """
        needle = fold_key(observed)
        if not needle:
            return False
        target_cor = fold_key(corrected) if corrected else None
        with self._fs.lock:
            data = self._fs.read([])
            before = len(data)
            kept: list[dict] = []
            for item in data:
                obs = fold_key(str(item.get("observed", "")))
                cor = fold_key(str(item.get("corrected", "")))
                if obs == needle and (target_cor is None or cor == target_cor):
                    continue
                kept.append(item)
            if len(kept) == before:
                return False
            self._fs.write(kept)
            return True
