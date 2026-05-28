"""Bag of Hallucinations (BoH) — Whisper silence-hallucination blocklist.

Whisper-large-v3-turbo was trained on YouTube transcripts and reverts to
end-of-video corpus when fed silence or low-signal audio. The arXiv:2501.11378
paper ran inference over 301,317 non-speech audio clips and catalogued the
top output phrases. The top 30 cover >70% of cases.

We intentionally keep this list small and matched at the SEGMENT level
(not substring), so legitimate "Thank you." utterances inside real speech
aren't suppressed. The conditional filter in ``streaming_core`` applies
this blocklist only when Whisper's own ``no_speech_prob`` is also elevated.

Sources:
- arXiv:2501.11378 Table III ("Investigation of Whisper ASR Hallucinations
  Induced by Non-Speech Audio") — top 15 phrases with frequencies.
- openai/whisper Discussion #1873 ("Share your hallucinations here") —
  community reports of additional patterns.
- openai/whisper Discussion #2378 (Pre-processings to reduce hallucinations).
"""

from __future__ import annotations

import re

# Phrases are stored in their NORMALIZED form (lowercase, no punctuation,
# collapsed whitespace). Matching is on _normalize(segment_text).
# Keep below ~120 entries — diminishing returns past the top 30 and risk of
# false positives grows. Add an entry only after seeing it in a real trace.
_HALLUCINATION_PHRASES: frozenset[str] = frozenset(
    {
        # arXiv 2501.11378 Table III, top 15 by frequency
        "thank you",                                      # 24.76%
        "thanks for watching",                            # 10.32%
        "so",                                             # 3.80%
        "thank you for watching",                         # 2.58%
        "the",                                            # 2.50%
        "you",                                            # 2.24%
        "oh",                                             # 1.83%
        "okay",                                           # 0.94%
        "im sorry",                                       # 0.77% ("i'm sorry" → "im sorry" after strip)
        "oh my god",                                      # 0.69%
        "bye",                                            # 0.56%
        "im not sure what im doing here",                 # 0.54%
        "uh",                                             # 0.53%
        "meow",                                           # 0.48%
        "subtitles by the amaraorg community",            # 0.46% ("amara.org" → "amaraorg")

        # arXiv 2501.11378 paper text mentions also
        "subtitles by steamteamextra",
        "hello everyone welcome to my channel",

        # openai/whisper Discussion #1873 — community reports
        "thanks for watching",
        "thank you for watching",
        "bye bye",
        "transcription by castingwords",
        "transcription by",
        "music",
        "applause",
        "laughter",
        "silence",
        "the end",
        "welcome to a new video",
        "like and subscribe",
        "please subscribe",
        "please like and subscribe",
        "subscribe to my channel",
        "dont forget to subscribe",
        "see you next time",
        "see you in the next video",
        "see you next video",
        "go to the next video",
        "going to go to the next video",
        "im going to go to the next video",
        "see you later",
        "see ya",
        "goodbye",
        "good bye",

        # Subtitle/credit artifacts
        "amaraorg",
        "by amaraorg",
        "subtitles by amaraorg",
        "subtitles",
        "captions by",
        "translation by",
        "subscribe and like",

        # Single fillers Whisper emits on near-silence
        "uh huh",
        "mm hmm",
        "mhm",
        "hmm",
        "um",
        "yeah",
        "yep",
        "right",
        "okay then",
        "all right",
        "alright",

        # Common short Whisper hallucinations from in-the-wild traces
        "thanks",
        "thats it",
        "thats all",
        "next",
        "and",
        "let me see",
        "let me know",
        "good luck",
        "youre welcome",
    }
)

# Sentinel chars Whisper occasionally emits whole-segment for music/silence
_HALLUCINATION_SENTINELS: frozenset[str] = frozenset(
    {
        "",
        "♪",
        "♫",
        "♩",
        "♬",
        "*",
        "...",
        "…",
        ".",
        "!",
        "?",
    }
)


# Tier-A subset: phrases whose presence at the END of a committed transcript
# is overwhelmingly hallucination, NOT legitimate dictation. These are
# multi-word YouTube-corpus signoffs and subtitle artifacts. We use this
# subset (not the full ``_HALLUCINATION_PHRASES`` set) for the
# "strip trailing BoH from committed before feeding initial_prompt" path —
# the full set includes single-word fillers like "so" and "you" that
# legitimately end real sentences and must not be stripped.
_TRAIL_STRIP_PHRASES: frozenset[str] = frozenset(
    {
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "thanks for watching everyone",
        "thank you very much",
        "thank you so much",
        "thanks",
        "subtitles by the amaraorg community",
        "subtitles by amaraorg",
        "subtitles by",
        "subtitles by steamteamextra",
        "amaraorg",
        "by amaraorg",
        "transcription by castingwords",
        "transcription by",
        "captions by",
        "translation by",
        "like and subscribe",
        "please subscribe",
        "please like and subscribe",
        "subscribe to my channel",
        "dont forget to subscribe",
        "subscribe and like",
        "see you next time",
        "see you in the next video",
        "see you next video",
        "go to the next video",
        "going to go to the next video",
        "im going to go to the next video",
        "welcome to a new video",
        "hello everyone welcome to my channel",
        "the end",
        "bye bye",
        "goodbye",
        "good bye",
        "youre welcome",
        "im not sure what im doing here",
    }
)

# Maximum number of trailing words to consider when stripping. The longest
# Tier-A phrases are YouTube signoffs such as
# "im going to go to the next video" (8 normalized words).
_TRAIL_STRIP_MAX_WORDS = 9


_PUNCT_STRIP_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation/symbols, collapse whitespace.

    Punctuation is replaced with empty string (NOT space) so apostrophes
    and inline periods collapse to glued tokens — that matches the
    blocklist entries.

    Examples:
        "Thank you."         → "thank you"
        "I'm sorry"          → "im sorry"
        "Subtitles by the Amara.org community" → "subtitles by the amaraorg community"
        "♪"                  → ""
        "Thanks for watching!" → "thanks for watching"
    """
    lowered = text.lower()
    stripped = _PUNCT_STRIP_RE.sub("", lowered)
    return _WS_RE.sub(" ", stripped).strip()


def is_whisper_silence_hallucination(text: str) -> bool:
    """True if the segment text (after normalization) is a known Whisper
    silence/YouTube hallucination.

    Matching is on the WHOLE normalized text — substring matches would
    catch legitimate "Thank you" inside real sentences. This is a strict
    full-segment match.

    The caller is expected to gate this with another signal (e.g. only
    suppress when ``no_speech_prob > some_threshold``) to avoid dropping
    legitimate short utterances like a user dictating "Thank you."
    """
    if text is None:
        return False
    # Whisper occasionally emits sentinel symbols as a whole segment
    raw = text.strip()
    if raw in _HALLUCINATION_SENTINELS:
        return True
    normalized = _normalize(text)
    if not normalized:
        return True
    return normalized in _HALLUCINATION_PHRASES


def phrase_count() -> int:
    """For visibility — total phrases plus sentinels in the blocklist."""
    return len(_HALLUCINATION_PHRASES) + len(_HALLUCINATION_SENTINELS)


def strip_trailing_boh(text: str) -> tuple[str, str | None]:
    """Strip trailing Tier-A BoH phrases from the end of ``text``.

    Returns ``(cleaned_text, removed_phrase_or_None)``. Iterates from
    longest suffix down to single-word, removing any trailing Tier-A
    match. Repeats until no more matches — handles cascading hallucinations
    like "thanks for watching thank you" (both phrases removed).

    Used for two purposes:
      1. Cleaning Whisper's ``initial_prompt`` so a previously-committed
         hallucination doesn't bias the next decode (preventing the
         "Thank you → Thank you → Thank you" lock-in observed in round 5).
      2. Cleaning committed preview state so a hallucination that slipped
         past Layers 1-3 doesn't persist in the HUD or get fed back as prompt
         text while the final paste catches up.

    Single-word fillers ("so", "you", "the", "uh") are intentionally NOT
    in the Tier-A list — they legitimately end real sentences and must
    not be stripped from displayed text. Those are handled at SEGMENT
    intake time via Layer 2's ``no_speech_prob > 0.6`` filter.

    Idempotent: ``strip_trailing_boh(strip_trailing_boh(x)[0])[0] == strip_trailing_boh(x)[0]``.
    """
    if not text:
        return text, None
    removed: list[str] = []
    current = text
    while True:
        stripped = current.strip()
        if not stripped:
            break
        words = stripped.split()
        if not words:
            break
        n_max = min(_TRAIL_STRIP_MAX_WORDS, len(words))
        match_n: int | None = None
        for n in range(n_max, 0, -1):
            suffix_raw = " ".join(words[-n:])
            suffix_norm = _normalize(suffix_raw)
            if suffix_norm in _TRAIL_STRIP_PHRASES:
                match_n = n
                removed.append(suffix_raw)
                break
        if match_n is None:
            break
        current = " ".join(words[:-match_n]).strip()
    if not removed:
        return text, None
    return current, " | ".join(removed)


def trail_strip_phrase_count() -> int:
    """Visibility for the Tier-A trail-strip list size."""
    return len(_TRAIL_STRIP_PHRASES)
