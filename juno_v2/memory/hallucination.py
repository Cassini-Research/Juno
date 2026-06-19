"""Hallucination heuristic used by the correction store.

Extracted from the monolithic ``memory/store.py`` so each decomposed store
can depend on just the signal it needs. The logic is unchanged —
``test_context_and_memory.py`` exercises the thresholds, so we keep them
identical.
"""

from __future__ import annotations

import re
from collections import Counter


# Confidence threshold above which the word-repetition heuristic is
# suppressed. mlx_whisper's avg_logprob sits around -0.3 to -0.5 on clean
# confident speech and -1.5 or worse on hallucinated silence or loops. A
# user legitimately saying "hello hello hello" reliably lands above -1.0,
# so we use that as the cutoff. When confidence is unknown we fall back to
# text-only heuristics to stay conservative.
HALLUCINATION_CONFIDENCE_FLOOR = -1.0


def looks_like_hallucination(text: str, *, confidence: float | None = None) -> bool:
    """Return True if *text* looks like ASR hallucination, not real speech.

    See ``juno_v2/memory/store.py`` history for the full rationale; we
    preserve the behaviour exactly. Six shapes the rule catches:

    1. All-short-token noise (``"A.D. A.D. A.D."``)
    2. Dominant substantive token (``"thu thu thu thu thu"``)
    3. CJK glyph loops / single-character dominance
    4. Substring loops with or without whitespace
       (``"ansaansaansaansa"``, ``"Dagmengdola, Dagmengdola, Dagmengdola"``)
       — added after observing real whisper-large-v3-turbo failures on
       continuous low-information audio.
    5. Pathologically low distinct-token ratio in long texts
       (``"cardiac cardiac cardiac ..."`` × 200,
       ``"make sure that we make sure that we ..."`` × 50). When the
       transcript runs to 50+ words and only a handful are unique,
       it's an autoregressive hallucination loop regardless of
       confidence. Real speech at that length has lexical diversity
       above 0.4 (heavy stopword usage still leaves room for nouns,
       verbs, adjectives). This check intentionally bypasses the
       confidence floor — whisper sometimes reports high
       avg_logprob on long token-loop hallucinations.
    6. Mixed-script characters within a single token (``"Clициц"``,
       ``"Hellодом"``). Whisper occasionally emits these on near-silent
       audio. Real speakers segment scripts with whitespace
       (``"Hello мир"``, not ``"Hellомир"``).

    And two legitimate shapes it spares:

    a. Version strings like ``"Version 3.0.0.0 released"`` — the dominant
       token ``'0'`` is short and context has substantive words.
    b. English with many stopwords — ratio threshold keeps this below the
       firing line.
    """
    if not text or len(text) < 3:
        return False
    alnum = sum(1 for c in text if c.isalnum())
    if alnum / max(len(text), 1) < 0.4:
        return True

    all_words = re.findall(r"\b\w+\b", text.lower())
    substantive_words = [w for w in all_words if len(w) >= 2]
    skip_word_repetition = (
        confidence is not None and confidence >= HALLUCINATION_CONFIDENCE_FLOOR
    )
    if not skip_word_repetition and len(all_words) >= 4:
        most_common_token, most_common_count = Counter(all_words).most_common(1)[0]
        repeat_ratio = most_common_count / len(all_words)
        if most_common_count >= 3 and repeat_ratio >= 0.4:
            if len(substantive_words) < 2:
                return True  # case (1): all-short-token noise
            if len(most_common_token) >= 2:
                return True  # case (2): dominant real word repeats

    # Consecutive same CJK character 3+ times.
    if re.search(
        r"([\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af])\1{2,}",
        text,
    ):
        return True

    # Single character dominates.
    if len(text) >= 8:
        char_counts = Counter(c for c in text if not c.isspace())
        if char_counts:
            _, top_count = char_counts.most_common(1)[0]
            non_space = sum(char_counts.values())
            if top_count / max(non_space, 1) > 0.5:
                return True

    # Case (5): pathologically low distinct-token ratio in long text.
    # ``"cardiac" × 200`` and ``"make sure that we" × 50`` were both
    # observed slipping past the existing checks even with high confidence.
    # Low diversity alone is not enough, though: domain speech such as chess
    # commentary can repeat a small vocabulary for minutes. When ASR confidence
    # is good, require loop structure in addition to low diversity.
    if len(all_words) >= 50:
        distinct = len(set(all_words))
        if distinct / len(all_words) < 0.15:
            if skip_word_repetition and not _has_token_ngram_loop(all_words):
                return False
            return True

    # Number-marker loops: on low-information pauses Whisper can emit
    # ``"1. 2. 3. 3. 3. ... 10."`` and then recover a plausible tail.
    # That shape should never be committed as dictated text. Keep this
    # structural and confidence-independent, but require an actual marker
    # run so legitimate short numbered lists survive.
    if _has_numeric_marker_loop(text):
        return True

    # Case (4): substring loops. Whisper hallucinates patterns like
    # "ansaansaansa..." (no whitespace) on continuous low-information
    # audio. These slip past the word-repetition heuristic because the
    # loop has no spaces and `\b\w+\b` returns one giant token.
    #
    # Gated on the same confidence floor as case (2): a confident
    # speaker repeating a phrase (avg_logprob >= -1.0) is legitimate
    # ("hello hello hello hello hello" said into the mic on purpose).
    # Whisper's hallucination loops report avg_logprob well below
    # -1.0, so the floor cleanly separates the two.
    if not skip_word_repetition and _has_substring_loop(text):
        return True

    # Case (6): mixed-script characters within a single token. Whisper
    # produced ``"Clициц"`` (2 Latin + 4 Cyrillic in one word) on
    # ~8 seconds of low-information audio in the example user's 2026-04-29
    # support bundle — too short for the loop / repetition / distinct
    # ratio checks to fire. Real speakers don't produce a single word
    # that mixes Latin + Cyrillic / CJK / Arabic / Devanagari; even
    # legitimate code-mixing puts a space between scripts ("Hello мир",
    # not "Hellомир"). Confidence-floor gated like case (2) so the
    # rare confident speaker who really does emit such a word is
    # spared.
    if not skip_word_repetition and _has_mixed_script_within_token(text):
        return True

    return False


def _has_numeric_marker_loop(text: str) -> bool:
    marker_run = re.search(
        r"(?<!\d)(?:[1-9]|10)[.)](?:\s+(?:[1-9]|10)[.)]){4,}",
        text,
    )
    if marker_run:
        return True

    markers = re.findall(r"(?<!\d)(?:[1-9]|10)[.)]", text)
    if len(markers) < 8:
        return False
    if len(set(markers)) < len(markers):
        return True
    content = re.sub(r"(?<!\d)(?:[1-9]|10)[.)]", " ", text)
    content_words = [w for w in re.findall(r"\b[a-zA-Z]{2,}\b", content)]
    return len(content_words) < len(markers) // 2


def _has_token_ngram_loop(words: list[str]) -> bool:
    """Detect long word-level loops that can report high ASR confidence."""
    if not words:
        return False
    total = len(words)
    top_count = Counter(words).most_common(1)[0][1]
    if top_count / total >= 0.33:
        return True
    for n in (2, 3, 4, 5):
        grams = [tuple(words[i : i + n]) for i in range(0, total - n + 1)]
        if not grams:
            continue
        _, count = Counter(grams).most_common(1)[0]
        if count >= 8 and (count * n) / total >= 0.55:
            return True
    return False


def _has_substring_loop(
    text: str,
    *,
    min_len: int = 3,
    max_len: int = 8,
    min_repeats: int = 4,
    coverage_threshold: float = 0.6,
) -> bool:
    """Detect whether *text* is dominated by a short repeating substring.

    Strips whitespace and ASCII punctuation, then tallies every n-gram in
    the stripped text and asks whether the most-common n-gram occurs at
    least ``min_repeats`` times AND covers at least
    ``coverage_threshold`` of the stripped text.

    The earlier implementation only scanned substrings starting at
    offsets 0..3 as a CPU optimization. That missed loops with a
    non-loop prefix — e.g. ``"www. alberalalalalal..."`` from the example user's
    fresh-install support bundle, where the actual ``ala``/``lal`` loop
    only starts at offset ~8 after a ``wwwalber`` prefix. A full Counter
    scan is O(len) per n-gram length and ``max_len - min_len + 1 = 6``,
    so total work is ~6 * len which is trivial for hallucination-sized
    inputs (under a few hundred chars).
    """
    stripped = re.sub(r"[\s,.\-_!?;:'\"]+", "", text.lower())
    if len(stripped) < min_len * min_repeats:
        return False
    upper_n = min(max_len, len(stripped) // min_repeats)
    needed_coverage = len(stripped) * coverage_threshold
    for n in range(min_len, upper_n + 1):
        ngrams: Counter[str] = Counter()
        for i in range(len(stripped) - n + 1):
            ngrams[stripped[i : i + n]] += 1
        if not ngrams:
            continue
        _, top_count = ngrams.most_common(1)[0]
        if top_count >= min_repeats and top_count * n >= needed_coverage:
            return True
    return False


# Per-script regex patterns. Each entry is (label, compiled regex). A
# token containing characters that match more than one of these patterns
# is treated as mixed-script — see ``_has_mixed_script_within_token``.
# Order matters only for readability; we count distinct hits, not the
# first one. Devanagari is included for en_IN / hi locales (the example user's
# laptop reports ``Locale: en_IN``); Arabic for general support.
_SCRIPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Latin", re.compile(r"[A-Za-z]")),
    ("Cyrillic", re.compile(r"[Ѐ-ӿԀ-ԯ]")),
    ("CJK", re.compile(r"[぀-ゟ゠-ヿ㐀-䶿一-鿿가-힯]")),
    ("Arabic", re.compile(r"[؀-ۿݐ-ݿ]")),
    ("Devanagari", re.compile(r"[ऀ-ॿ]")),
)


def _has_mixed_script_within_token(text: str) -> bool:
    """Return True if any single ``\\w+`` token in *text* contains
    characters from two or more incompatible writing systems.

    Real speakers segment script switches with whitespace
    (``"Hello мир"`` reads as two tokens, each pure-script). Whisper
    hallucinations occasionally emit a single word that fuses Latin +
    Cyrillic / CJK / Arabic — ``"Clициц"`` (the example user's 2026-04-29 bundle,
    6 chars from 8.4s audio) is the canonical example. Five-script
    coverage matches the language set Juno supports today plus the
    en_IN locale we've observed in the field.
    """
    for token in re.findall(r"\w+", text, flags=re.UNICODE):
        scripts_present = 0
        for _label, pattern in _SCRIPT_PATTERNS:
            if pattern.search(token):
                scripts_present += 1
                if scripts_present >= 2:
                    return True
    return False


# ---------------------------------------------------------------------------
# Silence-phrase guard
# ---------------------------------------------------------------------------
#
# Whisper-family models emit a small set of stock phrases when given silent
# or near-silent audio: "Thank you.", "Thanks for watching.", "you", a bare
# ".", etc. These are linguistically clean and pass the structural checks
# above (they are not loops, not mixed-script, not low-alnum), so the
# confidence-gated repetition rules can't catch them. They also frequently
# decode with avg_logprob > -1.0 — the model is internally "confident" in
# the hallucination — so a confidence-only gate does not work either.
#
# The discriminator that DOES work is corroboration with audio-side
# signals from whisper itself:
#   - ``no_speech_prob`` >= 0.6: whisper's own posterior that the segment
#     was non-speech. Whisper's internal silence skip uses this same
#     threshold, but only fires when avg_logprob is also < -1.0 — which
#     is why "Thank you." on silence (high logprob + high no_speech_prob)
#     leaks through the model and lands at our commit gate.
#   - ``avg_logprob`` < -1.0: low-confidence decode; treat with suspicion.
#   - very short audio (<= 1500 ms) when neither signal is reported:
#     a single stock phrase from a sub-second clip is almost always
#     a hallucinated transient, not a deliberate utterance.
#
# Real "thank you" speech has high audio energy and produces
# ``no_speech_prob`` well below 0.6 with ``avg_logprob`` above -0.5,
# so the AND of (phrase match) ∧ (audio-side suspicion) does not trip on
# the legitimate case.

#: Whisper-on-silence stock phrases. Stored lowercased and
#: punctuation-stripped; we normalize the input the same way before
#: comparing. Add new entries here as they appear in support bundles.
_SILENCE_HALLUCINATION_PHRASES: frozenset[str] = frozenset(
    {
        # English
        "thank you",
        "thanks",
        "thanks for watching",
        "thank you for watching",
        "thank you so much",
        "thank you very much",
        "thanks for listening",
        "thanks so much",
        "please subscribe",
        "subscribe",
        "you",
        "bye",
        "goodbye",
        "okay",
        "ok",
        "uh",
        "um",
        "hmm",
        "",  # bare punctuation collapses to empty after stripping
        # Auto-translated stock phrases that whisper occasionally emits
        # on silence when language detection drifts. Kept narrow so we
        # don't accidentally block real two-word utterances.
        "gracias",
        "merci",
        "danke",
        "ありがとう",
        "ありがとうございました",
        "감사합니다",
        "謝謝",
        "谢谢",
    }
)

#: Default ``no_speech_prob`` threshold above which a short stock phrase
#: is treated as a hallucination. Matches whisper's own internal
#: ``no_speech_threshold`` so we trust the model's own silence posterior.
SILENCE_NO_SPEECH_THRESHOLD = 0.6

#: Default ``avg_logprob`` threshold below which a short stock phrase is
#: treated as a hallucination. Matches whisper's internal
#: ``logprob_threshold``.
SILENCE_AVG_LOGPROB_THRESHOLD = -1.0

#: When neither audio-side signal is reported, fall back to duration:
#: a stock phrase from <= this many ms of audio is treated as silence
#: regardless of whisper's confidence (the hallucination is the only
#: shape that fits in that little time without real speech to anchor).
SILENCE_FALLBACK_AUDIO_MS = 1500.0

# Pre-compiled stripper used to normalize candidate text before phrase
# comparison. Kept module-level so the per-utterance call is cheap.
_PUNCTUATION_STRIP_RE = re.compile(r"[\s\.\!\?,;:'\"\-–—…]+")


def _normalize_for_phrase_match(text: str) -> str:
    """Lowercase, collapse runs of whitespace+punctuation to single spaces,
    then strip leading/trailing whitespace."""
    if not text:
        return ""
    collapsed = _PUNCTUATION_STRIP_RE.sub(" ", text.lower())
    return collapsed.strip()


def looks_like_silence_hallucination(
    text: str,
    *,
    no_speech_prob: float | None = None,
    avg_logprob: float | None = None,
    audio_duration_ms: float | None = None,
) -> bool:
    """Return True if *text* looks like a whisper-on-silence stock phrase.

    Conservative two-factor check: the text must match a known stock
    phrase AND at least one audio-side signal must corroborate that the
    audio was silence. Real speech that happens to say "thank you" has
    low ``no_speech_prob`` and high ``avg_logprob`` — both fail the
    corroboration step — so the legitimate case is preserved.

    Returns False on any input that doesn't match a stock phrase, so this
    function is safe to chain before the broader structural guard in
    :func:`looks_like_hallucination`.
    """
    if not text:
        return False
    normalized = _normalize_for_phrase_match(text)
    if normalized not in _SILENCE_HALLUCINATION_PHRASES:
        return False
    # Punctuation-only outputs ("!", ".", "...", "!?", etc.) normalize to the
    # empty string. A user cannot intentionally dictate just punctuation — the
    # input is always whisper hallucinating on silence or near-silence with no
    # detectable speech. Return True without requiring audio-side corroboration,
    # because the corroboration thresholds can fall through when whisper is
    # "confident" in its own hallucination (e.g. no_speech_prob ~ 0.4 and
    # avg_logprob ~ -0.6 on a near-silent 6s clip).
    if not normalized:
        return True
    # Phrase match (non-empty) — now require audio-side corroboration so a
    # real spoken "thank you" with confident audio signals is preserved.
    if no_speech_prob is not None and no_speech_prob >= SILENCE_NO_SPEECH_THRESHOLD:
        return True
    if avg_logprob is not None and avg_logprob < SILENCE_AVG_LOGPROB_THRESHOLD:
        return True
    # Neither model-side signal available — fall back to duration. Stock
    # phrases from sub-second audio are virtually always hallucinations.
    if (
        no_speech_prob is None
        and avg_logprob is None
        and audio_duration_ms is not None
        and audio_duration_ms <= SILENCE_FALLBACK_AUDIO_MS
    ):
        return True
    return False


#: Minimum word count we require to survive after a trailing-segment strip
#: before we trust the stripped result. Below this, we conservatively leave
#: the original text alone — the cost of cutting a real short utterance
#: ("Hi. Thank you.") is higher than letting one hallucinated tail through,
#: since the existing whole-utterance silence guard still backs us up for
#: pure stock-phrase utterances. Matches the existing four-word floor used
#: by surrounding memory heuristics.
TRAILING_STRIP_MIN_WORDS = 4
_REPEATED_STOCK_TAIL_WORDS = frozenset({"okay", "ok", "um", "uh", "hmm"})
_REPEATED_STOCK_TAIL_MIN_REPEATS = 3
_REPEATED_STOCK_TAIL_MIN_PREFIX_WORDS = 8
_WORD_SPAN_RE = re.compile(r"\b[\w']+\b")
_LOW_SIGNAL_ADJACENT_DUPLICATE_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "but",
        "is",
        "it",
        "just",
        "like",
        "so",
        "that",
        "the",
        "was",
        "were",
    }
)
_LOW_SIGNAL_ADJACENT_DUPLICATE_RE = re.compile(
    r"\b(?P<word>a|an|and|are|but|is|it|just|like|so|that|the|was|were)\b"
    r"(?P<repeat>(?:[\s,]+(?P=word)\b)+)",
    re.IGNORECASE,
)


def strip_trailing_silence_hallucination(
    text: str,
    *,
    segments: object = (),
    audio_duration_ms: float | None = None,
    no_speech_threshold: float = SILENCE_NO_SPEECH_THRESHOLD,
    avg_logprob_threshold: float = SILENCE_AVG_LOGPROB_THRESHOLD,
    min_remaining_words: int = TRAILING_STRIP_MIN_WORDS,
) -> str:
    """Strip whisper-on-silence stock-phrase tails from the end of *text*
    using per-segment audio-side corroboration.

    The existing :func:`looks_like_silence_hallucination` only fires when
    the *whole* utterance is a stock phrase, so a mixed transcript like
    ``"I had a great meeting today. Thank you. Thank you."`` slips
    through. This walks *segments* (the per-segment metadata from the
    whisper backend) from the end backwards: each tail segment whose
    normalized text is in :data:`_SILENCE_HALLUCINATION_PHRASES` AND
    whose own ``no_speech_prob`` / ``avg_logprob`` shows audio-side
    silence is marked for stripping. Stops at the first segment that is
    NOT a corroborated hallucination tail.

    **Per-segment corroboration is the whole point.** A trailing "Thank
    you." with ``no_speech_prob >= 0.6`` is whisper hallucinating on the
    silent tail of an otherwise-real utterance and is safe to strip. A
    trailing "Thank you." with ``no_speech_prob ~ 0.1`` and
    ``avg_logprob ~ -0.3`` is a real speaker saying "thank you" — the
    segment audio signals say speech, so we leave it alone.

    Returns the original *text* unchanged if:
      - *segments* is empty, falsy, or not iterable;
      - no tail segment matched the corroboration check;
      - stripping all matched tails would leave fewer than
        *min_remaining_words* words (safety floor: a heavily-stripped
        result is more likely to be wrong than the original).

    Parameters
    ----------
    text:
        The full utterance text.
    segments:
        Iterable of per-segment dicts (or objects with the same
        attributes) carrying ``text``, ``no_speech_prob``,
        ``avg_logprob``, and optionally ``start_ms`` / ``end_ms``.
        Anything else is silently ignored — the caller must tolerate
        backends that don't surface per-segment metadata.
    """
    if not text:
        return text
    try:
        segments_seq = tuple(segments)  # type: ignore[arg-type]
    except TypeError:
        return text
    if not segments_seq:
        return text

    def _get(seg: object, key: str) -> object:
        if isinstance(seg, dict):
            return seg.get(key)
        return getattr(seg, key, None)

    # Walk from the end backwards collecting corroborated stock-phrase
    # tails. Stop at the first segment that's NOT a hallucination tail.
    tail_texts_to_strip: list[str] = []
    for seg in reversed(segments_seq):
        seg_text_raw = _get(seg, "text")
        if not isinstance(seg_text_raw, str):
            break
        seg_text = seg_text_raw.strip()
        if not seg_text:
            # Empty trailing segment: skip (treat as already-trimmed) but
            # don't count as a hallucinated tail.
            continue
        normalized = _normalize_for_phrase_match(seg_text)
        if normalized not in _SILENCE_HALLUCINATION_PHRASES:
            break  # first real-content segment; stop walking
        # Phrase matched. Require audio-side corroboration from THIS
        # segment's own signals — the whole point is that we don't
        # cut a real "Thank you." that the speaker actually said.
        seg_no_speech_raw = _get(seg, "no_speech_prob")
        seg_avg_logprob_raw = _get(seg, "avg_logprob")
        seg_start_raw = _get(seg, "start_ms")
        seg_end_raw = _get(seg, "end_ms")
        try:
            seg_no_speech = (
                float(seg_no_speech_raw) if seg_no_speech_raw is not None else None
            )
        except (TypeError, ValueError):
            seg_no_speech = None
        try:
            seg_avg_logprob = (
                float(seg_avg_logprob_raw) if seg_avg_logprob_raw is not None else None
            )
        except (TypeError, ValueError):
            seg_avg_logprob = None
        try:
            seg_start_ms = float(seg_start_raw) if seg_start_raw is not None else None
        except (TypeError, ValueError):
            seg_start_ms = None
        try:
            seg_end_ms = float(seg_end_raw) if seg_end_raw is not None else None
        except (TypeError, ValueError):
            seg_end_ms = None
        corroborated = False
        if seg_no_speech is not None and seg_no_speech >= no_speech_threshold:
            corroborated = True
        elif seg_avg_logprob is not None and seg_avg_logprob < avg_logprob_threshold:
            corroborated = True
        elif _stock_tail_has_impossible_timing(
            audio_duration_ms=audio_duration_ms,
            start_ms=seg_start_ms,
            end_ms=seg_end_ms,
        ):
            corroborated = True
        if not corroborated:
            # Stock phrase but audio signals say speech — preserve it.
            break
        tail_texts_to_strip.append(seg_text)

    if not tail_texts_to_strip:
        return text

    # Reconstruct cleaned text by stripping each tail substring from the
    # end of *text* in order (most-recent tail first, which matches the
    # reverse-walk above). We strip in-place rather than rejoining the
    # surviving segments because:
    #   (a) the original text has gone through ITN, repetition collapse,
    #       and language normalization since segments were captured,
    #       so segment text is no longer a 1:1 substring slice of the
    #       final text in every case;
    #   (b) when it IS a substring, end-anchored strip is the safest
    #       op — we only ever cut from the tail.
    cleaned = text
    for tail in tail_texts_to_strip:
        # Try the literal segment text first; whisper preserves
        # punctuation in segments so most cases hit this branch.
        stripped_tail = cleaned.rstrip()
        if stripped_tail.endswith(tail):
            cleaned = stripped_tail[: -len(tail)]
            continue
        # Fall back to case-insensitive end-anchored match (segment
        # casing can drift after language normalization).
        if stripped_tail.lower().endswith(tail.lower()):
            cleaned = stripped_tail[: -len(tail)]
            continue
        # Couldn't locate this tail substring in the running cleaned
        # text — bail out rather than risk cutting the wrong characters.
        # The whole-utterance silence guard still backs us up downstream.
        return text

    cleaned = cleaned.rstrip(" \t\n")
    # Trim trailing punctuation/whitespace left over after stripping the
    # tail (e.g. cleaned text often ends with "...meeting today. " before
    # the period that separated it from the stripped tail; collapse it
    # back to "...meeting today.").
    cleaned = re.sub(r"[\s]+$", "", cleaned)

    if not cleaned:
        return text

    # Safety floor: refuse to strip if the remainder is too short to be
    # confident — a heavily-stripped result is more likely to be wrong
    # than the original. Below this threshold, leave the original alone
    # and let the whole-utterance silence guard handle it if it applies.
    remaining_words = re.findall(r"\b\w+\b", cleaned)
    if len(remaining_words) < min_remaining_words:
        return text

    return cleaned


def strip_repeated_stock_hallucination_tail(
    text: str,
    *,
    min_repeats: int = _REPEATED_STOCK_TAIL_MIN_REPEATS,
    min_prefix_words: int = _REPEATED_STOCK_TAIL_MIN_PREFIX_WORDS,
) -> str:
    """Strip repeated stock-silence words after substantive dictated text.

    Some final decodes append tails like ``"Okay, Okay Okay Okay"`` inside the
    same final result, without a separate segment whose no-speech metadata can
    corroborate the stock-phrase guard above. Keep this deliberately narrow:
    only a repeated stock token, only after enough prior words, and only when
    the suffix starts after a sentence/newline boundary.
    """
    if not text or not text.strip():
        return text
    matches = list(_WORD_SPAN_RE.finditer(text))
    if len(matches) < min_prefix_words + min_repeats:
        return text

    def normalized(match: re.Match[str]) -> str:
        return re.sub(r"[^a-z0-9]+", "", match.group(0).casefold())

    tail_token = normalized(matches[-1])
    if tail_token not in _REPEATED_STOCK_TAIL_WORDS:
        return text

    start = len(matches) - 1
    while start > 0 and normalized(matches[start - 1]) == tail_token:
        start -= 1
    repeats = len(matches) - start
    if repeats < min_repeats or start < min_prefix_words:
        return text

    separator = text[matches[start - 1].end() : matches[start].start()]
    if not any(ch in separator for ch in ".!?\n"):
        return text

    cleaned = text[: matches[start].start()].rstrip()
    cleaned = cleaned.rstrip(" ,;:")
    return cleaned or text


def strip_adjacent_low_signal_word_duplicates(text: str) -> str:
    """Collapse adjacent duplicate filler/connective words in final text.

    This is intentionally narrower than a general repetition cleaner. It only
    targets low-information glue words that ASR/writer cleanup can accidentally
    double ("just just", "that that") and leaves meaningful repetition such as
    names, numbers, commands, "very very", and code-like tokens alone.
    """
    if not text or not text.strip():
        return text

    def repl(match: re.Match[str]) -> str:
        word = match.group("word")
        normalized = re.sub(r"[^a-z0-9']+", "", word.casefold())
        if normalized not in _LOW_SIGNAL_ADJACENT_DUPLICATE_WORDS:
            return match.group(0)
        return word

    return _LOW_SIGNAL_ADJACENT_DUPLICATE_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Low-yield garbage guard
# ---------------------------------------------------------------------------
#
# Production 2026-06-11: 12.5 s of (mostly silent) audio decoded online to
# "Lamb &" with avg_logprob -2.11 — and pasted, replacing a better live
# preview ("I mean…"). The structural checks above can't catch it: one word,
# no loop, not a stock phrase. The signature IS the mismatch: a long buffer
# whose decode is both catastrophically low-confidence AND nearly empty.
# Real speech in a long buffer yields words; a real short utterance inside a
# long buffer decodes its few words confidently (whisper skips the silence).

#: avg_logprob at or below which a near-empty decode of long audio is junk.
#: Clean speech sits at -0.3…-0.5; even quiet-mic real speech stays above
#: -1.0 (see HALLUCINATION_CONFIDENCE_FLOOR). -1.4 leaves a wide margin.
LOW_YIELD_CONFIDENCE_FLOOR = -1.4

#: Only long buffers qualify — a short clip legitimately yields few words.
LOW_YIELD_MIN_AUDIO_MS = 4000.0

#: "Nearly empty" decode: at most this many words from the whole buffer.
LOW_YIELD_MAX_WORDS = 4


def looks_like_low_yield_garbage(
    text: str,
    *,
    confidence: float | None,
    audio_duration_ms: float | None,
) -> bool:
    """Return True for a near-empty, catastrophically low-confidence decode
    of a long audio buffer — whisper salvaging noise, not transcribing."""
    if not text:
        return False
    if confidence is None or confidence > LOW_YIELD_CONFIDENCE_FLOOR:
        return False
    if audio_duration_ms is None or audio_duration_ms < LOW_YIELD_MIN_AUDIO_MS:
        return False
    words = re.findall(r"\b\w+\b", text)
    return len(words) <= LOW_YIELD_MAX_WORDS


_PROMPT_ECHO_LEADING_RE = re.compile(
    r"^\s*(?:(?:app|category|title|vocabulary|prefer\s+exact\s+forms)\b\s*:"
    r"[^|\"\n]*(?:\|\s*|(?=\")|\s*$))+",
    re.IGNORECASE,
)


def strip_leading_prompt_echo(text: str) -> tuple[str, str | None]:
    """Strip a leading echo of Juno's structured Whisper initial prompt."""
    if not text:
        return text, None
    cleaned = _PROMPT_ECHO_LEADING_RE.sub("", text, count=1)
    if cleaned == text:
        return text, None
    removed = text[: len(text) - len(cleaned)].strip()
    cleaned = cleaned.strip()
    if (
        len(cleaned) >= 2
        and cleaned[0] in "\"“"
        and cleaned[-1] in "\"”"
        and cleaned.count("\"") + cleaned.count("“") + cleaned.count("”") <= 2
    ):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return text, None
    return cleaned, removed or "prompt_echo"


def _stock_tail_has_impossible_timing(
    *,
    audio_duration_ms: float | None,
    start_ms: float | None,
    end_ms: float | None,
) -> bool:
    if audio_duration_ms is None or start_ms is None or end_ms is None:
        return False
    if audio_duration_ms <= 0 or end_ms <= start_ms:
        return False
    duration_ms = end_ms - start_ms
    overrun_ms = end_ms - audio_duration_ms
    if overrun_ms >= 500:
        return True
    return duration_ms >= 8000 and start_ms >= max(0.0, audio_duration_ms - 6000)


__all__ = [
    "HALLUCINATION_CONFIDENCE_FLOOR",
    "LOW_YIELD_CONFIDENCE_FLOOR",
    "LOW_YIELD_MAX_WORDS",
    "LOW_YIELD_MIN_AUDIO_MS",
    "SILENCE_AVG_LOGPROB_THRESHOLD",
    "SILENCE_FALLBACK_AUDIO_MS",
    "SILENCE_NO_SPEECH_THRESHOLD",
    "TRAILING_STRIP_MIN_WORDS",
    "looks_like_hallucination",
    "looks_like_low_yield_garbage",
    "looks_like_silence_hallucination",
    "strip_leading_prompt_echo",
    "strip_adjacent_low_signal_word_duplicates",
    "strip_repeated_stock_hallucination_tail",
    "strip_trailing_silence_hallucination",
]
