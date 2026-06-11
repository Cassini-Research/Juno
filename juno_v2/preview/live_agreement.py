"""LocalAgreement-2 hypothesis buffer for Whisper streaming.

This is the algorithmic heart of the Juno Whisper-driven live HUD. It implements
the LocalAgreement-2 commit policy from:

    Macháček, Dabre, Bojar. "Turning Whisper into Real-Time Transcription System"
    (arXiv:2307.14743, EACL 2024 demo). MIT-licensed reference implementation:
    https://github.com/ufal/whisper_streaming  ``HypothesisBuffer.flush``

The port is functional, not structural — we keep absolute-timestamp ``Word``
records, push hypothesis lists through ``insert`` / ``flush``, and let the
caller own audio buffer trimming. We do NOT depend on mlx-whisper here so the
algorithm can be unit-tested without an Apple Silicon runtime.

Invariants the algorithm itself guarantees:

- Two consecutive hypotheses must agree on the next N words before any commit.
- ``tail`` is the remainder of the most recent hypothesis past the agreement
  point. It is render-only and intentionally allowed to flicker.
- The caller renders ``committed`` opaque and ``tail`` dimmed; final-stage
  replacement happens through the transcript patch path, not by mutating this
  buffer.

Important — ``committed`` is append-mostly INSIDE this module: ``flush`` extends
it and may immediately remove a newly committed adjacent replay phrase that
straddles the old/new boundary. The streaming-core wrapper also uses the same
boundary scrub after final tail promotion, plus its documented rollback/strip
paths.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field


@dataclass(slots=True)
class Word:
    """One transcribed word with absolute timestamps (seconds since utterance start)."""

    start: float
    end: float
    text: str

    @property
    def normalized(self) -> str:
        return "".join(ch.lower() for ch in self.text if ch.isalnum())


@dataclass(slots=True)
class HypothesisBuffer:
    """LocalAgreement-2 prefix-stability buffer.

    Usage per cadence tick::

        buf.insert(new_words_from_whisper)
        newly_committed: list[Word] = buf.flush()
        # render buf.committed opaque, buf.tail dimmed

    ``insert`` does dedupe + window-shift accounting.
    ``flush`` does the LocalAgreement-2 prefix match and moves agreed words into
    ``committed``. The remainder of the current hypothesis becomes the new tail.
    """

    committed: list[Word] = field(default_factory=list)
    tail: list[Word] = field(default_factory=list)
    last_committed_time: float = 0.0
    _staged_new: list[Word] = field(default_factory=list, repr=False)
    # Set by ``flush`` when the agreement prefix was found to be a replayed
    # committed phrase and dropped at commit time. The streaming-core wrapper
    # reads this for telemetry, then it's reset on the next ``insert``.
    last_flush_replay_drop_reason: str | None = field(default=None, repr=False)

    def insert(self, new_words: list[Word]) -> str | None:
        """Stage the new hypothesis and remove overlap with committed text.

        The live MLX path uses synthetic word timestamps for speed, so timestamps
        are not trustworthy enough to be the primary boundary detector. The
        durable signal is text agreement. We therefore anchor by normalized word
        overlap first, and use timestamps only as a fallback when no text anchor
        is available.

        Returns a telemetry reason when a replayed committed prefix was dropped.
        """
        words = [w for w in new_words if w.normalized]
        replay_drop_reason: str | None = None
        self.last_flush_replay_drop_reason = None

        if self.committed and words:
            words = self._drop_committed_overlap(words)
            words, replay_drop_reason = self._drop_committed_replay_prefix(words)

        self._staged_new = words
        return replay_drop_reason

    def _drop_committed_overlap(self, words: list[Word]) -> list[Word]:
        committed_norm = [w.normalized for w in self.committed if w.normalized]
        word_norm = [w.normalized for w in words if w.normalized]
        if not committed_norm or not word_norm:
            return words

        # Common rolling-window shape: the decoder re-emits the committed tail
        # at the beginning of its new hypothesis.
        max_anchor = min(12, len(committed_norm), len(word_norm))
        for n in range(max_anchor, 0, -1):
            if committed_norm[-n:] == word_norm[:n]:
                return words[n:]

        # Full-buffer shape: the decoder starts from earlier in the utterance,
        # so the committed tail appears inside the new hypothesis rather than at
        # its first word. Drop everything through that anchor.
        for n in range(max_anchor, 1, -1):
            anchor = committed_norm[-n:]
            for pos in range(0, len(word_norm) - n + 1):
                if word_norm[pos : pos + n] == anchor:
                    return words[pos + n :]

        # Deliberately no synthetic-timestamp fallback. Whisper word timestamps
        # in the streaming path are linearly interpolated from segment spans,
        # so ``last_committed_time``-based pruning would fire in arbitrary
        # directions after a buffer trim. The replay case this used to cover
        # is now handled by ``_drop_committed_replay_prefix`` below, which
        # operates on text n-grams and is robust under synthetic timestamps.
        return words

    def _drop_committed_replay_prefix(self, words: list[Word]) -> tuple[list[Word], str | None]:
        """Drop a staged prefix that is just an earlier committed phrase replay.

        MLX Whisper sometimes emits the prompt/history as if it were fresh audio
        after the buffer has been trimmed. LocalAgreement-2 then sees the same
        replay twice and would commit it as new text. We only fire on prefixes
        of ≥5 words AND ≥20 characters (see ``_find_replayed_prefix_len``), so
        normal short repetition like "very very" or "I want to" is preserved.
        """
        committed_norm = [w.normalized for w in self.committed if w.normalized]
        word_norm = [w.normalized for w in words if w.normalized]
        if len(committed_norm) < 5 or len(word_norm) < 5:
            match = _find_adjacent_duplicate_boundary(committed_norm, word_norm)
            if match is None:
                match = _find_short_reanchored_prefix(committed_norm, word_norm)
            if match is None:
                return words, None
            drop_len, reason = match
            return words[drop_len:], f"committed_{reason}"

        match = _find_adjacent_duplicate_boundary(committed_norm, word_norm)
        if match is not None:
            drop_len, reason = match
            return words[drop_len:], f"committed_{reason}"

        short_reanchor = _find_short_reanchored_prefix(committed_norm, word_norm)
        if short_reanchor is not None:
            drop_len, reason = short_reanchor
            return words[drop_len:], f"committed_{reason}"

        replay_len = _find_replayed_prefix_len(committed_norm, word_norm)
        if replay_len:
            return words[replay_len:], f"committed_replay_prefix_{replay_len}"
        return words, None

    def flush(self) -> list[Word]:
        """Run LocalAgreement-2 against the staged hypothesis. Returns newly
        committed words; updates ``tail`` to the unstable remainder.

        Returns the words that were newly committed THIS tick (empty list if
        no new agreement). Caller can use these to drive incremental UI
        animations, but the canonical source of truth is ``self.committed``.
        """
        agreement: list[Word] = []
        i = 0
        while i < len(self.tail) and i < len(self._staged_new):
            if self.tail[i].normalized and self.tail[i].normalized == self._staged_new[i].normalized:
                # Take the timestamps from the NEW hypothesis; they reflect
                # the model's most-recent re-estimation of the word boundaries.
                agreement.append(self._staged_new[i])
                i += 1
            else:
                break

        if agreement:
            agreement, boundary_reason = self.resolve_boundary_revision_before_append(agreement)
            if boundary_reason is not None:
                self.last_flush_replay_drop_reason = boundary_reason

        if agreement:
            agreement, replay_reason = self._drop_replayed_agreement_prefix(agreement)
            if replay_reason is not None:
                self.last_flush_replay_drop_reason = replay_reason

        if agreement:
            old_len = len(self.committed)
            self.committed.extend(agreement)
            agreement, duplicate_reason = self._drop_adjacent_duplicate_boundary(old_len, agreement)
            if duplicate_reason is not None:
                self.last_flush_replay_drop_reason = duplicate_reason
            if agreement:
                self.last_committed_time = agreement[-1].end

        self.tail = self._staged_new[i:]
        self._staged_new = []
        return agreement

    def _drop_replayed_agreement_prefix(self, agreement: list[Word]) -> tuple[list[Word], str | None]:
        """Remove agreement words that are just an older committed phrase.

        ``insert`` removes replay from each new hypothesis, but a replay can sit
        in ``tail`` first and become "stable" after the next matching decode.
        This guard applies the same long-prefix replay rule at commit time so
        a duplicated old phrase never graduates from dim tail to opaque text.

        Returns ``(agreement, reason_or_None)``. Reason is set when a replayed
        prefix was dropped — the streaming-core wrapper logs it as a telemetry
        counter so we can spot the matcher misfiring in production.
        """
        committed_norm = [w.normalized for w in self.committed if w.normalized]
        agreement_norm = [w.normalized for w in agreement if w.normalized]
        if len(committed_norm) < 5 or len(agreement_norm) < 5:
            return agreement, None

        replay_len = _find_replayed_prefix_len(committed_norm, agreement_norm)
        if replay_len:
            return agreement[replay_len:], f"agreement_replay_prefix_{replay_len}"
        return agreement, None

    def drop_adjacent_duplicate_boundary_after_append(
        self,
        old_len: int,
        appended: list[Word],
        *,
        allow_committed_suffix_drop: bool = False,
    ) -> tuple[list[Word], str | None]:
        return self._drop_adjacent_duplicate_boundary(
            old_len,
            appended,
            allow_committed_suffix_drop=allow_committed_suffix_drop,
        )

    def resolve_boundary_revision_before_append(
        self,
        agreement: list[Word],
    ) -> tuple[list[Word], str | None]:
        """Resolve agreement words that revise or replay the committed boundary.

        LocalAgreement's append-only default fails when Whisper refines the last
        committed token in the next stable hypothesis (``format`` →
        ``formatting``) or replays it with an article (``judge`` → ``the
        judge``). Handle those at the boundary before they become opaque text.
        """

        if not self.committed or not agreement:
            return agreement, None
        committed_norm = [w.normalized for w in self.committed if w.normalized]
        agreement_norm = [w.normalized for w in agreement if w.normalized]
        if len(committed_norm) != len(self.committed) or len(agreement_norm) != len(agreement):
            return agreement, None

        article_drop = _find_agreement_article_bridge_replay(committed_norm, agreement_norm)
        if article_drop:
            return agreement[article_drop:], "agreement_boundary_article_replay"

        if _same_stem_repetition(committed_norm[-1], agreement_norm[0]):
            self.committed[-1] = agreement[0]
            self._refresh_last_committed_time()
            return agreement[1:], "agreement_boundary_stem_revision"

        return agreement, None

    def drop_repeated_tail_suffix(self) -> str | None:
        """Drop a repeated phrase whose second copy is fully inside tail.

        Boundary replay can first show up as dim text instead of committed text.
        Example: committed ends with "the final word should", tail becomes
        "show up the final word should show up". The first complete phrase spans
        committed+tail; the second copy is entirely tail and should not render.
        """

        committed_norm = [w.normalized for w in self.committed if w.normalized]
        tail_norm = [w.normalized for w in self.tail if w.normalized]
        if len(committed_norm) != len(self.committed) or len(tail_norm) != len(self.tail):
            return None
        if not committed_norm or len(tail_norm) < 4:
            return None
        boundary_reason = self._drop_tail_prefix_repeating_committed_suffix(
            committed_norm,
            tail_norm,
        )
        if boundary_reason is not None:
            return boundary_reason
        recent_replay = _find_tail_prefix_repeating_recent_committed_phrase(
            committed_norm,
            tail_norm,
        )
        if recent_replay is not None:
            drop_len, reason = recent_replay
            del self.tail[:drop_len]
            self.last_flush_replay_drop_reason = reason
            return reason
        boundary = len(committed_norm)
        combined = committed_norm + tail_norm
        max_width = min(12, len(combined) // 2)
        for width in range(max_width, 3, -1):
            for start in range(0, len(combined) - (2 * width) + 1):
                second_start = start + width
                second_end = second_start + width
                if second_start < boundary:
                    continue
                left = combined[start:second_start]
                right = combined[second_start:second_end]
                if left != right or not _meaningful_replay_phrase(left):
                    continue
                del self.tail[second_start - boundary: second_end - boundary]
                reason = f"tail_contiguous_repeat_{width}"
                self.last_flush_replay_drop_reason = reason
                return reason
        return None

    def _drop_tail_prefix_repeating_committed_suffix(
        self,
        committed_norm: list[str],
        tail_norm: list[str],
    ) -> str | None:
        max_width = min(6, len(committed_norm), len(tail_norm))
        for width in range(max_width, 3, -1):
            left = committed_norm[-width:]
            right = tail_norm[:width]
            if not _meaningful_replay_phrase(left):
                continue
            if len("".join(left)) < 10:
                continue
            if left == right:
                del self.tail[:width]
                reason = f"tail_boundary_repeat_{width}"
                self.last_flush_replay_drop_reason = reason
                return reason
            if (
                left[:-1] == right[:-1]
                and right[-1].startswith(left[-1])
                and len(right[-1]) > len(left[-1])
            ):
                suffix = _word_suffix_after_normalized_prefix(self.tail[width - 1], left[-1])
                if suffix is None:
                    continue
                del self.tail[: width - 1]
                self.tail[0] = Word(
                    start=self.tail[0].start,
                    end=self.tail[0].end,
                    text=suffix,
                )
                reason = f"tail_boundary_repeat_{width}_suffix"
                self.last_flush_replay_drop_reason = reason
                return reason
        return None

    def _drop_adjacent_duplicate_boundary(
        self,
        old_len: int,
        agreement: list[Word],
        *,
        allow_committed_suffix_drop: bool = False,
    ) -> tuple[list[Word], str | None]:
        """Drop a newly committed phrase that repeats the committed tail.

        The long replay detector above catches old prompt/history chunks.
        Production replay runs also showed shorter adjacent repeats such as
        "ah today is may / ah today is may" and orthography drift such as
        "Project Atlas Travel Wrap up / Project Atlas Travel Wrap-up". This guard
        is narrower than lowering the global replay floor: it only fires when
        the duplicate phrase sits directly across the old/new commit boundary,
        so intentional later repetition remains untouched.
        """

        if old_len <= 0 or not agreement:
            return agreement, None
        committed_norm = [w.normalized for w in self.committed if w.normalized]
        if len(committed_norm) != len(self.committed):
            return agreement, None
        match = _find_adjacent_duplicate_boundary(
            committed_norm[:old_len],
            committed_norm[old_len:],
            allow_near_boundary=allow_committed_suffix_drop,
        )
        if match is None:
            return agreement, None
        drop_len, reason = match
        del self.committed[old_len: old_len + drop_len]
        stale_lag = _near_boundary_replay_lag(reason) if allow_committed_suffix_drop else 0
        if stale_lag and old_len >= stale_lag:
            del self.committed[old_len - stale_lag: old_len]
            self._refresh_last_committed_time()
        self.last_flush_replay_drop_reason = reason
        return agreement[drop_len:], reason

    def _refresh_last_committed_time(self) -> None:
        self.last_committed_time = self.committed[-1].end if self.committed else 0.0

    def committed_text(self) -> str:
        return " ".join(w.text for w in self.committed).strip()

    def tail_text(self) -> str:
        return " ".join(w.text for w in self.tail).strip()

    def reset(self) -> None:
        """Drop everything. Used when an utterance ends."""
        self.committed = []
        self.tail = []
        self.last_committed_time = 0.0
        self._staged_new = []


def absolute_words(
    word_dicts: list[dict] | tuple[dict, ...] | None,
    buffer_start_t: float,
) -> list[Word]:
    """Convert mlx-whisper word records (relative timestamps) to absolute Words.

    mlx-whisper returns each word as ``{"word": "Hello", "start": 0.12, "end": 0.55}``
    with start/end relative to the buffer fed to ``transcribe``. We shift by
    ``buffer_start_t`` so the LocalAgreement state machine sees a single
    monotone timeline across audio-buffer trims.

    Robust to missing keys: words without a usable text/start/end are skipped
    rather than crashing the decode loop.
    """
    if not word_dicts:
        return []
    out: list[Word] = []
    for item in word_dicts:
        if not isinstance(item, dict):
            continue
        text = str(item.get("word") or item.get("text") or "").strip()
        if not text:
            continue
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        try:
            start_f = float(start) + float(buffer_start_t)
            end_f = float(end) + float(buffer_start_t)
        except (TypeError, ValueError):
            continue
        if end_f < start_f:
            end_f = start_f
        out.append(Word(start=start_f, end=end_f, text=text))
    return out


def _find_replayed_prefix_len(committed_norm: list[str], candidate_norm: list[str]) -> int:
    """Return length of the candidate prefix that matches a window inside
    committed, or 0 if none.

    Two gates govern fire:

    - **Length floor**: ``n ≥ 5`` words AND prefix character-sum ≥ 20. Short
      phrasing repetition like "I want to" is 3 tokens / 7 chars and never
      passes this gate. A 5-word phrase averaging ≥4 chars/word (anything
      with real content) does.
    - **Match strength**: exact `window == prefix` OR fuzzy SequenceMatcher
      ratio ≥ 0.95. The fuzzy threshold was 0.90 prior to May 2026 — that
      was too loose for natural prose; 0.95 still catches all verbatim or
      one-token-drift prompt replays we see in production traces.

    Verified against the long golden HUD case: catches the "2026 I will
    speak fast now" replay loop (6 tokens, len_sum 23).
    """

    max_replay = min(20, len(candidate_norm), len(committed_norm))
    for n in range(max_replay, 4, -1):  # require prefix length ≥ 5
        prefix = candidate_norm[:n]
        if sum(len(token) for token in prefix) < 20:
            continue
        joined_prefix = " ".join(prefix)
        for pos in range(0, len(committed_norm) - n + 1):
            window = committed_norm[pos: pos + n]
            if window == prefix:
                return n
            if prefix[0] != window[0]:
                continue
            joined_window = " ".join(window)
            if difflib.SequenceMatcher(a=joined_window, b=joined_prefix, autojunk=False).ratio() >= 0.95:
                return n
    return 0


def _find_adjacent_duplicate_boundary(
    committed_norm: list[str],
    candidate_norm: list[str],
    *,
    allow_near_boundary: bool = False,
) -> tuple[int, str] | None:
    """Return candidate-prefix length to drop when it repeats committed suffix.

    This is intentionally boundary-only. It catches re-decoded carry audio that
    arrives as the first words of the next hypothesis, including orthographic
    tokenization drift such as ``wrap up`` versus ``wrap-up``. It does not scan
    arbitrary later text, because natural repeated phrasing is otherwise too easy
    to delete.
    """

    if len(committed_norm) < 3 or len(candidate_norm) < 3:
        return None
    max_left = min(6, len(committed_norm))
    max_right = min(6, len(candidate_norm))
    for left_width in range(max_left, 2, -1):
        left = committed_norm[-left_width:]
        if not _meaningful_replay_phrase(left):
            continue
        left_joined = "".join(left)
        if len(left_joined) < 10:
            continue
        for right_width in range(max_right, 2, -1):
            right = candidate_norm[:right_width]
            exact_tokens = left_width == right_width and left == right
            same_compound = left_joined == "".join(right)
            if exact_tokens:
                return right_width, f"adjacent_duplicate_phrase_{right_width}"
            if same_compound:
                return right_width, f"adjacent_duplicate_phrase_{left_width}x{right_width}"
            if _near_adjacent_revision_phrase(left, right):
                return right_width, f"adjacent_revision_phrase_{left_width}x{right_width}"
    if allow_near_boundary:
        near_boundary = _find_near_boundary_replay_prefix(committed_norm, candidate_norm)
        if near_boundary is not None:
            return near_boundary
    return None


def _find_near_boundary_replay_prefix(
    committed_norm: list[str],
    candidate_norm: list[str],
) -> tuple[int, str] | None:
    """Drop a final-tail prefix replayed just before the committed boundary.

    This is intentionally available only to true utterance-final tail promotion.
    Live commits must not remove committed words, but final drain may need to
    correct a one-word stale carry such as ``... text our`` followed by
    ``sentence readable and text awards``.
    """

    if len(committed_norm) < 4 or len(candidate_norm) < 4:
        return None
    max_lag = 3
    max_width = min(6, len(candidate_norm) - 1, len(committed_norm) - 1)
    for width in range(max_width, 2, -1):
        prefix = candidate_norm[:width]
        extended_prefix = candidate_norm[: min(len(candidate_norm), width + 2)]
        extended_meaningful = _meaningful_replay_phrase(extended_prefix)
        if not _meaningful_replay_phrase(prefix) and not (width >= 3 and extended_meaningful):
            continue
        min_prefix_chars = 8 if extended_meaningful else 10
        if len("".join(prefix)) < min_prefix_chars:
            continue
        search_start = max(0, len(committed_norm) - width - max_lag)
        search = committed_norm[search_start:]
        for pos in range(len(search) - width - 1, -1, -1):
            lag = len(search) - (pos + width)
            if lag < 1 or lag > max_lag:
                continue
            window = search[pos: pos + width]
            if window == prefix:
                return width, f"near_boundary_replay_phrase_{width}_lag{lag}"
            if "".join(window) == "".join(prefix):
                return width, f"near_boundary_replay_phrase_{width}_lag{lag}_compound"
            if _near_adjacent_revision_phrase(window, prefix):
                return width, f"near_boundary_revision_phrase_{width}_lag{lag}"
    return None


def _find_tail_prefix_repeating_recent_committed_phrase(
    committed_norm: list[str],
    tail_norm: list[str],
) -> tuple[int, str] | None:
    """Drop an unstable tail prefix that replays the latest committed phrase.

    Whisper's rolling preview window can re-anchor at a sentence boundary and
    emit a committed phrase again with a small opener/tokenization drift, e.g.
    ``... off-callback flow`` followed by tail ``to review the off call back
    flow find edge cases``. The existing adjacent-boundary guard misses that
    because the replay starts a few words before the committed suffix. Keep this
    recent and prefix-only so intentional later repetition remains visible.
    """

    if len(committed_norm) < 4 or len(tail_norm) < 4:
        return None
    search = committed_norm[-14:]
    max_tail_width = min(9, len(tail_norm))
    for tail_width in range(max_tail_width, 2, -1):
        prefix = tail_norm[:tail_width]
        if not _meaningful_replay_phrase(prefix):
            continue
        if len("".join(prefix)) < 10:
            continue
        max_window_width = min(8, len(search))
        for window_width in range(max_window_width, 2, -1):
            for start in range(len(search) - window_width, -1, -1):
                lag = len(search) - (start + window_width)
                if lag > 2:
                    continue
                window = search[start : start + window_width]
                if _phrase_compound_or_near_equal(window, prefix):
                    return (
                        tail_width,
                        f"tail_recent_replay_phrase_{window_width}x{tail_width}_lag{lag}",
                    )
    return None


def _find_agreement_article_bridge_replay(
    committed_norm: list[str],
    agreement_norm: list[str],
) -> int:
    if not committed_norm or len(agreement_norm) < 2:
        return 0
    first = committed_norm[-1]
    if len(first) < 4 and not any(ch.isdigit() for ch in first):
        return 0
    if agreement_norm[0] not in {"a", "an", "the"}:
        return 0
    if agreement_norm[1] != first:
        return 0
    return 2


def _same_stem_repetition(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) < 4:
        return False
    if not longer.startswith(shorter):
        return False
    suffix = longer[len(shorter):]
    if not suffix or len(suffix) > 5:
        return False
    return suffix in {
        "s",
        "es",
        "ed",
        "d",
        "ing",
        "ting",
        "tion",
        "ions",
        "er",
        "ers",
        "ly",
    }


_SHORT_REANCHOR_TOKENS = {
    "ah",
    "hmm",
    "mhm",
    "mm",
    "now",
    "ok",
    "okay",
    "uh",
    "um",
    "yeah",
    "yep",
}

_SHORT_REANCHOR_BRIDGE_WORDS = {
    "and",
    "but",
    "like",
    "no",
    "now",
    "okay",
    "ok",
    "so",
    "then",
    "well",
}


def _find_short_reanchored_prefix(
    committed_norm: list[str],
    candidate_norm: list[str],
) -> tuple[int, str] | None:
    """Drop a one-token filler/opening prefix replayed across a trim boundary.

    This catches the production shape ``Okay now`` followed by ``Okay I'm...``.
    Long replay matching deliberately ignores this because a single token is too
    weak globally; here it is allowed only at the immediate boundary, only for
    discourse markers, and only when the intervening committed words are bridge
    words rather than content.
    """

    if len(committed_norm) < 2 or len(candidate_norm) < 3:
        return None
    first = candidate_norm[0]
    if first not in _SHORT_REANCHOR_TOKENS:
        return None
    max_lag = min(2, len(committed_norm) - 1)
    for lag in range(0, max_lag + 1):
        anchor_index = len(committed_norm) - 1 - lag
        if anchor_index < 0 or committed_norm[anchor_index] != first:
            continue
        bridge = committed_norm[anchor_index + 1 :]
        if any(token not in _SHORT_REANCHOR_BRIDGE_WORDS for token in bridge):
            continue
        if lag == 0:
            return 1, "short_boundary_reanchor_1"
        return 1, f"short_near_boundary_reanchor_1_lag{lag}"
    return None


def _near_boundary_replay_lag(reason: str) -> int:
    if "near_boundary_" not in reason:
        return 0
    marker = "_lag"
    pos = reason.rfind(marker)
    if pos < 0:
        return 0
    start = pos + len(marker)
    end = start
    while end < len(reason) and reason[end].isdigit():
        end += 1
    if end == start:
        return 0
    try:
        return int(reason[start:end])
    except ValueError:
        return 0


_BOUNDARY_REVISION_WEAK_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "in",
    "is",
    "it",
    "keep",
    "of",
    "on",
    "or",
    "please",
    "that",
    "the",
    "then",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "with",
}


def _near_adjacent_revision_phrase(left: list[str], right: list[str]) -> bool:
    """Detect a short ASR boundary revision without rewriting committed text.

    This is a one-way duplicate guard: it may drop the candidate prefix that
    repeats the already committed boundary, but it never removes committed
    words. That preserves HUD stability while avoiding visible phrases such as
    "please keep the sentence please keep this sentence".
    """

    if len(left) != len(right) or len(left) < 3 or len(left) > 6:
        return False
    if left[0] != right[0]:
        return False
    if not _meaningful_replay_phrase(left) and not _meaningful_replay_phrase(right):
        return False

    mismatches = [(a, b) for a, b in zip(left, right, strict=True) if a != b]
    if len(mismatches) != 1:
        return False
    if len(left) - len(mismatches) < 2:
        return False

    mismatch_left, mismatch_right = mismatches[0]
    weak_mismatch = (
        mismatch_left in _BOUNDARY_REVISION_WEAK_WORDS
        or mismatch_right in _BOUNDARY_REVISION_WEAK_WORDS
    )
    if not weak_mismatch:
        token_ratio = difflib.SequenceMatcher(
            a=mismatch_left,
            b=mismatch_right,
            autojunk=False,
        ).ratio()
        if token_ratio < 0.72:
            return False

    phrase_ratio = difflib.SequenceMatcher(
        a=" ".join(left),
        b=" ".join(right),
        autojunk=False,
    ).ratio()
    return phrase_ratio >= 0.82


def _phrase_compound_or_near_equal(left: list[str], right: list[str]) -> bool:
    if len(left) < 3 or len(right) < 3:
        return False
    left_joined = "".join(left)
    right_joined = "".join(right)
    if left_joined == right_joined:
        return True
    if not left_joined or not right_joined:
        return False
    length_ratio = min(len(left_joined), len(right_joined)) / max(len(left_joined), len(right_joined))
    if length_ratio < 0.9:
        return False
    if left[0] != right[0] or left[-1] != right[-1]:
        return False
    return difflib.SequenceMatcher(a=left_joined, b=right_joined, autojunk=False).ratio() >= 0.94


def _meaningful_replay_phrase(tokens: list[str]) -> bool:
    filler_starters = {"ah", "um", "uh", "erm", "hmm"}
    date_words = {
        "today", "tomorrow", "yesterday", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday", "january", "february",
        "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december",
    }
    weak_words = {"and", "then", "the", "this", "that", "want", "now"}
    if tokens and tokens[0] in filler_starters and any(token in date_words or token.isdigit() for token in tokens):
        return True
    return any((len(token) >= 6 or token.isdigit()) and token not in weak_words for token in tokens)


def _word_suffix_after_normalized_prefix(word: Word, prefix_norm: str) -> str | None:
    """Return a small extension after a repeated token prefix, if punctuation-bound.

    This is intentionally narrow for command/file surfaces: ``foo.py`` after a
    committed ``foo`` should leave ``py`` visible after the replayed prefix is
    removed, while ordinary word growth such as ``project`` -> ``projection``
    must not be treated as a safe suffix split.
    """

    prefix = prefix_norm or ""
    if not prefix:
        return None
    matched = 0
    punctuation_between = False
    suffix_chars: list[str] = []
    for ch in word.text or "":
        if matched < len(prefix):
            if not ch.isalnum():
                continue
            if ch.lower() != prefix[matched]:
                return None
            matched += 1
            continue
        if not ch.isalnum():
            if not suffix_chars:
                punctuation_between = True
            continue
        suffix_chars.append(ch)
    if matched != len(prefix) or not punctuation_between:
        return None
    suffix = "".join(suffix_chars).strip()
    if not suffix:
        return None
    suffix_norm = "".join(ch.lower() for ch in suffix if ch.isalnum())
    if not suffix_norm or len(suffix_norm) > 6:
        return None
    return suffix


def trim_audio_at_segment_end(
    buffer_audio,
    buffer_start_t: float,
    sample_rate_hz: int,
    segment_end_t: float,
    carry_over_seconds: float = 0.0,
):
    """Drop buffer audio up to ``segment_end_t``. Returns (new_buffer, new_start_t).

    ``buffer_audio`` must be a numpy ndarray. We avoid importing numpy at module
    load — pass the array in.
    """
    trim_t = max(buffer_start_t, segment_end_t - max(0.0, float(carry_over_seconds)))
    drop_samples = int(round((trim_t - buffer_start_t) * sample_rate_hz))
    if drop_samples <= 0:
        return buffer_audio, buffer_start_t
    if drop_samples >= len(buffer_audio):
        return buffer_audio[0:0], buffer_start_t + len(buffer_audio) / sample_rate_hz
    return buffer_audio[drop_samples:], buffer_start_t + drop_samples / sample_rate_hz


def force_trim_audio(
    buffer_audio,
    buffer_start_t: float,
    sample_rate_hz: int,
    max_seconds: float = 25.0,
    carry_over_seconds: float = 5.0,
):
    """Safety trim. If buffer exceeds ``max_seconds``, drop everything but the
    most recent ``carry_over_seconds``. Returns (new_buffer, new_start_t, did_trim).

    The carry-over exists so LocalAgreement still has audio context to anchor
    its next hypothesis against; the cut text relies on the ``initial_prompt``
    biasing for continuity.
    """
    if len(buffer_audio) <= max_seconds * sample_rate_hz:
        return buffer_audio, buffer_start_t, False
    keep_samples = int(carry_over_seconds * sample_rate_hz)
    drop_samples = len(buffer_audio) - keep_samples
    return (
        buffer_audio[drop_samples:],
        buffer_start_t + drop_samples / sample_rate_hz,
        True,
    )
