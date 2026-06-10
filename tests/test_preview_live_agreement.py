import numpy as np

from juno_v2.contracts.preview import PreviewDecodeRequest
from juno_v2.preview.live_agreement import HypothesisBuffer, Word, _find_replayed_prefix_len
from juno_v2.preview.streaming_core import (
    StreamingPreviewSessionManager,
    WhisperDecodeOutput,
)


def _words(text: str, start: float = 0.0) -> list[Word]:
    out: list[Word] = []
    t = start
    for token in text.split():
        out.append(Word(start=t, end=t + 0.2, text=token))
        t += 0.2
    return out


def test_preview_does_not_replace_unrelated_committed_suffix() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("The deadline is Friday")
    buffer.tail = _words("No make that Monday", start=1.0)

    buffer.insert(_words("No make that Monday", start=1.0))
    buffer.flush()

    assert buffer.committed_text() == "The deadline is Friday No make that Monday"


def test_preview_drops_short_boundary_revision_overlap() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words(
        "Okay now I am speaking in Hindi and then I will switch back to English "
        "Please keep the sentence"
    )
    buffer.tail = _words("Please keep this sentence readable and text a word", start=5.0)

    replay_reason = buffer.insert(_words("Please keep this sentence readable and text our words", start=5.0))
    first_committed = buffer.flush()

    committed = buffer.committed_text()
    assert "Please keep the sentence Please keep this sentence" not in committed
    assert committed.endswith("Please keep the sentence")
    assert first_committed == []
    assert replay_reason == "committed_adjacent_revision_phrase_4x4"
    assert buffer.tail_text() == "readable and text our words"

    buffer.insert(_words("Please keep this sentence readable and text our words", start=5.0))
    newly_committed = buffer.flush()

    committed = buffer.committed_text()
    assert "Please keep the sentence Please keep this sentence" not in committed
    assert committed.endswith("Please keep the sentence readable and text our words")
    assert " ".join(w.text for w in newly_committed) == "readable and text our words"


def test_preview_drops_short_reanchored_opener_after_segment_trim() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("Okay now")
    buffer.tail = _words("Okay I'm speaking in Hindi", start=1.0)

    replay_reason = buffer.insert(_words("Okay I'm speaking in Hindi and then", start=1.0))
    first_committed = buffer.flush()

    assert replay_reason == "committed_short_near_boundary_reanchor_1_lag1"
    assert buffer.committed_text() == "Okay now"
    assert first_committed == []
    assert buffer.tail_text() == "I'm speaking in Hindi and then"

    buffer.insert(_words("I'm speaking in Hindi and then I will", start=1.0))
    newly_committed = buffer.flush()

    assert buffer.committed_text() == "Okay now I'm speaking in Hindi and then"
    assert "Okay now Okay" not in buffer.committed_text()
    assert " ".join(w.text for w in newly_committed) == "I'm speaking in Hindi and then"


def test_preview_live_append_does_not_drop_near_boundary_committed_suffix() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words(
        "Okay now I am speaking in Hindi and then I will switch back to English "
        "Please keep the sentence readable and text our"
    )
    promoted_tail = _words("sentence readable and text awards", start=8.0)

    old_len = len(buffer.committed)
    buffer.committed.extend(promoted_tail)
    kept_tail, reason = buffer.drop_adjacent_duplicate_boundary_after_append(old_len, promoted_tail)

    assert "text our sentence readable and text" in buffer.committed_text()
    assert " ".join(w.text for w in kept_tail) == "sentence readable and text awards"
    assert reason is None


def test_preview_final_tail_drops_near_boundary_replay_prefix() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words(
        "Okay now I am speaking in Hindi and then I will switch back to English "
        "Please keep the sentence readable and text our"
    )
    promoted_tail = _words("sentence readable and text awards", start=8.0)

    old_len = len(buffer.committed)
    buffer.committed.extend(promoted_tail)
    kept_tail, reason = buffer.drop_adjacent_duplicate_boundary_after_append(
        old_len,
        promoted_tail,
        allow_committed_suffix_drop=True,
    )

    committed = buffer.committed_text()
    assert "text our sentence readable and text" not in committed
    assert committed.endswith("Please keep the sentence readable and text awards")
    assert " ".join(w.text for w in kept_tail) == "awards"
    assert reason == "near_boundary_replay_phrase_4_lag1"


def test_preview_tail_drops_recent_committed_phrase_with_compound_drift() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("Ask ReviewMate to review the off-callback flow")
    buffer.tail = _words("To review the off call back flow find edge cases")

    reason = buffer.drop_repeated_tail_suffix()

    assert reason == "tail_recent_replay_phrase_5x7_lag0"
    assert buffer.committed_text() == "Ask ReviewMate to review the off-callback flow"
    assert buffer.tail_text() == "find edge cases"


def test_preview_drops_adjacent_stem_duplicate_at_commit_boundary() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("copy text style format")
    buffer.tail = _words("formatting anything", start=1.0)

    buffer.insert(_words("formatting anything", start=1.0))
    buffer.flush()

    assert buffer.committed_text() == "copy text style formatting anything"
    assert "format formatting" not in buffer.committed_text()
    assert buffer.last_flush_replay_drop_reason == "agreement_boundary_stem_revision"


def test_preview_drops_article_bridged_duplicate_at_commit_boundary() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("where I was invited as a judge")
    buffer.tail = _words("the judge was based", start=1.0)

    buffer.insert(_words("the judge was based", start=1.0))
    buffer.flush()

    assert buffer.committed_text() == "where I was invited as a judge was based"
    assert "judge the judge" not in buffer.committed_text()
    assert buffer.last_flush_replay_drop_reason == "agreement_boundary_article_replay"


def test_preview_agreement_does_not_roll_back_committed_suffix() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("Okay now I am going to")

    agreement, reason = buffer._drop_replayed_agreement_prefix(
        _words("Okay now I am speaking in Hindi")
    )

    assert " ".join(w.text for w in agreement) == "Okay now I am speaking in Hindi"
    assert buffer.committed_text() == "Okay now I am going to"
    assert reason is None


def test_preview_opening_revision_does_not_roll_back_committed_suffix() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("Okay now I'm talking about Hindi")

    agreement, reason = buffer._drop_replayed_agreement_prefix(
        _words("Okay now I'm speaking in Hindi")
    )

    assert " ".join(w.text for w in agreement) == "Okay now I'm speaking in Hindi"
    assert buffer.committed_text() == "Okay now I'm talking about Hindi"
    assert reason is None


def test_preview_revision_starting_at_anchor_does_not_roll_back_live_committed_text() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("Okay now I'm talking about Hindi")
    buffer.tail = _words("I'm speaking in Hindi and then", start=0.4)

    reason = buffer.insert(_words("I'm speaking in Hindi and then I will", start=0.4))
    newly_committed = buffer.flush()

    assert reason is None
    assert buffer.committed_text() == "Okay now I'm talking about Hindi I'm speaking in Hindi and then"
    assert " ".join(w.text for w in newly_committed) == "I'm speaking in Hindi and then"


def test_preview_does_not_roll_back_similar_new_sentence_after_boundary() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words("Okay now I'm talking about Hindi")
    buffer.tail = _words("I'm speaking in Hindi next", start=1.4)

    reason = buffer.insert(_words("I'm speaking in Hindi next", start=1.4))
    buffer.flush()

    assert reason is None
    assert buffer.committed_text() == "Okay now I'm talking about Hindi I'm speaking in Hindi next"


def test_preview_drops_long_replayed_prefix_from_inside_committed_history() -> None:
    buffer = HypothesisBuffer()
    buffer.committed = _words(
        "The customer meeting is in Japan No actually Korea "
        "The owner is Neel scratch that Neil Ofer"
    )

    reason = buffer.insert(
        _words("customer meeting is in Japan No The deadline is Friday no make that Monday")
    )

    assert reason == "committed_replay_prefix_6"
    assert " ".join(w.text for w in buffer._staged_new) == "The deadline is Friday no make that Monday"


def test_replayed_prefix_matcher_rejects_first_token_asr_drift() -> None:
    committed_norm = [
        "".join(ch.lower() for ch in token if ch.isalnum())
        for token in (
            "The customer meeting is in Japan No actually Korea "
            "The owner is Neel scratch that Neil Ofer"
        ).split()
    ]
    candidate_norm = [
        "".join(ch.lower() for ch in token if ch.isalnum())
        for token in "Neil scratch that Neil Ofer The deadline is Friday".split()
    ]

    assert _find_replayed_prefix_len(committed_norm, candidate_norm) == 0


class _SequenceDecoder:
    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.calls = 0

    def warm(self) -> None:
        return None

    def decode(self, **kwargs: object) -> WhisperDecodeOutput:
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        word_dicts = []
        t = 0.0
        for token in text.split():
            word_dicts.append({"word": token, "start": t, "end": t + 0.2})
            t += 0.2
        return WhisperDecodeOutput(
            text=text,
            language="en",
            decode_ms=1.0,
            word_dicts=word_dicts,
            segment_ends=[],
            metadata={"last_segment_no_speech_prob": 0.0},
        )


def _request(
    *,
    utterance_id: str = "utt",
    is_final: bool = False,
    preview_segment_final: bool = False,
) -> PreviewDecodeRequest:
    context_payload = {
        "preview_display_orthography": False,
        "preserve_state_on_decode_seq_zero": True,
    }
    if preview_segment_final:
        context_payload["preview_segment_final"] = True
        context_payload["retain_state_after_final"] = True
    audio = np.ones(1600, dtype=np.float32) * 0.01
    return PreviewDecodeRequest(
        utterance_id=utterance_id,
        audio=audio,
        sample_rate_hz=16000,
        start_ms=0.0,
        end_ms=100.0,
        is_final=is_final,
        decode_seq=0,
        reset_decoder_state=False,
        context_payload=context_payload,
    )


def test_preview_segment_final_does_not_force_commit_same_audio_tail() -> None:
    decoder = _SequenceDecoder(
        "The owner is Neil Scrassegg",
        "The owner is Neil Scrassegg",
    )
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=60_000.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
    )

    first = manager.process(_request())
    assert first.committed_text == ""
    assert manager._states["utt"].hypothesis.tail_text() == "The owner is Neil Scrassegg"

    segment_boundary = manager.process(_request(is_final=True, preview_segment_final=True))

    assert decoder.calls == 1
    assert segment_boundary.committed_text == ""
    assert manager._states["utt"].hypothesis.tail_text() == "The owner is Neil Scrassegg"
    assert segment_boundary.metadata["tail_final_promotion_status"] == "deferred"
    assert segment_boundary.metadata["tail_final_promotion_reason"] == "segment_boundary"


def test_root_final_can_commit_tail_at_true_utterance_end() -> None:
    decoder = _SequenceDecoder(
        "The final word is visible",
        "The final word is visible",
    )
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
    )

    manager.process(_request())
    final = manager.process(_request(is_final=True))

    assert decoder.calls == 2
    assert final.committed_text == "The final word is visible"


def test_root_final_promotes_short_complete_tail_when_nothing_committed() -> None:
    decoder = _SequenceDecoder(
        "",
        "Fuck the new website I am still going with the old website I will make some hand changes to it one by one",
    )
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
    )

    first = manager.process(_request())
    assert first.committed_text == ""

    final = manager.process(_request(is_final=True))

    assert final.committed_text == (
        "Fuck the new website I am still going with the old website "
        "I will make some hand changes to it one by one"
    )
    assert final.tail_text == ""
    assert final.metadata["tail_final_promotion_status"] == "promoted"
    assert final.metadata["tail_final_promotion_max_words"] == 32


def test_root_final_blocks_display_suppressed_single_word_tail() -> None:
    decoder = _SequenceDecoder(
        "Ask ReviewMate to review the auth callback flow",
        "Ask ReviewMate to review the auth callback flow",
        "Ask ReviewMate to review the auth callback flow Old.",
    )
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
    )

    manager.process(_request())
    manager.process(_request())
    final = manager.process(_request(is_final=True))

    assert decoder.calls == 3
    assert final.committed_text == "Ask ReviewMate to review the auth callback flow"
    assert final.tail_text == ""
    assert final.metadata["tail_display_suppress_reason"] == "tail_single_word_quarantine"
    assert final.metadata["tail_final_promotion_status"] == "blocked"
    assert final.metadata["tail_final_promotion_reason"] == "tail_single_word_quarantine"


def test_preview_canonicalizes_oke_before_agreement_overlap() -> None:
    decoder = _SequenceDecoder(
        "Oke",
        "Oke",
        "Okay this is a slow test",
        "Okay this is a slow test",
    )
    manager = StreamingPreviewSessionManager(
        decoder,
        decode_cadence_ms=0.0,
        min_decode_audio_ms=0.0,
        vad_enabled=False,
    )

    manager.process(_request())
    manager.process(_request())
    manager.process(_request())
    final = manager.process(_request(is_final=True))

    assert final.committed_text == "Okay this is a slow test"
    assert "Oke Okay" not in final.committed_text
