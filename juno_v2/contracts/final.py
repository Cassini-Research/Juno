from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from juno_v2.contracts.audio import AudioSamples


@dataclass(slots=True)
class FinalDecodeRequest:
    utterance_id: str
    audio: AudioSamples
    sample_rate_hz: int
    start_ms: float
    end_ms: float
    language: str | None = None
    allowed_languages: list[str] = field(default_factory=list)
    language_policy: str | None = None
    initial_prompt: str | None = None
    bias_phrases: list[str] = field(default_factory=list)
    context_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def audio_duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass(slots=True)
class FinalSegment:
    start_ms: float
    end_ms: float
    text: str
    # Per-segment audio-side signals produced by whisper-family backends.
    # ``no_speech_prob`` is whisper's posterior that the segment is
    # non-speech (>= 0.6 is the model's own silence threshold).
    # ``avg_logprob`` is the average token logprob for the segment
    # (< -1.0 is whisper's own low-confidence threshold). Both are
    # optional because non-whisper backends (qwen_asr, local_http_json)
    # don't surface them. Used by the commit-side
    # trailing-silence-hallucination guard to corroborate per-segment
    # stock-phrase matches with the audio signals from the *same*
    # segment, so a real "thank you" with confident decode is preserved
    # while a hallucinated "Thank you." tail with no_speech_prob >= 0.6
    # is stripped.
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FinalDecodeResult:
    utterance_id: str
    text: str
    start_ms: float
    end_ms: float
    audio_duration_ms: float
    backend_name: str
    # Absolute path (or model id) of the loaded weights. Populated by
    # every :class:`~juno_v2.final.backends.base.FinalAsrBackend`
    # implementation so per-utterance trace events and history readers
    # can attribute a transcript to the exact checkpoint that produced
    # it. Empty string when the backend wasn't configured with a path
    # (e.g. remote HTTP JSON backends that dispatch by endpoint).
    model_path: str = ""
    language: str | None = None
    decode_ms: float = 0.0
    end_of_turn_latency_ms: float = 0.0
    segments: List[FinalSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(slots=True)
class FinalTranscript:
    utterance_id: str
    text: str
    start_ms: float
    end_ms: float
    backend_name: str
    # Provenance for the per-utterance emission trace. Mirrors
    # :attr:`FinalDecodeResult.model_path` so the downstream ``final_
    # transcript_emitted`` event and the recovery history can attribute
    # each commit to a specific checkpoint.
    model_path: str = ""
    language: str | None = None
    decode_ms: float = 0.0
    end_of_turn_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FinalSessionSummary:
    session_id: str
    utterance_count: int
    decode_count: int
    emitted_count: int
    total_audio_ms: float
    average_decode_ms: float = 0.0
    average_end_of_turn_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptQualityReport:
    reference_text: str
    hypothesis_text: str
    word_error_rate: float
    char_error_rate: float
    word_distance: int
    char_distance: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Quality-report computation
# Placed here (contracts layer) so both language/ and final/ can import it
# without creating a cross-layer back-edge.
# ---------------------------------------------------------------------------

def compute_quality_report(reference_text: str, hypothesis_text: str) -> TranscriptQualityReport:
    """Compute word/character error rates between *reference_text* and *hypothesis_text*."""
    ref_words = _normalize_words(reference_text)
    hyp_words = _normalize_words(hypothesis_text)
    ref_chars = list(_normalize_chars(reference_text))
    hyp_chars = list(_normalize_chars(hypothesis_text))
    word_distance = _levenshtein(ref_words, hyp_words)
    char_distance = _levenshtein(ref_chars, hyp_chars)
    word_error_rate = (word_distance / len(ref_words)) if ref_words else float(word_distance > 0)
    char_error_rate = (char_distance / len(ref_chars)) if ref_chars else float(char_distance > 0)
    return TranscriptQualityReport(
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        word_error_rate=word_error_rate,
        char_error_rate=char_error_rate,
        word_distance=word_distance,
        char_distance=char_distance,
    )


def _normalize_words(text: str) -> list:
    return [token for token in text.strip().lower().split() if token]


def _normalize_chars(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        curr = [i]
        for j, right in enumerate(b, start=1):
            cost = 0 if left == right else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]
