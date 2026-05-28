"""One-shot dictation service used by the Mac shell and compatible clients."""

from juno_core_v3.dictation.pipeline import (
    OneShotDictationPipeline,
    OneShotDictationResult,
    UtteranceRecord,
    UtteranceRecordCache,
)
from juno_core_v3.dictation.transcriber import (
    DictationTranscriber,
    FinalBackendTranscriber,
    StubTranscriber,
    TranscribeResult,
    TranscribeUnavailable,
    UnavailableTranscriber,
    resolve_transcriber_from_env,
)

__all__ = [
    "DictationTranscriber",
    "FinalBackendTranscriber",
    "OneShotDictationPipeline",
    "OneShotDictationResult",
    "StubTranscriber",
    "TranscribeResult",
    "TranscribeUnavailable",
    "UnavailableTranscriber",
    "UtteranceRecord",
    "UtteranceRecordCache",
    "resolve_transcriber_from_env",
]
