"""Session runners for the broker plane.

Each ``SessionKind`` from ``juno_core_v3.contracts.session`` maps to a
concrete runner:

- ``SessionKind.INSERT``    -> :class:`InsertRunner`    (wraps the v3
  :class:`~juno_core_v3.dictation.pipeline.OneShotDictationPipeline`;
  drives a WAV-in, committed-transcript-out session with broker-policy
  knobs like ``reduce_optional_lanes``).
- ``SessionKind.TRANSFORM`` -> :class:`TransformRunner` (operates on
  selected text + a hint).

This split is the product contract: classifying a session is step one, but
each class has its *own* execution path and should never silently degenerate
into the others.

Note on the streaming live-mic engine: ``InsertRunner`` deliberately
does **not** host the long-running streaming loop
(:class:`~juno_v2.engine.session.DictationSessionRunner`). That
pathway is its own project — VAD, audio ring buffer, Metal scheduling,
trace, commit controller. ``InsertRunner`` is the broker-symmetric
runner for *one-shot* Insert (ingest a WAV, replay a stored utterance,
answer an HTTP transcription request); the streaming engine still
owns live dictation and may itself emit through this runner in a
future phase.
"""

from juno_core_v3.broker.runners.insert import InsertRequest, InsertResult, InsertRunner
from juno_core_v3.broker.runners.transform import (
    TransformHint,
    TransformRequest,
    TransformResult,
    TransformRunner,
)

__all__ = [
    "InsertRequest",
    "InsertResult",
    "InsertRunner",
    "TransformHint",
    "TransformRequest",
    "TransformResult",
    "TransformRunner",
]
