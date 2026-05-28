"""Meeting-output grammar layer.

Deterministic transforms for meeting-context voice writing:

- speaker-tag normalisation (``Alice: hello`` → ``**Alice:** hello``)
- action-items / next-steps heading detection + bullet grouping
- attendee extraction from the first few speaker lines

Only activated when ``app_category == 'meeting'`` (the new category
introduced in A3) or an explicit meeting-grammar overlay is
requested. Never leaks into prose / messaging / email / code flows —
those already have their own contracts.

No model, no network, no configuration. Mirrors the shape of
:mod:`juno_v2.code_grammar` so the two overlays are ergonomically
interchangeable from the transform runner's perspective.
"""
from juno_v2.meeting_grammar.engine import (
    MeetingGrammarEngine,
    MeetingGrammarMode,
    MeetingGrammarResult,
    detect_meeting_text,
)

__all__ = [
    "MeetingGrammarEngine",
    "MeetingGrammarMode",
    "MeetingGrammarResult",
    "detect_meeting_text",
]
