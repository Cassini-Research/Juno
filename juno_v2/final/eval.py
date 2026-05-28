from __future__ import annotations

# Re-export from the contracts layer so that existing callers of
# ``juno_v2.final.eval.compute_quality_report`` continue to work without
# change, while the canonical implementation lives in contracts/ (no
# upward dependencies).
from juno_v2.contracts.final import (  # noqa: F401
    TranscriptQualityReport,
    compute_quality_report,
)
