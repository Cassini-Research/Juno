"""Opaque provenance tags for deterministic reliability-test traffic.

The tags are deliberately optional and content-free.  Production requests do
not receive synthetic values, while test harnesses can attach stable run/case
identifiers that survive the transcript trace and Product History pipeline.
"""

from __future__ import annotations

import re
from typing import Any


_TEST_PROVENANCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def test_provenance_fields(
    *,
    test_run_id: Any = None,
    test_case_id: Any = None,
) -> dict[str, str]:
    """Return bounded, content-free correlation fields for supplied tags.

    Restricting the alphabet prevents callers from accidentally placing free
    form transcript/context text in fields intended only for correlation.
    """

    out: dict[str, str] = {}
    run_id = _sanitize_test_provenance_id(test_run_id)
    case_id = _sanitize_test_provenance_id(test_case_id)
    if run_id is not None:
        out["test_run_id"] = run_id
    if case_id is not None:
        out["test_case_id"] = case_id
    return out


def _sanitize_test_provenance_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or _TEST_PROVENANCE_ID_RE.fullmatch(candidate) is None:
        return None
    return candidate


__all__ = ["test_provenance_fields"]
