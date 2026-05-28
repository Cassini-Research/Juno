"""Populate ``CapabilityManifest`` quality/latency metrics from eval runs.

The route chooser wants to pick "best model for this surface under
this load". To do that it needs *measured* WER and latency, not just
the symbolic `min_ram_mb` / `streaming` fields that already live on
the manifest. P2-10 adds two fields (`wer_p50`, `latency_ms_p50`) and
this module plumbs them in from the eval gate outputs.

Report schema
-------------

The loader accepts a JSON document of shape::

    {
      "packages": {
        "<package_id>": {
          "wer_p50": 0.091,
          "latency_ms_p50": 1120.0
        },
        ...
      }
    }

It intentionally ignores unknown keys so operators can enrich the
file with provenance (corpus name, git sha, etc.) without schema
churn.

Resolution order
----------------

1. Explicit ``path=`` argument (tests / programmatic callers).
2. ``JUNO_EVAL_REPORT`` env var.

Returns ``None`` when nothing resolves; bundled packages will ship
with ``wer_p50=None`` / ``latency_ms_p50=None`` which is safe — the
chooser must treat those as "unknown, don't rank on quality/latency".
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from juno_core_v3.model_registry.registry import ModelPackage


def load_eval_report(path: str | os.PathLike[str] | None = None) -> dict[str, dict] | None:
    """Load a per-package metrics report.

    Returns a plain ``{package_id: {metric: value}}`` mapping so
    callers can consume it without tying themselves to an extra
    dataclass. ``None`` is returned when no report file can be
    resolved — this is the "fresh clone, nothing measured yet"
    state.
    """
    candidate: Path | None = None
    if path is not None:
        candidate = Path(path).expanduser()
    else:
        env = os.environ.get("JUNO_EVAL_REPORT")
        if env:
            candidate = Path(env).expanduser()

    if candidate is None:
        return None
    try:
        if not candidate.is_file():
            return None
    except (PermissionError, OSError):
        return None

    try:
        text = candidate.read_text("utf-8")
    except (PermissionError, OSError):
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"eval_report_invalid_json: {candidate}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"eval_report_not_object: {candidate}")

    packages = raw.get("packages")
    if not isinstance(packages, dict):
        # Allow a flat ``{id: metrics}`` shape for convenience.
        packages = raw

    out: dict[str, dict] = {}
    for pkg_id, metrics in packages.items():
        if not isinstance(pkg_id, str) or not isinstance(metrics, dict):
            continue
        wer = metrics.get("wer_p50")
        lat = metrics.get("latency_ms_p50")
        clean: dict[str, float] = {}
        if _is_finite_number(wer):
            clean["wer_p50"] = float(wer)
        if _is_finite_number(lat):
            clean["latency_ms_p50"] = float(lat)
        if clean:
            out[pkg_id] = clean
    return out


def apply_eval_report(
    packages: Iterable[ModelPackage],
    report: Mapping[str, Mapping[str, float]] | None,
) -> int:
    """Rewrite manifests in place with measured metrics.

    Returns the number of packages that received at least one
    metric. Packages missing from the report are untouched (their
    manifests keep ``None`` for unknown metrics, which the chooser
    treats as "no signal").

    Must be called BEFORE signing, because the manifest is part of
    the signed payload — applying this after signing would
    invalidate every signature.
    """
    if not report:
        return 0
    updated = 0
    for pkg in packages:
        metrics = report.get(pkg.package_id)
        if not metrics:
            continue
        pkg.manifest = replace(pkg.manifest, **metrics)  # type: ignore[arg-type]
        updated += 1
    return updated


# ---- helpers -------------------------------------------------------


def _is_finite_number(x: object) -> bool:
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    if x != x:  # NaN
        return False
    if x in (float("inf"), float("-inf")):
        return False
    return True
