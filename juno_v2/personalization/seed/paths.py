from __future__ import annotations

import os
from pathlib import Path


def repo_root_from_juno() -> Path:
    """``juno_v2/personalization/seed/`` → repository root (parent of ``juno_v2``)."""
    return Path(__file__).resolve().parents[3]


def default_seed_bundle_dir() -> Path:
    """Directory containing ``manifest.json`` and ``packs/``.

    Override with ``JUNO_SEED_BUNDLE_DIR`` for tests or custom installs.
    """
    env = (os.environ.get("JUNO_SEED_BUNDLE_DIR") or "").strip()
    if env:
        return Path(env).expanduser()
    return repo_root_from_juno() / "seed_data"
