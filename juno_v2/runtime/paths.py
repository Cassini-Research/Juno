"""Canonical filesystem locations for Juno consumer data.

- All shipping paths key off the bundle identifier (``JUNO_BUNDLE_ID``,
  default ``com.juno.shell``), matching how every production macOS app
  organizes user data. The legacy literal ``Juno`` directory is migrated
  once on first launch under the new convention.
- ``JUNO_APP_SUPPORT_DIR`` remains the highest-precedence override so
  tests and headless dev environments can isolate state.
- Runtime artifacts (instance lock, engine socket, endpoint metadata,
  local auth token) live under ``<support_root>/runtime`` with mode
  0700 — only the user's UID can read them.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DEFAULT_BUNDLE_ID = "com.juno.shell"
LEGACY_SUPPORT_DIR_NAME = "Juno"


def juno_dev_mode() -> bool:
    return os.environ.get("JUNO_DEV_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def juno_dev_demo_root(cwd: Path | None = None) -> Path | None:
    """Return repo demo root when dev mode is on; ``None`` otherwise."""
    if not juno_dev_mode():
        return None
    root = cwd or Path.cwd()
    return (root / ".juno_v2_demo").resolve()


def juno_bundle_id() -> str:
    raw = os.environ.get("JUNO_BUNDLE_ID", "").strip()
    return raw or DEFAULT_BUNDLE_ID


def _macos_support_parent() -> Path:
    return Path.home() / "Library" / "Application Support"


def juno_support_root() -> Path:
    """Root for all per-user Juno data on disk.

    Resolution order:
    1. ``JUNO_APP_SUPPORT_DIR`` (test/dev override).
    2. macOS: ``~/Library/Application Support/<bundle_id>``.
    3. Linux/CI: ``~/.juno``.
    """
    raw = os.environ.get("JUNO_APP_SUPPORT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if sys.platform == "darwin":
        return (_macos_support_parent() / juno_bundle_id()).resolve()
    return (Path.home() / ".juno").resolve()


def juno_runtime_dir() -> Path:
    """Mutable runtime artifacts (lock, socket, token, endpoint json).

    Created with mode 0700 so only the owning user can connect to the
    engine socket or read the local auth token. This is the structural
    defense against another local user impersonating the engine.
    """
    p = juno_support_root() / "runtime"
    try:
        p.mkdir(parents=True, exist_ok=True)
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p


def juno_engine_socket_path() -> Path:
    return juno_runtime_dir() / "engine.sock"


def juno_endpoint_metadata_path() -> Path:
    return juno_runtime_dir() / "broker_endpoint.json"


def migrate_legacy_support_root() -> Path | None:
    """One-shot migration from ``~/Library/Application Support/Juno`` to
    the bundle-id-keyed location. Returns the migration source if a
    move happened, else ``None``.

    Idempotent: safe to call on every launch. Refuses to overwrite an
    existing target — falls back to copying anything missing rather
    than clobbering a fresh install.
    """
    if sys.platform != "darwin":
        return None
    if os.environ.get("JUNO_APP_SUPPORT_DIR", "").strip():
        return None
    legacy = _macos_support_parent() / LEGACY_SUPPORT_DIR_NAME
    target = juno_support_root()
    if legacy == target:
        return None
    if not legacy.is_dir():
        return None
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    moved = False
    skip_names = {".migrated_from_legacy"}
    for entry in legacy.iterdir():
        if entry.name in skip_names:
            continue
        dest = target / entry.name
        if dest.exists():
            continue
        try:
            shutil.move(str(entry), str(dest))
            moved = True
        except OSError:
            continue
    # Marker lives in the *target* (the new canonical home), so the
    # legacy directory contents stop changing after the first call —
    # makes the migration idempotent.
    try:
        (target / ".migrated_from_legacy").write_text(
            f"{legacy}\n", encoding="utf-8"
        )
    except OSError:
        pass
    return legacy if moved else None


def juno_profile_root(cwd: Path | None = None) -> Path:
    """Root for the broker's demo-profile config and provisioning scratch.

    - Dev mode (``JUNO_DEV_MODE``): ``<cwd>/.juno_v2_demo``.
    - A repo checkout that already has ``<cwd>/.juno_v2_demo``: keep using
      it, so existing dev setups don't silently switch profiles.
    - Otherwise (packaged app): ``<support_root>/demo``. The packaged
      engine's cwd is inside the signed .app bundle — writing there is
      both blocked at runtime (EPERM) and would break the code-signing
      seal, which is exactly how a fresh DMG install's first model
      provisioning used to fail.
    """
    demo = juno_dev_demo_root(cwd)
    if demo is not None:
        return demo
    candidate = ((cwd or Path.cwd()) / ".juno_v2_demo").resolve()
    if candidate.is_dir():
        return candidate
    return (juno_support_root() / "demo").resolve()


def default_workbench_log_dir(cwd: Path | None = None) -> Path:
    """Default directory for workbench traces, broker settings, and JSON sidecars.

    - Dev mode: ``<cwd>/.juno_v2_demo/workbench`` (explicit repo coupling).
    - Otherwise: ``<juno_support_root>/Workbench``.
    """
    demo = juno_dev_demo_root(cwd)
    if demo is not None:
        return (demo / "workbench").resolve()
    return (juno_support_root() / "Workbench").resolve()


def product_history_db_path(*, workbench_log_dir: Path) -> Path:
    """SQLite path for durable product history."""
    override = os.environ.get("JUNO_PRODUCT_HISTORY_DIR", "").strip()
    if override:
        return (Path(override).expanduser() / "history.sqlite").resolve()
    return (Path(workbench_log_dir) / "product_history.sqlite").resolve()


def product_audio_root(*, workbench_log_dir: Path) -> Path:
    """Root for retained utterance WAVs."""
    override = os.environ.get("JUNO_PRODUCT_AUDIO_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path(workbench_log_dir) / "audio").resolve()


__all__ = [
    "DEFAULT_BUNDLE_ID",
    "default_workbench_log_dir",
    "juno_dev_demo_root",
    "juno_dev_mode",
    "juno_bundle_id",
    "juno_endpoint_metadata_path",
    "juno_engine_socket_path",
    "juno_profile_root",
    "juno_runtime_dir",
    "juno_support_root",
    "migrate_legacy_support_root",
    "product_audio_root",
    "product_history_db_path",
]
