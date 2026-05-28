"""Keystore helpers for HMAC-SHA256 package signing.

Production trust keys must be supplied explicitly with ``JUNO_KEYSTORE``
or a programmatic ``path=`` argument. The repository also includes an
example key file for local development, but it is loaded only when
``JUNO_ALLOW_EXAMPLE_KEYSTORE=1`` is set.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping

from juno_core_v3.model_registry.contracts import PackageSignature
from juno_core_v3.model_registry.registry import ModelPackage
from juno_core_v3.model_registry.signature import (
    canonical_payload,
    compute_hmac_signature,
)

_logger = logging.getLogger(__name__)

_EXAMPLE_KEYSTORE_RELPATH = Path("config") / "keystore.example.json"


def load_keystore(path: str | os.PathLike[str] | None = None) -> dict[str, bytes] | None:
    """Load trust keys from a JSON file.

    Precedence:

    1. Explicit ``path`` argument.
    2. ``JUNO_KEYSTORE`` env var.
    3. ``config/keystore.example.json`` relative to the repo root, only
       when ``JUNO_ALLOW_EXAMPLE_KEYSTORE=1`` is set.

    Returns ``None`` when nothing resolves to a readable file.
    Invalid JSON or malformed entries raise ``ValueError`` rather
    than silently skipping — a broken keystore must surface before
    the registry quietly accepts unsigned packages.
    """
    candidate: Path | None = None
    if path is not None:
        candidate = Path(path).expanduser()
    else:
        env = os.environ.get("JUNO_KEYSTORE")
        if env:
            candidate = Path(env).expanduser()
        elif os.environ.get("JUNO_ALLOW_EXAMPLE_KEYSTORE") == "1":
            repo_root = _find_repo_root()
            if repo_root is not None:
                default_path = repo_root / _EXAMPLE_KEYSTORE_RELPATH
                # ``is_file`` raises ``PermissionError`` when the example keystore
                # sits under a macOS-protected directory (e.g. ~/Documents)
                # and the engine runs sandboxed under launchd. Treat any
                # access denial as "not present" — signing is optional.
                try:
                    if default_path.is_file():
                        candidate = default_path
                except (PermissionError, OSError):
                    candidate = None

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
        # macOS sandbox can allow ``stat`` (is_file) but deny ``open`` for
        # paths under protected directories like ~/Documents. Treat as
        # "no keystore available" instead of crashing the engine.
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"keystore_invalid_json: {candidate}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"keystore_empty_or_not_object: {candidate}")

    out: dict[str, bytes] = {}
    for key_id, hex_value in raw.items():
        if not isinstance(key_id, str) or not key_id:
            raise ValueError(f"keystore_bad_key_id in {candidate}")
        if not isinstance(hex_value, str):
            raise ValueError(f"keystore_bad_key_value for {key_id} in {candidate}")
        try:
            out[key_id] = bytes.fromhex(hex_value)
        except ValueError as exc:
            raise ValueError(
                f"keystore_bad_hex for {key_id} in {candidate}"
            ) from exc
    return out


def sign_package(pkg: ModelPackage, *, key_id: str, key: bytes) -> None:
    """Populate ``pkg.signature`` so it verifies under ``key``.

    The signature covers the canonical JSON of the package with the
    ``signature`` field cleared, matching what
    :func:`verify_signature` reconstructs on the verify side. ``key_id``
    is recorded implicitly via the fact that the keystore maps it to
    the signing key; we don't put it inside the signature because the
    current :class:`PackageSignature` contract only has ``algo`` +
    ``value``.
    """
    # ``key_id`` is not stored in the signature itself (contract
    # mismatch) but is threaded through to keep the API aligned with
    # how the keystore is keyed — this way ops can rotate keys by
    # re-signing with the new key id without changing callers.
    _ = key_id
    payload = canonical_payload(pkg.to_dict())
    hex_sig = compute_hmac_signature(payload, key)
    pkg.signature = PackageSignature(algo="hmac-sha256", value=hex_sig)


def sign_packages(
    packages: list[ModelPackage],
    *,
    trust_keys: Mapping[str, bytes],
    key_id: str | None = None,
) -> None:
    """Sign every package in place using ``key_id`` (or the first key)."""
    if not trust_keys:
        raise ValueError("sign_packages_requires_nonempty_keystore")
    if key_id is None:
        key_id = next(iter(trust_keys))
    if key_id not in trust_keys:
        raise KeyError(f"sign_packages_unknown_key_id: {key_id}")
    key = trust_keys[key_id]
    for pkg in packages:
        sign_package(pkg, key_id=key_id, key=key)


def _find_repo_root() -> Path | None:
    """Walk up from this file to find a directory that looks like the
    repo root (contains ``config/`` and ``juno_core_v3/``).

    Each ``is_dir`` probe can raise ``PermissionError`` when the engine
    runs under a sandboxed launchd agent and the walk crosses a
    macOS-protected directory (e.g. ``~/Documents``). Treat any access
    denial as "not the repo root" — a packaged install must keep
    working without source-tree access.
    """
    try:
        here = Path(__file__).resolve()
    except (PermissionError, OSError):
        return None
    for parent in here.parents:
        try:
            if (parent / "config").is_dir() and (parent / "juno_core_v3").is_dir():
                return parent
        except (PermissionError, OSError):
            return None
    return None
