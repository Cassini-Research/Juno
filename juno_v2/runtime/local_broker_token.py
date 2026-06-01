"""Local-only shared secret between the macOS shell and the loopback broker (repair doc P18).

Token file: ``<juno_runtime_dir>/broker_local_token`` (0600).

Bypass (tests / local hacking): ``JUNO_DEV_NO_AUTH=1``.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from juno_v2.runtime.paths import juno_runtime_dir, juno_support_root

_TOKEN_NAME = "broker_local_token"


def dev_no_auth() -> bool:
    return os.environ.get("JUNO_DEV_NO_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}


def auth_enforcement_enabled() -> bool:
    """Mutating broker routes require ``X-Juno-Local-Token`` by default."""
    if dev_no_auth():
        return False
    raw = os.environ.get("JUNO_REQUIRE_LOCAL_BROKER_AUTH")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def token_path() -> Path:
    return juno_runtime_dir() / _TOKEN_NAME


def legacy_token_path() -> Path:
    return juno_support_root() / _TOKEN_NAME


def read_local_broker_token() -> str | None:
    if dev_no_auth():
        return None
    p = token_path()
    if not p.is_file():
        legacy = legacy_token_path()
        if legacy.is_file():
            p = legacy
        else:
            return None
    try:
        raw = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return raw or None


def ensure_local_broker_token() -> str | None:
    """Create the token file if missing. Returns the token, or ``None`` if auth is disabled."""
    if dev_no_auth():
        return None
    root = juno_runtime_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    p = token_path()
    if p.exists():
        return read_local_broker_token()
    legacy = legacy_token_path()
    if legacy.is_file():
        try:
            tok = legacy.read_text(encoding="utf-8").strip()
        except OSError:
            tok = ""
        if tok:
            try:
                p.write_text(tok + "\n", encoding="utf-8")
                try:
                    p.chmod(0o600)
                except OSError:
                    pass
                return tok
            except OSError:
                return tok
    tok = secrets.token_urlsafe(32)
    try:
        p.write_text(tok + "\n", encoding="utf-8")
        try:
            p.chmod(0o600)
        except OSError:
            pass
    except OSError:
        return None
    return tok


def regenerate_local_broker_token() -> str | None:
    """Create a fresh per-engine-spawn token in the runtime directory."""
    if dev_no_auth():
        return None
    root = juno_runtime_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    tok = secrets.token_urlsafe(32)
    try:
        p = token_path()
        p.write_text(tok + "\n", encoding="utf-8")
        try:
            p.chmod(0o600)
        except OSError:
            pass
    except OSError:
        return None
    return tok


def verify_request_token(header_value: str | None) -> bool:
    if dev_no_auth():
        return True
    expected = read_local_broker_token()
    if not expected:
        return False
    if not header_value:
        return False
    a = header_value.strip().encode("utf-8")
    b = expected.encode("utf-8")
    if len(a) > 512 or len(b) > 512:
        return False
    import hmac

    return hmac.compare_digest(a, b)


def route_requires_local_token(*, path: str, method: str) -> bool:
    """Return True when ``X-Juno-Local-Token`` must match (mutating broker routes)."""
    if not auth_enforcement_enabled():
        return False
    if method.upper() not in {"POST", "DELETE", "PUT", "PATCH"}:
        return False
    p = path.split("?", 1)[0].rstrip("/")
    if p == "/api/broker/dictation/ingest_wav":
        return True
    if p == "/api/broker/insertion/committed":
        return True
    if p == "/api/broker/learning/observe_correction":
        return True
    if p == "/api/broker/history/clear_all":
        return True
    if p == "/api/broker/history/cancel_draft":
        return True
    # Recovery actions on existing rows: keep aligned with cancel_draft /
    # clear_all (other mutating history routes that already require a
    # local token). reprocess and POST .../actions intentionally remain
    # off the list to preserve their pre-existing call surface.
    if p == "/api/broker/history/insert_again":
        return True
    if p == "/api/broker/retention/run_cleanup":
        return True
    if p == "/api/broker/storage/audio/prune_all":
        return True
    if p == "/api/broker/personalization/user_profile":
        return True
    if p.startswith("/api/broker/settings/"):
        return True
    if p in {"/api/broker/setup/install", "/api/broker/setup/repair"}:
        return True
    if p.startswith("/api/broker/memory/"):
        return True
    if p.startswith("/api/broker/privacy/"):
        return True
    if p.startswith("/api/broker/history/") and method.upper() == "DELETE":
        return True
    return False


__all__ = [
    "auth_enforcement_enabled",
    "dev_no_auth",
    "ensure_local_broker_token",
    "regenerate_local_broker_token",
    "read_local_broker_token",
    "route_requires_local_token",
    "token_path",
    "verify_request_token",
]
