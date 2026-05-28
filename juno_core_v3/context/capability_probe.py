"""Broker-side integration with the ``juno-capability`` helper.

The helper is a Swift binary that spawns briefly, reads the Accessibility
tree, and prints one JSON line describing the frontmost app + focused
UI element. This module:

- Shells out to the helper with a tight timeout so the broker never
  hangs on a stuck AX call.
- Parses the JSON into :class:`CapabilityReport`.
- Runs the decision logic that decides whether dictation is allowed
  right now - returning a :class:`CapabilityDecision` the HTTP layer
  can forward to the shell.

The blocklist of managed / banking / sensitive apps is data, not code.
See :func:`load_default_blocklist` for the baked-in defaults; callers
can extend with their own bundle IDs via :class:`CapabilityChecker`.

Important separation-of-concerns note: this module never paints UI,
never prompts for AX permissions, and never talks to the mic. It is a
pure "is it safe to start dictation right now?" gate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from juno_core_v3.context.suppression_config import SuppressionConfig

# Baked-in default blocklist of bundle IDs where we refuse to auto-paste.
# Conservative by design: banking, password managers, enterprise finance,
# tax, and known secure-vault tools. Users / operators can replace this
# list entirely; the goal is simply "don't start with a security
# embarrassment on day one".
_DEFAULT_BLOCKLIST: frozenset[str] = frozenset(
    {
        # Password managers (often have dictation-friendly note fields
        # right next to secret fields; easier to block the app outright).
        "com.1password.1password",
        "com.1password.1password7",
        "com.agilebits.onepassword7",
        "com.bitwarden.desktop",
        "com.lastpass.LastPass",
        "com.dashlane.dashlanephonefinalmac",
        "com.keepassxc.keepassxc",
        # Banking / finance desktop apps with known secure flows.
        "com.intuit.quickbooks.mac",
        "com.intuit.TurboTax",
        "com.hrblock.desktop",
        # VPN / security tools with token fields.
        "com.1password.1password-cli",
        "com.tailscale.ipn.macos",
    }
)


@dataclass(slots=True, frozen=True)
class CapabilityReport:
    """Parsed output of ``juno-capability``."""

    ok: bool
    has_ax_trust: bool
    frontmost_app_bundle_id: str | None
    frontmost_app_name: str | None
    frontmost_pid: int | None
    window_title: str | None
    focused_role: str | None
    focused_subrole: str | None
    focused_is_secure: bool
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str) -> "CapabilityReport":
        payload = json.loads(text) if text.strip() else {}
        return cls(
            ok=bool(payload.get("ok", False)),
            has_ax_trust=bool(payload.get("has_ax_trust", False)),
            frontmost_app_bundle_id=payload.get("frontmost_app_bundle_id"),
            frontmost_app_name=payload.get("frontmost_app_name"),
            frontmost_pid=payload.get("frontmost_pid"),
            window_title=payload.get("window_title"),
            focused_role=payload.get("focused_role"),
            focused_subrole=payload.get("focused_subrole"),
            focused_is_secure=bool(payload.get("focused_is_secure", False)),
            error=payload.get("error"),
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "has_ax_trust": self.has_ax_trust,
            "frontmost_app_bundle_id": self.frontmost_app_bundle_id,
            "frontmost_app_name": self.frontmost_app_name,
            "frontmost_pid": self.frontmost_pid,
            "window_title": self.window_title,
            "focused_role": self.focused_role,
            "focused_subrole": self.focused_subrole,
            "focused_is_secure": self.focused_is_secure,
            "error": self.error,
        }


@dataclass(slots=True)
class CapabilityDecision:
    """The broker's verdict on whether dictation may proceed."""

    ok: bool
    reason: str  # machine-readable reason code
    message: str  # human-readable explanation for UI / logs
    report: CapabilityReport
    warn: bool = field(default=False)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "reason": self.reason,
            "message": self.message,
            "report": self.report.to_dict(),
        }
        if self.warn:
            out["warn"] = True
        return out


def load_default_blocklist() -> frozenset[str]:
    """Return the immutable default blocklist of managed-app bundle IDs.

    Kept as a function (not a module-level export) so callers can't
    mutate the shared set by mistake.
    """
    return _DEFAULT_BLOCKLIST


class CapabilityChecker:
    """Invokes the Swift helper and applies policy.

    :param helper_path: absolute path to the ``juno-capability`` binary.
        When ``None`` we resolve on PATH or beside the running Python
        process. Missing helper -> decision ``ok=False`` with reason
        ``helper_not_installed`` so the shell knows to offer install
        instructions rather than showing a generic error.
    :param timeout_sec: hard upper bound for the helper invocation. The
        probe normally takes <100 ms; we keep the cap small so a hung
        AX call can't freeze the broker.
    :param extra_blocklist: bundle IDs to add on top of the defaults.
    :param suppression_config: optional JSON-driven block/warn rules; defaults
        to an empty additive config.
    """

    def __init__(
        self,
        helper_path: str | None = None,
        *,
        timeout_sec: float = 1.5,
        extra_blocklist: Iterable[str] = (),
        suppression_config: SuppressionConfig | None = None,
    ) -> None:
        self._helper_path = helper_path or self._resolve_helper()
        self._timeout_sec = timeout_sec
        self._suppression_config = suppression_config or SuppressionConfig.default()
        self._blocklist: frozenset[str] = (
            _DEFAULT_BLOCKLIST
            | frozenset(s.lower() for s in extra_blocklist)
            | self._suppression_config.blocklist_bundle_ids
        )

    @staticmethod
    def _resolve_helper() -> str | None:
        """Locate the ``juno-capability`` binary.

        Search order:
          1. ``JUNO_CAPABILITY_HELPER`` env var (explicit override).
          2. ``PATH`` (installed layout).
          3. The Swift package build output under
             ``shells/macos/.build/{release,debug}/`` relative to the
             repo root. This is how `swift build` produces it during
             development; we fall back to it so the gate works on a
             fresh clone without the user having to run an install
             script first.

        Returns ``None`` if nothing is found so :meth:`probe` can
        surface a structured ``helper_not_installed`` report.
        """
        override = os.environ.get("JUNO_CAPABILITY_HELPER")
        if override and Path(override).is_file() and os.access(override, os.X_OK):
            return override

        on_path = shutil.which("juno-capability")
        if on_path:
            return on_path

        # Walk up from this file to find the repo root (marker: the
        # ``shells/macos`` directory). Limited depth so we don't crawl
        # the filesystem on a misconfigured install.
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            candidate_root = parent / "shells" / "macos"
            if candidate_root.is_dir():
                for flavor in ("release", "debug"):
                    binary = candidate_root / ".build" / flavor / "juno-capability"
                    if binary.is_file() and os.access(binary, os.X_OK):
                        return str(binary)
                break
        return None

    def probe(self) -> CapabilityReport:
        """Run the helper and return a parsed report.

        Failure modes surface as ``ok=False`` inside the report rather
        than raising - the caller wants a decision either way.
        """
        if not self._helper_path:
            return CapabilityReport(
                ok=False,
                has_ax_trust=False,
                frontmost_app_bundle_id=None,
                frontmost_app_name=None,
                frontmost_pid=None,
                window_title=None,
                focused_role=None,
                focused_subrole=None,
                focused_is_secure=False,
                error="helper_not_installed",
            )

        try:
            result = subprocess.run(
                [self._helper_path],
                capture_output=True,
                text=True,
                timeout=self._timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CapabilityReport(
                ok=False,
                has_ax_trust=False,
                frontmost_app_bundle_id=None,
                frontmost_app_name=None,
                frontmost_pid=None,
                window_title=None,
                focused_role=None,
                focused_subrole=None,
                focused_is_secure=False,
                error="helper_timeout",
            )
        except Exception as exc:
            return CapabilityReport(
                ok=False,
                has_ax_trust=False,
                frontmost_app_bundle_id=None,
                frontmost_app_name=None,
                frontmost_pid=None,
                window_title=None,
                focused_role=None,
                focused_subrole=None,
                focused_is_secure=False,
                error=f"helper_exec_failed: {exc}",
            )

        try:
            return CapabilityReport.from_json(result.stdout)
        except json.JSONDecodeError:
            return CapabilityReport(
                ok=False,
                has_ax_trust=False,
                frontmost_app_bundle_id=None,
                frontmost_app_name=None,
                frontmost_pid=None,
                window_title=None,
                focused_role=None,
                focused_subrole=None,
                focused_is_secure=False,
                error="helper_bad_output",
            )

    def decide(
        self,
        report: CapabilityReport | None = None,
        *,
        app_bundle_id: str | None = None,
        window_title: str | None = None,
    ) -> CapabilityDecision:
        # Run the probe (if no report supplied) and apply blocking rules.
        #
        # app_bundle_id/window_title let callers pass surface-side hints
        # (e.g. the macOS shell already knows the target app at hotkey
        # press). When provided, they override the probed report fields
        # before we run blocklist / secure-field checks.
        r = report or self.probe()
        if app_bundle_id or window_title:
            r = CapabilityReport(
                ok=r.ok,
                has_ax_trust=r.has_ax_trust,
                frontmost_app_bundle_id=app_bundle_id or r.frontmost_app_bundle_id,
                frontmost_app_name=r.frontmost_app_name,
                frontmost_pid=r.frontmost_pid,
                window_title=window_title or r.window_title,
                focused_role=r.focused_role,
                focused_subrole=r.focused_subrole,
                focused_is_secure=r.focused_is_secure,
                error=r.error,
                raw=r.raw,
            )

        if not r.has_ax_trust:
            return CapabilityDecision(
                ok=False,
                reason="ax_permission_missing",
                message=(
                    "Accessibility permission not granted. Open System "
                    "Settings → Privacy & Security → Accessibility and "
                    "enable JunoShell (and the helper binaries)."
                ),
                report=r,
            )

        if r.error == "helper_not_installed":
            return CapabilityDecision(
                ok=False,
                reason="helper_not_installed",
                message=(
                    "juno-capability helper not found. Build it with "
                    "`swift build -c release --package-path shells/macos` "
                    "or set JUNO_CAPABILITY_HELPER to an absolute path."
                ),
                report=r,
            )

        if r.error == "helper_timeout":
            return CapabilityDecision(
                ok=False,
                reason="helper_timeout",
                message="Accessibility probe timed out.",
                report=r,
            )

        if r.focused_is_secure:
            return CapabilityDecision(
                ok=False,
                reason="secure_field",
                message=(
                    "The focused field is a secure-text input "
                    "(password / PIN). Dictation is blocked for safety."
                ),
                report=r,
            )

        wt = r.window_title or ""
        if wt:
            for pattern in self._suppression_config.blocklist_window_title_patterns:
                if pattern.search(wt):
                    return CapabilityDecision(
                        ok=False,
                        reason="window_title_blocked",
                        message="Window title matches a blocked pattern.",
                        report=r,
                    )

        bid = (r.frontmost_app_bundle_id or "").lower()
        if bid and bid in self._blocklist:
            return CapabilityDecision(
                ok=False,
                reason="app_blocked",
                message=(
                    f"App {r.frontmost_app_name or bid} is on the "
                    f"managed-app blocklist. Dictation is disabled."
                ),
                report=r,
            )

        paste_roles = frozenset(
            {
                "AXTextField",
                "AXTextArea",
                "AXComboBox",
                "AXSearchField",
                "AXWebArea",
            }
        )
        role = (r.focused_role or "").strip()
        if not role or role not in paste_roles:
            return CapabilityDecision(
                ok=False,
                reason="no_text_focus",
                message=(
                    "No editable text field appears focused. "
                    "Dictation will continue; text will be offered to copy instead of pasting."
                ),
                report=r,
            )
        warn = bool(bid and bid in self._suppression_config.warnlist_bundle_ids)
        return CapabilityDecision(
            ok=True,
            reason="allowed",
            message="ok",
            report=r,
            warn=warn,
        )


__all__ = [
    "CapabilityChecker",
    "CapabilityDecision",
    "CapabilityReport",
    "load_default_blocklist",
]
