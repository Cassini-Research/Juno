from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Pressure = Literal["unknown", "nominal", "warning", "critical"]


@dataclass(slots=True)
class HostResourceHints:
    """Optional signals from the host OS / shell (no heavy polling in-core).

    Shells should map platform APIs (e.g. ProcessInfo.thermalState, NSProcessInfo
    power state) into these coarse buckets so the broker can degrade safely without
    embedding Apple-specific code here.
    """

    memory_pressure: Pressure = "unknown"
    thermal_pressure: Pressure = "unknown"
    battery_low: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_pressure": self.memory_pressure,
            "thermal_pressure": self.thermal_pressure,
            "battery_low": self.battery_low,
        }

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> HostResourceHints | None:
        if not d:
            return None
        mem = d.get("memory_pressure")
        th = d.get("thermal_pressure")
        bl = bool(d.get("battery_low", False))
        if mem not in ("unknown", "nominal", "warning", "critical"):
            mem = "unknown"
        if th not in ("unknown", "nominal", "warning", "critical"):
            th = "unknown"
        return HostResourceHints(memory_pressure=mem, thermal_pressure=th, battery_low=bl)

    @classmethod
    def from_env(cls) -> HostResourceHints | None:
        """Test/dev hook: JUNO_MEMORY_PRESSURE, JUNO_THERMAL_PRESSURE, JUNO_BATTERY_LOW."""
        mem = os.environ.get("JUNO_MEMORY_PRESSURE")
        th = os.environ.get("JUNO_THERMAL_PRESSURE")
        if mem is None and th is None and "JUNO_BATTERY_LOW" not in os.environ:
            return None
        return cls(
            memory_pressure=cls._coerce_pressure(mem),
            thermal_pressure=cls._coerce_pressure(th),
            battery_low=os.environ.get("JUNO_BATTERY_LOW", "").lower() in ("1", "true", "yes"),
        )

    @classmethod
    def from_system(
        cls, *, timeout_sec: float = 0.8
    ) -> HostResourceHints | None:
        """Probe the real OS for thermal / battery / memory pressure.

        On macOS this shells out to the ``juno-host`` Swift helper which
        wraps ``ProcessInfo.thermalState``, ``IOPSCopyPowerSourcesInfo``,
        and ``host_statistics64(HOST_VM_INFO64)`` — none of which have a
        clean Python binding. On other platforms (or when the helper is
        missing) returns ``None`` so callers fall through to
        :meth:`from_env` or leave hints unset.

        Failure modes (missing helper, timeout, bad JSON) all return
        ``None`` rather than raising: resource hints are advisory. A
        broker that can't read them should keep running in its default
        (unconstrained) policy, not refuse dictation.
        """
        helper = cls._resolve_host_helper()
        if helper is None:
            return None
        try:
            proc = subprocess.run(  # noqa: S603
                [helper],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return cls.from_dict(payload)

    @classmethod
    def resolve(cls, *, timeout_sec: float = 0.8) -> HostResourceHints | None:
        """Combined resolver with env > system > None precedence.

        Env wins because it's the override tests and ops use to force
        a specific degradation path. System is the live probe. When
        neither produces a value we return ``None``, leaving the
        broker unconstrained (documented default behaviour).
        """
        env_hints = cls.from_env()
        if env_hints is not None:
            return env_hints
        return cls.from_system(timeout_sec=timeout_sec)

    # ---- helpers ---------------------------------------------------

    @staticmethod
    def _coerce_pressure(raw: str | None) -> Pressure:
        if raw is None:
            return "unknown"
        v = raw.lower().strip()
        if v in ("unknown", "nominal", "warning", "critical"):
            return v  # type: ignore[return-value]
        return "unknown"

    @staticmethod
    def _resolve_host_helper() -> str | None:
        """Locate the ``juno-host`` Swift helper.

        Resolution mirrors :class:`CapabilityChecker._resolve_helper` so
        ops only need to keep one set of binaries on ``PATH``:

        1. Explicit override ``JUNO_HOST_HELPER`` (absolute path).
        2. ``juno-host`` on ``PATH``.
        3. ``shells/macos/.build/{release,debug,arm64-apple-macosx/{release,debug}}/juno-host``
           relative to the repo root, which is where ``swift build``
           drops the artifact during dev.

        Returns ``None`` on non-darwin platforms even if a binary is
        somehow on ``PATH`` — the helper uses ``IOKit`` and
        ``host_statistics64``; running it under Linux would return
        noise, and the broker would rather have no hint than a wrong
        one.
        """
        override = os.environ.get("JUNO_HOST_HELPER")
        if override and Path(override).is_file() and os.access(override, os.X_OK):
            return override
        if sys.platform != "darwin":
            return None
        on_path = shutil.which("juno-host")
        if on_path:
            return on_path
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            root = parent / "shells" / "macos" / ".build"
            if root.is_dir():
                for variant in (
                    "release",
                    "debug",
                    "arm64-apple-macosx/release",
                    "arm64-apple-macosx/debug",
                    "x86_64-apple-macosx/release",
                    "x86_64-apple-macosx/debug",
                ):
                    candidate = root / variant / "juno-host"
                    if candidate.is_file() and os.access(candidate, os.X_OK):
                        return str(candidate)
                break
        return None
