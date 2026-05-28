from __future__ import annotations

from enum import Enum

from juno_core_v3.contracts.resource_hints import HostResourceHints


class SupportLevel(str, Enum):
    """Surface support policy."""

    GOLD = "gold"
    YELLOW = "yellow"
    RED = "red"


class SurfaceId(str, Enum):
    """Known surfaces for Phase 1 gating (extend as shells land)."""

    WORKBENCH_DEV = "workbench_dev"
    TERMINAL_TEST = "terminal_test"
    MAC_OVERLAY = "mac_overlay"
    TRANSFORM_SHORTCUT = "transform_shortcut"
    IPHONE_APP = "iphone_app"
    IPHONE_KEYBOARD = "iphone_keyboard"
    UNKNOWN = "unknown"


class SurfaceCapabilityGate:
    """Maps surface to support level."""

    _DEFAULT_MAP: dict[str, SupportLevel] = {
        SurfaceId.WORKBENCH_DEV.value: SupportLevel.GOLD,
        SurfaceId.TERMINAL_TEST.value: SupportLevel.GOLD,
        SurfaceId.MAC_OVERLAY.value: SupportLevel.GOLD,
        SurfaceId.TRANSFORM_SHORTCUT.value: SupportLevel.GOLD,
        SurfaceId.IPHONE_APP.value: SupportLevel.YELLOW,
        SurfaceId.IPHONE_KEYBOARD.value: SupportLevel.RED,
        SurfaceId.UNKNOWN.value: SupportLevel.RED,
    }

    def __init__(self, overrides: dict[str, SupportLevel] | None = None) -> None:
        self._map = dict(self._DEFAULT_MAP)
        if overrides:
            self._map.update({k: v for k, v in overrides.items()})

    def level_for(self, surface_id: str) -> SupportLevel:
        return self._map.get(surface_id, SupportLevel.RED)

    def allows_dictation(self, surface_id: str) -> bool:
        return self.level_for(surface_id) != SupportLevel.RED

    def policy_map(self) -> dict[str, str]:
        """Return the full surface → level map as plain strings for the API."""
        return {k: v.value for k, v in self._map.items()}

    def effective_level(self, surface_id: str, hints: HostResourceHints | None) -> SupportLevel:
        """Apply coarse resource hints on top of the static surface map (Gold/Yellow/Red)."""
        base = self.level_for(surface_id)
        if hints is None:
            return base
        if hints.memory_pressure == "critical" or hints.thermal_pressure == "critical":
            if base == SupportLevel.GOLD:
                return SupportLevel.YELLOW
            if base == SupportLevel.YELLOW:
                return SupportLevel.RED
            return base
        if hints.memory_pressure == "warning" and hints.battery_low and base == SupportLevel.GOLD:
            return SupportLevel.YELLOW
        return base
