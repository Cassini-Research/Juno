from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from juno_core_v3.contracts.context_packet import (
    ContextPacket,
    ContextPacketBudgets,
    FieldProvenance,
    build_context_packet_from_typed_bundle,
)
from juno_core_v3.context.clipboard_ring import ClipboardRingBuffer
from juno_core_v3.context.screen_fallback import ScreenFallbackPolicy

from juno_v2.contracts.context import TypedContextBundle


class ContextSuppressionClass(str, Enum):
    NONE = "none"
    SECURE_FIELD = "secure_field"
    SENSITIVE_APP = "sensitive_app"


class ContextDegradationClass(str, Enum):
    NONE = "none"
    REMOTE_DESKTOP = "remote_desktop"
    UNREADABLE_FIELD = "unreadable_field"


@dataclass(slots=True)
class ContextPlaneConfig:
    budgets: ContextPacketBudgets = field(default_factory=ContextPacketBudgets)
    screen_fallback_policy: ScreenFallbackPolicy = ScreenFallbackPolicy.DISABLED

    # Conservative defaults for v3: never enable screen fallback without explicit config.
    allow_screen_fallback: bool = False
    clipboard_ring: ClipboardRingBuffer | None = field(default=None)


def _looks_like_remote_desktop(app_name: str | None, window_title: str | None) -> bool:
    text = f"{app_name or ''} {window_title or ''}".casefold()
    return any(
        token in text
        for token in (
            "citrix",
            "remote desktop",
            "microsoft remote desktop",
            "vmware",
            "horizon",
            "vnc",
            "teamviewer",
            "anydesk",
        )
    )


def _looks_sensitive_app(app_name: str | None, window_title: str | None) -> bool:
    text = f"{app_name or ''} {window_title or ''}".casefold()
    return any(
        token in text
        for token in (
            "1password",
            "keychain",
            "passwords",
            "bank",
            "wallet",
            "authenticator",
            "okta",
        )
    )


class ContextPlane:
    """Context plane contract builder (Phase 3).

    Implements:
    - ordered-source bundling (using v2 TypedContextBundle as the current input)
    - provenance + budgets
    - suppression rules
    - degradation rules
    - screen fallback policy contract (not implemented)
    """

    def __init__(self, config: ContextPlaneConfig | None = None) -> None:
        self.config = config or ContextPlaneConfig()

    def build_from_typed_bundle(
        self,
        bundle: TypedContextBundle,
        *,
        default_provenance: FieldProvenance = FieldProvenance.WORKBENCH_SYNC,
        surface_id: str = "workbench_dev",
    ) -> ContextPacket:
        pkt = build_context_packet_from_typed_bundle(bundle, default_provenance=default_provenance)
        pkt.metadata = dict(pkt.metadata)
        pkt.metadata["surface_id"] = surface_id
        pkt.metadata["ordered_sources"] = [
            "selected_text",
            "focused_text_before",
            "focused_text_after",
            "field_text_excerpt",
            "app_name",
            "window_title",
            "clipboard_text",
            "candidate_entities",
        ]

        # 1) Degradation classification (may adjust budgets and disable screen fallback).
        degradation = ContextDegradationClass.NONE
        budgets = self.config.budgets
        if _looks_like_remote_desktop(pkt.app_name, pkt.window_title):
            degradation = ContextDegradationClass.REMOTE_DESKTOP
            budgets = ContextPacketBudgets(
                max_total_chars=min(budgets.max_total_chars, 12_000),
                max_selected_chars=min(budgets.max_selected_chars, 4_000),
                max_clipboard_chars=0,
                max_around_chars=min(budgets.max_around_chars, 4_000),
                max_field_excerpt_chars=min(budgets.max_field_excerpt_chars, 2_000),
            )
            pkt.metadata["screen_fallback_allowed"] = False

        # 2) Suppression classification.
        suppression = ContextSuppressionClass.NONE
        if getattr(bundle, "redaction", None) is not None and getattr(bundle.redaction, "secrets", 0) > 0:
            suppression = ContextSuppressionClass.SECURE_FIELD
        elif _looks_sensitive_app(pkt.app_name, pkt.window_title):
            suppression = ContextSuppressionClass.SENSITIVE_APP

        if suppression != ContextSuppressionClass.NONE:
            pkt.metadata["suppression"] = suppression.value
            pkt.metadata["suppressed_fields"] = [
                "selected_text",
                "focused_text_before",
                "focused_text_after",
                "clipboard_text",
                "field_text_excerpt",
            ]
            pkt.selected_text = ""
            pkt.focused_text_before = ""
            pkt.focused_text_after = ""
            pkt.clipboard_text = ""
            pkt.field_text_excerpt = ""

        pkt.metadata["degradation"] = degradation.value

        # 3) Budgets enforcement.
        pkt = pkt.enforce_budgets(budgets)
        pkt.metadata = dict(pkt.metadata)
        if self.config.clipboard_ring is not None and suppression == ContextSuppressionClass.NONE:
            recent = self.config.clipboard_ring.recent(limit=5)
            pkt.metadata["recent_clipboard"] = [e.text for e in recent]
            pkt.metadata["recent_clipboard_redacted"] = [e.redacted for e in recent]
        pkt.metadata["budgets"] = {
            "max_total_chars": budgets.max_total_chars,
            "max_selected_chars": budgets.max_selected_chars,
            "max_clipboard_chars": budgets.max_clipboard_chars,
            "max_around_chars": budgets.max_around_chars,
            "max_field_excerpt_chars": budgets.max_field_excerpt_chars,
        }

        # 4) Screen fallback contract (Phase 3: explicit disabled unless enabled).
        if not self.config.allow_screen_fallback:
            pkt.metadata["screen_fallback_policy"] = ScreenFallbackPolicy.DISABLED.value
            pkt.metadata.setdefault("screen_fallback_allowed", False)
        else:
            pkt.metadata["screen_fallback_policy"] = self.config.screen_fallback_policy.value
            pkt.metadata.setdefault(
                "screen_fallback_allowed",
                self.config.screen_fallback_policy != ScreenFallbackPolicy.DISABLED,
            )

        return pkt
