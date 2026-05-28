"""Map frontmost-surface signals to an editing-style label for macOS HUD.

The classifier taxonomy lives in :mod:`juno_v2.context.app_classifier`.
Shells POST a juno-capability-shaped JSON blob; we return ``app_category``
plus a short user-facing ``editing_style`` string aligned with product copy.
"""

from __future__ import annotations

from typing import Any, Dict

from juno_v2.context.app_classifier import classify_app_category

__all__ = ["editing_style_for_category", "surface_editing_profile"]


def editing_style_for_category(app_category: str) -> str:
    """User-facing cleanup posture label (three primary buckets + fallbacks)."""
    cat = (app_category or "unknown").strip().lower()
    if cat == "messaging":
        return "Minimal cleanup"
    if cat in ("code", "terminal"):
        return "Conservative"
    if cat in ("docs", "email", "forms", "meeting"):
        return "Punctuation focus"
    if cat == "unknown":
        return "Balanced"
    return "Balanced"


def surface_editing_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build ``{ok, app_category, editing_style, ...}`` from a loose JSON dict.

    Accepts keys emitted by ``juno-capability`` (``app_name``,
    ``window_title``, ``app_bundle_id`` / ``frontmost_app_bundle_id``,
    optional ``focused_role`` / ``focused_subrole`` for forward
    compatibility).
    """
    app_name = payload.get("app_name") or payload.get("frontmost_app_name")
    if isinstance(app_name, str):
        app_name = app_name.strip() or None
    else:
        app_name = None

    window_title = payload.get("window_title")
    if isinstance(window_title, str):
        window_title = window_title.strip() or None
    else:
        window_title = None

    bundle = (
        payload.get("app_bundle_id")
        or payload.get("frontmost_app_bundle_id")
    )
    if isinstance(bundle, str):
        bundle = bundle.strip() or None
    else:
        bundle = None

    category = classify_app_category(app_name, window_title, app_bundle_id=bundle)
    style = editing_style_for_category(category)

    focused_role = payload.get("focused_role")
    focused_subrole = payload.get("focused_subrole")
    if focused_role is not None and not isinstance(focused_role, str):
        focused_role = str(focused_role)
    if focused_subrole is not None and not isinstance(focused_subrole, str):
        focused_subrole = str(focused_subrole)

    return {
        "ok": True,
        "app_category": category,
        "editing_style": style,
        "app_name": app_name,
        "window_title": window_title,
        "app_bundle_id": bundle,
        "focused_role": focused_role,
        "focused_subrole": focused_subrole,
    }
