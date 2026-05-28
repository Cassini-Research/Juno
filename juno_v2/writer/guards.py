from __future__ import annotations

from juno_v2.writer.deterministic import AppCategory, apply_app_formatting


def is_no_touch_surface(app_category: str | None) -> bool:
    return (app_category or "").strip().lower() in {"code", "terminal"}


__all__ = ["AppCategory", "apply_app_formatting", "is_no_touch_surface"]

