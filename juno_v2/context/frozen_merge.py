"""Merge client-frozen Accessibility snapshots into :class:`TypedContextBundle`.

The macOS shell captures ``juno-capability`` JSON at hotkey press and sends it
with ``ingest_wav``. Those fields override the server-side context provider
snapshot for the utterance so the writer sees the same selection the user had
when they started speaking.
"""

from __future__ import annotations

from typing import Any, Mapping

from juno_v2.context.app_classifier import classify_app_category
from juno_v2.context.redaction import ContextRedactor
from juno_v2.contracts.context import TypedContextBundle


def merge_frozen_capability_into_bundle(
    context: TypedContextBundle,
    frozen: Mapping[str, Any] | None,
    *,
    max_field_chars: int = 240,
) -> bool:
    """Apply ``frozen`` (juno-capability-shaped dict) into ``context``.

    Returns ``True`` if any frozen key was applied (caller may trace).
    """
    if frozen is None:
        return False
    redactor = ContextRedactor()
    applied = False

    def _clip(s: str) -> str:
        return s[:max_field_chars]

    def _red(s: str) -> str:
        t, _ = redactor.redact(_clip(s))
        return t

    def _str(key: str) -> str:
        v = frozen.get(key)
        if v is None:
            return ""
        return str(v)

    # Text fields — frozen always wins when the client sent a frozen payload
    # (caller only passes non-None when multipart included frozen_context).
    sel = _red(_str("selected_text"))
    if "selected_text" in frozen:
        context.selected_text = sel
        applied = True
    fb = _red(_str("focused_text_before") or _str("surrounding_text_before"))
    if "focused_text_before" in frozen or "surrounding_text_before" in frozen:
        context.focused_text_before = fb
        applied = True
    fa = _red(_str("focused_text_after") or _str("surrounding_text_after"))
    if "focused_text_after" in frozen or "surrounding_text_after" in frozen:
        context.focused_text_after = fa
        applied = True
    clip = _red(_str("clipboard_text"))
    if "clipboard_text" in frozen:
        context.clipboard_text = clip
        applied = True
    field_excerpt = _red(_str("field_text_excerpt"))
    if "field_text_excerpt" in frozen:
        context.field_text_excerpt = field_excerpt
        applied = True

    bid = (_str("frontmost_app_bundle_id") or _str("app_bundle_id")).strip()
    if bid:
        context.metadata.setdefault("app_bundle_id", bid)
        context.metadata["app_bundle_id"] = bid
        applied = True
    name = (_str("frontmost_app_name") or _str("app_name")).strip()
    if name:
        context.app_name = name
        applied = True
    title = _str("window_title").strip()
    if title:
        context.window_title = title
        applied = True

    if "focused_is_secure" in frozen or "focused_secure" in frozen:
        sec = bool(frozen.get("focused_is_secure") or frozen.get("focused_secure"))
        context.metadata["focused_secure"] = sec
        applied = True

    loc = _str("locale_identifier").strip()
    if loc:
        context.metadata["locale_identifier"] = loc
        applied = True

    doc = (_str("focused_document_path") or _str("focused_file_path")).strip()
    if doc:
        context.focused_file_path = doc
        applied = True

    raw_entities = frozen.get("candidate_entities")
    if isinstance(raw_entities, list):
        merged = list(context.candidate_entities or [])
        seen = {item.casefold() for item in merged if isinstance(item, str)}
        for raw in raw_entities[:24]:
            value = _clip(str(raw or "").strip())
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
        if merged != list(context.candidate_entities or []):
            context.candidate_entities = merged[:40]
            applied = True

    raw_explicit_entities = frozen.get("explicit_candidate_entities")
    if isinstance(raw_explicit_entities, list):
        explicit_entities = list(context.metadata.get("explicit_candidate_entities") or [])
        explicit_seen = {item.casefold() for item in explicit_entities if isinstance(item, str)}
        for raw in raw_explicit_entities[:24]:
            value = _clip(str(raw or "").strip())
            if not value:
                continue
            key = value.casefold()
            if key in explicit_seen:
                continue
            explicit_seen.add(key)
            explicit_entities.append(value)
        context.metadata["explicit_candidate_entities"] = explicit_entities[:40]
        applied = True

    if applied and (context.app_name or context.metadata.get("app_bundle_id")):
        context.app_category = classify_app_category(
            context.app_name,
            context.window_title,
            app_bundle_id=context.metadata.get("app_bundle_id"),
        )

    return applied
