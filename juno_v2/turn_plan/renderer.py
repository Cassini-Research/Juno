from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.list_content import protect_list_render
from juno_v2.memory.store import JsonMemoryStore


@dataclass(slots=True)
class RenderResult:
    text: str
    rendered: bool
    reason: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


def render_turn_plan(
    plan: dict[str, Any],
    *,
    context: TypedContextBundle,
    memory_store: JsonMemoryStore | None = None,
) -> RenderResult:
    snippet = _render_snippet_only(plan, memory_store=memory_store, app_category=context.app_category)
    if snippet is not None:
        return snippet

    transform = plan.get("transform") if isinstance(plan.get("transform"), dict) else {}
    transformed = transform.get("transformed_text") if isinstance(transform, dict) else None
    if isinstance(transformed, str) and transformed.strip():
        return RenderResult(text=_guard_markdown(transformed.strip(), plan), rendered=True, metadata={"source": "transform"})

    render = plan.get("render_plan") if isinstance(plan.get("render_plan"), dict) else {}
    kind = str(render.get("render_kind") or "plain").strip()
    if kind == "none":
        return RenderResult(text="", rendered=True, reason="render_none")

    units = _content_units(render)
    if not units:
        corrected = plan.get("corrected_transcript") if isinstance(plan.get("corrected_transcript"), dict) else {}
        text = str(corrected.get("text") or "").strip()
        return RenderResult(text=_guard_markdown(text, plan), rendered=bool(text), reason="corrected_text_fallback")

    if kind in {"bulleted_list", "checklist"}:
        lines = []
        prefix = "[ ] " if kind == "checklist" and not bool(render.get("markdown_allowed")) else "- "
        if kind == "checklist" and bool(render.get("markdown_allowed")):
            prefix = "- [ ] "
        for unit in units:
            text = _clean_item(unit["text"])
            if text:
                lines.append(prefix + text)
        return _protected_list_result("\n".join(lines), plan=plan, render=render)

    if kind == "numbered_list":
        lines = []
        for idx, unit in enumerate(units, start=1):
            text = _clean_item(unit["text"])
            if text:
                lines.append(f"{idx}. {text}")
        return _protected_list_result("\n".join(lines), plan=plan, render=render)

    if kind == "table":
        text = _render_table(units, markdown_allowed=bool(render.get("markdown_allowed")))
        return RenderResult(text=_guard_markdown(text, plan), rendered=bool(text), metadata=_render_meta(render))

    if kind in {"code", "terminal"}:
        text = "\n".join(str(unit["text"]) for unit in units if str(unit["text"]))
        return RenderResult(text=text.strip("\n"), rendered=bool(text), metadata=_render_meta(render))

    if kind in {"email", "message", "note", "ai_prompt", "paragraphs", "plain"}:
        text = _render_blocks(units, compact=kind == "message")
        return RenderResult(text=_guard_markdown(text, plan), rendered=bool(text), metadata=_render_meta(render))

    text = _render_blocks(units, compact=False)
    return RenderResult(text=_guard_markdown(text, plan), rendered=bool(text), metadata=_render_meta(render))


def _content_units(render: dict[str, Any]) -> list[dict[str, Any]]:
    raw = render.get("content_units")
    if not isinstance(raw, list):
        return []
    units: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        if not text.strip():
            continue
        order_raw = item.get("order")
        try:
            order = int(order_raw)
        except (TypeError, ValueError):
            order = idx + 1
        units.append({
            "kind": str(item.get("kind") or "paragraph"),
            "text": text.strip(),
            "order": order,
        })
    return sorted(units, key=lambda x: x["order"])


def _render_blocks(units: list[dict[str, Any]], *, compact: bool) -> str:
    blocks: list[str] = []
    for unit in units:
        kind = str(unit.get("kind") or "")
        text = str(unit.get("text") or "").strip()
        if not text:
            continue
        if kind == "heading":
            blocks.append(text.rstrip(":") + ":")
        elif kind == "item":
            blocks.append("- " + _clean_item(text))
        else:
            blocks.append(text)
    sep = "\n" if compact else "\n\n"
    return sep.join(blocks).strip()


def _render_table(units: list[dict[str, Any]], *, markdown_allowed: bool) -> str:
    rows: list[list[str]] = []
    for unit in units:
        text = str(unit.get("text") or "").strip()
        if not text:
            continue
        if "|" in text:
            cells = [c.strip() for c in text.strip("|").split("|")]
        elif "\t" in text:
            cells = [c.strip() for c in text.split("\t")]
        else:
            cells = [text]
        rows.append(cells)
    if not rows:
        return ""
    if not markdown_allowed:
        return "\n".join("\t".join(row) for row in rows)
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    out = ["| " + " | ".join(padded[0]) + " |"]
    out.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in padded[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _clean_item(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip(" -•\t").rstrip()


def _guard_markdown(text: str, plan: dict[str, Any]) -> str:
    render = plan.get("render_plan") if isinstance(plan.get("render_plan"), dict) else {}
    if bool(render.get("markdown_allowed", False)):
        return text
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", text or "")
    out = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", out)
    return out


def _render_meta(render: dict[str, Any]) -> dict[str, Any]:
    return {
        "render_kind": render.get("render_kind"),
        "claimed_item_count": render.get("claimed_item_count"),
        "spoken_item_count": render.get("spoken_item_count"),
        "markdown_allowed": bool(render.get("markdown_allowed", False)),
    }


def _protected_list_result(
    text: str,
    *,
    plan: dict[str, Any],
    render: dict[str, Any],
) -> RenderResult:
    corrected = plan.get("corrected_transcript")
    corrected_text = str(corrected.get("text") or "").strip() if isinstance(corrected, dict) else ""
    protected = protect_list_render(corrected_text, _guard_markdown(text, plan))
    metadata = {**_render_meta(render), "content_preservation": protected.mode}
    reason = "list_content_fallback" if protected.mode == "complete_transcript_fallback" else "ok"
    return RenderResult(
        text=protected.text,
        rendered=bool(protected.text.strip()),
        reason=reason,
        metadata=metadata,
    )


def _render_snippet_only(
    plan: dict[str, Any],
    *,
    memory_store: JsonMemoryStore | None,
    app_category: str | None,
) -> RenderResult | None:
    snippets = plan.get("snippets") if isinstance(plan.get("snippets"), list) else []
    if not snippets or memory_store is None or getattr(memory_store, "snippets", None) is None:
        return None
    render = plan.get("render_plan") if isinstance(plan.get("render_plan"), dict) else {}
    units = render.get("content_units")
    if isinstance(units, list) and units:
        return None
    scope = (app_category or "global").strip().lower() or "global"
    for item in snippets:
        if not isinstance(item, dict):
            continue
        if str(item.get("operation") or "").strip() not in {"insert", "use"}:
            continue
        trigger = str(item.get("trigger") or "").strip()
        if not trigger:
            continue
        for candidate_scope in (scope, "global"):
            snippet = memory_store.snippets.resolve(trigger, scope=candidate_scope)
            body = str(getattr(snippet, "body", "") or "") if snippet is not None else ""
            if body:
                return RenderResult(
                    text=body,
                    rendered=True,
                    reason="snippet_insert",
                    metadata={
                        "snippet_expanded": True,
                        "trigger": trigger,
                        "scope": candidate_scope,
                        "body_chars": len(body),
                    },
                )
    return RenderResult(text="", rendered=False, reason="snippet_not_found")
