"""Attach recent clipboard entries to a :class:`TypedContextBundle`.

The workbench keeps a single
:class:`~juno_core_v3.context.clipboard_ring.ClipboardRingBuffer`
per process. Both the one-shot dictation pipeline and the streaming
:class:`~juno_v2.engine.session.DictationSessionRunner` need to
expose this ring to the writer / bias / tools layers without those
layers depending on the ring implementation.

This module provides a single function,
:func:`inject_clipboard_ring`, that reads the ring and stamps the
resulting entries onto a bundle. We populate two places for
compatibility:

1. ``bundle.recent_clipboard`` — first-class field (preferred).
2. ``bundle.metadata["recent_clipboard"]`` — legacy mirror so older
   consumers (writer service context payload, broker tools) keep
   working without a migration.

The function is best-effort: exceptions from the ring are swallowed
and the bundle is returned unchanged. That matches the streaming
session's existing contract that context enrichment must never block
ASR.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from juno_v2.contracts.context import TypedContextBundle


class _RingLike(Protocol):
    def recent(self, limit: int = ...) -> Iterable[Any]: ...


def inject_clipboard_ring(
    bundle: TypedContextBundle,
    ring: _RingLike | None,
    *,
    limit: int = 5,
) -> TypedContextBundle:
    """Attach up to ``limit`` recent clipboard entries to ``bundle``.

    Returns the same bundle for chaining. When ``ring`` is ``None``
    or raises, the bundle is returned untouched.
    """
    if ring is None:
        return bundle
    try:
        entries = ring.recent(limit=limit) or []
    except Exception:
        return bundle

    payload: list[dict[str, Any]] = []
    for entry in entries:
        text = getattr(entry, "text", None)
        if text is None and isinstance(entry, dict):
            text = entry.get("text")
        ts = getattr(entry, "ts_unix_ms", None)
        if ts is None and isinstance(entry, dict):
            ts = entry.get("ts_unix_ms")
        redacted = getattr(entry, "redacted", None)
        if redacted is None and isinstance(entry, dict):
            redacted = entry.get("redacted", False)
        if text is None:
            continue
        payload.append(
            {
                "text": str(text),
                "ts_unix_ms": int(ts) if ts is not None else 0,
                "redacted": bool(redacted),
            }
        )

    if not payload:
        return bundle

    bundle.recent_clipboard = payload
    bundle.metadata["recent_clipboard"] = payload
    return bundle


__all__ = ["inject_clipboard_ring"]
