from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass(slots=True)
class RedactionSummary:
    emails: int = 0
    urls: int = 0
    digit_sequences: int = 0
    secrets: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class TypedContextBundle:
    app_name: str | None = None
    window_title: str | None = None
    selected_text: str = ""
    focused_text_before: str = ""
    focused_text_after: str = ""
    clipboard_text: str = ""
    field_text_excerpt: str = ""
    candidate_entities: List[str] = field(default_factory=list)
    redaction: RedactionSummary = field(default_factory=RedactionSummary)
    # Coarse presentation category of the focused surface
    # (``messaging``, ``email``, ``docs``, ``code``, ``terminal``,
    # ``forms``, ``unknown``). Writers, transforms, and tools read this
    # to pick typographic policy (e.g. don't massage whitespace in
    # code). Populated by the context provider via
    # :func:`juno_v2.context.app_classifier.classify_app_category`;
    # surfaces can override by writing the value directly.
    app_category: str | None = None
    # Recent clipboard history (newest first), each entry is a dict
    # ``{"text": str, "ts_unix_ms": int, "redacted": bool}``. Populated
    # by :func:`juno_v2.context.clipboard_enrichment.inject_clipboard_ring`.
    # Kept as a plain list of dicts (not a typed dataclass) so callers
    # that ``to_dict()`` the bundle get a JSON-safe payload without
    # extra conversion, and legacy consumers reading
    # ``metadata["recent_clipboard"]`` keep working.
    recent_clipboard: List[Dict[str, Any]] = field(default_factory=list)
    # IDE / editor document context. Populated only when the focused
    # surface is a code-class app (see
    # :mod:`juno_v2.context.ide_symbol`):
    #
    # * ``focused_file_path``: absolute path (or URL) of the document
    #   the user is editing. Pulled from ``AXDocument`` by the Swift
    #   ``juno-capability`` helper, with a window-title heuristic
    #   fallback for apps that don't expose ``AXDocument``.
    # * ``symbol_under_cursor``: the identifier that straddles the
    #   caret, reconstructed from ``focused_text_before`` +
    #   ``focused_text_after``. Used by the recognition bias plan so
    #   the ASR is nudged to recognize domain-specific identifiers
    #   ("``transcribe_wav``", "``CapabilityChecker``") correctly.
    #
    # Both fields are ``None`` when unavailable; the writer must
    # treat them as advisory hints and never required.
    focused_file_path: str | None = None
    symbol_under_cursor: str | None = None
    # Arbitrary surface metadata. ITN reads (see :mod:`juno_v2.itn.format_policy`):
    # ``locale_identifier`` (e.g. ``en_GB`` from the macOS helper) and/or
    # ``itn_format`` explicit ``{"date_style","clock","currency_decimal"}``.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['redaction'] = self.redaction.to_dict()
        return data


@dataclass(slots=True)
class RecognitionBiasPlan:
    utterance_id: str
    context: TypedContextBundle
    bias_phrases: List[str] = field(default_factory=list)
    initial_prompt: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'utterance_id': self.utterance_id,
            'context': self.context.to_dict(),
            'bias_phrases': list(self.bias_phrases),
            'initial_prompt': self.initial_prompt,
            'metadata': dict(self.metadata),
        }


@runtime_checkable
class ContextWindowSource(Protocol):
    """Structural interface satisfied by any object that can provide a
    context window dict — notably :class:`~juno_v2.workbench.store.WorkbenchStore`.

    Defined here (contracts layer) so that :mod:`juno_v2.context.provider`
    can type its ``store`` field without importing from the higher-level
    ``workbench`` package, breaking the context↔workbench mutual dependency.
    """

    def context_window(self, *, max_field_chars: int) -> dict: ...
