"""Per-app surface presets: bundle-id keyed defaults for writer mode + ASR hints.

Persisted JSON lives next to custom modes under the workbench log root
(see :func:`default_surface_presets_path`). Shipped defaults are merged in
memory and overridden by user entries with the same ``bundle_id``.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from juno_v2.contracts.modes import CustomModeRecord, ModePolicy, ModeSelection, ModeSource
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.modes.policy import apply_custom_overrides, mode_policy_for, resolve_mode_selection

if TYPE_CHECKING:
    from juno_v2.modes.store import CustomModeStore


def default_surface_presets_path(log_dir: Path | str) -> Path:
    return Path(log_dir) / "juno_workbench_data" / "surface_presets.json"


@dataclass(slots=True)
class SurfacePresetRecord:
    """One preset row (JSON-serializable)."""

    id: str
    bundle_id: str
    fallback_app_category: str | None = None
    default_built_in_mode: str | None = None
    custom_mode_name: str | None = None
    asr_addon: str = ""
    writer_tone_addon: str = ""
    include_window_title_in_asr: bool = False
    lock_mode: bool = False
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurfacePresetRecord:
        return cls(
            id=str(data.get("id", "")),
            bundle_id=str(data.get("bundle_id", "")),
            fallback_app_category=data.get("fallback_app_category"),
            custom_mode_name=data.get("custom_mode_name"),
            default_built_in_mode=data.get("default_built_in_mode"),
            asr_addon=str(data.get("asr_addon", "")),
            writer_tone_addon=str(data.get("writer_tone_addon", "")),
            include_window_title_in_asr=bool(data.get("include_window_title_in_asr", False)),
            lock_mode=bool(data.get("lock_mode", False)),
            enabled=bool(data.get("enabled", True)),
        )


_CODE_ASR_ADDON = (
    "User is dictating into a code editor; preserve identifiers, file paths, "
    "code keywords, and case as spoken."
)
_CODE_WRITER_TONE = (
    "Code editor: prefer snake_case/camelCase per surrounding code; "
    "preserve symbols literally; no prose massaging."
)
_CHAT_ASR_ADDON = (
    "User is composing in a messaging surface; preserve natural wording, "
    "abbreviations, contractions, fragments, and conversational rhythm."
)
_CHAT_WRITER_TONE = (
    "Messaging: correct only clear transcription errors and explicit self-corrections. "
    "Do not formalize, expand, summarize, improve vocabulary, or add a terminal period "
    "unless necessary or explicitly dictated."
)
_DOCS_ASR_ADDON = (
    "User is dictating into a notes/docs surface; allow headings and bullets."
)
_DOCS_WRITER_TONE = "Docs: structured notes with headings, bullets, numbered lists."
_TASKS_ASR_ADDON = (
    "User is composing in a tasks/issues surface; bullet-friendly, action-oriented."
)
_TASKS_WRITER_TONE = "Tasks: short imperative sentences; bullet lists."


def _builtin_presets() -> list[SurfacePresetRecord]:
    """Conservative shipped defaults; user file overrides by ``bundle_id``."""
    return [
        # Email
        SurfacePresetRecord(
            id="builtin-apple-mail",
            bundle_id="com.apple.mail",
            fallback_app_category="email",
            default_built_in_mode="formal_email",
            asr_addon="User is composing email in Apple Mail; prefer full sentences and professional tone.",
            writer_tone_addon="Email composition: polite, clear, complete sentences.",
            include_window_title_in_asr=False,
            enabled=True,
        ),
        # Notes / docs
        SurfacePresetRecord(
            id="builtin-apple-notes",
            bundle_id="com.apple.Notes",
            fallback_app_category="docs",
            default_built_in_mode="structured_notes",
            asr_addon="User is dictating into Apple Notes; preserve note intent and spoken enumerations.",
            writer_tone_addon="Notes composition: use concise headings, paragraphs, and numbered lists for agendas.",
            include_window_title_in_asr=False,
            enabled=True,
        ),
        SurfacePresetRecord(
            id="builtin-notion",
            bundle_id="notion.id",
            fallback_app_category="docs",
            default_built_in_mode="structured_notes",
            asr_addon=_DOCS_ASR_ADDON,
            writer_tone_addon=_DOCS_WRITER_TONE,
            include_window_title_in_asr=True,
            enabled=True,
        ),
        SurfacePresetRecord(
            id="builtin-obsidian",
            bundle_id="md.obsidian",
            fallback_app_category="docs",
            default_built_in_mode="structured_notes",
            asr_addon=_DOCS_ASR_ADDON,
            writer_tone_addon=_DOCS_WRITER_TONE,
            include_window_title_in_asr=True,
            enabled=True,
        ),
        # Tasks
        SurfacePresetRecord(
            id="builtin-linear",
            bundle_id="com.linear",
            fallback_app_category="tasks",
            default_built_in_mode="structured_notes",
            asr_addon=_TASKS_ASR_ADDON,
            writer_tone_addon=_TASKS_WRITER_TONE,
            include_window_title_in_asr=True,
            enabled=True,
        ),
        # Code editors. ``default_built_in_mode='code_grammar'`` is an
        # intent marker — ``code_grammar`` is not a built-in writer mode,
        # so the resolver falls back to the base default while still
        # surfacing this preset record so the ``asr_addon`` and
        # ``fallback_app_category='code'`` flow through. The actual code
        # transforms are wired by the final-formatter post-pass keyed off
        # ``app_category`` (issue #12).
        SurfacePresetRecord(
            id="builtin-vscode",
            bundle_id="com.microsoft.VSCode",
            fallback_app_category="code",
            default_built_in_mode="code_grammar",
            asr_addon=_CODE_ASR_ADDON,
            writer_tone_addon=_CODE_WRITER_TONE,
            include_window_title_in_asr=True,
            enabled=True,
        ),
        SurfacePresetRecord(
            id="builtin-xcode",
            bundle_id="com.apple.dt.Xcode",
            fallback_app_category="code",
            default_built_in_mode="code_grammar",
            asr_addon=_CODE_ASR_ADDON,
            writer_tone_addon=_CODE_WRITER_TONE,
            include_window_title_in_asr=True,
            enabled=True,
        ),
        # Chat surfaces
        SurfacePresetRecord(
            id="builtin-slack",
            bundle_id="com.tinyspeck.slackmacgap",
            fallback_app_category="chat",
            default_built_in_mode="casual_chat",
            asr_addon=_CHAT_ASR_ADDON,
            writer_tone_addon=_CHAT_WRITER_TONE,
            include_window_title_in_asr=False,
            enabled=True,
        ),
        SurfacePresetRecord(
            id="builtin-telegram",
            bundle_id="ru.keepcoder.Telegram",
            fallback_app_category="chat",
            default_built_in_mode="casual_chat",
            asr_addon=_CHAT_ASR_ADDON,
            writer_tone_addon=_CHAT_WRITER_TONE,
            include_window_title_in_asr=False,
            enabled=True,
        ),
        SurfacePresetRecord(
            id="builtin-whatsapp",
            bundle_id="net.whatsapp.WhatsApp",
            fallback_app_category="chat",
            default_built_in_mode="casual_chat",
            asr_addon=_CHAT_ASR_ADDON,
            writer_tone_addon=_CHAT_WRITER_TONE,
            include_window_title_in_asr=False,
            enabled=True,
        ),
        SurfacePresetRecord(
            id="builtin-messages",
            bundle_id="com.apple.MobileSMS",
            fallback_app_category="chat",
            default_built_in_mode="casual_chat",
            asr_addon=_CHAT_ASR_ADDON,
            writer_tone_addon=_CHAT_WRITER_TONE,
            include_window_title_in_asr=False,
            enabled=True,
        ),
    ]


class SurfacePresetStore:
    """Load / save ``surface_presets.json`` (thread-safe)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write_raw({"schema_version": 1, "presets": []})

    def _read_raw(self) -> dict[str, Any]:
        with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"schema_version": 1, "presets": []}

    def _write_raw(self, data: dict[str, Any]) -> None:
        with self._lock:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_user_presets(self) -> list[SurfacePresetRecord]:
        raw = self._read_raw()
        out: list[SurfacePresetRecord] = []
        for item in raw.get("presets", []):
            if isinstance(item, dict):
                out.append(SurfacePresetRecord.from_dict(item))
        return out

    def list_presets_merged(self) -> list[SurfacePresetRecord]:
        """Built-in rows first, then user rows; user wins on duplicate ``bundle_id`` (casefold)."""
        by_bundle: dict[str, SurfacePresetRecord] = {}
        for p in _builtin_presets():
            if p.enabled and p.bundle_id.strip():
                by_bundle[p.bundle_id.strip().casefold()] = p
        for p in self.list_user_presets():
            if p.enabled and p.bundle_id.strip():
                by_bundle[p.bundle_id.strip().casefold()] = p
        return list(by_bundle.values())

    def upsert(self, record: SurfacePresetRecord) -> None:
        presets = self.list_user_presets()
        key_id = record.id.strip()
        key_bundle = record.bundle_id.strip().casefold()
        if not key_id or not record.bundle_id.strip():
            return
        replaced = False
        new_list: list[SurfacePresetRecord] = []
        for p in presets:
            if p.id.casefold() == key_id.casefold() or p.bundle_id.strip().casefold() == key_bundle:
                new_list.append(record)
                replaced = True
            else:
                new_list.append(p)
        if not replaced:
            new_list.append(record)
        self._write_raw({"schema_version": 1, "presets": [p.to_dict() for p in new_list]})

    def delete(self, preset_id: str) -> bool:
        key = (preset_id or "").strip()
        if not key:
            return False
        presets = self.list_user_presets()
        new_list = [p for p in presets if p.id.casefold() != key.casefold()]
        if len(new_list) == len(presets):
            return False
        self._write_raw({"schema_version": 1, "presets": [p.to_dict() for p in new_list]})
        return True

    def resolve(
        self,
        *,
        bundle_id: str | None,
        app_category: str | None,
        window_title: str | None = None,
    ) -> SurfacePresetRecord | None:
        """Pick the best matching enabled preset (bundle first, then category fallback)."""
        rows = self.list_presets_merged()
        bid = (bundle_id or "").strip().casefold()
        if bid:
            for p in rows:
                if p.bundle_id.strip().casefold() == bid:
                    return p
        cat = (app_category or "").strip().casefold()
        if cat:
            for p in rows:
                fb = (p.fallback_app_category or "").strip().casefold()
                if fb and fb == cat:
                    return p
        return None


def merge_prompt_parts(*parts: str | None, max_chars: int = 480) -> str | None:
    """Join non-empty prompt fragments with `` | ``; cap total length."""
    clean: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = (part or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        clean.append(value)
    if not clean:
        return None
    out = " | ".join(clean)
    if len(out) > max_chars:
        return out[: max_chars - 3].rstrip() + "..."
    return out


def build_surface_context_line(
    *,
    app_name: str | None,
    window_title: str | None,
    app_category: str | None,
    include_window_title: bool,
    max_len: int = 140,
) -> str | None:
    parts: list[str] = []
    if app_name and str(app_name).strip():
        parts.append(f"App: {str(app_name).strip()[:48]}")
    if include_window_title and window_title and str(window_title).strip():
        parts.append(f"Title: {str(window_title).strip()[:48]}")
    if app_category and str(app_category).strip():
        parts.append(f"Category: {str(app_category).strip()}")
    if not parts:
        return None
    s = " | ".join(parts)
    return s[:max_len]


def resolve_mode_with_surface_presets(
    *,
    manual_mode_name: str | None,
    custom_mode_name: str | None,
    custom_record: CustomModeRecord | None,
    surface_hint: str | None,
    surface_bundle_id: str | None,
    preset_store: SurfacePresetStore | None,
    custom_mode_store: CustomModeStore | None,
) -> tuple[ModeSelection, ModePolicy, SurfacePresetRecord | None]:
    """Resolve writer mode; apply surface preset only when manual + UI custom are unset.

    Precedence: manual > custom (UI) > surface preset > default_surface.
    """
    base = resolve_mode_selection(
        manual_mode_name=manual_mode_name,
        custom_mode_name=custom_mode_name,
        custom_record=custom_record,
        surface_hint=surface_hint,
    )
    if manual_mode_name or (custom_mode_name and custom_record is not None and custom_record.enabled):
        return base[0], base[1], None

    if preset_store is None:
        return base[0], base[1], None

    preset = preset_store.resolve(
        bundle_id=surface_bundle_id,
        app_category=surface_hint,
    )
    if preset is None:
        return base[0], base[1], None

    # Preset-driven custom mode
    if preset.custom_mode_name and custom_mode_store is not None:
        cr = custom_mode_store.get(preset.custom_mode_name)
        if cr is not None and cr.enabled:
            pol = apply_custom_overrides(
                mode_policy_for(cr.base_mode),
                cr,
                display_name=preset.custom_mode_name.strip(),
            )
            sel = ModeSelection(
                effective_mode=preset.custom_mode_name.strip(),
                mode_source=ModeSource.PRESET,
                manual_mode_name=None,
                custom_mode_name=preset.custom_mode_name.strip(),
                resolved_from_surface=surface_hint,
                surface_preset_id=preset.id,
                surface_bundle_id=surface_bundle_id,
            )
            return sel, pol, preset

    mode_name = (preset.default_built_in_mode or "").strip()
    if not mode_name or mode_name not in BUILTIN_MODES:
        # Unknown / non-built-in mode (e.g. ``code_grammar`` is an
        # engine name, not a writer mode). Surface the preset record
        # so its ``asr_addon`` / ``fallback_app_category`` propagate,
        # but use the base mode policy.
        return base[0], base[1], preset

    pol = mode_policy_for(mode_name)
    sel = ModeSelection(
        effective_mode=mode_name,
        mode_source=ModeSource.PRESET,
        manual_mode_name=None,
        custom_mode_name=None,
        resolved_from_surface=surface_hint,
        surface_preset_id=preset.id,
        surface_bundle_id=surface_bundle_id,
    )
    return sel, pol, preset
