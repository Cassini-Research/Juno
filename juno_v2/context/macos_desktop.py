from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass, field

from juno_v2.context.command import CommandRunner, SubprocessCommandRunner
from juno_v2.context.ide_symbol import derive_focused_file_path, derive_symbol_under_cursor
from juno_v2.context.provider import _build_bundle
from juno_v2.context.redaction import ContextRedactor
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.workbench import SyncClientStateRequest


@dataclass(slots=True)
class MacOSDesktopContextProviderConfig:
    max_field_chars: int = 1600
    helper_command: str | None = None
    osascript_bin: str = 'osascript'
    pbpaste_bin: str = 'pbpaste'
    timeout_sec: float = 0.8


@dataclass(slots=True)
class MacOSDesktopContextProvider:
    config: MacOSDesktopContextProviderConfig = field(default_factory=MacOSDesktopContextProviderConfig)
    runner: CommandRunner = field(default_factory=SubprocessCommandRunner)
    redactor: ContextRedactor = field(default_factory=ContextRedactor)

    def snapshot(self) -> TypedContextBundle:
        helper = self._helper_payload()
        app_name = helper.get('app_name') or self._frontmost_app_name()
        window_title = helper.get('window_title') or self._window_title()
        selected_text = helper.get('selected_text') or ''
        clipboard_text = helper.get('clipboard_text') or self._clipboard_text()
        focused_before = helper.get('focused_text_before') or helper.get('surrounding_text_before') or ''
        focused_after = helper.get('focused_text_after') or helper.get('surrounding_text_after') or ''
        bid = helper.get('app_bundle_id') or helper.get('frontmost_app_bundle_id')
        bundle_id_arg = bid.strip() if isinstance(bid, str) and bid.strip() else None
        bundle = _build_bundle(
            app_name=app_name,
            window_title=window_title,
            app_bundle_id=bundle_id_arg,
            selected_text=selected_text,
            focused_text_before=focused_before,
            focused_text_after=focused_after,
            clipboard_text=clipboard_text,
            max_field_chars=self.config.max_field_chars,
            redactor=self.redactor,
            last_committed_text='',
            last_committed_start=None,
            last_committed_end=None,
            last_committed_utterance_id=None,
        )
        bundle.metadata.update({
            'context_source': 'macos_desktop',
            'helper_used': bool(helper),
            'capabilities': self.capabilities(),
        })
        loc = helper.get('locale_identifier')
        if isinstance(loc, str) and loc.strip():
            bundle.metadata['locale_identifier'] = loc.strip()
        itn_fmt = helper.get('itn_format')
        if isinstance(itn_fmt, dict) and itn_fmt:
            bundle.metadata['itn_format'] = itn_fmt
        # IDE context — only populate when the focused surface is a
        # code-class app; for other categories the symbols/paths are
        # noise at best (emails, chat) and a privacy leak at worst.
        # ``app_category`` has already been stamped by
        # :func:`_build_bundle` via the app classifier.
        if bundle.app_category == 'code':
            file_path = derive_focused_file_path(helper, bundle.window_title)
            if file_path:
                bundle.focused_file_path = file_path
            symbol = derive_symbol_under_cursor(
                bundle.focused_text_before,
                bundle.focused_text_after,
            )
            if symbol:
                bundle.symbol_under_cursor = symbol
        return bundle

    def editable_sync_request(self) -> SyncClientStateRequest | None:
        helper = self._helper_payload()
        before = helper.get('focused_text_before') or helper.get('surrounding_text_before') or ''
        after = helper.get('focused_text_after') or helper.get('surrounding_text_after') or ''
        selected = helper.get('selected_text') or ''
        if not (before or after or selected):
            return None
        buffer_text = before + selected + after
        sel_start = len(before)
        sel_end = sel_start + len(selected)
        return SyncClientStateRequest(
            buffer_text=buffer_text,
            selection_start=sel_start,
            selection_end=sel_end,
            app_name=helper.get('app_name') or self._frontmost_app_name() or None,
            window_title=helper.get('window_title') or self._window_title() or None,
            clipboard_text=helper.get('clipboard_text') or self._clipboard_text(),
        )

    def capabilities(self) -> dict[str, object]:
        osascript_available = shutil.which(self.config.osascript_bin) is not None
        pbpaste_available = shutil.which(self.config.pbpaste_bin) is not None
        ui_enabled = False
        if osascript_available:
            try:
                raw = self.runner.run([
                    self.config.osascript_bin,
                    '-e',
                    'tell application "System Events" to UI elements enabled',
                ], timeout_sec=self.config.timeout_sec)
                ui_enabled = raw.strip().lower() == 'true'
            except Exception:
                ui_enabled = False
        return {
            'osascript_available': osascript_available,
            'pbpaste_available': pbpaste_available,
            'helper_command_configured': bool(self.config.helper_command),
            'accessibility_enabled': ui_enabled,
        }

    def _helper_payload(self) -> dict:
        if not self.config.helper_command:
            return {}
        argv = shlex.split(self.config.helper_command)
        if not argv:
            return {}
        try:
            payload = self.runner.run(argv, timeout_sec=self.config.timeout_sec)
            data = json.loads(payload)
            if not isinstance(data, dict):
                return {}
        except Exception:
            return {}
        # Bridge two key schemas the helper may emit:
        # - Historic / linux-style: ``app_name``, ``app_bundle_id``,
        #   ``selected_text``, ``focused_text_before/after``, ...
        # - Current ``juno-capability`` Swift helper: prefixes
        #   identity fields with ``frontmost_``.
        # We canonicalize to the first schema so the rest of the
        # provider doesn't need to care which helper produced the
        # payload. Never overwrite a key the helper already set.
        def _alias(src: str, dst: str) -> None:
            if dst not in data and data.get(src) is not None:
                data[dst] = data[src]

        _alias("frontmost_app_name", "app_name")
        _alias("frontmost_app_bundle_id", "app_bundle_id")
        _alias("frontmost_pid", "app_pid")
        _alias("focused_is_secure", "focused_secure")
        _alias("surrounding_text_before", "focused_text_before")
        _alias("surrounding_text_after", "focused_text_after")
        return data

    def _frontmost_app_name(self) -> str:
        script = 'tell application "System Events" to name of first application process whose frontmost is true'
        try:
            return self.runner.run([self.config.osascript_bin, '-e', script], timeout_sec=self.config.timeout_sec)
        except Exception:
            return ''

    def _window_title(self) -> str:
        script = '''tell application "System Events"
set frontApp to first application process whose frontmost is true
try
    tell front window of frontApp to get value of attribute "AXTitle"
on error
    try
        name of front window of frontApp
    on error
        return ""
    end try
end try
end tell'''
        try:
            return self.runner.run([self.config.osascript_bin, '-e', script], timeout_sec=self.config.timeout_sec)
        except Exception:
            return ''

    def _clipboard_text(self) -> str:
        try:
            return self.runner.run([self.config.pbpaste_bin], timeout_sec=self.config.timeout_sec)
        except Exception:
            return ''
