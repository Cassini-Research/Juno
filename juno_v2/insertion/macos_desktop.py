from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field

from juno_v2.context.command import CommandRunner, SubprocessCommandRunner
from juno_v2.context.macos_desktop import MacOSDesktopContextProvider
from juno_v2.contracts.workbench import CommitMode
from juno_v2.contracts.insertion import InsertionRequest
from juno_v2.insertion.base import ActiveAppInserter


@dataclass(slots=True)
class MacOSActiveAppInserterConfig:
    osascript_bin: str = 'osascript'
    pbcopy_bin: str = 'pbcopy'
    pbpaste_bin: str = 'pbpaste'
    timeout_sec: float = 0.8
    helper_command: str | None = None
    # Issue #18: clipboard restoration races with the cmd-v keystroke (the
    # keystroke returns when osascript exits, NOT when the receiving app has
    # consumed the paste event). Default OFF — matches the Swift production
    # path. Opt-in restoration is gated by ``restore_delay_ms`` to give the
    # receiving app time to consume the paste.
    restore_clipboard: bool = False
    restore_delay_ms: float = 150.0


@dataclass(slots=True)
class MacOSActiveAppInserter(ActiveAppInserter):
    config: MacOSActiveAppInserterConfig = field(default_factory=MacOSActiveAppInserterConfig)
    runner: CommandRunner = field(default_factory=SubprocessCommandRunner)
    state_provider: MacOSDesktopContextProvider | None = None
    name: str = 'macos_active_app'

    def warm(self) -> None:
        caps = self.capabilities()
        if not caps['platform_tools_ready']:
            raise RuntimeError('macOS insertion tools are unavailable')
        if not caps['accessibility_enabled']:
            raise RuntimeError('macOS Accessibility permission is not enabled for active-app insertion')

    def capabilities(self) -> dict[str, object]:
        osascript_available = shutil.which(self.config.osascript_bin) is not None
        pbcopy_available = shutil.which(self.config.pbcopy_bin) is not None
        pbpaste_available = shutil.which(self.config.pbpaste_bin) is not None
        accessibility_enabled = False
        if osascript_available:
            try:
                raw = self.runner.run([self.config.osascript_bin, '-e', 'tell application "System Events" to UI elements enabled'], timeout_sec=self.config.timeout_sec)
                accessibility_enabled = raw.strip().lower() == 'true'
            except Exception:
                accessibility_enabled = False
        return {
            'osascript_available': osascript_available,
            'pbcopy_available': pbcopy_available,
            'pbpaste_available': pbpaste_available,
            'accessibility_enabled': accessibility_enabled,
            'platform_tools_ready': osascript_available and pbcopy_available and pbpaste_available,
            'helper_command_configured': bool(self.config.helper_command),
        }

    def apply_commit(self, req: InsertionRequest) -> None:
        self.warm()
        if req.commit_mode == CommitMode.REPLACE_SELECTION and self.state_provider is not None:
            sync = self.state_provider.editable_sync_request()
            if sync is not None and sync.selection_start == sync.selection_end:
                raise RuntimeError('No active selection available for macOS replace-selection insertion')
        # Default: do NOT save the clipboard (matches the Swift production
        # inserter and eliminates the restore race). Opt-in restoration
        # honors ``restore_delay_ms`` to keep the keystroke and the restore
        # from racing through pasteboard.
        original_clipboard: str | None = None
        if self.config.restore_clipboard:
            original_clipboard = self.runner.run(
                [self.config.pbpaste_bin], timeout_sec=self.config.timeout_sec
            )
        try:
            self.runner.run([self.config.pbcopy_bin], timeout_sec=self.config.timeout_sec, stdin_text=req.text)
            if req.commit_mode == CommitMode.REPLACE_ALL:
                self._keystroke('a', modifiers=['command down'])
            self._keystroke('v', modifiers=['command down'])
        finally:
            if self.config.restore_clipboard and original_clipboard is not None:
                delay_ms = max(0.0, float(self.config.restore_delay_ms))
                if delay_ms > 0.0:
                    time.sleep(delay_ms / 1000.0)
                try:
                    self.runner.run(
                        [self.config.pbcopy_bin],
                        timeout_sec=self.config.timeout_sec,
                        stdin_text=original_clipboard,
                    )
                except Exception:
                    pass

    def _keystroke(self, key: str, *, modifiers: list[str]) -> None:
        mods = '{' + ', '.join(modifiers) + '}' if modifiers else '{}'
        script = f'tell application "System Events" to keystroke "{key}" using {mods}'
        self.runner.run([self.config.osascript_bin, '-e', script], timeout_sec=self.config.timeout_sec)
