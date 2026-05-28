from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from juno_v2.context.command import CommandRunner, SubprocessCommandRunner
from juno_v2.context.provider import _build_bundle
from juno_v2.context.redaction import ContextRedactor
from juno_v2.contracts.context import TypedContextBundle


@dataclass(slots=True)
class LinuxDesktopContextProviderConfig:
    max_field_chars: int = 240
    helper_command: str | None = None
    prefer_wayland: bool = True
    xdotool_bin: str = 'xdotool'
    xclip_bin: str = 'xclip'
    wl_paste_bin: str = 'wl-paste'
    timeout_sec: float = 0.5


@dataclass(slots=True)
class LinuxDesktopContextProvider:
    config: LinuxDesktopContextProviderConfig = field(default_factory=LinuxDesktopContextProviderConfig)
    runner: CommandRunner = field(default_factory=SubprocessCommandRunner)
    redactor: ContextRedactor = field(default_factory=ContextRedactor)

    def snapshot(self) -> TypedContextBundle:
        helper = self._helper_payload()
        app_name = helper.get('app_name') or self._app_name(helper.get('window_pid'))
        window_title = helper.get('window_title') or self._window_title()
        selected_text = helper.get('selected_text') or self._selected_text()
        clipboard_text = helper.get('clipboard_text') or self._clipboard_text()
        focused_before = helper.get('focused_text_before') or helper.get('surrounding_text_before') or ''
        focused_after = helper.get('focused_text_after') or helper.get('surrounding_text_after') or ''
        bundle = _build_bundle(
            app_name=app_name,
            window_title=window_title,
            app_bundle_id=None,
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
            'context_source': 'linux_desktop',
            'helper_used': bool(helper),
            'display_server': 'wayland' if os.getenv('WAYLAND_DISPLAY') else ('x11' if os.getenv('DISPLAY') else 'unknown'),
        })
        return bundle

    def _helper_payload(self) -> dict:
        if not self.config.helper_command:
            return {}
        argv = shlex.split(self.config.helper_command)
        if not argv:
            return {}
        try:
            payload = self.runner.run(argv, timeout_sec=self.config.timeout_sec)
            data = json.loads(payload)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _window_title(self) -> str:
        argv = [self.config.xdotool_bin, 'getactivewindow', 'getwindowname']
        try:
            return self.runner.run(argv, timeout_sec=self.config.timeout_sec)
        except Exception:
            return ''

    def _window_pid(self) -> int | None:
        argv = [self.config.xdotool_bin, 'getactivewindow', 'getwindowpid']
        try:
            raw = self.runner.run(argv, timeout_sec=self.config.timeout_sec)
            return int(raw)
        except Exception:
            return None

    def _app_name(self, helper_pid: object | None = None) -> str:
        pid: int | None
        try:
            pid = int(helper_pid) if helper_pid is not None else self._window_pid()
        except Exception:
            pid = self._window_pid()
        if pid is None:
            return ''
        try:
            return Path(f'/proc/{pid}/comm').read_text(encoding='utf-8').strip()
        except Exception:
            return ''

    def _selected_text(self) -> str:
        commands = self._selection_commands(primary=True)
        for argv in commands:
            try:
                text = self.runner.run(argv, timeout_sec=self.config.timeout_sec)
                if text:
                    return text
            except Exception:
                continue
        return ''

    def _clipboard_text(self) -> str:
        commands = self._selection_commands(primary=False)
        for argv in commands:
            try:
                text = self.runner.run(argv, timeout_sec=self.config.timeout_sec)
                if text:
                    return text
            except Exception:
                continue
        return ''

    def _selection_commands(self, *, primary: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        use_wayland = self.config.prefer_wayland and bool(os.getenv('WAYLAND_DISPLAY'))
        if use_wayland:
            commands.append([self.config.wl_paste_bin, '--no-newline', '--primary' if primary else '--clipboard'])
        selection = 'primary' if primary else 'clipboard'
        commands.append([self.config.xclip_bin, '-selection', selection, '-o'])
        return commands
