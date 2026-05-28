from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Protocol


class CommandRunner(Protocol):
    def run(self, argv: list[str], *, timeout_sec: float = 0.5, stdin_text: str | None = None) -> str: ...


@dataclass(slots=True)
class SubprocessCommandRunner:
    env: dict[str, str] | None = None

    def run(self, argv: list[str], *, timeout_sec: float = 0.5, stdin_text: str | None = None) -> str:
        proc = subprocess.run(  # noqa: S603
            argv,
            check=False,
            capture_output=True,
            timeout=timeout_sec,
            text=True,
            env=self.env,
            input=stdin_text,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f'Command failed: {shlex.join(argv)}')
        return proc.stdout.strip()
