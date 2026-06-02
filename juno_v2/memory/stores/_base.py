from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


_LOG = logging.getLogger(__name__)


class JsonFileStore:
    """JSON-file primitive shared by every decomposed store.

    Each domain store uses a distinct filename inside the same ``memory_dir``
    and a **shared** lock (passed in by the facade) so multi-store
    transactions can be serialized.

    The lock is re-entrant (``RLock``) so a caller that already holds the
    lock can call nested methods without deadlocking.

    Writes are *atomic* on POSIX: a sibling tempfile is written + fsynced
    and then ``os.replace``'d into position. This prevents a crash mid-write
    from leaving a truncated file readable by the next boot.

    Reads quarantine corrupt JSON: on ``json.JSONDecodeError`` /
    ``UnicodeDecodeError`` the bad file is renamed aside (``<name>.corrupt-<ts>``)
    and the caller-supplied default is returned. One bad shutdown does not
    take down the memory plane.
    """

    def __init__(self, memory_dir: Path | str, filename: str, *, lock: threading.RLock | None = None) -> None:
        self.memory_dir = Path(memory_dir)
        self.filename = filename
        self.lock = lock or threading.RLock()
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.memory_dir / self.filename

    def read(self, default: Any) -> Any:
        path = self.path
        if not path.exists():
            self._cleanup_orphaned_tmp_files()
            return default
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._cleanup_orphaned_tmp_files()
            return data
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            quarantine = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            try:
                path.rename(quarantine)
            except OSError:
                # Best-effort; if rename fails, leave the file in place.
                pass
            _LOG.warning(
                "JsonFileStore: quarantined corrupt %s -> %s (%s)",
                path,
                quarantine,
                exc,
            )
            self._cleanup_orphaned_tmp_files()
            return default

    def _cleanup_orphaned_tmp_files(self) -> None:
        """Remove temp files left by a crashed writer process.

        Active same-process writes use this process id in the temp filename and
        must not be unlinked by a concurrent read. Cross-process leftovers are
        only removed after their recorded owner process no longer exists.
        """

        prefix = f".{self.filename}."
        current_pid = os.getpid()
        for tmp_path in self.memory_dir.iterdir():
            name = tmp_path.name
            if not (name.startswith(prefix) and name.endswith(".tmp")):
                continue
            owner_pid = self._tmp_owner_pid(name, prefix)
            if owner_pid == current_pid:
                continue
            if owner_pid is not None and self._pid_exists(owner_pid):
                continue
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _tmp_owner_pid(name: str, prefix: str) -> int | None:
        rest = name[len(prefix):]
        raw_pid = rest.split(".", 1)[0]
        try:
            return int(raw_pid)
        except ValueError:
            return None

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def write(self, payload: Any) -> None:
        path = self.path
        # Atomic write: write to a sibling tmp file, fsync, then os.replace.
        # NamedTemporaryFile in the same dir guarantees same-filesystem rename.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.memory_dir),
            prefix=f".{self.filename}.{os.getpid()}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            try:
                json.dump(payload, tmp, ensure_ascii=False, indent=2)
                tmp.flush()
                try:
                    os.fsync(tmp.fileno())
                except OSError:
                    pass
            except Exception:
                # Best-effort cleanup of half-written temp file on failure.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        os.replace(tmp_path, path)
        # fsync the parent dir for durability (POSIX).
        try:
            dir_fd = os.open(str(self.memory_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    def ensure_default(self, default: Any) -> None:
        if not self.path.exists():
            self.write(default)
