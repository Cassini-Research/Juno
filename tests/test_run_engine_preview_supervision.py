"""Regression test for the bundled-engine preview-service supervisor.

2026-06-12: the streaming preview service (spawned once by run_engine.sh)
died ~20s after app launch and live HUD preview went permanently blank —
every broker preview chunk failed with connection-refused while final
transcription kept working, and engine health kept reporting the preview
backend ok. The fix wraps the service in a supervision loop inside
run_engine.sh. This test extracts that loop verbatim from the script and
exercises it with a stub service binary:

  * the service is restarted after it dies (with backoff),
  * each exit is logged with status and uptime,
  * SIGTERM to the supervisor also terminates the running child.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ENGINE = REPO_ROOT / "shells" / "macos" / "bundle_resources" / "engine" / "run_engine.sh"
MARKER_BEGIN = "# >>> juno-preview-supervisor"
MARKER_END = "# <<< juno-preview-supervisor"

# Stub stands in for `${PY} -m juno_v2.preview.streaming_service ...`.
# First two invocations die immediately with status 7; later invocations
# stay resident. Counter and child pid land in $STUB_DIR for assertions.
STUB = """#!/usr/bin/env bash
count_file="${STUB_DIR}/count"
n=$(cat "${count_file}" 2>/dev/null || echo 0)
n=$((n + 1))
echo "${n}" > "${count_file}"
echo $$ > "${STUB_DIR}/last_pid"
if [ "${n}" -le 2 ]; then
  exit 7
fi
sleep 600
"""


def _extract_supervisor() -> str:
    text = RUN_ENGINE.read_text(encoding="utf-8")
    assert MARKER_BEGIN in text, (
        "run_engine.sh no longer contains the preview-service supervisor block; "
        "a one-shot spawn regresses the permanently-dead-live-preview bug"
    )
    after_marker = text.split(MARKER_BEGIN, 1)[1]
    # Drop the remainder of the marker line itself; only the lines between
    # the markers are bash code.
    block = after_marker.split("\n", 1)[1].split(MARKER_END, 1)[0]
    return block


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


def test_supervisor_restarts_service_and_term_kills_child(tmp_path: Path) -> None:
    stub = tmp_path / "stub_service"
    stub.write_text(STUB, encoding="utf-8")
    stub.chmod(0o755)
    preview_log = tmp_path / "preview-service.log"
    preview_log.touch()

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'PY="{stub}"',
                f'PREVIEW_LOG="{preview_log}"',
                'JUNO_V2_PREVIEW_SERVICE_BACKEND="stub"',
                'PREVIEW_HOST="127.0.0.1"',
                'PREVIEW_PORT="0"',
                'JUNO_V2_PREVIEW_MODEL_PATH="stub-model"',
                'JUNO_V2_PREVIEW_HF_REPO_ID="stub-repo"',
                _extract_supervisor(),
                "supervise_preview_service &",
                'SUP="$!"',
                'echo "${SUP}" > "${STUB_DIR}/supervisor_pid"',
                'wait "${SUP}" || true',
            ]
        ),
        encoding="utf-8",
    )
    harness.chmod(0o755)

    env = dict(os.environ, STUB_DIR=str(tmp_path))
    proc = subprocess.Popen(["bash", str(harness)], env=env)
    try:
        count_file = tmp_path / "count"

        def third_start_happened() -> bool:
            try:
                return int(count_file.read_text()) >= 3
            except (FileNotFoundError, ValueError):
                return False

        # Two immediate deaths (status 7) with 2s + 4s backoff, then the
        # resident third start: comfortably under the deadline.
        _wait_for(third_start_happened, 30.0, "supervisor to restart the service twice")

        log_text = preview_log.read_text(encoding="utf-8")
        assert "exited status=7" in log_text, f"missing exit log line, got: {log_text!r}"
        assert "restarting in" in log_text

        child_pid = int((tmp_path / "last_pid").read_text().strip())
        supervisor_pid = int((tmp_path / "supervisor_pid").read_text().strip())

        # Resident child must be alive before the TERM.
        os.kill(child_pid, 0)

        os.kill(supervisor_pid, signal.SIGTERM)

        def child_gone() -> bool:
            try:
                os.kill(child_pid, 0)
                return False
            except ProcessLookupError:
                return True

        _wait_for(child_gone, 10.0, "TERM on supervisor to kill the resident service")
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
