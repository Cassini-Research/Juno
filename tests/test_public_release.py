from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)


def test_python_files_compile() -> None:
    result = run(sys.executable, "-m", "compileall", "-q", "juno_v2", "juno_core_v3", "scripts")
    assert result.returncode == 0, result.stderr + result.stdout


def test_tracked_json_files_parse() -> None:
    bad: list[str] = []
    for path in ROOT.rglob("*.json"):
        if any(part in {".git", ".venv", "build", "dist"} for part in path.parts):
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{path.relative_to(ROOT)}: {exc}")
    assert not bad, "\n".join(bad)


def test_public_scripts_exist_and_are_executable() -> None:
    for rel in (
        "scripts/bootstrap.sh",
        "scripts/bootstrap_full.sh",
        "scripts/doctor.sh",
        "scripts/run_live.sh",
        "scripts/run_workbench.sh",
        "scripts/install_macos.sh",
        "scripts/package_macos.sh",
        "scripts/smoke_test.sh",
    ):
        path = ROOT / rel
        assert path.is_file(), rel
        assert os.access(path, os.X_OK), rel


def test_readme_script_commands_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commands = re.findall(r"^(\./scripts/[A-Za-z0-9_./-]+)", readme, flags=re.M)
    assert commands
    missing = [cmd for cmd in commands if not (ROOT / cmd).exists()]
    assert not missing


def test_env_example_default_is_not_missing_replay_fixture() -> None:
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line in env.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    assert values.get("JUNO_V2_MODE") != "replay" or values.get("JUNO_V2_REPLAY_WAV")
    assert values.get("JUNO_V2_PLATFORM_NAME", "") != "linux"
    assert values.get("JUNO_V2_FINAL_BACKEND") != "mlx_whisper"


def test_doctor_ci_runs() -> None:
    result = run("./scripts/doctor.sh", "--ci", "--json")
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True


def test_workbench_help_runs() -> None:
    result = run("./scripts/run_workbench.sh", "--help")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Juno workbench" in result.stdout


def test_workbench_stays_alive_until_stopped() -> None:
    proc = subprocess.Popen(
        ["./scripts/run_workbench.sh", "--port", "9876"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1.5)
        assert proc.poll() is None, (proc.stdout.read() if proc.stdout else "") + (proc.stderr.read() if proc.stderr else "")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_forbidden_public_terms_are_allowlisted() -> None:
    result = run(sys.executable, "scripts/audit_public_terms.py")
    assert result.returncode == 0, result.stderr + result.stdout
