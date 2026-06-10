from __future__ import annotations

import json
import os
import plistlib
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


def test_macos_ota_plist_configures_sparkle(tmp_path: Path) -> None:
    key = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="
    plist_path = tmp_path / "Info.plist"
    plist_path.write_bytes(
        b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleShortVersionString</key><string>0.2.0</string>
  <key>CFBundleVersion</key><string>1</string>
</dict></plist>
"""
    )

    result = run(
        sys.executable,
        "scripts/configure_juno_macos_plist.py",
        str(plist_path),
        "--version",
        "1.2.3",
        "--build",
        "45",
        "--ota-feed-url",
        "https://updates.example.com/juno/appcast.xml",
        "--ota-public-ed-key",
        key,
        "--ota-channel",
        "beta",
        "--automatic-downloads",
        "--scheduled-interval",
        "3600",
    )

    assert result.returncode == 0, result.stderr + result.stdout

    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["CFBundleShortVersionString"] == "1.2.3"
    assert plist["CFBundleVersion"] == "45"
    assert plist["JunoOTAEnabled"] is True
    assert plist["SUFeedURL"] == "https://updates.example.com/juno/appcast.xml"
    assert plist["SUPublicEDKey"] == key
    assert plist["SUEnableAutomaticChecks"] is True
    assert plist["SUAllowsAutomaticUpdates"] is True
    assert plist["SUAutomaticallyUpdate"] is True
    assert plist["SUScheduledCheckInterval"] == 3600.0
    assert plist["SUShowReleaseNotes"] is True
    assert plist["JunoUpdateChannel"] == "beta"


def test_macos_ota_plist_disable_removes_sparkle_keys(tmp_path: Path) -> None:
    plist_path = tmp_path / "Info.plist"
    plist_path.write_bytes(
        plistlib.dumps(
            {
                "CFBundleShortVersionString": "0.2.0",
                "CFBundleVersion": "1",
                "JunoOTAEnabled": True,
                "JunoUpdateChannel": "beta",
                "SUFeedURL": "https://updates.example.com/juno/appcast.xml",
                "SUPublicEDKey": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
                "SUEnableAutomaticChecks": True,
                "SUAllowsAutomaticUpdates": True,
                "SUAutomaticallyUpdate": True,
                "SUScheduledCheckInterval": 3600.0,
                "SUShowReleaseNotes": True,
            }
        )
    )

    result = run(
        sys.executable,
        "scripts/configure_juno_macos_plist.py",
        str(plist_path),
        "--disable-ota",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["JunoOTAEnabled"] is False
    assert "JunoUpdateChannel" not in plist
    assert "SUFeedURL" not in plist
    assert "SUPublicEDKey" not in plist
    assert "SUEnableAutomaticChecks" not in plist


def test_macos_ota_plist_rejects_partial_configuration(tmp_path: Path) -> None:
    plist_path = tmp_path / "Info.plist"
    plist_path.write_bytes(b"<?xml version=\"1.0\"?><plist version=\"1.0\"><dict/></plist>")

    result = run(
        sys.executable,
        "scripts/configure_juno_macos_plist.py",
        str(plist_path),
        "--ota-feed-url",
        "https://updates.example.com/juno/appcast.xml",
    )

    assert result.returncode == 2
    assert "OTA requires both" in result.stderr
