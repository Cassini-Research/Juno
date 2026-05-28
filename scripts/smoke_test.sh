#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"$PYTHON_BIN" -m compileall -q juno_v2 juno_core_v3 scripts

"$PYTHON_BIN" - <<'PY'
import json
import pathlib
import sys

bad = []
for p in pathlib.Path(".").rglob("*.json"):
    if any(part in {".git", ".venv", "build", "dist"} for part in p.parts):
        continue
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        bad.append((str(p), str(e)))

if bad:
    for path, err in bad:
        print(f"{path}: {err}")
    sys.exit(1)
PY

./scripts/doctor.sh --ci
./scripts/run_workbench.sh --help >/dev/null
"$PYTHON_BIN" scripts/audit_public_terms.py
pytest
