#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export JUNO_V2_WRITER_RESIDENCY_POLICY="${JUNO_V2_WRITER_RESIDENCY_POLICY:-resident}"
export JUNO_V2_WRITER_IDLE_UNLOAD_TTL_S="${JUNO_V2_WRITER_IDLE_UNLOAD_TTL_S:-300}"

exec "$PYTHON_BIN" -m juno_v2.workbench.server "$@"
