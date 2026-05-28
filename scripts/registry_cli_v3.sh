#!/usr/bin/env bash
# Thin wrapper around `python -m juno_core_v3.model_registry.cli`.
# Matches the other `scripts/*_v2.sh` wrappers so operators don't need
# to remember the module path.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

exec "$PYTHON" -m juno_core_v3.model_registry.cli "$@"
