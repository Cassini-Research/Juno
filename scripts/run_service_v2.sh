#!/usr/bin/env bash
set -euo pipefail

if [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python
fi

exec "$PYTHON" -m juno_v2.runtime.service "$@"
