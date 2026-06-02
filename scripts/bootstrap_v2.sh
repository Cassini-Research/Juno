#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Honor an explicit PYTHON_BIN override (e.g. PYTHON_BIN=python3.11), and
# otherwise pick the newest python3.X >= 3.10 on PATH before falling back to
# bare `python3`. On stock macOS, `python3` resolves to 3.9, which fails
# deep inside `pip install -e .` with a hard-to-read requires-python error.
# Resolving up-front lets the user see a clear message before any venv work.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "bootstrap: no python3 interpreter found on PATH." >&2
  echo "  Install Python 3.10+ (e.g. via Homebrew: brew install python@3.11)" >&2
  echo "  or set PYTHON_BIN=/path/to/python3.11 and rerun." >&2
  exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "bootstrap: $PYTHON_BIN is Python $PY_VERSION but Juno requires 3.10 or newer." >&2
  echo "  Install a newer Python (e.g. brew install python@3.11) and rerun:" >&2
  echo "    PYTHON_BIN=python3.11 ./scripts/bootstrap.sh" >&2
  exit 1
fi

echo "bootstrap: using $PYTHON_BIN (Python $PY_VERSION)"
"$PYTHON_BIN" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
echo "Juno environment ready. Activate with: source .venv/bin/activate"
