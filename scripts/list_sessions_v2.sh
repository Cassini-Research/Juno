#!/usr/bin/env bash
# List recorded Juno session logs with metadata.
#
# Usage:
#   ./scripts/list_sessions_v2.sh
#   ./scripts/list_sessions_v2.sh --format json    # JSON output
#   ./scripts/list_sessions_v2.sh --log-dir /path  # custom log root

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

exec "$PYTHON" -m juno_v2.observability.session_reader --list "$@"
