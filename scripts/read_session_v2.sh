#!/usr/bin/env bash
# Read and summarize a Juno session trace log.
#
# Usage:
#   ./scripts/read_session_v2.sh                     # list all sessions
#   ./scripts/read_session_v2.sh SERVICE_SESSION_ID  # human-readable timeline
#   ./scripts/read_session_v2.sh SERVICE_SESSION_ID --format json  # JSON dump
#   ./scripts/read_session_v2.sh SERVICE_SESSION_ID --verbose      # include VAD noise
#
# The session ID or file path can be partial (e.g. "service_live_20260407")
# and the reader will find the best match in .juno_v2_logs/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

exec "$PYTHON" -m juno_v2.observability.session_reader "$@"
