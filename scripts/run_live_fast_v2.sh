#!/usr/bin/env bash
# Launch the live Juno service with the canonical MLX Whisper
# large-v3-turbo streaming preview service + final lane.
#
# The runtime talks to the preview lane over HTTP
# (streaming_local_http_json). Keeping the preview model in a separate
# resident process prevents the engine loop from competing with the
# final/writer lanes while Whisper is decoded on live checkpoints.
#
# Workbench UI: http://127.0.0.1:8765
# Ctrl-C to stop cleanly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python
fi

PREVIEW_MODEL="${JUNO_V2_PREVIEW_MODEL_PATH:-mlx-community/whisper-large-v3-turbo}"
PREVIEW_HF_REPO_ID="${JUNO_V2_PREVIEW_HF_REPO_ID:-mlx-community/whisper-large-v3-turbo}"

echo "preview model: $PREVIEW_MODEL"

PREVIEW_HOST="${JUNO_V2_PREVIEW_HOST:-127.0.0.1}"
PREVIEW_PORT="${JUNO_V2_PREVIEW_PORT:-8795}"
PREVIEW_ENDPOINT="${JUNO_V2_PREVIEW_ENDPOINT:-http://${PREVIEW_HOST}:${PREVIEW_PORT}}"
PREVIEW_SERVICE_BACKEND="${JUNO_V2_PREVIEW_SERVICE_BACKEND:-mlx_whisper}"
PREVIEW_DEVICE="${JUNO_V2_PREVIEW_DEVICE:-auto}"
PREVIEW_COMPUTE_TYPE="${JUNO_V2_PREVIEW_COMPUTE_TYPE:-default}"
PREVIEW_SERVICE_LOG="${JUNO_V2_PREVIEW_SERVICE_LOG:-.juno_v2_logs/preview_service.log}"

mkdir -p "$(dirname "$PREVIEW_SERVICE_LOG")"

PREVIEW_PID=""
cleanup_preview() {
  if [[ -n "${PREVIEW_PID:-}" ]] && kill -0 "$PREVIEW_PID" 2>/dev/null; then
    kill "$PREVIEW_PID" 2>/dev/null || true
    wait "$PREVIEW_PID" 2>/dev/null || true
  fi
}
trap cleanup_preview EXIT
trap 'trap - EXIT; cleanup_preview; exit 130' INT
trap 'trap - EXIT; cleanup_preview; exit 143' TERM

echo "preview service: $PREVIEW_ENDPOINT ($PREVIEW_SERVICE_BACKEND)"
"$PY" -m juno_v2.preview.streaming_service \
  --host "$PREVIEW_HOST" \
  --port "$PREVIEW_PORT" \
  --backend "$PREVIEW_SERVICE_BACKEND" \
  --model-path "$PREVIEW_MODEL" \
  --hf-repo-id "$PREVIEW_HF_REPO_ID" \
  --device "$PREVIEW_DEVICE" \
  --compute-type "$PREVIEW_COMPUTE_TYPE" \
  >"$PREVIEW_SERVICE_LOG" 2>&1 &
PREVIEW_PID=$!

for _ in $(seq 1 100); do
  if "$PY" - "$PREVIEW_ENDPOINT" <<'PY'
import json
import sys
import urllib.request

endpoint = sys.argv[1].rstrip("/") + "/healthz"
try:
    with urllib.request.urlopen(endpoint, timeout=0.5) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("ok") else 1)
PY
  then
    break
  fi
  if ! kill -0 "$PREVIEW_PID" 2>/dev/null; then
    echo "ERROR: preview service exited early. See $PREVIEW_SERVICE_LOG" >&2
    exit 2
  fi
  sleep 0.2
done

if ! "$PY" - "$PREVIEW_ENDPOINT" <<'PY'
import json
import sys
import urllib.request

endpoint = sys.argv[1].rstrip("/") + "/healthz"
try:
    with urllib.request.urlopen(endpoint, timeout=0.5) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("ok") else 1)
PY
then
  echo "ERROR: preview service did not become healthy. See $PREVIEW_SERVICE_LOG" >&2
  exit 2
fi

# Live-dictation latency logging — every committed utterance writes a
# one-line summary to stderr showing ttft / final / commit latency,
# the preview+final backend names, and the committed text. Comment
# out or unset if the stream becomes noisy during long sessions.
export JUNO_V2_LOG_LATENCY=${JUNO_V2_LOG_LATENCY:-1}

# Keep the local fast launcher on the same action-extraction contract as the
# packaged macOS engine. The HUD/audio E2E gate runs through this script, so
# leaving these unset silently exercises the older v2 action schema even when
# the installed app would use v3 containers, vague times, and compound chains.
export JUNO_ACTIONS_SCHEMA_V3="${JUNO_ACTIONS_SCHEMA_V3:-1}"
export JUNO_ACTIONS_OPERATIONS="${JUNO_ACTIONS_OPERATIONS:-1}"
export JUNO_ACTIONS_VAGUE_TIME="${JUNO_ACTIONS_VAGUE_TIME:-1}"
export JUNO_ACTIONS_CONTAINERS="${JUNO_ACTIONS_CONTAINERS:-1}"
export JUNO_ACTIONS_COMPOUND="${JUNO_ACTIONS_COMPOUND:-1}"

"$PY" -m juno_v2.runtime.service \
  --mode live \
  --preview-backend streaming_local_http_json \
  --preview-endpoint "$PREVIEW_ENDPOINT" \
  --preview-model-path "$PREVIEW_MODEL" \
  --final-backend mlx_whisper \
  --final-model-path mlx-community/whisper-large-v3-turbo \
  --final-hf-repo-id mlx-community/whisper-large-v3-turbo \
  --writer-backend mlx_lm \
  --writer-model-path mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --gpu-memory-budget-mb 12000 \
  --preview-gpu-memory-mb 4200 \
  --final-gpu-memory-mb 4200 \
  --writer-gpu-memory-mb 2600 \
  --writer-residency-policy resident \
  --language en \
  --language-policy fixed \
  --speech-profile standard \
  --serve-workbench \
  --workbench-host 127.0.0.1 \
  --workbench-port 8765 \
  "$@"
status=$?
exit "$status"
