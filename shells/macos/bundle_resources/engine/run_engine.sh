#!/usr/bin/env bash
# Juno bundled-engine entry.
# Installed as Juno.app/Contents/Resources/engine/run_engine.sh after
# ./scripts/build_juno_engine_bundle.sh populates this directory.
#
# Layout (typical):
#   engine/run_engine.sh          (this file)
#   engine/.venv/bin/python
#   engine/site/juno_v2/ ...     (packages copied or pip install --target)
#
set -euo pipefail
ENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ENGINE_ROOT"

if [[ -x "${ENGINE_ROOT}/.venv/bin/python" ]]; then
  PY="${ENGINE_ROOT}/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

# Bundle identifier propagated from the Swift parent so all on-disk paths
# (support root, runtime dir, lock, socket, token) key off the same id.
# Default kept in lockstep with shells/macos/Sources/JunoShell/JunoShellInfo.plist.
export JUNO_BUNDLE_ID="${JUNO_BUNDLE_ID:-com.juno.shell}"

# The engine runs from inside the signed app bundle. Python bytecode writes
# inside Contents/Resources mutate the bundle after signing, breaking the
# resource seal and potentially changing the identity macOS uses for privacy
# grants. Prefer no bytecode, and redirect any library-forced cache writes to
# the user's cache root outside Juno.app.
export PYTHONDONTWRITEBYTECODE=1
PY_BYTECODE_CACHE_ROOT="${HOME}/Library/Caches/${JUNO_BUNDLE_ID}/python-bytecode"
mkdir -p "${PY_BYTECODE_CACHE_ROOT}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${PY_BYTECODE_CACHE_ROOT}}"

# mlx_whisper.audio.load_audio shells out to `ffmpeg` (mlx_whisper/audio.py
# line 44-46). Apps launched via macOS LaunchServices (double-click,
# `open Juno.app`, Spotlight, Dock) inherit a minimal PATH —
# `/usr/bin:/bin:/usr/sbin:/sbin` — which doesn't include the standard
# package-manager prefixes. Without ffmpeg the engine emits
# `oneshot_transcribe_error: FileNotFoundError: ffmpeg` for every
# utterance and the History entries land with empty transcripts +
# `failure_reason: transcribe_failed`.
#
# Prefer a bundled ffmpeg (engine/bin/ffmpeg) when present so the
# self-contained install ships ffmpeg without external dependencies.
# Otherwise fall back to extending PATH with the common Homebrew /
# MacPorts prefixes so users who already have ffmpeg installed still
# work without a manual env tweak.
if [[ -x "${ENGINE_ROOT}/bin/ffmpeg" ]]; then
  export PATH="${ENGINE_ROOT}/bin:${PATH}"
fi
for p in /opt/homebrew/bin /usr/local/bin /opt/local/bin; do
  if [[ -d "$p" ]] && [[ ":${PATH}:" != *":${p}:"* ]]; then
    export PATH="${PATH}:${p}"
  fi
done

export JUNO_ENGINE_BUNDLE_ROOT="${ENGINE_ROOT}"

# Point the seed personalization runtime at the bundled seed_data directory
# that build_juno_engine_bundle.sh ships alongside .venv. Without this override
# juno_v2.personalization.seed.paths.repo_root_from_juno() resolves to
# Juno.app/Contents/Resources/engine/.venv/lib/python*/site-packages — where
# the seed dir isn't installed by pip — and the runtime logs "bundle dir
# missing" + silently disables vocab/style seeding.
if [[ -d "${ENGINE_ROOT}/seed_data" ]]; then
  export JUNO_SEED_BUNDLE_DIR="${ENGINE_ROOT}/seed_data"
fi

# Default logs when launched from Juno.app (not a repo checkout).
LOG_ROOT="${HOME}/Library/Logs/Juno"
mkdir -p "${LOG_ROOT}"
WB_LOG="${LOG_ROOT}/bundled-engine.log"

HOST="${JUNO_WORKBENCH_HOST:-127.0.0.1}"
PORT="${JUNO_WORKBENCH_PORT:-8765}"

# Production transport: framed JSON-RPC over a Unix domain socket.
# Default path mirrors juno_v2.runtime.paths.juno_engine_socket_path()
# so the Swift shell and Python engine agree on the location. The Swift
# parent may override JUNO_ENGINE_SOCKET to keep multiple installs
# (dev / staging / prod) from colliding on the same socket.
SUPPORT_PARENT="${HOME}/Library/Application Support"
SUPPORT_ROOT="${JUNO_APP_SUPPORT_DIR:-${SUPPORT_PARENT}/${JUNO_BUNDLE_ID}}"
RUNTIME_DIR="${SUPPORT_ROOT}/runtime"
SERVICE_LOG_DIR="${SUPPORT_ROOT}/logs/service"
MEMORY_DIR="${SUPPORT_ROOT}/memory"
mkdir -p "${RUNTIME_DIR}"
mkdir -p "${SERVICE_LOG_DIR}" "${MEMORY_DIR}"
chmod 700 "${RUNTIME_DIR}" 2>/dev/null || true
ENGINE_SOCKET="${JUNO_ENGINE_SOCKET:-${RUNTIME_DIR}/engine.sock}"
export JUNO_ENGINE_SOCKET="${ENGINE_SOCKET}"
export JUNO_V2_RUNTIME_DIR="${RUNTIME_DIR}"
export JUNO_V2_LOG_DIR="${SERVICE_LOG_DIR}"
export JUNO_V2_MEMORY_DIR="${MEMORY_DIR}"

# Full runtime.service mode with an isolated resident preview service.
# Live preview is driven by MLX Whisper large-v3-turbo on checkpointed
# utterance windows. The shell streams audio to the broker, the broker forwards
# preview chunks to this local preview process. The writer/action model remains
# on the final path by default; the live HUD stream should not be rewritten
# by a language model while the user is still speaking.
export JUNO_V2_MODE=live
export JUNO_V2_PREVIEW_BACKEND=streaming_local_http_json
# Hold back HUD commits for words ending inside the last 600ms of buffered
# audio — Whisper's truncated-decode zone, where stable hallucinated
# continuations come from ("…hey Juno" + cut speech → "I don't know.").
# Demoted words re-commit one decode later once more audio confirms them.
export JUNO_V2_PREVIEW_COMMIT_DRAFT_HORIZON_MS="${JUNO_V2_PREVIEW_COMMIT_DRAFT_HORIZON_MS:-600}"
export JUNO_V2_PREVIEW_MODEL_PATH=mlx-community/whisper-large-v3-turbo
export JUNO_V2_PREVIEW_HF_REPO_ID=mlx-community/whisper-large-v3-turbo
export JUNO_V2_PREVIEW_SERVICE_BACKEND=mlx_whisper
export JUNO_V2_FINAL_BACKEND=mlx_whisper
export JUNO_V2_FINAL_MODEL_PATH=mlx-community/whisper-large-v3-turbo
export JUNO_V2_FINAL_HF_REPO_ID=mlx-community/whisper-large-v3-turbo
export JUNO_V2_WRITER_BACKEND=mlx_lm
export JUNO_V2_WRITER_MODEL_PATH=mlx-community/Qwen3-4B-Instruct-2507-4bit
export JUNO_V2_TURN_PLANNER_ENABLED="${JUNO_V2_TURN_PLANNER_ENABLED:-1}"
export JUNO_V2_LIVE_CORRECTOR_ENABLED="${JUNO_V2_LIVE_CORRECTOR_ENABLED:-1}"
export JUNO_V2_LIVE_CORRECTOR_BACKEND=mlx_lm
export JUNO_V2_LIVE_CORRECTOR_MODEL_PATH=mlx-community/Qwen3-0.6B-4bit
export JUNO_V2_LIVE_CORRECTOR_MAX_TOKENS=160
export JUNO_V2_LIVE_CORRECTOR_TEMPERATURE=0.0
export JUNO_V2_LIVE_CORRECTOR_TOP_P=1.0
export JUNO_V2_LIVE_CORRECTOR_RESIDENCY_POLICY=resident
# Writer is resident in the packaged app. The Qwen3-4B writer/planner pays a
# 3-5s MLX cold start on every reload; with on_demand residency that cost
# landed on nearly every utterance because real pauses between dictations
# exceed any reasonable idle TTL. Resident is the documented production
# decision (2026-05-11) for an interactive dictation product on 16GB+ Macs.
export JUNO_V2_WRITER_RESIDENCY_POLICY=resident
# Old live-WAV adjudication is disabled. Final stop still uses Whisper + the
# 4B writer/action lane.
export JUNO_V2_LIVE_QWEN_ADJUDICATION=0
# When ON, the writer-LLM action extractor emits actions_intent_v3
# envelopes with Schedule (instant/series/vague) per action. Recurring
# intents ("for the next 10 days", "every weekday at 9") become a
# single reminder with EKRecurrenceRule instead of one instant. v2
# envelopes still validate unconditionally so older code paths are
# unaffected; flip to 0 to bisect any v3-specific regression.
export JUNO_ACTIONS_SCHEMA_V3="${JUNO_ACTIONS_SCHEMA_V3:-1}"
# The actions index records every executed action under a stable Juno id.
# Launch scope (2026-06-10): CREATION ONLY — operations on existing actions
# (update/complete/snooze/delete) and the followup correction window are OFF
# for go-live and tracked in docs/JUNO_TODO_POST_LAUNCH.md.
export JUNO_ACTIONS_INDEX="${JUNO_ACTIONS_INDEX:-1}"
export JUNO_ACTIONS_OPERATIONS="${JUNO_ACTIONS_OPERATIONS:-0}"
export JUNO_ACTIONS_VAGUE_TIME="${JUNO_ACTIONS_VAGUE_TIME:-1}"
export JUNO_ACTIONS_FOLLOWUP="${JUNO_ACTIONS_FOLLOWUP:-0}"
# Dictation editor: the AI lane for every non-wake dictation turn — one
# cached-prefix Qwen pass emitting an anchored edit script (corrections,
# restarts/fillers, spoken structure), applied deterministically. Floor on
# parse/deadline failure is the deterministic pipeline text, logged.
export JUNO_V2_DICTATION_EDITOR="${JUNO_V2_DICTATION_EDITOR:-1}"
export JUNO_V2_DICTATION_EDITOR_DEADLINE_MS="${JUNO_V2_DICTATION_EDITOR_DEADLINE_MS:-12000}"
export JUNO_ACTIONS_CONTAINERS="${JUNO_ACTIONS_CONTAINERS:-1}"
export JUNO_ACTIONS_COMPOUND="${JUNO_ACTIONS_COMPOUND:-1}"
export JUNO_V2_GPU_MEMORY_BUDGET_MB=12000
export JUNO_V2_PREVIEW_GPU_MEMORY_MB=4200
export JUNO_V2_FINAL_GPU_MEMORY_MB=4200
export JUNO_V2_WRITER_GPU_MEMORY_MB=2600
export JUNO_V2_LIVE_CORRECTOR_GPU_MEMORY_MB="${JUNO_V2_LIVE_CORRECTOR_GPU_MEMORY_MB:-800}"
export JUNO_V2_LANGUAGE=en
export JUNO_V2_LANGUAGE_POLICY=fixed
export JUNO_V2_SPEECH_PROFILE=standard
if [[ -n "${JUNO_DEV_WORKBENCH_HTTP:-}" ]]; then
  export JUNO_V2_SERVE_WORKBENCH=1
  export JUNO_V2_WORKBENCH_HOST="${HOST}"
  export JUNO_V2_WORKBENCH_PORT="${PORT}"
else
  export JUNO_V2_SERVE_WORKBENCH=0
fi

if "${PY}" -c "import juno_v2" 2>/dev/null; then
  PREVIEW_HOST="${JUNO_V2_PREVIEW_SERVICE_HOST:-127.0.0.1}"
  PREVIEW_PORT="$("${PY}" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"
  PREVIEW_ENDPOINT="http://${PREVIEW_HOST}:${PREVIEW_PORT}"
  export JUNO_V2_PREVIEW_ENDPOINT="${PREVIEW_ENDPOINT}"
  PREVIEW_LOG="${LOG_ROOT}/preview-service.log"
  PREVIEW_SUPERVISOR_PID=""
  RUNTIME_PID=""

  cleanup() {
    if [[ -n "${RUNTIME_PID}" ]]; then
      kill "${RUNTIME_PID}" 2>/dev/null || true
      wait "${RUNTIME_PID}" 2>/dev/null || true
    fi
    if [[ -n "${PREVIEW_SUPERVISOR_PID}" ]]; then
      kill "${PREVIEW_SUPERVISOR_PID}" 2>/dev/null || true
      wait "${PREVIEW_SUPERVISOR_PID}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  # >>> juno-preview-supervisor (extracted verbatim by tests/test_run_engine_preview_supervision.py)
  supervise_preview_service() {
    # The streaming preview service is a resident helper the runtime calls
    # over loopback HTTP for every live-caption chunk. A one-shot spawn
    # turns any death (signal, crash, OOM) into permanently dead live
    # preview: the runtime keeps the stale endpoint, health reporting keeps
    # saying ok, every chunk fails connection-refused, and only a full
    # engine-tree restart recovers (observed 2026-06-12). Restart forever
    # with bounded backoff so a boot-crash loop cannot peg the GPU, and log
    # every exit with status + uptime for post-mortem attribution.
    local child=""
    trap 'if [[ -n "${child}" ]]; then kill "${child}" 2>/dev/null || true; fi; exit 0' TERM INT
    local backoff=1
    while :; do
      local started_at
      started_at="$(date +%s)"
      "${PY}" -m juno_v2.preview.streaming_service \
        --backend "${JUNO_V2_PREVIEW_SERVICE_BACKEND}" \
        --host "${PREVIEW_HOST}" \
        --port "${PREVIEW_PORT}" \
        --model-path "${JUNO_V2_PREVIEW_MODEL_PATH}" \
        --hf-repo-id "${JUNO_V2_PREVIEW_HF_REPO_ID}" \
        --device auto \
        --compute-type default >>"${PREVIEW_LOG}" 2>&1 &
      child="$!"
      local status=0
      wait "${child}" || status=$?
      child=""
      local ended_at
      ended_at="$(date +%s)"
      if (( ended_at - started_at >= 30 )); then
        backoff=1
      else
        backoff=$(( backoff * 2 ))
        if (( backoff > 30 )); then backoff=30; fi
      fi
      echo "[supervisor] $(date -u +%Y-%m-%dT%H:%M:%SZ) preview service exited status=${status} uptime=$(( ended_at - started_at ))s; restarting in ${backoff}s" >>"${PREVIEW_LOG}"
      # Background sleep + wait keeps TERM delivery prompt: bash defers
      # traps while a foreground command runs, but interrupts `wait`.
      sleep "${backoff}" &
      wait "$!" || true
    done
  }
  # <<< juno-preview-supervisor

  supervise_preview_service &
  PREVIEW_SUPERVISOR_PID="$!"

  "${PY}" - <<PY
import json, sys, time
from urllib import request
url = "${PREVIEW_ENDPOINT}/healthz"
deadline = time.time() + 120
last = None
while time.time() < deadline:
    try:
        with request.urlopen(url, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("ok"):
            sys.exit(0)
    except Exception as exc:
        last = exc
    time.sleep(0.25)
raise SystemExit(f"preview service did not become healthy: {last}")
PY

  ARGS=(
    -m juno_v2.runtime.service
    --mode live \
    --preview-backend streaming_local_http_json \
    --preview-model-path mlx-community/whisper-large-v3-turbo \
    --preview-endpoint "${PREVIEW_ENDPOINT}" \
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
    --runtime-dir "${RUNTIME_DIR}" \
    --log-dir "${SERVICE_LOG_DIR}" \
    --memory-dir "${MEMORY_DIR}" \
    --engine-socket "${ENGINE_SOCKET}"
  )
  case "${JUNO_V2_LIVE_CORRECTOR_ENABLED}" in
    1|true|TRUE|yes|YES|on|ON)
      ARGS+=(
        --live-corrector-enabled \
        --live-corrector-backend "${JUNO_V2_LIVE_CORRECTOR_BACKEND}" \
        --live-corrector-model-path "${JUNO_V2_LIVE_CORRECTOR_MODEL_PATH}" \
        --live-corrector-max-tokens "${JUNO_V2_LIVE_CORRECTOR_MAX_TOKENS}" \
        --live-corrector-residency-policy "${JUNO_V2_LIVE_CORRECTOR_RESIDENCY_POLICY}" \
        --live-corrector-gpu-memory-mb "${JUNO_V2_LIVE_CORRECTOR_GPU_MEMORY_MB}"
      )
      ;;
  esac
  if [[ -n "${JUNO_DEV_WORKBENCH_HTTP:-}" ]]; then
    ARGS+=(--serve-workbench --workbench-host "${HOST}" --workbench-port "${PORT}")
  fi
  "${PY}" "${ARGS[@]}" >>"${WB_LOG}" 2>&1 &
  RUNTIME_PID="$!"
  wait "${RUNTIME_PID}"
  status=$?
  RUNTIME_PID=""
  exit "${status}"
fi

echo "error: juno_v2 is not importable under ${ENGINE_ROOT} (run scripts/build_juno_engine_bundle.sh)" >&2
exit 3
