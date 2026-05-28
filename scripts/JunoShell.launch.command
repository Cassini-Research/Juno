#!/bin/bash
# Double-click in Finder, or: open scripts/JunoShell.launch.command
# Starts the Juno voice engine (workbench + ASR) then the Juno menu-bar app.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi

RELEASE="${ROOT}/shells/macos/.build/release"
JUNO_BIN="${RELEASE}/Juno"

# ---- Build if binary is missing ------------------------------------------------------------
if [[ ! -x "$JUNO_BIN" ]]; then
  echo "Juno binary not found — building release now (this takes ~30 s)…"
  (cd "$ROOT/shells/macos" && swift build -c release)
fi

# ---- Resolve ASR model (mlx_whisper preferred; fall back to faster_whisper) ----------------
# ``find`` on a missing HF cache dir exits 1; with ``pipefail`` that would abort the script
# before we can try the next candidate — only run ``find`` when the parent path exists.
MLX_MODEL_DIR="$HOME/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots"
MLX_MODEL=""
if [[ -d "$MLX_MODEL_DIR" ]]; then
  MLX_MODEL=$(find "$MLX_MODEL_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
fi

FW_TURBO_DIR="$HOME/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots"
FW_BASE_DIR="$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots"
FW_SMALL_DIR="$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots"
FW_MODEL=""
if [[ -d "$FW_SMALL_DIR" ]]; then
  FW_MODEL=$(find "$FW_SMALL_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
fi
if [[ -z "$FW_MODEL" && -d "$FW_TURBO_DIR" ]]; then
  FW_MODEL=$(find "$FW_TURBO_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
fi
if [[ -z "$FW_MODEL" && -d "$FW_BASE_DIR" ]]; then
  FW_MODEL=$(find "$FW_BASE_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
fi

# ---- Kill any stale workbench, start fresh with ASR backend --------------------------------
pkill -f "juno_v2.workbench.server" 2>/dev/null || true
sleep 0.5

if [[ -n "$MLX_MODEL" ]]; then
  echo "Starting workbench with mlx_whisper…"
  BACKEND=mlx_whisper
  MODEL="$MLX_MODEL"
elif [[ -n "$FW_MODEL" ]]; then
  echo "Starting workbench with faster_whisper (fallback)…"
  BACKEND=faster_whisper
  MODEL="$FW_MODEL"
else
  echo "WARNING: No ASR model found locally — transcription will fail until a model is downloaded."
  BACKEND=""
  MODEL=""
fi

if [[ -n "$BACKEND" ]]; then
  JUNO_FINAL_BACKEND="$BACKEND" \
  JUNO_FINAL_MODEL_PATH="$MODEL" \
  JUNO_FINAL_LANGUAGE=en \
  PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
    "$PYTHON" -m juno_v2.workbench.server --host 127.0.0.1 --port 8765 \
    >> /tmp/juno-workbench.log 2>&1 &
  WB_PID=$!
  echo "Workbench started (PID $WB_PID). Logs: /tmp/juno-workbench.log"
else
  PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
    "$PYTHON" -m juno_v2.workbench.server --host 127.0.0.1 --port 8765 \
    >> /tmp/juno-workbench.log 2>&1 &
fi

# Wait up to 10 s for the workbench to accept connections
for i in $(seq 1 20); do
  if curl -sf --max-time 1 http://127.0.0.1:8765/healthz >/dev/null 2>&1; then
    echo "Workbench ready."
    break
  fi
  sleep 0.5
done

# ---- Launch Juno menu-bar app ---------------------------------------------------------------
export PATH="${RELEASE}:${PATH:-}"
echo "Starting Juno…"
# Do not ``exec`` — if Juno exits immediately (e.g. single-instance handoff), the shell
# can print a hint instead of looking like a silent crash.
"$JUNO_BIN"
code=$?
if [[ "$code" -eq 0 ]]; then
  # Normal quit, or duplicate launch (see stderr from JunoSingleInstance).
  :
else
  echo "Juno exited with code $code. If dictation fails, check /tmp/juno-workbench.log" >&2
fi
exit "$code"
