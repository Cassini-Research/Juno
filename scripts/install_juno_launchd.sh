#!/usr/bin/env bash
set -euo pipefail

# Installs a per-user LaunchAgent that runs the local Juno voice engine
# and serves the broker/workbench HTTP API on 127.0.0.1:8765.
#
# This is intentionally a developer/operator script. Packaged Juno.app may ship
# its own installer later; this script is the canonical way to set up launchd
# for a source checkout.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

LABEL="com.juno.voice-engine"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="${HOME}/Library/Logs/Juno"
STDOUT_LOG="${LOG_DIR}/voice-engine.out.log"
STDERR_LOG="${LOG_DIR}/voice-engine.err.log"

usage() {
  echo "Usage: $0 <install|uninstall|start|stop|restart|status> [--repo-root PATH] [--engine-bundle PATH]"
  echo "  --engine-bundle PATH   Run bundled engine (run_engine.sh + venv) instead of repo checkout."
  exit 1
}

ACTION="${1:-}"
shift || true

REPO_ROOT="$ROOT"
ENGINE_BUNDLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --engine-bundle) ENGINE_BUNDLE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [[ -z "$ACTION" ]]; then
  usage
fi

mkdir -p "$(dirname "$PLIST_PATH")" "$LOG_DIR"

GUI_DOMAIN="gui/$(id -u)"

write_plist() {
  local wd
  local run_cmd
  if [[ -n "$ENGINE_BUNDLE" ]]; then
    wd="$ENGINE_BUNDLE"
    if [[ ! -x "${ENGINE_BUNDLE}/run_engine.sh" ]]; then
      echo "error: missing executable ${ENGINE_BUNDLE}/run_engine.sh (build with scripts/build_juno_engine_bundle.sh)" >&2
      exit 2
    fi
    run_cmd="exec &quot;${ENGINE_BUNDLE}/run_engine.sh&quot;"
  else
    wd="$REPO_ROOT"
    run_cmd="cd &quot;${REPO_ROOT}&quot; &amp;&amp; exec ./scripts/run_live.sh"
  fi
  cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>WorkingDirectory</key><string>${wd}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>${run_cmd}</string>
  </array>
  <key>StandardOutPath</key><string>${STDOUT_LOG}</string>
  <key>StandardErrorPath</key><string>${STDERR_LOG}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>JUNO_V2_SERVE_WORKBENCH</key><string>1</string>
    <key>JUNO_V2_WORKBENCH_HOST</key><string>127.0.0.1</string>
    <key>JUNO_V2_WORKBENCH_PORT</key><string>8765</string>
    <key>JUNO_WORKBENCH_HOST</key><string>127.0.0.1</string>
    <key>JUNO_WORKBENCH_PORT</key><string>8765</string>
  </dict>
</dict>
</plist>
EOF
}

bootstrap_if_needed() {
  if [[ -n "$ENGINE_BUNDLE" ]]; then
    if [[ ! -x "${ENGINE_BUNDLE}/.venv/bin/python" ]]; then
      echo "Missing ${ENGINE_BUNDLE}/.venv — run ./scripts/build_juno_engine_bundle.sh first." >&2
      exit 2
    fi
    return
  fi
  if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "Missing .venv. Run ./scripts/bootstrap.sh first." >&2
    exit 2
  fi
}

launchctl_bootstrap() {
  launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
  launchctl bootstrap "$GUI_DOMAIN" "$PLIST_PATH"
  launchctl enable "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
}

case "$ACTION" in
  install)
    bootstrap_if_needed
    write_plist
    launchctl_bootstrap
    launchctl kickstart -k "$GUI_DOMAIN/$LABEL" || true
    echo "Installed: $PLIST_PATH"
    ;;
  uninstall)
    launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo "Uninstalled: $PLIST_PATH"
    ;;
  start)
    launchctl kickstart -k "$GUI_DOMAIN/$LABEL"
    ;;
  stop)
    launchctl kill SIGTERM "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
    ;;
  restart)
    launchctl kickstart -k "$GUI_DOMAIN/$LABEL"
    ;;
  status)
    launchctl print "$GUI_DOMAIN/$LABEL" 2>/dev/null || {
      echo "Not loaded: $LABEL"
      exit 3
    }
    ;;
  *)
    usage
    ;;
esac
