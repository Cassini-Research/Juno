#!/usr/bin/env bash
# Build (optional), then start Juno without opening a pile of Terminal windows.
#
# Juno is the menu-bar + Dock app; helpers (juno-hotkey, juno-paste, …) sit next to this binary.
#
# Why not only run from Cursor? Some agent environments lack a normal GUI session;
# use Terminal.app or Finder if the process starts but you see no UI.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REL_BIN="$ROOT/shells/macos/.build/release/Juno"
CMD_FILE="$ROOT/scripts/JunoShell.launch.command"

usage() {
  cat <<EOF
Build (optional) and launch Juno (no new Terminal tab by default).

Usage:
  $0 [--build] [--here] [--terminal]

Options:
  --build      Run swift build -c release first
  --here       Run Juno in this shell (foreground; good from Terminal/iTerm)
  --terminal   Open Terminal.app and run the .command file (legacy / debugging)
EOF
}

DO_BUILD=0
HERE=0
USE_TERMINAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; shift ;;
    --here) HERE=1; shift ;;
    --terminal) USE_TERMINAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$DO_BUILD" == 1 ]]; then
  echo "Building Juno (swift build -c release)…"
  (cd "$ROOT/shells/macos" && swift build -c release)
fi

if [[ ! -x "$REL_BIN" ]]; then
  echo "Missing $REL_BIN — run with --build first." >&2
  exit 1
fi

if [[ "$HERE" == 1 ]]; then
  exec "$REL_BIN"
fi

if [[ "$USE_TERMINAL" == 1 ]]; then
  chmod +x "$CMD_FILE" 2>/dev/null || true
  osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "cd \"$ROOT\" && open \"$CMD_FILE\""
end tell
APPLESCRIPT
  echo "Launched via Terminal (scripts/JunoShell.launch.command). Check the menu bar / Dock for Juno."
  exit 0
fi

# Default: LaunchServices opens the GUI binary in the user session — no new Terminal window.
if open -n -g "$REL_BIN" 2>/dev/null; then
  echo "Started Juno (menu bar / Dock). Quit from the Juno menu before relaunching."
  exit 0
fi

# Fallback if \`open\` is unavailable or rejects the binary: background exec, still no Terminal tab.
( cd "$ROOT" && exec "$REL_BIN" ) </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true
echo "Started Juno in background (open unavailable). Check the menu bar / Dock."
