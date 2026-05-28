#!/usr/bin/env bash
# Refresh LaunchServices registration for the Juno.app bundle under test.
#
# Local development commonly leaves multiple com.juno.shell rows behind
# (dist/Juno.app, /Applications/Juno.app, older build products). TCC and
# System Settings are much less confusing when the installed app path is the
# row LaunchServices prefers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${APP:-$ROOT/dist/Juno.app}"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
EVICT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --evict-only) EVICT_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--evict-only]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$LSREGISTER" ]]; then
  echo "lsregister not found: $LSREGISTER" >&2
  exit 2
fi

if [[ "$EVICT_ONLY" == 1 ]]; then
  echo "LaunchServices evict: common Juno bundle paths"
else
  if [[ ! -d "$APP" ]]; then
    echo "Juno.app not found: $APP" >&2
    exit 2
  fi
  echo "LaunchServices refresh: $APP"
fi

# Unregister the common local bundle paths first. Ignore misses; the goal is
# deterministic registration, not strict cleanup of every historical path.
for candidate in \
  "$ROOT/dist/Juno.app" \
  "/Applications/Juno.app"
do
  if [[ -d "$candidate" ]]; then
    "$LSREGISTER" -u "$candidate" >/dev/null 2>&1 || true
  fi
done

if [[ "$EVICT_ONLY" != 1 ]]; then
  "$LSREGISTER" -f "$APP"
fi
