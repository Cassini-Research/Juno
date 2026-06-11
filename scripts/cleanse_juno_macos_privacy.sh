#!/usr/bin/env bash
# Reset Juno's local macOS privacy state for fresh end-to-end testing.
#
# This intentionally targets Juno's bundle IDs and helper code-signing IDs.
# It does not reset every app's microphone/accessibility permissions.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${APP:-$ROOT/dist/Juno.app}"
LABEL="${JUNO_LAUNCHD_LABEL:-com.juno.voice-engine}"
GUI_DOMAIN="gui/$(id -u)"

usage() {
  echo "Usage: APP=/path/to/Juno.app $0"
  echo
  echo "What it does:"
  echo "  - quits Juno and stops the Juno voice-engine LaunchAgent if loaded"
  echo "  - resets Microphone, Accessibility, and Screen Recording for Juno"
  echo "  - resets Accessibility for packaged Juno helper binaries"
  echo "  - removes quarantine metadata from the local test app bundle"
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is macOS-only." >&2
  exit 2
fi

if [[ ! -d "$APP" ]]; then
  echo "Juno.app not found: $APP" >&2
  echo "Build first, or pass APP=/path/to/Juno.app." >&2
  exit 2
fi

echo "== stopping Juno =="
osascript -e 'tell application "Juno" to quit' >/dev/null 2>&1 || true
launchctl bootout "$GUI_DOMAIN/$LABEL" >/dev/null 2>&1 || true
pkill -x Juno >/dev/null 2>&1 || true
for _helper in juno-capability juno-host juno-hotkey juno-paste juno-textmon; do
  pkill -x "$_helper" >/dev/null 2>&1 || true
done
pkill -f "juno_v2.runtime.service" >/dev/null 2>&1 || true

# Give the UI process a moment to exit before TCC rows are reset.
sleep 1
pkill -9 -x Juno >/dev/null 2>&1 || true

echo "== app identity =="
APP_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP/Contents/Info.plist")"
echo "app: $APP"
echo "bundle id: $APP_ID"

FAILED_RESETS=0
tcc_missing_client() {
  local out="$1"
  [[ "$out" == *"No such bundle identifier"* || "$out" == *"OSStatus error -10814"* ]]
}

reset_service() {
  local service="$1"
  local client="$2"
  local optional="${3:-0}"
  echo "tccutil reset $service $client"
  local out
  if ! out="$(tccutil reset "$service" "$client" 2>&1)"; then
    if tcc_missing_client "$out"; then
      echo "info: no registered TCC row exists for $client ($service); already clean." >&2
      return 0
    fi
    echo "warning: tccutil reset $service $client failed: $out" >&2
    if [[ "$optional" != 1 ]]; then
      FAILED_RESETS=$((FAILED_RESETS + 1))
    fi
  fi
}

flush_tccd_cache() {
  killall -HUP tccd 2>/dev/null || true
  sleep 1
}

echo "== resetting app privacy rows =="
reset_service Microphone "$APP_ID"
reset_service Accessibility "$APP_ID"
reset_service ScreenCapture "$APP_ID"
# Voice Actions: Notes runs over AppleEvents (Automation), Reminders/Alarm
# runs over EventKit. Without these resets a previous "Don't Allow" sticks
# and the in-app Allow button silently no-ops because TCC already returned
# denied to AEDeterminePermissionToAutomateTarget — the symptom that made
# Voice Actions look broken even after a "fresh" reset.
reset_service AppleEvents "$APP_ID"
reset_service Reminders "$APP_ID"
reset_service Calendar "$APP_ID"
reset_service All "$APP_ID"

echo "== resetting helper accessibility rows =="
while IFS= read -r helper; do
  [[ -x "$helper" ]] || continue
  helper_codesign="$(codesign -dv "$helper" 2>&1 || true)"
  helper_id="$(awk -F= '/Identifier=/ {print $2; exit}' <<<"$helper_codesign")"
  if [[ -n "$helper_id" ]]; then
    reset_service Accessibility "$helper_id" 1
  fi
done < <(find "$APP/Contents/MacOS" -maxdepth 1 -type f -name 'juno-*' | sort)

echo "== flushing TCC daemon cache =="
flush_tccd_cache

echo "== clearing quarantine on this test bundle =="
xattr -dr com.apple.quarantine "$APP" >/dev/null 2>&1 || true

if [[ "$FAILED_RESETS" -gt 0 ]]; then
  echo
  echo "warning: $FAILED_RESETS TCC reset command(s) failed." >&2
  echo "         macOS may keep existing Microphone/Accessibility/Screen Recording permissions." >&2
  echo "         Try running from Terminal/Conductor with Full Disk Access, or manually remove Juno from System Settings > Privacy & Security." >&2
fi

echo
echo "Cleanse complete."
echo "Next:"
echo "  1. Open exactly this app: open \"$APP\""
echo "  2. Grant Microphone, Accessibility, and optional Screen Recording again when prompted."
echo "  3. If Accessibility still shows an old Juno row, remove it with the minus button and add this exact app bundle."
