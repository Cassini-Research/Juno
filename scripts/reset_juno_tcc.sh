#!/usr/bin/env bash
# Print / run macOS TCC resets for Juno so a new build can re-prompt permissions.
#
# Quit Juno completely first (menu bar → Quit). Legacy Juno builds used a
# different bundle id — use --juno with --run to clear those rows too.
#
# Usage:
#   ./scripts/reset_juno_tcc.sh                 # print Juno commands only
#   ./scripts/reset_juno_tcc.sh --juno          # print Juno + legacy Juno commands
#   ./scripts/reset_juno_tcc.sh --run           # execute Juno resets
#   ./scripts/reset_juno_tcc.sh --run --juno
#   ./scripts/reset_juno_tcc.sh --run --all     # reset every TCC service for the app id
#   ./scripts/reset_juno_tcc.sh --run --all --strict
set -euo pipefail

BUNDLE="${JUNO_BUNDLE_ID:-com.juno.shell}"
LEGACY_JUNO="${JUNO_LEGACY_BUNDLE_ID:-com.juno.shell}"

RUN=0
JUNO=0
ALL=0
STRICT=0
FAILURES=0
for a in "$@"; do
  case "$a" in
    --run) RUN=1 ;;
    --juno) JUNO=1 ;;
    --all) ALL=1 ;;
    --strict) STRICT=1 ;;
    -h|--help)
      echo "Usage: $0 [--juno] [--run] [--all] [--strict]"
      exit 0
      ;;
  esac
done

reset_bundle_print() {
  local b="$1"
  echo "Target bundle: $b"
  echo "  tccutil reset Microphone \"$b\""
  echo "  tccutil reset Accessibility \"$b\""
  echo "  tccutil reset ScreenCapture \"$b\""
  if [[ "$ALL" == 1 ]]; then
    echo "  tccutil reset All \"$b\""
  else
    echo "  # Stronger fresh-install reset: $0 --run --all"
  fi
}

reset_bundle_run() {
  local b="$1"
  # Microphone + dictation
  reset_service_run Microphone "$b"
  reset_service_run ScreenCapture "$b"
  # Keystroke + selection injection
  reset_service_run Accessibility "$b"
  # Voice Actions: Apple Events (Notes Automation) + EventKit (Reminders, Calendar)
  reset_service_run AppleEvents "$b"
  reset_service_run Reminders "$b"
  reset_service_run Calendar "$b"
  if [[ "$ALL" == 1 ]]; then
    reset_service_run All "$b"
  fi
}

# Restart tccd so its in-memory cache is invalidated. Without this the
# rows on disk may be cleared but the live daemon keeps serving the
# previous decision until the next user login — meaning a freshly
# installed Juno still finds the old "granted"/"denied" answer and the
# user never sees the re-prompt this reset was supposed to enable.
flush_tccd_cache() {
  killall -HUP tccd 2>/dev/null || true
  # Brief pause for the daemon to reload TCC.db from disk.
  sleep 1
}

tcc_missing_client() {
  local out="$1"
  [[ "$out" == *"No such bundle identifier"* || "$out" == *"OSStatus error -10814"* ]]
}

reset_service_run() {
  local service="$1"
  local client="$2"
  local optional="${3:-0}"
  local out
  if ! out="$(tccutil reset "$service" "$client" 2>&1)"; then
    if tcc_missing_client "$out"; then
      echo "info: tccutil reset $service $client skipped: no registered TCC row exists." >&2
      return 0
    fi
    echo "warning: tccutil reset $service $client failed: $out" >&2
    if [[ "$optional" != 1 ]]; then
      FAILURES=$((FAILURES + 1))
    fi
  fi
}

echo "Primary bundle: $BUNDLE"
reset_bundle_print "$BUNDLE"
if [[ "$JUNO" == 1 ]]; then
  if [[ -n "$LEGACY_JUNO" && "$LEGACY_JUNO" != "$BUNDLE" ]]; then
    echo
    echo "Legacy bundle (optional): $LEGACY_JUNO"
    reset_bundle_print "$LEGACY_JUNO"
  fi
fi

echo
echo "Helper Accessibility ids (run separately):"
for helper in capability host hotkey paste textmon; do
  echo "  tccutil reset Accessibility \"com.juno.shell.helper.${helper}\""
done

if [[ "$RUN" == 1 ]]; then
  echo
  echo "Executing resets for $BUNDLE ..."
  reset_bundle_run "$BUNDLE"
  if [[ "$JUNO" == 1 && -n "$LEGACY_JUNO" && "$LEGACY_JUNO" != "$BUNDLE" ]]; then
    echo "Executing resets for legacy $LEGACY_JUNO ..."
    reset_bundle_run "$LEGACY_JUNO"
  fi
  for helper in capability host hotkey paste textmon; do
    # Helper binaries are command-line tools, not app bundles registered
    # with LaunchServices, so tccutil may report "No such bundle
    # identifier" even when no stale row exists. Keep this diagnostic
    # visible, but do not make strict fresh-install reset hinge on it.
    reset_service_run Accessibility "com.juno.shell.helper.${helper}" 1
  done
  echo "Flushing tccd cache so the next launch re-prompts ..."
  flush_tccd_cache
  if [[ "$FAILURES" -gt 0 ]]; then
    echo "warning: $FAILURES TCC reset command(s) failed." >&2
    echo "         macOS may keep existing Microphone/Accessibility/Screen Recording permissions." >&2
    echo "         Try running from Terminal/Conductor with Full Disk Access, or manually remove Juno from System Settings > Privacy & Security." >&2
    if [[ "$STRICT" == 1 ]]; then
      exit 70
    fi
  fi
  echo "Done. If prompts still misbehave: sudo tccutil reset All \"$BUNDLE\""
fi
