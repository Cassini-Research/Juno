#!/usr/bin/env bash
# One-shot recovery: re-sign dist/Juno.app with the audio-input entitlement,
# unregister stale LaunchServices entries (e.g. the legacy JunoApp/build path
# that LaunchServices still has cached under bundle id com.juno.shell), reset
# TCC rows for Juno, then relaunch.
#
# Run this when:
#   - Juno never appears in System Settings → Privacy → Microphone, or
#   - the mic prompt never shows even on a fresh build.
#
# Why this is needed:
#   SwiftPM emits adhoc + hardened-runtime binaries with NO entitlements.
#   Under hardened runtime, AVCaptureDevice access requires
#   com.apple.security.device.audio-input — without it, the OS blocks the
#   request before TCC is consulted, so no prompt appears and no row is
#   created in Privacy → Microphone. Adhoc cdhash drift across rebuilds
#   compounds the problem by leaving stale TCC + LS rows behind.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${APP:-$ROOT/dist/Juno.app}"
ENTS="$ROOT/shells/macos/JunoApp/Juno.entitlements"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS only." >&2
  exit 2
fi

if [[ ! -d "$APP" ]]; then
  echo "Juno.app not found: $APP" >&2
  echo "Build first (./scripts/package_juno_macos_app.sh) or pass APP=/path." >&2
  exit 2
fi

if [[ ! -f "$ENTS" ]]; then
  echo "Missing entitlements: $ENTS" >&2
  exit 2
fi

echo "== quitting Juno =="
osascript -e 'tell application "Juno" to quit' >/dev/null 2>&1 || true
pkill -x Juno >/dev/null 2>&1 || true
sleep 1

echo "== re-signing $APP with entitlements =="
for helper in juno-capability juno-host juno-hotkey juno-paste juno-textmon; do
  hp="$APP/Contents/MacOS/$helper"
  if [[ -f "$hp" ]]; then
    codesign --force --options runtime --timestamp=none \
      --entitlements "$ENTS" \
      --identifier "com.juno.shell.helper.${helper#juno-}" \
      --sign - "$hp"
  fi
done
codesign --force --options runtime --timestamp=none \
  --entitlements "$ENTS" \
  --identifier "com.juno.shell" \
  --sign - "$APP"

echo "== verifying entitlements =="
codesign -d --entitlements - "$APP/Contents/MacOS/Juno" 2>&1 \
  | grep -q "com.apple.security.device.audio-input" \
  && echo "  audio-input entitlement: OK" \
  || { echo "  audio-input entitlement: MISSING — abort" >&2; exit 3; }

echo "== unregistering stale LaunchServices entries for com.juno.shell =="
APP="$APP" "$ROOT/scripts/juno_lsregister_refresh.sh" "$APP"

echo "== resetting TCC rows for com.juno.shell =="
for svc in Microphone Accessibility ScreenCapture; do
  echo "  tccutil reset $svc com.juno.shell"
  tccutil reset "$svc" com.juno.shell >/dev/null 2>&1 || true
done
for helper in capability host hotkey paste textmon; do
  tccutil reset Accessibility "com.juno.shell.helper.${helper}" >/dev/null 2>&1 || true
done

echo "== clearing quarantine on this bundle =="
xattr -dr com.apple.quarantine "$APP" >/dev/null 2>&1 || true

echo "== relaunching =="
open "$APP"

echo
echo "Done. On the next launch you should see a real macOS microphone prompt"
echo "for Juno. If you previously denied it, the tccutil reset above cleared"
echo "that decision so the prompt will reappear."
