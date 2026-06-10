#!/usr/bin/env bash
# One-script fresh install for Juno on macOS.
#
# Builds a self-contained engine bundle, packages Juno.app with the bundle
# embedded, and (optionally) installs to /Applications. End state: double-
# clicking Juno.app starts the local voice engine silently — no Terminal
# required for the end user.
#
# Usage:
#   ./scripts/install_juno_macos.sh                        # build + package → dist/Juno.app only
#   ./scripts/install_juno_macos.sh --install-to-apps      # also replace /Applications/Juno.app
#   ./scripts/install_juno_macos.sh --install-to-apps --cleanse
#                                                          # also reset TCC (re-prompts on launch)
#   ./scripts/install_juno_macos.sh --skip-build           # reuse existing dist/juno_engine_bundle
#
# --install-to-apps fully replaces an existing install: it quits Juno, removes the old
# /Applications/Juno.app, then copies the freshly built bundle, runs LaunchServices
# refresh (scripts/juno_lsregister_refresh.sh), then optional --cleanse TCC. (There is a
# single Juno.app; helper binaries ship inside it, not as separate Finder apps.)
#
# The packaged dist/Juno.app is the shippable artifact. Distribute it as a
# .dmg/.zip; users install by dragging to /Applications and double-clicking.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INSTALL_TO_APPS=0
CLEANSE=0
SKIP_BUILD=0
SIGN_ID="${CODESIGN_IDENTITY:--}"
SIGN_ID_EXPLICIT=0

usage() {
  echo "Usage: $0 [--install-to-apps] [--cleanse] [--skip-build] [--sign IDENTITY]"
  echo "  --install-to-apps  Replace /Applications/Juno.app with the new dist/Juno.app (quit + rm + copy)."
  echo "  --cleanse          Reset TCC permissions (Microphone/Accessibility/Screen Recording)"
  echo "                     for com.juno.shell so the next launch re-prompts. Implies a"
  echo "                     fresh-install simulation. Requires --install-to-apps to be"
  echo "                     useful since cleanse targets the installed bundle."
  echo "  --skip-build       Skip ./scripts/build_juno_engine_bundle.sh and reuse"
  echo "                     dist/juno_engine_bundle (must already exist)."
  echo "  --sign IDENTITY    codesign identity (default: \$CODESIGN_IDENTITY or '-' for ad-hoc)."
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-to-apps) INSTALL_TO_APPS=1; shift ;;
    --cleanse) CLEANSE=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --sign) SIGN_ID="$2"; SIGN_ID_EXPLICIT=1; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script is macOS-only" >&2
  exit 2
fi

# Prefer a stable developer cert over ad-hoc so TCC doesn't reset on every rebuild.
# Ad-hoc ("-") gives each rebuild a new cdhash, invalidating Mic/AX grants.
# CI overrides via CODESIGN_IDENTITY or --sign.
if [[ "$SIGN_ID" == "-" && "$SIGN_ID_EXPLICIT" == 0 ]]; then
  STABLE_ID="$(security find-identity -v -p codesigning 2>/dev/null \
    | awk -F'"' '/Developer ID Application/ {print $2; exit}')"
  if [[ -z "$STABLE_ID" ]]; then
    STABLE_ID="$(security find-identity -v -p codesigning 2>/dev/null \
      | awk -F'"' '/Apple Development/ {print $2; exit}')"
  fi
  if [[ -n "$STABLE_ID" ]]; then
    SIGN_ID="$STABLE_ID"
    echo "install: using stable codesign identity: $SIGN_ID"
  else
    echo "install: warning — falling back to ad-hoc signing; macOS will reset TCC on each rebuild." >&2
    echo "install: set CODESIGN_IDENTITY=<Apple Development: …> or pass --sign to keep grants." >&2
  fi
elif [[ "$SIGN_ID" != "-" ]]; then
  if ! security find-identity -v -p codesigning 2>/dev/null \
      | awk -F'"' -v id="$SIGN_ID" '$2 == id { found = 1 } END { exit found ? 0 : 1 }'; then
    echo "install: warning — requested codesign identity is not currently valid: $SIGN_ID" >&2
    echo "install: warning — falling back to ad-hoc signing for this local build." >&2
    SIGN_ID="-"
  fi
fi

BUNDLE_DIR="$ROOT/dist/juno_engine_bundle"
APP_DIR="$ROOT/dist/Juno.app"

echo "==> 1/3 Build self-contained engine bundle"
if [[ "$SKIP_BUILD" == 1 ]]; then
  if [[ ! -x "$BUNDLE_DIR/.venv/bin/python" ]]; then
    echo "error: --skip-build but $BUNDLE_DIR/.venv/bin/python is missing" >&2
    exit 3
  fi
  echo "    skip (reusing $BUNDLE_DIR)"
else
  ./scripts/build_juno_engine_bundle.sh "$BUNDLE_DIR"
fi

echo "==> 2/3 Package Juno.app with bundled engine"
if ! ./scripts/package_juno_macos_app.sh --engine "$BUNDLE_DIR" --sign "$SIGN_ID"; then
  if [[ "$SIGN_ID" != "-" ]]; then
    echo "install: warning — packaging failed with identity: $SIGN_ID" >&2
    echo "install: warning — retrying package with ad-hoc signing for this local build." >&2
    SIGN_ID="-"
    ./scripts/package_juno_macos_app.sh --engine "$BUNDLE_DIR" --sign "$SIGN_ID"
  else
    exit 5
  fi
fi

verify_packaged_app() {
  codesign --verify --deep --strict --verbose=2 "$APP_DIR" >/dev/null 2>&1
}

if ! verify_packaged_app; then
  if [[ "$SIGN_ID" != "-" ]]; then
    echo "install: warning — packaged app failed codesign verification with identity: $SIGN_ID" >&2
    echo "install: warning — re-signing ad-hoc so the local app can launch." >&2
    SIGN_ID="-"
    ./scripts/sign_juno_macos_app.sh \
      --app "$APP_DIR" \
      --identity "$SIGN_ID" \
      --entitlements "$ROOT/shells/macos/JunoApp/Juno.entitlements" \
      --timestamp none
  fi

  if ! verify_packaged_app; then
    echo "install: error — packaged app still fails codesign verification after signing." >&2
    codesign --verify --deep --strict --verbose=4 "$APP_DIR" >&2 || true
    exit 5
  fi
fi

# Sanity check: the bundled-engine path inside the .app must satisfy
# JunoEngineContract.bundledEngineRoot() — it requires run_engine.sh +
# one of (.venv/bin/python | site/juno_v2 | juno_v2). A .app missing
# .venv falls through to repo-checkout fallback at runtime, which is
# exactly the silent-failure mode this script exists to prevent.
if [[ ! -x "$APP_DIR/Contents/Resources/engine/.venv/bin/python" \
   && ! -d "$APP_DIR/Contents/Resources/engine/site/juno_v2" \
   && ! -d "$APP_DIR/Contents/Resources/engine/juno_v2" ]]; then
  echo "error: $APP_DIR is missing a bundled engine (no .venv, no site/juno_v2)." >&2
  echo "       JunoEngineContract.bundledEngineRoot() will return nil and the broker" >&2
  echo "       will silently fall through to repo-checkout fallback at runtime." >&2
  exit 4
fi

if [[ "$INSTALL_TO_APPS" != 1 ]]; then
  echo
  echo "Done. Artifact: $APP_DIR"
  echo "Distribute as .dmg/.zip; users drag to /Applications and double-click."
  echo "To install locally now:  ./scripts/install_juno_macos.sh --install-to-apps"
  exit 0
fi

echo "==> 3/3 Install to /Applications (replace existing)"
osascript -e 'tell application "Juno" to quit' >/dev/null 2>&1 || true
pkill -x Juno >/dev/null 2>&1 || true
for _helper in juno-capability juno-host juno-hotkey juno-paste juno-textmon; do
  pkill -x "$_helper" >/dev/null 2>&1 || true
done
_engine_pids="$(pgrep -f "[j]uno_v2.runtime.service" || true)"
if [[ -n "$_engine_pids" ]]; then
  kill $_engine_pids >/dev/null 2>&1 || true
fi
_engine_launcher_pids="$(pgrep -f "[J]uno.app/Contents/Resources/engine/run_engine.sh" || true)"
if [[ -n "$_engine_launcher_pids" ]]; then
  kill $_engine_launcher_pids >/dev/null 2>&1 || true
fi
sleep 1
# Stale lock from a previous crashed instance can block relaunch
# (followup: validate PID liveness in the Swift single-instance check).
rm -f "$HOME/Library/Application Support/Juno/JunoShell.lock"
rm -rf /Applications/Juno.app
ditto "$APP_DIR" /Applications/Juno.app
# `xattr -r` is not portable across macOS versions. Remove only quarantine:
# `xattr -c` also strips detached code-signature attributes from signed scripts.
find /Applications/Juno.app -print0 2>/dev/null | xargs -0 xattr -d com.apple.quarantine 2>/dev/null || true

verify_installed_app() {
  codesign --verify --deep --strict --verbose=2 /Applications/Juno.app >/dev/null 2>&1
}

if ! verify_installed_app; then
  if [[ "$SIGN_ID" != "-" ]]; then
    echo "install: warning — installed app failed codesign verification with identity: $SIGN_ID" >&2
    echo "install: warning — re-signing installed app ad-hoc for local launch." >&2
    SIGN_ID="-"
    ./scripts/sign_juno_macos_app.sh \
      --app /Applications/Juno.app \
      --identity "$SIGN_ID" \
      --entitlements "$ROOT/shells/macos/JunoApp/Juno.entitlements" \
      --timestamp none
  fi

  if ! verify_installed_app; then
    echo "install: error — installed app still fails codesign verification after signing." >&2
    codesign --verify --deep --strict --verbose=4 /Applications/Juno.app >&2 || true
    exit 6
  fi
fi

# Drop stale LaunchServices rows that share com.juno.shell (dist/build paths),
# then register only this install so System Settings / TCC match this binary.
APP=/Applications/Juno.app ./scripts/juno_lsregister_refresh.sh

if [[ "$CLEANSE" == 1 ]]; then
  echo "==> Cleanse TCC for com.juno.shell"
  APP=/Applications/Juno.app ./scripts/cleanse_juno_macos_privacy.sh || true
fi

# One last gate after privacy cleanup. The cleanse path should only remove
# quarantine metadata, but this is the final artifact the tester launches.
# If a local developer cert is unusable in the current keychain, fall back to
# ad-hoc signing rather than leaving /Applications/Juno.app in a broken state.
if ! verify_installed_app; then
  echo "install: warning — final installed app failed codesign verification after cleanse." >&2
  echo "install: warning — re-signing installed app ad-hoc for local launch." >&2
  SIGN_ID="-"
  ./scripts/sign_juno_macos_app.sh \
    --app /Applications/Juno.app \
    --identity "$SIGN_ID" \
    --entitlements "$ROOT/shells/macos/JunoApp/Juno.entitlements" \
    --timestamp none

  if ! verify_installed_app; then
    echo "install: error — final installed app still fails codesign verification after ad-hoc signing." >&2
    codesign --verify --deep --strict --verbose=4 /Applications/Juno.app >&2 || true
    exit 7
  fi
  APP=/Applications/Juno.app ./scripts/juno_lsregister_refresh.sh
fi

echo
echo "Installed: /Applications/Juno.app"
echo "Launch:    open /Applications/Juno.app"
echo
echo "Verify the local voice engine starts silently (product path):"
echo "  sed -n '1,120p' \"$HOME/Library/Application Support/com.juno.shell/runtime/health.json\""
echo "  tail -f ~/Library/Logs/Juno/bundled-engine.log"
echo "Note: /Applications/Juno.app uses the local UDS broker by default;"
echo "      127.0.0.1:8765 is enabled only for dev HTTP with JUNO_DEV_WORKBENCH_HTTP=1."
