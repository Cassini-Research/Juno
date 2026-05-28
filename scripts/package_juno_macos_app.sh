#!/usr/bin/env bash
# Build Juno.app (macOS) from SwiftPM release artifacts + bundle Info.plist.
# Produces dist/Juno.app at repo root by default. Codesign/notarize are optional flags.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ROOT="$ROOT/shells/macos/JunoApp"
DIST="$ROOT/dist/Juno.app"

SIGN_ID="${CODESIGN_IDENTITY:-}"
STAPLE="${STAPLE:-0}"
SWIFT_BUILD_FLAGS="${SWIFT_BUILD_FLAGS:---disable-sandbox}"
ENGINE_BUNDLE="${JUNO_ENGINE_BUNDLE:-}"
SWIFT_BUILD_ARTIFACT_DIR="${JUNO_SWIFT_BUILD_DIR:-}"
APP_VERSION="${JUNO_APP_VERSION:-}"
BUILD_NUMBER="${JUNO_BUILD_NUMBER:-}"
OTA_FEED_URL="${JUNO_OTA_FEED_URL:-}"
OTA_PUBLIC_ED_KEY="${JUNO_OTA_PUBLIC_ED_KEY:-}"
OTA_CHANNEL="${JUNO_OTA_CHANNEL:-}"
OTA_DISABLED="${JUNO_OTA_DISABLED:-}"
OTA_ALLOW_INSECURE_FEED="${JUNO_OTA_ALLOW_INSECURE_FEED:-}"
OTA_AUTOMATIC_CHECKS="${JUNO_OTA_AUTOMATIC_CHECKS:-}"
OTA_AUTOMATIC_DOWNLOADS="${JUNO_OTA_AUTOMATIC_DOWNLOADS:-}"
OTA_SCHEDULED_INTERVAL="${JUNO_OTA_SCHEDULED_INTERVAL:-}"

# Auto-detect the standard engine bundle location when --engine is not set.
# Without an engine bundle the .app launches into a permanent
# "Voice engine offline / no_engine_root" state, because
# JunoEngineContract.bundledEngineRoot() requires the venv + juno_v2
# package and JunoRepoPaths.guessRepoRoot() can't find pyproject.toml when
# the app runs from /Applications. Fail loudly here so packaging never
# silently produces a non-functional .app.
if [[ -z "$ENGINE_BUNDLE" && -d "$ROOT/dist/juno_engine_bundle/.venv" && -x "$ROOT/dist/juno_engine_bundle/run_engine.sh" ]]; then
  ENGINE_BUNDLE="$ROOT/dist/juno_engine_bundle"
  echo "Auto-detected engine bundle: $ENGINE_BUNDLE"
fi
ALLOW_NO_ENGINE="${JUNO_PACKAGE_ALLOW_NO_ENGINE:-0}"

usage() {
  echo "Usage: $0 [--dist PATH] [--sign IDENTITY] [--staple]"
  echo "  --dist PATH   Output Juno.app path (default: \$ROOT/dist/Juno.app)"
  echo "  --sign ID     codesign -s IDENTITY Juno.app (adhoc if ID is -)"
  echo "  --engine PATH Copy a prepared engine bundle into Contents/Resources/engine"
  echo "  --app-version VERSION       Override CFBundleShortVersionString"
  echo "  --build-number BUILD        Override CFBundleVersion"
  echo "  --ota-feed-url URL          Sparkle appcast URL"
  echo "  --ota-public-ed-key KEY     Sparkle public EdDSA key"
  echo "  --ota-channel CHANNEL       Optional Sparkle channel (for example: beta)"
  echo "  --disable-ota               Remove Sparkle feed/key settings from Info.plist"
  echo "  --allow-insecure-ota-feed   Permit http/file appcast URLs for local testing"
  echo "  --staple      Run xcrun stapler staple after sign (notarization ticket)"
  exit 1
}

remove_python_bytecode_caches() {
  local root="$1"
  [[ -d "$root" ]] || return 0
  find "$root" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "$root" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dist) DIST="$2"; shift 2 ;;
    --sign) SIGN_ID="$2"; shift 2 ;;
    --engine) ENGINE_BUNDLE="$2"; shift 2 ;;
    --app-version) APP_VERSION="$2"; shift 2 ;;
    --build-number) BUILD_NUMBER="$2"; shift 2 ;;
    --ota-feed-url) OTA_FEED_URL="$2"; shift 2 ;;
    --ota-public-ed-key) OTA_PUBLIC_ED_KEY="$2"; shift 2 ;;
    --ota-channel) OTA_CHANNEL="$2"; shift 2 ;;
    --disable-ota) OTA_DISABLED=1; shift ;;
    --allow-insecure-ota-feed) OTA_ALLOW_INSECURE_FEED=1; shift ;;
    --staple) STAPLE=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown: $1" >&2; usage ;;
  esac
done

mkdir -p "$(dirname "$DIST")"
MAKE_ARGS=(
  app
  "DIST_APP=$DIST"
  "SWIFT_BUILD_FLAGS=$SWIFT_BUILD_FLAGS"
)
if [[ -n "$SWIFT_BUILD_ARTIFACT_DIR" ]]; then
  MAKE_ARGS+=("BUILD=$SWIFT_BUILD_ARTIFACT_DIR")
fi
env \
  JUNO_APP_VERSION="$APP_VERSION" \
  JUNO_BUILD_NUMBER="$BUILD_NUMBER" \
  JUNO_OTA_FEED_URL="$OTA_FEED_URL" \
  JUNO_OTA_PUBLIC_ED_KEY="$OTA_PUBLIC_ED_KEY" \
  JUNO_OTA_CHANNEL="$OTA_CHANNEL" \
  JUNO_OTA_DISABLED="$OTA_DISABLED" \
  JUNO_OTA_ALLOW_INSECURE_FEED="$OTA_ALLOW_INSECURE_FEED" \
  JUNO_OTA_AUTOMATIC_CHECKS="$OTA_AUTOMATIC_CHECKS" \
  JUNO_OTA_AUTOMATIC_DOWNLOADS="$OTA_AUTOMATIC_DOWNLOADS" \
  JUNO_OTA_SCHEDULED_INTERVAL="$OTA_SCHEDULED_INTERVAL" \
  make -f "$APP_ROOT/Makefile" "${MAKE_ARGS[@]}"

if [[ -z "$ENGINE_BUNDLE" && "$ALLOW_NO_ENGINE" != "1" ]]; then
  cat >&2 <<EOF
error: no engine bundle and none auto-detected at \$ROOT/dist/juno_engine_bundle.

Without a bundled engine the .app launches into a permanent
"Voice engine offline / no_engine_root" state.

Build the engine bundle first:
  ./scripts/build_juno_engine_bundle.sh

Or, if you really want to ship a UI-only build (testing chrome only),
re-run with: JUNO_PACKAGE_ALLOW_NO_ENGINE=1 $0 ...
EOF
  exit 2
fi

if [[ -n "$ENGINE_BUNDLE" ]]; then
  if [[ ! -d "$ENGINE_BUNDLE" ]]; then
    echo "Engine bundle not found: $ENGINE_BUNDLE" >&2
    exit 2
  fi
  if [[ ! -x "$ENGINE_BUNDLE/run_engine.sh" ]]; then
    echo "Engine bundle missing executable run_engine.sh: $ENGINE_BUNDLE" >&2
    exit 2
  fi
  if [[ ! -x "$ENGINE_BUNDLE/.venv/bin/python" ]]; then
    echo "Engine bundle missing .venv/bin/python — run scripts/build_juno_engine_bundle.sh first." >&2
    exit 2
  fi
  echo "Copying engine bundle: $ENGINE_BUNDLE"
  rm -rf "$DIST/Contents/Resources/engine"
  mkdir -p "$DIST/Contents/Resources/engine"
  # The system rsync on macOS can fail mid-copy on large virtualenv trees with
  # transient mkstempat errors. `ditto` is the native bundle copier used by the
  # installer and preserves the directory shape without nesting ENGINE_BUNDLE.
  ditto "$ENGINE_BUNDLE" "$DIST/Contents/Resources/engine"
  remove_python_bytecode_caches "$DIST/Contents/Resources/engine"
  chmod +x "$DIST/Contents/Resources/engine/run_engine.sh"
fi

ENTS="$ROOT/shells/macos/JunoApp/Juno.entitlements"
if [[ ! -f "$ENTS" ]]; then
  echo "Missing entitlements file: $ENTS" >&2
  exit 2
fi

# The Makefile signs the app before optional engine resources are copied.
# If --engine is used, the resource seal must be refreshed after that copy
# or macOS launch and security checks will see a stale bundle signature.
if [[ -n "$ENGINE_BUNDLE" ]]; then
  POST_ENGINE_SIGN_ID="${SIGN_ID:--}"
  "$ROOT/scripts/sign_juno_macos_app.sh" \
    --app "$DIST" \
    --identity "$POST_ENGINE_SIGN_ID" \
    --entitlements "$ENTS" \
    --timestamp none
fi

if [[ -n "$SIGN_ID" ]]; then
  echo "Codesigning with: $SIGN_ID"
  TIMESTAMP_MODE="secure"
  if [[ "$SIGN_ID" == "-" ]]; then
    TIMESTAMP_MODE="none"
  fi
  "$ROOT/scripts/sign_juno_macos_app.sh" \
    --app "$DIST" \
    --identity "$SIGN_ID" \
    --entitlements "$ENTS" \
    --timestamp "$TIMESTAMP_MODE"
fi

if [[ "$STAPLE" == 1 ]]; then
  xcrun stapler staple "$DIST"
fi

echo "Juno.app → $DIST"
echo "Open with: open \"$DIST\""
