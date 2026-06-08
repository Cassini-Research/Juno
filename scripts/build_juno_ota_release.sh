#!/usr/bin/env bash
# Build a signed Juno.app archive and regenerate the Sparkle appcast.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_APP="$ROOT/dist/Juno.app"
UPDATES_DIR="$ROOT/dist/ota"
VERSION="${JUNO_APP_VERSION:-}"
BUILD_NUMBER="${JUNO_BUILD_NUMBER:-}"
SIGN_ID="${CODESIGN_IDENTITY:-}"
ENGINE_BUNDLE="${JUNO_ENGINE_BUNDLE:-}"
OTA_FEED_URL="${JUNO_OTA_FEED_URL:-}"
OTA_PUBLIC_ED_KEY="${JUNO_OTA_PUBLIC_ED_KEY:-}"
OTA_CHANNEL="${JUNO_OTA_CHANNEL:-}"
DOWNLOAD_URL_PREFIX="${JUNO_OTA_DOWNLOAD_URL_PREFIX:-}"
ALLOW_INSECURE_FEED="${JUNO_OTA_ALLOW_INSECURE_FEED:-}"
NOTARY_PROFILE="${JUNO_NOTARY_KEYCHAIN_PROFILE:-}"
STAPLE="${STAPLE:-0}"
RELEASE_NOTES_SOURCE="${JUNO_OTA_RELEASE_NOTES:-}"

usage() {
  cat <<'EOF'
Usage: build_juno_ota_release.sh --version VERSION --build-number BUILD --ota-feed-url URL --ota-public-ed-key KEY [options]

Options:
  --dist PATH                    Output app path (default: dist/Juno.app)
  --updates-dir PATH             Directory containing Sparkle update archives (default: dist/ota)
  --download-url-prefix URL      Public URL prefix for archives in appcast.xml
  --sign IDENTITY                Developer ID Application identity
  --engine PATH                  Prepared engine bundle to copy into Juno.app
  --ota-channel CHANNEL          Optional Sparkle channel (for example: beta)
  --release-notes PATH           Optional .html, .md, or .txt release notes file
  --allow-insecure-ota-feed      Permit http/file appcast URLs for local testing
  --notary-keychain-profile NAME Submit a temporary zip with notarytool before final archive
  --staple                       Staple an existing notarization ticket before final archive

The script creates Juno-VERSION-BUILD.zip and regenerates appcast.xml in --updates-dir.
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dist) DIST_APP="$2"; shift 2 ;;
    --updates-dir) UPDATES_DIR="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --build-number) BUILD_NUMBER="$2"; shift 2 ;;
    --sign) SIGN_ID="$2"; shift 2 ;;
    --engine) ENGINE_BUNDLE="$2"; shift 2 ;;
    --ota-feed-url) OTA_FEED_URL="$2"; shift 2 ;;
    --ota-public-ed-key) OTA_PUBLIC_ED_KEY="$2"; shift 2 ;;
    --ota-channel) OTA_CHANNEL="$2"; shift 2 ;;
    --download-url-prefix) DOWNLOAD_URL_PREFIX="$2"; shift 2 ;;
    --release-notes) RELEASE_NOTES_SOURCE="$2"; shift 2 ;;
    --allow-insecure-ota-feed) ALLOW_INSECURE_FEED=1; shift ;;
    --notary-keychain-profile) NOTARY_PROFILE="$2"; shift 2 ;;
    --staple) STAPLE=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown: $1" >&2; usage ;;
  esac
done

[[ -n "$VERSION" ]] || { echo "--version is required" >&2; exit 2; }
[[ -n "$BUILD_NUMBER" ]] || { echo "--build-number is required" >&2; exit 2; }
[[ -n "$OTA_FEED_URL" ]] || { echo "--ota-feed-url is required" >&2; exit 2; }
[[ -n "$OTA_PUBLIC_ED_KEY" ]] || { echo "--ota-public-ed-key is required" >&2; exit 2; }

find_sparkle_tool() {
  local tool="$1"
  local explicit="${SPARKLE_BIN_DIR:-}"
  if [[ -n "$explicit" && -x "$explicit/$tool" ]]; then
    printf '%s\n' "$explicit/$tool"
    return 0
  fi

  local found
  found="$(find "$ROOT/shells/macos/.build" -path "*/Sparkle/bin/$tool" -type f -perm -111 2>/dev/null | head -n 1 || true)"
  if [[ -n "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi

  swift package --package-path "$ROOT/shells/macos" resolve >/dev/null
  found="$(find "$ROOT/shells/macos/.build" -path "*/Sparkle/bin/$tool" -type f -perm -111 2>/dev/null | head -n 1 || true)"
  if [[ -n "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi

  echo "Could not locate Sparkle $tool. Run swift build --package-path shells/macos first." >&2
  return 1
}

PACKAGE_ARGS=(
  --dist "$DIST_APP"
  --app-version "$VERSION"
  --build-number "$BUILD_NUMBER"
  --ota-feed-url "$OTA_FEED_URL"
  --ota-public-ed-key "$OTA_PUBLIC_ED_KEY"
)

if [[ -n "$SIGN_ID" ]]; then
  PACKAGE_ARGS+=(--sign "$SIGN_ID")
fi
if [[ -n "$ENGINE_BUNDLE" ]]; then
  PACKAGE_ARGS+=(--engine "$ENGINE_BUNDLE")
fi
if [[ -n "$OTA_CHANNEL" ]]; then
  PACKAGE_ARGS+=(--ota-channel "$OTA_CHANNEL")
fi
if [[ -n "$ALLOW_INSECURE_FEED" ]]; then
  PACKAGE_ARGS+=(--allow-insecure-ota-feed)
fi

"$ROOT/scripts/package_juno_macos_app.sh" "${PACKAGE_ARGS[@]}"

mkdir -p "$UPDATES_DIR"

if [[ -n "$NOTARY_PROFILE" ]]; then
  NOTARY_DIR="$UPDATES_DIR/.notary"
  NOTARY_ZIP="$NOTARY_DIR/Juno-$VERSION-$BUILD_NUMBER-notary.zip"
  rm -rf "$NOTARY_DIR"
  mkdir -p "$NOTARY_DIR"
  ditto -c -k --sequesterRsrc --keepParent "$DIST_APP" "$NOTARY_ZIP"
  xcrun notarytool submit "$NOTARY_ZIP" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DIST_APP"
elif [[ "$STAPLE" == 1 ]]; then
  xcrun stapler staple "$DIST_APP"
fi

ARCHIVE="$UPDATES_DIR/Juno-$VERSION-$BUILD_NUMBER.zip"
rm -f "$ARCHIVE"
ditto -c -k --sequesterRsrc --keepParent "$DIST_APP" "$ARCHIVE"

release_notes_extension() {
  local path="$1"
  local base="${path##*/}"
  local ext="${base##*.}"
  printf '%s\n' "$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"
}

NOTES_FILE="$UPDATES_DIR/Juno-$VERSION-$BUILD_NUMBER.md"
if [[ -n "$RELEASE_NOTES_SOURCE" ]]; then
  if [[ ! -f "$RELEASE_NOTES_SOURCE" ]]; then
    echo "Release notes file not found: $RELEASE_NOTES_SOURCE" >&2
    exit 2
  fi
  NOTES_EXT="$(release_notes_extension "$RELEASE_NOTES_SOURCE")"
  case "$NOTES_EXT" in
    html|md|txt) ;;
    *)
      echo "Release notes must be .html, .md, or .txt: $RELEASE_NOTES_SOURCE" >&2
      exit 2
      ;;
  esac
  NOTES_FILE="$UPDATES_DIR/Juno-$VERSION-$BUILD_NUMBER.$NOTES_EXT"
  cp "$RELEASE_NOTES_SOURCE" "$NOTES_FILE"
else
  cat >"$NOTES_FILE" <<EOF
# Juno $VERSION

This update installs Juno $VERSION (build $BUILD_NUMBER).
EOF
fi

GENERATE_APPCAST="$(find_sparkle_tool generate_appcast)"
APPCAST_ARGS=()
if "$GENERATE_APPCAST" --help 2>&1 | grep -q -- '--embed-release-notes'; then
  APPCAST_ARGS+=(--embed-release-notes)
fi
if [[ -n "$DOWNLOAD_URL_PREFIX" ]]; then
  if "$GENERATE_APPCAST" --help 2>&1 | grep -q -- '--download-url-prefix'; then
    APPCAST_ARGS+=(--download-url-prefix "$DOWNLOAD_URL_PREFIX")
  else
    echo "warning: generate_appcast does not advertise --download-url-prefix; appcast URLs may need manual hosting adjustment" >&2
  fi
fi

"$GENERATE_APPCAST" "${APPCAST_ARGS[@]}" "$UPDATES_DIR"

echo "Archive: $ARCHIVE"
echo "Release notes: $NOTES_FILE"
echo "Appcast: $UPDATES_DIR/appcast.xml"
