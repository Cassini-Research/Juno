#!/usr/bin/env bash
# Generate or print the Sparkle EdDSA public key for Juno OTA releases.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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

GENERATE_KEYS="$(find_sparkle_tool generate_keys)"
exec "$GENERATE_KEYS" "$@"
