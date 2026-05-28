#!/usr/bin/env bash
# Sign Juno.app, helper tools, and bundled Sparkle framework in dependency order.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sign_juno_macos_app.sh --app PATH --identity IDENTITY --entitlements PATH [--timestamp none|secure]

Signs nested Sparkle code first, then helper tools, then the app bundle.
Use --identity - for local ad-hoc signing.
EOF
  exit 1
}

APP=""
IDENTITY=""
ENTS=""
TIMESTAMP_MODE="none"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="$2"; shift 2 ;;
    --identity) IDENTITY="$2"; shift 2 ;;
    --entitlements) ENTS="$2"; shift 2 ;;
    --timestamp) TIMESTAMP_MODE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown: $1" >&2; usage ;;
  esac
done

[[ -n "$APP" && -n "$IDENTITY" && -n "$ENTS" ]] || usage
[[ -d "$APP" ]] || { echo "App bundle not found: $APP" >&2; exit 2; }
[[ -f "$ENTS" ]] || { echo "Entitlements not found: $ENTS" >&2; exit 2; }

remove_python_bytecode_caches() {
  local root="$1"
  [[ -d "$root" ]] || return 0
  find "$root" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "$root" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
}

timestamp_args=(--timestamp=none)
if [[ "$TIMESTAMP_MODE" == "secure" ]]; then
  timestamp_args=(--timestamp)
elif [[ "$TIMESTAMP_MODE" != "none" ]]; then
  echo "Unsupported timestamp mode: $TIMESTAMP_MODE" >&2
  exit 2
fi

sign_plain() {
  local target="$1"
  codesign --force --options runtime "${timestamp_args[@]}" --sign "$IDENTITY" "$target"
}

sign_entitled() {
  local identifier="$1"
  local target="$2"
  codesign --force --options runtime "${timestamp_args[@]}" \
    --entitlements "$ENTS" \
    --identifier "$identifier" \
    --sign "$IDENTITY" \
    "$target"
}

sign_engine_code() {
  local engine_root="$APP/Contents/Resources/engine"
  [[ -d "$engine_root" ]] || return 0
  remove_python_bytecode_caches "$engine_root"

  while IFS= read -r target; do
    if file "$target" 2>/dev/null | grep -q 'Mach-O'; then
      if [[ -x "$target" && "$target" != *.so && "$target" != *.dylib ]]; then
        # The bundled venv's Python executable may dynamically load a
        # Homebrew Python.framework outside Juno.app. Under hardened runtime,
        # that requires the same library-validation exemption as the shell.
        codesign --force --options runtime "${timestamp_args[@]}" \
          --entitlements "$ENTS" \
          --sign "$IDENTITY" \
          "$target"
      else
        codesign --force --options runtime "${timestamp_args[@]}" --sign "$IDENTITY" "$target"
      fi
    fi
  done < <(
    find "$engine_root" -type f \( -name "*.so" -o -name "*.dylib" -o -perm -111 \) -print | sort
  )
}

SPARKLE_FW="$APP/Contents/Frameworks/Sparkle.framework"
if [[ -d "$SPARKLE_FW" ]]; then
  while IFS= read -r nested; do
    sign_plain "$nested"
  done < <(find "$SPARKLE_FW" \( -name "*.xpc" -o -name "*.app" \) -type d -prune | sort)

  while IFS= read -r executable; do
    sign_plain "$executable"
  done < <(find "$SPARKLE_FW" \( -name "*.xpc" -o -name "*.app" \) -type d -prune -o -type f -perm -111 \
    ! -path "*/Headers/*" \
    ! -path "*/Modules/*" \
    ! -path "*/Resources/*" \
    -print \
    | sort)

  sign_plain "$SPARKLE_FW"
fi

sign_engine_code

for helper in juno-capability juno-host juno-hotkey juno-paste juno-textmon; do
  helper_path="$APP/Contents/MacOS/$helper"
  if [[ -f "$helper_path" ]]; then
    sign_entitled "com.juno.shell.helper.${helper#juno-}" "$helper_path"
  fi
done

sign_entitled "com.juno.shell" "$APP"
