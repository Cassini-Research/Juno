#!/usr/bin/env bash
# Build and smoke-test the local Juno.app bundle.
#
# This is intentionally a packaging/runtime sanity check, not a full manual
# dictation test. Real mic capture, Accessibility paste, and notarized launch
# still need a signed app in the user's GUI session.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${APP:-$ROOT/dist/Juno.app}"
PORT="${JUNO_SMOKE_PORT:-8766}"

cd "$ROOT"

./scripts/package_juno_macos_app.sh --sign "${CODESIGN_IDENTITY:--}"

echo "== bundle files =="
test -x "$APP/Contents/MacOS/Juno"
for helper in juno-paste juno-hotkey juno-textmon juno-capability juno-host; do
  test -x "$APP/Contents/MacOS/$helper"
  file "$APP/Contents/MacOS/$helper"
done
test -f "$APP/Contents/Resources/AppIcon.icns"
file "$APP/Contents/Resources/AppIcon.icns"

echo "== plist =="
plutil -lint "$APP/Contents/Info.plist"

echo "== codesign =="
codesign --verify --deep --strict --verbose=2 "$APP"

echo "== helper probes =="
"$APP/Contents/MacOS/juno-host"
"$APP/Contents/MacOS/juno-capability" || true

echo "== current-tree voice engine smoke =="
ENGINE_LOG="/tmp/juno-smoke-workbench-${PORT}.log"
ENGINE_DIR="/tmp/juno-smoke-workbench-${PORT}"
rm -rf "$ENGINE_DIR"
JUNO_REQUIRE_LOCAL_BROKER_AUTH=1 \
  "$ROOT/.venv/bin/python" -m juno_v2.workbench.server \
  --host 127.0.0.1 --port "$PORT" --log-dir "$ENGINE_DIR" \
  >"$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!
cleanup() {
  kill "$ENGINE_PID" >/dev/null 2>&1 || true
  wait "$ENGINE_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 20); do
  if curl -sf --max-time 1 "http://127.0.0.1:${PORT}/healthz" >/dev/null; then
    break
  fi
  sleep 0.5
done

curl -sf --max-time 2 "http://127.0.0.1:${PORT}/healthz"
echo
curl -sf --max-time 2 "http://127.0.0.1:${PORT}/api/broker/engine/compatibility"
echo
curl -sf --max-time 2 "http://127.0.0.1:${PORT}/api/broker/privacy/context_settings"
echo
status="$(
  curl -sS --max-time 2 -o /tmp/juno-smoke-auth.json -w '%{http_code}' \
    -X POST "http://127.0.0.1:${PORT}/api/broker/privacy/context_settings" \
    -H 'Content-Type: application/json' \
    -d '{"use_clipboard":false}'
)"
test "$status" = "401"
cat /tmp/juno-smoke-auth.json
echo

echo "smoke ok: $APP"
