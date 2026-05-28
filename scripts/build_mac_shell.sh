#!/usr/bin/env bash
# Quick sanity: build the macOS shell + helper tools (SwiftPM release).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$ROOT/shells/macos" && swift build -c release)
echo "Built:"
ls -1 "$ROOT/shells/macos/.build/release/" | grep -E '^(Juno|juno-(paste|hotkey|textmon|capability|host))$' || true
echo ""
echo "Run (from repo root, in Terminal.app):"
echo "  PATH=\"\$PWD/shells/macos/.build/release:\$PATH\" ./shells/macos/.build/release/Juno"
