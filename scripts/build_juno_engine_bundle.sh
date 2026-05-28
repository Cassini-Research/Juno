#!/usr/bin/env bash
# Build a relocatable Python engine tree for Juno.app/Contents/Resources/engine.
#
# After this script:
#   1. Run ./scripts/package_juno_macos_app.sh — the Makefile copies run_engine.sh
#      into the .app; merge or rsync this output's .venv + site-packages into
#      dist/Juno.app/Contents/Resources/engine/ on the packaging host.
#
# Usage:
#   ./scripts/build_juno_engine_bundle.sh [OUTPUT_DIR]
#
# Default OUTPUT_DIR: <repo>/dist/juno_engine_bundle
#
# Notes:
#   - Installs the repo non-editable into an isolated venv so the bundle can move
#     without the source checkout.
#   - Model weights are NOT included; they remain under ~/Library/Application Support/Juno
#     or HF cache per existing broker setup flows.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/juno_engine_bundle}"

rm -rf "$OUT"
mkdir -p "$OUT"

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "error: pyproject.toml not found at $ROOT" >&2
  exit 2
fi

python3 -m venv --copies "$OUT/.venv"
# shellcheck source=/dev/null
source "$OUT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install "$ROOT"

cp "$ROOT/shells/macos/bundle_resources/engine/run_engine.sh" "$OUT/run_engine.sh"
chmod +x "$OUT/run_engine.sh"

# Ship the seed personalization data alongside the venv. Without this,
# JunoSeedPersonalizationRuntime.try_load logs "bundle dir missing" and
# silently disables the seed layer — which is how Juno's vocab + style
# memory gets pre-populated on first launch. run_engine.sh exports
# JUNO_SEED_BUNDLE_DIR so juno_v2.personalization.seed.paths uses this
# copy directly instead of guessing at the site-packages layout.
SEED_SRC="$ROOT/seed_data"
if [[ -d "$SEED_SRC" ]]; then
  rsync -a --delete "$SEED_SRC/" "$OUT/seed_data/"
  echo "Bundled seed data: $OUT/seed_data"
else
  echo "warning: $SEED_SRC not found — seed layer will be disabled at runtime" >&2
fi

echo "Engine bundle created at: $OUT"
echo "Merge into Juno.app: rsync -a \"$OUT/\" \"dist/Juno.app/Contents/Resources/engine/\""
