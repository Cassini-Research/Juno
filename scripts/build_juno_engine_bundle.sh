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
# Why a standalone interpreter instead of `python -m venv`:
#   A `python -m venv` (even with --copies) is NOT redistributable. It copies
#   only the interpreter *binary*; the standard library stays in the base
#   interpreter and is located at runtime through pyvenv.cfg's `home =` line,
#   which points back at the BUILD machine's Python (e.g.
#   /Users/<builder>/miniconda3/bin). On any other Mac that path is absent, so
#   the bundled interpreter boots, fails to find `encodings`, and dies with
#   "Could not find platform independent libraries" — run_engine.sh then reports
#   "juno_v2 is not importable" and the engine never starts. This is exactly the
#   class of failure that shipped in early DMGs.
#
#   We instead ship a python-build-standalone interpreter (astral-sh). Its
#   `install_only` distribution is a self-contained prefix: it carries its own
#   libpython + the COMPLETE standard library and resolves them via
#   @executable_path rpaths, so it runs from anywhere with no dependency on the
#   build host. Because it is a real prefix (not a venv) there is no pyvenv.cfg
#   umbilical at all.
#
# Notes:
#   - Installs the repo non-editable into the standalone prefix so the bundle
#     can move without the source checkout.
#   - Model weights are NOT included; they remain under ~/Library/Application Support/Juno
#     or HF cache per existing broker setup flows.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/juno_engine_bundle}"

# --- Relocatable CPython (python-build-standalone) -------------------------
# Pin both the build and its official SHA256. cp312 ABI is stable across patch
# releases, so 3.12.x matches the project's existing cp312 wheels (MLX, numpy,
# onnxruntime, …). Override via env to bump the interpreter.
PY_VERSION="${JUNO_PY_VERSION:-3.12.13}"
PBS_TAG="${JUNO_PBS_TAG:-20260610}"
PY_ARCH="${JUNO_PY_ARCH:-aarch64}"   # Juno ships Apple Silicon only (MLX / Metal)
# Official checksum for cpython-3.12.13+20260610-aarch64-apple-darwin-install_only.tar.gz
# (from the release's SHA256SUMS). Override JUNO_PY_SHA256 when bumping the pin.
PY_SHA256="${JUNO_PY_SHA256:-e18ddd4c1e8f4a1d6c4590b37f423d76aec734447edc20ed08e93983d95f2132}"
PBS_BASE="https://github.com/astral-sh/python-build-standalone/releases/download"
PY_ASSET="cpython-${PY_VERSION}+${PBS_TAG}-${PY_ARCH}-apple-darwin-install_only.tar.gz"
PY_URL="${PBS_BASE}/${PBS_TAG}/${PY_ASSET}"
CACHE_DIR="${JUNO_PY_CACHE_DIR:-$ROOT/dist/.python-build-standalone-cache}"

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "error: pyproject.toml not found at $ROOT" >&2
  exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"

# --- Download + verify the interpreter -------------------------------------
mkdir -p "$CACHE_DIR"
TARBALL="$CACHE_DIR/$PY_ASSET"
if [[ ! -f "$TARBALL" ]]; then
  echo "Downloading $PY_ASSET ..."
  curl -fL --retry 3 --proto '=https' --tlsv1.2 -o "$TARBALL.partial" "$PY_URL"
  mv "$TARBALL.partial" "$TARBALL"
fi
# Refuse to bundle an unverified interpreter — a tampered or truncated download
# must never reach codesign/notarization.
if ! echo "${PY_SHA256}  ${TARBALL}" | shasum -a 256 -c - >/dev/null 2>&1; then
  echo "error: checksum mismatch for $TARBALL" >&2
  echo "  expected: ${PY_SHA256}" >&2
  echo "  got:      $(shasum -a 256 "$TARBALL" | awk '{print $1}')" >&2
  echo "  deleting the cached file; re-run to re-download." >&2
  rm -f "$TARBALL"
  exit 4
fi
echo "Verified $PY_ASSET (sha256 ok)"

# --- Lay it down as engine/.venv -------------------------------------------
# The install_only tarball unpacks to a top-level python/ prefix (bin/, lib/,
# include/, share/). We place its contents at $OUT/.venv so the existing
# `.venv/bin/python` contract — shared by run_engine.sh,
# JunoEngineContract.swift, JunoSetupModel.swift, and
# package_juno_macos_app.sh — keeps working without touching those files.
TMP_EXTRACT="$(mktemp -d)"
trap 'rm -rf "$TMP_EXTRACT"' EXIT
tar -xzf "$TARBALL" -C "$TMP_EXTRACT"
if [[ ! -x "$TMP_EXTRACT/python/bin/python3" ]]; then
  echo "error: unexpected python-build-standalone layout (no python/bin/python3)" >&2
  exit 5
fi
mkdir -p "$OUT/.venv"
rsync -a "$TMP_EXTRACT/python/" "$OUT/.venv/"

VENV_PY="$OUT/.venv/bin/python3"
[[ -x "$VENV_PY" ]] || { echo "error: standalone python missing at $VENV_PY" >&2; exit 5; }
# python-build-standalone ships bin/python3 and bin/python3.x but not always
# bin/python. The shell/Swell contract probes `.venv/bin/python`; add a
# relative symlink so the chain stays inside the relocatable bundle.
if [[ ! -e "$OUT/.venv/bin/python" ]]; then
  ln -s python3 "$OUT/.venv/bin/python"
fi

# --- Install the repo into the standalone prefix ---------------------------
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install "$ROOT"

# --- Pin MLX to a macOS-floor wheel so the bundle runs on older macOS -------
# MLX publishes a SEPARATE wheel per macOS version. `mlx-metal` carries the
# compiled Metal backend + mlx.metallib, built for that macOS's Metal language
# version. pip on this build host picks the wheel matching the BUILD machine's
# macOS — and our build host is macOS 26 (Tahoe), so it grabs the macosx_26
# wheels, whose metallib uses Metal language 4.0. That metallib FAILS to load
# on macOS 15 (Sequoia):
#   "Failed to load the default metallib ... language version 4.0 which is not
#    supported on this OS"
# → the engine's MLX/Metal preflight aborts (recoverable=false), the engine
# never starts, and the app never reaches model download. Force the macosx_15
# wheels for both mlx and mlx-metal — they are forward-compatible (run on
# macOS 15 through 26), so one bundle works across all supported macOS.
# Override the floor with JUNO_MLX_WHEEL_PLATFORM if the support matrix changes.
MLX_WHEEL_PLATFORM="${JUNO_MLX_WHEEL_PLATFORM:-macosx_14_0_arm64}"
MLX_VER="$("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("mlx"))')"
MLX_METAL_VER="$("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("mlx-metal"))')"
SITE_PACKAGES="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "Pinning MLX to ${MLX_WHEEL_PLATFORM} wheels (mlx==${MLX_VER}, mlx-metal==${MLX_METAL_VER})"
"$VENV_PY" -m pip install --no-deps --upgrade --force-reinstall \
  --platform "$MLX_WHEEL_PLATFORM" --only-binary=:all: \
  --target "$SITE_PACKAGES" \
  "mlx==${MLX_VER}" "mlx-metal==${MLX_METAL_VER}"
# Fail the build if the swapped-in metallib's wheel didn't actually target an
# older macOS floor (guards against a silent fall-back to the host wheel).
INSTALLED_TAG="$(cat "$SITE_PACKAGES"/mlx_metal-*.dist-info/WHEEL 2>/dev/null | awk -F': ' '/^Tag:/{print $2}')"
echo "mlx-metal wheel tag after pin: ${INSTALLED_TAG:-<none>}"
case "$INSTALLED_TAG" in
  *macosx_26_*|"") echo "error: mlx-metal still targets macOS 26 (or missing) after pin — would break macOS 15 users" >&2; exit 6 ;;
esac
# Sanity: MLX must still import + run on THIS host (macOS-15 wheels are forward
# compatible to 26), confirming the swap produced a working runtime.
"$VENV_PY" -c 'import mlx.core as mx; print("mlx ok:", mx.array([1,2,3]).sum().item())' \
  || { echo "error: bundled MLX fails to import/run after macOS-floor pin" >&2; exit 6; }

# --- Strip build-machine path leaks ----------------------------------------
# pip writes console-script wrappers (hf, huggingface-cli, mlx_lm.*,
# mlx_whisper, normalizer, …) whose shebang hardcodes this build's
# .venv/bin/python3 absolute path. run_engine.sh never invokes them — it only
# ever runs `${PY} -m <module>` — so rather than ship 50+ build-host paths
# inside a signed bundle, drop them. The underlying modules remain importable
# via `python -m`. The interpreters themselves are kept.
shopt -s nullglob
for f in "$OUT/.venv/bin"/*; do
  [[ -f "$f" ]] || continue   # skip the python/python3.x symlinks
  case "$(basename "$f")" in
    python|python3|python3.*) continue ;;
  esac
  first="$(head -1 "$f" 2>/dev/null || true)"
  if [[ "$first" == "#!"*"/.venv/bin/python"* ]]; then
    rm -f "$f"
  fi
done
shopt -u nullglob

# PEP 610 records the build-time source path in dist-info/direct_url.json
# (file:///Users/<builder>/…/Juno). It is informational metadata the runtime
# never reads; remove it so the shipped bundle carries zero build-host paths.
find "$OUT/.venv" -name "direct_url.json" -path "*.dist-info/*" -delete

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
