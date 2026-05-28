#!/usr/bin/env bash
# Full local reset + rebuild + install of Juno for end-to-end testing as a new user:
# quit processes, evict stale LaunchServices rows for com.juno.shell, reset TCC
# (Juno + helpers + optional legacy com.juno.shell), optionally wipe prefs/support/logs,
# remove old app artifacts, then ./scripts/install_juno_macos.sh --install-to-apps.
#
# Default behavior of a fresh deploy:
#   - All TCC permissions for Juno are cleared (Microphone, Accessibility,
#     SpeechRecognition, Apple Events / Notes Automation, Reminders,
#     Calendar) and tccd is restarted so the next launch re-prompts.
#   - Onboarding state, preferences, caches and logs are wiped so you see
#     a true new-user flow.
#   - Product history (product_history.sqlite + audio/) is *preserved*
#     across the reset by default — pass --wipe-history if you want a
#     genuinely empty History page.
#
# After this script, open System Settings → Privacy & Security → Accessibility,
# remove any duplicate “Juno” rows with minus, add only /Applications/Juno.app if needed.
# Always start Juno with:  open /Applications/Juno.app
# Do not open dist/Juno.app or a bare helper binary — LaunchServices will register a second
# com.juno.shell path and Privacy lists may show a generic “exec” icon for the wrong row.
#
# Usage:
#   ./scripts/fresh_juno_macos_environment.sh
#   ./scripts/fresh_juno_macos_environment.sh --keep-user-data   # keep prefs + Application Support + Logs
#   ./scripts/fresh_juno_macos_environment.sh --keep-dist        # keep dist/ engine bundle + Juno.app build cache
#   ./scripts/fresh_juno_macos_environment.sh --skip-build       # reuse dist/juno_engine_bundle
#   ./scripts/fresh_juno_macos_environment.sh --sign IDENTITY    # override default ad-hoc local signing
#   ./scripts/fresh_juno_macos_environment.sh --no-cleanse       # skip all TCC resets/cleanses
#   ./scripts/fresh_juno_macos_environment.sh --wipe-history     # also wipe product history + retained audio
#   ./scripts/fresh_juno_macos_environment.sh --open             # open /Applications/Juno.app when done
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

KEEP_USER_DATA=0
KEEP_DIST=0
SKIP_BUILD=0
CLEANSE=1
OPEN_AFTER=0
WIPE_HISTORY=0
# Fresh-new-user resets intentionally wipe TCC, so stable local signing does
# not preserve useful permissions here. Default to ad-hoc for deterministic
# local verification; pass --sign to exercise a real distribution identity.
SIGN_ID="${JUNO_FRESH_SIGN_ID:--}"

usage() {
  echo "Usage: $0 [--keep-user-data] [--keep-dist] [--skip-build] [--sign IDENTITY] [--no-cleanse] [--wipe-history] [--open]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-user-data) KEEP_USER_DATA=1; shift ;;
    --keep-dist) KEEP_DIST=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --sign) SIGN_ID="$2"; shift 2 ;;
    --no-cleanse) CLEANSE=0; shift ;;
    --wipe-history) WIPE_HISTORY=1; shift ;;
    --open) OPEN_AFTER=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: macOS only" >&2
  exit 2
fi

GUI_DOMAIN="gui/$(id -u)"
LABEL="${JUNO_LAUNCHD_LABEL:-com.juno.voice-engine}"
BUNDLE_ID="${JUNO_BUNDLE_ID:-com.juno.shell}"
LEGACY_BUNDLE_ID="${JUNO_LEGACY_BUNDLE_ID:-}"

PREF_DOMAINS=("$BUNDLE_ID")
if [[ -n "$LEGACY_BUNDLE_ID" && "$LEGACY_BUNDLE_ID" != "$BUNDLE_ID" ]]; then
  PREF_DOMAINS+=("$LEGACY_BUNDLE_ID")
fi

echo "==> 1/6 Stop Juno, helpers, voice-engine subprocess, and LaunchAgent (if loaded)"
# Graceful quit attempt (capped at 2s — if Juno is hung/unclickable, AppleEvent
# round-trip can wedge for the full default timeout). Then force-kill below.
( osascript -e 'tell application "Juno" to quit' >/dev/null 2>&1 || true ) &
_quit_pid=$!
( sleep 2 && kill -9 $_quit_pid 2>/dev/null || true ) &
wait $_quit_pid 2>/dev/null || true

pkill -x Juno >/dev/null 2>&1 || true
for _helper in juno-capability juno-host juno-hotkey juno-paste juno-textmon; do
  pkill -x "$_helper" >/dev/null 2>&1 || true
done

# Kill the Python voice-engine subprocess. When Juno is force-killed via
# SIGKILL, the engine subprocess (started by run_engine.sh) is orphaned to
# launchd rather than terminated. A surviving engine holds the UDS socket
# at runtime/engine.sock and the next Juno launch's lifecycle probe sees a
# "healthy engine already up" — but it's a stale engine from the previous
# build, which can lead to "looks healthy, doesn't actually work" launch
# states. Kill all Python processes running juno_v2.runtime.service.
pkill -f "juno_v2.runtime.service" >/dev/null 2>&1 || true
# Also kill anything still bound to the Juno engine socket directory.
pkill -f "/Applications/Juno.app/Contents/Resources/engine/run_engine.sh" >/dev/null 2>&1 || true
pkill -f "Juno.app/Contents/Resources/engine" >/dev/null 2>&1 || true

launchctl bootout "$GUI_DOMAIN/$LABEL" >/dev/null 2>&1 || true
sleep 1
# Second pass: anything that respawned via launchd or was slow to die.
pkill -9 -x Juno >/dev/null 2>&1 || true
pkill -9 -f "juno_v2.runtime.service" >/dev/null 2>&1 || true

echo "==> 2/6 LaunchServices: unregister every stale com.juno.shell path"
"$ROOT/scripts/juno_lsregister_refresh.sh" --evict-only

if [[ "$CLEANSE" == 1 ]]; then
  echo "==> 3/6 Reset TCC for Juno + helpers + legacy com.juno.shell (best-effort)"
  "$ROOT/scripts/reset_juno_tcc.sh" --run --juno --all --strict
else
  echo "==> 3/6 Keep TCC permissions (--no-cleanse)"
fi

# History preservation. The wipe in step 4 nukes the entire
# Application Support directory for com.juno.shell. The current installed app
# stores product history under logs/service/workbench; older local builds used
# Workbench directly. Preserve both roots by default so a "fresh" rebuild does
# not silently erase the user's History page or replay audio. ``--wipe-history``
# opts out; ``--keep-user-data`` is already a superset.
HISTORY_STASH=""
SUPPORT_ROOT="$HOME/Library/Application Support/${BUNDLE_ID}"
CURRENT_WORKBENCH_DIR="$SUPPORT_ROOT/logs/service/workbench"
LEGACY_WORKBENCH_DIR="$SUPPORT_ROOT/Workbench"
HISTORY_PRIMARY_DIR="$CURRENT_WORKBENCH_DIR"

stash_workbench_history() {
  local label="$1"
  local src_dir="$2"
  local dest_dir="$3"
  local found=0
  mkdir -p "$dest_dir"
  for _base in product_history.sqlite actions_index.sqlite; do
    for _sfx in "" "-wal" "-shm"; do
      if [[ -f "$src_dir/${_base}${_sfx}" ]]; then
        cp -p "$src_dir/${_base}${_sfx}" "$dest_dir/" 2>/dev/null || true
        found=1
      fi
    done
  done
  for _f in history.jsonl product_history.sqlite.migrated_jsonl; do
    if [[ -f "$src_dir/$_f" ]]; then
      cp -p "$src_dir/$_f" "$dest_dir/" 2>/dev/null || true
      found=1
    fi
  done
  if [[ -d "$src_dir/audio" ]]; then
    cp -R "$src_dir/audio" "$dest_dir/audio" 2>/dev/null || true
    found=1
  fi
  if [[ "$found" == 1 ]]; then
    echo "==>    Stashed $label history → $dest_dir"
  else
    rm -rf "$dest_dir" 2>/dev/null || true
  fi
}

restore_workbench_history() {
  local label="$1"
  local dest_dir="$2"
  local src_dir="$3"
  [[ -d "$src_dir" ]] || return 0
  echo "==>    Restoring $label history to $dest_dir"
  mkdir -p "$dest_dir"
  for _base in product_history.sqlite actions_index.sqlite; do
    for _sfx in "" "-wal" "-shm"; do
      if [[ -f "$src_dir/${_base}${_sfx}" ]]; then
        cp -p "$src_dir/${_base}${_sfx}" "$dest_dir/" 2>/dev/null || true
      fi
    done
  done
  for _f in history.jsonl product_history.sqlite.migrated_jsonl; do
    if [[ -f "$src_dir/$_f" ]]; then
      cp -p "$src_dir/$_f" "$dest_dir/" 2>/dev/null || true
    fi
  done
  if [[ -d "$src_dir/audio" ]]; then
    rm -rf "$dest_dir/audio" 2>/dev/null || true
    cp -R "$src_dir/audio" "$dest_dir/audio" 2>/dev/null || true
  fi
}

if [[ "$KEEP_USER_DATA" == 0 && "$WIPE_HISTORY" == 0 ]]; then
  if [[ -d "$CURRENT_WORKBENCH_DIR" || -d "$LEGACY_WORKBENCH_DIR" ]]; then
    HISTORY_STASH="$(mktemp -d -t juno_history_stash)"
    stash_workbench_history "current" "$CURRENT_WORKBENCH_DIR" "$HISTORY_STASH/current"
    stash_workbench_history "legacy" "$LEGACY_WORKBENCH_DIR" "$HISTORY_STASH/legacy"
    if [[ ! -d "$HISTORY_STASH/current" && ! -d "$HISTORY_STASH/legacy" ]]; then
      rm -rf "$HISTORY_STASH" 2>/dev/null || true
      HISTORY_STASH=""
    fi
  fi
fi

if [[ "$KEEP_USER_DATA" == 0 ]]; then
  echo "==> 4/6 Remove preferences (fresh onboarding: no JunoOnboardingCompleted until in-app flow finishes)"
  # `defaults delete` drops the domain and helps cfprefsd; `rm` cleans any on-disk plist.
  for _domain in "${PREF_DOMAINS[@]}"; do
    defaults delete "$_domain" 2>/dev/null || true
    rm -f "$HOME/Library/Preferences/${_domain}.plist" || true
    rm -f "$HOME/Library/Preferences/ByHost/${_domain}".*.plist 2>/dev/null || true
  done
  # cfprefsd can keep a deleted domain in memory and write it back when
  # --open immediately launches Juno. Restart the per-user daemon so the
  # deleted plist is the source of truth before the app starts.
  killall cfprefsd >/dev/null 2>&1 || true
  sleep 1
  for _domain in "${PREF_DOMAINS[@]}"; do
    if defaults read "$_domain" >/dev/null 2>&1; then
      echo "error: defaults domain still exists after reset: $_domain" >&2
      defaults read "$_domain" >&2 || true
      echo "       refusing to open Juno with stale onboarding/preferences." >&2
      exit 4
    fi
  done
  echo "     (cleared ${PREF_DOMAINS[*]} — onboarding starts unset)"
else
  echo "==> 4/6 Keep user data (--keep-user-data)"
fi

if [[ "$KEEP_USER_DATA" == 0 ]]; then
  echo "==>    Application Support + Logs + Caches (Juno current + legacy roots)"
  for _domain in "${PREF_DOMAINS[@]}"; do
    rm -rf "$HOME/Library/Application Support/${_domain}" || true
    rm -rf "$HOME/Library/Logs/${_domain}" || true
    rm -rf "$HOME/Library/Caches/${_domain}" || true
  done
  rm -rf "$HOME/Library/Application Support/Juno" || true
  rm -rf "$HOME/Library/Logs/Juno" || true
  rm -rf "$HOME/Library/Caches/Juno" || true
fi

rm -f "$HOME/Library/Application Support/Juno/JunoShell.lock" 2>/dev/null || true

# Always remove the runtime socket + lock + token, even with --keep-user-data.
# These are session artifacts; a stale UDS socket file from a killed engine
# makes the next launch's lifecycle probe report "healthy" against nothing,
# wedging the splash. Pref + history are kept; only ephemeral runtime state
# is cleared.
rm -rf "$HOME/Library/Application Support/${BUNDLE_ID}/runtime" 2>/dev/null || true

echo "==> 5/6 Remove installed app and optional dist artifacts"
rm -rf /Applications/Juno.app || true
if [[ "$KEEP_DIST" == 0 ]]; then
  rm -rf "$ROOT/dist/Juno.app" "$ROOT/dist/juno_engine_bundle" || true
fi

INSTALL_ARGS=(--install-to-apps)
[[ "$SKIP_BUILD" == 1 ]] && INSTALL_ARGS+=(--skip-build)
[[ -n "$SIGN_ID" ]] && INSTALL_ARGS+=(--sign "$SIGN_ID")
[[ "$CLEANSE" == 1 ]] && INSTALL_ARGS+=(--cleanse)

echo "==> 6/6 Build + install to /Applications"
"$ROOT/scripts/install_juno_macos.sh" "${INSTALL_ARGS[@]}"

verify_installed_permission_plumbing() {
  local app="/Applications/Juno.app"
  local info="$app/Contents/Info.plist"
  local missing=0
  for _key in \
    NSMicrophoneUsageDescription \
    NSAccessibilityUsageDescription \
    NSSpeechRecognitionUsageDescription \
    NSAppleEventsUsageDescription \
    NSRemindersUsageDescription \
    NSRemindersFullAccessUsageDescription \
    NSCalendarsUsageDescription \
    NSCalendarsFullAccessUsageDescription \
    NSCalendarsWriteOnlyAccessUsageDescription
  do
    if ! /usr/libexec/PlistBuddy -c "Print :$_key" "$info" >/dev/null 2>&1; then
      echo "error: installed Juno.app is missing Info.plist permission key: $_key" >&2
      missing=1
    fi
  done
  local ents
  ents="$(mktemp -t juno_entitlements)"
  if codesign -d --entitlements :- "$app" >"$ents" 2>/dev/null; then
    for _ent in \
      com.apple.security.device.audio-input \
      com.apple.security.automation.apple-events \
      com.apple.security.personal-information.calendars
    do
      if ! grep -q "$_ent" "$ents"; then
        echo "error: installed Juno.app is missing entitlement: $_ent" >&2
        missing=1
      fi
    done
  else
    echo "error: could not read installed Juno.app entitlements" >&2
    missing=1
  fi
  rm -f "$ents" 2>/dev/null || true
  if [[ "$missing" != 0 ]]; then
    exit 8
  fi
  echo "==>    Verified permission plumbing: mic, AX, speech, Notes Automation, Reminders, Calendar"
}

verify_installed_permission_plumbing

# Restore product history if it was stashed. Done *after* install so
# the install step (which can also touch Application Support to seed
# defaults) is the one that creates the parent directory tree — we
# then drop the prior SQLite + audio back into the same location.
if [[ -n "$HISTORY_STASH" && -d "$HISTORY_STASH" ]]; then
  restore_workbench_history "current" "$CURRENT_WORKBENCH_DIR" "$HISTORY_STASH/current"
  restore_workbench_history "legacy" "$LEGACY_WORKBENCH_DIR" "$HISTORY_STASH/legacy"
  rm -rf "$HISTORY_STASH" || true
fi

if [[ "$OPEN_AFTER" == 1 ]]; then
  open /Applications/Juno.app
fi

echo
echo "Fresh environment steps complete."
if [[ "$CLEANSE" == 1 ]]; then
  echo "Permissions reset for $BUNDLE_ID: Microphone, Accessibility, SpeechRecognition,"
  echo "                                 AppleEvents (Notes Automation), Reminders, Calendar."
  echo "  → All Voice Actions + Voice Commands prompts will re-fire on next launch."
fi
if [[ -n "${HISTORY_STASH:-}" ]] || [[ "$KEEP_USER_DATA" == 1 ]] && [[ "$WIPE_HISTORY" == 0 ]]; then
  echo "Product history preserved at: $HISTORY_PRIMARY_DIR/product_history.sqlite"
elif [[ "$WIPE_HISTORY" == 1 ]]; then
  echo "Product history was wiped (--wipe-history)."
fi
echo
echo "Manual follow-up (recommended once):"
echo "  System Settings → Privacy & Security → Accessibility → remove duplicate/wrong-icon Juno rows (-),"
echo "  then add only: /Applications/Juno.app"
echo "Launch (correct Dock + Privacy icon):  open /Applications/Juno.app"
echo "Do not: open dist/Juno.app (registers a second com.juno.shell path)"
