#!/usr/bin/env bash
# Build and activate the exact Juno source used by the phone-facing voice API.
#
# The DGX Spark is an authenticated relay, not a second Juno runtime. A deploy
# therefore has two independently verifiable parts:
#   1. build, sign and atomically activate Juno.app on this Mac;
#   2. write a non-secret receipt beside the Spark relay after end-to-end health.
#
# The current app is retained as the one rollback build. The script restores it
# automatically if the local bridge or public relay does not become healthy.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mode="audit"
source_root="$REPO_ROOT"
allow_dirty=0
dgx_host="${JUNO_VOICE_DGX_HOST:-}"
dgx_receipt_dir="${JUNO_VOICE_DGX_RECEIPT_DIR:-}"
dgx_relay_path="${JUNO_VOICE_DGX_RELAY_PATH:-qwen38-dgx-spark/juno-relay/juno_relay.py}"
public_health_url="${JUNO_VOICE_PUBLIC_HEALTH_URL:-}"
local_health_url="${JUNO_VOICE_LOCAL_HEALTH_URL:-http://127.0.0.1:8797/healthz}"
installed_app="${JUNO_VOICE_INSTALLED_APP:-/Applications/Juno.app}"
deploy_root=""
sign_identity="${JUNO_VOICE_SIGN_IDENTITY:-${CODESIGN_IDENTITY:-}}"
developer_dir="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
skip_tests=0
skip_build=0

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_juno_voice_backend.sh audit [options]
  scripts/deploy_juno_voice_backend.sh deploy [options]

Options:
  --source-root PATH     Juno checkout to inspect/build (default: this repo)
  --allow-dirty          permit a dirty source snapshot and record its digest
  --dgx-host USER@HOST   Spark SSH destination; never stored in source
  --dgx-receipt-dir PATH Spark directory for current/previous receipts
  --public-health URL    public relay health URL checked after activation
  --local-health URL     local bridge health URL (default: 127.0.0.1:8797)
  --sign IDENTITY        stable macOS signing identity
  --skip-tests           skip the focused final-text regression suite
  --skip-build           reuse the already staged deploy app

The deploy command intentionally has no embedded hostname, endpoint or key.
Use an owner-only local environment file or pass options at invocation time.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '==> %s\n' "$*"
}

safe_delete_owned_tree() {
  local target="$1"
  local allowed_parent="$2"
  [[ -e "$target" ]] || return 0
  case "$target" in
    "$allowed_parent"/*) ;;
    *) die "refusing to delete outside the owned deployment root: $target" ;;
  esac
  if /usr/sbin/lsof +D "$target" >/dev/null 2>&1; then
    die "refusing to delete a deployment tree with open files: $target"
  fi
  find "$target" -depth -delete
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    audit|deploy) mode="$1"; shift ;;
    --source-root) source_root="$2"; shift 2 ;;
    --allow-dirty) allow_dirty=1; shift ;;
    --dgx-host) dgx_host="$2"; shift 2 ;;
    --dgx-receipt-dir) dgx_receipt_dir="$2"; shift 2 ;;
    --public-health) public_health_url="$2"; shift 2 ;;
    --local-health) local_health_url="$2"; shift 2 ;;
    --sign) sign_identity="$2"; shift 2 ;;
    --skip-tests) skip_tests=1; shift ;;
    --skip-build) skip_build=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

source_root="$(cd "$source_root" && pwd)"
deploy_root="${JUNO_VOICE_DEPLOY_ROOT:-$source_root/dist/voice-backend-deploy}"
staged_app="$deploy_root/staged/Juno.app"
rollback_app="$deploy_root/rollback/Juno.app"
receipt_dir="$deploy_root/receipts"
current_receipt="$receipt_dir/current.receipt"
previous_receipt="$receipt_dir/previous.receipt"

[[ "$(uname -s)" == "Darwin" ]] || die "this deployment is macOS-only"
[[ -d "$source_root/.git" || -f "$source_root/.git" ]] || die "not a Git checkout: $source_root"
[[ -x "$source_root/scripts/build_juno_engine_bundle.sh" ]] || die "Juno build scripts are missing"
[[ -x "$developer_dir/usr/bin/xcodebuild" ]] || die "full Xcode is required at $developer_dir"

sha256_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    printf 'missing\n'
  fi
}

source_delta_sha256() {
  local root="$1"
  {
    git -C "$root" diff --binary HEAD --
    git -C "$root" status --porcelain=v1 --untracked-files=normal
    while IFS= read -r -d '' untracked_path; do
      printf '%s  %s\n' \
        "$(sha256_file "$root/$untracked_path")" \
        "$untracked_path"
    done < <(git -C "$root" ls-files -z --others --exclude-standard)
  } | shasum -a 256 | awk '{print $1}'
}

http_health() {
  local url="$1"
  [[ -n "$url" ]] || return 0
  curl --noproxy '*' --fail --silent --show-error --max-time 15 "$url" >/dev/null
}

wait_for_health() {
  local url="$1"
  local attempts="${2:-30}"
  local delay="${3:-2}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if http_health "$url"; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

source_commit="$(git -C "$source_root" rev-parse HEAD)"
source_branch="$(git -C "$source_root" symbolic-ref --quiet --short HEAD || printf 'detached')"
source_status="$(git -C "$source_root" status --porcelain=v1 --untracked-files=normal)"
source_dirty=0
if [[ -n "$source_status" ]]; then
  source_dirty=1
fi

# A dirty private deployment is fingerprinted from the tracked binary diff,
# status listing, and every untracked file's content hash. Clean releases use
# the immutable commit directly.
source_delta_digest="$(source_delta_sha256 "$source_root")"
source_identity="$source_commit"
if [[ "$source_dirty" == 1 ]]; then
  source_identity="${source_commit}-dirty-${source_delta_digest:0:12}"
fi

critical_paths=(
  juno_core_v3/dictation/pipeline.py
  juno_core_v3/dictation/self_corrections.py
  juno_v2/writer/dictation_editor.py
  juno_v2/writer/service.py
  juno_v2/workbench/server.py
  juno_v2/runtime/uds_dispatch.py
)

installed_site="$installed_app/Contents/Resources/engine/.venv/lib/python3.12/site-packages"
critical_match="yes"
critical_lines=()
for relative_path in "${critical_paths[@]}"; do
  source_sha="$(sha256_file "$source_root/$relative_path")"
  installed_sha="$(sha256_file "$installed_site/$relative_path")"
  if [[ "$source_sha" != "$installed_sha" ]]; then
    critical_match="no"
  fi
  critical_lines+=("critical_${relative_path//\//_}=$source_sha:$installed_sha")
done

local_health="fail"
if http_health "$local_health_url"; then
  local_health="ok"
fi
public_health="not_configured"
if [[ -n "$public_health_url" ]]; then
  public_health="fail"
  if http_health "$public_health_url"; then
    public_health="ok"
  fi
fi

remote_relay_sha="not_configured"
remote_service_state="not_configured"
if [[ -n "$dgx_host" ]]; then
  [[ "$dgx_relay_path" =~ ^[A-Za-z0-9._/-]+$ ]] || die "DGX relay path contains unsupported characters"
  remote_relay_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$dgx_host" \
    "path='$dgx_relay_path'; case \"\$path\" in /*) ;; *) path=\"\$HOME/\$path\" ;; esac; sha256sum \"\$path\" 2>/dev/null | awk '{print \$1}' || printf missing" \
    2>/dev/null || printf unreachable)"
  remote_service_state="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$dgx_host" \
    'systemctl --user is-active juno-relay.service 2>/dev/null || true' \
    2>/dev/null || printf unreachable)"
fi

print_audit() {
  printf 'source_root=%s\n' "$source_root"
  printf 'source_branch=%s\n' "$source_branch"
  printf 'source_identity=%s\n' "$source_identity"
  printf 'source_dirty=%s\n' "$source_dirty"
  printf 'installed_app=%s\n' "$installed_app"
  printf 'installed_matches_source=%s\n' "$critical_match"
  printf 'local_health=%s\n' "$local_health"
  printf 'public_health=%s\n' "$public_health"
  printf 'dgx_relay_sha256=%s\n' "$remote_relay_sha"
  printf 'dgx_service=%s\n' "$remote_service_state"
  printf '%s\n' "${critical_lines[@]}"
}

if [[ "$mode" == "audit" ]]; then
  print_audit
  exit 0
fi

if [[ "$source_dirty" == 1 && "$allow_dirty" != 1 ]]; then
  die "source checkout is dirty; commit the release or pass --allow-dirty for an explicitly fingerprinted private build"
fi
[[ -n "$public_health_url" ]] || die "deploy requires --public-health so the phone path is verified"
[[ -n "$dgx_host" ]] || die "deploy requires --dgx-host so the Spark receipt is updated"
[[ -n "$dgx_receipt_dir" ]] || die "deploy requires --dgx-receipt-dir"
[[ "$dgx_receipt_dir" == /* ]] || die "--dgx-receipt-dir must be an absolute path on Spark"

available_kb="$(df -Pk "$source_root" | awk 'NR == 2 {print $4}')"
minimum_kb=$((105 * 1024 * 1024))
(( available_kb >= minimum_kb )) || die "deployment would violate the 100 GB free-space floor"

if [[ "$skip_tests" != 1 ]]; then
  python_bin="$source_root/.venv/bin/python"
  [[ -x "$python_bin" ]] || die "missing $python_bin; bootstrap Juno before deployment"
  log "Run final-text and engine-contract regressions"
  "$python_bin" -m pytest -q \
    "$source_root/tests/test_self_correction_retakes.py" \
    "$source_root/tests/test_dictation_editor.py" \
    "$source_root/tests/test_ai_first_final_resolution.py" \
    "$source_root/tests/test_fresh_install_provisioning.py"
fi

if [[ "$skip_build" != 1 ]]; then
  if [[ -z "$sign_identity" ]]; then
    sign_identity="$(security find-identity -v -p codesigning 2>/dev/null \
      | awk -F'"' '/Developer ID Application/ {print $2; exit}')"
  fi
  if [[ -z "$sign_identity" ]]; then
    sign_identity="$(security find-identity -v -p codesigning 2>/dev/null \
      | awk -F'"' '/Apple Development/ {print $2; exit}')"
  fi
  [[ -n "$sign_identity" ]] || die "a stable signing identity is required; ad-hoc signing would reset microphone permission"

  log "Build the exact source snapshot into the stable engine cache"
  "$source_root/scripts/build_juno_engine_bundle.sh" "$source_root/dist/juno_engine_bundle"
  log "Package and sign the staged application"
  if [[ -d "$deploy_root/staged" ]]; then
    safe_delete_owned_tree "$deploy_root/staged" "$deploy_root"
  fi
  mkdir -p "$deploy_root/staged"
  DEVELOPER_DIR="$developer_dir" "$source_root/scripts/package_juno_macos_app.sh" \
    --dist "$staged_app" \
    --engine "$source_root/dist/juno_engine_bundle" \
    --sign "$sign_identity"
fi

[[ -d "$staged_app" ]] || die "staged app does not exist: $staged_app"
codesign --verify --deep --strict --verbose=2 "$staged_app" >/dev/null
[[ -x "$staged_app/Contents/Resources/engine/.venv/bin/python" ]] || die "staged app has no bundled engine"

for relative_path in "${critical_paths[@]}"; do
  cmp -s "$source_root/$relative_path" \
    "$staged_app/Contents/Resources/engine/.venv/lib/python3.12/site-packages/$relative_path" \
    || die "staged app does not contain current source: $relative_path"
done

current_source_commit="$(git -C "$source_root" rev-parse HEAD)"
current_source_digest="$(source_delta_sha256 "$source_root")"
if [[ "$current_source_commit" != "$source_commit" || "$current_source_digest" != "$source_delta_digest" ]]; then
  die "source changed during build; staged app was not activated"
fi

next_app="/Applications/.Juno.app.next.$$"
restore_required=0
cleanup_next() {
  if [[ -d "$next_app" ]]; then
    if ! /usr/sbin/lsof +D "$next_app" >/dev/null 2>&1; then
      find "$next_app" -depth -delete 2>/dev/null || true
    fi
  fi
}
trap cleanup_next EXIT

log "Stage the verified application beside the active install"
ditto "$staged_app" "$next_app"
codesign --verify --deep --strict --verbose=2 "$next_app" >/dev/null

log "Activate with one rollback build"
osascript -e 'tell application "Juno" to quit' >/dev/null 2>&1 || true
pkill -x Juno >/dev/null 2>&1 || true
engine_pids="$(pgrep -f '[j]uno_v2.runtime.service' || true)"
if [[ -n "$engine_pids" ]]; then
  kill $engine_pids >/dev/null 2>&1 || true
fi
sleep 1

if [[ -d "$rollback_app" ]]; then
  safe_delete_owned_tree "$rollback_app" "$deploy_root"
fi
mkdir -p "$(dirname "$rollback_app")"
if [[ -d "$installed_app" ]]; then
  mv "$installed_app" "$rollback_app"
  restore_required=1
fi
mv "$next_app" "$installed_app"
codesign --verify --deep --strict --verbose=2 "$installed_app" >/dev/null
open -gja "$installed_app"

if ! wait_for_health "$local_health_url" 45 2 || ! wait_for_health "$public_health_url" 12 3; then
  log "Health failed; restore the known-good application"
  osascript -e 'tell application "Juno" to quit' >/dev/null 2>&1 || true
  pkill -x Juno >/dev/null 2>&1 || true
  engine_pids="$(pgrep -f '[j]uno_v2.runtime.service' || true)"
  if [[ -n "$engine_pids" ]]; then
    kill $engine_pids >/dev/null 2>&1 || true
  fi
  if /usr/sbin/lsof +D "$installed_app" >/dev/null 2>&1; then
    die "failed build still has open files; manual rollback is required from $rollback_app"
  fi
  find "$installed_app" -depth -delete 2>/dev/null || true
  if [[ "$restore_required" == 1 && -d "$rollback_app" ]]; then
    mv "$rollback_app" "$installed_app"
    open -gja "$installed_app"
    wait_for_health "$local_health_url" 45 2 || true
  fi
  die "new build did not pass local and public health; rollback restored"
fi

installed_site="$installed_app/Contents/Resources/engine/.venv/lib/python3.12/site-packages"
deployed_pipeline_sha="$(sha256_file "$installed_site/juno_core_v3/dictation/pipeline.py")"
deployed_editor_sha="$(sha256_file "$installed_site/juno_v2/writer/dictation_editor.py")"
created_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
mkdir -p "$receipt_dir"
receipt_tmp="$receipt_dir/.current.receipt.$$"
{
  printf 'format=juno-voice-backend-deploy-v1\n'
  printf 'created_at=%s\n' "$created_at"
  printf 'source_commit=%s\n' "$source_commit"
  printf 'source_branch=%s\n' "$source_branch"
  printf 'source_dirty=%s\n' "$source_dirty"
  printf 'source_delta_sha256=%s\n' "$source_delta_digest"
  printf 'source_identity=%s\n' "$source_identity"
  printf 'pipeline_sha256=%s\n' "$deployed_pipeline_sha"
  printf 'dictation_editor_sha256=%s\n' "$deployed_editor_sha"
  printf 'dgx_relay_sha256=%s\n' "$remote_relay_sha"
  printf 'local_health=ok\n'
  printf 'public_health=ok\n'
} >"$receipt_tmp"
chmod 600 "$receipt_tmp"
if [[ -f "$current_receipt" ]]; then
  mv "$current_receipt" "$previous_receipt"
fi
mv "$receipt_tmp" "$current_receipt"

log "Publish the non-secret deployment receipt to Spark"
remote_tmp="$dgx_receipt_dir/.current.receipt.$$"
ssh -o BatchMode=yes "$dgx_host" "mkdir -p '$dgx_receipt_dir' && chmod 700 '$dgx_receipt_dir'"
scp -q "$current_receipt" "$dgx_host:$remote_tmp"
ssh -o BatchMode=yes "$dgx_host" "
  set -eu
  chmod 600 '$remote_tmp'
  if test -f '$dgx_receipt_dir/current.receipt'; then
    mv '$dgx_receipt_dir/current.receipt' '$dgx_receipt_dir/previous.receipt'
  fi
  mv '$remote_tmp' '$dgx_receipt_dir/current.receipt'
"

# The packaged application has already been copied into /Applications and its
# predecessor is the single retained rollback. Keeping this third full bundle
# would consume another multi-gigabyte engine copy without adding a recovery
# path. Refresh the open-file check immediately before removing only this
# script-owned staging tree.
log "Remove the verified staging copy"
safe_delete_owned_tree "$deploy_root/staged" "$deploy_root"

log "Deployment complete"
printf 'source_identity=%s\n' "$source_identity"
printf 'receipt=%s\n' "$current_receipt"
printf 'rollback=%s\n' "$rollback_app"
