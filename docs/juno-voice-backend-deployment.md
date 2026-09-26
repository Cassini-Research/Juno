# Juno voice backend deployment

The phone-facing API does not run a second Juno model on the DGX Spark. The
Spark process is an authenticated, memory-only relay. Its upstream is the
installed Juno engine on the Mac, reached through a reverse SSH tunnel.

```text
iPhone keyboard
  -> public HTTPS speech route
  -> DGX Spark relay
  -> reverse SSH forward
  -> Mac-local authenticated speech bridge
  -> /Applications/Juno.app engine
  -> final ASR + vocabulary + memory + writer
```

Consequently, pulling source on either host does not update the API. The
installed Mac application is the executable backend. A release is current only
after the exact source snapshot is packaged, activated, verified through the
public phone route, and recorded on both machines.

## Audit without changing the service

```sh
scripts/deploy_juno_voice_backend.sh audit \
  --source-root /path/to/Juno \
  --dgx-host "$JUNO_VOICE_DGX_HOST" \
  --public-health "$JUNO_VOICE_PUBLIC_HEALTH_URL"
```

The audit compares the final-text pipeline, self-correction logic, writer,
workbench and socket dispatcher inside `/Applications/Juno.app` with the chosen
checkout. It also checks the local bridge, public route, Spark relay hash and
service state. It never reads or prints either bearer key.

## Deploy a tested source revision

Keep hostnames and endpoints in an owner-only local environment, not in Git:

```sh
export JUNO_VOICE_DGX_HOST='user@spark-host'
export JUNO_VOICE_DGX_RECEIPT_DIR='/home/user/path/to/juno-relay/deployments'
export JUNO_VOICE_PUBLIC_HEALTH_URL='https://voice.example.com/healthz'

scripts/deploy_juno_voice_backend.sh deploy --source-root /path/to/Juno
```

The default path requires a clean Git source revision, runs the focused
final-text regressions, reuses Juno's stable engine cache, signs with a stable
developer identity, verifies that the staged bundle contains the selected
source, and keeps one rollback application. It then activates the build and
requires both the Mac bridge and the public phone route to become healthy. A
failed health gate restores the previous app automatically.

After a successful activation, the packaging copy is removed. The active app
and one intentional rollback are the only retained application bundles; the
stable engine cache remains for the next incremental build.

The build pins the full Xcode toolchain at
`/Applications/Xcode.app/Contents/Developer` unless `DEVELOPER_DIR` is already
set. This prevents a machine whose active developer directory is only the
Command Line Tools from silently producing a shell without SwiftUI macros.

After success, a mode-600, non-secret receipt is rotated as `current` and
`previous` locally and on the Spark. The receipt identifies the source commit,
dirty-snapshot digest when explicitly allowed, critical installed-file hashes,
relay hash and health result. The script does not copy model weights to the
Spark, expose an admin port, rotate credentials or change the public route.

`--allow-dirty` exists for an explicitly fingerprinted private build, but a
clean commit is the normal release boundary. `--skip-build` is for retrying an
already verified staged application; `--skip-tests` should be reserved for a
retry where the same source identity already passed the recorded test run.

## What final-text behavior the API receives

The Mac bridge calls Juno's `POST /api/broker/dictation/ingest_wav` route with
`transcript_stage=final_delivery`. This is the same one-shot pipeline used by
the Mac product: ASR normalization, personal vocabulary/memory, conservative
self-correction handling, and the dictation editor run before the API returns
`text`. The DGX relay forwards that final response; it does not replace it with
raw ASR.

This supports corrections contained within one utterance, such as “3 PM,
scratch that, 4 PM.” It does not let a later keyboard utterance edit arbitrary
text already owned by another iOS app; cross-utterance editing requires an
explicit client/session protocol and undo boundary.

## Mac-independent Spark route

`scripts/juno_voice_spark_relay.py` is the bounded alternative for an always-on
DGX Spark that already hosts the locked Parakeet service. It adapts the phone's
`juno-local` multipart contract to the Spark's pinned Parakeet model and runs
Juno's deterministic self-correction pass on the result. Deploy it beside an
exact copy of `juno_core_v3/dictation/self_corrections.py` named
`self_corrections.py`; record both hashes in the release receipt. The supplied
`juno-voice-spark.service` binds the relay to loopback and keeps the public and
Parakeet credentials in separate owner-only files.

Publish that loopback service with HTTPS Funnel on the Spark itself. Do not
switch a phone from the Mac endpoint until independent public DNS, pinned TLS,
unauthenticated rejection, authenticated session lifecycle, and a real speech
upload all pass. Merely moving public ingress is not Mac independence: the
relay's upstream must point to Spark loopback, never the reverse-SSH port.
