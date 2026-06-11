# Juno OTA Updates

Juno uses Sparkle 2 for macOS over-the-air updates. The app already contains
the updater code and packaging scripts; this document explains the external
accounts, keys, and release steps needed to make updates work in production.

## What You Need

You need three pieces of infrastructure:

1. Apple Developer Program membership.
   This is required for a Developer ID Application certificate and notarization.
   Juno can build locally with ad-hoc signing, but production OTA builds should
   be Developer ID signed and notarized so macOS and Sparkle can trust the
   installed update.

2. A static HTTPS host for update files.
   This can be a website, CDN, S3 plus CloudFront, Cloudflare R2, GitHub
   Releases plus a stable raw appcast URL, or any service that can serve public
   files over HTTPS. Juno's scripts generate the files; they do not upload them.

3. Sparkle EdDSA signing keys.
   Sparkle signs update archives with an Ed25519 key pair. The public key is
   embedded in `Info.plist` as `SUPublicEDKey`. The private key stays in the
   release machine's Keychain, or in an exported private key file that you keep
   offline and restrict tightly.

No app-side API key is needed for Sparkle itself. API keys may be needed only
for your hosting provider if you automate uploads.

The Sparkle update host and website download host can be the same server or
different servers. Treat them as different release surfaces:

- Sparkle OTA host: `appcast.xml`, versioned Sparkle `.zip` archives, release
  notes, and generated delta files.
- Website download host: versioned `.dmg` files for fresh installs.

Do not put the website `.dmg` in Sparkle's appcast. Sparkle expects the `.zip`
archive generated and signed by `generate_appcast`.

## One-Time Setup

### 1. Install Apple signing assets

Create or install a Developer ID Application certificate in the release Mac's
login Keychain. The signing identity should appear in:

```bash
security find-identity -v -p codesigning
```

Look for an identity like:

```text
Developer ID Application: Your Company Name (TEAMID)
```

That string is passed to release scripts as `--sign` or
`CODESIGN_IDENTITY`.

Use the exact identity string printed by `security find-identity`. The display
name is fixed by the Apple Developer Program membership that issued the
certificate. Individual memberships show the person's legal name; organization
memberships show the legal entity name.

### 2. Store notarization credentials

Create a notarytool Keychain profile on the release Mac. Apple supports Apple
ID app-specific-password credentials and App Store Connect API key credentials.
For a local release machine, the app-specific-password flow is usually enough:

```bash
xcrun notarytool store-credentials "juno-notary" \
  --apple-id "developer@example.com" \
  --team-id "TEAMID" \
  --password "app-specific-password"
```

The profile name, `juno-notary` in this example, is passed to
`--notary-keychain-profile` or `JUNO_NOTARY_KEYCHAIN_PROFILE`.

### 3. Generate the Sparkle EdDSA key pair

Build or resolve the Swift package once so Sparkle's tools are available:

```bash
swift build -c release --package-path shells/macos
```

Generate the Sparkle key pair:

```bash
./scripts/generate_juno_sparkle_keys.sh
```

Sparkle stores the private key in the login Keychain and prints a base64 public
key. Keep the public key for release builds:

```bash
export JUNO_OTA_PUBLIC_ED_KEY="paste-public-key-here"
```

Do not commit the private key. If releases need to move to another Mac or CI
runner, use Sparkle's `generate_keys` import/export options through
`./scripts/generate_juno_sparkle_keys.sh --help`, and store the exported private
key in a restricted secret manager.

### 4. Choose the appcast URL

Pick the final public URL for the appcast before shipping the first OTA-enabled
build. Example:

```bash
export JUNO_OTA_FEED_URL="https://updates.example.com/juno/appcast.xml"
export JUNO_OTA_DOWNLOAD_URL_PREFIX="https://updates.example.com/juno/"
```

The first OTA-enabled Juno build must be distributed manually. Future builds are
discovered from the appcast URL embedded in that first build.

## Per-Release Checklist

### 1. Build the engine bundle

```bash
./scripts/build_juno_engine_bundle.sh
```

This creates `dist/juno_engine_bundle`, which the OTA release script copies
into `Juno.app`.

### 2. Pick version numbers

Use a user-facing version and an incrementing build number:

```bash
export JUNO_APP_VERSION="0.2.1"
export JUNO_BUILD_NUMBER="2"
```

Sparkle compares `CFBundleVersion`, so `JUNO_BUILD_NUMBER` must increase for
every public update, even if the visible version is unchanged.

### 3. Build, sign, notarize, archive, and regenerate the appcast

```bash
./scripts/build_juno_ota_release.sh \
  --version "$JUNO_APP_VERSION" \
  --build-number "$JUNO_BUILD_NUMBER" \
  --sign "Developer ID Application: Your Company Name (TEAMID)" \
  --engine dist/juno_engine_bundle \
  --ota-feed-url "$JUNO_OTA_FEED_URL" \
  --ota-public-ed-key "$JUNO_OTA_PUBLIC_ED_KEY" \
  --download-url-prefix "$JUNO_OTA_DOWNLOAD_URL_PREFIX" \
  --notary-keychain-profile "juno-notary"
```

The script produces:

- `dist/Juno.app`
- `dist/ota/Juno-VERSION-BUILD.zip`
- `dist/ota/Juno-VERSION-BUILD.md` or copied release notes from
  `--release-notes`
- `dist/ota/appcast.xml`
- any Sparkle delta update files generated by `generate_appcast`

By default, the script creates a minimal markdown release note so Sparkle's
update dialog has content to show. Pass `--release-notes path/to/notes.md` for
custom `.html`, `.md`, or `.txt` notes.

### 4. Upload update files

Upload everything generated under `dist/ota/` to your update host. The public
URLs must match the feed URL and download URL prefix passed into the release
script. For the example above:

- `https://updates.example.com/juno/appcast.xml`
- `https://updates.example.com/juno/Juno-0.2.1-2.zip`
- `https://updates.example.com/juno/Juno-0.2.1-2.md` if release notes are not
  embedded by your Sparkle toolchain
- any generated `.delta` files next to the archive

Set cache headers conservatively for `appcast.xml` while testing. A short TTL
such as 5 minutes is easier to debug than a CDN-cached appcast.

Upload `appcast.xml` last. Sparkle clients use the appcast as the source of
truth, so publishing it before the archive is publicly readable can make update
checks fail.

### 5. Test from an older installed build

Install an older OTA-enabled build, launch Juno, and use:

- Juno menu bar: `Updates`
- Settings: `Updates & app` -> `Software updates`
- App menu command: `Check for Updates...`

For automatic-check testing, clear Sparkle's last-check timestamp:

```bash
defaults delete com.juno.shell SULastCheckTime
```

Then relaunch Juno.

## Production Release Flow

Use this flow for a public release from a release Mac. It builds one
Developer ID signed and notarized `Juno.app`, publishes Sparkle OTA files, and
optionally creates a notarized DMG from that same app for website downloads.

### 1. Set release variables

Use the real production update host. Keep a separate output directory for
production artifacts so old localhost appcasts are never uploaded by mistake.

```bash
export JUNO_APP_VERSION="0.2.1"
export JUNO_BUILD_NUMBER="2"
export CODESIGN_IDENTITY="Developer ID Application: Your Company Name (TEAMID)"
export JUNO_OTA_PUBLIC_ED_KEY="paste-public-key-here"
export JUNO_OTA_FEED_URL="https://updates.example.com/appcast.xml"
export JUNO_OTA_DOWNLOAD_URL_PREFIX="https://updates.example.com/"
export JUNO_NOTARY_KEYCHAIN_PROFILE="juno-notary"
export JUNO_RELEASE_UPDATES_DIR="dist/ota-public"
```

`JUNO_BUILD_NUMBER` must increase for every public release. Reusing a build
number can make Sparkle ignore the update.

### 2. Build the OTA release

```bash
./scripts/build_juno_engine_bundle.sh

rm -rf "$JUNO_RELEASE_UPDATES_DIR"
mkdir -p "$JUNO_RELEASE_UPDATES_DIR"

./scripts/build_juno_ota_release.sh \
  --version "$JUNO_APP_VERSION" \
  --build-number "$JUNO_BUILD_NUMBER" \
  --sign "$CODESIGN_IDENTITY" \
  --engine dist/juno_engine_bundle \
  --updates-dir "$JUNO_RELEASE_UPDATES_DIR" \
  --ota-feed-url "$JUNO_OTA_FEED_URL" \
  --ota-public-ed-key "$JUNO_OTA_PUBLIC_ED_KEY" \
  --download-url-prefix "$JUNO_OTA_DOWNLOAD_URL_PREFIX" \
  --notary-keychain-profile "$JUNO_NOTARY_KEYCHAIN_PROFILE"
```

Pass `--release-notes path/to/notes.md` if you want custom notes in the Sparkle
update dialog. Without it, the script writes minimal release notes.

### 3. Verify before publishing

```bash
xcrun stapler validate dist/Juno.app
spctl -a -vv dist/Juno.app
codesign --verify --deep --strict --verbose=2 dist/Juno.app

grep -E "sparkle:version|sparkle:shortVersionString|enclosure url" \
  "$JUNO_RELEASE_UPDATES_DIR/appcast.xml"

grep -E "127\\.0\\.0\\.1|localhost|updates\\.example\\.com" \
  "$JUNO_RELEASE_UPDATES_DIR/appcast.xml" && {
    echo "error: appcast contains a non-production URL" >&2
    exit 1
  }
```

Expected results:

- `stapler validate` works.
- `spctl` reports `accepted` and `source=Notarized Developer ID`.
- `codesign --verify` succeeds.
- The appcast enclosure URL points at the production update host and the new
  versioned `.zip`.

### 4. Publish OTA files

Publish these files from `$JUNO_RELEASE_UPDATES_DIR`:

- `appcast.xml`
- `Juno-VERSION-BUILD.zip`
- `Juno-VERSION-BUILD.md`, `.html`, or `.txt`, if present
- generated `.delta` files, if present

Do not publish `$JUNO_RELEASE_UPDATES_DIR/.notary` or temporary staging
directories.

For an SSH-managed static host, use a staging directory and move `appcast.xml`
last:

```bash
REMOTE_HOST="deploy@example.com"
REMOTE_DIR="/var/www/updates.example.com"
STAGING="$REMOTE_DIR/.incoming-$(date -u +%Y%m%dT%H%M%SZ)"

ssh "$REMOTE_HOST" "mkdir -p '$STAGING'"
rsync -av \
  --exclude '.notary/' \
  --exclude 'appcast.xml' \
  "$JUNO_RELEASE_UPDATES_DIR"/ \
  "$REMOTE_HOST:$STAGING"/
rsync -av "$JUNO_RELEASE_UPDATES_DIR/appcast.xml" "$REMOTE_HOST:$STAGING/appcast.xml"
ssh "$REMOTE_HOST" "\
  set -e; \
  find '$STAGING' -maxdepth 1 -type f ! -name appcast.xml -exec mv {} '$REMOTE_DIR'/ \\;; \
  mv '$STAGING/appcast.xml' '$REMOTE_DIR/appcast.xml'; \
  rmdir '$STAGING'"
```

On nginx or CDN-backed static hosting, use short cache headers for the appcast
and immutable headers for versioned archives:

```text
appcast.xml: no-cache, max-age=60
Juno-*.zip: public, max-age=31536000, immutable
Juno-*.delta: public, max-age=31536000, immutable
Juno-*.md/html/txt: public, max-age=31536000, immutable
```

### 5. Verify public OTA URLs

```bash
curl -I "$JUNO_OTA_FEED_URL"
curl -fsSL "$JUNO_OTA_FEED_URL" \
  | grep -E "sparkle:version|sparkle:shortVersionString|enclosure url"

ARCHIVE_URL="$(curl -fsSL "$JUNO_OTA_FEED_URL" \
  | sed -n 's/.*<enclosure url="\([^"]*\)".*/\1/p' \
  | head -n 1)"
curl -I "$ARCHIVE_URL"
```

Both the appcast and archive should return HTTP 200. Test from an older
installed Juno build after the public URLs are verified.

## Website DMG Release

The DMG is for fresh installs from the website. Build it from the same
Developer ID signed and stapled `dist/Juno.app` produced by the production OTA
release flow. Do not rebuild ad-hoc for the DMG.

### 1. Create the DMG

```bash
DMG_NAME="Juno-$JUNO_APP_VERSION-$JUNO_BUILD_NUMBER.dmg"
DMG_PATH="dist/downloads/$DMG_NAME"

rm -rf dist/dmg-stage dist/downloads
mkdir -p dist/dmg-stage dist/downloads
ditto dist/Juno.app dist/dmg-stage/Juno.app
ln -s /Applications dist/dmg-stage/Applications

hdiutil create \
  -volname "Juno" \
  -srcfolder dist/dmg-stage \
  -ov \
  -format UDZO \
  "$DMG_PATH"
```

This creates a simple drag-to-install disk image containing `Juno.app` and an
`Applications` symlink.

### 2. Sign, notarize, and staple the DMG

```bash
codesign --force \
  --sign "$CODESIGN_IDENTITY" \
  --timestamp \
  "$DMG_PATH"

xcrun notarytool submit "$DMG_PATH" \
  --keychain-profile "$JUNO_NOTARY_KEYCHAIN_PROFILE" \
  --wait

xcrun stapler staple "$DMG_PATH"
```

If `notarytool` cannot find the profile, recreate it with
`xcrun notarytool store-credentials` on the release Mac. Do not put the Apple
ID password, app-specific password, API key, or Keychain profile contents in the
repository.

### 3. Verify the DMG

```bash
xcrun stapler validate "$DMG_PATH"
spctl -a -t open --context context:primary-signature -vv "$DMG_PATH"
codesign --verify --verbose=2 "$DMG_PATH"
hdiutil verify "$DMG_PATH"
```

Expected results:

- `stapler validate` works.
- `spctl` reports `accepted` and `source=Notarized Developer ID`.
- `codesign --verify` succeeds.
- `hdiutil verify` reports a valid checksum.

### 4. Publish the website download

Upload the versioned DMG to the website download host, for example:

```text
https://downloads.example.com/Juno-0.2.1-2.dmg
```

Use a long immutable cache header for versioned DMGs:

```text
Juno-*.dmg: public, max-age=31536000, immutable
```

If the website also has a stable `latest` download URL, update that pointer only
after the versioned DMG is uploaded and returns HTTP 200:

```bash
curl -I "https://downloads.example.com/Juno-$JUNO_APP_VERSION-$JUNO_BUILD_NUMBER.dmg"
```

## Local Appcast Testing

For local testing, serve the appcast over `http://127.0.0.1`. Sparkle may reject
`file://` feed URLs during update checks even when Juno's packaging script
allows insecure feed URLs for local builds. Production feeds should use HTTPS.

The installed app must be older than the item in `appcast.xml`. This example
installs `0.2.0` build `1`, then serves `0.2.1` build `2` as the update:

```bash
export JUNO_OTA_FEED_URL="http://127.0.0.1:8000/appcast.xml"
export JUNO_OTA_DOWNLOAD_URL_PREFIX="http://127.0.0.1:8000/"
export JUNO_OTA_PUBLIC_ED_KEY="paste-public-key-here"

./scripts/build_juno_engine_bundle.sh
rm -rf dist/ota
mkdir -p dist/ota
```

Build and install the older OTA-enabled app:

```bash
./scripts/build_juno_ota_release.sh \
  --version 0.2.0 \
  --build-number 1 \
  --sign "-" \
  --engine dist/juno_engine_bundle \
  --ota-feed-url "$JUNO_OTA_FEED_URL" \
  --ota-public-ed-key "$JUNO_OTA_PUBLIC_ED_KEY" \
  --download-url-prefix "$JUNO_OTA_DOWNLOAD_URL_PREFIX" \
  --allow-insecure-ota-feed

osascript -e 'tell application "Juno" to quit' 2>/dev/null || true
rm -rf /Applications/Juno.app
ditto dist/Juno.app /Applications/Juno.app
```

Build the newer update, but do not manually install this build:

```bash
./scripts/build_juno_ota_release.sh \
  --version 0.2.1 \
  --build-number 2 \
  --sign "-" \
  --engine dist/juno_engine_bundle \
  --ota-feed-url "$JUNO_OTA_FEED_URL" \
  --ota-public-ed-key "$JUNO_OTA_PUBLIC_ED_KEY" \
  --download-url-prefix "$JUNO_OTA_DOWNLOAD_URL_PREFIX" \
  --allow-insecure-ota-feed
```

Wait for the script to finish. `appcast.xml` is created after the archive is
fully written and Sparkle's `generate_appcast` command has inspected it:

```bash
ls -la dist/ota
grep -E "sparkle:version|shortVersionString|enclosure url" dist/ota/appcast.xml
```

Serve the update directory:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory dist/ota
```

In another terminal, verify both URLs before checking from Juno:

```bash
curl -I http://127.0.0.1:8000/appcast.xml
curl -I http://127.0.0.1:8000/Juno-0.2.1-2.zip
```

Both should return `200 OK`. Then launch the older installed app and check for
updates:

```bash
defaults delete com.juno.shell SULastCheckTime 2>/dev/null || true
open /Applications/Juno.app
```

Use Juno's `Updates` menu bar item, Settings -> `Updates & app`, or the app menu
command to check for updates.

Use this only to verify feed wiring and UI behavior. For production-like
testing, use Developer ID signing, notarization, and HTTPS.

## Developer ID Permission Testing

Use this flow when testing whether macOS permissions persist across OTA updates.
Ad-hoc signing (`--sign "-"`) is useful for feed wiring, but it is not a valid
permission-persistence test because each rebuild can have a different code
identity.

First, confirm the release Mac has a Developer ID Application identity installed:

```bash
security find-identity -v -p codesigning
```

Look for:

```text
Developer ID Application: Your Company or Name (TEAMID)
```

Then export that exact identity string:

```bash
export CODESIGN_IDENTITY="Developer ID Application: Your Company or Name (TEAMID)"
```

If the Developer Program membership is an individual account, add that Apple ID
to Xcode on the release Mac or import the Developer ID Application certificate
with its private key into the login Keychain. Adding another Apple ID to an
individual membership only grants App Store Connect access; it does not make
that Apple ID a full Developer Program team member for signing assets.

Store notarization credentials once:

```bash
xcrun notarytool store-credentials "juno-notary" \
  --apple-id "developer@example.com" \
  --team-id "TEAMID" \
  --password "app-specific-password"
```

Build and install a Developer ID signed baseline, then grant Juno's permissions:

```bash
./scripts/build_juno_ota_release.sh \
  --version 0.2.0 \
  --build-number 1 \
  --sign "$CODESIGN_IDENTITY" \
  --engine dist/juno_engine_bundle \
  --ota-feed-url "$JUNO_OTA_FEED_URL" \
  --ota-public-ed-key "$JUNO_OTA_PUBLIC_ED_KEY" \
  --download-url-prefix "$JUNO_OTA_DOWNLOAD_URL_PREFIX" \
  --notary-keychain-profile "juno-notary" \
  --allow-insecure-ota-feed

osascript -e 'tell application "Juno" to quit' 2>/dev/null || true
rm -rf /Applications/Juno.app
ditto dist/Juno.app /Applications/Juno.app
open /Applications/Juno.app
```

Verify the installed app is not ad-hoc signed:

```bash
codesign -dv --verbose=4 /Applications/Juno.app 2>&1 \
  | grep -E "Authority|TeamIdentifier|Signature"
```

Expected output includes `Authority=Developer ID Application` and
`TeamIdentifier=TEAMID`. It should not include `Signature=adhoc`.

After permissions are granted, build the next update with the same signing
identity and notary profile:

```bash
./scripts/build_juno_ota_release.sh \
  --version 0.2.1 \
  --build-number 2 \
  --sign "$CODESIGN_IDENTITY" \
  --engine dist/juno_engine_bundle \
  --ota-feed-url "$JUNO_OTA_FEED_URL" \
  --ota-public-ed-key "$JUNO_OTA_PUBLIC_ED_KEY" \
  --download-url-prefix "$JUNO_OTA_DOWNLOAD_URL_PREFIX" \
  --notary-keychain-profile "juno-notary" \
  --allow-insecure-ota-feed
```

Serve `dist/ota`, check for updates from the installed app, and install the
update. Permissions may prompt once when moving from an ad-hoc build to
Developer ID signing because the app identity changed. The real persistence
test is Developer ID baseline -> Developer ID update with the same bundle ID,
same Team ID, and same signing identity.

## Channels

Stable releases do not need a channel:

```bash
--ota-channel stable
```

or omit `--ota-channel` entirely.

For beta releases:

```bash
--ota-channel beta
```

The packaging script writes `JunoUpdateChannel` into `Info.plist`; the updater
allows only that non-stable channel. A stable build will not automatically see a
beta-only update.

## Secrets And Where They Live

| Secret | Required | Where it lives | Commit it? |
| --- | --- | --- | --- |
| Developer ID private key | Production | macOS Keychain on release machine | No |
| Notary credentials | Production notarization | notarytool Keychain profile or CI secret | No |
| Sparkle EdDSA private key | All Sparkle update signing | macOS Keychain or restricted exported key | No |
| Sparkle EdDSA public key | All OTA-enabled builds | `SUPublicEDKey` in app `Info.plist` | OK |
| Hosting API token | Only automated upload | Hosting provider secret manager | No |

## Troubleshooting

- `Updates are not configured for this build`: the app was packaged without
  both `--ota-feed-url` and `--ota-public-ed-key`, or `--disable-ota` was set.
- Sparkle says no update is available: confirm `CFBundleVersion` increased and
  that the installed app is older than the appcast item.
- Local update checks fail with `404`: confirm the local HTTP server is serving
  the directory that contains `appcast.xml`, and confirm the release script has
  finished running `generate_appcast`.
- Download fails: confirm the archive URL in `appcast.xml` is public and
  matches the uploaded file.
- Signature validation fails: regenerate the appcast on the Mac that has the
  Sparkle private key, then upload the refreshed archive, delta files, and
  appcast together.
- macOS blocks the app: confirm the app was Developer ID signed, notarized, and
  stapled before the final Sparkle archive was created.
- `notarytool` cannot find the Keychain profile: rerun
  `xcrun notarytool store-credentials` on the release Mac, or pass the correct
  profile name to `--notary-keychain-profile`.
- DMG is signed but macOS reports `Unnotarized Developer ID`: submit the
  DMG itself to `notarytool`, wait for `Accepted`, then run
  `xcrun stapler staple` on the DMG. Notarizing only `Juno.app` is not enough
  for a website DMG download.

## References

- Sparkle setup and EdDSA keys: https://sparkle-project.org/documentation/
- Sparkle publishing guide: https://sparkle-project.org/documentation/publishing/
- Apple Developer ID certificates: https://developer.apple.com/help/account/certificates/create-developer-id-certificates
- Apple notarization workflow: https://help.apple.com/xcode/mac/current/en.lproj/dev88332a81e.html
