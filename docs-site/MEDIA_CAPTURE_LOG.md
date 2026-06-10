# Juno Docs Media Capture Log

Date: 2026-05-20

## Source of truth used

- App repo: current Juno source tree
- Docs repo: `docs-site/`
- Latest app branch inspected: current app audit branch on 2026-05-19.
- Current Swift UI surfaces: `JunoMainChrome.swift`, `JunoOnboarding.swift`, `JunoActionsPage.swift`, `JunoVoiceCommandsPage.swift`, `JunoModesView.swift`, `MemoryManagementView.swift`, `SurfacePresetsView.swift`, `JunoPrivacyView.swift`, `JunoSettingsView.swift`.
- Current preview source: `juno_v2/preview/streaming_core.py`, `juno_v2/preview/live_agreement.py`, `juno_v2/preview/streaming_service.py`.

## Media produced

The media set was corrected from generated composites to real captures from the installed app. Juno was rebuilt, reset, reinstalled, opened, permissioned, and then driven through onboarding and the main app before capture:

```bash
cd <juno-source-tree>
env APP=/Applications/Juno.app ./scripts/fresh_juno_macos_environment.sh --open
```

Raw live evidence is stored in `public/images/screenshots/live/real/`. The canonical production screenshot filenames below now point at those real installed-app captures where a current app surface exists.

Current screenshots now published:

- `public/images/screenshots/actions-view.png`
- `public/images/screenshots/copy-ready-output.png`
- `public/images/screenshots/dictionary-memory.png`
- `public/images/screenshots/first-launch-permissions.png`
- `public/images/screenshots/first-launch-actions.png`
- `public/images/screenshots/first-launch-ready.png`
- `public/images/screenshots/first-launch-welcome.png`
- `public/images/screenshots/history-view.png`
- `public/images/screenshots/home-overview.png`
- `public/images/screenshots/hud-listening-target-app.png`
- `public/images/screenshots/hud-state-grid.svg`
- `public/images/screenshots/per-app-writing-controls.png`
- `public/images/screenshots/privacy-view.png`
- `public/images/screenshots/styles-view.png`
- `public/images/screenshots/voice-commands-view.png`
- `public/images/screenshots/notes-view.png`
- `public/images/screenshots/quickstart-cursor-before-dictation.png`
- `public/images/screenshots/quickstart-inserted-result.png`
- `public/images/screenshots/recording-retention-setting.png`
- `public/images/screenshots/settings-overview.png`
- `public/images/screenshots/settings-permissions.png`

No generated product screenshots are referenced by the public docs. The secure-field page uses the real Privacy screen rather than a password-field mockup.

Live-derived videos:

- `public/videos/demos/juno-product-overview.mp4`
- `public/videos/demos/juno-first-launch.mp4`
- `public/videos/demos/juno-first-dictation.mp4`
- `public/videos/demos/juno-styles-demo.mp4`
- `public/videos/demos/juno-copy-ready-fallback.mp4`
- `public/videos/demos/juno-shortcut-hud-states.mp4`
- `public/videos/demos/juno-speak-and-insert.mp4`

Video properties:

- Source: real macOS screen recordings of `/Applications/Juno.app`.
- `juno-first-launch.mp4`: real onboarding recording from Welcome through Try It.
- `juno-product-overview.mp4`: real main-app navigation recording across Home, History, Actions, Voice Commands, Styles, Dictionary & Memory, Per-app writing, Privacy, and Settings.
- The shorter task-specific demo MP4s intentionally reuse the relevant real recording instead of generated or composite media.

## Regeneration

Run from `docs-site/`:

```bash
npm run media:live
```

This script maps the live cropped captures into the canonical `public/images/screenshots/*.png` paths and rebuilds the published MP4 files from live app media.

The older generator remains in the repo for fallback/composite creation, but it is not the current production-media source of truth:

```bash
node scripts/generate-production-media.mjs
```

Use `node scripts/generate-production-media.mjs` only for temporary fallback art while waiting on a real app capture pass. Do not treat generated composites as production screenshots or product videos.

## 2026-05-20 real capture note

The installed app was driven through onboarding and the main product after Accessibility was confirmed. Real foreground-window screenshots were captured for onboarding, Home, History, Actions, Voice Commands, Styles, Dictionary & Memory, Per-app writing, Privacy, and Settings. Real screen recordings were captured for first launch and product overview.

## 2026-05-19 validation

- Superseded by the 2026-05-20 real capture pass above. Do not use the generated media pass as the production source of truth.
- `npm run validate:stubs` passed on 76 production MDX pages.
- `npm run typecheck` passed.
- Media-reference check produced no missing asset paths.
- `npm run build` passed.
- Browser visual QA passed on the docs homepage, First Launch, Voice Commands, Per-app writing, and Privacy pages with no broken images on those pages.
