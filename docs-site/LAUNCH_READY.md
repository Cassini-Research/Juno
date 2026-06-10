# Juno Docs Launch Readiness

Date: 2026-05-20

## Current state

- The `/docs` landing page now uses a custom Juno product homepage instead of the default generated docs title stack.
- The homepage includes a launch-video placeholder with a concrete storyboard for the future intro video.
- The visual system now follows the current Juno app and website direction: warm paper, editorial display type, dark HUD island, rounded product cards, and direct first-run CTAs.
- Reusable docs blocks for steps, videos, screenshots, and diagrams have been restyled so inner pages no longer look like raw technical scaffolding.
- The core first-use pages have been rewritten in product language: What is Juno, Quickstart, First Launch, Home, Speak and Insert, Actions, Safe Insertion and Copy Ready, and Settings.
- Public docs now include current Home and Actions pages.
- Settings docs use the current section names: General, Dictation, Audio input, Writing & language, Storage, Privacy & learning, Permissions, Models, Updates & app, and Developer.
- HUD docs use the current public state model and remove the stale extra finalizing state.
- Developer reference pages were tightened against the current UDS/HTTP route inventory and no longer promise the removed tools, streaming, clipboard, or OpenAI-compatible route examples.
- All referenced screenshot and video assets exist under `public/images` and `public/videos`.
- Media has been corrected to use live installed-app captures from `/Applications/Juno.app` after a clean rebuild/reset/reinstall.
- Canonical screenshot paths now point to real installed-app captures for first launch, Home, History, Actions, Voice Commands, Styles, Dictionary & Memory, Per-app writing, Privacy, Settings, permissions, recording retention, quickstart, and HUD/try-it surfaces.
- `public/videos/demos/juno-product-overview.mp4` and `public/videos/demos/juno-first-launch.mp4` were rebuilt from real macOS recordings of `/Applications/Juno.app`; the task-specific demo MP4s reuse those real recordings instead of generated composites.
- `scripts/build-live-media-from-captures.mjs` is the current media publish script. `scripts/generate-production-media.mjs` is now only a fallback composite generator.
- `npm run validate:stubs` passed on 79 production MDX pages.
- `npm run typecheck` passed.
- `npm run build` passed and generated 82 static pages.
- The media reference check produced no `MISSING` lines.
- The live media correction was validated after the media replacement and caption updates.
- Browser testing passed across all 79 MDX docs routes.
- Search now has a backing `/api/search` route and was verified from the visible search dialog with the query `Actions`.
- After the May 19 refresh, `npm run validate:stubs`, `npm run typecheck`, media-reference check, route sweep, browser visual QA, and `npm run build` all pass.
- The May 19 build generated 80 static pages, including `/api/search`.
- A post-redesign route sweep passed across all 79 docs routes with no bad routes, and `/api/search?query=Actions` returned Actions results.
- Architecture diagrams and poster diagrams now come from `scripts/generate-juno-diagrams.mjs` and share one Juno visual system.
- Public Architecture pages have been rewritten in product-facing language and no longer expose historical package names, developer workbench details, source-file/source-map provenance, token details, or socket details.
- Diagram-heavy Architecture, Privacy Overview, Clipboard and Pasteboard, and What is Juno pages use wide layout for readable diagrams.
- `DiagramBlock` no longer renders source-file provenance in public docs.
- `RouteTable` and `PrivacyMatrix` use the same Juno-styled table treatment; `RouteTable` keeps internal source props in MDX only and does not render them.
- The public-section internal-term sweep passed across Start, Use Juno, Privacy, Troubleshooting, Architecture, components, and public diagram/poster assets.
- The Reference section has been rebuilt as user-facing product reference instead of internal API documentation.
- Removed the public Reference pages for environment variables, storage paths, helper binaries, tests, and local API route groups.
- Added product reference pages for shortcuts/HUD, Settings, privacy, History and recordings, Actions, styles/writing, and quick troubleshooting.
- Refreshed the docs against the current Whisper HUD branch: the sidebar is now Home, History, Actions, Voice Commands, Styles, Dictionary & Memory, Per-app writing, Privacy, Settings; first launch is Welcome, Access, Shortcut, Setup, Actions, Try it; and public docs describe the current Whisper-driven live preview path.
- The latest Styles/Voice Commands correction pass checked the current open PR, kept public routes and copy on Styles, split editing commands into Voice Commands, refreshed current screenshot/video assets, removed stale public per-app naming, and verified that remaining `mode` mentions are intentional internal/API or generic-language terms.
- Developer docs now use the current `juno_v2/` and `juno_core_v3/` package names.
- A latest public-doc sweep found no remaining old package-name references in docs content.
- The latest validation pass after the Reference rebuild passed: `npm run validate:stubs`, `npm run typecheck`, and `npm run build`.
- The latest build generated 79 static pages after deleting stale internal Reference routes.
- Local route sweep passed across all 75 current docs routes; deleted internal Reference routes returned 404; search returned Actions results.
- The latest validation pass after the Styles correction passed: `npm run validate:stubs` on 75 production MDX pages, `npm run typecheck`, media-reference check, and `npm run build`.
- The latest build generated 79 static pages.
- Local route sweep passed across all 75 current docs routes; old Modes routes now return 404; `/api/search?query=Styles` returned status 200 with Styles results.

## Final checks

Run from `docs-site/`:

```bash
npm run validate:stubs
npm run typecheck
npm run build
```

Then confirm:

```bash
find content/docs -name '*.mdx' -print0 | xargs -0 rg -o 'src="(/images/[^"]+|/videos/[^"]+)"' | sed 's/.*src="//;s/"$//' | sort -u | while read p; do test -e "public${p}" || echo "MISSING $p"; done
```

Expected result: no `MISSING` lines.

## Remaining media note

The public docs now reference real installed-app captures for product screenshots and real installed-app recordings for demo videos. The secure-field page uses the real Privacy screen rather than a password-field mockup.

## Remaining editorial note

The homepage, first-use path, architecture visuals, and Reference section now set the quality bar. The deeper troubleshooting, privacy, developer, and release pages should still get a final human editorial pass before public launch.
