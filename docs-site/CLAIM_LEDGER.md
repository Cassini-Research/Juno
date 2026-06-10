# Claim Ledger

Last updated: 2026-05-19

This ledger tracks production-doc claims against the current Juno source tree. Public docs should use product language; developer docs may name internal routes, storage paths, and helper binaries only when required for implementation work.

## Product Claims

| Claim | Current source of truth | Docs status |
| --- | --- | --- |
| Juno is a local-first macOS voice writing app. | `CURRENT_PRODUCT_TRUTH.md`; `MAC_RUNTIME_TRUTH.md` | Reflected across `index.mdx`, `start/what-is-juno.mdx`, privacy pages, and architecture pages. |
| The default shortcut is Fn / Globe, configurable from Settings → Dictation → Shortcut. | `JunoSettingsView.swift`; `JunoShellApp.swift` | Reflected in quickstart, shortcuts, and Settings docs. |
| The current app sidebar is Home, History, Actions, Voice Commands, Styles, Dictionary & Memory, Per-app writing, Privacy, Settings. | `JunoMainChrome.swift`; current app audit branch | Reflected in `use-juno/meta.json`, the homepage guide list, current user-guide pages, and generated media. |
| First launch is a six-step flow: Welcome, Access, Shortcut, Setup, Actions, Try it. Setup shows Starting up, Downloading models, and Warming up. | `JunoOnboarding.swift`; installed `/Applications/Juno.app` rebuilt from `./scripts/fresh_juno_macos_environment.sh --open` on 2026-05-19 | Reflected in `start/first-launch.mdx`, onboarding screenshots, and launch media. |
| The live HUD preview is Whisper-driven with LocalAgreement-2. The removed streaming preview backends should not be described as the production preview path. | `juno_v2/preview/streaming_core.py`; `juno_v2/preview/live_agreement.py`; `juno_v2/preview/streaming_service.py`; PR #88 | Reflected in developer engine docs and public HUD copy without exposing unnecessary backend names. |
| The old Dictionary & Memory Styles category has been removed. Dictionary now documents Vocabulary, Corrections, Snippets, and Replacements. | `MemoryManagementView.swift`; PR #76 | Reflected in `use-juno/dictionary-and-memory.mdx` and Smart Context docs. |
| Home is the readiness surface for setup, voice-engine connection, stats, recent dictations, and Actions entry. | `JunoMainChrome.swift`; `JunoHomeHeroCard.swift`; `JunoHomeStatsGraph.swift`; `JunoHomeRecentList.swift` | Reflected in new `use-juno/home.mdx` and `home-overview.png`. |
| Actions is the dedicated surface for Notes, Reminders, and Alarms. Dictation and Voice Commands work when Actions are off. | `JunoActionsPage.swift`; `JunoActionCatalog.swift`; `JunoVoiceCommandsPage.swift` | Reflected in `use-juno/actions.mdx`, `use-juno/voice-commands.mdx`, legacy `use-juno/notes.mdx`, and current generated media. |
| Wake word alone is not enough to create an action. Juno needs explicit action intent. | `JunoActionsPage.swift`; `JunoActionDTOs.swift`; action eval docs/tests | Reflected in `use-juno/actions.mdx`. |
| HUD should not show Listening until microphone frames and speech energy are detected. | `CURRENT_PRODUCT_TRUTH.md`; `MAC_RUNTIME_TRUTH.md`; `JunoShellApp.swift` | Reflected in `use-juno/hud-states.mdx`, `start/shortcuts-and-hud.mdx`, and `hud-state-grid.svg`. |
| Current public HUD model is Checking, Checking microphone, Waiting for speech, Listening, Polishing, Transcribing, Inserted, Copy Ready, Blocked, Error. | `JunoShellApp.swift`; `HUDTranscriptStore.swift`; current generated HUD media | Reflected in HUD docs and current HUD state media. |
| Secure/password fields suppress context, learning, history, audio save, and paste. | `CURRENT_PRODUCT_TRUTH.md`; `JunoLocalCapability.swift`; privacy policy implementation | Reflected in privacy, app integration, safe insertion, and HUD docs. |
| Smart Context, selected text, current app/field, app/window title, and learning from corrections default on; recent clipboard defaults off. | `CURRENT_PRODUCT_TRUTH.md`; `JunoSettingsView.swift` | Reflected in Settings and privacy docs. |
| History defaults to 90 days and recordings default to 30 days. | `CURRENT_PRODUCT_TRUTH.md`; settings/backend defaults | Reflected in History, Recordings, Settings, and privacy retention docs. |
| Current language choices are Auto-detect, English, Hindi + English, Mandarin Chinese, Spanish, and Keep original. | `JunoSettingsView.swift` | Reflected in Settings and Writing Workflows docs. |

## Current Settings Labels

| Section | Exact current labels used in docs |
| --- | --- |
| General | Appearance, Show in Dock, Keep recordings for, Live transcriptions, HUD position. |
| Dictation | Shortcut, HUD sounds, Show word count after dictation, Pause sensitivity, Home greeting. |
| Audio input | Mic processing. |
| Writing & language | Language, Speaking environment. |
| Storage | Keep history for, Reveal storage folder, Clear history…, Delete recordings…, Clean up now. |
| Privacy & learning | Smart Context, Use selected text, Use current app/field, Use app/window title, Use recent clipboard, Learn from corrections. |
| Permissions | Microphone, Accessibility, Live captions, Check again. |
| Models | Fast preview model, Final dictation model, Writing engine, Install, Repair, Model details. |
| Updates & app | Software updates, Check, Install…, Check automatically, Download updates automatically, Launch on login, Developer mode. |

## Current Internal Route Inventory

This is the current maintenance route surface from `juno_v2/runtime/uds_dispatch.py` plus the HTTP compatibility surface in `juno_v2/workbench/server.py`. It is kept here as an internal claim-audit aid only. Public Reference pages must not expose this route catalog.

### GET

- `/healthz`
- `/api/runtime`
- `/api/state`
- `/api/broker/engine/compatibility`
- `/api/broker/personalization/summary`
- `/api/broker/personalization/user_profile`
- `/api/broker/settings`
- `/api/broker/privacy/context_settings`
- `/api/broker/privacy/app_overrides`
- `/api/broker/writer/warm`
- `/api/broker/preview/warm`
- `/api/broker/model/routes`
- `/api/broker/stats/summary`
- `/api/broker/storage/stats`
- `/api/broker/runtime/backends`
- `/api/broker/surface/active`
- `/api/broker/surface/policy`
- `/api/broker/surface/capability`
- `/api/broker/modes/builtin`
- `/api/broker/modes/custom`
- `/api/broker/modes/current`
- `/api/broker/surface_presets/user`
- `/api/broker/surface_presets/merged`
- `/api/broker/transforms/builtin`
- `/api/broker/transforms/custom`
- `/api/broker/recovery/paste_last`
- `/api/broker/recovery/history`
- `/api/broker/recovery/replay`
- `/api/broker/memory/snapshot`
- `/api/broker/memory/vocab`
- `/api/broker/memory/replacement`
- `/api/broker/memory/snippet`
- `/api/broker/memory/correction`
- `/api/broker/setup/status`
- `/api/broker/export/data.zip`
- `/api/broker/audio/{utterance_id}/replay`
- `/api/broker/history`

### POST

- `/api/broker/dictation/ingest_wav`
- `/api/broker/session/start`
- `/api/broker/session/transform`
- `/api/broker/shell/home_greeting`
- `/api/broker/modes/manual/set`
- `/api/broker/modes/manual/clear`
- `/api/broker/dictation/replay_all_finals`
- `/api/broker/dictation/preview/chunk`
- `/api/broker/runtime/swap_final`
- `/api/broker/modes/custom/upsert`
- `/api/broker/modes/custom/delete`
- `/api/broker/modes/custom/activate`
- `/api/broker/surface_presets/upsert`
- `/api/broker/surface_presets/delete`
- `/api/broker/surface/editing_profile`
- `/api/broker/transforms/custom/upsert`
- `/api/broker/transforms/custom/delete`
- `/api/broker/insertion/committed`
- `/api/broker/learning/observe_correction`
- `/api/broker/personalization/user_profile`
- `/api/broker/settings/retention`
- `/api/broker/settings/language_environment`
- `/api/broker/settings/writer`
- `/api/broker/settings/itn`
- `/api/broker/settings/audio`
- `/api/broker/settings/live_caption`
- `/api/broker/privacy/context_settings`
- `/api/broker/privacy/app_overrides`
- `/api/broker/memory/vocab`
- `/api/broker/memory/vocab/remove`
- `/api/broker/memory/replacement`
- `/api/broker/memory/replacement/remove`
- `/api/broker/memory/snippet`
- `/api/broker/memory/snippet/remove`
- `/api/broker/memory/correction/remove`
- `/api/broker/writer/extract`
- `/api/broker/recovery/ingest`
- `/api/broker/recovery/retry_append`
- `/api/broker/recovery/audio/delete`
- `/api/broker/setup/install`
- `/api/broker/setup/repair`
- `/api/broker/storage/audio/prune_all`
- `/api/broker/history/clear_all`
- `/api/broker/history/cancel_draft`
- `/api/broker/history/reprocess`
- `/api/broker/history/{utterance_id}/actions`
- `/api/broker/retention/run_cleanup`

### DELETE

- `/api/broker/history/{utterance_id}`

## Route Guardrails

- Mutating routes require `X-Juno-Local-Token`.
- User-facing docs must not expose route names, helper process names, workbench terminology, or Python/venv details.
- There is no dedicated `/api/broker/notes/*`, `/api/broker/reminders/*`, or `/api/broker/alarms/*` route group in the current docs-backed API. Actions execute in the macOS shell and Apple frameworks; only action outcomes are attached to history.
