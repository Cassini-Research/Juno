/**
 * swap-media.mjs
 * Replace MediaSlot components with ScreenshotFrame, DiagramBlock, or VideoDemo
 * based on the asset mapping from MEDIA_PLACEHOLDER_MANIFEST.md.
 *
 * Run: node scripts/swap-media.mjs
 */

import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

const ROOT = new URL('..', import.meta.url).pathname;
const CONTENT = join(ROOT, 'content/docs');

// Each entry: { file, replacements: [ { old slot block, new block } ] }
// old block is matched as a regex on the full file content.

const SCREENSHOT_IMPORT = `import { ScreenshotFrame } from '@/components/docs/ScreenshotFrame';`;
const DIAGRAM_IMPORT = `import { DiagramBlock } from '@/components/docs/DiagramBlock';`;
const VIDEO_IMPORT = `import { VideoDemo } from '@/components/docs/VideoDemo';`;

const MEDIASLOT_IMPORT_RE = /import \{ MediaSlot \} from '@\/components\/docs\/MediaSlot';\n/;

/** Strip the MediaSlot import and the slot block, replace with new component. */
function swap(filePath, newImport, newBlock) {
  let src = readFileSync(filePath, 'utf8');

  // Remove MediaSlot import (only once, if present)
  src = src.replace(MEDIASLOT_IMPORT_RE, '');

  // Ensure the new import is present
  if (!src.includes(newImport)) {
    // Insert after the last existing import line
    const lastImportIdx = src.lastIndexOf('\nimport ');
    if (lastImportIdx !== -1) {
      const end = src.indexOf('\n', lastImportIdx + 1);
      src = src.slice(0, end + 1) + newImport + '\n' + src.slice(end + 1);
    } else {
      src = newImport + '\n' + src;
    }
  }

  return src;
}

/** Replace a <MediaSlot ... /> block (single or multiline) with new markup. */
function replaceSlot(src, assetPath, newBlock) {
  // Match <MediaSlot\n ... /> including the assetPath attribute
  const escaped = assetPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`<MediaSlot[^>]*${escaped}[\\s\\S]*?/>`, 'm');
  return src.replace(re, newBlock);
}

// ─── Mapping ─────────────────────────────────────────────────────────────────

const OPS = [
  // #10 clipboard-vs-pasteboard diagram
  {
    file: 'privacy-and-data/clipboard-and-pasteboard.mdx',
    importLine: DIAGRAM_IMPORT,
    assetPath: '/images/posters/clipboard-vs-pasteboard.png',
    newBlock: `<DiagramBlock
  title="Pasteboard insertion vs. clipboard context"
  caption="Two separate flows: Juno always writes to the pasteboard on insert; clipboard context is a separate, opt-in read."
  takeaway="Juno never reads your clipboard unless you explicitly enable recent-clipboard context in Settings."
>
  <img src="/images/posters/clipboard-vs-pasteboard.svg" alt="Pasteboard insertion vs clipboard context diagram" style={{ width: '100%', height: 'auto' }} />
</DiagramBlock>`,
  },

  // #11 storage cleanup controls (no matching panel — use settings-overview)
  {
    file: 'privacy-and-data/delete-export-and-cleanup.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/storage-cleanup-controls.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/settings-overview.png"
  alt="Juno Settings showing Keep recordings for control"
  caption="Settings — the Keep recordings for dropdown controls how long audio is retained. Use History → Delete or the per-entry Delete action to remove individual records."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #12 privacy-data-flow poster diagram
  {
    file: 'privacy-and-data/overview.mdx',
    importLine: DIAGRAM_IMPORT,
    assetPath: '/images/posters/privacy-data-flow.png',
    newBlock: `<DiagramBlock
  title="Privacy data flow"
  caption="How audio, context, and optional clipboard flow through Juno's local engine and what is stored."
  takeaway="Every path through Juno stays on this Mac. Secure fields block capture, context, learning, history, audio, and paste regardless of settings."
>
  <img src="/images/posters/privacy-data-flow-poster.svg" alt="Juno privacy data flow diagram" style={{ width: '100%', height: 'auto' }} />
</DiagramBlock>`,
  },

  // #14 smart-context-settings (no panel — use per-app writing controls)
  {
    file: 'privacy-and-data/smart-context.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/smart-context-settings.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/per-app-writing-controls.png"
  alt="Per-app writing showing privacy controls: Context, Learn corrections, Save history, Keep recordings"
  caption="Per-app writing — per-app privacy overrides let you restrict context, correction learning, history, or recordings for specific apps. Global defaults apply to all others."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #16 first-launch-permissions
  {
    file: 'start/first-launch.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/first-launch-permissions.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/first-launch-permissions.png"
  alt="Juno onboarding Access step showing Microphone REQUIRED, Accessibility REQUIRED, and Live captions OPTIONAL — all granted"
  caption="Access step — Microphone and Accessibility are required. Live captions is optional and uses on-device speech recognition."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #17 first-launch-ready
  {
    file: 'start/first-launch.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/first-launch-ready.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/first-launch-ready.png"
  alt="Juno onboarding Setup step showing Live captions, High-quality transcription, and Smart formatting all ready"
  caption="Setup step — all three engine capabilities confirmed ready. Models live on this Mac; nothing is sent to the cloud."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #18 settings-permissions
  {
    file: 'start/permissions.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/settings-permissions.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/settings-permissions.png"
  alt="Juno permissions screen showing Microphone REQUIRED, Accessibility REQUIRED, and Live captions OPTIONAL"
  caption="Required permissions: Microphone (to hear speech) and Accessibility (to place text). Live captions is optional."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #20 quickstart cursor before
  {
    file: 'start/quickstart.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/quickstart-cursor-before-dictation.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/quickstart-cursor-before-dictation.png"
  alt="Cursor placed in a blank Apple Notes field before starting Juno dictation"
  caption="Place your cursor in any writable field before pressing the shortcut — Juno will insert exactly there."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #21 quickstart inserted result
  {
    file: 'start/quickstart.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/quickstart-inserted-result.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/quickstart-inserted-result.png"
  alt="Juno HUD showing INSERTED state with the transcribed text placed in the target field"
  caption="After dictation completes, the HUD shows Inserted and the polished text appears at the cursor."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #23 voice-to-text-flow diagram
  {
    file: 'start/what-is-juno.mdx',
    importLine: DIAGRAM_IMPORT,
    assetPath: '/images/posters/juno-voice-to-text-flow.png',
    newBlock: `<DiagramBlock
  title="Voice to written text — how Juno works"
  caption="One shortcut press starts the flow; Juno handles the rest locally and lands text exactly where your cursor is."
  takeaway="If a field blocks insertion, Juno falls back to Copy Ready so your text is never lost."
>
  <img src="/images/posters/voice-to-text-flow.svg" alt="Juno voice to text flow: shortcut → HUD → local engine → insert or copy ready → history" style={{ width: '100%', height: 'auto' }} />
</DiagramBlock>`,
  },

  // #24 per-app-writing-controls
  {
    file: 'use-juno/per-app-writing.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/per-app-writing-controls.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/per-app-writing-controls.png"
  alt="Per-app writing settings showing writing style and privacy overrides"
  caption="Per-app writing — assign a default writing style per app, and override context, learning, history, and recording settings for that app independently."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #25 dictionary-memory
  {
    file: 'use-juno/dictionary-and-memory.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/dictionary-memory.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/dictionary-memory.png"
  alt="Dictionary and memory Vocabulary tab showing seed and user-added terms"
  caption="Vocabulary — add names, product terms, and abbreviations Juno should recognise reliably. Seed terms are promoted automatically from your history."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #26 history-view
  {
    file: 'use-juno/history.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/history-view.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/history-view.png"
  alt="Juno History view showing transcript list with search and filter controls, and a selected entry with Copy, Save phrase, and Delete actions"
  caption="History — search and filter every past dictation. Select an entry to copy, save as a phrase, or delete it."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #29 notes-view (Voice Actions page)
  {
    file: 'use-juno/notes.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/notes-view.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/notes-view.png"
  alt="Juno Actions page showing Notes, Reminders, Alarms, and Hey Juno examples"
  caption="Actions — use Hey Juno requests to save thoughts to a Juno folder in Apple Notes during dictation."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #30 recording-retention-setting
  {
    file: 'use-juno/recordings.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/recording-retention-setting.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/recording-retention-setting.png"
  alt="Juno Settings showing Keep recordings for set to 30 days"
  caption="Keep recordings for — controls how long audio is retained for replay and troubleshooting. Defaults to 30 days; set to Never to disable recording."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #33 settings-overview
  {
    file: 'use-juno/settings.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/settings-overview.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/settings-overview.png"
  alt="Juno Settings showing Quick Picks section with Appearance, Show in Dock, Pause sensitivity, Keep recordings for, Live transcriptions, and HUD position"
  caption="Settings — Quick Picks and Dictation sections cover appearance, the shortcut key, HUD behaviour, recording retention, and live transcriptions."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },

  // #35 hud-listening-target-app
  {
    file: 'use-juno/speak-and-insert.mdx',
    importLine: SCREENSHOT_IMPORT,
    assetPath: '/images/screenshots/hud-listening-target-app.png',
    newBlock: `<ScreenshotFrame
  src="/images/screenshots/hud-listening-target-app.png"
  alt="Juno HUD in Waiting for speech state floating over Apple Notes"
  caption="The HUD appears as a floating overlay above the target app. It shows the active state and the app Juno will insert into."
  surface="Juno 0.2"
  dateCaptured="2026-05-04"
/>`,
  },
];

// ─── Run ─────────────────────────────────────────────────────────────────────

let changed = 0;
for (const op of OPS) {
  const filePath = join(CONTENT, op.file);
  let src;
  try {
    src = readFileSync(filePath, 'utf8');
  } catch {
    console.warn(`SKIP (not found): ${op.file}`);
    continue;
  }

  // Remove MediaSlot import, add new import
  src = src.replace(MEDIASLOT_IMPORT_RE, '');
  if (!src.includes(op.importLine)) {
    const lastImportEnd = (() => {
      const lines = src.split('\n');
      let last = -1;
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('import ')) last = i;
      }
      if (last === -1) return 0;
      return lines.slice(0, last + 1).join('\n').length + 1;
    })();
    src = src.slice(0, lastImportEnd) + op.importLine + '\n' + src.slice(lastImportEnd);
  }

  // Replace the MediaSlot block
  src = replaceSlot(src, op.assetPath, op.newBlock);

  writeFileSync(filePath, src, 'utf8');
  console.log(`✓ ${op.file}`);
  changed++;
}

console.log(`\nDone. ${changed} files updated.`);
