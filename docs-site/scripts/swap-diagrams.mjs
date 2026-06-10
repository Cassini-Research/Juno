#!/usr/bin/env node
// Replace single architecture <MediaSlot type="diagram" .../> with a <DiagramBlock> + <img>.
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ARCH = join(here, '..', 'content', 'docs', 'architecture');

const TAKEAWAYS = {
  'architecture-runtime-overview.svg':
    'The native app is the user-facing controller; the local voice engine is the private processing layer.',
  'architecture-dictation-lifecycle.svg':
    'Copy Ready is a first-class recovery path, not an error afterthought.',
  'architecture-failure-recovery.svg':
    'When text exists but insertion is unsafe, Juno hands you Copy Ready instead of guessing at the destination.',
  'architecture-local-voice-engine.svg':
    'The shell controls the user session; the engine controls private processing and durable local state.',
  'architecture-macos-helpers.svg':
    'Each helper binary does one OS-level job; keeping them narrow makes signing and review tractable.',
  'architecture-memory-personalization.svg':
    'Personalization is only applied when privacy policy permits it; secure surfaces suppress it entirely.',
  'architecture-privacy-data-flow.svg':
    'Secure-field policy is a gate before capture, context, learning, history, audio save, and paste.',
  'architecture-storage-lifecycle.svg':
    'Durable product data and short-lived runtime artifacts live in separate paths with separate retention.',
};

const SOURCE_FILES = {
  'architecture-runtime-overview.svg': [
    'harpy_v2/runtime/paths.py',
    'harpy_v2/workbench/server.py',
    'shells/macos/Sources/JunoShell/JunoShellApp.swift',
  ],
  'architecture-dictation-lifecycle.svg': [
    'shells/macos/Sources/JunoShell/JunoShellApp.swift',
    'harpy_v2/workbench/server.py',
  ],
  'architecture-failure-recovery.svg': [
    'shells/macos/Sources/JunoShell/JunoShellApp.swift',
    'shells/macos/Sources/JunoShell/JunoLocalCapability.swift',
  ],
  'architecture-local-voice-engine.svg': [
    'harpy_v2/workbench/server.py',
    'harpy_v2/runtime/paths.py',
  ],
  'architecture-macos-helpers.svg': [
    'PRODUCTION_SHIPPING_PLAN.md',
    'shells/macos/Sources/JunoShell/',
  ],
  'architecture-memory-personalization.svg': [
    'harpy_v2/workbench/server.py',
    'shells/macos/Sources/JunoShell/MemoryManagementView.swift',
  ],
  'architecture-privacy-data-flow.svg': [
    'CURRENT_PRODUCT_TRUTH.md',
    'harpy_v2/workbench/server.py',
    'shells/macos/Sources/JunoShell/JunoLocalCapability.swift',
  ],
  'architecture-storage-lifecycle.svg': [
    'harpy_v2/runtime/paths.py',
    'harpy_v2/runtime/local_broker_token.py',
  ],
};

const PAGES = [
  'overview.mdx',
  'dictation-lifecycle.mdx',
  'failure-and-recovery-model.mdx',
  'local-voice-engine.mdx',
  'macos-app-and-helpers.mdx',
  'memory-and-personalization.mdx',
  'privacy-data-flow.mdx',
  'storage-lifecycle.mdx',
];

const slotRe =
  /<MediaSlot\s+type="diagram"[\s\S]*?asset="\/images\/diagrams\/([^"]+)"[\s\S]*?title="([^"]+)"[\s\S]*?(?:caption="([^"]*)"[\s\S]*?)?\/>/;

// Loose: title may appear before asset; capture both with separate regex
function extract(slot) {
  const asset = slot.match(/asset="\/images\/diagrams\/([^"]+)"/)?.[1];
  const title = slot.match(/title="([^"]+)"/)?.[1];
  const caption = slot.match(/caption="([^"]*)"/)?.[1] ?? '';
  return { asset, title, caption };
}

for (const page of PAGES) {
  const path = join(ARCH, page);
  let src = readFileSync(path, 'utf8');

  // Find <MediaSlot ... type="diagram" ... />
  const m = src.match(/<MediaSlot[\s\S]*?type="diagram"[\s\S]*?\/>/);
  if (!m) {
    console.log(`skip ${page}: no diagram slot`);
    continue;
  }
  const slot = m[0];
  const { asset, title, caption } = extract(slot);
  if (!asset || !title) {
    console.log(`skip ${page}: could not parse slot`);
    continue;
  }
  const takeaway = TAKEAWAYS[asset] ?? 'See architecture page for context.';
  const sources = SOURCE_FILES[asset] ?? [];
  const srcArr = sources.length
    ? `\n  sourceFiles={${JSON.stringify(sources)}}`
    : '';
  const replacement =
    `<DiagramBlock\n  title=${JSON.stringify(title)}\n  caption=${JSON.stringify(
      caption,
    )}\n  takeaway=${JSON.stringify(takeaway)}${srcArr}\n>\n  <img src="/images/diagrams/${asset}" alt=${JSON.stringify(
      title,
    )} style={{ width: '100%', height: 'auto' }} />\n</DiagramBlock>`;

  src = src.replace(slot, replacement);

  // Fix imports: ensure DiagramBlock imported, drop MediaSlot if unused.
  const stillUsesMediaSlot = src.includes('<MediaSlot');
  // Remove existing MediaSlot import line.
  src = src.replace(
    /import\s*\{\s*MediaSlot\s*\}\s*from\s*'@\/components\/docs\/MediaSlot';\s*\n?/,
    '',
  );
  // Remove existing DiagramBlock import to avoid duplicates, then re-add.
  src = src.replace(
    /import\s*\{\s*DiagramBlock\s*\}\s*from\s*'@\/components\/docs\/DiagramBlock';\s*\n?/,
    '',
  );
  // Insert imports after frontmatter (after the second --- line).
  const fmEnd = src.indexOf('\n---\n');
  if (fmEnd === -1) {
    console.log(`skip ${page}: no frontmatter`);
    continue;
  }
  const insertAt = fmEnd + 5; // after \n---\n
  let importBlock = `\nimport { DiagramBlock } from '@/components/docs/DiagramBlock';\n`;
  if (stillUsesMediaSlot) {
    importBlock += `import { MediaSlot } from '@/components/docs/MediaSlot';\n`;
  }
  src = src.slice(0, insertAt) + importBlock + src.slice(insertAt);

  writeFileSync(path, src);
  console.log(`updated ${page} -> ${asset}`);
}
