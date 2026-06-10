import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const OUT = 'public';

const colors = {
  paper: '#f7f8fa',
  paperWarm: '#eef2f6',
  ink: '#101828',
  inkSoft: '#344054',
  muted: '#667085',
  hair: '#d0d5dd',
  navy: '#101828',
  navy2: '#1f2937',
  amber: '#b54708',
  amberSoft: '#fff4ed',
  meadow: '#087443',
  meadowSoft: '#ecfdf3',
  card: '#ffffff',
  block: '#f2f4f7',
  danger: '#b42318',
  dangerSoft: '#fef3f2',
};

function esc(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function wrap(text, max = 24) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = '';
  for (const word of words) {
    if (!line) {
      line = word;
    } else if ((line + ' ' + word).length <= max) {
      line += ' ' + word;
    } else {
      lines.push(line);
      line = word;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function textLines(text, x, y, opts = {}) {
  const {
    max = 24,
    size = 18,
    weight = 700,
    color = colors.ink,
    anchor = 'middle',
    lineHeight = 1.2,
    className = '',
  } = opts;
  const lines = wrap(text, max);
  const dy = size * lineHeight;
  const start = y - ((lines.length - 1) * dy) / 2;
  return `<text class="${className}" x="${x}" y="${start}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}" fill="${color}">${lines
    .map((line, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : dy}">${esc(line)}</tspan>`)
    .join('')}</text>`;
}

function eyebrow(text, x, y) {
  return `<text x="${x}" y="${y}" font-family="JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12" font-weight="700" letter-spacing="2.2" fill="${colors.muted}">${esc(text.toUpperCase())}</text>`;
}

function node(n) {
  const {
    x,
    y,
    w = 178,
    h = 86,
    title,
    note,
    kind = 'default',
  } = n;
  const palette = {
    default: [colors.card, colors.hair, colors.ink],
    primary: [colors.navy, colors.navy, '#f8f2e7'],
    amber: [colors.amberSoft, '#e7c7a4', '#704316'],
    meadow: [colors.meadowSoft, '#b8d8ca', '#174d3c'],
    muted: [colors.block, colors.hair, colors.inkSoft],
    danger: [colors.dangerSoft, '#e0b4a8', '#763625'],
  }[kind];
  const cx = x + w / 2;
  const titleY = note ? y + h * 0.38 : y + h / 2 + 1;
  return `<g>
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${palette[0]}" stroke="${palette[1]}" stroke-width="1.2"/>
    ${textLines(title, cx, titleY, { max: Math.max(14, Math.floor(w / 8.4)), size: 22, weight: 780, color: palette[2] })}
    ${note ? textLines(note, cx, y + h * 0.72, { max: Math.max(18, Math.floor(w / 7.4)), size: 14, weight: 580, color: kind === 'primary' ? 'rgba(248,242,231,.78)' : colors.muted }) : ''}
  </g>`;
}

function pointFor(n, side) {
  if (side === 'left') return [n.x, n.y + n.h / 2];
  if (side === 'right') return [n.x + n.w, n.y + n.h / 2];
  if (side === 'top') return [n.x + n.w / 2, n.y];
  return [n.x + n.w / 2, n.y + n.h];
}

function edge(nodes, e) {
  const from = nodes[e.from];
  const to = nodes[e.to];
  const [x1, y1] = pointFor(from, e.fromSide ?? 'right');
  const [x2, y2] = pointFor(to, e.toSide ?? 'left');
  const mid = e.mid ?? Math.round((x1 + x2) / 2);
  const d = e.path ?? `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
  const stroke = e.kind === 'danger' ? colors.danger : e.kind === 'muted' ? colors.muted : colors.navy2;
  const label = e.label
    ? `<text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 10}" text-anchor="middle" font-size="14" font-weight="650" fill="${colors.muted}">${esc(e.label)}</text>`
    : '';
  return `<g>
    <path d="${d}" fill="none" stroke="${stroke}" stroke-width="2" stroke-linecap="round" marker-end="url(#arrow)"/>
    ${label}
  </g>`;
}

function badge(text, x, y, fill = colors.navy, color = '#f8f2e7') {
  const w = Math.max(94, text.length * 7.8 + 24);
  return `<g>
    <rect x="${x}" y="${y}" width="${w}" height="30" rx="15" fill="${fill}"/>
    <text x="${x + w / 2}" y="${y + 20}" text-anchor="middle" font-size="12" font-weight="750" fill="${color}">${esc(text)}</text>
  </g>`;
}

function render({ title, caption, width = 1280, height = 620, nodes: nodeList, edges = [], badges = [], footer }) {
  const byId = Object.fromEntries(nodeList.map((n) => [n.id, n]));
  const bg = `<rect width="${width}" height="${height}" rx="0" fill="${colors.paper}"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}">
  <defs>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#120e09" flood-opacity=".10"/>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="${colors.navy2}"/>
    </marker>
  </defs>
  <style>
    text { font-family: "Inter Tight", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  </style>
  ${bg}
  <rect x="22" y="22" width="${width - 44}" height="${height - 44}" rx="14" fill="${colors.card}" stroke="${colors.hair}" filter="url(#softShadow)"/>
  <g>
    ${edges.map((e) => edge(byId, e)).join('\n')}
    ${nodeList.map(node).join('\n')}
    ${badges.map((b) => badge(...b)).join('\n')}
  </g>
  ${footer ? textLines(footer, width / 2, height - 48, { max: 88, size: 16, weight: 640, color: colors.muted }) : ''}
</svg>`;
}

function write(relativePath, svg) {
  const path = join(OUT, relativePath);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, svg);
  console.log(path);
}

function simple(relativePath, diagram) {
  write(relativePath, render(diagram));
}

simple('images/diagrams/architecture-runtime-overview.svg', {
  title: 'Runtime architecture',
  caption: 'Juno keeps the product surface in the Mac app and private processing on this Mac.',
  nodes: [
    { id: 'user', x: 82, y: 278, w: 132, h: 74, title: 'You', note: 'start dictation', kind: 'amber' },
    { id: 'app', x: 276, y: 250, w: 184, h: 118, title: 'Juno app', note: 'shortcut, HUD, permissions, safe insertion', kind: 'primary' },
    { id: 'engine', x: 552, y: 250, w: 210, h: 118, title: 'Local voice engine', note: 'speech, writing, memory', kind: 'default' },
    { id: 'insert', x: 852, y: 202, w: 190, h: 86, title: 'Safe insert', note: 'text lands at cursor', kind: 'meadow' },
    { id: 'copy', x: 852, y: 330, w: 190, h: 86, title: 'Copy Ready', note: 'blocked fields stay safe', kind: 'amber' },
    { id: 'data', x: 1092, y: 250, w: 136, h: 118, title: 'Local data', note: 'history, settings, retention', kind: 'muted' },
  ],
  edges: [
    { from: 'user', to: 'app' },
    { from: 'app', to: 'engine' },
    { from: 'engine', to: 'insert' },
    { from: 'engine', to: 'copy' },
    { from: 'insert', to: 'data' },
    { from: 'copy', to: 'data' },
  ],
  footer: 'The user sees Juno. The engine stays local and private. Unsafe destinations use Copy Ready instead of a forced paste.',
});

simple('images/diagrams/architecture-dictation-lifecycle.svg', {
  title: 'Dictation lifecycle',
  caption: 'A single session from shortcut press to inserted text or Copy Ready fallback.',
  nodes: [
    { id: 'shortcut', x: 80, y: 250, w: 150, h: 94, title: 'Press shortcut', note: 'Fn / Globe by default', kind: 'amber' },
    { id: 'check', x: 280, y: 250, w: 172, h: 94, title: 'Juno checks', note: 'permissions and focused field', kind: 'default' },
    { id: 'listen', x: 502, y: 250, w: 172, h: 94, title: 'Speak', note: 'HUD shows truthful state', kind: 'primary' },
    { id: 'write', x: 724, y: 250, w: 172, h: 94, title: 'Juno writes', note: 'local transcription and cleanup', kind: 'default' },
    { id: 'insert', x: 946, y: 194, w: 172, h: 88, title: 'Inserted', note: 'safe target', kind: 'meadow' },
    { id: 'copy', x: 946, y: 322, w: 172, h: 88, title: 'Copy Ready', note: 'blocked target', kind: 'amber' },
  ],
  edges: [
    { from: 'shortcut', to: 'check' },
    { from: 'check', to: 'listen' },
    { from: 'listen', to: 'write' },
    { from: 'write', to: 'insert' },
    { from: 'write', to: 'copy' },
  ],
  badges: [['local-first', 724, 180, colors.navy], ['safe fallback', 946, 424, colors.amber]],
  footer: 'Juno does not show Listening until the microphone is really active, and it never treats Copy Ready as a failure.',
});

simple('images/diagrams/architecture-failure-recovery.svg', {
  title: 'Failure and recovery',
  caption: 'Every blocked state should tell the user what happened and what to do next.',
  nodes: [
    { id: 'start', x: 84, y: 246, w: 150, h: 92, title: 'Start dictation', note: 'shortcut pressed', kind: 'amber' },
    { id: 'permission', x: 306, y: 148, w: 190, h: 86, title: 'Permission missing', note: 'open Settings', kind: 'danger' },
    { id: 'mic', x: 306, y: 258, w: 190, h: 86, title: 'No microphone', note: 'show mic guidance', kind: 'danger' },
    { id: 'secure', x: 306, y: 368, w: 190, h: 86, title: 'Secure field', note: 'block capture and paste', kind: 'danger' },
    { id: 'engine', x: 578, y: 204, w: 190, h: 96, title: 'Engine not ready', note: 'install or repair', kind: 'amber' },
    { id: 'text', x: 578, y: 334, w: 190, h: 96, title: 'Text exists', note: 'protect the result', kind: 'default' },
    { id: 'copy', x: 854, y: 334, w: 190, h: 96, title: 'Copy Ready', note: 'manual placement', kind: 'meadow' },
    { id: 'support', x: 854, y: 204, w: 190, h: 96, title: 'Support path', note: 'diagnostics if repair fails', kind: 'muted' },
  ],
  edges: [
    { from: 'start', to: 'permission' },
    { from: 'start', to: 'mic' },
    { from: 'start', to: 'secure' },
    { from: 'mic', to: 'engine' },
    { from: 'secure', to: 'text' },
    { from: 'engine', to: 'support' },
    { from: 'text', to: 'copy' },
  ],
  footer: 'No paste is safer than wrong paste. When Juno has text but cannot trust the field, Copy Ready is the right outcome.',
});

simple('images/diagrams/architecture-local-voice-engine.svg', {
  title: 'Local voice engine',
  caption: 'Private processing turns audio and safe context into useful writing on this Mac.',
  nodes: [
    { id: 'audio', x: 80, y: 246, w: 156, h: 92, title: 'Audio input', note: 'current session only', kind: 'amber' },
    { id: 'preview', x: 292, y: 178, w: 174, h: 84, title: 'Live preview', note: 'fast feedback', kind: 'default' },
    { id: 'final', x: 292, y: 318, w: 174, h: 84, title: 'Final transcript', note: 'higher quality pass', kind: 'default' },
    { id: 'writing', x: 540, y: 246, w: 190, h: 92, title: 'Writing cleanup', note: 'punctuation, format, commands', kind: 'primary' },
    { id: 'memory', x: 804, y: 178, w: 178, h: 84, title: 'Memory', note: 'allowed terms and corrections', kind: 'meadow' },
    { id: 'history', x: 804, y: 318, w: 178, h: 84, title: 'History', note: 'local retention rules', kind: 'muted' },
    { id: 'result', x: 1052, y: 246, w: 150, h: 92, title: 'Text result', note: 'back to Juno app', kind: 'amber' },
  ],
  edges: [
    { from: 'audio', to: 'preview' },
    { from: 'audio', to: 'final' },
    { from: 'preview', to: 'writing' },
    { from: 'final', to: 'writing' },
    { from: 'memory', to: 'writing', fromSide: 'left', toSide: 'right' },
    { from: 'writing', to: 'history' },
    { from: 'writing', to: 'result' },
  ],
  footer: 'The engine is not a separate app for users. It is the local processing layer behind Juno.',
});

simple('images/diagrams/architecture-macos-helpers.svg', {
  title: 'Mac app and system permissions',
  caption: 'Juno uses narrow Mac capabilities to listen, understand the focused field, and place text safely.',
  nodes: [
    { id: 'app', x: 536, y: 240, w: 208, h: 112, title: 'Juno app', note: 'the product surface users control', kind: 'primary' },
    { id: 'shortcut', x: 104, y: 138, w: 190, h: 84, title: 'Shortcut listener', note: 'start and stop dictation', kind: 'default' },
    { id: 'field', x: 104, y: 360, w: 190, h: 84, title: 'Focused field check', note: 'safe or blocked target', kind: 'default' },
    { id: 'paste', x: 986, y: 138, w: 190, h: 84, title: 'Insertion control', note: 'paste only when safe', kind: 'meadow' },
    { id: 'observer', x: 986, y: 360, w: 190, h: 84, title: 'Correction observer', note: 'learns only when allowed', kind: 'muted' },
    { id: 'permissions', x: 536, y: 408, w: 208, h: 84, title: 'macOS permissions', note: 'Microphone and Accessibility', kind: 'amber' },
  ],
  edges: [
    { from: 'shortcut', to: 'app' },
    { from: 'field', to: 'app' },
    { from: 'app', to: 'paste' },
    { from: 'app', to: 'observer' },
    { from: 'app', to: 'permissions', fromSide: 'bottom', toSide: 'top' },
  ],
  footer: 'Users should understand the capability, not the binary behind it.',
});

simple('images/diagrams/architecture-memory-personalization.svg', {
  title: 'Memory and personalization',
  caption: 'Juno improves repeated words and style only when the current surface allows learning.',
  nodes: [
    { id: 'context', x: 90, y: 250, w: 168, h: 92, title: 'Allowed context', note: 'selected text, app, field', kind: 'amber' },
    { id: 'gate', x: 324, y: 250, w: 170, h: 92, title: 'Privacy gate', note: 'secure fields stop learning', kind: 'primary' },
    { id: 'dictionary', x: 562, y: 146, w: 174, h: 82, title: 'Dictionary', note: 'names and terms', kind: 'meadow' },
    { id: 'snippets', x: 562, y: 250, w: 174, h: 82, title: 'Snippets', note: 'phrases you reuse', kind: 'meadow' },
    { id: 'corrections', x: 562, y: 354, w: 174, h: 82, title: 'Corrections', note: 'edits Juno may learn', kind: 'meadow' },
    { id: 'bias', x: 818, y: 250, w: 184, h: 92, title: 'Better output', note: 'spelling, style, formatting', kind: 'default' },
    { id: 'delete', x: 1068, y: 250, w: 138, h: 92, title: 'User control', note: 'edit or delete', kind: 'muted' },
  ],
  edges: [
    { from: 'context', to: 'gate' },
    { from: 'gate', to: 'dictionary' },
    { from: 'gate', to: 'snippets' },
    { from: 'gate', to: 'corrections' },
    { from: 'dictionary', to: 'bias' },
    { from: 'snippets', to: 'bias' },
    { from: 'corrections', to: 'bias' },
    { from: 'bias', to: 'delete' },
  ],
  footer: 'Memory should be useful, inspectable, and removable. Secure or suppressed surfaces do not learn.',
});

simple('images/diagrams/architecture-privacy-data-flow.svg', {
  title: 'Privacy data flow',
  caption: 'Each sensitive behavior has its own switch or safety gate.',
  nodes: [
    { id: 'secure', x: 88, y: 114, w: 190, h: 88, title: 'Secure field gate', note: 'blocks capture and paste', kind: 'danger' },
    { id: 'audio', x: 92, y: 300, w: 170, h: 82, title: 'Voice audio', note: 'during dictation only', kind: 'amber' },
    { id: 'context', x: 318, y: 300, w: 170, h: 82, title: 'Safe context', note: 'selected text and field', kind: 'default' },
    { id: 'clipboard', x: 544, y: 300, w: 170, h: 82, title: 'Clipboard context', note: 'off by default', kind: 'muted' },
    { id: 'engine', x: 794, y: 250, w: 188, h: 112, title: 'Local processing', note: 'speech and writing on this Mac', kind: 'primary' },
    { id: 'output', x: 1044, y: 176, w: 158, h: 82, title: 'Inserted or Copy Ready', note: 'user-visible output', kind: 'meadow' },
    { id: 'retention', x: 1044, y: 332, w: 158, h: 82, title: 'Local retention', note: 'history and recordings', kind: 'amber' },
  ],
  edges: [
    { from: 'secure', to: 'audio', fromSide: 'bottom', toSide: 'top', kind: 'danger' },
    { from: 'audio', to: 'engine' },
    { from: 'context', to: 'engine' },
    { from: 'clipboard', to: 'engine' },
    { from: 'engine', to: 'output' },
    { from: 'engine', to: 'retention' },
  ],
  footer: 'Secure fields force context, learning, history, audio save, and paste off.',
});

simple('images/diagrams/architecture-storage-lifecycle.svg', {
  title: 'Storage lifecycle',
  caption: 'Juno keeps user data local and applies separate retention windows.',
  nodes: [
    { id: 'session', x: 96, y: 250, w: 164, h: 92, title: 'Dictation session', note: 'audio and final text', kind: 'amber' },
    { id: 'history', x: 336, y: 150, w: 180, h: 86, title: 'Transcript history', note: '90 days by default', kind: 'default' },
    { id: 'audio', x: 336, y: 286, w: 180, h: 86, title: 'Recordings', note: '30 days by default', kind: 'default' },
    { id: 'memory', x: 336, y: 422, w: 180, h: 86, title: 'Memory', note: 'until edited or deleted', kind: 'meadow' },
    { id: 'local', x: 604, y: 250, w: 198, h: 104, title: 'Juno support area', note: 'local files on this Mac', kind: 'primary' },
    { id: 'cleanup', x: 884, y: 196, w: 182, h: 86, title: 'Cleanup', note: 'retention and delete actions', kind: 'amber' },
    { id: 'export', x: 884, y: 330, w: 182, h: 86, title: 'Export or support', note: 'user-controlled sharing', kind: 'muted' },
  ],
  edges: [
    { from: 'session', to: 'history' },
    { from: 'session', to: 'audio' },
    { from: 'session', to: 'memory' },
    { from: 'history', to: 'local' },
    { from: 'audio', to: 'local' },
    { from: 'memory', to: 'local' },
    { from: 'local', to: 'cleanup' },
    { from: 'local', to: 'export' },
  ],
  footer: 'Deleting a history entry also removes its linked recording when one exists.',
});

simple('images/posters/voice-to-text-flow.svg', {
  title: 'Voice to written text',
  caption: 'The first path every new Juno user should understand.',
  nodes: [
    { id: 'shortcut', x: 92, y: 248, w: 158, h: 92, title: 'Press shortcut', note: 'Fn / Globe', kind: 'amber' },
    { id: 'hud', x: 310, y: 248, w: 158, h: 92, title: 'HUD opens', note: 'waiting, listening, polishing', kind: 'primary' },
    { id: 'speak', x: 528, y: 248, w: 158, h: 92, title: 'Speak naturally', note: 'say the thought', kind: 'default' },
    { id: 'juno', x: 746, y: 248, w: 158, h: 92, title: 'Juno writes', note: 'local engine', kind: 'default' },
    { id: 'insert', x: 964, y: 190, w: 170, h: 84, title: 'Inserted', note: 'safe field', kind: 'meadow' },
    { id: 'copy', x: 964, y: 318, w: 170, h: 84, title: 'Copy Ready', note: 'blocked field', kind: 'amber' },
  ],
  edges: [
    { from: 'shortcut', to: 'hud' },
    { from: 'hud', to: 'speak' },
    { from: 'speak', to: 'juno' },
    { from: 'juno', to: 'insert' },
    { from: 'juno', to: 'copy' },
  ],
  footer: 'Start in the app you are writing in. Juno places text there when it is safe.',
});

simple('images/posters/privacy-data-flow-poster.svg', {
  title: 'Privacy basics',
  caption: 'What Juno can use, what stays local, and what gets blocked.',
  nodes: [
    { id: 'settings', x: 86, y: 226, w: 190, h: 104, title: 'Your settings', note: 'context, learning, history, recordings', kind: 'amber' },
    { id: 'secure', x: 86, y: 370, w: 190, h: 86, title: 'Secure fields', note: 'always block sensitive paths', kind: 'danger' },
    { id: 'engine', x: 390, y: 270, w: 210, h: 112, title: 'Local processing', note: 'voice and writing on this Mac', kind: 'primary' },
    { id: 'output', x: 718, y: 190, w: 178, h: 86, title: 'Text output', note: 'inserted or Copy Ready', kind: 'meadow' },
    { id: 'history', x: 718, y: 326, w: 178, h: 86, title: 'Local records', note: 'history and recordings', kind: 'default' },
    { id: 'control', x: 1004, y: 270, w: 158, h: 112, title: 'User control', note: 'review, delete, export', kind: 'muted' },
  ],
  edges: [
    { from: 'settings', to: 'engine' },
    { from: 'secure', to: 'engine', kind: 'danger' },
    { from: 'engine', to: 'output' },
    { from: 'engine', to: 'history' },
    { from: 'output', to: 'control' },
    { from: 'history', to: 'control' },
  ],
  footer: 'Juno does not use your audio, transcripts, or corrections to train AI models.',
});

simple('images/posters/clipboard-vs-pasteboard.svg', {
  title: 'Pasteboard vs clipboard context',
  caption: 'One places Juno output. The other is optional context Juno may read.',
  nodes: [
    { id: 'output', x: 110, y: 188, w: 190, h: 92, title: 'Juno output', note: 'the text Juno just wrote', kind: 'meadow' },
    { id: 'pasteboard', x: 388, y: 188, w: 202, h: 92, title: 'Pasteboard insertion', note: 'used to place text', kind: 'primary' },
    { id: 'field', x: 678, y: 188, w: 190, h: 92, title: 'Focused field', note: 'safe destination', kind: 'default' },
    { id: 'clip', x: 110, y: 362, w: 190, h: 92, title: 'Recent clipboard', note: 'anything you copied', kind: 'amber' },
    { id: 'setting', x: 388, y: 362, w: 202, h: 92, title: 'Clipboard context', note: 'off by default', kind: 'muted' },
    { id: 'engine', x: 678, y: 362, w: 190, h: 92, title: 'Writing context', note: 'only if enabled', kind: 'default' },
  ],
  edges: [
    { from: 'output', to: 'pasteboard' },
    { from: 'pasteboard', to: 'field' },
    { from: 'clip', to: 'setting' },
    { from: 'setting', to: 'engine' },
  ],
  badges: [['always separate', 930, 270, colors.navy], ['opt-in read', 930, 354, colors.amber]],
  footer: 'Juno can use the pasteboard to insert output without reading your recent clipboard as context.',
});
