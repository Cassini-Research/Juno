import { spawnSync } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import sharp from 'sharp';

const W = 1280;
const H = 720;
const screenshotW = 1964;
const screenshotH = 1200;
const outScreens = 'public/images/screenshots';
const outVideos = 'public/videos/demos';
const tmpRoot = '/private/tmp/juno-docs-media';

const colors = {
  ink: '#09080e',
  paper: '#f4f1ea',
  card: '#faf7f1',
  elevated: '#f6f2eb',
  border: 'rgba(9,8,14,0.10)',
  muted: '#5a566a',
  dim: '#9c97a8',
  navy: '#0d1a2e',
  accent: '#6a8aff',
  accentDim: '#e2e7ff',
  meadow: '#348a6e',
  danger: '#ff3b30',
  amber: '#d9852b',
};

const font = "'SF Pro Display', 'Helvetica Neue', Arial, sans-serif";
const mono = "'SF Mono', 'Menlo', monospace";

function esc(s) {
  return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function text(x, y, value, size = 24, fill = colors.ink, weight = 600, family = font, anchor = 'start') {
  return `<text x="${x}" y="${y}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${esc(value)}</text>`;
}

function wrapText(x, y, value, width, size = 24, fill = colors.ink, weight = 500, line = 1.28) {
  const words = value.split(/\s+/);
  const max = Math.max(16, Math.floor(width / (size * 0.52)));
  const lines = [];
  let current = '';
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > max && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  }
  if (current) lines.push(current);
  return lines.map((lineText, i) => text(x, y + i * size * line, lineText, size, fill, weight)).join('');
}

function rect(x, y, w, h, r = 20, fill = colors.card, stroke = colors.border, sw = 1, extra = '') {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" ${extra}/>`;
}

function gradientDefs() {
  return `
  <defs>
    <linearGradient id="paperGlow" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#fffaf0"/>
      <stop offset="0.55" stop-color="${colors.paper}"/>
      <stop offset="1" stop-color="#e8edff"/>
    </linearGradient>
    <linearGradient id="navyGlow" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#15284b"/>
      <stop offset="1" stop-color="${colors.navy}"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#09080e" flood-opacity="0.16"/>
    </filter>
    <filter id="hudShadow" x="-20%" y="-20%" width="140%" height="160%">
      <feDropShadow dx="0" dy="16" stdDeviation="18" flood-color="#09080e" flood-opacity="0.28"/>
    </filter>
  </defs>`;
}

function bars(x, y, scale = 1, fill = '#ffffff', phase = 0) {
  const heights = [16, 26, 38, 29, 18];
  return heights.map((h, i) => {
    const pulse = 0.72 + Math.abs(Math.sin(phase * 0.12 + i * 0.9)) * 0.34;
    const hh = h * pulse * scale;
    return `<rect x="${x + i * 14 * scale}" y="${y - hh / 2}" width="${6 * scale}" height="${hh}" rx="${3 * scale}" fill="${fill}" opacity="${0.55 + i * 0.08}"/>`;
  }).join('');
}

function chromeDots(x, y) {
  return `<circle cx="${x}" cy="${y}" r="8" fill="#ff5f57"/><circle cx="${x + 26}" cy="${y}" r="8" fill="#febc2e"/><circle cx="${x + 52}" cy="${y}" r="8" fill="#28c840"/>`;
}

function noteWindow(x, y, w, h, title = 'Notes') {
  return `
    ${rect(x, y, w, h, 28, '#fffdf8', 'rgba(9,8,14,0.08)', 1, 'filter="url(#shadow)"')}
    ${chromeDots(x + 30, y + 30)}
    ${text(x + 36, y + 88, title, 28, colors.ink, 700)}
    ${rect(x + 36, y + 116, w - 72, h - 154, 20, '#fbf8f1', 'rgba(9,8,14,0.07)')}
  `;
}

function junoWindow(x, y, w, h, section = 'Home') {
  const items = ['Home', 'History', 'Actions', 'Voice Commands', 'Styles', 'Dictionary & Memory', 'Per-app writing', 'Privacy', 'Settings'];
  return `
    ${rect(x, y, w, h, 28, '#fffaf2', 'rgba(9,8,14,0.08)', 1, 'filter="url(#shadow)"')}
    ${chromeDots(x + 30, y + 30)}
    ${rect(x + 22, y + 62, 205, h - 84, 22, '#f2ede3', 'rgba(9,8,14,0.06)')}
    ${text(x + 54, y + 112, 'Juno', 30, colors.ink, 800)}
    ${items.map((item, i) => `
      ${rect(x + 40, y + 146 + i * 38, 158, 30, 12, item === section ? '#ffffff' : 'transparent', 'transparent')}
      ${text(x + 56, y + 166 + i * 38, item, item.length > 14 ? 11 : 14, item === section ? colors.ink : colors.muted, item === section ? 700 : 500)}
    `).join('')}
    ${text(x + 264, y + 112, section, 34, colors.ink, 800)}
  `;
}

function hud(x, y, state, body, opts = {}) {
  const tone = opts.tone ?? 'normal';
  const fill = tone === 'blocked' ? '#241520' : 'url(#navyGlow)';
  const accent = tone === 'blocked' ? colors.danger : tone === 'copy' ? colors.amber : colors.accent;
  const h = body ? 154 : 66;
  const w = opts.width ?? 560;
  return `
    ${rect(x, y, w, h, 32, fill, 'rgba(255,255,255,0.08)', 1, 'filter="url(#hudShadow)"')}
    <circle cx="${x + 42}" cy="${y + 34}" r="22" fill="rgba(255,255,255,0.10)"/>
    ${bars(x + 32, y + 34, 0.58, '#ffffff', opts.phase ?? 1)}
    ${text(x + 82, y + 40, state, 18, '#f4f1ea', 800)}
    <circle cx="${x + w - 44}" cy="${y + 34}" r="7" fill="${accent}"/>
    ${body ? wrapText(x + 34, y + 86, body, w - 70, 24, '#f4f1ea', 650) : ''}
  `;
}

function captionStrip(label, step) {
  return `
    ${rect(48, 48, 1184, 76, 24, 'rgba(255,255,255,0.70)', 'rgba(9,8,14,0.07)')}
    ${text(82, 96, label, 28, colors.ink, 800)}
    ${text(1196, 96, step, 20, colors.muted, 700, mono, 'end')}
  `;
}

function base(svgBody, w = W, h = H) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    ${gradientDefs()}
    <rect width="${w}" height="${h}" fill="url(#paperGlow)"/>
    <circle cx="${w - 120}" cy="92" r="260" fill="${colors.accent}" opacity="0.10"/>
    <circle cx="118" cy="${h - 96}" r="240" fill="${colors.meadow}" opacity="0.08"/>
    ${svgBody}
  </svg>`;
}

function quickNoteBody(progress, insertedText = '') {
  const caretX = 228 + Math.min(1, progress) * 330;
  return `
    ${noteWindow(132, 136, 1008, 452, 'Untitled note')}
    ${text(188, 316, insertedText || 'Send a note saying I will share the proposal by Friday afternoon.', 32, insertedText ? colors.ink : colors.dim, 650)}
    <rect x="${caretX}" y="284" width="3" height="44" rx="1.5" fill="${colors.accent}" opacity="${insertedText ? 0 : 1}"/>
  `;
}

function screenshotCopyReady() {
  return base(`
    ${noteWindow(126, 132, 1712, 670, 'Target app')}
    ${text(202, 374, 'Customer reply', 38, colors.ink, 800)}
    ${rect(202, 424, 1460, 188, 22, '#fbf8f1', 'rgba(9,8,14,0.08)')}
    ${wrapText(244, 500, 'The target changed while Juno was finishing, so the final text stayed available instead of being pasted into the wrong field.', 1250, 32, colors.muted, 560)}
    ${hud(458, 708, 'Copy Ready', 'I will share the proposal by Friday afternoon. If anything changes before then, I will send an updated timeline.', { tone: 'copy', width: 1048 })}
    ${rect(662, 1016, 640, 72, 28, colors.navy, 'transparent')}
    ${text(982, 1062, 'Copy output', 26, '#f4f1ea', 800, font, 'middle')}
  `, screenshotW, screenshotH);
}

function screenshotSecureField() {
  return base(`
    ${noteWindow(154, 150, 1656, 720, 'Secure sign-in')}
    ${text(248, 350, 'Password', 34, colors.ink, 800)}
    ${rect(248, 390, 1050, 94, 24, '#fffdf8', 'rgba(9,8,14,0.10)')}
    ${text(288, 451, '••••••••••••••', 38, colors.muted, 700, mono)}
    ${rect(1326, 390, 250, 94, 24, '#eeeeee', 'rgba(9,8,14,0.08)')}
    ${text(1451, 451, 'Sign in', 28, colors.dim, 800, font, 'middle')}
    ${hud(478, 660, 'Blocked', 'Secure field detected. Juno skips capture, context, learning, history, audio save, and paste.', { tone: 'blocked', width: 1008 })}
  `, screenshotW, screenshotH);
}

function junoAppScreenshot(section, title, body, active = section) {
  return base(`
    ${junoWindow(128, 120, 1708, 820, active)}
    ${rect(470, 196, 1180, 528, 26, '#fbf8f1', 'rgba(9,8,14,0.07)')}
    ${text(526, 276, title, 42, colors.ink, 850)}
    ${wrapText(526, 332, body, 920, 30, colors.muted, 560)}
  `, screenshotW, screenshotH);
}

function listRows(x, y, rows, selected = 0) {
  return rows.map((row, i) => `
    ${rect(x, y + i * 82, 700, 60, 18, i === selected ? colors.accentDim : '#fffdf8', 'rgba(9,8,14,0.06)')}
    ${text(x + 24, y + 37 + i * 82, row, 23, colors.ink, 760)}
  `).join('');
}

function onboardingScreenshot(step) {
  const copy = {
    welcome: ['Speak naturally.', 'Juno writes where you already are.', 'Your name (optional)'],
    permissions: ["You're all set", 'Microphone and Accessibility unlock normal dictation. Live captions are optional.', 'Continue'],
    ready: ['Ready when you are', 'Models live on this Mac. Try one sentence, then open Juno.', 'Open Juno'],
    actions: ['Juno can do things too', 'Enable Hey Juno actions if you want Notes, Reminders, and Alarms. Voice Commands work either way.', 'Grant all three'],
  }[step];
  return base(`
    ${rect(222, 104, 1518, 940, 34, '#fffaf2', 'rgba(9,8,14,0.08)', 1, 'filter="url(#shadow)"')}
    ${chromeDots(266, 148)}
    ${['Welcome', 'Access', 'Shortcut', 'Setup', 'Actions', 'Try it'].map((s, i) => `<rect x="${288 + i * 230}" y="224" width="180" height="8" rx="4" fill="${colors.accent}"/><text x="${288 + i * 230}" y="268" font-family="${font}" font-size="20" font-weight="800" fill="${colors.ink}">${s}</text>`).join('')}
    <circle cx="982" cy="410" r="46" fill="${colors.navy}"/>
    ${bars(954, 410, 1, '#ffffff', 2)}
    ${text(982, 520, copy[0], 58, colors.ink, 850, font, 'middle')}
    ${wrapText(710, 582, copy[1], 540, 30, colors.muted, 560)}
    ${step === 'permissions' ? listRows(560, 690, ['Microphone active', 'Accessibility active', 'Live captions optional'], 0) : ''}
    ${step === 'actions' ? listRows(560, 690, ['Reminders', 'Calendar-backed alarms', 'Apple Notes'], 0) : ''}
    ${rect(756, 866, 450, 70, 26, colors.navy, 'transparent')}
    ${text(982, 911, copy[2], 27, '#f4f1ea', 800, font, 'middle')}
  `, screenshotW, screenshotH);
}

function settingsScreenshot(kind) {
  const title = kind === 'permissions' ? 'Permissions' : kind === 'recording' ? 'Storage' : 'Make Juno feel right on this Mac';
  const rows = kind === 'permissions'
    ? ['Microphone: Granted', 'Accessibility: Granted', 'Live captions: Optional']
    : kind === 'recording'
      ? ['Keep history for: 90 days', 'Keep recordings for: 30 days', 'Clear history...', 'Delete recordings...', 'Clean up now']
      : ['General: Appearance, Dock, recordings, HUD position', 'Dictation: Shortcut, HUD sounds, pause sensitivity', 'Writing & language: Language, speaking environment', 'Privacy & learning: Smart Context and corrections', 'Updates & app: auto-update, login, Developer mode'];
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'Settings')}
    ${kind === 'overview' ? '' : text(492, 310, title, 38, colors.ink, 850)}
    ${listRows(492, kind === 'overview' ? 306 : 380, rows, 0)}
    ${rect(1240, 340, 420, 320, 26, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(1282, 406, 'Production defaults', 28, colors.ink, 850)}
    ${wrapText(1282, 470, 'User-facing docs describe the final product behavior and avoid internal route, helper, or engine implementation names.', 320, 24, colors.muted, 560)}
  `, screenshotW, screenshotH);
}

function homeScreenshot() {
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'Home')}
    ${rect(492, 232, 680, 220, 28, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(536, 306, 'Good morning, Jas.', 44, colors.ink, 850)}
    ${wrapText(536, 366, 'Juno is ready. Press Fn / Globe and start speaking in any writing surface.', 540, 26, colors.muted, 560)}
    ${rect(1212, 232, 408, 220, 28, '#f7fbf6', 'rgba(9,8,14,0.07)')}
    ${text(1256, 302, 'Today', 24, colors.muted, 750)}
    ${text(1256, 364, '1,248 words', 42, colors.ink, 850)}
    ${text(1256, 408, 'most in Notes', 22, colors.meadow, 750)}
    ${rect(492, 500, 496, 276, 26, '#fbf8f1', 'rgba(9,8,14,0.07)')}
    ${text(536, 568, 'Actions', 32, colors.ink, 850)}
    ${wrapText(536, 622, 'Notes, reminders, and alarms are controlled from the dedicated Actions page.', 380, 24, colors.muted, 560)}
    ${rect(1036, 500, 584, 276, 26, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(1080, 568, 'Recent dictations', 32, colors.ink, 850)}
    ${listRows(1080, 612, ['Proposal follow-up', 'Bug triage summary'], 0)}
  `, screenshotW, screenshotH);
}

function perAppWritingScreenshot() {
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'Per-app writing')}
    ${wrapText(492, 312, 'Choose how Juno writes, learns, saves, and inserts text in specific apps.', 940, 27, colors.muted, 560)}
    ${listRows(492, 420, ['Apple Notes: Structured notes', 'Mail: Formal email', 'Terminal: Copy Ready only', 'Remote desktop: context off'], 0)}
  `, screenshotW, screenshotH);
}

function privacyScreenshot() {
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'Privacy')}
    ${wrapText(492, 312, 'Review what Juno can observe, save, learn, and block while dictating.', 940, 27, colors.muted, 560)}
    ${listRows(492, 420, ['Local data stays on this Mac', 'Secure fields block capture and paste', 'History and recordings are separate', 'Cleanup controls are explicit'], 0)}
  `, screenshotW, screenshotH);
}

function memoryScreenshot() {
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'Dictionary')}
    ${rect(492, 308, 340, 492, 24, '#fbf8f1', 'rgba(9,8,14,0.07)')}
    ${listRows(526, 356, ['Vocabulary', 'Corrections', 'Replacements', 'Snippets'], 0)}
    ${rect(880, 308, 620, 492, 24, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(934, 374, 'Vocabulary', 34, colors.ink, 850)}
    ${wrapText(934, 438, 'Names, terms, and product spellings Juno should preserve when it writes.', 460, 26, colors.muted, 560)}
    ${['Jas', 'Juno', 'Qwen', 'Whisper', 'Product Hunt'].map((s, i) => text(938, 534 + i * 48, s, 26, colors.ink, 700)).join('')}
  `, screenshotW, screenshotH);
}

function historyScreenshot() {
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'History')}
    ${rect(492, 308, 520, 520, 24, '#fbf8f1', 'rgba(9,8,14,0.07)')}
    ${listRows(526, 358, ['Proposal follow-up', 'Hiring note', 'Weekly update', 'Bug triage summary'], 0)}
    ${rect(1060, 308, 560, 520, 24, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(1114, 384, 'Proposal follow-up', 34, colors.ink, 850)}
    ${wrapText(1114, 454, 'Final text, raw transcript, target app, insertion outcome, and deletion controls live here.', 410, 26, colors.muted, 560)}
  `, screenshotW, screenshotH);
}

function stylesScreenshot() {
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'Styles')}
    ${listRows(492, 278, ['Default (balanced)', 'Verbatim', 'Casual chat', 'Formal email', 'Structured notes', 'Polished rewrite', 'Commands'], 0)}
    ${rect(1240, 326, 420, 310, 26, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(1282, 388, 'Default style', 30, colors.ink, 850)}
    ${wrapText(1282, 446, 'Balanced cleanup that keeps meaning while improving punctuation, casing, and clarity.', 320, 24, colors.muted, 560)}
  `, screenshotW, screenshotH);
}

function voiceActionsScreenshot() {
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'Actions')}
    ${text(492, 296, 'Let Juno save the small things', 44, colors.ink, 850)}
    ${wrapText(492, 358, 'Say Hey Juno for Juno Actions. Hey Siri belongs to Apple Siri.', 920, 28, colors.muted, 560)}
    ${rect(492, 468, 1060, 86, 24, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(532, 520, 'Voice Actions are off', 27, colors.ink, 850)}
    ${rect(1202, 482, 136, 48, 18, colors.navy, 'transparent')}
    ${text(1270, 520, 'Turn on', 22, '#f4f1ea', 800, font, 'middle')}
    ${listRows(492, 594, ['Reminders: Off', 'Notes: Set up', 'Alarms: Off'], 1)}
    ${rect(1248, 594, 330, 210, 22, '#fbf8f1', 'rgba(9,8,14,0.07)')}
    ${text(1282, 646, 'Say this', 20, colors.meadow, 850)}
    ${wrapText(1282, 690, 'Hey Juno, remind me to call mom at 6pm.', 250, 22, colors.ink, 700)}
    ${text(1282, 752, 'Not this', 20, colors.danger, 850)}
    ${wrapText(1282, 786, 'Hey Siri is Apple, not Juno.', 250, 20, colors.muted, 650)}
  `, screenshotW, screenshotH);
}

function voiceCommandsScreenshot() {
  return base(`
    ${junoWindow(128, 116, 1708, 840, 'Voice Commands')}
    ${text(492, 286, 'Edit without lifting your hands', 40, colors.ink, 850)}
    ${wrapText(492, 340, 'Voice Commands edit text during dictation. They do not need Reminders, Calendar, or Notes permissions.', 900, 26, colors.muted, 560)}
    ${rect(492, 438, 520, 330, 24, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(532, 498, "While you're talking", 27, colors.ink, 850)}
    ${listRows(532, 544, ['scratch that', 'new paragraph', 'next bullet'], 0)}
    ${rect(1060, 438, 520, 330, 24, '#fffdf8', 'rgba(9,8,14,0.07)')}
    ${text(1100, 498, 'Edit what you just said', 27, colors.ink, 850)}
    ${listRows(1100, 544, ['fix that', 'make that shorter', 'turn that into bullets'], 0)}
  `, screenshotW, screenshotH);
}

function quickstartBefore() {
  return base(`
    ${noteWindow(154, 142, 1656, 720, 'Quick Notes')}
    ${text(248, 350, 'Place the cursor where Juno should write.', 42, colors.ink, 850)}
    ${rect(248, 414, 1240, 230, 24, '#fffdf8', 'rgba(9,8,14,0.08)')}
    <rect x="292" y="474" width="4" height="58" rx="2" fill="${colors.accent}"/>
  `, screenshotW, screenshotH);
}

function quickstartInserted() {
  return base(`
    ${noteWindow(154, 142, 1656, 720, 'Quick Notes')}
    ${text(248, 350, 'First dictation', 42, colors.ink, 850)}
    ${rect(248, 414, 1240, 230, 24, '#fffdf8', 'rgba(9,8,14,0.08)')}
    ${wrapText(292, 502, 'I will share the proposal by Friday afternoon.', 900, 34, colors.ink, 650)}
    ${hud(504, 682, 'Inserted', 'Text placed at the cursor.', { width: 956 })}
  `, screenshotW, screenshotH);
}

function hudListeningTarget() {
  return base(`
    ${noteWindow(154, 142, 1656, 720, 'Target app')}
    ${text(248, 350, 'Writing surface', 42, colors.ink, 850)}
    ${rect(248, 414, 1240, 230, 24, '#fffdf8', 'rgba(9,8,14,0.08)')}
    ${wrapText(292, 502, 'Send a note saying I will share the proposal by Friday afternoon.', 900, 34, colors.muted, 650)}
    ${hud(504, 682, 'Listening', 'Send a note saying I will share the proposal by Friday afternoon.', { width: 956, phase: 4 })}
  `, screenshotW, screenshotH);
}

function renderScreenshotSet() {
  return [
    ['actions-view.png', voiceActionsScreenshot()],
    ['voice-commands-view.png', voiceCommandsScreenshot()],
    ['dictionary-memory.png', memoryScreenshot()],
    ['first-launch-permissions.png', onboardingScreenshot('permissions')],
    ['first-launch-ready.png', onboardingScreenshot('ready')],
    ['first-launch-welcome.png', onboardingScreenshot('welcome')],
    ['first-launch-actions.png', onboardingScreenshot('actions')],
    ['history-view.png', historyScreenshot()],
    ['home-overview.png', homeScreenshot()],
    ['hud-listening-target-app.png', hudListeningTarget()],
    ['per-app-writing-controls.png', perAppWritingScreenshot()],
    ['privacy-view.png', privacyScreenshot()],
    ['styles-view.png', stylesScreenshot()],
    ['notes-view.png', voiceActionsScreenshot()],
    ['quickstart-cursor-before-dictation.png', quickstartBefore()],
    ['quickstart-inserted-result.png', quickstartInserted()],
    ['recording-retention-setting.png', settingsScreenshot('recording')],
    ['settings-overview.png', settingsScreenshot('overview')],
    ['settings-permissions.png', settingsScreenshot('permissions')],
    ['copy-ready-output.png', screenshotCopyReady()],
    ['secure-field-blocked.png', screenshotSecureField()],
  ];
}

function videoFrame(kind, frame, total) {
  const p = frame / Math.max(1, total - 1);
  if (kind === 'juno-first-launch') {
    const steps = ['Welcome', 'Access', 'Shortcut', 'Setup', 'Actions', 'Try it'];
    const idx = Math.min(5, Math.floor(p * 6));
    return base(`
      ${captionStrip('First launch walkthrough', `${idx + 1} / 6`)}
      ${rect(214, 152, 852, 494, 34, '#fffaf2', 'rgba(9,8,14,0.08)', 1, 'filter="url(#shadow)"')}
      ${chromeDots(254, 190)}
      ${steps.map((s, i) => `<rect x="${258 + i * 126}" y="244" width="88" height="8" rx="4" fill="${i <= idx ? colors.accent : '#ded8ce'}"/>${text(258 + i * 126, 282, s, 13, i === idx ? colors.ink : colors.muted, 800)}`).join('')}
      ${text(640, 390, idx === 0 ? 'Write with your voice.' : idx === 1 ? 'Grant microphone and accessibility.' : idx === 2 ? 'Choose Fn / Globe.' : idx === 3 ? 'Prepare the local voice engine.' : idx === 4 ? 'Choose action permissions.' : 'Try one sentence.', 42, colors.ink, 850, font, 'middle')}
      ${wrapText(410, 452, idx === 3 ? 'Setup shows Starting up, Downloading models, and Warming up before Juno opens.' : idx === 4 ? 'Actions are optional. Dictation and Voice Commands keep working even when they are off.' : 'Juno guides the first run without requiring Terminal or developer setup.', 460, 22, colors.muted, 560)}
      ${idx === 3 ? bars(562, 548, 1.2, colors.accent, frame) : rect(512, 530, 256, 58, 24, colors.navy, 'transparent') + text(640, 568, idx === 5 ? 'Open Juno' : 'Continue', 22, '#f4f1ea', 800, font, 'middle')}
    `);
  }
  if (kind === 'juno-first-dictation' || kind === 'juno-speak-and-insert') {
    const state = p < 0.18 ? 'Waiting for speech' : p < 0.62 ? 'Listening' : p < 0.82 ? 'Polishing' : 'Inserted';
    const body = p < 0.18 ? '' : p < 0.62 ? 'Send a note saying I will share the proposal by Friday afternoon.' : p < 0.82 ? 'Turning speech into clean text...' : 'I will share the proposal by Friday afternoon.';
    return base(`
      ${captionStrip(kind === 'juno-speak-and-insert' ? 'Speak and insert' : 'First dictation', state)}
      ${quickNoteBody(p > 0.82 ? 1 : 0, p > 0.82 ? 'I will share the proposal by Friday afternoon.' : '')}
      ${hud(358, 520, state, body, { phase: frame, width: 564 })}
    `);
  }
  if (kind === 'juno-styles-demo') {
    const formal = p > 0.48;
    return base(`
      ${captionStrip('Style change demo', formal ? 'Formal email' : 'Default')}
      ${junoWindow(124, 120, 1030, 510, 'Styles')}
      ${rect(294, 162, 330, 382, 22, '#fbf8f1', 'rgba(9,8,14,0.07)')}
      ${['Default (balanced)', 'Formal email', 'Team update'].map((s, i) => rect(320, 204 + i * 70, 250, 46, 16, (formal && i === 1) || (!formal && i === 0) ? colors.accentDim : 'transparent', 'rgba(9,8,14,0.04)') + text(344, 234 + i * 70, s, 20, colors.ink, 750)).join('')}
      ${rect(670, 206, 386, 270, 24, '#ffffff', 'rgba(9,8,14,0.07)')}
      ${text(704, 258, formal ? 'Output preview' : 'Default style', 24, colors.ink, 800)}
      ${wrapText(704, 314, formal ? 'I will send the revised proposal by Friday afternoon and follow up with the final timeline.' : 'I’ll send the proposal by Friday afternoon and follow up if anything changes.', 296, 22, colors.muted, 560)}
    `);
  }
  if (kind === 'juno-copy-ready-fallback') {
    const copy = p > 0.48;
    return base(`
      ${captionStrip('Safe insertion fallback', copy ? 'Copy Ready' : 'Inserted')}
      ${copy ? screenshotSecureField().replace(/^[\s\S]*?<rect width="1964" height="1200"[^>]*>/, '').replace('</svg>', '') : quickNoteBody(1, 'I will share the proposal by Friday afternoon.')}
      ${hud(356, 526, copy ? 'Copy Ready' : 'Inserted', copy ? 'The target is blocked, so the final text stays ready to copy.' : 'Text placed at the cursor.', { tone: copy ? 'copy' : 'normal', width: 568, phase: frame })}
    `);
  }
  if (kind === 'juno-shortcut-hud-states') {
    const states = ['Checking microphone', 'Waiting for speech', 'Listening', 'Polishing', 'Inserted', 'Copy Ready'];
    const idx = Math.min(states.length - 1, Math.floor(p * states.length));
    return base(`
      ${captionStrip('Shortcut and HUD states', states[idx])}
      ${rect(188, 198, 904, 326, 34, '#fffaf2', 'rgba(9,8,14,0.08)', 1, 'filter="url(#shadow)"')}
      ${text(640, 300, 'Press Fn / Globe once to start.', 38, colors.ink, 850, font, 'middle')}
      ${wrapText(402, 360, 'Juno moves through explicit states instead of pretending to listen before microphone frames and speech are detected.', 476, 22, colors.muted, 560)}
      ${hud(358, 482, states[idx], idx >= 2 ? 'A short spoken note appears here while Juno listens and polishes.' : '', { tone: idx === 5 ? 'copy' : 'normal', width: 564, phase: frame })}
    `);
  }
  return base(`
    ${captionStrip('Juno product overview', p < 0.34 ? 'Start' : p < 0.68 ? 'Speak' : 'Insert safely')}
    ${quickNoteBody(p > 0.68 ? 1 : 0, p > 0.68 ? 'I will share the proposal by Friday afternoon.' : '')}
    ${hud(356, 522, p < 0.34 ? 'Waiting for speech' : p < 0.68 ? 'Listening' : 'Inserted', p < 0.34 ? '' : p < 0.68 ? 'Speak naturally. Juno keeps the destination in view.' : 'Text placed at the cursor when the target is safe.', { phase: frame, width: 568 })}
  `);
}

async function renderPng(svg, path) {
  await sharp(Buffer.from(svg)).png().toFile(path);
}

async function renderVideo(kind, file, frames = 180) {
  const dir = join(tmpRoot, kind);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  for (let i = 0; i < frames; i += 1) {
    await renderPng(videoFrame(kind, i, frames), join(dir, `${String(i).padStart(4, '0')}.png`));
  }
  const out = join(outVideos, file);
  const result = spawnSync('ffmpeg', [
    '-y',
    '-r', '30',
    '-i', join(dir, '%04d.png'),
    '-f', 'lavfi',
    '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
    '-shortest',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'veryfast',
    '-crf', '23',
    '-c:a', 'aac',
    '-b:a', '64k',
    '-movflags', '+faststart',
    out,
  ], { stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`ffmpeg failed for ${file}`);
}

async function main() {
  const screenshotsOnly = process.argv.includes('--screenshots-only');
  const stylesOnly = process.argv.includes('--styles-only');
  mkdirSync(outScreens, { recursive: true });
  mkdirSync(outVideos, { recursive: true });
  mkdirSync(tmpRoot, { recursive: true });
  if (stylesOnly) {
    await renderPng(stylesScreenshot(), join(outScreens, 'styles-view.png'));
    await renderVideo('juno-styles-demo', 'juno-styles-demo.mp4');
    return;
  }
  await renderPng(screenshotCopyReady(), join(outScreens, 'copy-ready-output.png'));
  await renderPng(screenshotSecureField(), join(outScreens, 'secure-field-blocked.png'));
  for (const [file, svg] of renderScreenshotSet()) {
    await renderPng(svg, join(outScreens, file));
  }
  if (screenshotsOnly) return;
  await renderVideo('juno-product-overview', 'juno-product-overview.mp4');
  await renderVideo('juno-first-launch', 'juno-first-launch.mp4');
  await renderVideo('juno-first-dictation', 'juno-first-dictation.mp4');
  await renderVideo('juno-styles-demo', 'juno-styles-demo.mp4');
  await renderVideo('juno-copy-ready-fallback', 'juno-copy-ready-fallback.mp4');
  await renderVideo('juno-shortcut-hud-states', 'juno-shortcut-hud-states.mp4');
  await renderVideo('juno-speak-and-insert', 'juno-speak-and-insert.mp4');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
