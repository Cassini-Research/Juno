import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const liveDir = 'public/images/screenshots/live';
const screenshotDir = 'public/images/screenshots';
const videoDir = 'public/videos/demos';
const tmpRoot = '/private/tmp/juno-docs-live-video';
const videoFilter =
  'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0xf4f1ea';

const screenshotMap = [
  [['juno-onboarding-welcome-20260519.png', 'juno-first-launch-welcome-live-cropped.png'], 'first-launch-welcome.png'],
  [['juno-onboarding-access-20260519.png', 'juno-first-launch-access-live-cropped.png'], 'first-launch-permissions.png'],
  ['juno-first-launch-setup-live-cropped.png', 'first-launch-ready.png'],
  ['juno-home-live-cropped.png', 'home-overview.png'],
  ['juno-actions-live-cropped.png', 'actions-view.png'],
  ['juno-actions-live-cropped.png', 'notes-view.png'],
  ['juno-history-live-cropped.png', 'history-view.png'],
  ['juno-styles-live-cropped.png', 'styles-view.png'],
  ['juno-dictionary-memory-live-cropped.png', 'dictionary-memory.png'],
  ['juno-per-app-writing-live-cropped.png', 'per-app-writing-controls.png'],
  ['juno-settings-live-cropped.png', 'settings-overview.png'],
  ['juno-settings-live-cropped.png', 'recording-retention-setting.png'],
  [['juno-onboarding-access-20260519.png', 'juno-first-launch-access-live-cropped.png'], 'settings-permissions.png'],
  ['juno-first-launch-try-live-cropped.png', 'quickstart-cursor-before-dictation.png'],
  ['juno-first-launch-try-live-cropped.png', 'quickstart-inserted-result.png'],
  ['juno-first-launch-try-live-cropped.png', 'hud-listening-target-app.png'],
  ['juno-first-launch-try-live-cropped.png', 'copy-ready-output.png'],
];

const videos = [
  {
    out: 'juno-first-launch.mp4',
    frames: [
      'juno-first-launch-welcome-live-cropped.png',
      'juno-first-launch-access-live-cropped.png',
      'juno-first-launch-shortcut-live-cropped.png',
      'juno-first-launch-setup-live-cropped.png',
      'juno-first-launch-actions-live-cropped.png',
      'juno-first-launch-try-live-cropped.png',
    ],
  },
  {
    out: 'juno-first-dictation.mp4',
    frames: [
      'juno-first-launch-shortcut-live-cropped.png',
      'juno-first-launch-try-live-cropped.png',
      'juno-home-live-cropped.png',
      'juno-history-live-cropped.png',
    ],
  },
  {
    out: 'juno-shortcut-hud-states.mp4',
    frames: [
      'juno-first-launch-shortcut-live-cropped.png',
      'juno-first-launch-try-live-cropped.png',
      'juno-settings-live-cropped.png',
    ],
  },
  {
    out: 'juno-styles-demo.mp4',
    frames: [
      'juno-styles-live-cropped.png',
      'juno-per-app-writing-live-cropped.png',
      'juno-settings-live-cropped.png',
    ],
  },
  {
    out: 'juno-copy-ready-fallback.mp4',
    frames: [
      'juno-first-launch-try-live-cropped.png',
      'juno-per-app-writing-live-cropped.png',
      'juno-history-live-cropped.png',
    ],
  },
  {
    out: 'juno-speak-and-insert.mp4',
    frames: [
      'juno-first-launch-shortcut-live-cropped.png',
      'juno-first-launch-try-live-cropped.png',
      'juno-home-live-cropped.png',
    ],
  },
];

function ensureFile(path) {
  if (!existsSync(path)) {
    throw new Error(`Missing required live capture: ${path}`);
  }
}

function resolveLiveCapture(source) {
  const candidates = Array.isArray(source) ? source : [source];
  for (const candidate of candidates) {
    const path = join(liveDir, candidate);
    if (existsSync(path)) return path;
  }
  throw new Error(`Missing required live capture: ${candidates.map((candidate) => join(liveDir, candidate)).join(' or ')}`);
}

function run(command, args) {
  const result = spawnSync(command, args, { stdio: 'inherit' });
  if (result.status !== 0) {
    throw new Error(`${command} failed with status ${result.status}`);
  }
}

function copyScreenshots() {
  for (const [source, destination] of screenshotMap) {
    copyFileSync(resolveLiveCapture(source), join(screenshotDir, destination));
  }
}

function makeSlideshow({ out, frames }) {
  const workDir = join(tmpRoot, out.replace(/[^a-z0-9.-]/gi, '_'));
  rmSync(workDir, { recursive: true, force: true });
  mkdirSync(workDir, { recursive: true });

  let index = 0;
  for (const frame of frames) {
    const srcPath = join(liveDir, frame);
    if (!existsSync(srcPath) && frame === 'juno-first-launch-actions-live-cropped.png') {
      continue;
    }
    ensureFile(srcPath);
    for (let repeat = 0; repeat < 42; repeat += 1) {
      copyFileSync(srcPath, join(workDir, `${String(index).padStart(5, '0')}.png`));
      index += 1;
    }
  }

  run('ffmpeg', [
    '-y',
    '-framerate',
    '30',
    '-i',
    join(workDir, '%05d.png'),
    '-f',
    'lavfi',
    '-i',
    'anullsrc=channel_layout=stereo:sample_rate=48000',
    '-shortest',
    '-vf',
    videoFilter,
    '-c:v',
    'libx264',
    '-pix_fmt',
    'yuv420p',
    '-preset',
    'veryfast',
    '-crf',
    '21',
    '-c:a',
    'aac',
    '-b:a',
    '64k',
    '-movflags',
    '+faststart',
    join(videoDir, out),
  ]);
}

function transcodeProductOverview() {
  const liveRecording = join(videoDir, 'juno-product-overview-live.mov');
  if (!existsSync(liveRecording)) {
    makeSlideshow({
      out: 'juno-product-overview.mp4',
      frames: [
        'juno-home-live-cropped.png',
        'juno-actions-live-cropped.png',
        'juno-history-live-cropped.png',
        'juno-styles-live-cropped.png',
        'juno-dictionary-memory-live-cropped.png',
        'juno-per-app-writing-live-cropped.png',
        'juno-settings-live-cropped.png',
      ],
    });
    return;
  }

  run('ffmpeg', [
    '-y',
    '-i',
    liveRecording,
    '-f',
    'lavfi',
    '-i',
    'anullsrc=channel_layout=stereo:sample_rate=48000',
    '-shortest',
    '-vf',
    videoFilter,
    '-c:v',
    'libx264',
    '-pix_fmt',
    'yuv420p',
    '-preset',
    'veryfast',
    '-crf',
    '21',
    '-c:a',
    'aac',
    '-b:a',
    '64k',
    '-movflags',
    '+faststart',
    join(videoDir, 'juno-product-overview.mp4'),
  ]);
}

mkdirSync(videoDir, { recursive: true });
copyScreenshots();
transcodeProductOverview();
for (const video of videos) {
  makeSlideshow(video);
}

console.log('Live Juno media published from captured app screens.');
