import fs from 'node:fs';
import path from 'node:path';

const root = path.join(process.cwd(), 'content', 'docs');
const userFacingSections = new Set([
  'index.mdx',
  'start',
  'use-juno',
  'privacy-and-data',
  'troubleshooting',
  'releases',
]);
const developerFacingSections = new Set([
  'architecture',
  'developers',
  'reference',
  'contribute',
]);

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else if (entry.name.endsWith('.mdx')) out.push(full);
  }
  return out;
}

function sectionFor(relPath) {
  const parts = relPath.split(path.sep);
  return parts.length === 1 ? parts[0] : parts[0];
}

function assertMissing(text, relPath, token, failures) {
  if (text.includes(token)) failures.push(`${relPath} should not include ${token}`);
}

function assertPresent(text, relPath, token, failures) {
  if (!text.includes(token)) failures.push(`${relPath} missing ${token}`);
}

function getFrontmatter(text) {
  const match = text.match(/^---\n([\s\S]*?)\n---/);
  return match ? match[1] : '';
}

const files = walk(root);
const failures = [];

for (const file of files) {
  const relPath = path.relative(root, file);
  const text = fs.readFileSync(file, 'utf8');
  const section = sectionFor(relPath);
  const frontmatter = getFrontmatter(text);

  assertPresent(frontmatter, relPath, 'title:', failures);
  assertPresent(frontmatter, relPath, 'description:', failures);

  assertMissing(text, relPath, '## Page brief', failures);
  assertMissing(text, relPath, '## Draft outline', failures);
  assertMissing(text, relPath, '## Acceptance test', failures);
  assertMissing(text, relPath, 'StatusBadge', failures);
  assertMissing(text, relPath, 'AudienceTag', failures);

  if (userFacingSections.has(section)) {
    assertMissing(frontmatter, relPath, 'status:', failures);
    assertMissing(frontmatter, relPath, 'audience:', failures);
    assertMissing(frontmatter, relPath, 'pageType:', failures);
  }

  if (developerFacingSections.has(section)) {
    assertMissing(frontmatter, relPath, 'status:', failures);
  }
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`Validated ${files.length} production MDX pages.`);
