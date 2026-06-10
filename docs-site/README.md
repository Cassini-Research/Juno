# Juno Docs Site

Fumadocs scaffold for Juno’s final public launch documentation.

## Run locally

```bash
npm install
npm run typecheck
npm run build
npm run dev
```

## Structure

- `content/docs/` — MDX documentation pages
- `components/docs/` — reusable documentation components
- `public/images/screenshots/live/real/` — real app screenshots with sanitized sample data
- `public/images/diagrams/` — architecture diagrams
- `VIDEO_NOTES.md` — future video capture notes; no public page depends on videos
- `DOCS_STYLE.md` — writing and claim rules
- `AGENTS.md` — assistant/editing rules

The public docs are screenshot-first. Do not use private user history, private app windows, generated UI mockups, or video placeholders in published pages.
