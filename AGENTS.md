# Agent Workflow

This repository uses a Git-flow style release process. Keep changes scoped,
reviewable, and merged through the branch that matches the work stage.

## Branch Roles

- `main`: production source of truth. Only release-ready code belongs here.
  Public releases and production tags are cut from this branch.
- `develop`: active integration branch for ongoing development. Feature and
  non-emergency fix PRs merge here first.
- `release`: stabilization branch for a planned release, updated from
  `develop` when the release scope is frozen. Only release blockers, packaging
  changes, version bumps, and release notes should land here.
- `feature/<short-name>`: short-lived branch for new product work, created from
  `develop`, merged back into `develop`.
- `fix/<short-name>`: short-lived branch for normal bug fixes, created from
  `develop`, merged back into `develop`.
- `hotfix/<short-name>`: urgent production fix, created from `main`, merged into
  `main`, then back-merged into `develop` and any active `release` branch.
- `master`: do not use for new work. If it exists, treat it only as a legacy
  alias of `main`; do not let `main` and `master` diverge.

## Standard Development Flow

1. Start new work from the latest `develop`.
2. Create a scoped branch:
   - `feature/<short-name>` for new behavior
   - `fix/<short-name>` for normal bug fixes
3. Keep commits focused and include tests or verification appropriate to the
   changed surface.
4. Open the PR into `develop`.
5. Merge to `develop` only after review and required checks pass.

Do not open normal feature or fix PRs directly into `main`.

## Release Flow

1. Update `release` from `develop` when the release scope is frozen.
2. Stabilize only on `release`:
   - release-blocking bug fixes
   - version/build number changes
   - release notes
   - packaging, signing, notarization, and OTA release adjustments
3. Open a PR from `release` into `main`.
4. After approval and passing checks, merge into `main`.
5. Tag the release from `main` using the shipped version, for example
   `v0.2.1`.
6. Back-merge `main` into `develop` so release fixes and version changes are not
   lost.
7. Back-merge `main` into `release` or reset `release` from `develop` when the
   next release cycle begins.

## Hotfix Flow

Use this only for production issues that cannot wait for the next release.

1. Create `hotfix/<short-name>` from `main`.
2. Fix only the production issue and add focused verification.
3. Open the PR into `main`.
4. After merge, tag or patch-release from `main` if needed.
5. Back-merge `main` into `develop`.
6. If the `release` branch is active, also merge the hotfix into that release
   branch.

## Agent Expectations

- Check the current branch before making changes.
- Preserve user changes already present in the worktree.
- Target PRs according to the branch roles above.
- Keep generated release artifacts out of source control unless a release
  process explicitly requires them.
- For macOS release work, follow `docs/ota-updates.md` for signing,
  notarization, Sparkle OTA, and DMG steps.
