"""Auto-enable HF Hub offline mode when all required models are cached.

Juno is a local-first product: once the user has downloaded the models,
the app must work without an internet connection. The underlying
``huggingface_hub`` library, however, defaults to revalidating cached
artifacts against the hub on every ``snapshot_download``. With no
network, that revalidation can stall for ~10 s per file before falling
back to the cache — multiplied across four model repos (preview, final,
writer, optional live corrector), the user sees the app "fail to work"
for the better part of a minute on offline boot.

This module exposes :func:`enable_offline_mode_if_cache_complete`. The
engine entrypoints call it once at startup. When every required repo is
present in the local cache it sets ``HF_HUB_OFFLINE=1`` in
``os.environ``; all downstream ``snapshot_download`` / ``hf_hub_download``
calls inside ``mlx_whisper`` and ``mlx_lm`` then read straight from the
cache. The explicit install/repair flow temporarily pops the env var so
deliberate downloads still work — see
:func:`hub_online_for_explicit_download`.

Import is cheap: we only depend on ``huggingface_hub.try_to_load_from_cache``,
which doesn't touch the network or load any model weight files.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterable, Iterator


def enable_offline_mode_if_cache_complete(repo_ids: Iterable[str]) -> bool:
    """Set ``HF_HUB_OFFLINE=1`` iff every ``repo_id`` is cached locally.

    Probes each repo via ``huggingface_hub.try_to_load_from_cache`` for
    its ``config.json`` (the canonical "this model is present" marker).
    A miss on any repo is treated as "we may still need the network" —
    the env var is left alone so the default revalidate-then-fallback
    behaviour stays in play.

    Returns ``True`` when offline mode was applied. Returns ``False``
    when at least one repo is uncached, when ``repo_ids`` is empty,
    when ``huggingface_hub`` is unavailable, or when ``HF_HUB_OFFLINE``
    was already set by the caller (in which case we don't change it).

    Idempotent: calling repeatedly is safe.

    Safe to call before importing ``mlx_whisper`` / ``mlx_lm`` — those
    libraries read ``HF_HUB_OFFLINE`` lazily on each download call, so
    setting it first ensures the very first model load already sees it.
    """

    if os.environ.get("HF_HUB_OFFLINE"):
        # Caller (or operator) already pinned it; leave their value.
        return False

    ids = [r.strip() for r in repo_ids if r and r.strip()]
    if not ids:
        return False

    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import-not-found]
    except Exception:
        return False

    for repo_id in ids:
        try:
            result = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
        except Exception:
            print(
                f"[OFFLINE]     skipped (cache probe error) repo={repo_id}",
                file=sys.stderr,
                flush=True,
            )
            return False
        if not isinstance(result, str):
            print(
                f"[OFFLINE]     skipped (cache miss) repo={repo_id}",
                file=sys.stderr,
                flush=True,
            )
            return False

    os.environ["HF_HUB_OFFLINE"] = "1"
    print(
        f"[OFFLINE]     enabled HF_HUB_OFFLINE=1 (cache complete for {len(ids)} repo(s))",
        file=sys.stderr,
        flush=True,
    )
    return True


@contextmanager
def hub_online_for_explicit_download() -> Iterator[None]:
    """Temporarily clear ``HF_HUB_OFFLINE`` for the duration of a
    deliberate download (install or repair). Restores the prior value
    on exit so the rest of the process returns to its offline-safe
    posture.

    Note: ``os.environ`` is process-global, so this affects every
    thread for the duration. Callers must hold whatever lock protects
    against concurrent installs (the workbench server holds
    ``_setup_install_lock``).
    """

    prior = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        yield
    finally:
        if prior is not None:
            os.environ["HF_HUB_OFFLINE"] = prior


def required_hf_repo_ids_from_config(config: object) -> list[str]:
    """Extract HF repo IDs from a ``ProductionServiceConfig`` (or a
    duck-typed equivalent that exposes the same attributes). Returns the
    set of model paths that look like HF repo identifiers.

    A value is treated as an HF repo id when it contains a slash and
    doesn't resolve to an existing filesystem path — same rule used
    by ``juno_core_v3.dictation.transcriber.is_hf_repo_id``.
    """

    from pathlib import Path

    out: list[str] = []
    seen: set[str] = set()
    for attr in (
        "preview_model_path",
        "final_model_path",
        "writer_model_path",
        "live_corrector_model_path",
    ):
        raw = str(getattr(config, attr, "") or "").strip()
        if not raw or "/" not in raw or raw in seen:
            continue
        # Only treat as repo id when there's no on-disk path with that
        # value (production uses HF repo ids, tests + demos sometimes
        # point at a local dir).
        if Path(raw).exists():
            continue
        seen.add(raw)
        out.append(raw)
    return out
