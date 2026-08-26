from __future__ import annotations

import os
from dataclasses import dataclass, field

from juno_v2.contracts.writer import WriterMode


@dataclass(slots=True)
class WriterConfig:
    backend_name: str | None = None
    local_http_endpoint: str | None = None
    local_http_timeout_sec: float = 30.0
    model_path: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    default_mode: WriterMode = WriterMode.DEFAULT_SURFACE
    enable_model_transforms: bool = True
    enable_deterministic_transforms: bool = True
    enable_turn_planner: bool = field(
        default_factory=lambda: _env_bool("JUNO_V2_TURN_PLANNER_ENABLED", False)
    )
    # The model turn planner re-emits the whole transcript as JSON, so its
    # latency scales with utterance length (observed 8-28s on Qwen3-4B for
    # real dictation, on the paste critical path). Plain dictation never
    # consumes planner text — the renderer treats render_kind=plain as
    # corrected_text_fallback — so long non-wake utterances skip the model
    # and use the deterministic structural plan (list/checklist rendering)
    # at zero model cost. Short utterances, selection-anchored commands, and
    # explicit structure requests keep the model planner: that is where
    # transforms / memory mutations / message rendering actually live.
    # Flip the env to restore the model planner on every utterance.
    turn_plan_dictation_enabled: bool = field(
        default_factory=lambda: _env_bool("JUNO_V2_TURN_PLAN_DICTATION", False)
    )
    turn_plan_max_dictation_words: int = field(
        default_factory=lambda: _env_int("JUNO_V2_TURN_PLAN_MAX_DICTATION_WORDS", 16)
    )
    # Short utterances are under the word gate above, yet most of them are
    # still plain dictation with nothing for the planner to do — production
    # traces showed 4-24s of paste-path wait and zero actions across 25
    # utterances (issue #75). When enabled, a non-wake, unselected utterance
    # that carries no action / transform / structure / memory cue skips the
    # model planner entirely (reason ``plain_dictation_without_intent``).
    # Any cue keeps the planner, so the gate can only cost latency, never a
    # dropped command.
    #
    # Two scope notes: the cue lexicon is English, so the gate only applies
    # when the turn's language hint is absent or starts with "en" — a
    # non-English turn always keeps the planner. And like the word gate
    # above, this gate is disabled wholesale by JUNO_V2_TURN_PLAN_DICTATION=1
    # (which restores the model planner on every utterance); the repair skip
    # and the paste-path deadline below are independent of that flag and
    # still apply.
    turn_plan_plain_dictation_skip: bool = field(
        default_factory=lambda: _env_bool("JUNO_V2_TURN_PLAN_PLAIN_SKIP", True)
    )
    # Wall-clock decode budget for the planner when the turn is on the paste
    # critical path (no wake verification, no selection) and text delivery is
    # therefore already guaranteed. Handed to the backend as ``deadline_ms``,
    # the same mechanism the dictation editor uses; backends without a
    # streaming loop ignore it. 0 disables the budget. Wake/selection turns
    # never get a budget — truncating those would drop actions or transforms.
    # Applies whatever JUNO_V2_TURN_PLAN_DICTATION is set to: that flag only
    # governs which utterances reach the planner, not how long a reached
    # decode may run. A decode stopped on this budget is reported as
    # decode_timed_out in the turn_plan_generated trace.
    turn_plan_paste_deadline_ms: int = field(
        default_factory=lambda: _env_int("JUNO_V2_TURN_PLAN_PASTE_DEADLINE_MS", 6000)
    )
    # The dictation editor is the AI lane for every non-wake dictation turn
    # (cached-prefix edit-script model pass, deterministic application).
    # When enabled it supersedes the model turn planner for dictation —
    # selection transforms still use the planner.
    dictation_editor_enabled: bool = field(
        default_factory=lambda: _env_bool("JUNO_V2_DICTATION_EDITOR", True)
    )
    dictation_editor_deadline_ms: int = field(
        default_factory=lambda: _env_int("JUNO_V2_DICTATION_EDITOR_DEADLINE_MS", 12000)
    )
    # When >0 and the backend uses residency_policy='on_demand', the
    # BackendLifecycleManager defers unload by this many seconds after the
    # last release. If a new acquire arrives within the window, the pending
    # unload is cancelled. This trades a little GPU memory for much lower
    # writer-lane TTFT when a user strings commands together back-to-back.
    # 0.0 preserves the unload-on-every-release legacy behavior.
    # 300s: real dictation sessions pause well over 30s between utterances,
    # so a 30s TTL meant a 3-5s Qwen3-4B cold start on nearly every turn.
    idle_unload_ttl_s: float = 300.0
    # Residency policy used when the lifecycle manager registers the writer
    # backend via the DictationSessionRunner fallback path. The factory path
    # already passes its own value via spec.writer_residency_policy; this
    # field only governs the in-process fallback registration in
    # engine/session.py. Default 'resident' matches the rest of the stack —
    # the writer (Qwen3-4B via MLX) pays a 3-5s cold-start on every reload,
    # so keeping it resident is the correct default for an interactive
    # dictation product.
    residency_policy: str = 'resident'


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default
