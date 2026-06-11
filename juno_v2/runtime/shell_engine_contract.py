"""Loopback contract between the Juno macOS shell and the local engine (workbench).

Bump ``SHELL_ENGINE_PROTOCOL_VERSION`` in lockstep with the Swift constant
``JunoEngineContract.shellEngineProtocolVersion`` whenever broker HTTP
semantics change incompatibly for the shell.
"""

from __future__ import annotations

# Monotonic integer; shell treats mismatch as incompatible (stale or too-new engine).
#
# v2 (2026-04-30, production-grade revamp): compatibility response now
# carries ``runtime_role``, ``instance_id``, ``bundle_id``, and ``pid``.
# The Swift shell refuses to attach unless ``runtime_role`` is the
# canonical production role string below — eliminates the silent
# misattach to ``juno_v2.workbench.server`` on a shared dev port.
#
# v3 (2026-05-03): compatibility response also carries the active
# deployment profile (preview/final/writer backends plus writer
# residency). The Swift shell refuses to attach to a stale engine whose
# runtime contract no longer matches the bundled production profile.
#
# v4 (2026-05-05): the bundled macOS production profile reports the
# active deployment profile, including the writer residency policy, so the
# shell can reject stale packaged engines before broker calls.
#
# v5 (2026-05-13): packaged live preview reports the isolated local HTTP
# preview backend, keeping captions out of the Qwen/runtime MLX lock.
#
# v6 (2026-05-23): packaged production writer residency changed to on-demand
# with TTL reaping so background idle does not keep Qwen pinned.
#
# v7 (2026-06-10): packaged production writer residency changed back to
# resident after the Qwen turn-planner layer moved onto the hot path for
# actions and short structured requests. Shell and engine must agree so
# onboarding/setup-status does not reject a healthy bundled runtime.
SHELL_ENGINE_PROTOCOL_VERSION: int = 7

# Canonical role string the shell requires before attaching to an engine.
# Standalone workbench (the dev tool) reports a different role and is
# rejected by ``JunoBroker.ensureCompatible``.
PRODUCTION_RUNTIME_ROLE: str = "juno_runtime_service"
WORKBENCH_STANDALONE_ROLE: str = "workbench_standalone"

__all__ = [
    "PRODUCTION_RUNTIME_ROLE",
    "SHELL_ENGINE_PROTOCOL_VERSION",
    "WORKBENCH_STANDALONE_ROLE",
]
