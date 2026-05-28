from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from juno_v2.contracts.context import TypedContextBundle

from juno_v2.contracts.tracing import TraceKind
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.workbench.store import WorkbenchStore

from juno_v2.writer.backends.base import WriterBackend

from juno_core_v3.broker.runners import TransformRequest, TransformRunner
from juno_core_v3.broker.session_manager import SessionManager
from juno_core_v3.context.plane import ContextPlane, ContextPlaneConfig
from juno_core_v3.contracts.resource_hints import HostResourceHints
from juno_core_v3.contracts.session import BrokerSession, SessionKind, UserIntentSignals
from juno_core_v3.model_registry.defaults import build_default_registry
from juno_core_v3.policy.broker_policy import BrokerPolicyEngine
from juno_core_v3.policy.surface_gate import SupportLevel
from juno_core_v3.model_registry.routing import RouteChooser, RouteRequest
from juno_core_v3.model_registry.contracts import ModelSlot, SurfaceClass
from juno_core_v3.recovery.session import RecoverySession


@dataclass
class BrokerFacade:
    """Thin-surface broker API for workbench-facing routes.

    Uses the same ``session_id`` as the workbench trace session so recovery ledger
    and traces correlate.
    """

    workbench_session_id: str
    recorder: TraceRecorder
    log_dir: Path
    session_manager: SessionManager = field(default_factory=SessionManager)
    policy_engine: BrokerPolicyEngine = field(default_factory=BrokerPolicyEngine)
    route_chooser: RouteChooser = field(default_factory=lambda: RouteChooser(build_default_registry()))
    context_plane: ContextPlane = field(default_factory=lambda: ContextPlane(ContextPlaneConfig()))
    # ``writer_backend`` is optional. When wired (the runtime service
    # and registry launcher pass the engine's loaded writer backend),
    # free-form Transform hints like "make this shorter" hit the model
    # instead of degrading to a deterministic polish. Without it we
    # still honour deterministic hints (polish/bullets/numbered/upper/
    # lower/title) and mark free-form results as ``degraded=True`` so
    # surfaces can explain to the user.
    writer_backend: WriterBackend | None = None
    broker_session: BrokerSession | None = None
    recovery: RecoverySession | None = None
    transform_runner: TransformRunner = field(default_factory=TransformRunner)

    def __post_init__(self) -> None:
        rroot = Path(self.log_dir) / "recovery"
        self.recovery = RecoverySession(
            broker_session_id=self.workbench_session_id,
            recovery_root=rroot,
            recorder=self.recorder,
        )
        # When the caller handed us a writer backend (production runtime
        # service + registry launcher do), upgrade the transform runner
        # so free-form hints like "make this shorter" actually hit the
        # model instead of silently degrading to deterministic polish.
        # Callers that pre-built their own ``transform_runner`` (tests)
        # keep the one they injected — we only overwrite when the
        # default factory produced a backend-less runner.
        if self.writer_backend is not None and self.transform_runner.writer_backend is None:
            self.transform_runner = TransformRunner(writer_backend=self.writer_backend)

    def start_session(self, signals: UserIntentSignals | None = None) -> dict:
        """Classify and bind broker session to the workbench session id."""
        sig = signals or UserIntentSignals()
        # Prefer caller-supplied hints (shell already probed the host);
        # fall back to env override (tests/ops) and then a live system
        # probe via the juno-host helper. All three paths are
        # best-effort: a missing signal leaves the broker in its
        # default unconstrained policy, which is safer than blocking
        # dictation on a transient probe failure.
        if sig.host_hints is None:
            resolved = HostResourceHints.resolve()
            if resolved is not None:
                sig = replace(sig, host_hints=resolved)
        kind = self.session_manager.classify_only(sig)
        decision = self.policy_engine.decide(sig, kind)
        if not decision.ok:
            payload = {"ok": False, "error": decision.blocked_reason or "policy_denied", "error_code": decision.blocked_reason or "policy_denied", "policy": decision.to_dict()}
            self.recorder.record(TraceKind.SYSTEM, "broker_session_policy_denied", payload)
            return payload
        meta = {
            "signals": sig.to_dict(),
            "surface": "workbench",
            "policy": decision.to_dict(),
        }
        self.broker_session = BrokerSession.new(
            kind,
            metadata=meta,
            session_id=self.workbench_session_id,
        )
        self.recorder.record(TraceKind.SYSTEM, "broker_session_started", self.broker_session.to_dict())
        return {"ok": True, "broker_session": self.broker_session.to_dict(), "policy": decision.to_dict()}

    def model_routes(self) -> dict:
        preview = self.route_chooser.choose(RouteRequest(slot=ModelSlot.PREVIEW_ASR, surface=SurfaceClass.DESKTOP))
        final = self.route_chooser.choose(RouteRequest(slot=ModelSlot.FINAL_ASR, surface=SurfaceClass.DESKTOP))
        writer = self.route_chooser.choose(RouteRequest(slot=ModelSlot.WRITER, surface=SurfaceClass.DESKTOP))
        out = {
            "preview": None if preview.chosen is None else preview.chosen.to_dict(),
            "final": None if final.chosen is None else final.chosen.to_dict(),
            "writer": None if writer.chosen is None else writer.chosen.to_dict(),
            "reasons": {"preview": preview.reason, "final": final.reason, "writer": writer.reason},
        }
        return out

    def ingest_recovery(self, store: WorkbenchStore) -> dict:
        assert self.recovery is not None
        self.recovery.ingest_from_workbench_snapshot(store.snapshot())
        return {"ok": True}

    def paste_last(self) -> dict:
        assert self.recovery is not None
        text = self.recovery.paste_last_transcript()
        return {"ok": True, "text": text}

    def retry_append(self, store: WorkbenchStore) -> dict:
        assert self.recovery is not None
        snap = self.recovery.retry_last_commit_append(store)
        return {"ok": True, "snapshot": snap}

    def recovery_history(self) -> dict:
        assert self.recovery is not None
        return {"ok": True, "entries": self.recovery.history()}

    # ------------------------------------------------------------------ #
    # Session enforcement (Phase A)
    # ------------------------------------------------------------------ #

    def _signals_for_kind(self, kind: SessionKind, signals: UserIntentSignals) -> UserIntentSignals:
        """Normalize caller-provided signals for an explicit action kind.

        This is the only place we should stamp explicit_* flags for the broker
        action methods.  It avoids ambiguous mixed signals like
        explicit_transform=True + explicit_insert=True.
        """
        return UserIntentSignals(
            has_selected_text=signals.has_selected_text,
            explicit_transform=(kind == SessionKind.TRANSFORM),
            explicit_insert=(kind == SessionKind.INSERT),
            surface_id=signals.surface_id,
            host_hints=signals.host_hints,
        )

    def _session_error(self, *, code: str, kind: SessionKind, detail: dict | None = None) -> dict:
        payload = {
            "ok": False,
            "error": code,
            "error_code": code,
            "required_kind": kind.value,
        }
        if detail:
            payload.update(detail)
        self.recorder.record(TraceKind.SYSTEM, code, payload)
        return payload

    def _ensure_session(self, kind: SessionKind, signals: UserIntentSignals | None = None) -> dict | None:
        """Guarantee an active broker session of *kind* exists.

        Important policy change: we no longer invent a GOLD ``workbench_dev``
        surface when the caller omitted surface metadata.  If the runtime wants
        an auto-started session it must supply explicit signals so surface tiers
        and resource gates are evaluated against the real caller.
        """
        if signals is None:
            if self.broker_session is None:
                return self._session_error(code="broker_session_required", kind=kind)
            if self.broker_session.kind != kind:
                return self._session_error(
                    code="broker_session_kind_mismatch",
                    kind=kind,
                    detail={"actual_kind": self.broker_session.kind.value},
                )
            return None

        normalized = self._signals_for_kind(kind, signals)
        result = self.start_session(normalized)
        if not result.get("ok"):
            return result
        if self.broker_session is None or self.broker_session.kind != kind:
            return self._session_error(code="broker_session_start_failed", kind=kind)
        return None

    def ensure_session(self, *, kind: SessionKind, signals: UserIntentSignals | None = None) -> dict:
        """Public wrapper so surfaces / server code can force session truth.

        Returns a standard ``ok`` envelope with the active broker_session and
        policy when successful.
        """
        err = self._ensure_session(kind, signals=signals)
        if err is not None:
            return err
        assert self.broker_session is not None
        return {
            "ok": True,
            "broker_session": self.broker_session.to_dict(),
            "policy": dict(self.broker_session.metadata.get("policy", {})),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _current_policy(self) -> dict:
        if self.broker_session is None:
            return {}
        return dict(self.broker_session.metadata.get("policy", {}))

    def _current_support_level(self) -> SupportLevel:
        """Return the support level of the active session, or RED if none."""
        policy = self._current_policy()
        raw = policy.get("support_level", SupportLevel.RED.value)
        try:
            return SupportLevel(raw)
        except ValueError:
            return SupportLevel.RED

    @staticmethod
    def _policy_degradation_reason(policy: dict) -> str | None:
        degraded = list(policy.get("degraded_features") or [])
        if not degraded:
            return None
        for preferred in (
            "terminal_safe_transform",
            "writer_lane_budget_tight",
            "reduce_optional_lanes",
        ):
            if preferred in degraded:
                return preferred
        return str(degraded[0])

    # ------------------------------------------------------------------ #
    # Transform session class
    # ------------------------------------------------------------------ #

    def run_transform(
        self,
        *,
        selected_text: str,
        hint: str,
        app_category: str | None = None,
        metadata: dict | None = None,
        signals: UserIntentSignals | None = None,
    ) -> dict:
        """Run a Transform session: rewrite selected text given a hint."""
        err = self._ensure_session(SessionKind.TRANSFORM, signals=signals)
        if err is not None:
            return err
        if not selected_text.strip():
            return {"ok": False, "error": "empty_selection", "error_code": "transform_requires_selection"}
        policy = self._current_policy()
        level = self._current_support_level()
        degraded_features = list(policy.get("degraded_features") or [])
        force_det = level == SupportLevel.YELLOW or "terminal_safe_transform" in degraded_features
        md = dict(metadata or {})
        req = TransformRequest(
            selected_text=selected_text,
            hint=hint,
            app_category=app_category,
            metadata=md,
            force_deterministic=force_det,
        )
        result = self.transform_runner.run(req)
        self.recorder.record(TraceKind.SYSTEM, "broker_transform", result.to_dict())
        if self.recovery is not None:
            self.recovery.record_transform(
                selected_text=selected_text,
                replacement_text=result.replacement_text,
                hint=hint,
                deterministic=result.deterministic,
                degraded=result.degraded,
                session_id=self.workbench_session_id,
            )
        resp = {
            "ok": True,
            "result": result.to_dict(),
            "support_level": level.value,
            "degraded": bool(degraded_features) or bool(result.degraded),
        }
        reason = self._policy_degradation_reason(policy)
        if reason is not None:
            resp["degraded_reason"] = reason
            resp["degraded_features"] = degraded_features
        elif result.degraded:
            resp["degraded_reason"] = result.metadata.get("reason") or "transform_degraded"
        return resp

    # ------------------------------------------------------------------ #
    # Context inspector (Phase 02)
    # ------------------------------------------------------------------ #

    def context_inspect(self, bundle: "TypedContextBundle | None" = None) -> dict:
        """Build a ContextPacket from *bundle* (or an empty bundle) and return
        the packet fields alongside provenance, suppression, and degradation
        summaries.  This is a diagnostic / inspector endpoint — it does not
        change any state.
        """
        from juno_v2.contracts.context import TypedContextBundle as _TCB

        b = bundle if bundle is not None else _TCB()
        pkt = self.context_plane.build_from_typed_bundle(b, surface_id="inspector")
        suppression = pkt.metadata.get("suppression", "none")
        suppressed_fields = pkt.metadata.get("suppressed_fields", [])
        degradation = pkt.metadata.get("degradation", "none")
        screen_fallback_policy = pkt.metadata.get("screen_fallback_policy", "disabled")
        ordered_sources = pkt.metadata.get("ordered_sources", [])
        recent_clipboard = pkt.metadata.get("recent_clipboard", [])
        budgets = pkt.metadata.get("budgets", {})
        return {
            "ok": True,
            "packet": pkt.to_dict(),
            "summary": {
                "suppression": suppression,
                "suppressed_fields": suppressed_fields,
                "degradation": degradation,
                "screen_fallback_policy": screen_fallback_policy,
                "ordered_sources": ordered_sources,
                "recent_clipboard_count": len(recent_clipboard),
                "budgets": budgets,
                "truncation_applied": dict(pkt.truncation_applied),
            },
        }
