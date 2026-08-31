"""Maps JSON-RPC method names to WorkbenchApp handlers.

This module is the Python-side "broker route table" for the UDS
transport. It mirrors — but does not duplicate — the HTTP handler
table in ``juno_v2/workbench/server.py``: both ultimately call the
same ``broker_*`` methods on ``WorkbenchApp``, so adding a handler in
one place is the only place a method's behavior actually changes.

Method-naming convention: dotted, mirroring HTTP route shape minus the
``/api/`` prefix. ``/api/broker/engine/compatibility`` → ``broker.engine.compatibility``.
The Swift client calls ``client.call("broker.engine.compatibility", ...)``;
human readers can map straight back to the HTTP route.

This initial cut wires the **identity + readiness** core that the Swift
shell needs to attach. Full migration of every broker route to UDS is
intentionally a follow-on so each batch can be reviewed and tested
without growing this file unreviewably. Routes still served only over
HTTP keep working for the dev workbench browser UI.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from juno_v2.runtime.local_broker_token import (
    auth_enforcement_enabled,
    verify_request_token,
)
from juno_v2.runtime.uds_jsonrpc import JsonRpcServer

LOGGER = logging.getLogger(__name__)


def _auth_check(token: Optional[str]) -> bool:
    """Wrap ``verify_request_token`` so the server can be constructed
    without the workbench app — useful for tests of the transport."""
    if not auth_enforcement_enabled():
        return True
    if token is None:
        return False
    return verify_request_token(token)


def make_uds_server(socket_path, app, *, enforce_auth: bool = True) -> JsonRpcServer:
    """Build a configured ``JsonRpcServer`` with the production handler
    set wired up. Caller owns ``server.start()`` / ``server.stop()``."""
    server = JsonRpcServer(
        socket_path=socket_path,
        auth_check=_auth_check if enforce_auth else None,
    )
    register_core_handlers(server, app)
    return server


# ---------------------------------------------------------------------------
# Core handler set (Phase 2 initial slice)
# ---------------------------------------------------------------------------


def register_core_handlers(server: JsonRpcServer, app) -> None:
    """Wire the engine identity + warm-state core. The Swift shell needs
    these before any other broker call can succeed (they gate
    ``ensureCompatible`` and the ``Setting up voice engine…`` UI state)."""

    def engine_compatibility(_params: Dict[str, Any], _binary):
        return app.broker_engine_compatibility()

    def engine_healthz(_params: Dict[str, Any], _binary):
        return {
            "ok": True,
            "session_id": app.session_id,
            "warm": app.warm_status(),
            "instance_id": getattr(app, "instance_id", ""),
        }

    def engine_warm_status(_params: Dict[str, Any], _binary):
        return app.warm_status()

    def broker_setup_status(_params: Dict[str, Any], _binary):
        return app.broker_setup_status()

    def broker_settings_get(_params: Dict[str, Any], _binary):
        return app.broker_settings_get()

    def surface_capability(_params: Dict[str, Any], _binary):
        return app.broker_capability()

    def broker_http(params: Dict[str, Any], binary):
        return dispatch_broker_http_like(app, params, binary)

    server.register("broker.http", broker_http)
    server.register("broker.engine.compatibility", engine_compatibility)
    server.register("broker.engine.healthz", engine_healthz)
    server.register("broker.engine.warm_status", engine_warm_status)
    server.register("broker.setup.status", broker_setup_status)
    server.register("broker.settings.get", broker_settings_get)
    server.register("broker.surface.capability", surface_capability)


def _query(params: Dict[str, Any]) -> Dict[str, list[str]]:
    raw = params.get("query")
    if isinstance(raw, dict):
        out: Dict[str, list[str]] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                out[str(key)] = [str(v) for v in value]
            elif value is not None:
                out[str(key)] = [str(value)]
        return out
    if isinstance(raw, str):
        return parse_qs(raw)
    return {}


def _payload(params: Dict[str, Any]) -> Dict[str, Any]:
    raw = params.get("payload")
    return raw if isinstance(raw, dict) else {}


def _normalize_path(raw: Any) -> str:
    path = str(raw or "").split("?", 1)[0].rstrip("/") or "/"
    if path == "/":
        return path
    return path if path.startswith("/") else f"/{path}"


def dispatch_broker_http_like(app, params: Dict[str, Any], binary):
    """Dispatch a broker request described as method/path/query/payload.

    The Swift shell uses this as a narrow compatibility bridge while the
    production transport moves from HTTP/TCP to JSON-RPC/UDS. Behavior
    mirrors ``WorkbenchRequestHandler`` by calling the same ``WorkbenchApp``
    methods, but avoids HTTP framing and loopback ports entirely.
    """
    method = str(params.get("http_method") or params.get("method") or "GET").upper()
    path = _normalize_path(params.get("path"))
    qs = _query(params)
    payload = _payload(params)

    if method == "GET":
        if path == "/healthz" or path == "healthz":
            return {"ok": True, "session_id": app.session_id, "warm": app.warm_status()}
        if path == "/api/runtime" or path == "api/runtime":
            return app.runtime_snapshot()
        if path == "/api/state" or path == "api/state":
            # The macOS shell polls this during streaming to drive HUD
            # text and pending-commit state. Without an explicit route
            # the UDS dispatcher falls through to the "unknown route"
            # branch, which floods bundled-engine.log (~thousands of
            # lines/minute) and buries every real diagnostic.
            return app.state()
        get_routes = {
            "/api/broker/engine/compatibility": app.broker_engine_compatibility,
            "/api/broker/personalization/summary": app.broker_personalization_summary,
            "/api/broker/personalization/user_profile": app.broker_user_profile_get,
            "/api/broker/settings": app.broker_settings_get,
            "/api/broker/privacy/context_settings": app.broker_privacy_context_settings_get,
            "/api/broker/privacy/app_overrides": app.broker_privacy_app_overrides_get,
            "/api/broker/writer/warm": app.broker_writer_warm,
            "/api/broker/preview/warm": app.broker_preview_warm,
            "/api/broker/model/routes": app.broker_model_routes,
            "/api/broker/stats/summary": app.broker_stats_summary,
            "/api/broker/storage/stats": app.broker_storage_stats,
            "/api/broker/runtime/backends": app.broker_runtime_backends,
            "/api/broker/surface/active": app.broker_surface_active,
            "/api/broker/modes/builtin": app.broker_modes_builtin_list,
            "/api/broker/modes/custom": app.broker_modes_custom_list,
            "/api/broker/modes/current": app.broker_modes_current,
            "/api/broker/surface_presets/user": app.broker_surface_presets_user,
            "/api/broker/surface_presets/merged": app.broker_surface_presets_merged,
            "/api/broker/transforms/builtin": app.broker_transforms_builtin_list,
            "/api/broker/transforms/custom": app.broker_transforms_custom_list,
            "/api/broker/recovery/paste_last": app.broker_paste_last,
            "/api/broker/recovery/history": app.broker_recovery_history,
            "/api/broker/surface/capability": app.broker_capability,
            "/api/broker/memory/snapshot": app.broker_memory_snapshot,
            "/api/broker/memory/vocab": app.broker_memory_vocab_list,
            "/api/broker/memory/replacement": app.broker_memory_replacement_list,
            "/api/broker/memory/snippet": app.broker_memory_snippet_list,
            "/api/broker/memory/correction": app.broker_memory_correction_list,
            "/api/broker/setup/status": app.broker_setup_status,
        }
        handler = get_routes.get(path)
        if handler is not None:
            return handler()
        if path == "/api/broker/export/data.zip":
            return {"ok": True, "content_type": "application/zip"}, app.broker_export_data_zip_bytes()
        if path.startswith("/api/broker/audio/") and path.endswith("/replay"):
            uid = path.split("/api/broker/audio/", 1)[1].rsplit("/replay", 1)[0].strip("/")
            data = app.broker_audio_replay_bytes(uid)
            if data:
                return {"ok": True, "content_type": "audio/wav"}, data
            return {"ok": False, "error": "not_found", "error_code": "not_found"}
        if path == "/api/broker/recovery/replay":
            uid = (qs.get("utterance_id") or [""])[0]
            route = (qs.get("route") or [None])[0]
            return app.broker_replay_utterance(uid, route=route)
        if path == "/api/broker/history":
            try:
                limit = int((qs.get("limit") or ["50"])[0])
            except ValueError:
                limit = 50
            cursor: int | None = None
            raw_cursor = (qs.get("before_updated_at_ms") or [""])[0]
            if raw_cursor:
                try:
                    cursor = int(raw_cursor)
                    if cursor <= 0:
                        cursor = None
                except ValueError:
                    cursor = None
            return app.broker_utterance_history(
                limit=limit,
                before_updated_at_ms=cursor,
                test_run_id=(qs.get("test_run_id") or [None])[0],
                test_case_id=(qs.get("test_case_id") or [None])[0],
            )
    if method == "POST":
        if path == "/api/broker/dictation/ingest_wav":
            wav_bytes = binary or b""
            host_hints = None
            if isinstance(payload.get("host_hints"), dict):
                from juno_core_v3.contracts.resource_hints import HostResourceHints

                host_hints = HostResourceHints.from_dict(payload["host_hints"])
            return app.broker_dictation_transcribe(
                wav_bytes,
                language=payload.get("language") or (qs.get("language") or [None])[0],
                app_bundle_id=payload.get("app_bundle_id") or (qs.get("app_bundle_id") or [None])[0],
                window_title_hint=payload.get("window_title_hint")
                or payload.get("window_title")
                or (qs.get("window_title_hint") or qs.get("window_title") or [None])[0],
                utterance_id=payload.get("utterance_id") or (qs.get("utterance_id") or [None])[0],
                surface_id=payload.get("surface_id"),
                host_hints=host_hints,
                frozen_context=payload.get("frozen_context") if isinstance(payload.get("frozen_context"), dict) else None,
                transcript_stage=payload.get("transcript_stage"),
                session_context_tape=payload.get("session_context_tape")
                if isinstance(payload.get("session_context_tape"), (dict, list))
                else None,
                transcript_hint=payload.get("transcript_hint") if isinstance(payload.get("transcript_hint"), str) else None,
                pause_sensitivity_seconds=payload.get("pause_sensitivity_seconds")
                if isinstance(payload.get("pause_sensitivity_seconds"), (int, float))
                else None,
                shell_timeline=payload.get("shell_timeline") if isinstance(payload.get("shell_timeline"), dict) else None,
                test_run_id=payload.get("test_run_id"),
                test_case_id=payload.get("test_case_id"),
            )
        if path == "/api/broker/dictation/live_correct":
            return app.broker_dictation_live_correct(payload)
        post_routes = {
            "/api/broker/session/start": app.broker_start_session,
            "/api/broker/session/transform": app.broker_transform,
            "/api/broker/shell/home_greeting": app.broker_shell_home_greeting,
            "/api/broker/modes/manual/set": app.broker_modes_manual_set,
            "/api/broker/dictation/replay_all_finals": app.broker_replay_all_finals,
            "/api/broker/runtime/swap_final": app.broker_runtime_swap_final,
            "/api/broker/modes/custom/upsert": app.broker_modes_custom_set,
            "/api/broker/modes/custom/delete": app.broker_modes_custom_delete,
            "/api/broker/modes/custom/activate": app.broker_modes_custom_activate,
            "/api/broker/surface_presets/upsert": app.broker_surface_presets_upsert,
            "/api/broker/surface_presets/delete": app.broker_surface_presets_delete,
            "/api/broker/surface/editing_profile": app.broker_surface_editing_profile,
            "/api/broker/transforms/custom/upsert": app.broker_transforms_custom_upsert,
            "/api/broker/transforms/custom/delete": app.broker_transforms_custom_delete,
            "/api/broker/insertion/committed": app.broker_insertion_committed,
            "/api/broker/learning/observe_correction": app.broker_observe_correction,
            "/api/broker/personalization/user_profile": app.broker_user_profile_set,
            "/api/broker/settings/retention": app.broker_settings_set_retention,
            "/api/broker/settings/language_environment": app.broker_language_environment_set,
            "/api/broker/privacy/context_settings": app.broker_privacy_context_settings_set,
            "/api/broker/privacy/app_overrides": app.broker_privacy_app_overrides_set,
            "/api/broker/memory/vocab": app.broker_memory_vocab_upsert,
            "/api/broker/memory/vocab/remove": app.broker_memory_vocab_remove,
            "/api/broker/memory/replacement": app.broker_memory_replacement_upsert,
            "/api/broker/memory/replacement/remove": app.broker_memory_replacement_remove,
            "/api/broker/memory/snippet": app.broker_memory_snippet_upsert,
            "/api/broker/memory/snippet/remove": app.broker_memory_snippet_remove,
            "/api/broker/memory/correction/remove": app.broker_memory_correction_remove,
            "/api/broker/memory/clear_all": app.broker_memory_clear_all,
            "/api/broker/dictation/preview/chunk": app.broker_dictation_preview_chunk,
            "/api/broker/writer/extract": app.broker_writer_extract,
        }
        if path == "/api/broker/recovery/ingest":
            return app.broker_ingest_recovery()
        if path == "/api/broker/recovery/retry_append":
            return app.broker_retry_append()
        if path == "/api/broker/recovery/audio/delete":
            return app.broker_delete_replay_audio(str(payload.get("utterance_id") or ""))
        if path == "/api/broker/modes/manual/clear":
            return app.broker_modes_manual_clear()
        if path == "/api/broker/setup/install":
            return app.broker_setup_install(payload, force=False)
        if path == "/api/broker/setup/repair":
            return app.broker_setup_install(payload, force=True)
        if path == "/api/broker/storage/audio/prune_all":
            return app.broker_storage_prune_all_audio()
        if path == "/api/broker/history/clear_all":
            return app.broker_history_clear_all()
        if path == "/api/broker/history/cancel_draft":
            return app.broker_history_cancel_draft(payload)
        if path == "/api/broker/history/reprocess":
            return app.broker_history_reprocess(
                str(payload.get("utterance_id") or ""),
                str(payload.get("mode_name") or ""),
                is_custom=bool(payload.get("is_custom") or False),
            )
        if path == "/api/broker/history/insert_again":
            return app.broker_history_insert_again(payload)
        if path.startswith("/api/broker/history/") and path.endswith("/actions"):
            uid = path[len("/api/broker/history/"): -len("/actions")]
            return app.broker_history_update_actions(uid, payload)
        if path == "/api/broker/settings/writer":
            return app.broker_settings_set_writer(bool(payload.get("enabled")))
        if path == "/api/broker/settings/itn":
            return app.broker_settings_set_itn(bool(payload.get("enabled")))
        if path == "/api/broker/settings/audio":
            return app.broker_settings_set_audio_save(bool(payload.get("enabled")))
        if path == "/api/broker/settings/live_caption":
            return app.broker_settings_set_live_caption(bool(payload.get("enabled")))
        if path == "/api/broker/retention/run_cleanup":
            return app.broker_retention_run_cleanup()
        handler = post_routes.get(path)
        if handler is not None:
            return handler(payload)

    if method == "DELETE":
        if path.startswith("/api/broker/history/"):
            return app.broker_history_delete(path.split("/api/broker/history/", 1)[1].strip("/"))
    LOGGER.warning("unknown UDS broker route method=%s path=%s", method, path)
    return {"ok": False, "error": "not_found", "error_code": "not_found", "path": path}


__all__ = [
    "make_uds_server",
    "register_core_handlers",
    "dispatch_broker_http_like",
]
