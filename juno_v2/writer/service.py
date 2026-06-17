from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from juno_v2.commands.resolver import focused_text_before_tail, resolve_command_target
from juno_v2.commands.semantic import interpret_semantic_command
from juno_v2.contracts.commands import CommandTargetClass
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.modes import ModePolicy, ModeSelection
from juno_v2.contracts.tracing import TraceKind
from juno_v2.contracts.workbench import ClientSelection, CommitMode
from juno_v2.contracts.writer import (
    WriterActionKind,
    WriterIntent,
    WriterIntentKind,
    WriterMode,
    WriterOutcome,
    WriterTransformRequest,
    WriterTransformResult,
)
from juno_v2.memory.entity_policy import common_english_single_word
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.memory.term_policy import learned_term_allowed
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.turn_plan import (
    TurnPlanPacket,
    TurnPlanResult,
    TurnPlanner,
    fallback_structural_turn_plan,
    render_turn_plan,
    validate_turn_plan,
)
from juno_v2.turn_plan.planner import structural_instruction_present
from juno_v2.writer.dictation_editor import (
    DICTATION_EDIT_TASK,
    FILLER_STRIP_MODES,
    apply_edit_script,
    build_editor_suffix,
    capitalize_sentence_starts,
    parse_edit_script,
    strip_hesitation_fillers,
)
from juno_v2.turn_plan.validators import span_present
from juno_v2.writer.backends.base import WriterBackend
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.deterministic import (
    AppCategory,
    expand_snippets,
    render_bullets,
    render_explicit_bullet_list_command,
    render_lowercase,
    render_natural_bullet_list_dictation,
    render_numbered,
    render_title_case,
    render_uppercase,
)
from juno_v2.context.compiler import FormattingPacket
from juno_v2.writer.final_formatter import (
    FinalFormatter,
    apply_commit_boundary_rules,
    apply_grammar_postpass,
    should_run_final_formatting,
)
from juno_v2.writer.parser import WriterIntentParser
from juno_v2.writer.state import WriterState


_MEMORY_COMMAND_TERMS = {
    "juno",
    "remember",
    "teach",
    "teach juno",
    "term",
    "terms",
    "word",
    "words",
}
_RECENT_TRANSFORM_TARGET_RE = re.compile(
    r"\b(?:that|the\s+last\s+(?:sentence|line|paragraph|answer|thing|paste|text))\b",
    re.IGNORECASE,
)
_TRANSFORM_VERB_RE = re.compile(
    r"\b(?:make|turn|rewrite|change|convert|summari[sz]e|shorten|simplify|fix|correct|clean\s+up|translate)\b",
    re.IGNORECASE,
)


def _mentions_recent_transform_target(text: str) -> bool:
    current = str(text or "")
    return bool(_RECENT_TRANSFORM_TARGET_RE.search(current) and _TRANSFORM_VERB_RE.search(current))


def _unresolved_correction_cues_present(text: str) -> bool:
    """True when a spoken edit marker survived the deterministic retake pass.

    The pipeline applies unambiguous retakes before the writer runs, so any
    marker still present needs meaning-level judgment. Deterministic list
    rendering must not ship "scratch that" verbatim inside a bullet — the
    dictation editor owns these turns (it resolves the correction AND emits
    the list structure).
    """
    if not text:
        return False
    # Lazy import: juno_core_v3.dictation's package __init__ pulls the
    # pipeline, which imports this module back.
    from juno_core_v3.dictation.self_corrections import MID_UTTERANCE_EDIT_MARKER_RE

    return bool(MID_UTTERANCE_EDIT_MARKER_RE.search(text))


def _replace_commit_fields(target: dict, text: str) -> tuple[str, dict]:
    """Output text + metadata for a TRANSFORM_COMMIT that replaces a target.

    The shell deletes ``target_text_chars`` characters back from the caret
    before pasting (gated on the target name), so the count must be the
    caret-anchored ``delete_chars`` when the target provides one — the
    stripped text length undercounts whenever whitespace sits between the
    target and the caret. Non-empty output re-appends that whitespace so a
    "pressed Enter twice, then asked for a rewrite" caret keeps its blank
    line.
    """
    out = (text or "").strip()
    if out:
        out += str(target.get("trailing_text") or "")
    meta = {
        "target": target.get("target"),
        "target_text_chars": int(target.get("delete_chars") or len(target.get("text") or "")),
    }
    return out, meta


@dataclass(slots=True)
class WriterService:
    config: WriterConfig
    recorder: TraceRecorder
    backend: WriterBackend | None = None
    parser: WriterIntentParser = field(default_factory=WriterIntentParser)
    state: WriterState = field(default_factory=WriterState)
    backend_acquire: Callable[[], object] | None = None
    backend_release: Callable[[], object] | None = None

    def rewrite(self, req: WriterTransformRequest) -> WriterTransformResult:
        return self._rewrite_with_backend(req)

    def extract_memory_candidates(self, *, text: str, kind: str, limit: int = 6) -> list[dict] | None:
        if self.backend is None:
            return None
        extractor = getattr(self.backend, "extract_memory_candidates", None)
        if not callable(extractor):
            return None
        acquired = False
        if self.backend_acquire is not None:
            self.backend_acquire()
            acquired = True
        try:
            return extractor(text=text, kind=kind, limit=limit)
        finally:
            if acquired and self.backend_release is not None:
                self.backend_release()

    def _rewrite_with_backend(self, req: WriterTransformRequest) -> WriterTransformResult:
        if self.backend is None:
            raise RuntimeError("writer backend unavailable")
        acquired = False
        if self.backend_acquire is not None:
            self.backend_acquire()
            acquired = True
        try:
            return self.backend.rewrite(req)
        finally:
            if acquired and self.backend_release is not None:
                self.backend_release()

    def _model_rewrite_allowed_for_target(
        self,
        pol: ModePolicy | None,
        target_class: CommandTargetClass,
    ) -> bool:
        if pol is None or self.state.mode == WriterMode.COMMAND_MODE:
            return True
        if target_class == CommandTargetClass.SELECTED_TEXT:
            return bool(pol.allow_selection_commands)
        if target_class in (CommandTargetClass.RECENT_COMMIT, CommandTargetClass.FOCUSED_TEXT):
            return bool(pol.allow_recent_target_commands)
        return bool(pol.allow_model_insert_rewrite)

    def _model_rewrite_blocked_noop(
        self,
        utterance_id: str,
        intent,
        target_class: CommandTargetClass,
    ) -> WriterOutcome:
        return self._noop(
            utterance_id,
            'mode_disallows_model_semantics',
            intent.kind.value,
            extra={'target_class': target_class.value},
        )

    def warm(self) -> None:
        if self.backend is not None:
            self.backend.warm()

    def plan_turn(
        self,
        *,
        utterance_id: str,
        final_text: str,
        raw_text: str,
        context: TypedContextBundle,
        memory_store: JsonMemoryStore | None,
        memory_snapshot: MemorySnapshot | None,
        memory_packet: dict | None = None,
        language_hint: str | None = None,
        mode_policy: ModePolicy | None = None,
        mode_selection: ModeSelection | None = None,
        partial_text: str | None = None,
        writer_tone_addon: str | None = None,
        wake_verified: bool = False,
        now_iso: str | None = None,
    ) -> TurnPlanResult | None:
        if not self.config.enable_turn_planner or self.backend is None or not self.config.enable_model_transforms:
            return None
        if self._skip_model_planner_for_dictation(
            final_text=final_text,
            context=context,
            wake_verified=wake_verified,
        ):
            structural = self._structural_turn_plan_for_dictation(
                utterance_id=utterance_id,
                final_text=final_text,
                context=context,
            )
            if structural is not None:
                return structural
            if not structural_instruction_present(final_text):
                self.recorder.record(
                    TraceKind.WRITER,
                    "turn_plan_skipped",
                    {
                        "utterance_id": utterance_id,
                        "reason": "long_dictation_without_structure",
                        "words": _approx_word_count(final_text),
                    },
                )
                return None
            # Explicit spoken structure request ("note down five points …")
            # that the deterministic itemizer could not split — that narrow
            # case still earns a model decode.
        self._apply_runtime_mode(
            mode_policy=mode_policy,
            mode_selection=mode_selection,
            writer_tone_addon=writer_tone_addon,
        )
        packet = TurnPlanPacket(
            utterance_id=utterance_id,
            final_text=final_text,
            raw_text=raw_text,
            context=context,
            mode_policy=self.state.mode_policy,
            mode_selection=self.state.mode_selection,
            memory_store=memory_store,
            memory_snapshot=memory_snapshot,
            memory_packet=memory_packet,
            language_hint=language_hint,
            partial_text=partial_text,
            writer_tone_addon=self.state.writer_tone_addon,
            wake_verified=wake_verified,
            now_iso=now_iso,
        )
        planner = TurnPlanner(self)
        result = planner.plan(packet)
        validation_after_plan = None
        if result.ok and isinstance(result.plan, dict):
            validation_after_plan = validate_turn_plan(result.plan, source_text=final_text, context=context)
        if result.plan is None or result.status == "invalid_json" or (
            validation_after_plan is not None and not validation_after_plan.ok
        ):
            result = planner.repair(
                packet,
                result,
                validation_errors=[] if validation_after_plan is None else list(validation_after_plan.errors),
                validation_warnings=[] if validation_after_plan is None else list(validation_after_plan.warnings),
            )
            validation_after_plan = None
            if result.ok and isinstance(result.plan, dict):
                validation_after_plan = validate_turn_plan(result.plan, source_text=final_text, context=context)
        if validation_after_plan is not None and not validation_after_plan.ok:
            fallback = fallback_structural_turn_plan(final_text)
            if fallback is not None:
                fallback_validation = validate_turn_plan(fallback, source_text=final_text, context=context)
                if fallback_validation.ok:
                    result = TurnPlanResult(
                        plan=fallback,
                        status="ok",
                        backend_name=result.backend_name,
                        decode_ms=result.decode_ms,
                        raw_output=result.raw_output,
                        errors=[],
                        repair_attempted=True,
                        repair_status="fallback",
                        initial_status=result.initial_status or result.status,
                        initial_errors=list(result.initial_errors or result.errors),
                        validation_errors_before_repair=[
                            *result.validation_errors_before_repair,
                            *validation_after_plan.errors,
                        ],
                        validation_warnings_before_repair=[
                            *result.validation_warnings_before_repair,
                            *validation_after_plan.warnings,
                        ],
                        normalization_notes=[
                            *result.normalization_notes,
                            "structural_render_fallback_after_validation_failure",
                        ],
                    )
                    validation_after_plan = fallback_validation
        payload: dict[str, Any] = {
            "utterance_id": utterance_id,
            "status": result.status,
            "backend": result.backend_name,
            "decode_ms": result.decode_ms,
            "errors": list(result.errors),
            "repair_attempted": result.repair_attempted,
            "repair_status": result.repair_status,
            "initial_status": result.initial_status,
            "initial_errors": list(result.initial_errors),
            "validation_errors_before_repair": list(result.validation_errors_before_repair),
            "validation_warnings_before_repair": list(result.validation_warnings_before_repair),
            "normalization_notes": list(result.normalization_notes),
            "raw_output_chars": len(result.raw_output or ""),
        }
        if result.repair_attempted or not result.ok:
            payload["raw_output_preview"] = (result.raw_output or "")[:800]
        if validation_after_plan is not None:
            payload.update({
                "validation_ok": validation_after_plan.ok,
                "validation_errors": list(validation_after_plan.errors),
                "validation_warnings": list(validation_after_plan.warnings),
            })
        if isinstance(result.plan, dict):
            payload.update({
                "utterance_kind": result.plan.get("utterance_kind"),
                "render_kind": (result.plan.get("render_plan") or {}).get("render_kind")
                if isinstance(result.plan.get("render_plan"), dict)
                else None,
                "action_count": len(result.plan.get("actions") or [])
                if isinstance(result.plan.get("actions"), list)
                else 0,
            })
        self.recorder.record(TraceKind.WRITER, "turn_plan_generated", payload)
        return result

    def _skip_model_planner_for_dictation(
        self,
        *,
        final_text: str,
        context: TypedContextBundle,
        wake_verified: bool,
    ) -> bool:
        """Decide whether this utterance is long plain dictation.

        Wake-verified turns (actions), short utterances (where transforms,
        memory mutations, and message rendering live), and selection-anchored
        commands keep the model planner. Long plain dictation skips it: the
        renderer never uses planner text for plain dictation, and the decode
        cost (8-28s observed on Qwen3-4B) lands on the paste critical path.
        """
        if wake_verified or self.config.turn_plan_dictation_enabled:
            return False
        limit = int(self.config.turn_plan_max_dictation_words or 0)
        if limit <= 0:
            return False
        if _approx_word_count(final_text) <= limit:
            return False
        if (getattr(context, "selected_text", "") or "").strip():
            return False
        return True

    def _structural_turn_plan_for_dictation(
        self,
        *,
        utterance_id: str,
        final_text: str,
        context: TypedContextBundle,
    ) -> TurnPlanResult | None:
        """Zero-model turn plan for non-wake utterances.

        Keeps explicit spoken-structure rendering (lists / checklists) alive
        without paying a multi-second model decode on the paste path. Returns
        ``None`` when the deterministic itemizer finds nothing to render,
        which sends the writer down its existing deterministic/legacy lanes.
        """
        if _is_no_touch_context(context):
            return None
        fallback = fallback_structural_turn_plan(final_text)
        if fallback is None:
            return None
        if self._defer_list_to_editor(utterance_id, final_text, lane="structural_turn_plan"):
            return None
        validation = validate_turn_plan(fallback, source_text=final_text, context=context)
        if not validation.ok:
            return None
        result = TurnPlanResult(
            plan=fallback,
            status="ok",
            backend_name="deterministic_structural",
            decode_ms=0.0,
            normalization_notes=["model_planner_wake_gated"],
        )
        self.recorder.record(
            TraceKind.WRITER,
            "turn_plan_generated",
            {
                "utterance_id": utterance_id,
                "status": result.status,
                "backend": result.backend_name,
                "decode_ms": 0.0,
                "errors": [],
                "repair_attempted": False,
                "repair_status": None,
                "normalization_notes": list(result.normalization_notes),
                "validation_ok": True,
                "validation_errors": [],
                "validation_warnings": list(validation.warnings),
                "utterance_kind": fallback.get("utterance_kind"),
                "render_kind": (fallback.get("render_plan") or {}).get("render_kind")
                if isinstance(fallback.get("render_plan"), dict)
                else None,
                "action_count": 0,
            },
        )
        return result

    def _pre_editor_route_match(
        self,
        *,
        utterance_id: str,
        final_text: str,
        raw_text: str,
        memory_store: JsonMemoryStore | None,
        context: TypedContextBundle,
        language_hint: str | None,
    ) -> bool:
        """Deterministic routes that must win over the dictation editor.

        Commands ("next bullet", "make that shorter"), direct snippet
        inserts, and explicit structure requests are advertised product
        behaviors handled by the parser / snippet / structural lanes below.
        The editor consuming them as prose pasted the literal words
        (production 2026-06-11). Returns True ⇒ skip the editor.
        """
        reason: str | None = None
        selection_present = bool((getattr(context, "selected_text", "") or "").strip())
        for candidate in (final_text, raw_text):
            cleaned = (candidate or "").strip()
            if not cleaned or len(cleaned.split()) > 12:
                continue
            try:
                intent = self.parser.parse(
                    cleaned,
                    language_hint=language_hint,
                    selection_present=selection_present,
                )
            except Exception:  # noqa: BLE001 — probe must never break the turn
                continue
            kind = getattr(intent, "kind", None)
            if kind is not None and kind is not WriterIntentKind.DICTATE:
                reason = f"parser_intent:{getattr(kind, 'value', kind)}"
                break
        if (
            reason is None
            and not _is_no_touch_context(context)
            and structural_instruction_present(final_text)
            # A surviving correction cue means the structural lane will
            # defer anyway — keep the editor in play so it can resolve the
            # correction AND emit the structure in one pass.
            and not _unresolved_correction_cues_present(final_text)
        ):
            reason = "structural_instruction"
        if reason is None:
            try:
                snippet_hit = self._direct_snippet_insert(
                    final_text,
                    memory_store=memory_store,
                    app_category=getattr(context, "app_category", None),
                )
            except Exception:  # noqa: BLE001
                snippet_hit = None
            if snippet_hit is not None:
                reason = "direct_snippet"
        if reason is None:
            return False
        self.recorder.record(
            TraceKind.WRITER,
            "dictation_edit_bypassed",
            {"utterance_id": utterance_id, "reason": reason},
        )
        return True

    def edit_dictation(
        self,
        *,
        utterance_id: str,
        final_text: str,
        raw_text: str,
        context: TypedContextBundle,
        memory_packet: dict | None = None,
        memory_store: Any = None,
    ) -> WriterOutcome | None:
        """Run the cached-prefix dictation editor. None ⇒ deterministic floor."""
        packet = memory_packet or {}
        memory_terms: list[str] = []
        for value in (packet.get("lexicon_terms") or [])[:48]:
            if str(value).strip():
                memory_terms.append(str(value).strip())
        for row in (packet.get("corrections") or [])[:8]:
            if isinstance(row, dict) and str(row.get("corrected") or "").strip():
                memory_terms.append(str(row["corrected"]).strip())
        screen_terms: list = []
        for entity in (getattr(context, "candidate_entities", None) or [])[:16]:
            text = entity.get("text") if isinstance(entity, dict) else entity
            if str(text or "").strip():
                screen_terms.append(str(text).strip())
        # Same decision layer as the ASR prompt and HUD repair: screen terms
        # first, family-deduped memory terms after.
        from juno_v2.memory.bias import _diversify_bias_phrases

        terms = _diversify_bias_phrases(memory_terms, screen_terms=screen_terms, cap=24)
        behavior = str(getattr(self.state.mode_policy, "writer_behavior", "") or "")
        filler_policy = (
            "light cleanup allowed" if "filler" in behavior else "keep the speaker's filler words"
        )
        style_hint = str(getattr(self.state.mode_policy, "prompt_prefix", "") or "") or None
        suffix = build_editor_suffix(
            transcript=final_text,
            app_name=getattr(context, "app_name", None),
            app_category=getattr(context, "app_category", None),
            known_terms=list(dict.fromkeys(terms)),
            filler_policy=filler_policy,
            style_hint=style_hint,
        )
        request = WriterTransformRequest(
            utterance_id=utterance_id,
            instruction="dictation_edit",
            source_text=final_text,
            mode=self.state.mode,
            context_payload={"task": DICTATION_EDIT_TASK, "payload_text": suffix},
            metadata={
                "kind": DICTATION_EDIT_TASK,
                "cache_prefix": True,
                "deadline_ms": int(self.config.dictation_editor_deadline_ms),
                "max_tokens": 352,
            },
        )
        try:
            result = self.backend.rewrite(request)
        except Exception as exc:  # noqa: BLE001 — editor must never break the turn
            self.recorder.record(
                TraceKind.WRITER,
                "dictation_edit_floor",
                {"utterance_id": utterance_id, "reason": f"backend_error:{exc}"},
            )
            return None
        raw_output = str(getattr(result, "text", "") or "")
        script = parse_edit_script(raw_output)
        decode_ms = float(getattr(result, "decode_ms", 0.0) or 0.0)
        if script is None:
            self.recorder.record(
                TraceKind.WRITER,
                "dictation_edit_floor",
                {
                    "utterance_id": utterance_id,
                    "reason": "unparseable_output",
                    "decode_ms": decode_ms,
                    "raw_output_preview": raw_output[:200],
                },
            )
            return None
        if script.verdict == "clean" and not script.has_ops:
            edited, applied = final_text, {"edits": 0, "deletes": 0, "skipped": 0, "struct": None}
        else:
            applied_result = apply_edit_script(final_text, script)
            if applied_result is None:
                self.recorder.record(
                    TraceKind.WRITER,
                    "dictation_edit_floor",
                    {
                        "utterance_id": utterance_id,
                        "reason": "application_rejected",
                        "decode_ms": decode_ms,
                    },
                )
                return None
            edited, applied = applied_result
        self.recorder.record(
            TraceKind.WRITER,
            "dictation_edit_generated",
            {
                "utterance_id": utterance_id,
                "verdict": script.verdict,
                "decode_ms": decode_ms,
                "applied": applied,
                "changed": edited != final_text,
                **{k: v for k, v in (result.metadata or {}).items() if k in (
                    "prompt_cache_hit", "cached_prefix_tokens", "suffix_tokens", "timed_out",
                    "generation_api",
                )},
            },
        )
        structural = applied.get("struct") is not None
        category = str(getattr(context, "app_category", "") or "").lower()
        if category not in {"terminal", "code", "developer_tools"}:
            edited = capitalize_sentence_starts(edited)
            # Mode-gated hesitation cleanup — the final guard after the
            # editor (production 2026-06-11: formal Mail kept "uh").
            mode_name = str(getattr(self.state.mode_policy, "mode_name", "") or "").lower()
            if mode_name in FILLER_STRIP_MODES:
                edited = strip_hesitation_fillers(edited)
        # Expand user snippets on the editor's output. The editor path returns
        # here, before the deterministic expand_snippets step, so without this
        # a saved snippet ("signoff" -> "Best, Juno") never came up whenever the
        # dictation editor was enabled (the production default).
        edited, editor_snippets_expanded, editor_snippet_scopes = self._expand_snippets_for_commit(
            edited, context=context, memory_store=memory_store
        )
        if editor_snippets_expanded:
            self.recorder.record(
                TraceKind.WRITER,
                "writer_snippets_expanded",
                {
                    "utterance_id": utterance_id,
                    "scopes": editor_snippet_scopes,
                    "pipeline": "dictation_editor",
                },
            )
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.PASS_THROUGH_COMMIT,
            output_text=edited,
            learn_from_commit=not structural,
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            deterministic_used=False,
            model_used=True,
            metadata={
                "reason": "dictation_editor",
                "snippet_expanded": editor_snippets_expanded,
                "raw_text": raw_text,
                "input_text": final_text,
                "editor": {"verdict": script.verdict, "applied": applied, "decode_ms": decode_ms},
            },
        )

    def process_transcript(
        self,
        *,
        utterance_id: str,
        final_text: str,
        raw_text: str,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
        memory_store: JsonMemoryStore | None,
        memory_snapshot: MemorySnapshot | None,
        memory_packet: dict | None = None,
        language_hint: str | None = None,
        mode_policy: ModePolicy | None = None,
        mode_selection: ModeSelection | None = None,
        partial_text: str | None = None,
        writer_tone_addon: str | None = None,
        turn_plan_result: TurnPlanResult | None = None,
        wake_verified: bool = False,
        now_iso: str | None = None,
    ) -> WriterOutcome:
        self._apply_runtime_mode(
            mode_policy=mode_policy,
            mode_selection=mode_selection,
            writer_tone_addon=writer_tone_addon,
        )

        explicit_bullet_list = self._explicit_bullet_list_outcome(
            utterance_id=utterance_id,
            final_text=final_text,
            raw_text=raw_text,
            context=context,
            anchor_selection=anchor_selection,
            wake_verified=wake_verified,
        )
        if explicit_bullet_list is not None:
            return self._annotate_outcome(explicit_bullet_list)

        natural_bullet_list = self._natural_bullet_list_outcome(
            utterance_id=utterance_id,
            final_text=final_text,
            raw_text=raw_text,
            context=context,
            anchor_selection=anchor_selection,
            wake_verified=wake_verified,
        )
        if natural_bullet_list is not None:
            return self._annotate_outcome(natural_bullet_list)

        structural_list = self._structural_list_outcome_when_planner_disabled(
            utterance_id=utterance_id,
            final_text=final_text,
            raw_text=raw_text,
            context=context,
            anchor_selection=anchor_selection,
            wake_verified=wake_verified,
            memory_store=memory_store,
            require_turn_planner_disabled=True,
        )
        if structural_list is not None:
            return self._annotate_outcome(structural_list)

        # The dictation editor is the primary AI lane for non-wake dictation:
        # one cached-prefix edit-script pass, deterministically applied. On
        # any failure (parse, grounding, deadline) it returns None and the
        # lanes below act as the floor.
        if (
            not wake_verified
            and self.config.dictation_editor_enabled
            and self.backend is not None
            and self.config.enable_model_transforms
            and self.state.mode != WriterMode.VERBATIM
            and not (getattr(context, "selected_text", "") or "").strip()
            and (final_text or "").strip()
            and not _is_no_touch_context(context)
            and not self._pre_editor_route_match(
                utterance_id=utterance_id,
                final_text=final_text,
                raw_text=raw_text,
                memory_store=memory_store,
                context=context,
                language_hint=language_hint,
            )
        ):
            editor_outcome = self.edit_dictation(
                utterance_id=utterance_id,
                final_text=final_text,
                raw_text=raw_text,
                context=context,
                memory_packet=memory_packet,
                memory_store=memory_store,
            )
            if editor_outcome is not None:
                return self._annotate_outcome(editor_outcome)

        if (
            turn_plan_result is None
            and self.config.enable_turn_planner
            and self.state.mode != WriterMode.VERBATIM
        ):
            turn_plan_result = self.plan_turn(
                utterance_id=utterance_id,
                final_text=final_text,
                raw_text=raw_text,
                context=context,
                memory_store=memory_store,
                memory_snapshot=memory_snapshot,
                memory_packet=memory_packet,
                language_hint=language_hint,
                mode_policy=self.state.mode_policy,
                mode_selection=self.state.mode_selection,
                partial_text=partial_text,
                writer_tone_addon=self.state.writer_tone_addon,
                wake_verified=wake_verified,
                now_iso=now_iso,
            )
        turn_plan_outcome = self._outcome_from_turn_plan(
            utterance_id=utterance_id,
            final_text=final_text,
            raw_text=raw_text,
            context=context,
            anchor_selection=anchor_selection,
            memory_store=memory_store,
            memory_snapshot=memory_snapshot,
            memory_packet=memory_packet,
            turn_plan_result=turn_plan_result,
            partial_text=partial_text,
            wake_verified=wake_verified,
        )
        if turn_plan_outcome is not None:
            return self._annotate_outcome(turn_plan_outcome)

        memory_learning_outcome = self._memory_learning_outcome_from_extractor(
            utterance_id=utterance_id,
            final_text=final_text,
            partial_text=partial_text,
            memory_store=memory_store,
            turn_plan_result=turn_plan_result,
        )
        if memory_learning_outcome is not None:
            return self._annotate_outcome(memory_learning_outcome)

        selection_present = bool(anchor_selection is not None and anchor_selection.start != anchor_selection.end and context.selected_text.strip())
        intent = self.parser.parse(
            final_text,
            language_hint=language_hint,
            selection_present=selection_present,
            active_mode=self.state.mode,
            mode_policy=self.state.mode_policy,
            partial_text=partial_text,
        )
        self.recorder.record(TraceKind.WRITER, 'writer_intent_parsed', {
            'utterance_id': utterance_id,
            'intent': intent.to_dict(),
            'active_mode': self.state.mode.value,
            'active_structure_mode': self.state.structure_mode,
            'effective_mode': None if self.state.mode_selection is None else self.state.mode_selection.effective_mode,
            'mode_source': None if self.state.mode_selection is None else self.state.mode_selection.mode_source.value,
        })

        if intent.kind == WriterIntentKind.DICTATE and selection_present and self.state.mode != WriterMode.VERBATIM:
            from juno_v2.writer.selection_intent_gate import resolve_selection_intent_after_dictate_parse

            gate = resolve_selection_intent_after_dictate_parse(
                recorder=self.recorder,
                utterance_id=utterance_id,
                spoken_final=final_text,
                context=context,
                anchor_selection=anchor_selection,
                mode_policy=self.state.mode_policy,
                backend=self,
            )
            if gate.kind == 'ambiguous':
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.NOOP,
                    output_text='',
                    learn_from_commit=False,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    metadata={
                        'reason': 'selection_intent_ambiguous',
                        'spoken_preview': final_text[:120],
                        'gate_confidence': gate.confidence,
                    },
                ))
            if gate.kind == 'edit' and gate.instruction:
                intent = WriterIntent(
                    kind=WriterIntentKind.MODEL_TRANSFORM,
                    raw_text=final_text,
                    instruction=gate.instruction.strip(),
                    metadata={'selection_nl_gate': gate.source},
                )

        if intent.kind == WriterIntentKind.DICTATE:
            structural_list = self._structural_list_outcome_when_planner_disabled(
                utterance_id=utterance_id,
                final_text=final_text,
                raw_text=raw_text,
                context=context,
                anchor_selection=anchor_selection,
                wake_verified=wake_verified,
                memory_store=memory_store,
                require_turn_planner_disabled=False,
            )
            if structural_list is not None:
                return self._annotate_outcome(structural_list)

            text = final_text
            cleanup_meta = {'pipeline': 'already_adjudicated_passthrough'}
            model_used = False
            direct_snippet = self._direct_snippet_insert(
                text,
                memory_store=memory_store,
                app_category=context.app_category,
            )
            if direct_snippet is not None:
                body, snippet_meta = direct_snippet
                self.recorder.record(
                    TraceKind.WRITER,
                    'writer_snippet_direct_inserted',
                    {
                        'utterance_id': utterance_id,
                        **snippet_meta,
                    },
                )
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.PASS_THROUGH_COMMIT,
                    output_text=body,
                    learn_from_commit=True,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    model_used=False,
                    metadata={
                        'raw_text': raw_text,
                        'input_text': final_text,
                        'snippet_expanded': True,
                        'dictation_cleanup': {
                            'pipeline': 'snippet_direct_insert',
                            **snippet_meta,
                        },
                        'grammar_postpass': None,
                    },
                ))
            # Snippet expansion: expand user-defined triggers (e.g. ``brb`` ->
            # ``be right back``) before structure-mode formatting so a
            # snippet body participates in bullet / numbered rendering.
            # Scope is the app category when known, falling back to
            # ``global``. We skip expansion in VERBATIM mode because that
            # mode promises an as-spoken paste with no writer massage.
            snippet_expanded = False
            _is_raw_surface = (context.app_category or '').strip().lower() in ('code', 'terminal')
            snippet_scopes: list[str] = []
            if (
                self.state.mode != WriterMode.VERBATIM
                and not _is_raw_surface
                and memory_store is not None
                and getattr(memory_store, "snippets", None) is not None
                and text
            ):
                snippet_scopes = self._snippet_scopes_for_policy(context.app_category)
                before_snippets = text
                for scope in snippet_scopes:
                    text = expand_snippets(text, resolver=memory_store.snippets, scope=scope)
                snippet_expanded = text != before_snippets
                if snippet_expanded:
                    self.recorder.record(
                        TraceKind.WRITER,
                        'writer_snippets_expanded',
                        {
                            'utterance_id': utterance_id,
                            'scopes': snippet_scopes,
                            'before_chars': len(before_snippets),
                            'after_chars': len(text),
                        },
                    )
            grammar_postpass_meta: dict[str, Any] | None = None
            if self.state.mode != WriterMode.VERBATIM:
                text = apply_commit_boundary_rules(text, app_category=context.app_category)
                # Issue #12 — code_grammar / meeting_grammar auto-fire,
                # gated by ``app_category``. Runs after boundary rules
                # (which is a no-op for code/terminal) so code surfaces
                # still get their grammar pass even when the
                # final-formatter chain below bails out.
                grammar_result = apply_grammar_postpass(
                    text, app_category=context.app_category
                )
                if grammar_result.applied:
                    text = grammar_result.text
                    grammar_postpass_meta = {
                        'engine': grammar_result.engine,
                        'rules_applied': list(grammar_result.rules_applied),
                    }
                    self.recorder.record(
                        TraceKind.WRITER,
                        'writer_grammar_postpass_applied',
                        {
                            'utterance_id': utterance_id,
                            'app_category': context.app_category,
                            **grammar_postpass_meta,
                        },
                    )
                text = self._apply_structure_mode_to_dictation(text, context=context, anchor_selection=anchor_selection)
                formatting_result = self._run_final_formatting_if_needed(
                    utterance_id=utterance_id,
                    text=text,
                    context=context,
                    memory_store=memory_store,
                )
                if formatting_result is not None:
                    text = formatting_result.text
                    model_used = model_used or not bool(formatting_result.metadata.get('deterministic'))
                    cleanup_meta = {
                        'pipeline': 'writer_final_formatting_v1',
                        'writer_backend': formatting_result.backend_name,
                        'writer_decode_ms': formatting_result.decode_ms,
                        **formatting_result.metadata,
                    }
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.PASS_THROUGH_COMMIT,
                output_text=text,
                learn_from_commit=True,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=(not model_used and text != final_text),
                model_used=model_used,
                metadata={
                    'raw_text': raw_text,
                    'input_text': final_text,
                    'snippet_expanded': snippet_expanded,
                    'dictation_cleanup': cleanup_meta,
                    'grammar_postpass': grammar_postpass_meta,
                },
            ))

        if intent.kind == WriterIntentKind.COMMAND_RESULT:
            return self._execute_deterministic_command(
                utterance_id=utterance_id,
                intent=intent,
                context=context,
                anchor_selection=anchor_selection,
                partial_text=partial_text or '',
                memory_store=memory_store,
                memory_snapshot=memory_snapshot,
                memory_packet=memory_packet,
            )

        if intent.kind == WriterIntentKind.INSERT_CONTEXT_FIELD:
            return self._annotate_outcome(self._insert_context_field(utterance_id, intent, context, anchor_selection))

        if intent.kind == WriterIntentKind.INSERT_FORMATTING:
            insert_text = intent.insert_text
            if intent.transform_kind == 'next_numbered':
                insert_text = self._next_numbered_marker(context=context)
            elif intent.transform_kind == 'new_bullet' and self.state.structure_mode == 'bullets':
                return self._noop(utterance_id, 'structure_mode_handles_items', intent.kind.value)
            selection = None
            if anchor_selection is not None:
                selection = ClientSelection(start=anchor_selection.end, end=anchor_selection.end)
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.DIRECT_COMMIT,
                output_text=insert_text,
                commit_mode=CommitMode.INSERT_AT_CARET,
                selection_override=selection,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={'transform_kind': intent.transform_kind},
            ))

        if intent.kind == WriterIntentKind.SET_STRUCTURE_MODE:
            self.state.structure_mode = intent.structure_mode
            self.state.structure_item_index = 1
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.STATE_MUTATION,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                metadata={'state_action': 'set_structure_mode', 'structure_mode': self.state.structure_mode},
            ))

        if intent.kind == WriterIntentKind.SWITCH_MODE and intent.mode is not None:
            self.state.mode = intent.mode
            self.state.mode_selection = None
            self.state.mode_policy = None
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.MODE_SWITCH,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                metadata={
                    'mode': self.state.mode.value,
                    'manual_writer_mode': intent.mode.value,
                    'set_manual_writer_mode': intent.mode.value,
                },
            ))

        if intent.kind == WriterIntentKind.ADD_TERM and memory_store is not None and intent.term:
            memory_store.add_lexicon_entry(term=intent.term, canonical_form=intent.term, aliases=[], source='voice_command')
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.MEMORY_MUTATION,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                memory_updated=True,
                metadata={'memory_action': 'add_term', 'term': intent.term},
            ))

        if intent.kind == WriterIntentKind.ADD_REPLACEMENT and memory_store is not None and intent.trigger and intent.replacement:
            memory_store.add_replacement(trigger=intent.trigger, replacement=intent.replacement, source='voice_command')
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.MEMORY_MUTATION,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                memory_updated=True,
                metadata={'memory_action': 'add_replacement', 'trigger': intent.trigger, 'replacement': intent.replacement},
            ))

        if intent.kind == WriterIntentKind.DETERMINISTIC_TRANSFORM:
            target = self._selected_target(context, anchor_selection)
            if target is None:
                return self._noop(utterance_id, 'missing_selection_for_transform', intent.kind.value)
            transformed = self._deterministic_transform(intent.transform_kind or '', target['text'])
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.TRANSFORM_COMMIT,
                output_text=transformed,
                commit_mode=CommitMode.REPLACE_SELECTION,
                selection_override=target['selection'],
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={'transform_kind': intent.transform_kind, 'source_text': target['text'], 'target': target['target']},
            ))

        if intent.kind == WriterIntentKind.MODEL_TRANSFORM:
            target = self._selected_target(context, anchor_selection)
            if target is None:
                return self._noop(utterance_id, 'missing_selection_for_transform', intent.kind.value)
            return self._annotate_outcome(self._run_model_transform(
                utterance_id=utterance_id,
                intent=intent,
                target=target,
                context=context,
                memory_store=memory_store,
                memory_snapshot=memory_snapshot,
                memory_packet=memory_packet,
            ))

        if intent.kind == WriterIntentKind.RECENT_DETERMINISTIC_TRANSFORM:
            target = self._selected_target(context, anchor_selection) or self._recent_target(context)
            if target is None:
                return self._noop(utterance_id, 'missing_recent_target', intent.kind.value)
            transformed = self._deterministic_transform(intent.transform_kind or '', target['text'])
            out_text, replace_meta = _replace_commit_fields(target, transformed)
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.TRANSFORM_COMMIT,
                output_text=out_text,
                commit_mode=CommitMode.REPLACE_SELECTION,
                selection_override=target['selection'],
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={'transform_kind': intent.transform_kind, 'source_text': target['text'], **replace_meta},
            ))

        if intent.kind == WriterIntentKind.RECENT_MODEL_TRANSFORM:
            target = self._selected_target(context, anchor_selection) or self._recent_target(context)
            if target is None and partial_text and partial_text.strip() and self.backend is not None and self.config.enable_model_transforms:
                result = self._rewrite_with_backend(WriterTransformRequest(
                    utterance_id=utterance_id,
                    instruction=intent.instruction or 'Rewrite the current text.',
                    source_text=partial_text.strip(),
                    mode=self.state.mode,
                    target_selection=None,
                    context_payload={
                        'app_name': context.app_name,
                        'window_title': context.window_title,
                        'app_category': context.app_category,
                        'selected_text': context.selected_text,
                        'field_text_excerpt': context.field_text_excerpt,
                        'candidate_entities': context.candidate_entities,
                        'recent_clipboard': list(context.recent_clipboard) or list(
                            context.metadata.get('recent_clipboard') or []
                        ),
                        'memory_packet': memory_packet,
                        'target_kind': 'active_partial',
                    },
                    metadata={'raw_command': intent.raw_text, 'fallback_target': 'active_partial'},
                ))
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.STATE_MUTATION,
                    output_text=result.text,
                    learn_from_commit=False,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    model_used=True,
                    metadata={
                        'instruction': intent.instruction,
                        'writer_backend': result.backend_name,
                        'writer_decode_ms': result.decode_ms,
                        'target': 'active_partial',
                        'state_action': 'active_partial_rewrite',
                        'pending_partial_text': result.text,
                        **result.metadata,
                    },
                ))
            if target is None:
                return self._noop(utterance_id, 'missing_recent_target', intent.kind.value)
            return self._annotate_outcome(self._run_model_transform(
                utterance_id=utterance_id,
                intent=intent,
                target=target,
                context=context,
                memory_store=memory_store,
                memory_snapshot=memory_snapshot,
                memory_packet=memory_packet,
            ))

        return self._noop(utterance_id, 'unsupported_intent', intent.kind.value, extra={'intent': intent.to_dict()})

    def _apply_runtime_mode(
        self,
        *,
        mode_policy: ModePolicy | None,
        mode_selection: ModeSelection | None,
        writer_tone_addon: str | None,
    ) -> None:
        if mode_policy is not None:
            self.state.mode_policy = mode_policy
        if mode_selection is not None:
            self.state.mode_selection = mode_selection
            self.state.mode = self._writer_mode_from_string(mode_selection.effective_mode)
        elif mode_policy is not None:
            self.state.mode = self._writer_mode_from_string(mode_policy.base_mode)
        self.state.writer_tone_addon = (writer_tone_addon or "").strip() or None

    def _outcome_from_turn_plan(
        self,
        *,
        utterance_id: str,
        final_text: str,
        raw_text: str,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
        memory_store: JsonMemoryStore | None,
        memory_snapshot: MemorySnapshot | None,
        memory_packet: dict | None,
        turn_plan_result: TurnPlanResult | None,
        partial_text: str | None = None,
        wake_verified: bool = False,
    ) -> WriterOutcome | None:
        if turn_plan_result is None:
            return None
        if not turn_plan_result.ok or not isinstance(turn_plan_result.plan, dict):
            return None
        if self.state.mode == WriterMode.VERBATIM:
            return None

        plan = turn_plan_result.plan
        validation = validate_turn_plan(plan, source_text=final_text, context=context)
        self.recorder.record(
            TraceKind.WRITER,
            "turn_plan_validated",
            {
                "utterance_id": utterance_id,
                "ok": validation.ok,
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
                "utterance_kind": plan.get("utterance_kind"),
            },
        )
        if not validation.ok:
            return None

        kind = str(plan.get("utterance_kind") or "").strip()
        safety = plan.get("safety") if isinstance(plan.get("safety"), dict) else {}
        commit_policy = str(safety.get("commit_policy") or "commit").strip()
        memory_source_text = _memory_learning_source_text(final_text, partial_text)
        explicit_memory_request = (
            kind == "memory_mutation"
            or _explicit_memory_learning_requested(memory_source_text)
            or _explicit_memory_learning_requested(final_text)
        )
        memory_plan = plan
        explicit_vocab_request = (
            _explicit_vocab_learning_requested(memory_source_text)
            or _explicit_vocab_learning_requested(final_text)
        )
        if explicit_memory_request and explicit_vocab_request:
            extracted = self.extract_memory_candidates(text=memory_source_text, kind="vocab", limit=8)
            if extracted:
                existing = plan.get("memory_candidates")
                memory_plan = {
                    **plan,
                    "memory_candidates": [
                        *(existing if isinstance(existing, list) else []),
                        *extracted,
                    ],
                }
        memory_mutation = self._apply_turn_plan_memory_candidates(
            memory_plan,
            source_text=memory_source_text,
            memory_store=memory_store,
            explicit_request=explicit_memory_request,
        )
        if memory_mutation is not None:
            if memory_mutation["entries"]:
                return WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.MEMORY_MUTATION,
                    output_text="",
                    learn_from_commit=False,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    model_used=True,
                    memory_updated=True,
                    metadata={
                        "memory_action": "turn_plan_memory_candidates",
                        **memory_mutation,
                        "turn_plan": self._turn_plan_metadata(turn_plan_result, validation=validation),
                    },
                )
            if kind == "memory_mutation":
                return WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.NOOP,
                    output_text="",
                    learn_from_commit=False,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    model_used=True,
                    metadata={
                        "reason": memory_mutation["reason"],
                        **memory_mutation,
                        "turn_plan": self._turn_plan_metadata(turn_plan_result, validation=validation),
                    },
                )
        if explicit_vocab_request:
            return None
        if kind != "actions" and commit_policy in {"no_commit", "confirm"}:
            if not wake_verified and kind in {"dictation", "format_dictation", "mixed"}:
                self.recorder.record(
                    TraceKind.WRITER,
                    "turn_plan_text_commit_policy_ignored",
                    {
                        "utterance_id": utterance_id,
                        "utterance_kind": kind,
                        "commit_policy": commit_policy,
                    },
                )
                return None
            return WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.NOOP,
                output_text="",
                learn_from_commit=False,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                model_used=True,
                metadata={
                    "reason": f"turn_plan_commit_policy_{commit_policy}",
                    "turn_plan": self._turn_plan_metadata(turn_plan_result, validation=validation),
                },
            )
        if kind in {"no_op", "ambiguous"}:
            return WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.NOOP,
                output_text="",
                learn_from_commit=False,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                model_used=True,
                metadata={
                    "reason": f"turn_plan_{kind}",
                    "turn_plan": self._turn_plan_metadata(turn_plan_result, validation=validation),
                },
            )

        transform = plan.get("transform") if isinstance(plan.get("transform"), dict) else {}
        transform_operation = str(transform.get("operation") or "none").strip()
        if kind == "transform" or transform_operation != "none":
            outcome = self._turn_plan_transform_outcome(
                utterance_id=utterance_id,
                plan=plan,
                result=turn_plan_result,
                final_text=final_text,
                context=context,
                anchor_selection=anchor_selection,
                memory_snapshot=memory_snapshot,
                memory_packet=memory_packet,
                validation=validation,
            )
            if outcome is not None:
                return outcome
            return None

        if kind in {"command", "memory_mutation"}:
            return None

        if kind == "actions":
            # Action dispatch is pipeline-owned. By the time the writer sees
            # an actions plan, the pipeline has already either dispatched the
            # coerced actions (and the writer is never called) or dispatched
            # nothing. Returning a paste-suppressing NOOP here therefore
            # always meant "no paste AND no action" — the production
            # data-loss black hole (2026-06-10 history rows with
            # failure_reason=turn_plan_action_only and empty transcripts).
            # Always fall back to text delivery.
            actions = plan.get("actions")
            self.recorder.record(
                TraceKind.WRITER,
                "turn_plan_actions_fell_back_to_text",
                {
                    "utterance_id": utterance_id,
                    "wake_verified": wake_verified,
                    "action_count": len(actions) if isinstance(actions, list) else 0,
                    "reason": "text_delivery_guaranteed",
                },
            )
            return None

        render_result = render_turn_plan(plan, context=context, memory_store=memory_store)
        if render_result.rendered and render_result.text.strip() and render_result.reason != "corrected_text_fallback":
            render_kind = str((plan.get("render_plan") or {}).get("render_kind") or "").strip()
            structural = render_kind in {"bulleted_list", "numbered_list", "checklist", "table", "email", "ai_prompt"}
            return WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.PASS_THROUGH_COMMIT,
                output_text=render_result.text,
                learn_from_commit=not structural and kind == "dictation",
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=False,
                model_used=True,
                metadata={
                    "raw_text": raw_text,
                    "input_text": final_text,
                    "turn_plan": self._turn_plan_metadata(
                        turn_plan_result,
                        validation=validation,
                        render_metadata=render_result.metadata,
                        render_reason=render_result.reason,
                    ),
                },
            )

        if kind == "mixed" and wake_verified and isinstance(plan.get("actions"), list) and plan.get("actions"):
            return WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.NOOP,
                output_text="",
                learn_from_commit=False,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                model_used=True,
                metadata={
                    "reason": "turn_plan_action_only",
                    "turn_plan": self._turn_plan_metadata(
                        turn_plan_result,
                        validation=validation,
                        render_metadata=render_result.metadata,
                        render_reason=render_result.reason,
                    ),
                },
            )
        return None

    def _defer_list_to_editor(self, utterance_id: str, final_text: str, *, lane: str) -> bool:
        """Deterministic list lanes yield to the editor on surviving cues."""
        if not _unresolved_correction_cues_present(final_text):
            return False
        self.recorder.record(
            TraceKind.WRITER,
            "deterministic_list_deferred_to_editor",
            {"utterance_id": utterance_id, "lane": lane},
        )
        return True

    def _explicit_bullet_list_outcome(
        self,
        *,
        utterance_id: str,
        final_text: str,
        raw_text: str,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
        wake_verified: bool,
    ) -> WriterOutcome | None:
        if (
            wake_verified
            or self.state.mode == WriterMode.VERBATIM
            or (anchor_selection is not None and anchor_selection.start != anchor_selection.end)
        ):
            return None
        if _is_no_touch_context(context):
            return None
        rendered = render_explicit_bullet_list_command(final_text)
        if rendered is None:
            return None
        if self._defer_list_to_editor(utterance_id, final_text, lane="explicit_bullet_list"):
            return None
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.PASS_THROUGH_COMMIT,
            output_text=rendered,
            learn_from_commit=False,
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            deterministic_used=True,
            model_used=False,
            metadata={
                "raw_text": raw_text,
                "input_text": final_text,
                "dictation_cleanup": {"pipeline": "explicit_same_utterance_bullet_list"},
                "structure": "bullets",
            },
        )

    def _natural_bullet_list_outcome(
        self,
        *,
        utterance_id: str,
        final_text: str,
        raw_text: str,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
        wake_verified: bool,
    ) -> WriterOutcome | None:
        if (
            wake_verified
            or self.state.mode == WriterMode.VERBATIM
            or (anchor_selection is not None and anchor_selection.start != anchor_selection.end)
        ):
            return None
        if _is_no_touch_context(context):
            return None
        rendered = render_natural_bullet_list_dictation(final_text)
        if rendered is None:
            return None
        if self._defer_list_to_editor(utterance_id, final_text, lane="natural_bullet_list"):
            return None
        mismatch = (
            rendered.claimed_item_count is not None
            and rendered.claimed_item_count != rendered.spoken_item_count
        )
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.PASS_THROUGH_COMMIT,
            output_text=rendered.text,
            learn_from_commit=False,
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            deterministic_used=True,
            model_used=False,
            metadata={
                "raw_text": raw_text,
                "input_text": final_text,
                "dictation_cleanup": {
                    "pipeline": rendered.pipeline,
                    "claimed_item_count": rendered.claimed_item_count,
                    "spoken_item_count": rendered.spoken_item_count,
                    "claimed_count_mismatch": mismatch,
                },
                "structure": "bullets",
            },
        )

    def _structural_list_outcome_when_planner_disabled(
        self,
        *,
        utterance_id: str,
        final_text: str,
        raw_text: str,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
        wake_verified: bool,
        memory_store: JsonMemoryStore | None,
        require_turn_planner_disabled: bool,
    ) -> WriterOutcome | None:
        if (
            wake_verified
            or (require_turn_planner_disabled and self.config.enable_turn_planner)
            or self.state.mode == WriterMode.VERBATIM
            or _is_no_touch_context(context)
            or (anchor_selection is not None and anchor_selection.start != anchor_selection.end)
        ):
            return None
        fallback = fallback_structural_turn_plan(final_text)
        if fallback is None:
            return None
        if self._defer_list_to_editor(utterance_id, final_text, lane="structural_list"):
            return None
        validation = validate_turn_plan(fallback, source_text=final_text, context=context)
        if not validation.ok:
            return None
        render_result = render_turn_plan(fallback, context=context, memory_store=memory_store)
        if not render_result.rendered or not render_result.text.strip():
            return None
        render_kind = str((fallback.get("render_plan") or {}).get("render_kind") or "").strip()
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.PASS_THROUGH_COMMIT,
            output_text=render_result.text,
            learn_from_commit=False,
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            deterministic_used=True,
            model_used=False,
            metadata={
                "raw_text": raw_text,
                "input_text": final_text,
                "turn_plan": {
                    "status": "ok",
                    "backend": "deterministic_structural_no_planner",
                    "utterance_kind": fallback.get("utterance_kind"),
                    "render_kind": render_kind,
                    "validation_ok": True,
                    "validation_warnings": list(validation.warnings),
                    "render_reason": render_result.reason,
                    "render_metadata": render_result.metadata,
                },
                "dictation_cleanup": {"pipeline": "deterministic_structural_no_planner"},
                "structure": render_kind,
            },
        )

    def _apply_turn_plan_memory_candidates(
        self,
        plan: dict[str, Any],
        *,
        source_text: str,
        memory_store: JsonMemoryStore | None,
        explicit_request: bool,
    ) -> dict[str, Any] | None:
        if not explicit_request:
            return None
        raw_candidates = plan.get("memory_candidates")
        model_candidates = raw_candidates if isinstance(raw_candidates, list) else []
        spelled_candidates = _spelled_vocab_memory_candidates(source_text)
        source_candidates = _explicit_vocab_source_phrase_candidates(source_text)
        spelled_keys = {
            _memory_term_key(candidate.get("canonical_form") or candidate.get("term"))
            for candidate in spelled_candidates
        }
        spelled_keys.discard("")
        raw_candidates = [*spelled_candidates, *source_candidates, *model_candidates]
        if not raw_candidates:
            return None
        if memory_store is None:
            return {"entries": [], "skipped": [], "reason": "memory_store_unavailable"}

        normalized_candidates: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for raw in raw_candidates[:12]:
            candidate = _normalize_turn_plan_memory_candidate(raw, source_text=source_text)
            if candidate is None:
                skipped.append({"reason": "invalid_or_ungrounded"})
                continue
            if _memory_candidate_overlaps_spelled_acronym_tail(candidate, spelled_keys=spelled_keys):
                skipped.append({"term": candidate["canonical_form"], "reason": "overlaps_spelled_candidate"})
                continue
            normalized_candidates.append(candidate)

        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in normalized_candidates:
            if _memory_candidate_contained_by_longer_candidate(candidate, normalized_candidates):
                skipped.append({"term": candidate["canonical_form"], "reason": "contained_in_longer_candidate"})
                continue
            key = _memory_term_key(candidate["canonical_form"])
            if not key or key in seen:
                skipped.append({"term": candidate["canonical_form"], "reason": "duplicate"})
                continue
            seen.add(key)
            memory_store.add_lexicon_entry(
                term=candidate["term"],
                canonical_form=candidate["canonical_form"],
                aliases=candidate["aliases"],
                pronunciation_hint=candidate["pronunciation_hint"],
                source="voice_teach_turn_plan",
            )
            entries.append(candidate)
        if not entries:
            return {"entries": [], "skipped": skipped, "reason": "no_valid_memory_candidates"}
        return {"entries": entries, "skipped": skipped, "reason": "ok"}

    def _memory_learning_outcome_from_extractor(
        self,
        *,
        utterance_id: str,
        final_text: str,
        partial_text: str | None,
        memory_store: JsonMemoryStore | None,
        turn_plan_result: TurnPlanResult | None,
    ) -> WriterOutcome | None:
        memory_source_text = _memory_learning_source_text(final_text, partial_text)
        if not _explicit_vocab_learning_requested(memory_source_text):
            return None
        if memory_store is None:
            return WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.NOOP,
                output_text="",
                learn_from_commit=False,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                model_used=False,
                metadata={"reason": "memory_store_unavailable"},
            )
        extracted = self.extract_memory_candidates(text=memory_source_text, kind="vocab", limit=8)
        if extracted is None:
            return None
        mutation = self._apply_turn_plan_memory_candidates(
            {"memory_candidates": extracted},
            source_text=memory_source_text,
            memory_store=memory_store,
            explicit_request=True,
        )
        turn_plan_meta: dict[str, Any] = {}
        if turn_plan_result is not None:
            turn_plan_meta = {
                "status": turn_plan_result.status,
                "errors": list(turn_plan_result.errors),
                "repair_attempted": turn_plan_result.repair_attempted,
                "repair_status": turn_plan_result.repair_status,
            }
        if mutation is not None and mutation["entries"]:
            return WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.MEMORY_MUTATION,
                output_text="",
                learn_from_commit=False,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                model_used=True,
                memory_updated=True,
                metadata={
                    "memory_action": "extractor_memory_candidates",
                    **mutation,
                    "turn_plan_fallback": turn_plan_meta,
                },
            )
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.NOOP,
            output_text="",
            learn_from_commit=False,
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            model_used=True,
            metadata={
                "reason": "no_valid_memory_candidates",
                "memory_action": "extractor_memory_candidates",
                "extracted_count": len(extracted),
                "turn_plan_fallback": turn_plan_meta,
            },
        )

    def _turn_plan_transform_outcome(
        self,
        *,
        utterance_id: str,
        plan: dict[str, Any],
        result: TurnPlanResult,
        final_text: str,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
        memory_snapshot: MemorySnapshot | None,
        memory_packet: dict | None,
        validation,
    ) -> WriterOutcome | None:
        transform = plan.get("transform") if isinstance(plan.get("transform"), dict) else {}
        target = self._turn_plan_target(plan, context=context, anchor_selection=anchor_selection)
        if target is None:
            if _mentions_recent_transform_target(final_text):
                target = self._selected_target(context, anchor_selection) or self._recent_target(context)
            if target is None:
                self.recorder.record(
                    TraceKind.WRITER,
                    "turn_plan_transform_target_deferred_to_parser",
                    {
                        "utterance_id": utterance_id,
                        "reason": "missing_transform_target",
                        "recent_target_command": _mentions_recent_transform_target(final_text),
                    },
                )
                return None

        transformed = transform.get("transformed_text")
        if not isinstance(transformed, str) or not transformed.strip():
            if bool(transform.get("requires_second_pass")):
                transformed = self._run_turn_plan_transform_generation(
                    utterance_id=utterance_id,
                    plan=plan,
                    source_text=target["text"],
                    context=context,
                    target_kind=target["target"],
                    memory_snapshot=memory_snapshot,
                    memory_packet=memory_packet,
                )
            else:
                transformed = ""
        if not isinstance(transformed, str) or not transformed.strip():
            return None

        out_text, replace_meta = _replace_commit_fields(target, transformed)
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.TRANSFORM_COMMIT,
            output_text=out_text,
            commit_mode=CommitMode.REPLACE_SELECTION,
            selection_override=target["selection"],
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            model_used=True,
            metadata={
                "instruction": str(transform.get("instruction") or final_text),
                "turn_plan": self._turn_plan_metadata(result, validation=validation),
                **replace_meta,
            },
        )

    def _turn_plan_target(
        self,
        plan: dict[str, Any],
        *,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
    ):
        target = plan.get("target") if isinstance(plan.get("target"), dict) else {}
        kind = str(target.get("kind") or "").strip()
        if kind == "selection":
            return self._selected_target(context, anchor_selection)
        if kind in {"recent_commit", "recent_clipboard"}:
            return self._recent_target(context)
        if kind == "explicit_span":
            return self._selected_target(context, anchor_selection) or self._recent_target(context)
        return None

    def _run_turn_plan_transform_generation(
        self,
        *,
        utterance_id: str,
        plan: dict[str, Any],
        source_text: str,
        context: TypedContextBundle,
        target_kind: str,
        memory_snapshot: MemorySnapshot | None,
        memory_packet: dict | None,
    ) -> str:
        if self.backend is None or not self.config.enable_model_transforms:
            return ""
        payload = {
            "task": "transform_generation_v1",
            "schema_version": "transform_generation_v1",
            "turn_plan": plan,
            "source_text": source_text,
            "target_kind": target_kind,
            "context": {
                "app_name": context.app_name,
                "window_title": context.window_title,
                "app_category": context.app_category,
                "selected_text": context.selected_text,
                "field_text_excerpt": context.field_text_excerpt,
                "candidate_entities": list(context.candidate_entities or [])[:32],
            },
            "memory": {
                "packet": memory_packet or {},
                "snapshot_counts": None if memory_snapshot is None else {
                    "lexicon": len(memory_snapshot.lexicon),
                    "replacements": len(memory_snapshot.replacements),
                    "corrections": len(memory_snapshot.corrections),
                    "session_entities": len(memory_snapshot.session_entities),
                },
            },
        }
        response = self._rewrite_with_backend(WriterTransformRequest(
            utterance_id=utterance_id,
            instruction="Generate transformed_text for this typed turn plan.",
            source_text=source_text,
            mode=self.state.mode,
            target_selection=None,
            context_payload={"task": "transform_generation_v1", "payload": payload},
            metadata={"kind": "transform_generation_v1", "max_tokens": max(512, self.config.max_tokens)},
        ))
        obj = _json_object_from_model_text(response.text)
        if isinstance(obj, dict):
            text = obj.get("transformed_text")
            return str(text or "").strip()
        return ""

    def _turn_plan_metadata(
        self,
        result: TurnPlanResult,
        *,
        validation,
        render_metadata: dict[str, Any] | None = None,
        render_reason: str | None = None,
    ) -> dict[str, Any]:
        plan = result.plan if isinstance(result.plan, dict) else {}
        render = plan.get("render_plan") if isinstance(plan.get("render_plan"), dict) else {}
        return {
            "enabled": True,
            "backend": result.backend_name,
            "decode_ms": result.decode_ms,
            "status": result.status,
            "errors": list(result.errors),
            "repair_attempted": result.repair_attempted,
            "repair_status": result.repair_status,
            "initial_status": result.initial_status,
            "initial_errors": list(result.initial_errors),
            "validation_errors_before_repair": list(result.validation_errors_before_repair),
            "validation_warnings_before_repair": list(result.validation_warnings_before_repair),
            "raw_output_chars": len(result.raw_output or ""),
            "utterance_kind": plan.get("utterance_kind"),
            "render_kind": render.get("render_kind"),
            "action_count": len(plan.get("actions") or []) if isinstance(plan.get("actions"), list) else 0,
            "commit_policy": (plan.get("safety") or {}).get("commit_policy") if isinstance(plan.get("safety"), dict) else None,
            "execute_policy": (plan.get("safety") or {}).get("execute_policy") if isinstance(plan.get("safety"), dict) else None,
            "validation_errors": list(validation.errors),
            "validation_warnings": list(validation.warnings),
            "render": render_metadata or {},
            "render_reason": render_reason,
        }

    def _execute_deterministic_command(
        self,
        *,
        utterance_id: str,
        intent,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
        partial_text: str,
        memory_store: JsonMemoryStore | None,
        memory_snapshot: MemorySnapshot | None,
        memory_packet: dict | None,
    ) -> WriterOutcome:
        raw_cmd = intent.metadata.get('deterministic_command') or {}
        kind = str(raw_cmd.get('kind', ''))
        payload = dict(raw_cmd.get('payload') or {})
        pol = self.state.mode_policy
        command_shape = True
        tgt_class, _, tgt_text = resolve_command_target(
            text=intent.raw_text,
            context=context,
            anchor_selection=anchor_selection,
            active_partial=partial_text,
            command_shape=command_shape,
        )
        if kind in {'discard_utterance', 'undo_last'} and tgt_class == CommandTargetClass.ACTIVE_UTTERANCE:
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.STATE_MUTATION,
                output_text='',
                learn_from_commit=False,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={'command': kind, 'target_class': tgt_class.value, 'state_action': 'discard_active_partial'},
            ))
        if kind == 'delete_words' and tgt_class == CommandTargetClass.ACTIVE_UTTERANCE and partial_text.strip():
            n = int(payload.get('words', 1))
            words = partial_text.strip().split()
            new_words = words[:-n] if len(words) > n else []
            out = ' '.join(new_words)
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.STATE_MUTATION,
                output_text=out,
                learn_from_commit=False,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={
                    'command': 'delete_words',
                    'removed': n,
                    'target_class': tgt_class.value,
                    'state_action': 'active_partial_rewrite',
                    'pending_partial_text': out,
                },
            ))
        if kind == 'delete_sentence' and payload.get('scope') == 'utterance' and partial_text.strip():
            parts = re.split(r'(?<=[.!?])\s+', partial_text.strip())
            out = ' '.join(parts[:-1]).strip() if len(parts) > 1 else ''
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.STATE_MUTATION,
                output_text=out,
                learn_from_commit=False,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={
                    'command': 'delete_last_sentence',
                    'target_class': tgt_class.value,
                    'state_action': 'active_partial_rewrite',
                    'pending_partial_text': out,
                },
            ))
        if kind == 'delete_sentence' and payload.get('scope') == 'recent':
            if tgt_class == CommandTargetClass.SELECTED_TEXT:
                target = self._selected_target(context, anchor_selection)
                missing_reason = 'missing_selection_for_transform'
            else:
                target = self._recent_target(context)
                missing_reason = 'missing_recent_target'
            if target is None:
                return self._noop(utterance_id, missing_reason, intent.kind.value)
            parts = re.split(r'(?<=[.!?])\s+', target['text'])
            out = ' '.join(parts[:-1]).strip() if len(parts) > 1 else ''
            out_text, replace_meta = _replace_commit_fields(target, out)
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.TRANSFORM_COMMIT,
                output_text=out_text,
                commit_mode=CommitMode.REPLACE_SELECTION,
                selection_override=target['selection'],
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={'command': 'delete_last_sentence_recent', **replace_meta},
            ))
        if kind == 'insert' and 'text' in payload:
            ins = str(payload['text'])
            sel = None
            if anchor_selection is not None:
                sel = ClientSelection(start=anchor_selection.end, end=anchor_selection.end)
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.DIRECT_COMMIT,
                output_text=ins,
                commit_mode=CommitMode.INSERT_AT_CARET,
                selection_override=sel,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={'command': 'insert', 'target_class': tgt_class.value},
            ))
        if kind == 'structure':
            if payload.get('mode') == 'bullet_item':
                if self.state.structure_mode == 'bullets':
                    return self._noop(utterance_id, 'structure_mode_handles_items', WriterIntentKind.INSERT_FORMATTING.value)
                self.state.structure_mode = 'bullets'
                self.state.structure_item_index = 1
                sel = None
                if anchor_selection is not None:
                    sel = ClientSelection(start=anchor_selection.end, end=anchor_selection.end)
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.DIRECT_COMMIT,
                    output_text='\n- ',
                    commit_mode=CommitMode.INSERT_AT_CARET,
                    selection_override=sel,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'next_bullet', 'transform_kind': 'new_bullet'},
                ))
            if payload.get('mode') == 'numbered_item':
                self.state.structure_mode = 'numbered'
                self.state.structure_item_index = 1
                ins = self._next_numbered_marker(context=context)
                sel = None
                if anchor_selection is not None:
                    sel = ClientSelection(start=anchor_selection.end, end=anchor_selection.end)
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.DIRECT_COMMIT,
                    output_text=ins,
                    commit_mode=CommitMode.INSERT_AT_CARET,
                    selection_override=sel,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'next_number', 'transform_kind': 'next_numbered'},
                ))
        if kind == 'quote' and 'text' in payload:
            ins = str(payload['text'])
            sel = None
            if anchor_selection is not None:
                sel = ClientSelection(start=anchor_selection.end, end=anchor_selection.end)
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.DIRECT_COMMIT,
                output_text=ins,
                commit_mode=CommitMode.INSERT_AT_CARET,
                selection_override=sel,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={'command': 'quote'},
            ))
        if kind == 'recent_edit':
            # Selection takes priority over recent commit. If user has text
            # selected when they say "fix that" / "make that shorter" etc.,
            # the command must operate on the selection — that's what they
            # explicitly pointed at.
            if tgt_class == CommandTargetClass.SELECTED_TEXT:
                target = self._selected_target(context, anchor_selection)
            else:
                target = self._recent_target(context)
            if target is None and partial_text.strip() and self.backend is not None and self.config.enable_model_transforms:
                if payload.get('transform') == 'bullets':
                    transformed = self._deterministic_transform('bullets', partial_text.strip())
                    return self._annotate_outcome(WriterOutcome(
                        utterance_id=utterance_id,
                        action=WriterActionKind.STATE_MUTATION,
                        output_text=transformed,
                        learn_from_commit=False,
                        writer_mode=self.state.mode,
                        structure_mode=self.state.structure_mode,
                        deterministic_used=True,
                        metadata={
                            'command': 'recent_bullets',
                            'target': 'active_partial',
                            'state_action': 'active_partial_rewrite',
                            'pending_partial_text': transformed,
                        },
                    ))
                if payload.get('transform') == 'numbered':
                    transformed = self._deterministic_transform('numbered', partial_text.strip())
                    return self._annotate_outcome(WriterOutcome(
                        utterance_id=utterance_id,
                        action=WriterActionKind.STATE_MUTATION,
                        output_text=transformed,
                        learn_from_commit=False,
                        writer_mode=self.state.mode,
                        structure_mode=self.state.structure_mode,
                        deterministic_used=True,
                        metadata={
                            'command': 'recent_numbered',
                            'target': 'active_partial',
                            'state_action': 'active_partial_rewrite',
                            'pending_partial_text': transformed,
                        },
                    ))
                if payload.get('transform') == 'delete_paragraph':
                    parts = re.split(r'\n\n+', partial_text.strip())
                    out = '\n\n'.join(parts[:-1]).strip() if len(parts) > 1 else ''
                    return self._annotate_outcome(WriterOutcome(
                        utterance_id=utterance_id,
                        action=WriterActionKind.STATE_MUTATION,
                        output_text=out,
                        learn_from_commit=False,
                        writer_mode=self.state.mode,
                        structure_mode=self.state.structure_mode,
                        deterministic_used=True,
                        metadata={
                            'command': 'delete_last_paragraph',
                            'target': 'active_partial',
                            'state_action': 'active_partial_rewrite',
                            'pending_partial_text': out,
                        },
                    ))
                instr = str(payload.get('instruction') or '')
                if instr and not self._model_rewrite_allowed_for_target(
                    pol,
                    CommandTargetClass.ACTIVE_UTTERANCE,
                ):
                    return self._model_rewrite_blocked_noop(
                        utterance_id,
                        intent,
                        CommandTargetClass.ACTIVE_UTTERANCE,
                    )
                if instr and (pol is None or pol.allow_recent_target_commands):
                    result = self._rewrite_with_backend(WriterTransformRequest(
                        utterance_id=utterance_id,
                        instruction=instr,
                        source_text=partial_text.strip(),
                        mode=self.state.mode,
                        target_selection=None,
                        context_payload={
                            'app_name': context.app_name,
                            'window_title': context.window_title,
                            'app_category': context.app_category,
                            'selected_text': context.selected_text,
                            'field_text_excerpt': context.field_text_excerpt,
                            'candidate_entities': context.candidate_entities,
                            'recent_clipboard': list(context.recent_clipboard) or list(
                                context.metadata.get('recent_clipboard') or []
                            ),
                            'memory_packet': memory_packet,
                            'target_kind': 'active_partial',
                        },
                        metadata={'raw_command': intent.raw_text, 'fallback_target': 'active_partial'},
                    ))
                    return self._annotate_outcome(WriterOutcome(
                        utterance_id=utterance_id,
                        action=WriterActionKind.STATE_MUTATION,
                        output_text=result.text,
                        learn_from_commit=False,
                        writer_mode=self.state.mode,
                        structure_mode=self.state.structure_mode,
                        model_used=True,
                        metadata={
                            'command': 'recent_edit',
                            'target': 'active_partial',
                            'state_action': 'active_partial_rewrite',
                            'pending_partial_text': result.text,
                            'writer_backend': result.backend_name,
                            'writer_decode_ms': result.decode_ms,
                            **result.metadata,
                        },
                    ))
            if target is None:
                missing_reason = (
                    'missing_selection_for_transform'
                    if tgt_class == CommandTargetClass.SELECTED_TEXT
                    else 'missing_recent_target'
                )
                return self._noop(utterance_id, missing_reason, intent.kind.value)
            if target.get('target') == 'selection':
                target_class_for_policy = CommandTargetClass.SELECTED_TEXT
            elif target.get('target') == 'focused_text_before':
                target_class_for_policy = CommandTargetClass.FOCUSED_TEXT
            else:
                target_class_for_policy = CommandTargetClass.RECENT_COMMIT
            if payload.get('transform') == 'bullets':
                transformed = self._deterministic_transform('bullets', target['text'])
                out_text, replace_meta = _replace_commit_fields(target, transformed)
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.TRANSFORM_COMMIT,
                    output_text=out_text,
                    commit_mode=CommitMode.REPLACE_SELECTION,
                    selection_override=target['selection'],
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'recent_bullets', **replace_meta},
                ))
            if payload.get('transform') == 'numbered':
                transformed = self._deterministic_transform('numbered', target['text'])
                out_text, replace_meta = _replace_commit_fields(target, transformed)
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.TRANSFORM_COMMIT,
                    output_text=out_text,
                    commit_mode=CommitMode.REPLACE_SELECTION,
                    selection_override=target['selection'],
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'recent_numbered', **replace_meta},
                ))
            if payload.get('transform') == 'delete_paragraph':
                parts = re.split(r'\n\n+', target['text'])
                out = '\n\n'.join(parts[:-1]).strip() if len(parts) > 1 else ''
                out_text, replace_meta = _replace_commit_fields(target, out)
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.TRANSFORM_COMMIT,
                    output_text=out_text,
                    commit_mode=CommitMode.REPLACE_SELECTION,
                    selection_override=target['selection'],
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'delete_last_paragraph', **replace_meta},
                ))
            instr = str(payload.get('instruction') or '')
            if instr and not self._model_rewrite_allowed_for_target(pol, target_class_for_policy):
                return self._model_rewrite_blocked_noop(utterance_id, intent, target_class_for_policy)
            if instr and (pol is None or pol.allow_recent_target_commands):
                fake = WriterIntent(kind=WriterIntentKind.RECENT_MODEL_TRANSFORM, raw_text=intent.raw_text, instruction=instr)
                return self._annotate_outcome(self._run_model_transform(
                    utterance_id=utterance_id,
                    intent=fake,
                    target=target,
                    context=context,
                    memory_store=memory_store,
                    memory_snapshot=memory_snapshot,
                    memory_packet=memory_packet,
                ))
        if kind == 'translate' and tgt_text and tgt_class in (
            CommandTargetClass.SELECTED_TEXT,
            CommandTargetClass.RECENT_COMMIT,
            CommandTargetClass.FOCUSED_TEXT,
        ):
            if not self._model_rewrite_allowed_for_target(pol, tgt_class):
                return self._model_rewrite_blocked_noop(utterance_id, intent, tgt_class)
            if tgt_class == CommandTargetClass.SELECTED_TEXT:
                target = self._selected_target(context, anchor_selection)
                missing_reason = 'missing_selection_for_transform'
                xform_kind = WriterIntentKind.MODEL_TRANSFORM
            else:
                target = self._recent_target(context)
                missing_reason = 'missing_recent_target'
                xform_kind = WriterIntentKind.RECENT_MODEL_TRANSFORM
            if target is None:
                return self._noop(utterance_id, missing_reason, intent.kind.value)
            lang = str(payload.get('language', 'English'))
            fake = WriterIntent(
                kind=xform_kind,
                raw_text=intent.raw_text,
                instruction=f'Translate to {lang}. Preserve meaning.',
            )
            return self._annotate_outcome(self._run_model_transform(
                utterance_id=utterance_id,
                intent=fake,
                target=target,
                context=context,
                memory_store=memory_store,
                memory_snapshot=memory_snapshot,
                memory_packet=memory_packet,
            ))
        if kind == 'replace':
            a = str(payload.get('from', '')).strip()
            b = str(payload.get('to', '')).strip()
            if tgt_class == CommandTargetClass.SELECTED_TEXT:
                target = self._selected_target(context, anchor_selection)
            elif tgt_class in (CommandTargetClass.RECENT_COMMIT, CommandTargetClass.FOCUSED_TEXT):
                target = self._recent_target(context)
            else:
                target = None
            if target is not None and a:
                transformed = target['text'].replace(a, b)
                out_text, replace_meta = _replace_commit_fields(target, transformed)
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.TRANSFORM_COMMIT,
                    output_text=out_text,
                    commit_mode=CommitMode.REPLACE_SELECTION,
                    selection_override=target['selection'],
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'replace', **replace_meta},
                ))
            # No recent-commit target available. Promote to a persistent memory
            # rule so "replace X with Y" matches the intent users express when
            # they say it in an empty context — same semantics as the already-
            # working "replace X to Y" and "always replace X with Y" shapes.
            if memory_store is not None and a and b and len(a) < 80 and len(b) < 80:
                memory_store.add_replacement(trigger=a, replacement=b, source='voice_command')
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.MEMORY_MUTATION,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    memory_updated=True,
                    metadata={
                        'memory_action': 'add_replacement',
                        'trigger': a,
                        'replacement': b,
                        'promoted_from': 'deterministic_replace',
                    },
                ))
            if tgt_class in (CommandTargetClass.RECENT_COMMIT, CommandTargetClass.FOCUSED_TEXT):
                return self._noop(utterance_id, 'missing_recent_target', intent.kind.value)
            if tgt_class == CommandTargetClass.SELECTED_TEXT:
                return self._noop(utterance_id, 'missing_selection_for_transform', intent.kind.value)
        sem = interpret_semantic_command(
            intent.raw_text,
            mode_policy=pol,
            active_mode=self.state.mode,
            target_class=tgt_class,
            target_text=tgt_text,
        )
        if sem is not None and sem.intent_name == 'declined_semantic':
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.NOOP,
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                metadata={
                    'command_ambiguity': True,
                    'reason': sem.ambiguity_reason,
                    'target_class': tgt_class.value,
                },
            ))
        if sem is not None and sem.rewrite_instruction:
            if tgt_class in (CommandTargetClass.RECENT_COMMIT, CommandTargetClass.FOCUSED_TEXT):
                target = self._recent_target(context)
                if target is None:
                    return self._noop(utterance_id, 'missing_recent_target', intent.kind.value)
                fake = WriterIntent(
                    kind=WriterIntentKind.RECENT_MODEL_TRANSFORM,
                    raw_text=intent.raw_text,
                    instruction=sem.rewrite_instruction,
                )
                return self._annotate_outcome(self._run_model_transform(
                    utterance_id=utterance_id,
                    intent=fake,
                    target=target,
                    context=context,
                    memory_store=memory_store,
                    memory_snapshot=memory_snapshot,
                    memory_packet=memory_packet,
                ))
            if tgt_class == CommandTargetClass.SELECTED_TEXT:
                target = self._selected_target(context, anchor_selection)
                if target is None:
                    return self._noop(utterance_id, 'missing_selection_for_transform', intent.kind.value)
                fake = WriterIntent(
                    kind=WriterIntentKind.MODEL_TRANSFORM,
                    raw_text=intent.raw_text,
                    instruction=sem.rewrite_instruction,
                )
                return self._annotate_outcome(self._run_model_transform(
                    utterance_id=utterance_id,
                    intent=fake,
                    target=target,
                    context=context,
                    memory_store=memory_store,
                    memory_snapshot=memory_snapshot,
                    memory_packet=memory_packet,
                ))
            if tgt_class == CommandTargetClass.ACTIVE_UTTERANCE and partial_text.strip():
                if self.backend is None or not self.config.enable_model_transforms:
                    return self._noop(utterance_id, 'writer_backend_unavailable', intent.kind.value)
                fake = WriterIntent(
                    kind=WriterIntentKind.MODEL_TRANSFORM,
                    raw_text=intent.raw_text,
                    instruction=sem.rewrite_instruction,
                )
                style_card = _resolve_style_card(
                    memory_store,
                    context.app_category,
                    mode_policy=self.state.mode_policy,
                )
                style_prefix = _style_card_to_prompt(style_card)
                prefix_parts: list[str] = []
                if style_prefix:
                    prefix_parts.append(style_prefix.strip())
                if pol is not None and (pol.prompt_prefix or "").strip():
                    prefix_parts.append(pol.prompt_prefix.strip())
                if (self.state.writer_tone_addon or "").strip():
                    prefix_parts.append(self.state.writer_tone_addon.strip())
                prefix = "\n".join(prefix_parts) if prefix_parts else ""
                instruction = f"{prefix}\n{sem.rewrite_instruction}" if prefix else sem.rewrite_instruction
                req = WriterTransformRequest(
                    utterance_id=utterance_id,
                    instruction=instruction,
                    source_text=partial_text.strip(),
                    mode=self.state.mode,
                    target_selection=None,
                    context_payload={
                        'app_name': context.app_name,
                        'window_title': context.window_title,
                        'app_category': context.app_category,
                        'selected_text': context.selected_text,
                        'field_text_excerpt': context.field_text_excerpt,
                        'candidate_entities': context.candidate_entities,
                        'recent_clipboard': list(context.recent_clipboard) or list(
                            context.metadata.get('recent_clipboard') or []
                        ),
                        'style_card': style_card.to_dict() if style_card is not None else None,
                        'memory_packet': memory_packet,
                        'target_kind': 'active_partial',
                    },
                    metadata={'raw_command': intent.raw_text, 'semantic_intent': sem.intent_name},
                )
                result = self._rewrite_with_backend(req)
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.STATE_MUTATION,
                    output_text=result.text,
                    learn_from_commit=False,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    model_used=True,
                    metadata={
                        'command': sem.intent_name,
                        'target_class': tgt_class.value,
                        'state_action': 'active_partial_rewrite',
                        'pending_partial_text': result.text,
                        'writer_backend': result.backend_name,
                        'writer_decode_ms': result.decode_ms,
                    },
                ))
        if pol is not None and pol.command_ambiguity_policy == 'confidence_gated' and self.state.mode == WriterMode.COMMAND_MODE:
            if tgt_class == CommandTargetClass.NONE and kind not in {'insert', 'structure', 'quote'}:
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.NOOP,
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    metadata={
                        'command_ambiguity': True,
                        'reason': 'command_mode_low_confidence',
                        'utterance': intent.raw_text,
                    },
                ))
        return self._noop(utterance_id, 'command_unhandled', intent.kind.value, extra={'deterministic_command': raw_cmd})

    def _apply_structure_mode_to_dictation(self, text: str, *, context: TypedContextBundle, anchor_selection: ClientSelection | None) -> str:
        if not text or self.state.structure_mode is None:
            return text
        if anchor_selection is not None and anchor_selection.start != anchor_selection.end:
            return text
        prefix = ''
        if context.focused_text_before and not context.focused_text_before.endswith(('\n', '\r')):
            prefix = '\n'
        if self.state.structure_mode == 'bullets':
            return f"{prefix}- {text}"
        if self.state.structure_mode == 'numbered':
            marker = f"{self.state.structure_item_index}. "
            self.state.structure_item_index += 1
            return f"{prefix}{marker}{text}"
        return text

    def _next_numbered_marker(self, *, context: TypedContextBundle) -> str:
        prefix = ''
        if context.focused_text_before and not context.focused_text_before.endswith(('\n', '\r')):
            prefix = '\n'
        marker = f"{self.state.structure_item_index}. "
        self.state.structure_item_index += 1
        return prefix + marker

    def _selected_target(self, context: TypedContextBundle, anchor_selection: ClientSelection | None):
        if not context.selected_text.strip() or anchor_selection is None or anchor_selection.start == anchor_selection.end:
            return None
        return {'text': context.selected_text.strip(), 'selection': anchor_selection, 'target': 'selection'}

    def _recent_target(self, context: TypedContextBundle):
        text = str(context.metadata.get('last_committed_text') or '').strip()
        start = context.metadata.get('last_committed_start')
        end = context.metadata.get('last_committed_end')
        if text and start is not None and end is not None:
            return {'text': text, 'selection': ClientSelection(start=int(start), end=int(end)), 'target': 'recent_commit'}

        recent_clipboard = list(context.recent_clipboard) or list(
            context.metadata.get('recent_clipboard') or []
        )
        for entry in recent_clipboard:
            clip_text = ""
            if isinstance(entry, dict):
                clip_text = str(entry.get('text') or '').strip()
            else:
                clip_text = str(getattr(entry, 'text', '') or '').strip()
            if not clip_text:
                continue
            return {
                'text': clip_text,
                'selection': ClientSelection(start=0, end=len(clip_text)),
                'target': 'recent_clipboard',
            }
        focused_tail = focused_text_before_tail(context.focused_text_before)
        if focused_tail is not None:
            return {
                'text': focused_tail['text'],
                'selection': ClientSelection(start=focused_tail['start'], end=focused_tail['end']),
                'target': 'focused_text_before',
                'delete_chars': focused_tail['delete_chars'],
                'trailing_text': focused_tail['trailing_text'],
            }
        return None

    def _run_model_transform(
        self,
        *,
        utterance_id: str,
        intent,
        target,
        context: TypedContextBundle,
        memory_store: JsonMemoryStore | None,
        memory_snapshot: MemorySnapshot | None,
        memory_packet: dict | None,
    ) -> WriterOutcome:
        if self.backend is None or not self.config.enable_model_transforms:
            return self._noop(utterance_id, 'writer_backend_unavailable', intent.kind.value)

        # Resolve a style card: mode policy prefers certain categories.
        style_card = _resolve_style_card(
            memory_store,
            context.app_category,
            mode_policy=self.state.mode_policy,
        )
        style_prefix = _style_card_to_prompt(style_card)
        base_instruction = intent.instruction or 'Rewrite the selected text.'
        pol = self.state.mode_policy
        # D5: route mode_prompt_prefix and other tone/style guidance through
        # the system prompt via context_payload['mode_prompt_prefix'], the
        # same wiring the final-formatting path uses. The mlx_lm backend
        # reads this for the selection_transform_v1 task. We keep the user
        # instruction free of these prefixes so the model sees a clean
        # editing directive in the user turn.
        prefix_parts: list[str] = []
        if style_prefix:
            prefix_parts.append(style_prefix.strip())
        if pol is not None and (pol.prompt_prefix or "").strip():
            prefix_parts.append(pol.prompt_prefix.strip())
        if (self.state.writer_tone_addon or "").strip():
            prefix_parts.append(self.state.writer_tone_addon.strip())
        mode_prompt_prefix = "\n".join(prefix_parts) if prefix_parts else ""
        instruction = base_instruction

        # D2: tag selection-transform calls so the backend activates the
        # dedicated selection_transform_v1 system prompt. The recent-edit
        # path keeps the generic writer system prompt; it is not a
        # transform-on-highlighted-text request.
        is_selection_transform = target.get('target') == 'selection'
        context_payload = {
            'app_name': context.app_name,
            'window_title': context.window_title,
            'app_category': context.app_category,
            'selected_text': context.selected_text,
            'field_text_excerpt': context.field_text_excerpt,
            'candidate_entities': context.candidate_entities,
            'recent_clipboard': list(context.recent_clipboard) or list(
                context.metadata.get('recent_clipboard') or []
            ),
            'style_card': style_card.to_dict() if style_card is not None else None,
            'memory_packet': memory_packet,
            'memory_snapshot_counts': None if memory_snapshot is None else {
                'lexicon': len(memory_snapshot.lexicon),
                'replacements': len(memory_snapshot.replacements),
                'corrections': len(memory_snapshot.corrections),
                'session_entities': len(memory_snapshot.session_entities),
            },
            'target_kind': target['target'],
            'mode_prompt_prefix': mode_prompt_prefix,
        }
        if is_selection_transform:
            context_payload['task'] = 'selection_transform_v1'

        req = WriterTransformRequest(
            utterance_id=utterance_id,
            instruction=instruction,
            source_text=target['text'],
            mode=self.state.mode,
            target_selection=target['selection'],
            context_payload=context_payload,
            metadata={
                'raw_command': intent.raw_text,
                'style_card_name': style_card.name if style_card is not None else None,
            },
        )
        if style_card is not None:
            self.recorder.record(
                TraceKind.WRITER,
                'writer_style_card_applied',
                {
                    'utterance_id': utterance_id,
                    'style_card': style_card.name,
                    'app_category': context.app_category,
                },
            )
        result = self._rewrite_with_backend(req)
        out_text, replace_meta = _replace_commit_fields(target, result.text)
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.TRANSFORM_COMMIT,
            output_text=out_text,
            commit_mode=CommitMode.REPLACE_SELECTION,
            selection_override=target['selection'],
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            model_used=True,
            metadata={
                'instruction': req.instruction,
                'writer_backend': result.backend_name,
                'writer_decode_ms': result.decode_ms,
                'style_card': style_card.name if style_card is not None else None,
                **result.metadata,
                **replace_meta,
            },
        )

    def _run_final_formatting_if_needed(
        self,
        *,
        utterance_id: str,
        text: str,
        context: TypedContextBundle,
        memory_store: JsonMemoryStore | None,
    ) -> WriterTransformResult | None:
        pol = self.state.mode_policy
        policy = getattr(pol, "final_formatting_policy", None)
        if self.state.mode == WriterMode.VERBATIM:
            return None
        if not should_run_final_formatting(policy, app_category=context.app_category):
            return None
        style_card = _resolve_style_card(
            memory_store,
            context.app_category,
            mode_policy=self.state.mode_policy,
        )
        packet = FormattingPacket(
            utterance_id=utterance_id,
            corrected_text=text,
            app_name=context.app_name,
            app_category=context.app_category,
            window_title=context.window_title,
            mode_name=self.state.mode_selection.effective_mode if self.state.mode_selection is not None else self.state.mode.value,
            final_formatting_policy=str(policy or "minimal"),
            style_card=style_card.to_dict() if style_card is not None else None,
            focused_text_before=context.focused_text_before[-1000:],
            focused_text_after=context.focused_text_after[:600],
            selected_text_excerpt=context.selected_text[:1200],
            writer_tone_addon=self.state.writer_tone_addon,
            mode_prompt_prefix=(
                (getattr(self.state.mode_policy, "prompt_prefix", "") or None)
                if self.state.mode_policy is not None
                else None
            ),
            metadata={
                'candidate_entities': list(context.candidate_entities or [])[:32],
                'recent_screen_terms': list(context.candidate_entities or [])[:32],
            },
        )
        formatter = FinalFormatter(
            backend=self.backend if self.config.enable_model_transforms else None,
            acquire=self.backend_acquire,
            release=self.backend_release,
        )
        self.recorder.record(
            TraceKind.WRITER,
            'oneshot_final_formatting_started',
            {
                'utterance_id': utterance_id,
                'policy': packet.final_formatting_policy,
                'mode_name': packet.mode_name,
                'app_category': context.app_category,
            },
        )
        try:
            result = formatter.format(packet)
        except Exception as exc:  # noqa: BLE001
            self.recorder.record(
                TraceKind.WRITER,
                'writer_final_formatting_failed',
                {
                    'utterance_id': utterance_id,
                    'error': f'{type(exc).__name__}: {exc}',
                },
            )
            return None
        rejection = getattr(formatter, "last_rejection", None)
        if result is None and rejection is not None:
            self.recorder.record(
                TraceKind.WRITER,
                'writer_final_formatting_rejected',
                {
                    'utterance_id': utterance_id,
                    'policy': packet.final_formatting_policy,
                    'mode_name': packet.mode_name,
                    'app_category': context.app_category,
                    **rejection,
                },
            )
        if result is not None:
            self.recorder.record(
                TraceKind.WRITER,
                'oneshot_final_formatting_ok',
                {
                    'utterance_id': utterance_id,
                    'policy': packet.final_formatting_policy,
                    'backend': result.backend_name,
                    'deterministic': bool(result.metadata.get('deterministic')),
                },
            )
        if result is not None and style_card is not None:
            self.recorder.record(
                TraceKind.WRITER,
                'writer_style_card_applied',
                {
                    'utterance_id': utterance_id,
                    'style_card': style_card.name,
                    'app_category': context.app_category,
                    'target': 'final_formatting',
                },
            )
        return result

    def _insert_context_field(
        self,
        utterance_id: str,
        intent,
        context: TypedContextBundle,
        anchor_selection: ClientSelection | None,
    ) -> WriterOutcome:
        field = intent.context_field or ''
        if field == 'focused_file_path':
            value = (context.focused_file_path or '').strip()
        elif field == 'symbol_under_cursor':
            value = (context.symbol_under_cursor or '').strip()
        else:
            return self._noop(utterance_id, f'unknown_context_field:{field}', intent.kind.value)
        if not value:
            return self._noop(utterance_id, f'context_field_empty:{field}', intent.kind.value)
        sel = None
        if anchor_selection is not None:
            sel = ClientSelection(start=anchor_selection.end, end=anchor_selection.end)
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.DIRECT_COMMIT,
            output_text=value,
            commit_mode=CommitMode.INSERT_AT_CARET,
            selection_override=sel,
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            deterministic_used=True,
            metadata={'context_field': field, 'value_len': len(value)},
        )

    def _noop(self, utterance_id: str, reason: str, intent_kind: str, extra: dict | None = None) -> WriterOutcome:
        return self._annotate_outcome(WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.NOOP,
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            metadata={'reason': reason, 'intent_kind': intent_kind, **(extra or {})},
        ))

    def _annotate_outcome(self, outcome: WriterOutcome) -> WriterOutcome:
        sel = self.state.mode_selection
        pol = self.state.mode_policy
        if sel is not None:
            outcome.effective_mode = sel.effective_mode
            outcome.mode_source = sel.mode_source.value
            outcome.manual_mode_name = sel.manual_mode_name
            outcome.custom_mode_name = sel.custom_mode_name
        if pol is not None:
            outcome.base_mode = pol.base_mode
        trace = {
            'effective_mode': outcome.effective_mode,
            'mode_source': outcome.mode_source,
            'base_mode': outcome.base_mode,
            'manual_mode_name': outcome.manual_mode_name,
            'custom_mode_name': outcome.custom_mode_name,
        }
        outcome.metadata = {**outcome.metadata, 'mode_trace': trace}
        return outcome

    @staticmethod
    def _writer_mode_from_string(name: str) -> WriterMode:
        key = (name or '').strip()
        if key == 'command_mode':
            return WriterMode.COMMAND_MODE
        try:
            return WriterMode(key)
        except ValueError:
            return WriterMode.DEFAULT_SURFACE

    def _expand_snippets_for_commit(
        self,
        text: str,
        *,
        context: Any,
        memory_store: Any,
    ) -> tuple[str, bool, list[str]]:
        """Expand user snippet triggers for a committed dictation.

        Shared by the deterministic pipeline AND the AI dictation-editor path
        so a saved snippet expands regardless of which lane produced the final
        text. (The editor path used to return before the deterministic
        expansion step, so snippets never came up in production where the
        editor is enabled.) No-op in VERBATIM mode, on code/terminal surfaces,
        when there's no snippet store, or when the active mode policy yields no
        snippet scopes. Returns ``(text, expanded, scopes)``.
        """
        if (
            self.state.mode == WriterMode.VERBATIM
            or (getattr(context, "app_category", "") or "").strip().lower() in ("code", "terminal")
            or memory_store is None
            or getattr(memory_store, "snippets", None) is None
            or not text
        ):
            return text, False, []
        scopes = self._snippet_scopes_for_policy(getattr(context, "app_category", None))
        if not scopes:
            return text, False, []
        before = text
        for scope in scopes:
            text = expand_snippets(text, resolver=memory_store.snippets, scope=scope)
        return text, text != before, scopes

    def _snippet_scopes_for_policy(self, app_category: str | None) -> list[str]:
        pol = self.state.mode_policy
        if pol is None:
            return [_snippet_scope_from_category(app_category), 'global']
        s = pol.snippet_scope_policy
        if s in {'none', 'none_unless_invoked'}:
            return []
        if s == 'surface_plus_global':
            return list(dict.fromkeys([_snippet_scope_from_category(app_category), 'global']))
        if s == 'messaging_plus_global':
            return ['messaging', 'global']
        if s == 'email_plus_global':
            return ['email', 'global']
        if s == 'docs_plus_global':
            return ['docs', 'global']
        if s == 'all_scopes':
            return ['messaging', 'email', 'docs', 'global']
        if s.startswith('custom:'):
            return [s.split(':', 1)[1].strip(), 'global']
        return [_snippet_scope_from_category(app_category), 'global']

    def _direct_snippet_insert(
        self,
        text: str,
        *,
        memory_store: JsonMemoryStore | None,
        app_category: str | None,
    ) -> tuple[str, dict[str, Any]] | None:
        if self.state.mode == WriterMode.VERBATIM:
            return None
        if not text or memory_store is None or getattr(memory_store, "snippets", None) is None:
            return None
        if (app_category or '').strip().lower() in {'code', 'terminal'}:
            return None
        match = _DIRECT_SNIPPET_INSERT_RE.match(text)
        if match is None:
            return None
        raw_trigger = re.sub(r"\s+", " ", (match.group("trigger") or "").strip(" .,!?:;\"'"))
        if not raw_trigger:
            return None
        trigger_candidates = _direct_snippet_trigger_candidates(raw_trigger)
        if not trigger_candidates:
            return None
        scopes = self._snippet_scopes_for_policy(app_category)
        for scope in scopes:
            for trigger in trigger_candidates:
                snippet = memory_store.snippets.resolve(trigger, scope=scope)
                body = str(getattr(snippet, "body", "") or "") if snippet is not None else ""
                if body:
                    return body, {
                        'trigger': trigger,
                        'spoken_trigger': raw_trigger,
                        'scope': scope,
                        'body_chars': len(body),
                    }
        return None

    def _deterministic_transform(self, transform_kind: str, text: str) -> str:
        if transform_kind == 'bullets':
            return render_bullets(text)
        if transform_kind == 'numbered':
            return render_numbered(text)
        if transform_kind == 'uppercase':
            return render_uppercase(text)
        if transform_kind == 'lowercase':
            return render_lowercase(text)
        if transform_kind == 'title_case':
            return render_title_case(text)
        return text


def _resolve_style_card(
    memory_store: "JsonMemoryStore | None",
    app_category: str | None,
    *,
    mode_policy: "ModePolicy | None" = None,
) -> None:
    """Style cards were removed as a memory category.

    The function is kept as a no-op stub so the writer's call sites
    don't need to change shape; every caller treats a ``None`` return
    as "no style preset applied" and that path is well-exercised. If
    we ever want named writing presets back, they should be modeled
    as references on Modes / Per-app rules (see the kill-styles
    discussion), not as a separate memory category.
    """
    _ = (memory_store, app_category, mode_policy)
    return None


def _style_card_to_prompt(card: object | None) -> str:
    """Returns the empty string. See :func:`_resolve_style_card`."""
    _ = card
    return ""


def _approx_word_count(text: str) -> int:
    return len((text or "").split())


def _is_no_touch_context(context: TypedContextBundle) -> bool:
    category = str(getattr(context, "app_category", "") or "").strip().lower()
    return category in {"code", "terminal", "developer_tools"}


def _explicit_memory_learning_requested(text: str) -> bool:
    value = str(text or "")
    return bool(
        re.search(r"\b(?:teach|remember|learn)\s+(?:juno\b)?", value, flags=re.IGNORECASE)
        or re.search(r"\b(?:should|must)\s+be\s+(?:remembered|learned)\b", value, flags=re.IGNORECASE)
        or re.search(r"\badd\s+.+\s+to\s+(?:my\s+)?(?:dictionary|lexicon|vocab(?:ulary)?|wordlist)\b", value, flags=re.IGNORECASE)
    )


def _explicit_vocab_learning_requested(text: str) -> bool:
    value = str(text or "")
    if re.search(
        r"\bremember\s+(?:that\s+)?[\"']?.+?[\"']?\s+(?:is|means|equals|should\s+be)\s+[\"']?.+?[\"']?\s*$",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:always\s+)?(?:replace|change|use)\s+[\"']?.+?[\"']?\s+(?:with|as|to)\s+[\"']?.+?[\"']?\s*$",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    return _explicit_memory_learning_requested(value)


def _normalize_turn_plan_memory_candidate(raw: Any, *, source_text: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    canonical = _clean_memory_term(
        raw.get("canonical")
        or raw.get("canonical_form")
        or raw.get("term")
        or raw.get("surface")
    )
    surface = _clean_memory_term(raw.get("surface") or raw.get("term") or canonical)
    if not canonical:
        return None
    if _memory_candidate_overlaps_source_spelling(canonical, source_text=source_text):
        return None
    if not _memory_candidate_allowed(canonical, source_text=source_text):
        return None
    if not span_present(canonical, source_text) and not span_present(surface, source_text):
        return None
    aliases = []
    raw_aliases = raw.get("aliases")
    if isinstance(raw_aliases, list):
        for alias in raw_aliases:
            cleaned = _clean_memory_term(alias)
            if (
                cleaned
                and _memory_alias_allowed(cleaned, canonical=canonical, source_text=source_text)
                and span_present(cleaned, source_text)
            ):
                aliases.append(cleaned)
    pronunciation_hint = str(raw.get("pronunciation_hint") or raw.get("phonetic") or "").strip() or None
    return {
        "term": surface or canonical,
        "canonical_form": canonical,
        "aliases": list(dict.fromkeys(aliases)),
        "pronunciation_hint": pronunciation_hint,
    }


def _memory_learning_source_text(final_text: str, partial_text: str | None) -> str:
    final = str(final_text or "").strip()
    partial = str(partial_text or "").strip()
    if not partial or not _explicit_vocab_learning_requested(partial):
        return final
    if not _explicit_vocab_learning_requested(final):
        return partial
    partial_spelled = len(re.findall(r"\bspelled\b", partial, flags=re.IGNORECASE))
    final_spelled = len(re.findall(r"\bspelled\b", final, flags=re.IGNORECASE))
    if partial_spelled >= final_spelled:
        return partial
    return final


def _normalize_spelled_learning_text(source_text: str) -> str:
    text = re.sub(r"\s+", " ", str(source_text or "")).strip()
    if not text:
        return ""
    if re.search(r"\bspelled\b", text, flags=re.IGNORECASE):
        text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _spelled_vocab_memory_candidates(source_text: str) -> list[dict[str, Any]]:
    if not _explicit_vocab_learning_requested(source_text):
        return []
    text = _normalize_spelled_learning_text(source_text)
    if not text:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b(?:is\s+)?spelled\b", text, flags=re.IGNORECASE):
        surface = _spelled_vocab_surface_before_marker(text[: match.start()])
        letters, raw_spelling = _consume_spelled_letters(text[match.end() :], surface=surface)
        if len(letters) < 2:
            continue
        canonical = _canonical_from_spelled_vocab_surface(surface, letters)
        if not canonical:
            continue
        candidate = {
            "term": surface or canonical,
            "canonical_form": canonical,
            "aliases": [surface] if surface and _memory_term_key(surface) != _memory_term_key(canonical) else [],
            "pronunciation_hint": f"spelled {raw_spelling}",
        }
        key = _memory_term_key(canonical)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _explicit_vocab_source_phrase_candidates(source_text: str) -> list[dict[str, Any]]:
    if not _explicit_vocab_learning_requested(source_text):
        return []
    text = _normalize_spelled_learning_text(source_text)
    if not text:
        return []
    text = re.sub(
        r"(?i)^\s*(?:teach|remember|learn|save|add)\s+(?:juno\s+)?"
        r"(?:(?:these|this|the)\s+)?(?:terms?|words?|names?|vocabulary|vocab)?\s*[:.]?\s*",
        "",
        text,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_part in re.split(r"[,.;]\s*", text):
        for part in _explicit_vocab_source_phrase_parts(raw_part):
            if not part or len(part.split()) > 5:
                continue
            if not _explicit_vocab_phrase_has_term_signal(part):
                continue
            if not _memory_candidate_allowed(part, source_text=source_text):
                continue
            term_key = _memory_term_key(part)
            if term_key and term_key not in seen:
                seen.add(term_key)
                out.append({"term": part, "canonical_form": part, "aliases": []})
    return out


def _explicit_vocab_source_phrase_parts(raw_part: str) -> list[str]:
    part = _clean_memory_term(raw_part)
    if not part or re.search(r"(?i)\bspelled\b", part):
        return []
    part = re.sub(r"(?i)^\s*(?:and|or)\s+", "", part).strip()
    part = re.sub(
        r"(?i)\b(?:is|are)\s+(?:a\s+|an\s+)?(?:term|word|name)\b.*$",
        "",
        part,
    ).strip()
    if not part:
        return []
    pieces = [
        _clean_memory_term(piece)
        for piece in re.split(r"\b(?:and|or)\b", part, flags=re.IGNORECASE)
    ]
    pieces = [piece for piece in pieces if piece]
    if len(pieces) > 1 and all(_explicit_vocab_phrase_has_term_signal(piece) for piece in pieces):
        return pieces
    return [part]


def _explicit_vocab_phrase_has_term_signal(value: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9._-]*", str(value or ""))
    if not tokens:
        return False
    for token in tokens:
        if token.isupper() and len(token) >= 2:
            return True
        if re.search(r"[a-z][A-Z]", token):
            return True
        if token[:1].isupper() and token[1:].islower() and not common_english_single_word(token):
            return True
        if re.search(r"\d", token):
            return True
    return False


def _spelled_vocab_surface_before_marker(before_marker: str) -> str:
    value = re.sub(r"\s+", " ", str(before_marker or "")).strip(" ,.;:-")
    if not value:
        return ""
    if re.search(r"\bspelled\b", value, flags=re.IGNORECASE):
        value = re.split(r"\bspelled\b", value, flags=re.IGNORECASE)[-1].strip(" ,.;:-")
    parts = re.split(r"[,.;:]|\b(?:and|or)\b", value, flags=re.IGNORECASE)
    surface = parts[-1] if parts else value
    surface = re.sub(
        r"(?i)\b(?:teach|remember|learn)\s+(?:juno\s+)?(?:these|this|the)?\s*"
        r"(?:terms?|words?|names?|vocabulary|vocab)?\b",
        " ",
        surface,
    )
    surface = re.sub(
        r"(?i)\b(?:add|save)\s+(?:these|this|the)?\s*(?:terms?|words?|names?)\b",
        " ",
        surface,
    )
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9._-]*", surface)
    if not tokens:
        return ""
    letter_suffix: list[str] = []
    for token in reversed(tokens):
        if len(token) == 1 and token.isalpha():
            letter_suffix.append(token)
            continue
        break
    if len(letter_suffix) >= 2:
        suffix = list(reversed(letter_suffix))
        return " ".join(suffix[-8:])
    return " ".join(tokens[-4:])


def _consume_spelled_letters(after_marker: str, *, surface: str = "") -> tuple[list[str], str]:
    text = str(after_marker or "")
    idx = 0
    letters: list[str] = []
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n-":
            idx += 1
        if idx >= len(text):
            break
        match = re.match(r"[A-Za-z]+", text[idx:])
        if match is None:
            break
        token = match.group(0)
        if len(token) != 1:
            break
        letters.append(token.upper())
        idx += len(token)
    letters = _anchor_spelled_letters_to_surface(letters, surface=surface)
    return letters, " ".join(letters)


def _anchor_spelled_letters_to_surface(letters: list[str], *, surface: str) -> list[str]:
    if len(letters) < 3:
        return letters
    anchors = _spelled_surface_anchor_keys(surface)
    if not anchors:
        return letters
    best_len = 0
    best_score: tuple[int, int] | None = None
    joined = "".join(letters).casefold()
    for anchor in anchors:
        if not anchor:
            continue
        max_distance = max(1, min(2, round(len(anchor) * 0.25)))
        if (
            len(letters) > len(anchor)
            and joined[: len(anchor)] == anchor
            and joined[len(anchor) : len(anchor) + 1] in {"a", "e", "i", "o", "u", "y"}
            and max_distance >= 1
        ):
            n = len(anchor) + 1
            score = (0, -n)
            if best_score is None or score < best_score:
                best_score = score
                best_len = n
        min_len = max(2, len(anchor) - max_distance)
        max_len = min(len(letters), len(anchor) + max_distance)
        for n in range(min_len, max_len + 1):
            prefix = joined[:n]
            distance = _small_edit_distance(prefix, anchor, max_distance=max_distance)
            if distance > max_distance:
                continue
            score = (distance, -n)
            if best_score is None or score < best_score:
                best_score = score
                best_len = n
    if best_len >= 2 and best_len < len(letters):
        return []
    return letters


def _spelled_surface_anchor_keys(surface: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9._-]*", str(surface or ""))
    keys: list[str] = []

    def add(value: str) -> None:
        cleaned = re.sub(r"[^a-z0-9]+", "", value.casefold())
        if cleaned and cleaned not in keys:
            keys.append(cleaned)

    if tokens:
        add("".join(tokens))
        if len(tokens[0]) > 1:
            add(tokens[0])
    return keys


def _small_edit_distance(left: str, right: str, *, max_distance: int) -> int:
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    previous = list(range(len(right) + 1))
    for i, ch_left in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, ch_right in enumerate(right, start=1):
            cost = 0 if ch_left == ch_right else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def _memory_candidate_overlaps_spelled_acronym_tail(
    candidate: dict[str, Any],
    *,
    spelled_keys: set[str],
) -> bool:
    if not spelled_keys:
        return False
    canonical = str(candidate.get("canonical_form") or candidate.get("term") or "")
    candidate_key = _memory_term_key(canonical)
    if not candidate_key:
        return False
    for spelled_key in spelled_keys:
        if candidate_key == spelled_key:
            continue
        if candidate_key.startswith(spelled_key) and not candidate_key.startswith(f"{spelled_key} "):
            compressed_tail = candidate_key[len(spelled_key) :].strip()
            if compressed_tail and (
                _looks_like_acronym_word(compressed_tail)
            ):
                return True
        if not candidate_key.startswith(f"{spelled_key} "):
            continue
        tail = candidate_key[len(spelled_key) :].strip()
        tail_words = tail.split()
        if tail_words and all(_looks_like_acronym_word(word) for word in tail_words):
            return True
    return False


def _memory_candidate_overlaps_source_spelling(candidate: str, *, source_text: str) -> bool:
    spelled_keys = {
        _memory_term_key(item.get("canonical_form") or item.get("term"))
        for item in _spelled_vocab_memory_candidates(source_text)
    }
    spelled_keys.discard("")
    if not spelled_keys:
        return False
    probe = {
        "term": candidate,
        "canonical_form": candidate,
        "aliases": [],
        "pronunciation_hint": None,
    }
    return _memory_candidate_overlaps_spelled_acronym_tail(probe, spelled_keys=spelled_keys)


def _memory_candidate_contained_by_longer_candidate(
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> bool:
    key = _memory_term_key(candidate.get("canonical_form") or candidate.get("term"))
    words = key.split()
    if not words:
        return False
    for other in candidates:
        if other is candidate:
            continue
        other_key = _memory_term_key(other.get("canonical_form") or other.get("term"))
        other_words = other_key.split()
        if len(other_words) <= len(words):
            continue
        for idx in range(0, len(other_words) - len(words) + 1):
            if other_words[idx : idx + len(words)] == words:
                return True
    return False


def _looks_like_acronym_word(word: str) -> bool:
    value = re.sub(r"[^a-z0-9]", "", str(word or "").casefold())
    if len(value) < 2 or len(value) > 6:
        return False
    return not re.search(r"[aeiouy]", value)



def _canonical_from_spelled_vocab_surface(surface: str, letters: list[str]) -> str:
    spelled = "".join(ch for ch in letters if ch).upper()
    if not spelled:
        return ""
    cleaned_surface = _clean_memory_term(surface)
    if cleaned_surface:
        pieces = cleaned_surface.split()
        if pieces and all(len(piece) == 1 and piece.isalpha() for piece in pieces):
            surfaced = "".join(pieces).upper()
            if surfaced == spelled:
                return spelled
            return ""
        if len(pieces) > 1 and all(re.fullmatch(r"[A-Z0-9]{2,}", piece) for piece in pieces):
            surfaced = "".join(pieces).upper()
            if surfaced == spelled:
                return " ".join(pieces)
        if pieces and _memory_term_key(pieces[0]) == _memory_term_key(spelled):
            return " ".join([_preferred_spelled_piece_casing(pieces[0], spelled), *pieces[1:]])
        leading_initials = []
        for piece in pieces:
            if len(piece) == 1 and piece.isalpha():
                leading_initials.append(piece)
                continue
            break
        if leading_initials and len(pieces) > len(leading_initials):
            initials = "".join(leading_initials).casefold()
            target = spelled.casefold()
            max_distance = max(1, min(2, len(target) - len(initials)))
            if _small_edit_distance(initials, target[: len(initials)], max_distance=max_distance) <= max_distance:
                return " ".join([
                    _spelled_letters_default_canonical(spelled),
                    *pieces[len(leading_initials) :],
                ])
            return ""
        if len(pieces) > 1 and _spelled_letters_replace_surface_piece(pieces[0], spelled):
            return " ".join([_spelled_letters_default_canonical(spelled), *pieces[1:]])
        if _memory_term_key(cleaned_surface) == _memory_term_key(spelled):
            return _preferred_spelled_piece_casing(cleaned_surface, spelled)
    return _spelled_letters_default_canonical(spelled)


def _spelled_letters_replace_surface_piece(surface_piece: str, spelled: str) -> bool:
    piece = re.sub(r"[^A-Za-z0-9]+", "", str(surface_piece or ""))
    target = re.sub(r"[^A-Za-z0-9]+", "", str(spelled or ""))
    if len(piece) < 3 or len(target) < 3:
        return False
    if piece[:1].casefold() != target[:1].casefold():
        return False
    if abs(len(piece) - len(target)) > 2:
        return False
    return True


def _preferred_spelled_piece_casing(surface_piece: str, spelled: str) -> str:
    piece = str(surface_piece or "").strip()
    if piece and piece.isupper():
        return spelled
    if piece and re.search(r"[a-z][A-Z]", piece):
        return piece
    if piece and piece[:1].isupper():
        return piece[:1].upper() + piece[1:]
    return _spelled_letters_default_canonical(spelled)


def _spelled_letters_default_canonical(spelled: str) -> str:
    value = str(spelled or "").strip().upper()
    if not value:
        return ""
    if len(value) <= 4 and not re.search(r"[AEIOUY]", value[1:], flags=re.IGNORECASE):
        return value
    return value[:1] + value[1:].lower()


def _memory_candidate_allowed(term: str, *, source_text: str) -> bool:
    value = _clean_memory_term(term)
    if not value or not learned_term_allowed(value):
        return False
    key = _memory_term_key(value)
    if key in _MEMORY_COMMAND_TERMS:
        return False
    words = value.split()
    if words and all(common_english_single_word(word) for word in words):
        return False
    if key.startswith("teach juno") or key.endswith(" product terms"):
        return False
    if len(key.split()) > 8:
        return False
    source_key = _memory_term_key(source_text)
    if source_key and source_key == key:
        return False
    return True


def _memory_alias_allowed(alias: str, *, canonical: str, source_text: str) -> bool:
    value = _clean_memory_term(alias)
    if not value or not learned_term_allowed(value):
        return False
    key = _memory_term_key(value)
    if not key or key in _MEMORY_COMMAND_TERMS or key == _memory_term_key(canonical):
        return False
    if key.startswith("teach juno") or key.endswith(" product terms"):
        return False
    if len(key.split()) > 8:
        return False
    source_key = _memory_term_key(source_text)
    if source_key and source_key == key:
        return False
    return True


def _clean_memory_term(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" ,.;:-\"'"))
    text = re.sub(r"^(?:teach|remember|learn)\s+juno\s+(?:that\s+|these\s+|this\s+)?", "", text, flags=re.IGNORECASE)
    return text.strip(" ,.;:-\"'")


def _memory_term_key(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


_DIRECT_SNIPPET_INSERT_RE = re.compile(
    r"^\s*(?:insert|paste|use|add)\s+(?:the\s+)?(?P<trigger>.+?)\s*[.!?]?\s*$",
    flags=re.IGNORECASE,
)


def _direct_snippet_trigger_candidates(raw_trigger: str) -> list[str]:
    trigger = re.sub(r"\s+", " ", (raw_trigger or "").strip())
    if not trigger:
        return []
    words = trigger.casefold().split()
    candidates: list[str] = []

    def add(value: str) -> None:
        cleaned = re.sub(r"\s+", " ", value.strip(" .,!?:;\"'"))
        if cleaned and cleaned.casefold() not in {item.casefold() for item in candidates}:
            candidates.append(cleaned)

    add(trigger)
    if "snippet" in words:
        add(re.sub(r"(?i)^snippet\s+", "", trigger))
        add(re.sub(r"(?i)\s+snippet$", "", trigger))
    return candidates


def _json_object_from_model_text(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if "\n" in raw:
            raw = raw.split("\n", 1)[1].strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _snippet_scope_from_category(category: str | None) -> str:
    """Translate a context ``app_category`` into a snippet scope.

    Snippets are stored with scopes that mirror the app-category
    values (``messaging``, ``email``, ``docs``, ``code``, ``terminal``,
    ``forms``, ``global``). ``code`` / ``terminal`` surfaces should
    not trigger snippets at all — typing ``brb`` into a terminal
    should stay as ``brb`` — so we return ``global`` and rely on the
    snippet store to only surface entries the user explicitly scoped
    there.
    """
    if not category:
        return "global"
    normalized = category.strip().lower()
    if normalized in {"code", "terminal"}:
        return "global"
    known = {
        AppCategory.MESSAGING.value,
        AppCategory.EMAIL.value,
        AppCategory.DOCS.value,
        AppCategory.FORMS.value,
    }
    if normalized in known:
        return normalized
    return "global"
