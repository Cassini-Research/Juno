from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from juno_v2.commands.resolver import resolve_command_target
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
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.writer.backends.base import WriterBackend
from juno_v2.writer.config import WriterConfig
from juno_v2.writer.deterministic import (
    AppCategory,
    expand_snippets,
    render_bullets,
    render_lowercase,
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
    ) -> WriterOutcome:
        if mode_policy is not None:
            self.state.mode_policy = mode_policy
        if mode_selection is not None:
            self.state.mode_selection = mode_selection
            self.state.mode = self._writer_mode_from_string(mode_selection.effective_mode)
        elif mode_policy is not None:
            self.state.mode = self._writer_mode_from_string(mode_policy.base_mode)
        self.state.writer_tone_addon = (writer_tone_addon or "").strip() or None

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
            text = final_text
            cleanup_meta = {'pipeline': 'already_adjudicated_passthrough'}
            model_used = False
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
            return self._annotate_outcome(WriterOutcome(
                utterance_id=utterance_id,
                action=WriterActionKind.TRANSFORM_COMMIT,
                output_text=out,
                commit_mode=CommitMode.REPLACE_SELECTION,
                selection_override=target['selection'],
                writer_mode=self.state.mode,
                structure_mode=self.state.structure_mode,
                deterministic_used=True,
                metadata={'command': 'delete_last_sentence_recent', 'target': target['target']},
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
            if payload.get('transform') == 'bullets':
                transformed = self._deterministic_transform('bullets', target['text'])
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.TRANSFORM_COMMIT,
                    output_text=transformed,
                    commit_mode=CommitMode.REPLACE_SELECTION,
                    selection_override=target['selection'],
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'recent_bullets', 'target': target['target']},
                ))
            if payload.get('transform') == 'numbered':
                transformed = self._deterministic_transform('numbered', target['text'])
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.TRANSFORM_COMMIT,
                    output_text=transformed,
                    commit_mode=CommitMode.REPLACE_SELECTION,
                    selection_override=target['selection'],
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'recent_numbered', 'target': target['target']},
                ))
            if payload.get('transform') == 'delete_paragraph':
                parts = re.split(r'\n\n+', target['text'])
                out = '\n\n'.join(parts[:-1]).strip() if len(parts) > 1 else ''
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.TRANSFORM_COMMIT,
                    output_text=out,
                    commit_mode=CommitMode.REPLACE_SELECTION,
                    selection_override=target['selection'],
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'delete_last_paragraph', 'target': target['target']},
                ))
            instr = str(payload.get('instruction') or '')
            if instr and not self._model_rewrite_allowed_for_target(pol, tgt_class):
                return self._model_rewrite_blocked_noop(utterance_id, intent, tgt_class)
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
            elif tgt_class == CommandTargetClass.RECENT_COMMIT:
                target = self._recent_target(context)
            else:
                target = None
            if target is not None and a:
                transformed = target['text'].replace(a, b)
                return self._annotate_outcome(WriterOutcome(
                    utterance_id=utterance_id,
                    action=WriterActionKind.TRANSFORM_COMMIT,
                    output_text=transformed,
                    commit_mode=CommitMode.REPLACE_SELECTION,
                    selection_override=target['selection'],
                    writer_mode=self.state.mode,
                    structure_mode=self.state.structure_mode,
                    deterministic_used=True,
                    metadata={'command': 'replace', 'target': target['target']},
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
            if tgt_class == CommandTargetClass.RECENT_COMMIT:
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
            if tgt_class == CommandTargetClass.RECENT_COMMIT:
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
        if not text or start is None or end is None:
            return None
        return {'text': text, 'selection': ClientSelection(start=int(start), end=int(end)), 'target': 'recent_commit'}

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
        return WriterOutcome(
            utterance_id=utterance_id,
            action=WriterActionKind.TRANSFORM_COMMIT,
            output_text=result.text,
            commit_mode=CommitMode.REPLACE_SELECTION,
            selection_override=target['selection'],
            writer_mode=self.state.mode,
            structure_mode=self.state.structure_mode,
            model_used=True,
            metadata={
                'instruction': req.instruction,
                'writer_backend': result.backend_name,
                'writer_decode_ms': result.decode_ms,
                'target': target['target'],
                'style_card': style_card.name if style_card is not None else None,
                **result.metadata,
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
