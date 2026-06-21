from __future__ import annotations

from dataclasses import dataclass, field

from juno_v2.contracts.commit import ActiveCommitSession, CommitAnchor, CommitDecision, CommitStatus
from juno_v2.contracts.final import FinalTranscript
from juno_v2.contracts.insertion import InsertionRequest
from juno_v2.contracts.preview import PreviewEmission
from juno_v2.contracts.tracing import TraceKind
from juno_v2.memory.hallucination import (
    looks_like_silence_hallucination,
    strip_adjacent_low_signal_word_duplicates,
    strip_repeated_stock_hallucination_tail,
    strip_trailing_silence_hallucination,
)
from juno_v2.memory.store import _looks_like_hallucination
from juno_v2.contracts.workbench import (
    ClientSelection,
    CommitMode,
    FinalCandidateRequest,
    FinalCommitRequest,
    PartialCommitRequest,
    SyncClientStateRequest,
)
from juno_v2.insertion.base import ActiveAppInserter
from juno_v2.workbench.store import WorkbenchStore


@dataclass(slots=True)
class CommitController:
    store: WorkbenchStore
    inserter: ActiveAppInserter | None = None
    _sessions: dict[str, ActiveCommitSession] = field(default_factory=dict, init=False)
    _current_utterance_id: str | None = field(default=None, init=False)

    @property
    def recorder(self):
        return self.store.recorder

    @property
    def active(self) -> ActiveCommitSession | None:
        if self._current_utterance_id is None:
            return None
        return self._sessions.get(self._current_utterance_id)

    def sync_client_state(self, req: SyncClientStateRequest):
        with self.store.lock:
            state = self.store.sync_client_state(req)
            self._refresh_conflict_marker()
            return state

    def reset_all(self) -> None:
        with self.store.lock:
            self._sessions.clear()
            self._current_utterance_id = None
            self.store.state.active_utterance_id = None
            self.store.state.anchor_revision = None
            self.store.state.active_commit_mode = None
            self.store.state.pending_commit = False
            self.store.state.commit_conflict = None
            self.store.clear_partial()
            self.store.clear_final_candidate()

    def begin_utterance(self, utterance_id: str, commit_mode: CommitMode | None = None) -> None:
        with self.store.lock:
            session = self._sessions.get(utterance_id)
            if session is None:
                selection = self.store.state.selection.normalized(len(self.store.state.buffer_text))
                resolved_mode = commit_mode or self._resolve_mode(selection)
                session = ActiveCommitSession(
                    utterance_id=utterance_id,
                    anchor=CommitAnchor(
                        buffer_text=self.store.state.buffer_text,
                        selection=selection,
                        revision=self.store.state.revision,
                        commit_mode=resolved_mode,
                    ),
                )
                self._sessions[utterance_id] = session
                self.recorder.record(
                    TraceKind.COMMIT,
                    "utterance_began",
                    {
                        "utterance_id": utterance_id,
                        "commit_mode": resolved_mode.value,
                        "anchor_revision": session.anchor.revision,
                        "selection_start": selection.start,
                        "selection_end": selection.end,
                        "parallel_session_count": len(self._sessions),
                    },
                )
            elif commit_mode is not None:
                session.anchor.commit_mode = commit_mode
            self._current_utterance_id = utterance_id
            self._publish_session_state(utterance_id, clear_preview=True)

    def apply_preview(self, emission: PreviewEmission) -> None:
        with self.store.lock:
            session = self._ensure_session(emission.utterance_id)
            session.latest_partial_text = emission.text
            if self._current_utterance_id == emission.utterance_id:
                self.store.apply_partial(PartialCommitRequest(text=emission.text))
                self._refresh_conflict_marker()
            self.recorder.record(
                TraceKind.COMMIT,
                "preview_applied_to_commit_session",
                {
                    "utterance_id": emission.utterance_id,
                    "text_length": len(emission.text),
                    "is_final_preview": emission.is_final,
                    "stability_delta_chars": emission.stability_delta_chars,
                },
            )

    def stage_final(self, transcript: FinalTranscript) -> None:
        with self.store.lock:
            session = self._ensure_session(transcript.utterance_id)
            session.final_text = transcript.text
            session.final_metadata = dict(transcript.metadata or {})
            if self._current_utterance_id == transcript.utterance_id:
                self.store.set_final_candidate(FinalCandidateRequest(text=transcript.text))
                self.store.state.pending_commit = bool(transcript.text)
                self._refresh_conflict_marker()
            self.recorder.record(
                TraceKind.COMMIT,
                "final_staged_for_commit",
                {
                    "utterance_id": transcript.utterance_id,
                    "text_length": len(transcript.text),
                    "decode_ms": transcript.decode_ms,
                    "eot_latency_ms": transcript.end_of_turn_latency_ms,
                },
            )

    def stage_and_maybe_commit(self, transcript: FinalTranscript, auto_commit: bool = True) -> CommitDecision:
        with self.store.lock:
            self.stage_final(transcript)
            if not auto_commit:
                return CommitDecision(utterance_id=transcript.utterance_id, committed=False)
            return self.commit_utterance(transcript.utterance_id)

    def commit_text_for_active(
        self,
        *,
        utterance_id: str,
        text: str,
        auto_commit: bool = True,
        commit_mode_override: CommitMode | None = None,
        selection_override: ClientSelection | None = None,
        metadata: dict | None = None,
    ) -> CommitDecision:
        with self.store.lock:
            session = self._ensure_session(utterance_id)
            session.final_text = text
            if self._current_utterance_id == utterance_id:
                self.store.set_final_candidate(FinalCandidateRequest(text=text))
                self.store.state.pending_commit = bool(text)
                self._refresh_conflict_marker()
            self.recorder.record(
                TraceKind.COMMIT,
                "writer_text_staged_for_commit",
                {
                    "utterance_id": utterance_id,
                    "text_length": len(text),
                    "commit_mode_override": commit_mode_override.value if commit_mode_override else None,
                    "selection_override": selection_override.to_dict() if selection_override else None,
                    "metadata": metadata or {},
                },
            )
            if not auto_commit:
                return CommitDecision(utterance_id=utterance_id, committed=False)
            return self.commit_utterance(utterance_id, commit_mode_override=commit_mode_override, selection_override=selection_override)

    def complete_active_without_commit(self, *, reason: str, metadata: dict | None = None, utterance_id: str | None = None) -> None:
        with self.store.lock:
            target_id = utterance_id or self._current_utterance_id
            if target_id is None:
                return
            session = self._sessions.pop(target_id, None)
            if session is None:
                return
            if self._current_utterance_id == target_id:
                self.store.clear_partial()
                self.store.clear_final_candidate()
                self._clear_display_state()
                self._select_fallback_session()
            self.recorder.record(TraceKind.COMMIT, "utterance_completed_without_commit", {"utterance_id": target_id, "reason": reason, "metadata": metadata or {}})

    def commit_active(self) -> CommitDecision:
        with self.store.lock:
            if self._current_utterance_id is None:
                raise RuntimeError("No active utterance to commit")
            return self._commit_session(self._current_utterance_id)

    def commit_utterance(
        self,
        utterance_id: str,
        *,
        commit_mode_override: CommitMode | None = None,
        selection_override: ClientSelection | None = None,
    ) -> CommitDecision:
        with self.store.lock:
            return self._commit_session(utterance_id, commit_mode_override=commit_mode_override, selection_override=selection_override)

    def _commit_session(self, utterance_id: str, *, commit_mode_override: CommitMode | None = None, selection_override: ClientSelection | None = None) -> CommitDecision:
        session = self._sessions.get(utterance_id)
        if session is None:
            raise RuntimeError(f"No active utterance to commit: {utterance_id}")
        if not session.final_text.strip():
            self.abort_utterance(utterance_id, "empty_final_transcript")
            return CommitDecision(
                utterance_id=utterance_id,
                committed=False,
                conflict_reason="empty_final_transcript",
            )
        # Strip trailing whisper-on-silence tails BEFORE the whole-utterance
        # guards see them. We only strip when each trailing tail segment has
        # its own audio-side corroboration (high no_speech_prob OR low
        # avg_logprob), so a real "Thank you" with confident audio signals is
        # preserved. See strip_trailing_silence_hallucination for the
        # per-segment corroboration logic — this is the discriminator that
        # prevents the function from cutting real user-spoken stock phrases.
        segments = session.final_metadata.get("segments") or ()
        audio_duration_raw = session.final_metadata.get("audio_duration_ms")
        try:
            audio_duration_ms = float(audio_duration_raw) if audio_duration_raw is not None else None
        except (TypeError, ValueError):
            audio_duration_ms = None
        cleaned = strip_trailing_silence_hallucination(
            session.final_text,
            segments=segments,
            audio_duration_ms=audio_duration_ms,
        )
        if cleaned and cleaned != session.final_text:
            self.recorder.record(
                TraceKind.COMMIT,
                "trailing_silence_hallucination_stripped",
                {
                    "utterance_id": utterance_id,
                    "before_text": session.final_text,
                    "after_text": cleaned,
                },
            )
            session.final_text = cleaned
            if self._current_utterance_id == utterance_id:
                self.store.set_final_candidate(FinalCandidateRequest(text=cleaned))
        cleaned = strip_repeated_stock_hallucination_tail(session.final_text)
        if cleaned and cleaned != session.final_text:
            self.recorder.record(
                TraceKind.COMMIT,
                "repeated_stock_hallucination_tail_stripped",
                {
                    "utterance_id": utterance_id,
                    "before_text": session.final_text,
                    "after_text": cleaned,
                },
            )
            session.final_text = cleaned
            if self._current_utterance_id == utterance_id:
                self.store.set_final_candidate(FinalCandidateRequest(text=cleaned))
        # Resolve the ASR's avg_logprob up front: the adjacent-duplicate
        # stripper and the repetition/hallucination guards below all skip when
        # the transcript is confident. Real speech saying "Hello hello hello" —
        # or the band "The The" — has avg_logprob ~ -0.3 to -0.5 on mlx_whisper;
        # hallucinated repetition on silence sits well below -1.0.
        avg_logprob = session.final_metadata.get("avg_logprob")
        try:
            confidence = float(avg_logprob) if avg_logprob is not None else None
        except (TypeError, ValueError):
            confidence = None
        cleaned = strip_adjacent_low_signal_word_duplicates(
            session.final_text, confidence=confidence
        )
        if cleaned and cleaned != session.final_text:
            self.recorder.record(
                TraceKind.COMMIT,
                "adjacent_low_signal_duplicate_words_stripped",
                {
                    "utterance_id": utterance_id,
                    "before_text": session.final_text,
                    "after_text": cleaned,
                },
            )
            session.final_text = cleaned
            if self._current_utterance_id == utterance_id:
                self.store.set_final_candidate(FinalCandidateRequest(text=cleaned))
        # Silence-phrase guard: catches whisper-on-silence one-shots like
        # "Thank you." that pass the structural checks because they're
        # linguistically clean. Requires audio-side corroboration
        # (no_speech_prob / avg_logprob / short audio) to avoid blocking
        # legitimate short utterances.
        no_speech_prob_raw = session.final_metadata.get("no_speech_prob")
        try:
            no_speech_prob = float(no_speech_prob_raw) if no_speech_prob_raw is not None else None
        except (TypeError, ValueError):
            no_speech_prob = None
        if looks_like_silence_hallucination(
            session.final_text,
            no_speech_prob=no_speech_prob,
            avg_logprob=confidence,
            audio_duration_ms=audio_duration_ms,
        ):
            self.abort_utterance(utterance_id, "silence_hallucination")
            return CommitDecision(
                utterance_id=utterance_id,
                committed=False,
                conflict_reason="silence_hallucination",
            )
        if _looks_like_hallucination(session.final_text, confidence=confidence):
            self.abort_utterance(utterance_id, "hallucinated_final_transcript")
            return CommitDecision(
                utterance_id=utterance_id,
                committed=False,
                conflict_reason="hallucinated_final_transcript",
            )
        resolved_mode = commit_mode_override or session.anchor.commit_mode
        resolved_selection = selection_override or session.anchor.selection
        conflict_reason = self._conflict_reason(session, selection_override=resolved_selection)
        if conflict_reason:
            session.status = CommitStatus.CONFLICT
            session.conflict_reason = conflict_reason
            if self._current_utterance_id == utterance_id:
                self.store.state.pending_commit = True
                self.store.state.commit_conflict = conflict_reason
                self.store.state.revision += 1
            self.recorder.record(
                TraceKind.COMMIT,
                "commit_blocked_by_conflict",
                {"utterance_id": utterance_id, "conflict_reason": conflict_reason},
            )
            return CommitDecision(
                utterance_id=utterance_id,
                committed=False,
                conflict_reason=conflict_reason,
                commit_mode=resolved_mode,
                committed_text=session.final_text,
            )
        req = FinalCommitRequest(
            text=session.final_text,
            commit_mode=resolved_mode,
            selection_start=resolved_selection.start,
            selection_end=resolved_selection.end,
            utterance_id=utterance_id,
        )
        if self.inserter is not None:
            try:
                self.inserter.apply_commit(InsertionRequest(
                    utterance_id=utterance_id,
                    text=session.final_text,
                    commit_mode=resolved_mode,
                    selection=resolved_selection,
                    anchor_text=session.anchor.buffer_text,
                ))
            except Exception as exc:
                if self._current_utterance_id == utterance_id:
                    self.store.state.pending_commit = True
                    self.store.state.commit_conflict = 'external_insertion_failed'
                    self.store.state.revision += 1
                self.recorder.record(
                    TraceKind.COMMIT,
                    'commit_blocked_by_external_insertion_failure',
                    {'utterance_id': utterance_id, 'error': str(exc), 'inserter': getattr(self.inserter, 'name', 'unknown')},
                )
                return CommitDecision(
                    utterance_id=utterance_id,
                    committed=False,
                    conflict_reason='external_insertion_failed',
                    commit_mode=resolved_mode,
                    committed_text=session.final_text,
                )
        old_buffer = self.store.state.buffer_text
        committed_start = resolved_selection.start if resolved_mode != CommitMode.APPEND else len(old_buffer)
        committed_end = resolved_selection.end if resolved_mode == CommitMode.REPLACE_SELECTION else committed_start
        self.store.commit_final(req)
        new_buffer = self.store.state.buffer_text
        self.store.state.last_committed_utterance_id = utterance_id
        self.store.state.last_committed_text = session.final_text
        self.store.state.revision += 1
        decision = CommitDecision(
            utterance_id=utterance_id,
            committed=True,
            commit_mode=resolved_mode,
            committed_text=session.final_text,
        )
        self.recorder.record(TraceKind.COMMIT, "commit_completed", decision.to_dict())
        session.status = CommitStatus.COMMITTED
        self._sessions.pop(utterance_id, None)
        self._rebase_pending_sessions_after_commit(
            committed_utterance_id=utterance_id,
            old_buffer=old_buffer,
            new_buffer=new_buffer,
            committed_start=committed_start,
            committed_end=committed_end,
        )
        if self._current_utterance_id == utterance_id:
            self._clear_display_state()
            self._select_fallback_session()
        elif self.active is not None:
            self._refresh_conflict_marker()
        return decision

    def abort_active(self, reason: str) -> None:
        with self.store.lock:
            if self._current_utterance_id is None:
                return
            self.abort_utterance(self._current_utterance_id, reason)

    def abort_utterance(self, utterance_id: str, reason: str) -> None:
        with self.store.lock:
            session = self._sessions.pop(utterance_id, None)
            if session is None:
                return
            session.status = CommitStatus.ABORTED
            session.conflict_reason = reason
            if self._current_utterance_id == utterance_id:
                self.store.clear_partial()
                self.store.clear_final_candidate()
                self.store.state.pending_commit = False
                self.store.state.commit_conflict = reason
                self._clear_display_state()
                self._select_fallback_session()
            self.recorder.record(TraceKind.COMMIT, "utterance_aborted", {"utterance_id": utterance_id, "reason": reason})

    def session_selection(self, utterance_id: str) -> ClientSelection | None:
        with self.store.lock:
            session = self._sessions.get(utterance_id)
            if session is None:
                return None
            return session.anchor.selection

    def _ensure_session(self, utterance_id: str) -> ActiveCommitSession:
        session = self._sessions.get(utterance_id)
        if session is None:
            self.begin_utterance(utterance_id)
            session = self._sessions[utterance_id]
        return session

    def _resolve_mode(self, selection: ClientSelection) -> CommitMode:
        if selection.start != selection.end:
            return CommitMode.REPLACE_SELECTION
        return CommitMode.INSERT_AT_CARET

    def _conflict_reason(self, session: ActiveCommitSession, *, selection_override: ClientSelection | None = None) -> str | None:
        current = self.store.state.buffer_text
        anchor = session.anchor
        original = anchor.buffer_text
        if current == original:
            return None
        sel = (selection_override or anchor.selection).normalized(len(original))
        if current[: sel.start] != original[: sel.start]:
            return "pre_anchor_buffer_mutation"
        if original[sel.start : sel.end] != current[sel.start : min(sel.end, len(current))]:
            return "anchored_region_mutation"
        return None

    def _refresh_conflict_marker(self) -> None:
        session = self.active
        if session is None:
            self.store.state.commit_conflict = None
            return
        conflict_reason = self._conflict_reason(session)
        self.store.state.commit_conflict = conflict_reason
        if conflict_reason:
            session.status = CommitStatus.CONFLICT
            session.conflict_reason = conflict_reason

    def _publish_session_state(self, utterance_id: str, *, clear_preview: bool = False) -> None:
        session = self._sessions[utterance_id]
        self.store.state.active_utterance_id = utterance_id
        self.store.state.anchor_revision = session.anchor.revision
        self.store.state.active_commit_mode = session.anchor.commit_mode.value
        self.store.state.pending_commit = bool(session.final_text)
        self.store.state.commit_conflict = session.conflict_reason
        if clear_preview:
            self.store.state.partial_text = session.latest_partial_text if session.latest_partial_text else ""
            self.store.state.final_candidate_text = session.final_text
        self.store.state.revision += 1

    def _clear_display_state(self) -> None:
        self.store.state.active_utterance_id = None
        self.store.state.anchor_revision = None
        self.store.state.active_commit_mode = None
        self.store.state.pending_commit = False
        self.store.state.commit_conflict = None

    def _select_fallback_session(self) -> None:
        if not self._sessions:
            self._current_utterance_id = None
            return
        self._current_utterance_id = next(reversed(self._sessions))
        session = self._sessions[self._current_utterance_id]
        self.store.state.active_utterance_id = session.utterance_id
        self.store.state.anchor_revision = session.anchor.revision
        self.store.state.active_commit_mode = session.anchor.commit_mode.value
        self.store.state.pending_commit = bool(session.final_text)
        self.store.state.commit_conflict = session.conflict_reason
        self.store.state.partial_text = session.latest_partial_text
        self.store.state.final_candidate_text = session.final_text
        self.store.state.revision += 1

    def _rebase_pending_sessions_after_commit(
        self,
        *,
        committed_utterance_id: str,
        old_buffer: str,
        new_buffer: str,
        committed_start: int,
        committed_end: int,
    ) -> None:
        delta = len(new_buffer) - len(old_buffer)
        if not self._sessions:
            return
        for utterance_id, session in list(self._sessions.items()):
            if utterance_id == committed_utterance_id:
                continue
            anchor = session.anchor
            if anchor.buffer_text != old_buffer:
                continue
            sel = anchor.selection.normalized(len(old_buffer))
            if committed_end <= sel.start:
                # safe upstream insertion/replacement before a later utterance anchor.
                session.anchor = CommitAnchor(
                    buffer_text=new_buffer,
                    selection=ClientSelection(start=max(0, sel.start + delta), end=max(0, sel.end + delta)),
                    revision=self.store.state.revision,
                    commit_mode=anchor.commit_mode,
                )
                self.recorder.record(
                    TraceKind.COMMIT,
                    'pending_session_rebased_after_commit',
                    {
                        'utterance_id': utterance_id,
                        'committed_utterance_id': committed_utterance_id,
                        'delta_chars': delta,
                        'selection_start': session.anchor.selection.start,
                        'selection_end': session.anchor.selection.end,
                    },
                )
            elif committed_start < sel.end:
                session.status = CommitStatus.CONFLICT
                session.conflict_reason = 'upstream_commit_overlap'
                self.recorder.record(
                    TraceKind.COMMIT,
                    'pending_session_marked_overlap_conflict',
                    {
                        'utterance_id': utterance_id,
                        'committed_utterance_id': committed_utterance_id,
                    },
                )
                if utterance_id == self._current_utterance_id:
                    self.store.state.commit_conflict = session.conflict_reason
        self._refresh_conflict_marker()
