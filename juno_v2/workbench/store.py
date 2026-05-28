from __future__ import annotations

import threading
from dataclasses import replace
from typing import Dict

from juno_v2.contracts.tracing import TraceKind
from juno_v2.contracts.workbench import (
    ClientSelection,
    CommitMode,
    FinalCandidateRequest,
    FinalCommitRequest,
    PartialCommitRequest,
    SyncClientStateRequest,
    WorkbenchState,
)
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.workbench.mode_selection_store import WriterModeSelectionStore


class WorkbenchStore:
    def __init__(
        self,
        recorder: TraceRecorder,
        mode_selection_store: WriterModeSelectionStore | None = None,
    ) -> None:
        self.recorder = recorder
        self.state = WorkbenchState()
        self.lock = threading.RLock()
        # Issue #9: persist manual / custom writer-mode selection so a
        # broker restart doesn't silently drop the user's choice. The
        # store is optional — passing ``None`` preserves the legacy
        # in-memory-only behavior used by tests that don't care about
        # disk persistence.
        self.mode_selection_store = mode_selection_store
        if mode_selection_store is not None:
            manual, custom = mode_selection_store.load()
            self.state.manual_writer_mode = manual
            self.state.custom_writer_mode = custom
        self.recorder.record(TraceKind.SYSTEM, "workbench_initialized", {"revision": self.state.revision})

    def _persist_mode_selection(self) -> None:
        store = self.mode_selection_store
        if store is None:
            return
        try:
            store.save(
                manual=self.state.manual_writer_mode,
                custom=self.state.custom_writer_mode,
            )
        except OSError as exc:
            # Best-effort: log via recorder, but never raise out of a
            # locked mutation — callers expect snapshot() to return.
            try:
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "writer_mode_selection_persist_failed",
                    {"error": str(exc), "error_type": type(exc).__name__},
                )
            except OSError:
                pass

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            state = replace(self.state)
            state.recent_events = self.recorder.recent_events()
            return state.to_dict()


    def context_window(self, *, max_field_chars: int) -> dict[str, object]:
        with self.lock:
            state = replace(self.state)
            selection = state.selection.normalized(len(state.buffer_text))
            before = state.buffer_text[max(0, selection.start - max_field_chars): selection.start]
            after = state.buffer_text[selection.end: selection.end + max_field_chars]
            selected_text = state.buffer_text[selection.start: selection.end]
            return {
                'app_name': state.app_name,
                'app_bundle_id': state.app_bundle_id,
                'window_title': state.window_title,
                'clipboard_text': state.clipboard_text,
                'focused_text_before': before,
                'focused_text_after': after,
                'selected_text': selected_text,
                'last_committed_text': state.last_committed_text,
                'last_committed_start': state.last_committed_start,
                'last_committed_end': state.last_committed_end,
                'last_committed_utterance_id': state.last_committed_utterance_id,
            }

    def sync_client_state(self, req: SyncClientStateRequest) -> Dict[str, object]:
        with self.lock:
            selection = req.selection().normalized(len(req.buffer_text))
            self.state.buffer_text = req.buffer_text
            self.state.selection = selection
            self.state.app_name = req.app_name
            self.state.app_bundle_id = (req.app_bundle_id or None)
            self.state.window_title = req.window_title
            self.state.clipboard_text = req.clipboard_text
            self.state.revision += 1
            self.recorder.record(
                TraceKind.UI,
                "sync_client_state",
                {
                    "buffer_length": len(req.buffer_text),
                    "selection_start": selection.start,
                    "selection_end": selection.end,
                    "revision": self.state.revision,
                },
            )
            return self.snapshot()

    def apply_partial(self, req: PartialCommitRequest) -> Dict[str, object]:
        with self.lock:
            self.state.partial_text = req.text
            self.state.revision += 1
            self.recorder.record(
                TraceKind.WORKBENCH,
                "partial_updated",
                {"partial_length": len(req.text), "revision": self.state.revision},
            )
            return self.snapshot()

    def clear_partial(self) -> Dict[str, object]:
        with self.lock:
            self.state.partial_text = ""
            self.state.revision += 1
            self.recorder.record(TraceKind.WORKBENCH, "partial_cleared", {"revision": self.state.revision})
            return self.snapshot()

    def set_final_candidate(self, req: FinalCandidateRequest) -> Dict[str, object]:
        with self.lock:
            self.state.final_candidate_text = req.text
            self.state.revision += 1
            self.recorder.record(
                TraceKind.WORKBENCH,
                "final_candidate_updated",
                {"candidate_length": len(req.text), "revision": self.state.revision},
            )
            return self.snapshot()

    def clear_final_candidate(self) -> Dict[str, object]:
        with self.lock:
            self.state.final_candidate_text = ""
            self.state.revision += 1
            self.recorder.record(TraceKind.WORKBENCH, "final_candidate_cleared", {"revision": self.state.revision})
            return self.snapshot()

    def set_writer_mode(self, mode: str) -> Dict[str, object]:
        with self.lock:
            self.state.writer_mode = mode
            self.state.revision += 1
            self.recorder.record(TraceKind.WRITER, "writer_mode_updated", {"mode": mode, "revision": self.state.revision})
            return self.snapshot()

    def set_manual_writer_mode(self, mode: str | None) -> Dict[str, object]:
        """Pin effective writer mode to a built-in id until cleared."""
        with self.lock:
            self.state.manual_writer_mode = (mode or "").strip() or None
            self.state.revision += 1
            self._persist_mode_selection()
            self.recorder.record(
                TraceKind.WRITER,
                "manual_writer_mode_set",
                {"manual_writer_mode": self.state.manual_writer_mode, "revision": self.state.revision},
            )
            return self.snapshot()

    def clear_manual_writer_mode(self) -> Dict[str, object]:
        with self.lock:
            self.state.manual_writer_mode = None
            self.state.revision += 1
            self._persist_mode_selection()
            self.recorder.record(TraceKind.WRITER, "manual_writer_mode_cleared", {"revision": self.state.revision})
            return self.snapshot()

    def set_custom_writer_mode(self, name: str | None) -> Dict[str, object]:
        with self.lock:
            self.state.custom_writer_mode = (name or "").strip() or None
            self.state.revision += 1
            self._persist_mode_selection()
            self.recorder.record(
                TraceKind.WRITER,
                "custom_writer_mode_set",
                {"custom_writer_mode": self.state.custom_writer_mode, "revision": self.state.revision},
            )
            return self.snapshot()

    def clear_custom_writer_mode(self) -> Dict[str, object]:
        with self.lock:
            self.state.custom_writer_mode = None
            self.state.revision += 1
            self._persist_mode_selection()
            self.recorder.record(TraceKind.WRITER, "custom_writer_mode_cleared", {"revision": self.state.revision})
            return self.snapshot()

    def set_language_state(self, *, requested_language: str | None, observed_language: str | None, language_policy: str | None) -> Dict[str, object]:
        with self.lock:
            self.state.requested_language = requested_language
            self.state.observed_language = observed_language
            self.state.language_policy = language_policy
            self.state.revision += 1
            self.recorder.record(TraceKind.SYSTEM, "language_state_updated", {
                "requested_language": requested_language,
                "observed_language": observed_language,
                "language_policy": language_policy,
                "revision": self.state.revision,
            })
            return self.snapshot()

    def set_last_writer_action(self, action: str | None, *, payload: dict | None = None) -> Dict[str, object]:
        with self.lock:
            self.state.last_writer_action = action
            self.state.revision += 1
            self.recorder.record(TraceKind.WRITER, "writer_action_updated", {"action": action, "revision": self.state.revision, "payload": payload or {}})
            return self.snapshot()

    @staticmethod
    def _needs_separator(text: str, pos: int) -> bool:
        """Return True if a space is needed before inserting at *pos*."""
        if not text or pos <= 0 or pos > len(text):
            return False
        prev = text[pos - 1]
        return prev not in (' ', '\n', '\t', '\r')

    def commit_final(self, req: FinalCommitRequest) -> Dict[str, object]:
        with self.lock:
            original = self.state.buffer_text
            selection = req.selection_override() or self.state.selection
            selection = selection.normalized(len(original))
            incoming = req.text
            if incoming and incoming[0].isalnum():
                if req.commit_mode == CommitMode.APPEND and self._needs_separator(original, len(original)):
                    incoming = ' ' + incoming
                elif req.commit_mode == CommitMode.INSERT_AT_CARET and self._needs_separator(original, selection.end):
                    incoming = ' ' + incoming
            if req.commit_mode == CommitMode.REPLACE_ALL:
                updated = incoming
                committed_start = 0
                caret = len(updated)
            elif req.commit_mode == CommitMode.APPEND:
                committed_start = len(original)
                updated = original + incoming
                caret = len(updated)
            elif req.commit_mode == CommitMode.REPLACE_SELECTION:
                committed_start = selection.start
                updated = original[: selection.start] + incoming + original[selection.end :]
                caret = selection.start + len(incoming)
            else:
                committed_start = selection.end
                updated = original[: selection.end] + incoming + original[selection.end :]
                caret = selection.end + len(incoming)

            self.state.buffer_text = updated
            self.state.partial_text = ""
            self.state.final_candidate_text = ""
            self.state.selection = ClientSelection(start=caret, end=caret)
            self.state.last_committed_start = committed_start
            self.state.last_committed_end = committed_start + len(incoming)
            self.state.revision += 1
            self.recorder.record(
                TraceKind.WORKBENCH,
                "final_committed",
                {
                    "commit_mode": req.commit_mode.value,
                    "inserted_length": len(incoming),
                    "buffer_length": len(updated),
                    "caret": caret,
                    "revision": self.state.revision,
                    "utterance_id": req.utterance_id,
                    "selection_start": selection.start,
                    "selection_end": selection.end,
                    "committed_start": committed_start,
                    "committed_end": committed_start + len(incoming),
                },
            )
            return self.snapshot()

    def reset(self, keep_buffer: bool = False) -> Dict[str, object]:
        with self.lock:
            buffer_text = self.state.buffer_text if keep_buffer else ""
            writer_mode = self.state.writer_mode
            manual_wm = self.state.manual_writer_mode
            custom_wm = self.state.custom_writer_mode
            self.state = WorkbenchState(
                buffer_text=buffer_text,
                writer_mode=writer_mode,
                manual_writer_mode=manual_wm,
                custom_writer_mode=custom_wm,
                runtime_mode=self.state.runtime_mode,
            )
            if buffer_text:
                self.state.selection = ClientSelection(start=len(buffer_text), end=len(buffer_text))
            self.state.revision += 1
            self.recorder.record(
                TraceKind.SYSTEM,
                "workbench_reset",
                {"keep_buffer": keep_buffer, "buffer_length": len(self.state.buffer_text), "revision": self.state.revision},
            )
            return self.snapshot()
