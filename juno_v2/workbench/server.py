from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import replace
import datetime as dt
import json
import logging
import os
import re
import threading
import time
import uuid
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from juno_v2.context.provider import ContextProvider
    from juno_v2.engine.session import DictationSessionRunner
    from juno_v2.language.policy import LanguagePlanner
    from juno_v2.memory.bias import RecognitionBiasEngine
    from juno_v2.memory.store import JsonMemoryStore
    from juno_v2.observability.actions_index import ActionsIndex
    from juno_v2.personalization.seed.runtime import JunoSeedPersonalizationRuntime
    from juno_v2.runtime.swappable_final import SwappableFinalBackend
    from juno_v2.writer.backends.base import WriterBackend
    from juno_v2.writer.service import WriterService

from juno_v2.commit.controller import CommitController
from juno_v2.contracts.tracing import TraceKind
from juno_v2.contracts.workbench import CommitMode, FinalCandidateRequest, FinalCommitRequest, ResetRequest, SyncClientStateRequest
from juno_v2.observability.tracing import TraceRecorder
from juno_v2.runtime.config import WorkbenchRuntimeConfig
from juno_v2.runtime.deployment import _env_bool
from juno_v2.runtime.ids import new_session_id
from juno_v2.runtime.inference_scheduler import InferenceJobCancelled, InferenceScheduler
from juno_v2.runtime.paths import product_audio_root
from juno_v2.memory.hallucination import looks_like_silence_hallucination
# LiveTranscriptStateManager and the live patch envelope path were removed
# with the LocalAgreement-2 rewrite.
# Final-stage reconciliation patches still flow through
# juno_v2.transcript.final_reconciliation.
from juno_v2.workbench.store import WorkbenchStore

from juno_core_v3.context.capability_probe import CapabilityChecker
from juno_core_v3.context.clipboard_ring import ClipboardRingBuffer
from juno_core_v3.context.plane import ContextPlane
from juno_core_v3.context.suppression_config import SuppressionConfig
from juno_core_v3.contracts.resource_hints import HostResourceHints
from juno_core_v3.dictation import (
    DictationTranscriber,
    OneShotDictationPipeline,
    resolve_transcriber_from_env,
)
from juno_core_v3.broker.runners import InsertRequest, InsertRunner
from juno_core_v3.contracts.session import SessionKind, UserIntentSignals


def _preview_candidates_from_session_context_tape(tape: Any, *, limit: int = 24) -> list[str]:
    """Extract screen terms for HUD repair from the macOS context tape."""
    if tape is None:
        return []
    if isinstance(tape, dict):
        maybe = tape.get("snapshots")
        raw_items = maybe if isinstance(maybe, list) else [tape]
    elif isinstance(tape, list):
        raw_items = tape
    else:
        raw_items = []

    chunks: list[str] = []
    phrase_candidates: list[str] = []
    explicit: list[str] = []
    seen_snapshots: set[tuple[str, str, str]] = set()
    for item in raw_items[:12]:
        if not isinstance(item, dict):
            continue
        app = str(item.get("app_name") or item.get("frontmost_app_name") or "").strip()
        title = str(item.get("window_title") or "").strip()
        selected = str(item.get("selected_text") or "").strip()
        before = str(item.get("focused_text_before") or item.get("focused_text") or "").strip()
        after = str(item.get("focused_text_after") or "").strip()
        field_excerpt = str(item.get("field_text_excerpt") or "").strip()
        doc = str(item.get("focused_document_path") or item.get("focused_file_path") or "").strip()
        key = (
            app.casefold(),
            title.casefold(),
            (selected or before or after or field_excerpt).casefold(),
        )
        if key in seen_snapshots:
            continue
        seen_snapshots.add(key)
        chunks.extend([app, title, selected, before, after, field_excerpt, doc])
        for chunk in (title, selected, before, after, field_excerpt, doc):
            phrase_candidates.extend(
                match.group(0).strip()
                for match in re.finditer(
                    r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){1,3}\b",
                    chunk,
                )
            )
        raw_candidates = item.get("candidate_entities") or item.get("candidate_terms")
        if isinstance(raw_candidates, list):
            explicit.extend(str(raw or "").strip() for raw in raw_candidates[:limit])

    try:
        from juno_v2.context.provider import _extract_candidates
    except ImportError:
        extracted: list[str] = []
    else:
        extracted = _extract_candidates([chunk[:240] for chunk in chunks if chunk])

    out: list[str] = []
    seen: set[str] = set()
    for candidate in [*explicit, *phrase_candidates, *extracted]:
        value = str(candidate or "").strip()
        if not value or len(value) > 80:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= limit:
            break
    return out


_ACTION_PREVIEW_WAKE_RE = re.compile(r"^\s*(?:hey\s+juno|juno)\b", re.IGNORECASE)
_ACTION_PREVIEW_KIND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "note",
        re.compile(
            r"\b(?:take|create|add|write|save)\s+(?:a\s+)?note\b|\bnote\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reminder",
        re.compile(r"\bremind(?:er|ers)?\b|\bremind\s+me\b", re.IGNORECASE),
    ),
    (
        "alarm",
        re.compile(r"\b(?:set\s+)?alarm\b", re.IGNORECASE),
    ),
)


def _action_preview_display_text(committed_text: str, tail_text: str) -> str | None:
    """Return a compact HUD status for wake-phrase action utterances.

    The preview lane is acoustic text, but action utterances are not pasted as
    dictation. Showing every partial action fragment in the committed HUD lane
    makes successful multi-action requests look broken because overlapping
    preview windows repeat note/reminder/alarm text. Keep this display-only:
    final action extraction still runs from the final transcript.
    """
    combined = re.sub(" +", " ", f"{committed_text or ''} {tail_text or ''}").strip()
    if not combined or _ACTION_PREVIEW_WAKE_RE.search(combined) is None:
        return None

    seen: set[str] = set()
    found: list[tuple[int, str]] = []
    for label, pattern in _ACTION_PREVIEW_KIND_PATTERNS:
        match = pattern.search(combined)
        if match is None or label in seen:
            continue
        seen.add(label)
        found.append((match.start(), label))
    if not found:
        return "Hey Juno"
    labels = [label for _, label in sorted(found, key=lambda item: item[0])]
    return f"Hey Juno: {', '.join(labels)}"


def _merge_action_preview_display_text(previous: str | None, current: str) -> str:
    """Keep wake-action HUD status monotonic across overlapping preview chunks."""
    if not previous:
        return current
    if current.startswith(previous):
        return current
    if previous.startswith(current):
        return previous
    prev_labels = _action_preview_display_labels(previous)
    cur_labels = _action_preview_display_labels(current)
    if not prev_labels:
        return current if cur_labels else previous
    if not cur_labels:
        return previous
    merged: list[str] = []
    for label in [*prev_labels, *cur_labels]:
        if label not in merged:
            merged.append(label)
    return f"Hey Juno: {', '.join(merged)}"


def _action_preview_display_labels(text: str) -> list[str]:
    match = re.match(r"^\s*Hey\s+Juno(?:\s*:\s*(?P<labels>.+))?\s*$", text or "", flags=re.IGNORECASE)
    if match is None:
        return []
    labels = match.group("labels")
    if not labels:
        return []
    out: list[str] = []
    for raw in labels.split(","):
        label = raw.strip().lower()
        if label in {"note", "reminder", "alarm"} and label not in out:
            out.append(label)
    return out
from juno_core_v3.policy.surface_gate import SurfaceId
from juno_core_v3.actions.timeparse import parse_when
from juno_core_v3.workbench.broker_facade import BrokerFacade
from juno_v2.audio.diagnostics import AudioDiagnostics, analyze_audio_signal
from juno_v2.contracts.modes import CustomModeRecord
from juno_v2.contracts.transforms import CustomTransformRecord
from juno_v2.modes.defaults import BUILTIN_MODES, builtin_mode_names
from juno_v2.modes.store import CustomModeStore, default_modes_data_path
from juno_v2.presets.surface_presets import (
    SurfacePresetRecord,
    SurfacePresetStore,
    default_surface_presets_path,
    resolve_mode_with_surface_presets,
)
from juno_v2.transforms.catalog import BUILTIN_CATALOG
from juno_v2.transforms.store import CustomTransformStore, default_transforms_data_path
from juno_core_v3.broker.tools import ToolRegistry, register_builtin_tools

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)


_PREVIEW_BARE_SILENCE_PHRASE_RE = re.compile(
    r"^\s*((?:thank\s+you|thanks(\s+(for\s+(watching|listening)|so\s+much))?|please\s+subscribe|subscribe|bye|goodbye|yeah|yep|ps)(?:\s+(?:ps|yeah|yep))*)[\s\.\!\?,;:'\"\-–—…]*$",
    re.IGNORECASE,
)

_PREVIEW_BARE_FILLER_PHRASE_RE = re.compile(
    r"^\s*(?:yeah|yep|yes|no|okay|ok|whoa|wow|mm+|mhm|mm\s*hmm|uh\s*huh|uhh?|um+|hmm+)[\s\.\!\?,;:'\"\-–—…]*$",
    re.IGNORECASE,
)

def _preview_bare_silence_phrase(text: str) -> bool:
    return bool(_PREVIEW_BARE_SILENCE_PHRASE_RE.match(text or ""))


def _preview_bare_filler_phrase(text: str) -> bool:
    return bool(_PREVIEW_BARE_FILLER_PHRASE_RE.match(text or ""))


def _preview_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text or "")


def _preview_has_substantive_words(text: str) -> bool:
    return bool(_preview_words(text))


def _preview_compatibility_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    contraction_expansions = {
        "i'd": ("i", "would"),
        "you'd": ("you", "would"),
        "we'd": ("we", "would"),
        "they'd": ("they", "would"),
        "he'd": ("he", "would"),
        "she'd": ("she", "would"),
        "it'd": ("it", "would"),
        "i'm": ("i", "am"),
        "you're": ("you", "are"),
        "we're": ("we", "are"),
        "they're": ("they", "are"),
        "i'll": ("i", "will"),
        "you'll": ("you", "will"),
        "we'll": ("we", "will"),
        "they'll": ("they", "will"),
        "i've": ("i", "have"),
        "you've": ("you", "have"),
        "we've": ("we", "have"),
        "they've": ("they", "have"),
    }
    for word in _preview_words(text):
        token = word.casefold()
        expanded = contraction_expansions.get(token)
        if expanded:
            tokens.extend(expanded)
        else:
            tokens.append(token)
    return tokens


def _preview_repetitive_noise(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", (text or "").casefold())
    if not lowered.strip():
        return False
    if lowered.count("little bit") >= 1 and (
        "little bit of a little bit" in lowered
        or "a little bit more than a little bit" in lowered
        or lowered.count("little bit") >= 2
    ):
        return True
    words = _preview_words(lowered)
    if len(words) < 10:
        return False
    if len(set(words)) / max(1, len(words)) < 0.34:
        return True
    for n in (2, 3, 4):
        grams = [tuple(words[i : i + n]) for i in range(0, len(words) - n + 1)]
        if grams and max(grams.count(g) for g in set(grams)) >= 3:
            return True
    return False


def _normalized_preview_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()


def _preview_hypotheses_compatible(previous: str, incoming: str) -> bool:
    prev_tokens = _preview_compatibility_tokens(previous)
    inc_tokens = _preview_compatibility_tokens(incoming)
    if not prev_tokens or not inc_tokens:
        return False
    if len(prev_tokens) <= len(inc_tokens):
        return inc_tokens[: len(prev_tokens)] == prev_tokens
    return prev_tokens[: len(inc_tokens)] == inc_tokens


def _preview_hypothesis_revision_compatible(previous: str, incoming: str) -> bool:
    prev_tokens = _preview_compatibility_tokens(previous)
    inc_tokens = _preview_compatibility_tokens(incoming)
    if len(prev_tokens) < 4 or len(inc_tokens) <= len(prev_tokens):
        return False

    common_tokens = 0
    for prev_token, inc_token in zip(prev_tokens, inc_tokens):
        if prev_token != inc_token:
            break
        common_tokens += 1

    if common_tokens < 4:
        return False

    # Live preview can repair one or two trailing words while adding new
    # words. Keep that fluid, but do not allow a visible prefix to be
    # replaced by a different sentence.
    revised_prev_tail = len(prev_tokens) - common_tokens
    incoming_tail = len(inc_tokens) - common_tokens
    if revised_prev_tail <= 2 and incoming_tail >= revised_prev_tail + 2:
        return True
    if (
        common_tokens >= max(6, int(len(prev_tokens) * 0.55))
        and incoming_tail >= revised_prev_tail + 1
    ):
        return True

    prev_norm = _normalized_preview_text(previous)
    inc_norm = _normalized_preview_text(incoming)
    common_chars = _common_prefix_chars(prev_norm, inc_norm)
    return common_chars >= max(28, int(len(prev_norm) * 0.68)) and len(inc_norm) - len(prev_norm) >= 12


def _preview_hypothesis_overlaps_visible_hint(visible_hint: str, incoming: str) -> bool:
    if _preview_hypotheses_compatible(visible_hint, incoming):
        return True
    if _preview_hypothesis_revision_compatible(visible_hint, incoming):
        return True
    visible_tokens = _preview_compatibility_tokens(visible_hint)
    inc_tokens = _preview_compatibility_tokens(incoming)
    if len(visible_tokens) < 4 or len(inc_tokens) < 2:
        return False
    max_overlap = min(len(visible_tokens), len(inc_tokens), 8)
    for size in range(max_overlap, 1, -1):
        if visible_tokens[-size:] == inc_tokens[:size]:
            return True
    return False


def _preview_cumulative_audio_ms(metadata: dict[str, Any] | None) -> float | None:
    raw_meta = dict(metadata or {})
    raw_payload = raw_meta.get("raw") if isinstance(raw_meta.get("raw"), dict) else {}
    for value in (
        raw_meta.get("cumulative_audio_ms"),
        raw_payload.get("cumulative_audio_ms"),
        raw_meta.get("request_audio_ms"),
        raw_payload.get("request_audio_ms"),
    ):
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _common_prefix_chars(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def _select_divergent_preview_text(previous: str, incoming: str) -> tuple[str, str]:
    """Choose between competing cumulative preview hypotheses.

    The preview service returns whole-utterance hypotheses. A divergent
    hypothesis is not a new chunk to append; treating it that way causes the
    HUD repetition failure seen in live testing. We only allow a divergent
    replacement when it is still early in the utterance, or when the incoming
    text is a clear longer version of the same prefix with minor boundary
    repair. Otherwise the already-visible HUD text wins until final ASR.
    """
    prev = re.sub(r"\s+", " ", (previous or "").strip())
    inc = re.sub(r"\s+", " ", (incoming or "").strip())
    if not prev or not inc:
        return inc or prev, "preview_empty_divergent"
    if _preview_repetitive_noise(inc):
        return prev, "preview_repetitive_regression"

    prev_words = _preview_words(prev)
    inc_words = _preview_words(inc)
    if len(prev_words) <= 6 and len(inc_words) <= 14:
        return inc, "preview_early_hypothesis_replace"

    prev_norm = _normalized_preview_text(prev)
    inc_norm = _normalized_preview_text(inc)
    common = _common_prefix_chars(prev_norm, inc_norm)
    longer_by = len(inc_norm) - len(prev_norm)
    if common >= max(28, int(len(prev_norm) * 0.68)) and longer_by >= 18:
        return inc, "preview_compatible_tail_rewrite"

    return prev, "preview_divergent_regression"


def _merge_final_preview_tail(previous: str, incoming: str) -> tuple[str, str | None]:
    """Merge a final preview flush that only contains the rolling-window tail.

    MLX Whisper preview runs on a bounded audio window. On stop, that window can
    decode just the last phrase while the HUD already contains the earlier text.
    Treating that as a short replacement freezes the HUD exactly when the final
    preview should catch up.
    """

    prev = re.sub(r"\s+", " ", (previous or "").strip())
    inc = re.sub(r"\s+", " ", (incoming or "").strip())
    if not prev or not inc:
        return inc or prev, None
    if inc == prev or prev.startswith(inc) or inc in prev:
        return prev, "preview_final_short_tail_held"
    if inc.startswith(prev):
        return inc, None

    prev_spans = _preview_word_spans(prev)
    inc_spans = _preview_word_spans(inc)
    prev_tokens = [span[0] for span in prev_spans]
    inc_tokens = [span[0] for span in inc_spans]
    max_overlap = min(16, len(prev_tokens), len(inc_tokens))
    for count in range(max_overlap, 1, -1):
        if prev_tokens[-count:] != inc_tokens[:count]:
            continue
        suffix_start = inc_spans[count - 1][2]
        suffix = inc[suffix_start:].lstrip(" ,.;:!?-")
        if not suffix:
            return prev, "preview_final_short_tail_held"
        return _join_preview_tail(prev, suffix), "preview_final_tail_merged"

    if _looks_like_final_tail_continuation(prev, inc):
        return _join_preview_tail(prev, inc), "preview_final_tail_appended"

    if len(inc_tokens) < len(prev_tokens):
        return prev, "preview_final_short_tail_held"
    return inc, None


def _preview_word_spans(value: str) -> list[tuple[str, int, int]]:
    return [
        ("".join(ch.casefold() for ch in match.group(0) if ch.isalnum()), match.start(), match.end())
        for match in re.finditer(r"\S+", value or "")
        if "".join(ch.casefold() for ch in match.group(0) if ch.isalnum())
    ]


def _looks_like_final_tail_continuation(previous: str, incoming: str) -> bool:
    prev_words = _preview_words(previous)
    inc_words = _preview_words(incoming)
    if len(prev_words) < 5 or len(inc_words) < 4:
        return False
    if (
        _preview_bare_silence_phrase(incoming)
        or _preview_bare_filler_phrase(incoming)
        or _preview_repetitive_noise(incoming)
    ):
        return False
    if inc_words[0].casefold() in {
        "and",
        "or",
        "but",
        "so",
        "then",
        "also",
        "because",
        "it",
        "that",
        "this",
    }:
        return True
    # On the MLX Whisper preview lane the final flush can decode only the
    # bounded rolling-window tail. That tail can start on any normal content
    # word or verb after a pause ("Add the owner...", "Schedule the follow
    # up..."), not only on conjunctions. If the final candidate is a shorter,
    # substantive phrase and not a known low-signal hallucination, prefer
    # forward HUD motion over freezing the visible transcript until final paste.
    return len(incoming) >= 20 and len(inc_words) < len(prev_words)


def _join_preview_tail(previous: str, suffix: str) -> str:
    left = re.sub(r"\s+", " ", (previous or "").strip())
    right = re.sub(r"\s+", " ", (suffix or "").strip())
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


_V3_RECURRENCE_PROMPT_ADDENDUM = """

When the user speaks a recurring intent ("every day", "for the next 10 days",
"every weekday at 9", "on the first of every month"), DO NOT collapse it to a
single instant. Emit one action whose schedule.kind is "series" and whose
schedule.series carries the ICS-shaped rule:

  {
    "freq": "DAILY" | "WEEKLY" | "MONTHLY" | "YEARLY",
    "interval": 1,
    "by_day": ["MO","TU","WE","TH","FR"]?,    // weekly only
    "by_month_day": [1,15]?,                   // monthly only
    "count": 10?,                              // when bounded
    "until_iso": "..."?,                       // when bounded by date
    "first_occurrence_iso": "..."              // anchors time-of-day; required
  }

Examples:

  "remind me for the next 10 days to do hiring for hardware" →
    schedule.kind = "series",
    schedule.series.freq = "DAILY", interval = 1, count = 10,
    schedule.series.first_occurrence_iso anchors tomorrow at the user's
    morning default if no time was spoken.

  "every weekday at 9 stand-up" →
    schedule.kind = "series",
    schedule.series.freq = "WEEKLY", interval = 1,
    schedule.series.by_day = ["MO","TU","WE","TH","FR"],
    schedule.series.first_occurrence_iso anchors next weekday at 09:00.

  "on the first of every month pay rent" →
    schedule.kind = "series",
    schedule.series.freq = "MONTHLY", interval = 1,
    schedule.series.by_month_day = [1].

For a single-shot reminder, emit schedule.kind = "instant" with
schedule.instant.iso. Notes never carry schedule.

"""

_V3_CREATE_ONLY_PROMPT_ADDENDUM = """

Set operation = "create" for every action emitted in this version.
"""

_V3_OPERATIONS_PROMPT_ADDENDUM = """

When the user asks to change, move, complete, snooze, cancel, delete, stop,
undo, list, read, show, or ask whether a reminder/alarm/note exists, emit a
non-create operation instead of pretending it is a new reminder.
Questions like "did I set a reminder about Karim" or "do I have a reminder
about rent" are executable queries, not general questions.

Allowed Phase 2 operations:
  update   = change time/title/body of an existing item
  complete = mark an existing reminder done
  snooze   = delay an existing reminder by an offset
  delete   = cancel/delete/undo an existing item
  query    = list/show/read/find matching existing items

Do not emit append_to or remove_from unless the Container/list Phase 4
addendum is included below. Every non-create action MUST include target:
  by_id          when a Juno id was explicitly referenced
  by_pronoun     only for bare "that", "it", "the last one", "the one I just made"
  by_description for "the gym reminder", "the 4 pm hiring reminder"
  by_query       for "all reminders for tomorrow", "what do I have today"

If a pronoun is followed by descriptive words ("that hiring reminder series",
"that laundry one"), use by_description, not by_pronoun. The spoken word
"reminder" always means kind="reminder", including recurring reminders and
reminder series; do not convert reminder series into alarms.

For target.filter use kind, text_match, list_name, or date_range when spoken.
Examples:
  "move my 4 pm hiring reminder to 6" -> operation update, kind reminder,
    target.ref_kind by_description, target.description "4 pm hiring reminder",
    schedule.kind instant for the new 6 pm time.
  "mark the laundry one done" -> operation complete, kind reminder,
    target.ref_kind by_description, target.description "laundry".
  "snooze that for an hour" -> operation snooze, target.ref_kind by_pronoun,
    target.pronoun "that", snooze_offset_seconds 3600.
  "push the gym reminder by 30 minutes" -> operation update, kind reminder,
    target.ref_kind by_description, target.description "gym reminder",
    relative_offset_seconds 1800. Use snooze only when the user says snooze.
  "what reminders do I have tomorrow" -> operation query, kind reminder,
    target.ref_kind by_query, target.filter.date_range for tomorrow.
  "did I set a reminder about Karim" -> operation query, kind reminder,
    target.ref_kind by_query, target.filter.text_match "Karim".
"""

_V3_VAGUE_TIME_PROMPT_ADDENDUM = """

Vague time Phase 3 is enabled. For vague reminder times, emit
schedule.kind="vague" instead of schedule.kind="instant", and set BOTH
action.needs_confirmation=true and schedule.vague.needs_confirmation=true.
Use schedule.vague.default_iso=null; the resolver fills the concrete default.

Bucket mapping:
  "later" -> bucket later
  "soon" or "in a bit" -> bucket soon
  "tonight" or "this evening" -> bucket evening
  "tomorrow morning" -> bucket morning
  "after lunch" -> bucket afternoon
  "this weekend" -> bucket weekend
  "next week" -> bucket next_week

Examples:
  "remind me later about the slack thread" -> create reminder, body "slack thread",
    schedule.kind vague, schedule.vague.bucket later, needs_confirmation true.
  "in a bit remind me to check the build" -> create reminder, body "check the build",
    schedule.kind vague, schedule.vague.bucket soon, needs_confirmation true.
  "this weekend remind me to call dad" -> create reminder, body "call dad",
    schedule.kind vague, schedule.vague.bucket weekend, needs_confirmation true.
"""

_V3_CONTAINERS_PROMPT_ADDENDUM = """

Container/list Phase 4 is enabled. For Reminders lists, use
container.list_name with the spoken list name. Supported operations:
  append_to = add item(s) to an existing or new Reminders list
  remove_from = remove matching item(s) from a Reminders list
  query = show/read/list a Reminders list
  create = create a named Reminders list, optionally with item text in body

Examples:
  "add eggs and oat milk to my groceries list" -> operation append_to,
    kind reminder, body "eggs and oat milk", container.list_name "Groceries".
  "create a packing list for tokyo with passport headphones and charger" ->
    operation create, kind reminder, body "passport headphones and charger",
    container.list_name "Tokyo".
  "show me my groceries list" -> operation query, kind reminder, body "",
    container.list_name "Groceries".
  "remove milk from groceries" -> operation remove_from, kind reminder,
    body "milk", target.description "milk", container.list_name "Groceries".
  "move pay rent to my work list" -> operation update, kind reminder,
    target.ref_kind by_description, target.description "pay rent",
    container.list_name "Work".
"""

_V3_COMPOUND_PROMPT_ADDENDUM = """

Compound Phase 6 is enabled. Split one utterance into ordered actions when the
user asks for multiple concrete actions in the same sentence. Keep declaration
order. If a later action refers to an earlier action or container, set
links_to to the earlier action's link_id.

Examples:
  "at 5 pm remind me to leave then 5:30 to call mom" -> two reminder actions.
  "set an alarm for 6 am and a reminder at 6:15 to take meds" -> alarm then
    reminder.
  "create a packing list with passport and headphones and remind me tomorrow
    at 8 am to check it" -> list create with link_id "create-list-1", then
    reminder with links_to "create-list-1".
"""


def build_writer_action_extractor(writer_backend: Any):
    """Adapt the live writer backend to the action LLM fallback contract."""

    from juno_core_v3.actions.llm_extractor import expected_envelope_schema
    from juno_v2.contracts.writer import WriterMode, WriterTransformRequest

    use_v3 = _env_bool("JUNO_ACTIONS_SCHEMA_V3", False)
    use_operations = _env_bool("JUNO_ACTIONS_OPERATIONS", False)
    instruction_cache: dict[str, str] = {}

    def _instruction_for(version: str) -> str:
        cached = instruction_cache.get(version)
        if cached is not None:
            return cached
        schema = expected_envelope_schema(version=version)
        instruction = (
            "Extract Juno voice actions from the transcript. The wake phrase has already been removed.\n\n"
            "First classify the speech act. Use execute_action whenever the user is directing Juno to "
            "perform a concrete operation on a note, reminder, alarm, or list — including creating, "
            "scheduling, or waking them with a new item, AND including updating, completing, snoozing, "
            "deleting, or querying an existing item when those operations are enabled. The execute_action "
            "intent is the umbrella for every Juno-directed operation; the per-action 'operation' field "
            "distinguishes create from update/complete/snooze/delete/query/append_to/remove_from.\n\n"
            "Do not execute when the utterance is ordinary dictation, praise, a product/capability description, "
            "a question (other than executable queries about existing reminders), an example, or quoted text, "
            "even if it contains words like Juno, note, remind, reminder, alarm, timer, or wake me.\n\n"
            "Resolve false starts and corrections using the last correction the user settled on. Split multiple "
            "executable requests into ordered actions. For each executable action, provide an evidence_span copied "
            "from the spoken transcript. Use context only as reference after spoken intent is proven; never use "
            "context as evidence that an action was requested. If now_iso is present in the prompt, use it as the "
            "current local time for relative dates. Preserve the spoken time phrase in when_text for reminders and "
            "alarms; any schedule.instant.iso must resolve from that when_text and now_iso.\n\n"
            "Return one JSON object matching the schema. No markdown, no prose. Schema: "
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        if version == "actions_intent_v3":
            instruction = instruction + _V3_RECURRENCE_PROMPT_ADDENDUM
            instruction = instruction + (
                _V3_OPERATIONS_PROMPT_ADDENDUM if use_operations else _V3_CREATE_ONLY_PROMPT_ADDENDUM
            )
            if _env_bool("JUNO_ACTIONS_VAGUE_TIME", False):
                instruction = instruction + _V3_VAGUE_TIME_PROMPT_ADDENDUM
            if _env_bool("JUNO_ACTIONS_CONTAINERS", False):
                instruction = instruction + _V3_CONTAINERS_PROMPT_ADDENDUM
            if _env_bool("JUNO_ACTIONS_COMPOUND", False):
                instruction = instruction + _V3_COMPOUND_PROMPT_ADDENDUM
        instruction_cache[version] = instruction
        return instruction

    def _extract(post_wake_text: str, now: Any = None) -> dict[str, Any] | None:
        version = _select_actions_schema_version(post_wake_text, use_v3=use_v3)
        try:
            result = writer_backend.rewrite(
                WriterTransformRequest(
                    utterance_id=f"actions-{uuid.uuid4().hex[:12]}",
                    instruction=_instruction_for(version),
                    source_text=post_wake_text,
                    mode=WriterMode.DEFAULT_SURFACE,
                    context_payload={
                        "task": "voice_action_extraction",
                        "schema_version": version,
                        "wake_gate_verified_from_raw_asr": True,
                        "adjudicated_text": post_wake_text,
                        "now_iso": now.isoformat() if hasattr(now, "isoformat") else None,
                        "allowed_action_kinds": ["note", "reminder", "alarm"],
                    },
                    metadata={"feature": "voice_actions_llm_fallback"},
                )
            )
        except Exception:
            logger.exception("actions_llm_extractor_failed")
            return None
        envelope = _extract_json_object(getattr(result, "text", "") or "")
        if envelope is not None:
            envelope.setdefault("schema_version", version)
            _normalize_action_required_fields(envelope, source_text=post_wake_text)
            if version == "actions_intent_v3":
                _normalize_create_envelope(envelope, source_text=post_wake_text)
                _normalize_recurrence_envelope(envelope, source_text=post_wake_text, now=now)
            if version == "actions_intent_v3" and use_operations:
                _normalize_operations_envelope(envelope, source_text=post_wake_text)
            if version == "actions_intent_v3" and _env_bool("JUNO_ACTIONS_VAGUE_TIME", False):
                _normalize_vague_time_envelope(envelope, source_text=post_wake_text)
            if version == "actions_intent_v3" and _env_bool("JUNO_ACTIONS_CONTAINERS", False):
                _normalize_container_envelope(envelope, source_text=post_wake_text)
            if version == "actions_intent_v3" and _env_bool("JUNO_ACTIONS_COMPOUND", False):
                _normalize_compound_envelope(envelope, source_text=post_wake_text, now=now)
        return envelope

    return _extract


def _select_actions_schema_version(text: str, *, use_v3: bool) -> str:
    from juno_core_v3.actions.llm_extractor import select_actions_schema_version

    return select_actions_schema_version(text, use_v3=use_v3)


def _normalize_action_required_fields(envelope: dict[str, Any], *, source_text: str) -> None:
    actions = envelope.get("actions")
    if not isinstance(actions, list):
        return
    top_confidence = _coerce_float(envelope.get("confidence"), default=0.9)
    for action in actions:
        if not isinstance(action, dict):
            continue
        if _coerce_float(action.get("confidence"), default=-1.0) < 0.0:
            action["confidence"] = top_confidence
        evidence = str(action.get("evidence_span") or "").strip()
        if not evidence:
            repaired = _repair_action_evidence_span(action, source_text=source_text)
            if repaired:
                action["evidence_span"] = repaired


def _repair_action_evidence_span(action: dict[str, Any], *, source_text: str) -> str | None:
    source = re.sub(r"\s+", " ", (source_text or "").strip())
    if not source:
        return None
    body = str(action.get("body") or "").strip()
    when_text = str(action.get("when_text") or "").strip()
    if body and _token_overlap_ratio(body, source) >= 0.6:
        return source
    if when_text and _span_key_local(when_text) in _span_key_local(source):
        return source
    return None


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _span_key_local(text: str) -> str:
    s = str(text or "").casefold().strip()
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def _token_overlap_ratio(needle: str, haystack: str) -> float:
    needle_tokens = [t for t in _span_key_local(needle).split() if len(t) > 2]
    if not needle_tokens:
        return 0.0
    haystack_tokens = set(_span_key_local(haystack).split())
    matches = sum(1 for token in needle_tokens if token in haystack_tokens)
    return matches / len(needle_tokens)


def _normalize_compound_envelope(envelope: dict[str, Any], *, source_text: str, now: Any) -> None:
    parsed = _compound_parse(source_text, now=now)
    if parsed is None:
        return
    envelope["intent"] = "execute_action"
    envelope["should_execute"] = True
    envelope["confidence"] = max(float(envelope.get("confidence") or 0.0), 0.9)
    envelope["decision_evidence_span"] = source_text
    envelope["actions"] = parsed


def _compound_parse(text: str, *, now: Any) -> list[dict[str, Any]] | None:
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).strip(" .!?")
    lowered = cleaned.lower()

    list_reminder_chain = _compound_parse_list_reminder_chain(cleaned, now=now)
    if list_reminder_chain is not None:
        return list_reminder_chain

    match = re.search(
        r"\bat (?P<t1>\d{1,2}(?::\d{2})?\s*(?:am|pm)) remind me to (?P<body1>.+?) "
        r"then (?P<t2>\d{1,2}(?::\d{2})?\s*(?:am|pm)?) to (?P<body2>.+)$",
        lowered,
    )
    if match:
        return [
            _compound_action(
                kind="reminder",
                body=match.group("body1"),
                evidence=match.group(0),
                when_text=match.group("t1"),
                now=now,
            ),
            _compound_action(
                kind="reminder",
                body=match.group("body2"),
                evidence=match.group(0),
                when_text=match.group("t2"),
                now=now,
            ),
        ]

    match = re.search(
        r"\bset an alarm for (?P<t1>\d{1,2}(?::\d{2})?\s*(?:am|pm)) and "
        r"a reminder at (?P<t2>\d{1,2}(?::\d{2})?\s*(?:am|pm)?) to (?P<body>.+)$",
        lowered,
    )
    if match:
        return [
            _compound_action(
                kind="alarm",
                body=f"alarm for {match.group('t1')}",
                evidence=match.group(0),
                when_text=match.group("t1"),
                now=now,
            ),
            _compound_action(
                kind="reminder",
                body=match.group("body"),
                evidence=match.group(0),
                when_text=match.group("t2"),
                now=now,
            ),
        ]

    match = re.search(
        r"\bcreate (?:a )?(?P<list>.+?) list with (?P<items>.+?) and remind me "
        r"(?P<when>tomorrow at \d{1,2}(?::\d{2})?\s*(?:am|pm)|at \d{1,2}(?::\d{2})?\s*(?:am|pm)) "
        r"to (?P<body>.+)$",
        lowered,
    )
    if match:
        link_id = "create-list-1"
        return [
            {
                "kind": "reminder",
                "operation": "create",
                "body": match.group("items").strip(),
                "evidence_span": match.group(0),
                "container": {"list_name": _title_name(match.group("list"))},
                "link_id": link_id,
            },
            _compound_action(
                kind="reminder",
                body=match.group("body"),
                evidence=match.group(0),
                when_text=match.group("when"),
                now=now,
                links_to=link_id,
            ),
        ]

    match = re.search(
        r"\bremind me at (?P<times>.+?) to (?P<body>.+)$",
        lowered,
    )
    if match:
        times = re.findall(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)", match.group("times"))
        if len(times) >= 2:
            return [
                _compound_action(
                    kind="reminder",
                    body=match.group("body"),
                    evidence=match.group(0),
                    when_text=t,
                    now=now,
                )
                for t in times
            ]

    return None


def _compound_parse_list_reminder_chain(text: str, *, now: Any) -> list[dict[str, Any]] | None:
    clauses = [
        _strip_compound_clause_prefix(clause)
        for clause in re.split(r"[.!?]+|\bthen\b", text or "", flags=re.IGNORECASE)
    ]
    clauses = [clause for clause in clauses if clause]
    if len(clauses) < 2:
        return None

    actions: list[dict[str, Any]] = []
    saw_list_append = False
    saw_reminder = False
    for clause in clauses:
        list_match = re.search(
            r"\b(?:add|put|append)\s+(?P<body>.+?)\s+to\s+(?:my|the)\s+(?P<list>.+?)\s+list\b",
            clause,
            flags=re.IGNORECASE,
        )
        if list_match is not None:
            saw_list_append = True
            actions.append({
                "kind": "reminder",
                "operation": "append_to",
                "body": list_match.group("body").strip(" ,.;:!?-"),
                "evidence_span": clause,
                "container": {"list_name": _title_name(list_match.group("list"))},
            })
            continue

        reminder = _compound_reminder_from_clause(clause, now=now)
        if reminder is not None:
            saw_reminder = True
            actions.append(reminder)

    if len(actions) >= 2 and saw_list_append and saw_reminder:
        return actions
    return None


def _strip_compound_clause_prefix(clause: str) -> str:
    out = re.sub(r"\s+", " ", str(clause or "").strip(" \t\r\n,.;:!?-"))
    for _ in range(4):
        next_out = re.sub(
            r"^(?:and|then|also|please|okay|ok|um+|uh+|a+|hey\s+juno)\b[\s,.;:!?-]*",
            "",
            out,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n,.;:!?-")
        if next_out == out:
            break
        out = next_out
    return out


_COMPOUND_VAGUE_WHEN_RE = (
    r"this weekend|in a bit|soon|later|tonight|this evening|after lunch|"
    r"this morning|this afternoon|tomorrow|next week"
)


def _compound_reminder_from_clause(clause: str, *, now: Any) -> dict[str, Any] | None:
    match = re.search(
        rf"\b(?P<when_before>{_COMPOUND_VAGUE_WHEN_RE})\b[\s,.;:!?-]*"
        r"(?:please\s+)?remind me\s+to\s+(?P<body>.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    when_text = ""
    body = ""
    if match is not None:
        when_text = match.group("when_before").strip()
        body = match.group("body").strip(" ,.;:!?-")
    else:
        match = re.search(
            rf"\b(?:please\s+)?remind me\s+(?:(?P<when_after>{_COMPOUND_VAGUE_WHEN_RE})\s+)?"
            r"to\s+(?P<body>.+)$",
            clause,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        when_text = (match.group("when_after") or "").strip()
        body = match.group("body").strip(" ,.;:!?-")
    if not body:
        return None
    action = {
        "kind": "reminder",
        "operation": "create",
        "body": body,
        "evidence_span": clause,
    }
    if when_text:
        action["when_text"] = when_text
        bucket = _compound_vague_bucket(when_text)
        if bucket:
            action["schedule"] = {"kind": "vague", "vague": {"bucket": bucket}}
    return action


def _compound_vague_bucket(when_text: str) -> str | None:
    key = re.sub(r"\s+", " ", str(when_text or "").strip().casefold())
    if key in {"this weekend", "weekend"}:
        return "weekend"
    if key in {"in a bit", "soon"}:
        return "soon"
    if key == "later":
        return "later"
    if key == "tonight":
        return "tonight"
    if key in {"this evening"}:
        return "evening"
    if key == "after lunch":
        return "afternoon"
    if key == "this morning":
        return "morning"
    if key == "this afternoon":
        return "afternoon"
    if key == "next week":
        return "next_week"
    return None


def _compound_action(
    *,
    kind: str,
    body: str,
    evidence: str,
    when_text: str,
    now: Any,
    links_to: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": kind,
        "operation": "create",
        "body": body.strip(" .!?"),
        "evidence_span": evidence,
        "when_text": when_text.strip(),
    }
    iso = _instant_iso(when_text, now=now)
    if iso:
        out["schedule"] = {"kind": "instant", "instant": {"iso": iso, "tz": None}}
    if links_to:
        out["links_to"] = links_to
    return out


def _instant_iso(when_text: str, *, now: Any) -> str | None:
    try:
        parsed = parse_when(when_text, now=now if isinstance(now, dt.datetime) else None)
    except Exception:  # noqa: BLE001
        parsed = None
    return parsed.iso if parsed is not None else None


def _normalize_container_envelope(envelope: dict[str, Any], *, source_text: str) -> None:
    actions = envelope.get("actions")
    if not isinstance(actions, list) or not actions:
        return
    for action in actions:
        if not isinstance(action, dict):
            continue
        evidence = action.get("evidence_span")
        clause = str(evidence or "").strip() if isinstance(evidence, str) else ""
        if not clause or (len(actions) > 1 and clause == source_text):
            continue
        parsed = _container_parse(clause)
        if parsed is None and len(actions) == 1:
            parsed = _container_parse(source_text)
        if parsed is None:
            continue
        operation, body, list_name, target_desc = parsed
        action["kind"] = "reminder"
        action["operation"] = operation
        action["container"] = {"list_name": list_name}
        if body is not None:
            action["body"] = body
        if target_desc:
            action["target"] = {"ref_kind": "by_description", "description": target_desc}


def _container_parse(text: str) -> tuple[str, str | None, str, str | None] | None:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .!?")
    clauses = [c.strip(" ,.;:!?-") for c in re.split(r"[.!?;]+|\bthen\b", cleaned) if c.strip(" ,.;:!?-")]
    for clause in clauses or [cleaned]:
        match = re.search(r"\badd (?P<body>.+?) to (?:my |the )?(?P<list>.+?)(?: list)?$", clause)
        if match:
            return ("append_to", match.group("body").strip(), _title_name(match.group("list")), None)
        match = re.search(r"\bcreate (?:a )?(?P<list>.+?) list(?: for (?P<for>.+?))?(?: with (?P<body>.+))?$", clause)
        if match:
            name = match.group("for") or match.group("list")
            return ("create", (match.group("body") or "").strip() or None, _title_name(name), None)
        match = re.search(r"\b(?:show|read|list) (?:me )?(?:my )?(?P<list>.+?) list$", clause)
        if match:
            return ("query", "", _title_name(match.group("list")), None)
        match = re.search(r"\bremove (?P<body>.+?) from (?:my |the )?(?P<list>.+?)(?: list)?$", clause)
        if match:
            body = match.group("body").strip()
            return ("remove_from", body, _title_name(match.group("list")), body)
        match = re.search(r"\bmove (?P<body>.+?) to (?:my |the )?(?P<list>.+?)(?: list)?$", clause)
        if match:
            body = match.group("body").strip()
            return ("update", body, _title_name(match.group("list")), body)
    return None


def _title_name(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = re.sub(r"\s+list$", "", cleaned, flags=re.IGNORECASE).strip()
    return " ".join(part.capitalize() for part in cleaned.split())


def _normalize_create_envelope(envelope: dict[str, Any], *, source_text: str) -> None:
    actions = envelope.get("actions")
    if not isinstance(actions, list):
        return
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("operation") or "create").strip().lower() != "create":
            continue
        local_text = " ".join(
            str(x or "")
            for x in (
                action.get("evidence_span"),
                action.get("body"),
                action.get("title"),
            )
        )
        text = local_text if local_text.strip() else source_text
        sink_kind = _create_sink_kind_for_text(text)
        if sink_kind is not None:
            action["kind"] = sink_kind
        if action.get("kind") == "note":
            action.pop("schedule", None)
            action.pop("when_text", None)
            body = _note_body_for_text(source_text)
            if body:
                action["body"] = body


def _create_sink_kind_for_text(text: str) -> str | None:
    lowered = (text or "").casefold()
    if "set an alarm" in lowered or " alarm" in lowered or "wake me" in lowered:
        return "alarm"
    if "remind me" in lowered or " reminder" in lowered:
        return "reminder"
    if "take a note" in lowered or "save a note" in lowered or "note that" in lowered or "save this quote" in lowered:
        return "note"
    return None


def _note_body_for_text(text: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    lowered = cleaned.casefold()
    patterns = (
        r"\bsave a note about (?P<body>.+)$",
        r"\btake a note about (?P<body>.+)$",
        r"\btake a note that (?P<body>.+)$",
        r"\bnote that (?P<body>.+)$",
        r"\bsave this quote:?\s*(?P<body>.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            body = match.group("body").strip(" .!?")
            return body or None
    return None


def _normalize_recurrence_envelope(envelope: dict[str, Any], *, source_text: str, now: Any) -> None:
    actions = envelope.get("actions")
    if not isinstance(actions, list):
        return
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("operation") or "create").strip().lower() != "create":
            continue
        if str(action.get("kind") or "").strip().lower() not in {"reminder", "alarm"}:
            continue
        text = " ".join(
            str(x or "")
            for x in (
                action.get("evidence_span"),
                action.get("body"),
                action.get("when_text"),
                source_text,
            )
        )
        count = _daily_recurrence_count_for_text(text)
        if count is None:
            continue
        schedule = action.get("schedule") if isinstance(action.get("schedule"), dict) else {}
        series = schedule.get("series") if isinstance(schedule.get("series"), dict) else {}
        updated_series = dict(series)
        updated_series["freq"] = "DAILY"
        updated_series["interval"] = int(updated_series.get("interval") or 1)
        updated_series["count"] = count
        if not str(updated_series.get("first_occurrence_iso") or "").strip():
            anchor = _series_anchor_iso(schedule, now=now)
            if anchor:
                updated_series["first_occurrence_iso"] = anchor
        action["schedule"] = {"kind": "series", "series": updated_series}


def _daily_recurrence_count_for_text(text: str) -> int | None:
    lowered = re.sub(r"\s+", " ", (text or "").casefold())
    if "every day" not in lowered and "daily" not in lowered and "for the next" not in lowered:
        return None
    match = re.search(r"\bnext (?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten|thirty) days?\b", lowered)
    if match:
        return _small_number(match.group("n"))
    match = re.search(r"\bfor (?:the )?next (?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten|thirty) days?\b", lowered)
    if match:
        return _small_number(match.group("n"))
    if re.search(r"\bfor (?:the )?next week\b", lowered):
        return 7
    if re.search(r"\bfor (?:a|one|the next) month\b", lowered):
        return 30
    return None


def _series_anchor_iso(schedule: dict[str, Any], *, now: Any) -> str | None:
    instant = schedule.get("instant") if isinstance(schedule.get("instant"), dict) else {}
    iso = str(instant.get("iso") or "").strip()
    if iso:
        return iso
    if isinstance(now, dt.datetime):
        anchor = now
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=dt.timezone.utc)
        anchor = anchor + dt.timedelta(days=1)
        anchor = anchor.replace(hour=9, minute=0, second=0, microsecond=0)
        return anchor.isoformat()
    return None


def _small_number(raw: str) -> int | None:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "thirty": 30,
    }
    text = str(raw or "").strip().lower()
    if text.isdigit():
        return int(text)
    return words.get(text)


def _normalize_operations_envelope(envelope: dict[str, Any], *, source_text: str) -> None:
    actions = envelope.get("actions")
    if not isinstance(actions, list):
        return
    for action in actions:
        if not isinstance(action, dict):
            continue
        operation = str(action.get("operation") or "create").strip().lower()
        if operation == "create":
            continue
        text = " ".join(
            str(x or "")
            for x in (
                action.get("evidence_span"),
                action.get("body"),
                action.get("title"),
                source_text,
            )
        )
        sink_kind = _sink_kind_for_text(text)
        if sink_kind is not None:
            action["kind"] = sink_kind
        target = action.get("target") if isinstance(action.get("target"), dict) else None
        if target is None:
            target = _target_for_operation_text(operation=operation, text=text)
            if target is not None:
                action["target"] = target
        if not str(action.get("body") or "").strip() and isinstance(action.get("target"), dict):
            desc = str(action["target"].get("description") or "").strip()
            if desc:
                action["body"] = desc


def _sink_kind_for_text(text: str) -> str | None:
    lowered = (text or "").casefold()
    # Prefer explicit spoken sink words over the model's guess. "Reminder
    # series" is still a reminder, not an alarm.
    if "reminder" in lowered or "reminders" in lowered:
        return "reminder"
    if "alarm" in lowered or "wake me" in lowered:
        return "alarm"
    if "note" in lowered:
        return "note"
    return None


def _target_for_operation_text(*, operation: str, text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .!?")
    if operation == "query":
        return {"ref_kind": "by_query", "filter": {"text_match": _query_text_match(cleaned)}}
    if operation == "snooze" and re.search(r"\b(that|it|this|the last one)\b", cleaned):
        return {"ref_kind": "by_pronoun", "pronoun": "that"}
    desc = _description_target_text(cleaned)
    if desc:
        return {"ref_kind": "by_description", "description": desc}
    if re.search(r"\b(that|it|this|the last one)\b", cleaned):
        return {"ref_kind": "by_pronoun", "pronoun": "that"}
    return None


def _description_target_text(text: str) -> str | None:
    patterns = (
        r"\bmark (?:the )?(?P<desc>.+?) (?:one )?(?:as )?done\b",
        r"\b(?:cancel|delete|stop) (?:the )?(?P<desc>.+?)(?: reminder| alarm| note|$)",
        r"\b(?:move|push|shift|delay|bump) (?:the |my )?(?P<desc>.+?)(?: by | to | from |$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            desc = match.group("desc").strip(" .!?")
            desc = re.sub(r"\b(?:that|the|my)\b", "", desc).strip()
            return desc or None
    return None


def _query_text_match(text: str) -> str | None:
    match = re.search(r"\babout (?P<desc>.+)$", text)
    if match:
        return match.group("desc").strip(" .!?") or None
    return None


def _normalize_vague_time_envelope(envelope: dict[str, Any], *, source_text: str) -> None:
    actions = envelope.get("actions")
    if not isinstance(actions, list):
        return
    decision_text = str(envelope.get("decision_evidence_span") or "")
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("operation") or "create").strip().lower() != "create":
            continue
        if str(action.get("kind") or "").strip().lower() not in {"reminder", "alarm"}:
            continue
        spoken_body = _vague_body_for_text(source_text)
        if spoken_body:
            action["body"] = spoken_body
        elif not str(action.get("body") or "").strip():
            title = str(action.get("title") or "").strip()
            if title:
                action["body"] = title
        schedule = action.get("schedule")
        if isinstance(schedule, dict) and str(schedule.get("kind") or "").strip().lower() == "vague":
            _canonicalize_vague_schedule(action, schedule)
            continue
        bucket = _vague_bucket_for_text(
            " ".join(
                str(x or "")
                for x in (
                    action.get("evidence_span"),
                    action.get("when_text"),
                    decision_text,
                    source_text,
                )
            )
        )
        if bucket is not None:
            action["schedule"] = {
                "kind": "vague",
                "vague": {
                    "bucket": bucket,
                    "default_iso": None,
                    "tz": None,
                    "needs_confirmation": True,
                },
            }
            action["needs_confirmation"] = True


def _canonicalize_vague_schedule(action: dict[str, Any], schedule: dict[str, Any]) -> None:
    vague = schedule.get("vague") if isinstance(schedule.get("vague"), dict) else {}
    bucket = str(vague.get("bucket") or schedule.get("bucket") or "").strip().lower()
    if not bucket:
        return
    action["schedule"] = {
        "kind": "vague",
        "vague": {
            "bucket": bucket,
            "default_iso": vague.get("default_iso", schedule.get("default_iso")),
            "tz": vague.get("tz", schedule.get("tz")),
            "needs_confirmation": True,
        },
    }
    action["needs_confirmation"] = True


def _vague_bucket_for_text(text: str) -> str | None:
    lowered = re.sub(r"\s+", " ", (text or "").casefold())
    checks = (
        ("this weekend", "weekend"),
        ("the weekend", "weekend"),
        ("tomorrow morning", "morning"),
        ("this morning", "morning"),
        ("tonight", "evening"),
        ("this evening", "evening"),
        ("in the evening", "evening"),
        ("after lunch", "afternoon"),
        ("after dinner", "evening"),
        ("in a bit", "soon"),
        ("soon", "soon"),
        ("later", "later"),
        ("next week", "next_week"),
    )
    for needle, bucket in checks:
        if needle in lowered:
            return bucket
    return None


def _vague_body_for_text(text: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    lowered = cleaned.casefold()
    patterns = (
        r"\bremind me to (?P<body>.+)$",
        r"\bremind me later about (?P<body>.+)$",
        r"\bremind me in a bit to (?P<body>.+)$",
        r"\bremind me soon to (?P<body>.+)$",
        r"\bremind me about (?P<body>.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            body = match.group("body").strip(" .!?")
            return body or None
    return None


def resolve_dictation_surface_id(
    surface_id: str | None,
    app_bundle_id: str | None,
) -> str:
    """Pick INSERT ``surface_id`` for one-shot dictation.

    When the Mac shell sends ``app_bundle_id`` but omits ``surface_id``,
    treat the request as ``mac_overlay`` — not ``workbench_dev`` — so
    product surface policy applies (repair doc P5).
    """
    sid = str(surface_id).strip() if surface_id else ""
    if sid:
        return sid
    if app_bundle_id and str(app_bundle_id).strip():
        return SurfaceId.MAC_OVERLAY.value
    return SurfaceId.WORKBENCH_DEV.value


class WorkbenchApp:
    def __init__(
        self,
        config: WorkbenchRuntimeConfig,
        *,
        session_id: str | None = None,
        recorder: TraceRecorder | None = None,
        store: WorkbenchStore | None = None,
        commit: CommitController | None = None,
        transcriber: DictationTranscriber | None = None,
        memory: JsonMemoryStore | None = None,
        # --- Engine-component injections (wired by ProductionServiceRunner
        # and the registry launcher so the one-shot dictation HTTP path
        # runs through the exact same context / bias / writer stack that
        # the streaming DictationSessionRunner uses). Each is optional:
        # standalone workbench runs without a loaded engine still work,
        # just with fewer features.
        context_provider: ContextProvider | None = None,
        writer_service: WriterService | None = None,
        live_corrector_service: WriterService | None = None,
        writer_backend: WriterBackend | None = None,
        language_planner: LanguagePlanner | None = None,
        bias_engine: RecognitionBiasEngine | None = None,
        clipboard_ring: ClipboardRingBuffer | None = None,
        context_plane: ContextPlane | None = None,
        # Hot-swap shim around the final-ASR backend. When set (the
        # ProductionServiceRunner wires this), the workbench exposes
        # /api/broker/runtime/backends and /swap_final so users can
        # compare different ASR models live without restarting the
        # service. None for standalone workbench runs.
        final_swap: SwappableFinalBackend | None = None,
        juno_seed_runtime: JunoSeedPersonalizationRuntime | None = None,
    ) -> None:
        self.final_swap = final_swap
        self.config = config
        from juno_v2.runtime.local_broker_token import ensure_local_broker_token
        from juno_v2.runtime.paths import (
            juno_bundle_id,
            migrate_legacy_support_root,
        )

        # Ensure on-disk layout matches the bundle-id-keyed convention
        # before any subsystem reads from disk (token, settings, history).
        migrate_legacy_support_root()
        ensure_local_broker_token()

        # Engine identity (production-grade revamp, Phase 1). The
        # ``ProductionServiceRunner`` overrides ``runtime_role`` to
        # ``"juno_runtime_service"`` after constructing the app; the Swift
        # shell refuses to attach to anything else. Defaults below
        # describe the standalone ``python -m juno_v2.workbench.server``
        # path that powers the dev workbench UI.
        import os
        import time
        import uuid

        self.runtime_role: str = "workbench_standalone"
        self.instance_id: str = uuid.uuid4().hex
        self.bundle_id: str = juno_bundle_id()
        self.process_pid: int = os.getpid()
        self.started_at: float = time.time()
        self.deployment_profile: Dict[str, Any] = {}
        # Optional reference to the streaming :class:`DictationSessionRunner`,
        # set by ``ProductionServiceRunner`` after the engine is built. The
        # broker live-caption setter mutates ``preview_decode_enabled`` on
        # this runner so the next utterance reflects the toggle without
        # needing to rebuild the engine.
        self.dictation_runner: DictationSessionRunner | None = None

        self.session_id = session_id or new_session_id(prefix="workbench")
        self.recorder = recorder or TraceRecorder(self.session_id, config.log_dir, recent_limit=config.recent_event_limit)
        # Issue #9: persist manual / custom writer mode across broker
        # restart. The selection store reads from disk on construct so
        # ``WorkbenchStore`` can hydrate state.{manual,custom}_writer_mode
        # before the first snapshot. ``default_surface`` and unknown ids
        # are coerced to ``None`` on load (#5 interaction).
        from juno_v2.workbench.mode_selection_store import (
            WriterModeSelectionStore,
            default_writer_mode_selection_path,
        )

        self.mode_selection_store = WriterModeSelectionStore(
            default_writer_mode_selection_path(self.config.log_dir)
        )
        self.store = store or WorkbenchStore(
            self.recorder,
            mode_selection_store=self.mode_selection_store,
        )
        self.commit = commit or CommitController(self.store)
        self._lock = self.store.lock
        self.actions_index: ActionsIndex | None = None
        self._inference_scheduler = InferenceScheduler(name="juno-broker-inference")
        self._utterance_lifecycle_guard = threading.Lock()
        self._utterance_lifecycle_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="juno-utterance-lifecycle",
        )
        self._action_preview_display_lock = threading.Lock()
        self._action_preview_display_by_utterance: dict[str, str] = {}
        try:
            from juno_v2.observability.actions_index import get_actions_index

            self.actions_index = get_actions_index(Path(config.log_dir))
        except Exception:
            logger.exception("failed_to_initialize_actions_index")
        self.broker = BrokerFacade(
            workbench_session_id=self.session_id,
            recorder=self.recorder,
            log_dir=Path(config.log_dir),
            writer_backend=writer_backend,
        )
        action_extractor_source = writer_service if writer_service is not None else writer_backend
        if action_extractor_source is not None:
            try:
                from juno_core_v3.actions.llm_extractor import set_llm_extractor

                set_llm_extractor(build_writer_action_extractor(action_extractor_source))
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "actions_llm_extractor_registered",
                    {
                        "writer_backend": getattr(
                            action_extractor_source,
                            "backend_name",
                            type(action_extractor_source).__name__,
                        )
                    },
                )
            except Exception:
                logger.exception("failed_to_register_actions_llm_extractor")
        else:
            try:
                from juno_core_v3.actions.llm_extractor import set_llm_extractor

                set_llm_extractor(None)
            except Exception:
                logger.exception("failed_to_clear_actions_llm_extractor")
        if context_plane is not None:
            self.broker.context_plane = context_plane
        # Personalization store: caller-injected in production (the
        # runtime service passes the engine's store so corrections and
        # snippets are shared with the live session). Standalone workbench
        # creates its own on disk so the tool registry still works.
        if memory is not None:
            self.memory = memory
        else:
            self.memory = None
            memory_dir = Path(config.log_dir) / "memory"
            try:
                from juno_v2.memory.store import JsonMemoryStore

                self.memory = JsonMemoryStore(memory_dir)
            except Exception:
                self.memory = None
        self.tools = ToolRegistry()
        register_builtin_tools(self.tools, memory=self.memory)
        # One-shot dictation transcriber resolution order:
        #   1) caller-injected (production runtime service, tests).
        #   2) JUNO_FINAL_BACKEND env vars — standalone workbench mode.
        #   3) UnavailableTranscriber — explicit, loud "not configured"
        #      state. We never paste a placeholder transcript.
        # We resolve at __init__ time so doctor/health endpoints can
        # report the actual backend name without a request.
        self.transcriber: DictationTranscriber = (
            transcriber if transcriber is not None else resolve_transcriber_from_env()
        )
        self._transcriber_was_injected = transcriber is not None

        # Cold-start warmup tracking. resolve_transcriber_from_env now uses
        # ``snapshot_download(local_files_only=True)`` so workbench startup
        # is non-blocking even when the configured model is an HF repo id
        # not yet cached. We schedule a background pre-warm to populate
        # the cache so the first ingest_wav call doesn't pay the multi-GB
        # download latency, and surface progress via ``/healthz`` and
        # ``/api/broker/engine/compatibility`` so the shell can show a
        # "Setting up voice engine…" surface instead of "engine
        # unreachable" during the warmup window.
        #
        # warm_state values:
        #   "ready"   — model is on disk (cache hit at startup, or already
        #               local path / non-HF backend).
        #   "warming" — pre-warm thread is running snapshot_download.
        #   "error"   — pre-warm raised; the first dictation call will
        #               surface the underlying failure.
        self.warm_state: str = "ready"
        self.warm_error: str | None = None
        self._warm_thread: threading.Thread | None = None
        self._maybe_start_prewarm()
        # Mac capability probe. Resolves the Swift helper lazily; if the
        # helper binary isn't installed, ``decide()`` returns a
        # ``helper_not_installed`` decision so the shell can show a
        # helpful message. We keep the checker as an instance member so
        # callers / tests can inject a fake via ``app.capability``.
        #
        # Suppression config: the capability gate consults an optional
        # JSON file (``config/suppression_apps.json`` by default) for
        # extra bundle-ID / window-title block rules on top of the
        # baked-in defaults. We resolve the path in this order:
        #   1) JUNO_SUPPRESSION_CONFIG env var (explicit override).
        #   2) ``config/suppression_apps.json`` relative to the repo root.
        # Missing or malformed files are logged but non-fatal — we fall
        # back to the empty default config so the gate still works.
        suppression_cfg = self._load_suppression_config()
        self.capability = CapabilityChecker(suppression_config=suppression_cfg)
        # Clipboard ring buffer: bounded, redacted history of text the
        # user has pasted via juno. The context plane reads from this
        # when building a ContextPacket so the writer / tools see recent
        # clipboard entries as context. The ring is populated lazily on
        # ``/api/broker/insertion/committed`` (and explicitly via
        # ``/api/broker/clipboard/push``).
        #
        # Production deployments share a single ring with the streaming
        # ``DictationSessionRunner`` so clipboard history is unified
        # across one-shot and streaming paths. Standalone workbench
        # runs create a private ring.
        self.clipboard_ring = clipboard_ring if clipboard_ring is not None else ClipboardRingBuffer()
        self.custom_mode_store = CustomModeStore(default_modes_data_path(self.config.log_dir))
        self.surface_preset_store = SurfacePresetStore(default_surface_presets_path(self.config.log_dir))
        self.custom_transform_store = CustomTransformStore(default_transforms_data_path(self.config.log_dir))

        self.juno_seed_runtime = juno_seed_runtime
        if self.juno_seed_runtime is None and self.memory is not None:
            from juno_v2.personalization.seed.runtime import JunoSeedPersonalizationRuntime

            self.juno_seed_runtime = JunoSeedPersonalizationRuntime.try_load(memory_store=self.memory)
        if self.juno_seed_runtime is not None and self.memory is not None:
            memory_dir = Path(getattr(self.memory, "memory_dir", Path(self.config.log_dir) / "memory"))
            sentinel = memory_dir / ".seeded_v1"
            if not sentinel.exists():
                try:
                    memory_dir.mkdir(parents=True, exist_ok=True)
                    self.juno_seed_runtime.promotion.run_initial_promotion(self.memory)
                    sentinel.touch()
                except Exception:
                    # Never block startup on seeding failures.
                    pass

        # Context / bias / writer components for the one-shot dictation
        # pipeline. Production deployments (``ProductionServiceRunner``
        # and ``registry_launcher.serve_from_config``) inject these from
        # the engine artifacts so the HTTP transcribe endpoint reuses
        # the same context provider and writer backend the streaming
        # ``DictationSessionRunner`` uses. Standalone workbench runs
        # (the live macOS shell launches via ``python -m
        # juno_v2.workbench.server``) used to leave this as ``None``,
        # which silently bypassed the entire deterministic writer plane
        # (filler-strip, backtrack, chant-strip, list rendering) on the
        # primary user dictation path. The deterministic pipeline does
        # not need an LLM backend, so we construct a backendless
        # service when no injection happens -- plain dictation then
        # gets the same writer rules covered by replay validation.
        self.context_provider = context_provider
        if writer_service is not None:
            self.writer_service = writer_service
        else:
            from juno_v2.writer.config import WriterConfig
            from juno_v2.writer.service import WriterService

            self.writer_service = WriterService(
                config=WriterConfig(),
                recorder=self.recorder,
                backend=None,
            )
        self.language_planner = language_planner
        self.live_corrector_service = live_corrector_service or self._build_env_live_corrector_service()
        self.live_transcript_adjudicator = None
        if self.live_corrector_service is not None:
            from juno_v2.transcript.adjudicator import TranscriptAdjudicator, TranscriptAdjudicatorConfig

            self.live_transcript_adjudicator = TranscriptAdjudicator(
                backend=self.live_corrector_service,
                recorder=self.recorder,
                config=TranscriptAdjudicatorConfig(
                    live_enabled=True,
                    max_tokens_live=int(os.getenv("JUNO_V2_LIVE_CORRECTOR_MAX_TOKENS", "160") or "160"),
                ),
            )

        self._settings_path = Path(self.config.log_dir) / "broker_settings.json"
        self._settings = self._load_settings()

        from juno_v2.language.normalize import LanguageAwareNormalizer
        from juno_v2.memory.bias import RecognitionBiasEngine

        self.bias_engine = bias_engine or RecognitionBiasEngine()
        self.transcript_normalizer = LanguageAwareNormalizer()
        # One-shot dictation pipeline: context -> bias -> ASR ->
        # normalize -> writer -> learning. Lives here (rather than
        # per-request) so the utterance-id cache survives across calls
        # and ``/api/broker/insertion/committed`` can trigger
        # correction learning with the real (raw, committed) pair.
        _audio_root = product_audio_root(workbench_log_dir=Path(self.config.log_dir))
        # Recovery-floor invariant: audio is always written so any session
        # (including paste/action failures) has a recoverable transcript +
        # audio for at least RECOVERY_AUDIO_FLOOR_DAYS. The user's
        # ``audio_save_enabled`` and ``audio_retention_policy`` settings now
        # control retention *beyond* the floor, never whether audio is
        # written at all. See ``_pipeline_time_retention_days``.
        _audio_save_dir = _audio_root
        self.oneshot_pipeline = OneShotDictationPipeline(
            transcriber=self.transcriber,
            recorder=self.recorder,
            context_provider=self.context_provider,
            memory_store=self.memory,
            bias_engine=self.bias_engine,
            writer_service=self.writer_service,
            language_planner=self.language_planner,
            transcript_normalizer=self.transcript_normalizer,
            capability_gate=None,  # enable via JUNO_ONESHOT_GATE_CAPABILITY=1
            clipboard_ring=self.clipboard_ring,
            context_plane=self.broker.context_plane,
            audio_save_dir=_audio_save_dir,
            audio_retention_limit=1000,
            audio_retention_days=self._pipeline_time_retention_days(),
            writer_enabled=bool(self._settings.get("writer_enabled", True)),
            itn_enabled=bool(self._settings.get("itn_enabled", True)),
            custom_mode_store=self.custom_mode_store,
            surface_preset_store=self.surface_preset_store,
            juno_seed_runtime=self.juno_seed_runtime,
            live_transcript_adjudicator=self.live_transcript_adjudicator,
        )
        self._maybe_attach_capability_gate()
        # Setup install state: None = unknown, "downloading" = in progress,
        # "ready" = done, "failed:..." = error message.
        self._setup_install_state: str | None = None
        self._setup_install_lock = threading.Lock()
        # Live download progress for the model-provisioning step. Populated
        # by ``broker_setup_install`` while a download is in flight and read
        # by ``broker_setup_status`` so the onboarding UI can render a real
        # progress bar with bytes-downloaded / total + a rolling speed and
        # ETA. ``None`` whenever no install is active.
        self._setup_install_progress: dict[str, Any] | None = None

    def _build_env_live_corrector_service(self) -> Any | None:
        if not _env_bool("JUNO_V2_LIVE_CORRECTOR_ENABLED", False):
            return None
        backend_name = (os.getenv("JUNO_V2_LIVE_CORRECTOR_BACKEND") or "").strip()
        if not backend_name:
            return None
        from juno_v2.runtime.backends import create_writer_backend
        from juno_v2.writer.config import WriterConfig
        from juno_v2.writer.service import WriterService

        cfg = WriterConfig(
            backend_name=backend_name,
            local_http_endpoint=os.getenv("JUNO_V2_LIVE_CORRECTOR_ENDPOINT") or None,
            model_path=os.getenv("JUNO_V2_LIVE_CORRECTOR_MODEL_PATH") or None,
            max_tokens=int(os.getenv("JUNO_V2_LIVE_CORRECTOR_MAX_TOKENS", "160") or "160"),
            temperature=float(os.getenv("JUNO_V2_LIVE_CORRECTOR_TEMPERATURE", "0.0") or "0.0"),
            top_p=float(os.getenv("JUNO_V2_LIVE_CORRECTOR_TOP_P", "1.0") or "1.0"),
            residency_policy=os.getenv("JUNO_V2_LIVE_CORRECTOR_RESIDENCY_POLICY", "resident"),
        )
        return WriterService(config=cfg, recorder=self.recorder, backend=create_writer_backend(cfg))

    # Minimum days every retained audio file is kept, regardless of the
    # user's audio retention setting. Guarantees a window during which the
    # History detail pane can offer "Insert again" / "Re-run" recovery
    # actions on a failed session without the audio having been pruned.
    RECOVERY_AUDIO_FLOOR_DAYS = 7

    def _pipeline_time_retention_days(self) -> int:
        """In-pipeline mtime prune horizon; ``0`` skips time prune (``forever``).

        Honours the user's preference but enforces a recovery floor: when
        the user has set the audio policy to ``off`` or has the per-session
        ``audio_save_enabled`` toggle off, audio is still retained for
        :attr:`RECOVERY_AUDIO_FLOOR_DAYS` so we can recover from paste /
        action failures. ``forever`` still means forever.
        """
        floor = max(1, int(self.RECOVERY_AUDIO_FLOOR_DAYS))
        policy = str(self._settings.get("audio_retention_policy") or "days").strip().lower()
        if policy == "forever":
            # No time-based prune. The retention_limit cap still applies.
            return 0
        # Respect ``audio_save_enabled`` as the user's "I don't want
        # long-term storage" lever, but never below the recovery floor.
        if not bool(self._settings.get("audio_save_enabled", True)):
            return floor
        if policy == "off":
            return floor
        try:
            d = int(self._settings.get("audio_retention_days") or 30)
        except Exception:
            d = 30
        return max(floor, d)

    def _load_settings(self) -> Dict[str, Any]:
        defaults = {
            "writer_enabled": True,
            "itn_enabled": True,
            "audio_save_enabled": True,
            # When True (default) the engine streams per-utterance preview
            # decodes for the live HUD caption. When False the engine session
            # skips preview-lane decode invocations to save CPU/GPU; the model
            # remains downloaded and the resident streaming-preview service
            # stays warm so toggling back on is cheap. Read once at session
            # start; mutating mid-session is undefined. The shell mirrors the
            # Settings toggle via JUNO_V2_LIVE_CAPTION_START_ENABLED and
            # force-disables ineligible hosts via JUNO_V2_LIVE_CAPTION_ALLOWED.
            "live_caption_enabled": _env_bool('JUNO_V2_LIVE_CAPTION_DEFAULT_ENABLED', True),
            "language_mode": "auto",
            "smart_context": True,
            "use_selected_text": True,
            "use_focused_text": True,
            "use_window_title": True,
            "use_clipboard": False,
            "learn_from_corrections": True,
            "save_history": True,
            "save_audio": True,
            "per_app": {},
            # Retention policies used by the macOS Settings surface.
            # policy: "forever" | "off" | "days"
            "audio_retention_policy": "days",
            "audio_retention_days": 30,
            "history_retention_policy": "days",
            "history_retention_days": 90,
        }
        try:
            if self._settings_path.exists():
                raw = json.loads(self._settings_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    defaults.update(raw)
        except Exception:
            pass
        if 'JUNO_V2_LIVE_CAPTION_START_ENABLED' in os.environ:
            defaults["live_caption_enabled"] = _env_bool('JUNO_V2_LIVE_CAPTION_START_ENABLED', False)
        if not _env_bool('JUNO_V2_LIVE_CAPTION_ALLOWED', True):
            defaults["live_caption_enabled"] = False
        return defaults

    def _persist_settings(self) -> None:
        try:
            self._settings_path.write_text(
                json.dumps(self._settings, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def broker_settings_get(self) -> Dict[str, Any]:
        return {"ok": True, "settings": dict(self._settings)}

    def broker_privacy_context_settings_get(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "settings": self._privacy_context_settings(),
        }

    def broker_privacy_context_settings_set(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for key in (
            "smart_context",
            "use_selected_text",
            "use_focused_text",
            "use_window_title",
            "use_clipboard",
            "learn_from_corrections",
            "save_history",
            "save_audio",
        ):
            if key in payload:
                self._settings[key] = bool(payload.get(key))
        self._persist_settings()
        if getattr(self, "oneshot_pipeline", None) is not None:
            # Audio is always written (recovery-floor invariant); the
            # user's privacy setting only adjusts retention horizon, which
            # is read fresh from ``_pipeline_time_retention_days``.
            root = product_audio_root(workbench_log_dir=Path(self.config.log_dir))
            self.oneshot_pipeline.audio_save_dir = root
            self.oneshot_pipeline.audio_retention_days = self._pipeline_time_retention_days()
        return self.broker_privacy_context_settings_get()

    def broker_privacy_app_overrides_get(self) -> Dict[str, Any]:
        per_app = self._settings.get("per_app")
        if not isinstance(per_app, dict):
            per_app = {}
        return {"ok": True, "per_app": per_app}

    def broker_privacy_app_overrides_set(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        bundle_id = str(payload.get("bundle_id") or payload.get("app_bundle_id") or "").strip()
        if not bundle_id:
            return {"ok": False, "error": "bundle_id_required"}
        per_app = self._settings.get("per_app")
        if not isinstance(per_app, dict):
            per_app = {}
        current = dict(per_app.get(bundle_id) or {})
        for key in ("use_context", "learn", "save_history", "save_audio"):
            if key in payload:
                raw = payload.get(key)
                if isinstance(raw, bool):
                    current[key] = raw
                else:
                    val = str(raw or "default").strip().lower()
                    current[key] = val if val in {"default", "on", "off"} else "default"
        per_app[bundle_id] = current
        self._settings["per_app"] = per_app
        self._persist_settings()
        return {"ok": True, "bundle_id": bundle_id, "settings": current, "per_app": per_app}

    def broker_language_environment_set(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        lang = str(payload.get("language_mode") or "auto").strip().lower()
        allowed_lang = {"auto", "en", "pair:en,hi", "zh", "es", "keep_original"}
        self._settings["language_mode"] = lang if lang in allowed_lang else "auto"
        self._persist_settings()
        return {"ok": True, "settings": dict(self._settings)}

    def _privacy_context_settings(self) -> Dict[str, Any]:
        return {
            "smart_context": bool(self._settings.get("smart_context", True)),
            "use_selected_text": bool(self._settings.get("use_selected_text", True)),
            "use_focused_text": bool(self._settings.get("use_focused_text", True)),
            "use_window_title": bool(self._settings.get("use_window_title", True)),
            "use_clipboard": bool(self._settings.get("use_clipboard", False)),
            "learn_from_corrections": bool(self._settings.get("learn_from_corrections", True)),
            "save_history": bool(self._settings.get("save_history", True)),
            "save_audio": bool(self._settings.get("save_audio", True)),
        }

    def _app_override_bool(self, app_bundle_id: str | None, key: str, default: bool) -> bool:
        bid = str(app_bundle_id or "").strip()
        per_app = self._settings.get("per_app")
        if not bid or not isinstance(per_app, dict):
            return default
        raw = dict(per_app.get(bid) or {}).get(key, "default")
        if isinstance(raw, bool):
            return raw
        val = str(raw or "default").strip().lower()
        if val == "on":
            return True
        if val == "off":
            return False
        return default

    def _is_secure_context(self, frozen_context: Dict[str, Any] | None) -> bool:
        if not isinstance(frozen_context, dict):
            return False
        for key in ("secure_field", "is_secure_field", "password_field", "is_password_field"):
            if bool(frozen_context.get(key)):
                return True
        role = str(frozen_context.get("focused_role") or "").lower()
        return "secure" in role or "password" in role

    def _sanitize_frozen_context(
        self,
        frozen_context: Dict[str, Any] | None,
        *,
        app_bundle_id: str | None,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        ctx = dict(frozen_context or {})
        privacy = self._privacy_context_settings()
        secure = self._is_secure_context(ctx)
        use_context = self._app_override_bool(app_bundle_id, "use_context", True)
        allow_context = bool(privacy["smart_context"] and use_context and not secure)
        allow_history = self._app_override_bool(app_bundle_id, "save_history", bool(privacy["save_history"])) and not secure
        allow_audio = self._app_override_bool(app_bundle_id, "save_audio", bool(privacy["save_audio"])) and not secure
        allow_learning = self._app_override_bool(app_bundle_id, "learn", bool(privacy["learn_from_corrections"])) and not secure

        if not allow_context or not privacy["use_selected_text"]:
            for key in ("selected_text", "selection_text", "selection"):
                ctx.pop(key, None)
        if not allow_context or not privacy["use_focused_text"]:
            for key in ("focused_text", "focused_value", "field_text"):
                ctx.pop(key, None)
        if not allow_context or not privacy["use_clipboard"]:
            for key in ("clipboard_text", "clipboard", "clipboard_items"):
                ctx.pop(key, None)
        if not allow_context or not privacy["use_window_title"]:
            for key in ("window_title", "frontmost_window_title"):
                ctx.pop(key, None)

        summary = {
            "secure_field": secure,
            "use_context": allow_context,
            "save_history": allow_history,
            "save_audio": allow_audio,
            "learn_from_corrections": allow_learning,
        }
        return (ctx or None), summary

    def broker_settings_set_writer(self, enabled: bool) -> Dict[str, Any]:
        self._settings["writer_enabled"] = bool(enabled)
        self._persist_settings()
        if getattr(self, "oneshot_pipeline", None) is not None:
            self.oneshot_pipeline.writer_enabled = bool(enabled)
        return {"ok": True, "writer_enabled": bool(enabled)}

    def broker_settings_set_itn(self, enabled: bool) -> Dict[str, Any]:
        self._settings["itn_enabled"] = bool(enabled)
        self._persist_settings()
        if getattr(self, "oneshot_pipeline", None) is not None:
            self.oneshot_pipeline.itn_enabled = bool(enabled)
        return {"ok": True, "itn_enabled": bool(enabled)}

    def broker_settings_set_audio_save(self, enabled: bool) -> Dict[str, Any]:
        self._settings["audio_save_enabled"] = bool(enabled)
        self._persist_settings()
        if getattr(self, "oneshot_pipeline", None) is not None:
            # Recovery-floor invariant: audio is always written. This
            # toggle now adjusts retention horizon only (see
            # ``_pipeline_time_retention_days``). When disabled, audio
            # ages out at ``RECOVERY_AUDIO_FLOOR_DAYS``.
            root = product_audio_root(workbench_log_dir=Path(self.config.log_dir))
            self.oneshot_pipeline.audio_save_dir = root
            self.oneshot_pipeline.audio_retention_days = self._pipeline_time_retention_days()
        return {"ok": True, "audio_save_enabled": bool(enabled)}

    def broker_settings_set_live_caption(self, enabled: bool) -> Dict[str, Any]:
        """Persist the live-caption (HUD live transcription) toggle.

        The engine gate lives on the streaming
        :class:`DictationSessionRunner`. Each call to ``_decode_preview``
        re-reads ``runner.preview_decode_enabled`` at the top of the
        function, so flipping this toggle takes effect on the *next*
        utterance — never mid-decode. The model on disk, the resident
        streaming-preview service, and the lifecycle registration of the
        preview backend are all unchanged.
        """
        allowed = _env_bool('JUNO_V2_LIVE_CAPTION_ALLOWED', True)
        enabled = bool(enabled) and allowed
        self._settings["live_caption_enabled"] = enabled
        self._persist_settings()
        runner = getattr(self, "dictation_runner", None)
        if runner is not None and hasattr(runner, "preview_decode_enabled"):
            runner.preview_decode_enabled = enabled
        out = {"ok": True, "live_caption_enabled": enabled}
        if not allowed:
            out["disabled_reason"] = "host_not_eligible"
        return out

    def broker_settings_set_retention(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def _parse_policy(key: str, *, allow_days: bool = True) -> str | None:
            v = str(payload.get(key) or "").strip().lower()
            if v in {"forever", "off"}:
                return v
            if allow_days and v == "days":
                return v
            return None

        def _parse_days(key: str) -> int | None:
            raw = payload.get(key)
            if raw is None:
                return None
            try:
                d = int(raw)
            except Exception:
                return None
            if d <= 0:
                return None
            return d

        audio_policy = _parse_policy("audio_policy")
        hist_policy = _parse_policy("history_policy")
        if audio_policy is None or hist_policy is None:
            return {"ok": False, "error": "invalid_policy"}
        audio_days = _parse_days("audio_days")
        hist_days = _parse_days("history_days")

        self._settings["audio_retention_policy"] = audio_policy
        if audio_policy == "days":
            self._settings["audio_retention_days"] = audio_days or 30
        else:
            self._settings["audio_retention_days"] = None

        self._settings["history_retention_policy"] = hist_policy
        if hist_policy == "days":
            self._settings["history_retention_days"] = hist_days or 90
        else:
            self._settings["history_retention_days"] = None

        self._persist_settings()
        if getattr(self, "oneshot_pipeline", None) is not None:
            self.oneshot_pipeline.audio_retention_days = self._pipeline_time_retention_days()
        return {"ok": True, "settings": dict(self._settings)}

    def broker_retention_run_cleanup(self) -> Dict[str, Any]:
        """Apply the current retention policies (best-effort)."""
        import time as _time
        from juno_v2.observability.history_store import prune_persistent_history_by_days

        out: Dict[str, Any] = {"ok": True}

        # --- Audio pruning (retained per-utterance WAVs)
        # Recovery-floor invariant: even when policy is ``off`` we keep at
        # least RECOVERY_AUDIO_FLOOR_DAYS so the History detail pane can
        # offer Insert again / Re-run on recent failures. The user's own
        # "delete all audio" is a different action (history_clear_all).
        floor = max(1, int(self.RECOVERY_AUDIO_FLOOR_DAYS))
        audio_policy = str(self._settings.get("audio_retention_policy") or "days")
        audio_days_raw = self._settings.get("audio_retention_days")
        candidate = self._candidate_wavs()
        removed_audio = 0
        keep_days: int | None
        if audio_policy == "forever":
            keep_days = None
        elif audio_policy == "off":
            keep_days = floor
        else:  # "days"
            try:
                keep_days = max(floor, int(audio_days_raw or 30))
            except Exception:
                keep_days = max(floor, 30)
        if keep_days is not None:
            cutoff = _time.time() - (keep_days * 86400)
            for p in candidate:
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink(missing_ok=True)
                        removed_audio += 1
                except Exception:
                    continue

        out["audio_removed"] = removed_audio
        out["audio_retention_floor_days"] = floor

        # --- History pruning (persistent compact history.jsonl)
        hist_policy = str(self._settings.get("history_retention_policy") or "days")
        hist_days_raw = self._settings.get("history_retention_days")
        if hist_policy == "off":
            hist_res = prune_persistent_history_by_days(self.config.log_dir, keep_days=0)
        elif hist_policy == "days":
            try:
                keep_days = int(hist_days_raw or 90)
            except Exception:
                keep_days = 90
            hist_res = prune_persistent_history_by_days(self.config.log_dir, keep_days=keep_days)
        else:
            hist_res = {"ok": True, "kept": None, "removed": 0}

        out["history"] = hist_res
        out["settings"] = dict(self._settings)
        return out

    def broker_history_clear_all(self) -> Dict[str, Any]:
        audio_removed = 0
        for p in self._candidate_wavs():
            try:
                p.unlink(missing_ok=True)
                audio_removed += 1
            except Exception:
                continue
        try:
            from juno_v2.observability.product_history import get_product_history_store

            get_product_history_store(Path(self.config.log_dir)).clear_all()
        except Exception:
            pass
        path = Path(self.config.log_dir) / "history.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            return {"ok": True, "audio_removed": audio_removed}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def broker_export_data_zip_bytes(self) -> bytes:
        import io
        import zipfile

        buf = io.BytesIO()
        root = Path(self.config.log_dir)
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            try:
                from juno_v2.observability.product_history import get_product_history_store

                db = get_product_history_store(root).db_path
                if db.exists():
                    zf.write(db, arcname="product_history.sqlite")
            except Exception:
                pass
            history = root / "history.jsonl"
            if history.exists():
                zf.write(history, arcname="history.jsonl")
            mem = root / "memory"
            if mem.exists():
                for p in mem.glob("*.json"):
                    zf.write(p, arcname=f"memory/{p.name}")
        return buf.getvalue()

    def _maybe_attach_capability_gate(self) -> None:
        """Wire the Swift capability probe into the one-shot pipeline.

        Opt-in via ``JUNO_ONESHOT_GATE_CAPABILITY=1``. Off by default
        because the Swift helper needs Accessibility permission; we
        don't want to fail surprised users on first run. When on, we
        hand the pipeline a tiny adapter that matches the
        ``CapabilityGate`` protocol by delegating to
        :class:`CapabilityChecker`.
        """
        import os as _os

        if _os.environ.get("JUNO_ONESHOT_GATE_CAPABILITY", "0") != "1":
            return

        capability_checker = self.capability

        # Translate ``CapabilityDecision`` (ok / warn / message / reason)
        # into the shape the :class:`OneShotDictationPipeline` expects
        # (``blocked`` / ``mode`` / ``reason``). Without this translation
        # the pipeline would read ``decision.blocked``, find it missing,
        # and treat every capability verdict as "allow" — which is the
        # exact opposite of a safety gate.
        class _GatedDecision:
            __slots__ = ("blocked", "mode", "reason", "message")

            def __init__(self, decision) -> None:
                # ``no_text_focus`` is a soft verdict: dictation may run, but the
                # Mac shell should not claim a successful paste (no AX text field).
                soft_allow = decision.reason == "no_text_focus"
                self.blocked = (not decision.ok) and not soft_allow
                if self.blocked:
                    self.mode = "block"
                elif soft_allow:
                    self.mode = "allow"
                elif decision.warn:
                    self.mode = "warn"
                else:
                    self.mode = "allow"
                self.reason = decision.reason
                self.message = decision.message

        class _GateAdapter:
            def decide(
                self, *, app_bundle_id: str | None, window_title: str | None
            ):
                decision = capability_checker.decide(
                    app_bundle_id=app_bundle_id, window_title=window_title
                )
                return _GatedDecision(decision)

        self.oneshot_pipeline.capability_gate = _GateAdapter()

    @staticmethod
    def _load_suppression_config() -> SuppressionConfig:
        """Load the suppression config from env or repo defaults.

        Never raises — broken JSON or IO errors fall back to an empty
        config so the capability gate still runs. We log at INFO when
        we couldn't load so ops can notice in the trace.
        """
        import logging
        import os as _os

        explicit = _os.environ.get("JUNO_SUPPRESSION_CONFIG", "").strip()
        candidates: list[Path] = []
        if explicit:
            candidates.append(Path(explicit).expanduser())
        # Repo-root fallback: this file lives at juno_v2/workbench/server.py
        # so the repo root is three parents up.
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "config" / "suppression_apps.json")

        logger = logging.getLogger(__name__)
        for path in candidates:
            if not path.is_file():
                continue
            try:
                return SuppressionConfig.load(path)
            except Exception as exc:
                logger.warning("suppression_config_load_failed path=%s err=%s", path, exc)
        return SuppressionConfig.default()

    def warm_status(self) -> Dict[str, Any]:
        """Snapshot of pre-warm progress for ``/healthz`` and the engine
        compatibility endpoint. Stable shape: ``ready: bool``,
        ``state: "ready"|"warming"|"error"``, optional ``error: str``."""
        out: Dict[str, Any] = {
            "ready": self.warm_state == "ready",
            "state": self.warm_state,
        }
        if self.warm_state == "error" and self.warm_error:
            out["error"] = self.warm_error
        return out

    def set_warm_state(self, state: str, *, error: str | None = None) -> None:
        """Update warm state from outside the WorkbenchApp.

        Used by ``ProductionServiceRunner`` so it can publish ``state =
        "warming"`` to ``/healthz`` and engine-compat consumers
        immediately after the HTTP server starts, then flip to
        ``"ready"`` (or ``"error"``) when ``warm_all()`` completes.
        Without this hook the runtime had to warm models before
        starting HTTP — meaning ``/healthz`` was unreachable for the
        entire 60–180 s cold-cache download window and the macOS
        shell's ``engineWarmingCard`` never had a chance to render.
        """
        if state not in ("ready", "warming", "error"):
            raise ValueError(f"invalid warm state: {state!r}")
        self.warm_state = state
        self.warm_error = error if state == "error" else None

    def set_transcriber(self, transcriber: DictationTranscriber) -> None:
        """Atomically swap the active dictation transcriber.

        Used by the runtime service to inject the warmed final-ASR
        backend after ``warm_all()`` completes — workbench startup can
        race ahead and serve ``/healthz`` with ``state="warming"``
        while the heavy model load happens behind it. Both the
        instance attribute and the OneShotDictationPipeline's reference
        get updated under ``self._lock`` so an in-flight ingest_wav
        request observes a consistent transcriber.
        """
        with self._lock:
            self.transcriber = transcriber
            if hasattr(self, "oneshot_pipeline") and self.oneshot_pipeline is not None:
                self.oneshot_pipeline.transcriber = transcriber

    def _maybe_start_prewarm(self) -> None:
        """Schedule a background snapshot_download for the configured HF
        repo id when the synchronous resolver couldn't satisfy it from the
        local cache. Called once during ``__init__``."""
        if self._transcriber_was_injected:
            return
        repo_id = self._discover_warm_target_repo_id()
        if not repo_id:
            return
        self.warm_state = "warming"
        self._warm_thread = threading.Thread(
            target=self._run_prewarm,
            args=(repo_id,),
            name="juno-workbench-prewarm",
            daemon=True,
        )
        self._warm_thread.start()

    def _discover_warm_target_repo_id(self) -> str | None:
        """Inspect the env vars resolve_transcriber_from_env reads and
        return an HF repo id when one is configured but not yet cached.
        Returns ``None`` if the path is local, the cache is hot, or no
        backend is configured."""
        from juno_core_v3.dictation.transcriber import is_hf_repo_id

        for name in (
            "JUNO_FINAL_MODEL_PATH",
            "JUNO_V2_FINAL_MODEL_PATH",
        ):
            raw = os.environ.get(name, "").strip()
            if not raw:
                continue
            if not is_hf_repo_id(raw):
                return None
            try:
                from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

                snapshot_download(raw, local_files_only=True)
                return None  # cache hit — no warmup needed
            except Exception:
                return raw
        return None

    def _run_prewarm(self, repo_id: str) -> None:
        from juno_v2.runtime.offline_mode import hub_online_for_explicit_download
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

            # Prewarm only fires when the synchronous cache-resolver
            # said this repo isn't fully present. If HF_HUB_OFFLINE was
            # auto-set by the boot-time check (it shouldn't be, given
            # the cache was incomplete) we clear it here so the prewarm
            # is allowed to actually download.
            with hub_online_for_explicit_download():
                snapshot_download(repo_id)
            self.warm_state = "ready"
            self.warm_error = None
        except Exception as exc:  # pragma: no cover — network / permission paths
            self.warm_state = "error"
            self.warm_error = f"{type(exc).__name__}: {exc}"

    def state(self) -> Dict[str, Any]:
        with self._lock:
            snap: Dict[str, Any] = dict(self.store.snapshot())
        bs = self.broker.broker_session
        snap["broker_observability"] = {
            "session_active": bs is not None,
            "session_kind": bs.kind.value if bs else None,
            "policy": dict(bs.metadata.get("policy", {})) if bs else {},
        }
        snap["trace_hint"] = {
            "workbench_session_id": self.session_id,
            "trace_jsonl": str(self.recorder.log_path.resolve()),
            "log_dir": str(Path(self.config.log_dir).resolve()),
        }
        return snap

    def runtime_snapshot(self) -> Dict[str, Any]:
        """Aggregate health/summary/startup/doctor/logs when runtime_dir and logs_dir are set."""
        out: Dict[str, Any] = {}
        cfg = self.config
        if cfg.runtime_dir is not None:
            rd = Path(cfg.runtime_dir)
            for name, key in (
                ("health.json", "health"),
                ("summary.json", "summary"),
                ("startup_profile.json", "startup_profile"),
                ("doctor_report.json", "doctor_report"),
                ("demo_meta.json", "demo_meta"),
            ):
                p = rd / name
                if p.exists():
                    try:
                        out[key] = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        out[key] = {}
            if out.get('summary') and 'metadata' in out['summary']:
                out['runtime_truth'] = out['summary'].get('metadata', {}).get('runtime_truth', {})
                out['memory_packet_summaries'] = out['summary'].get('metadata', {}).get('memory_packet_summaries', [])
                out['utterance_records'] = out['summary'].get('metadata', {}).get('utterance_records', [])
        if cfg.logs_dir is not None:
            ld = Path(cfg.logs_dir)
            preview_log = ld / "preview_service.log"
            runtime_log = ld / "runtime_service.log"
            preview_tail = ""
            runtime_tail = ""
            if preview_log.exists():
                text = preview_log.read_text(encoding="utf-8")
                preview_tail = "\n".join(text.splitlines()[-8:])
            if runtime_log.exists():
                text = runtime_log.read_text(encoding="utf-8")
                runtime_tail = "\n".join(text.splitlines()[-8:])
            if cfg.runtime_dir is not None:
                meta = out.get("demo_meta") or {}
                if isinstance(meta, dict):
                    if meta.get("preview_log"):
                        pl = Path(str(meta["preview_log"]))
                        if pl.exists():
                            preview_tail = "\n".join(pl.read_text(encoding="utf-8").splitlines()[-8:])
                    if meta.get("runtime_log"):
                        rl = Path(str(meta["runtime_log"]))
                        if rl.exists():
                            runtime_tail = "\n".join(rl.read_text(encoding="utf-8").splitlines()[-8:])
            trace_tail = ''
            if self.recorder.log_path.exists():
                trace_tail = '\n'.join(self.recorder.log_path.read_text(encoding='utf-8').splitlines()[-32:])
            out['logs'] = {
                'preview_service': {'tail': preview_tail, 'path': str(preview_log) if preview_log.exists() else ''},
                'runtime_service': {'tail': runtime_tail, 'path': str(runtime_log) if runtime_log.exists() else ''},
                'workbench_trace': {'tail': trace_tail, 'path': str(self.recorder.log_path)},
            }
        elif self.recorder.log_path.exists():
            trace_tail = "\n".join(self.recorder.log_path.read_text(encoding="utf-8").splitlines()[-32:])
            out["logs"] = {
                "preview_service": {"tail": "", "path": ""},
                "runtime_service": {"tail": "", "path": ""},
                "workbench_trace": {"tail": trace_tail, "path": str(self.recorder.log_path)},
            }
        jsonl_index: list[dict[str, Any]] = []
        try:
            logd = self.recorder.log_path.parent
            if logd.is_dir():
                for p in sorted(logd.glob('*.jsonl'), key=lambda x: x.stat().st_mtime, reverse=True)[:16]:
                    try:
                        st = p.stat()
                        jsonl_index.append(
                            {
                                'file': p.name,
                                'path': str(p.resolve()),
                                'bytes': st.st_size,
                                'mtime_unix': int(st.st_mtime),
                            }
                        )
                    except OSError:
                        continue
        except OSError:
            pass
        out['workbench_trace_info'] = {
            'workbench_session_id': self.session_id,
            'primary_jsonl': str(self.recorder.log_path.resolve()),
            'log_dir': str(Path(self.config.log_dir).resolve()),
            'jsonl_files_newest_first': jsonl_index,
        }
        if cfg.runtime_dir is None and cfg.logs_dir is None:
            out['workbench_standalone'] = True
        return out

    # ------------------------------------------------------------------
    # Setup / model readiness
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_hf_repo_id_from_hub_cache_dir(path: str) -> str:
        """Best-effort: map a HF hub cache snapshot dir to a `org/model` repo id.

        Example::
            ~/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/<hash>
            → Systran/faster-whisper-base
        """

        p = (path or "").strip()
        if not p:
            return ""
        try:
            parts = Path(p).parts
        except Exception:
            return ""
        for part in parts:
            if part.startswith("models--") and "--" in part:
                inner = part[len("models--") :]
                # HF hub uses `--` as namespace separators in folder names.
                return inner.replace("--", "/")
        return ""

    @staticmethod
    def _setup_model_ui_fields_from_production_service_config(cfg: Any) -> Dict[str, str]:
        """Model labels for macOS onboarding when the embedded workbench is wired to ProductionServiceConfig.

        The macOS onboarding renders fixed role-based titles ("Live captions",
        "High-quality transcription", "Smart formatting"). We only emit the
        repo id here as a tiny secondary line — backend-name guesses
        ("Faster Whisper preview") would otherwise leak implementation
        details into the user-facing title.
        """

        preview_backend = str(getattr(cfg, "preview_backend", "") or "").strip().lower()
        preview_path = str(getattr(cfg, "preview_model_path", "") or "").strip()
        final_path = str(getattr(cfg, "final_model_path", "") or "").strip()
        final_hf_repo = str(getattr(cfg, "final_hf_repo_id", "") or "").strip()
        writer_path = str(getattr(cfg, "writer_model_path", "") or "").strip()
        live_corrector_path = str(getattr(cfg, "live_corrector_model_path", "") or "").strip()

        preview_repo = WorkbenchApp._infer_hf_repo_id_from_hub_cache_dir(preview_path) or preview_path

        final_repo = final_hf_repo or final_path
        writer_repo = writer_path
        live_corrector_repo = live_corrector_path

        # Titles intentionally left empty — Swift renders fixed role-based
        # labels. Repo id appears as a tiny subtitle when present.
        return {
            "preview_repo_id": preview_repo,
            "final_repo_id": final_repo,
            "writer_repo_id": writer_repo,
            "live_corrector_repo_id": live_corrector_repo,
            "preview_model_title": "",
            "final_model_title": "",
            "writer_model_title": "",
            "live_corrector_model_title": "",
        }

    @staticmethod
    def _setup_model_ui_fields_from_config(config: Any) -> Dict[str, str]:
        """Repo ids for macOS onboarding. Titles stay empty: Swift renders fixed
        role-based labels and uses the repo id only as a small subtitle."""

        preview_rid = str(getattr(config, "preview_repo_id", "") or "")
        final_rid = str(getattr(config, "final_repo_id", "") or "")
        writer_rid = str(getattr(config, "writer_model_path", "") or "")
        live_corrector_rid = str(getattr(config, "live_corrector_model_path", "") or "")
        return {
            "preview_repo_id": preview_rid,
            "final_repo_id": final_rid,
            "writer_repo_id": writer_rid,
            "live_corrector_repo_id": live_corrector_rid,
            "preview_model_title": "",
            "final_model_title": "",
            "writer_model_title": "",
            "live_corrector_model_title": "",
        }

    @staticmethod
    def _is_cached_hf_repo(repo_id: str) -> bool:
        """Best-effort local HF cache probe without any network access.

        Probes snapshot *completeness* (config + weights), not just
        config.json presence: setup status uses this to tell onboarding
        a lane is ready, and a config.json left behind by an interrupted
        download must read as "needs install", not "ready".
        """
        raw = str(repo_id or "").strip()
        if not raw:
            return False
        try:
            from juno_core_v3.dictation.transcriber import is_hf_repo_id

            if not is_hf_repo_id(raw):
                return False
            from juno_v2.demo.models import is_hf_model_cache_complete

            return is_hf_model_cache_complete(raw)
        except Exception:
            return False

    @classmethod
    def _production_backend_ready(
        cls,
        *,
        backend: str,
        model_path: str,
        endpoint: str | None = None,
    ) -> bool:
        """Readiness probe for production-service lane config when runtime
        health.json is missing or not yet populated during relaunch."""
        backend_norm = str(backend or "").strip().lower()
        raw_path = str(model_path or "").strip()

        if backend_norm in {"local_http_json", "streaming_local_http_json"}:
            if not endpoint:
                return False
            # The local service loads repo-id model paths from the shared
            # HF cache: a configured endpoint alone can't serve until that
            # snapshot is complete. Reporting "ready" here on a fresh
            # install would stop onboarding from ever starting the
            # download.
            if raw_path and not Path(raw_path).exists():
                try:
                    from juno_core_v3.dictation.transcriber import is_hf_repo_id

                    if is_hf_repo_id(raw_path):
                        return cls._is_cached_hf_repo(raw_path)
                except Exception:
                    return True
            return True
        if backend_norm == "mlx_whisper":
            return cls._is_cached_hf_repo(raw_path)
        if raw_path and Path(raw_path).exists():
            try:
                from juno_v2.demo.models import is_model_ready

                return bool(is_model_ready(Path(raw_path)))
            except Exception:
                return False
        return cls._is_cached_hf_repo(raw_path)

    @classmethod
    def _production_config_setup_status(cls, svc_cfg: Any, tracked: str | None) -> Dict[str, Any]:
        """Fallback setup snapshot derived from ProductionServiceConfig itself.

        This avoids bouncing to the source profile when the embedded
        runtime has started serving HTTP but health.json is still absent,
        stale, or only partially written.
        """
        preview_ready = cls._production_backend_ready(
            backend=getattr(svc_cfg, "preview_backend", ""),
            model_path=str(getattr(svc_cfg, "preview_model_path", "") or ""),
            endpoint=getattr(svc_cfg, "preview_endpoint", None),
        )
        final_ready = cls._production_backend_ready(
            backend=getattr(svc_cfg, "final_backend", ""),
            model_path=str(getattr(svc_cfg, "final_model_path", "") or ""),
            endpoint=getattr(svc_cfg, "final_endpoint", None),
        )
        writer_backend = str(getattr(svc_cfg, "writer_backend", "") or "none")
        writer_model_path = str(getattr(svc_cfg, "writer_model_path", "") or "")
        live_corrector_enabled = bool(getattr(svc_cfg, "live_corrector_enabled", False))
        live_corrector_backend = str(getattr(svc_cfg, "live_corrector_backend", "") or "none")
        live_corrector_model_path = str(getattr(svc_cfg, "live_corrector_model_path", "") or "")
        writer_backend_norm = writer_backend.strip().lower()
        writer_required = writer_backend_norm not in {"", "none"}
        writer_cached = True
        if writer_required and writer_model_path:
            try:
                from juno_v2.demo.models import is_writer_model_cached

                writer_cached = bool(is_writer_model_cached(writer_model_path))
            except Exception:
                writer_cached = cls._is_cached_hf_repo(writer_model_path)
        elif writer_required:
            writer_cached = False
        writer_ready = writer_cached if writer_required else True
        live_corrector_backend_norm = live_corrector_backend.strip().lower()
        live_corrector_required = live_corrector_enabled and live_corrector_backend_norm not in {"", "none"}
        live_corrector_cached = True
        if live_corrector_required and live_corrector_model_path:
            try:
                from juno_v2.demo.models import is_hf_model_cached

                live_corrector_cached = bool(is_hf_model_cached(live_corrector_model_path))
            except Exception:
                live_corrector_cached = cls._is_cached_hf_repo(live_corrector_model_path)
        elif live_corrector_required:
            live_corrector_cached = False
        live_corrector_ready = live_corrector_cached if live_corrector_required else True
        overall_ready = bool(preview_ready and final_ready and writer_ready and live_corrector_ready)
        install_state = tracked if tracked and tracked != "ready" else ("ready" if overall_ready else "needs_setup")

        out: Dict[str, Any] = {
            "ok": True,
            "source": "production_config_quick_check",
            "overall_ready": overall_ready,
            "install_state": install_state,
            "broker_reachable": True,
            "broker_version": "1",
            "runtime_present": True,
            "preview_model_ready": preview_ready,
            "final_model_ready": final_ready,
            "writer_model_ready": writer_ready,
            "live_corrector_model_ready": live_corrector_ready,
            "live_corrector_required": live_corrector_required,
            "live_corrector_model_cached": live_corrector_cached,
            "live_corrector_runtime_warm": False,
            "live_corrector_runtime_loaded": False,
            "writer_required": writer_required,
            "writer_model_cached": writer_cached,
            "writer_runtime_warm": False,
            "writer_runtime_loaded": False,
            "final_backend": getattr(svc_cfg, "final_backend", ""),
            "writer_backend": writer_backend,
            "writer_model_path": writer_model_path,
            "live_corrector_backend": live_corrector_backend,
            "live_corrector_model_path": live_corrector_model_path,
            "checks": [
                {
                    "name": "preview_model",
                    "ok": preview_ready,
                    "detail": f"backend={getattr(svc_cfg, 'preview_backend', '')}, path={getattr(svc_cfg, 'preview_model_path', '')}",
                },
                {
                    "name": "final_model",
                    "ok": final_ready,
                    "detail": f"backend={getattr(svc_cfg, 'final_backend', '')}, path={getattr(svc_cfg, 'final_model_path', '')}",
                },
                {
                    "name": "writer_model",
                    "ok": writer_ready,
                    "detail": f"backend={writer_backend or 'none'}, path={writer_model_path or ''}",
                    "metadata": {
                        "writer_required": writer_required,
                        "writer_model_cached": writer_cached,
                        "writer_runtime_warm": False,
                        "writer_runtime_loaded": False,
                    },
                },
                {
                    "name": "live_corrector_model",
                    "ok": live_corrector_ready,
                    "detail": f"backend={live_corrector_backend or 'none'}, path={live_corrector_model_path or ''}",
                    "metadata": {
                        "live_corrector_required": live_corrector_required,
                        "live_corrector_model_cached": live_corrector_cached,
                        "live_corrector_runtime_warm": False,
                        "live_corrector_runtime_loaded": False,
                    },
                },
            ],
        }
        out.update(cls._setup_model_ui_fields_from_production_service_config(svc_cfg))
        return out

    def broker_setup_status(self) -> Dict[str, Any]:
        """Return current setup/readiness state using quick file checks.

        Returns truthful state without running heavy doctor subprocess checks.
        Uses a cached doctor_report.json when available (written by the full
        demo launcher), otherwise does quick model-path presence checks.
        """
        import os

        # If the embedded workbench is running alongside the production service,
        # prefer the *actual* ProductionServiceConfig + runtime health snapshot over
        # the standalone profile defaults (which are only for repo bootstrap).
        if self.config.runtime_dir is not None:
            try:
                from juno_v2.runtime.deployment import ProductionServiceConfig

                svc_cfg = ProductionServiceConfig.from_env()
                # The embedded workbench is constructed with the runtime
                # directory owned by the active ProductionService instance.
                # ``ProductionServiceConfig.from_env()`` can point at a stale
                # shell/runtime path in tests or local dev, so use the attached
                # workbench runtime for health while keeping svc_cfg lane/model
                # metadata.
                runtime_dir = Path(self.config.runtime_dir)
                svc_cfg.paths.runtime_dir = runtime_dir
                svc_cfg.paths.health_json = runtime_dir / "health.json"
                try:
                    svc_cfg.validate()
                    production_cfg_valid = True
                except Exception:
                    production_cfg_valid = False
                with self._setup_install_lock:
                    tracked = self._setup_install_state
                if production_cfg_valid:
                    health_path = svc_cfg.paths.resolved_health_json()
                    if health_path.exists():
                        try:
                            health = json.loads(health_path.read_text(encoding="utf-8"))
                            if isinstance(health, dict) and health.get("status") == "error":
                                raise RuntimeError("runtime health snapshot is an error state")
                            lifecycle = health.get("lifecycle") if isinstance(health, dict) else None
                            comps = (lifecycle or {}).get("components") if isinstance(lifecycle, dict) else None
                            if not isinstance(comps, list) or not comps:
                                raise RuntimeError("runtime health snapshot has no model components")
                            preview_warmed = False
                            preview_loaded = False
                            final_warmed = False
                            final_loaded = False
                            writer_warmed = False
                            writer_loaded = False
                            live_corrector_warmed = False
                            live_corrector_loaded = False
                            preview_name = ""
                            final_name = ""
                            writer_ready = False
                            writer_backend = str(getattr(svc_cfg, "writer_backend", "") or "none")
                            writer_model_path = str(getattr(svc_cfg, "writer_model_path", "") or "")
                            live_corrector_enabled = bool(getattr(svc_cfg, "live_corrector_enabled", False))
                            live_corrector_backend = str(getattr(svc_cfg, "live_corrector_backend", "") or "none")
                            live_corrector_model_path = str(getattr(svc_cfg, "live_corrector_model_path", "") or "")

                            if isinstance(comps, list):
                                for c in comps:
                                    if not isinstance(c, dict):
                                        continue
                                    role = str(c.get("role") or "")
                                    if role == "preview_asr":
                                        preview_name = str(c.get("name") or "")
                                        preview_warmed = c.get("warmed") is True
                                        preview_loaded = c.get("loaded") is True
                                    if role == "final_asr":
                                        final_name = str(c.get("name") or "")
                                        final_warmed = c.get("warmed") is True
                                        final_loaded = c.get("loaded") is True
                                    if role == "writer":
                                        writer_warmed = c.get("warmed") is True
                                        writer_loaded = c.get("loaded") is True
                                        writer_ready = bool(writer_warmed and writer_loaded)
                                    if role == "live_corrector":
                                        live_corrector_warmed = c.get("warmed") is True
                                        live_corrector_loaded = c.get("loaded") is True

                            # health.json is a startup snapshot; on_demand components
                            # like the writer always read False there because warm_all
                            # skips them (lifecycle.py:156-158). Worse, the snapshot
                            # is never re-emitted during runtime, so the manual warm
                            # path (broker_writer_warm → svc.warm()) cannot flip the
                            # snapshot to True even after the backend is fully loaded.
                            # Trust the live ``backend._loaded`` attribute instead —
                            # it's the authoritative source on whether
                            # ``mlx_lm.load(...)`` has populated the model + tokenizer.
                            live_writer_backend = getattr(getattr(self, "writer_service", None), "backend", None)
                            if live_writer_backend is not None and hasattr(live_writer_backend, "_loaded"):
                                live_loaded = bool(getattr(live_writer_backend, "_loaded"))
                                writer_loaded = writer_loaded or live_loaded
                                writer_warmed = writer_warmed or live_loaded
                            live_corrector_backend_obj = getattr(getattr(self, "live_corrector_service", None), "backend", None)
                            if live_corrector_backend_obj is not None and hasattr(live_corrector_backend_obj, "_loaded"):
                                live_loaded = bool(getattr(live_corrector_backend_obj, "_loaded"))
                                live_corrector_loaded = live_corrector_loaded or live_loaded
                                live_corrector_warmed = live_corrector_warmed or live_loaded

                            # setup/status is an install-readiness surface, not a
                            # residency report. Streaming preview can be available
                            # in the HF cache while not yet warmed/loaded in the
                            # runtime snapshot, so use the production config cache
                            # probe as the fallback readiness source.
                            preview_config_ready = self.__class__._production_backend_ready(
                                backend=svc_cfg.preview_backend,
                                model_path=str(svc_cfg.preview_model_path or ""),
                                endpoint=getattr(svc_cfg, "preview_endpoint", None),
                            )
                            final_config_ready = self.__class__._production_backend_ready(
                                backend=svc_cfg.final_backend,
                                model_path=str(svc_cfg.final_model_path or ""),
                                endpoint=getattr(svc_cfg, "final_endpoint", None),
                            )
                            preview_ready = bool(preview_name) and bool(
                                preview_warmed or preview_loaded or preview_config_ready
                            )
                            final_ready = bool(final_name) and bool(
                                final_warmed or final_loaded or final_config_ready
                            )

                            writer_backend_norm = (writer_backend or "").strip().lower()
                            writer_required = writer_backend_norm not in {"", "none"}
                            writer_cached = True
                            if writer_required and writer_model_path:
                                try:
                                    from juno_v2.demo.models import is_writer_model_cached

                                    writer_cached = bool(is_writer_model_cached(writer_model_path))
                                except Exception:
                                    # Keep setup/status resilient when cache probes fail.
                                    writer_cached = bool(writer_warmed or writer_loaded)
                            elif writer_required:
                                writer_cached = False

                            # Setup-readiness is cache-based for on-demand writer residency.
                            writer_ready = writer_cached if writer_required else True
                            live_corrector_backend_norm = (live_corrector_backend or "").strip().lower()
                            live_corrector_required = live_corrector_enabled and live_corrector_backend_norm not in {"", "none"}
                            live_corrector_cached = True
                            if live_corrector_required and live_corrector_model_path:
                                try:
                                    from juno_v2.demo.models import is_hf_model_cached

                                    live_corrector_cached = bool(is_hf_model_cached(live_corrector_model_path))
                                except Exception:
                                    live_corrector_cached = bool(live_corrector_warmed or live_corrector_loaded)
                            elif live_corrector_required:
                                live_corrector_cached = False

                            live_corrector_ready = live_corrector_cached if live_corrector_required else True

                            overall_ready = bool(preview_ready and final_ready and writer_ready and live_corrector_ready)

                            if tracked and tracked != "ready":
                                install_state = tracked
                            else:
                                install_state = "ready" if overall_ready else "needs_setup"

                            out_rt: Dict[str, Any] = {
                                "ok": True,
                                "source": "runtime_health",
                                "overall_ready": overall_ready,
                                "install_state": install_state,
                                "broker_reachable": True,
                                "broker_version": "1",
                                "runtime_present": True,
                                "preview_model_ready": preview_ready,
                                "final_model_ready": final_ready,
                                "writer_model_ready": writer_ready,
                                "live_corrector_model_ready": live_corrector_ready,
                                "live_corrector_required": live_corrector_required,
                                "live_corrector_model_cached": live_corrector_cached,
                                "live_corrector_runtime_warm": live_corrector_warmed,
                                "live_corrector_runtime_loaded": live_corrector_loaded,
                                "writer_required": writer_required,
                                "writer_model_cached": writer_cached,
                                "writer_runtime_warm": writer_warmed,
                                "writer_runtime_loaded": writer_loaded,
                                "final_backend": svc_cfg.final_backend,
                                "writer_backend": writer_backend,
                                "writer_model_path": writer_model_path,
                                "live_corrector_backend": live_corrector_backend,
                                "live_corrector_model_path": live_corrector_model_path,
                                "checks": [
                                    {
                                        "name": "preview_model",
                                        "ok": preview_ready,
                                        "detail": f"backend={svc_cfg.preview_backend}, path={svc_cfg.preview_model_path}",
                                        "metadata": {"engine_component": "preview_asr", "component_name": preview_name},
                                    },
                                    {
                                        "name": "final_model",
                                        "ok": final_ready,
                                        "detail": f"backend={svc_cfg.final_backend}, path={svc_cfg.final_model_path}",
                                        "metadata": {"engine_component": "final_asr", "component_name": final_name},
                                    },
                                    {
                                        "name": "writer_model",
                                        "ok": writer_ready,
                                        "detail": f"backend={writer_backend or 'none'}, path={writer_model_path or ''}",
                                        "metadata": {
                                            "writer_required": writer_required,
                                            "writer_model_cached": writer_cached,
                                            "writer_runtime_warm": writer_warmed,
                                            "writer_runtime_loaded": writer_loaded,
                                        },
                                    },
                                    {
                                        "name": "live_corrector_model",
                                        "ok": live_corrector_ready,
                                        "detail": f"backend={live_corrector_backend or 'none'}, path={live_corrector_model_path or ''}",
                                        "metadata": {
                                            "engine_component": "live_corrector",
                                            "live_corrector_required": live_corrector_required,
                                            "live_corrector_model_cached": live_corrector_cached,
                                            "live_corrector_runtime_warm": live_corrector_warmed,
                                            "live_corrector_runtime_loaded": live_corrector_loaded,
                                        },
                                    },
                                ],
                            }
                            out_rt.update(self.__class__._setup_model_ui_fields_from_production_service_config(svc_cfg))
                            # Attach the live download-progress snapshot when a
                            # provisioning install is running. Client renders
                            # bytes/total/speed/ETA for a real progress bar.
                            with self._setup_install_lock:
                                if self._setup_install_progress is not None:
                                    out_rt["download_progress"] = dict(self._setup_install_progress)
                            return out_rt
                        except Exception:
                            pass
                    fallback_out = self.__class__._production_config_setup_status(svc_cfg, tracked)
                    with self._setup_install_lock:
                        if self._setup_install_progress is not None:
                            fallback_out["download_progress"] = dict(self._setup_install_progress)
                    return fallback_out
            except Exception:
                # Fall through to cached doctor / quick demo checks.
                pass

        # If a full runtime is attached, use its cached doctor report.
        if self.config.runtime_dir is not None:
            doctor_file = Path(self.config.runtime_dir) / "doctor_report.json"
            if doctor_file.exists():
                try:
                    cached = json.loads(doctor_file.read_text(encoding="utf-8"))
                    checks = cached.get("results", [])
                    overall_ready = all(c.get("ok", False) for c in checks)
                    install_state = self._setup_install_state or ("ready" if overall_ready else "needs_setup")
                    out_cached: Dict[str, Any] = {
                        "ok": True,
                        "source": "cached_doctor_report",
                        "overall_ready": overall_ready,
                        "install_state": install_state,
                        "broker_reachable": True,
                        "broker_version": "1",
                        "checks": checks,
                    }
                    try:
                        from juno_v2.demo.config import DEFAULT_DEMO_PROFILE, DemoPaths, load_demo_config
                        from juno_v2.runtime.paths import juno_profile_root

                        _cfg = load_demo_config(
                            paths=DemoPaths(root_dir=juno_profile_root()),
                            profile_name=DEFAULT_DEMO_PROFILE,
                        )
                        out_cached.update(self.__class__._setup_model_ui_fields_from_config(_cfg))
                    except Exception:
                        out_cached["preview_repo_id"] = ""
                        out_cached["final_repo_id"] = ""
                        out_cached["writer_repo_id"] = ""
                        out_cached["live_corrector_repo_id"] = ""
                        out_cached["preview_model_title"] = ""
                        out_cached["final_model_title"] = ""
                        out_cached["writer_model_title"] = ""
                        out_cached["live_corrector_model_title"] = ""
                    return out_cached
                except Exception:
                    pass

        # Quick path: model file presence check without heavy doctor run.
        try:
            from juno_v2.demo.config import DEFAULT_DEMO_PROFILE, DemoPaths, load_demo_config
            from juno_v2.demo.models import is_hf_model_cached, is_model_ready, is_writer_model_cached
            from juno_v2.runtime.paths import juno_profile_root

            paths = DemoPaths(root_dir=juno_profile_root())
            config = load_demo_config(paths=paths, profile_name=DEFAULT_DEMO_PROFILE)
            preview_service_backend = str(getattr(config, "preview_service_backend", "") or "").strip().lower()
            preview_backend_value = str(getattr(config, "preview_backend", "") or "").strip().lower()
            preview_uses_mlx = (
                preview_backend_value == "streaming_local_http_json"
                and preview_service_backend in {"mlx_whisper", "mlx_whisper_streaming"}
            )
            preview_ready = (
                bool(is_hf_model_cached(str(config.preview_repo_id))) or is_model_ready(config.preview_model_path)
                if preview_uses_mlx
                else is_model_ready(config.preview_model_path)
            )
            final_uses_mlx = (config.final_backend or "").strip().lower() == "mlx_whisper"
            if final_uses_mlx:
                final_ready = bool(is_hf_model_cached(str(config.final_repo_id))) or is_model_ready(config.final_model_path)
            else:
                final_ready = is_model_ready(config.final_model_path)
            writer_backend = str(config.writer_backend or "").strip()
            writer_required = writer_backend.lower() not in {"", "none"}
            writer_ready = (
                writer_required and bool(config.writer_model_path) and is_writer_model_cached(config.writer_model_path)
            ) if writer_required else True
            live_corrector_backend = str(getattr(config, "live_corrector_backend", "") or "").strip()
            live_corrector_required = live_corrector_backend.lower() not in {"", "none"}
            live_corrector_model_path = str(getattr(config, "live_corrector_model_path", "") or "")
            if live_corrector_required and live_corrector_model_path:
                from juno_v2.demo.models import is_hf_model_cached

                live_corrector_ready = is_hf_model_cached(live_corrector_model_path)
            else:
                live_corrector_ready = not live_corrector_required
            runtime_present = preview_ready

            # Current install state: use in-progress tracking first, then
            # derive from file presence.
            with self._setup_install_lock:
                tracked = self._setup_install_state
            if tracked and tracked != "ready":
                install_state = tracked
            else:
                install_state = "ready" if (preview_ready and final_ready and writer_ready and live_corrector_ready) else "not_started"

            out_quick: Dict[str, Any] = {
                "ok": True,
                "source": "quick_check",
                "overall_ready": preview_ready and final_ready and writer_ready and live_corrector_ready,
                "install_state": install_state,
                "broker_reachable": True,
                "broker_version": "1",
                "runtime_present": runtime_present,
                "preview_model_ready": preview_ready,
                "final_model_ready": final_ready,
                "writer_model_ready": writer_ready,
                "live_corrector_model_ready": live_corrector_ready,
                "live_corrector_required": live_corrector_required,
                "live_corrector_model_cached": live_corrector_ready if live_corrector_required else True,
                "live_corrector_runtime_warm": False,
                "live_corrector_runtime_loaded": False,
                "writer_required": writer_required,
                "writer_model_cached": writer_ready if writer_required else True,
                "writer_runtime_warm": False,
                "writer_runtime_loaded": False,
                "final_backend": config.final_backend or "faster_whisper",
                "writer_backend": config.writer_backend or "none",
                "writer_model_path": config.writer_model_path or "",
                "live_corrector_backend": live_corrector_backend or "none",
                "live_corrector_model_path": live_corrector_model_path,
                "checks": [
                    {
                        "name": "preview_model",
                        "ok": preview_ready,
                        "detail": (
                            f"path={config.preview_model_path}"
                            + (" (mlx_whisper HF cache)" if preview_uses_mlx else "")
                        ),
                        "metadata": {"preview_uses_mlx": preview_uses_mlx},
                    },
                    {
                        "name": "final_model",
                        "ok": final_ready,
                        "detail": (
                            f"path={config.final_model_path}"
                            + (" (mlx_whisper HF cache)" if final_uses_mlx else "")
                        ),
                        "metadata": {"final_uses_mlx": final_uses_mlx},
                    },
                    {
                        "name": "writer_model",
                        "ok": writer_ready,
                        "detail": f"backend={config.writer_backend or 'none'}, path={config.writer_model_path or ''}",
                        "metadata": {
                            "writer_required": writer_required,
                            "writer_model_cached": writer_ready if writer_required else True,
                        },
                    },
                    {
                        "name": "live_corrector_model",
                        "ok": live_corrector_ready,
                        "detail": f"backend={live_corrector_backend or 'none'}, path={live_corrector_model_path}",
                        "metadata": {
                            "live_corrector_required": live_corrector_required,
                            "live_corrector_model_cached": live_corrector_ready if live_corrector_required else True,
                        },
                    },
                ],
            }
            out_quick.update(self.__class__._setup_model_ui_fields_from_config(config))
            return out_quick
        except Exception as exc:
            return {
                "ok": False,
                "source": "error",
                "overall_ready": False,
                "install_state": "error",
                "broker_reachable": True,
                "error": str(exc),
                "checks": [],
            }

    def broker_setup_install(self, payload: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
        """Trigger model provisioning in a background thread.

        Safe to call multiple times: a second call while downloading is a no-op.
        Pass ``force=True`` (via the repair route) to re-download even if
        models are already present.

        Spawns a sidecar poller that watches the HF cache directory for each
        target repo and computes (bytes_so_far, bytes_total, bytes_per_second,
        eta_seconds). The onboarding UI reads that snapshot via
        ``broker_setup_status`` to render a real progress bar — bytes, speed,
        ETA — instead of an indeterminate spinner.
        """
        with self._setup_install_lock:
            if self._setup_install_state == "downloading":
                return {"ok": True, "install_state": "downloading", "message": "Install already in progress"}
            self._setup_install_state = "downloading"
            # Reset progress on a new install so a stale snapshot from a
            # previous run doesn't leak into the UI.
            self._setup_install_progress = {
                "bytes_so_far": 0,
                "bytes_total": 0,
                "bytes_per_second": 0.0,
                "eta_seconds": None,
                "started_at": time.time(),
                "elapsed_seconds": 0.0,
                "repos": [],
                "current_repo": None,
                "current_lane": None,
                "repos_done": 0,
                "log": [],
            }

        stop_event = threading.Event()

        def _do_install() -> None:
            from juno_v2.runtime.offline_mode import hub_online_for_explicit_download
            try:
                from juno_v2.demo.config import DEFAULT_DEMO_PROFILE, DemoPaths, load_demo_config
                from juno_v2.demo.models import provision_demo_models
                from juno_v2.runtime.paths import juno_profile_root

                # Writable root: the packaged engine's cwd is inside the
                # sealed .app bundle — provisioning scratch must live in
                # Application Support (see juno_profile_root).
                paths = DemoPaths(root_dir=juno_profile_root())
                cfg = load_demo_config(paths=paths, profile_name=DEFAULT_DEMO_PROFILE)

                # Kick off the progress poller in parallel — it self-stops
                # when ``stop_event`` is set in the ``finally`` below.
                repos = self.__class__._setup_install_repos_from_demo_config(cfg)
                with self._setup_install_lock:
                    self._setup_install_progress["repos"] = [r for r, _ in repos]

                # Disk-space precheck. A ~4.5-5 GB model download that runs out
                # of space fails deep inside snapshot_download with an opaque
                # OSError that the onboarding UI shows as "check your
                # connection" — misleading and unactionable. Fail fast with a
                # specific reason. The real total needs a network round-trip;
                # when it's unavailable (offline) we skip the check and let the
                # download surface the genuine network error instead.
                required_bytes = 0
                try:
                    required_bytes = self.__class__._fetch_repos_total_bytes(repos)
                except Exception:
                    required_bytes = 0
                if required_bytes > 0:
                    import shutil as _shutil
                    try:
                        from huggingface_hub import constants as _hf_constants
                        cache_root = Path(
                            getattr(_hf_constants, "HF_HUB_CACHE", None)
                            or (Path.home() / ".cache" / "huggingface" / "hub")
                        )
                    except Exception:
                        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
                    # disk_usage needs an existing path; walk up to the nearest one.
                    probe = cache_root
                    while not probe.exists() and probe != probe.parent:
                        probe = probe.parent
                    try:
                        free_bytes = _shutil.disk_usage(str(probe)).free
                    except Exception:
                        free_bytes = None
                    # 1.3x headroom: HF writes the blob plus a snapshot symlink
                    # tree and needs temp space during extraction.
                    needed_bytes = int(required_bytes * 1.3)
                    if free_bytes is not None and free_bytes < needed_bytes:
                        need_gb = needed_bytes / 1e9
                        free_gb = free_bytes / 1e9
                        self._setup_install_log(
                            f"Not enough disk space: need ~{need_gb:.1f} GB free, "
                            f"have {free_gb:.1f} GB"
                        )
                        logger.error(
                            "setup_install_insufficient_disk need_bytes=%d free_bytes=%d path=%s",
                            needed_bytes, free_bytes, probe,
                        )
                        with self._setup_install_lock:
                            self._setup_install_state = "failed:insufficient_disk"
                        return

                self._setup_install_log(
                    f"Starting download of {len(repos)} model(s)"
                )
                poller = threading.Thread(
                    target=self._poll_setup_install_progress,
                    args=(repos, stop_event),
                    daemon=True,
                    name="juno-setup-install-progress",
                )
                poller.start()

                # Deliberate install/repair: the process may have set
                # HF_HUB_OFFLINE=1 at boot when the cache was already
                # complete; clear it here so this download can talk to
                # the hub. Restored on exit.
                with hub_online_for_explicit_download():
                    provision_demo_models(cfg, paths=paths, force=force)
                self._setup_install_log("All model files downloaded")
                lifecycle = getattr(self, "lifecycle_manager", None)
                warm_component = getattr(lifecycle, "warm_component", None)
                warm_errors: list[str] = []
                if callable(warm_component):
                    # Fresh install: the engine deliberately skipped warming
                    # ASR lanes whose weights weren't on disk at startup
                    # (see _initial_warm_skip_roles). Now that the download
                    # landed, warm them here — preview first so the HUD
                    # shows words on the first utterance — so the engine
                    # becomes ready without a restart.
                    self._setup_install_log("Loading models into memory")
                    for role in ("preview_asr", "final_asr", "live_corrector", "writer"):
                        try:
                            warm_component(role)
                        except Exception as exc:
                            logger.exception("setup_install_component_warm_failed role=%s", role)
                            warm_errors.append(f"{role}: {type(exc).__name__}: {exc}")
                    set_warm_state = getattr(self, "set_warm_state", None)
                    if callable(set_warm_state):
                        try:
                            if warm_errors:
                                set_warm_state("error", error="; ".join(warm_errors))
                            else:
                                set_warm_state("ready")
                        except Exception:
                            logger.exception("setup_install_set_warm_state_failed")
                self._setup_install_log("Setup complete")
                with self._setup_install_lock:
                    self._setup_install_state = "ready"
                    # One final snapshot at 100% so the UI shows a clean
                    # completion frame before transitioning out of the
                    # downloading state.
                    if self._setup_install_progress is not None:
                        total = self._setup_install_progress.get("bytes_total") or 0
                        if total > 0:
                            self._setup_install_progress["bytes_so_far"] = total
                        self._setup_install_progress["bytes_per_second"] = 0.0
                        self._setup_install_progress["eta_seconds"] = 0.0
            except Exception as exc:
                self._setup_install_log(f"Install failed: {exc}")
                with self._setup_install_lock:
                    self._setup_install_state = f"failed:{exc!s}"
            finally:
                stop_event.set()

        t = threading.Thread(target=_do_install, daemon=True, name="juno-setup-install")
        t.start()
        return {"ok": True, "install_state": "downloading", "message": "Model download started"}

    @staticmethod
    def _setup_install_repos_from_demo_config(cfg: Any) -> "list[tuple[str, str]]":
        """Return ``[(repo_id, lane_name), ...]`` for every repo a
        provision-demo-models call will try to download. Order follows the
        actual provisioning order so the UI can show "Now: <repo>" later if
        we want to surface per-lane state."""
        seen: set[str] = set()
        out: list[tuple[str, str]] = []

        def _add(repo: Any, lane: str) -> None:
            s = str(repo or "").strip()
            if not s or "/" not in s or s in seen:
                return
            seen.add(s)
            out.append((s, lane))

        _add(getattr(cfg, "preview_repo_id", None), "preview")
        _add(getattr(cfg, "final_repo_id", None), "final")
        if (getattr(cfg, "writer_backend", "") or "").strip().lower() == "mlx_lm":
            _add(getattr(cfg, "writer_model_path", None), "writer")
        if (getattr(cfg, "live_corrector_backend", "") or "").strip().lower() == "mlx_lm":
            _add(getattr(cfg, "live_corrector_model_path", None), "live_corrector")
        return out

    def _setup_install_log(self, line: str) -> None:
        """Append a human-readable line to the install status log.

        Copy-on-write (a new list each append) so ``broker_setup_status``'s
        shallow ``dict(...)`` snapshot can be serialized concurrently
        without racing an in-place append. Bounded so a pathological
        install can't grow the payload without limit.
        """
        with self._setup_install_lock:
            if self._setup_install_progress is None:
                return
            log = list(self._setup_install_progress.get("log") or [])
            log.append({"t": time.time(), "line": str(line)})
            self._setup_install_progress["log"] = log[-30:]

    @staticmethod
    def _fetch_repos_total_bytes(repos: "list[tuple[str, str]]") -> int:
        """Network probe of the summed repo sizes; 0 when unavailable."""
        total = 0
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            api = HfApi()
            for repo_id, _lane in repos:
                try:
                    info = api.repo_info(repo_id=repo_id, files_metadata=True)
                    siblings = getattr(info, "siblings", None) or []
                    total += sum(int(getattr(s, "size", 0) or 0) for s in siblings)
                except Exception:
                    # Offline / private / wrong-API path: report "no total
                    # known" so the caller keeps retrying.
                    return 0
        except Exception:
            return 0
        return total

    def _poll_setup_install_progress(
        self,
        repos: "list[tuple[str, str]]",
        stop_event: threading.Event,
    ) -> None:
        """Background thread: every 2s sample the HF cache size for each
        target repo, then compute (bytes_so_far, bytes_total, bytes_per_second,
        eta_seconds) plus the lane currently downloading, and publish the
        dict so ``broker_setup_status`` can read it. Self-terminates when
        ``stop_event`` is set.

        The total-size probe needs the network; a transient failure at
        install start must not freeze the UI at "indeterminate" for the
        whole download, so it is retried inside the loop until it
        succeeds.
        """
        bytes_total = 0
        next_total_attempt = 0.0
        current_repo: str | None = None

        history: list[tuple[float, int]] = []
        while not stop_event.is_set():
            try:
                now = time.time()
                if bytes_total <= 0 and now >= next_total_attempt:
                    bytes_total = self._fetch_repos_total_bytes(repos)
                    next_total_attempt = now + 15.0
                    if bytes_total > 0:
                        gb = bytes_total / 1e9
                        self._setup_install_log(f"Download size: {gb:.1f} GB total")

                # First repo (in provisioning order) whose snapshot is not
                # complete is the one provision_demo_models is working on.
                active_repo: str | None = None
                active_lane: str | None = None
                repos_done = 0
                for repo_id, lane in repos:
                    if self.__class__._is_cached_hf_repo(repo_id):
                        repos_done += 1
                        continue
                    if active_repo is None:
                        active_repo, active_lane = repo_id, lane
                if active_repo is not None and active_repo != current_repo:
                    current_repo = active_repo
                    self._setup_install_log(
                        f"Downloading {active_repo} ({repos_done + 1} of {len(repos)})"
                    )

                bytes_so_far = self.__class__._hf_cache_bytes_for_repos([r for r, _ in repos])
                history.append((now, bytes_so_far))
                # Keep only the last 10s of samples for a stable rolling rate.
                history = [(t, b) for (t, b) in history if now - t <= 10.0]

                speed = 0.0
                if len(history) >= 2:
                    dt = history[-1][0] - history[0][0]
                    db = max(history[-1][1] - history[0][1], 0)
                    if dt > 0.25:
                        speed = db / dt

                eta: float | None = None
                if bytes_total > 0 and speed > 0:
                    remaining = max(bytes_total - bytes_so_far, 0)
                    eta = remaining / speed

                with self._setup_install_lock:
                    if self._setup_install_progress is None:
                        # broker_setup_install resets this when a new install
                        # starts; bail out cleanly if it's been cleared.
                        return
                    started_at = self._setup_install_progress.get("started_at") or now
                    self._setup_install_progress["bytes_total"] = bytes_total
                    self._setup_install_progress["bytes_so_far"] = bytes_so_far
                    self._setup_install_progress["bytes_per_second"] = speed
                    self._setup_install_progress["eta_seconds"] = eta
                    self._setup_install_progress["elapsed_seconds"] = max(now - started_at, 0)
                    self._setup_install_progress["current_repo"] = active_repo
                    self._setup_install_progress["current_lane"] = active_lane
                    self._setup_install_progress["repos_done"] = repos_done
            except Exception:
                # Never let the poller crash the install thread.
                pass
            stop_event.wait(2.0)

    @staticmethod
    def _hf_cache_bytes_for_repos(repo_ids: "list[str]") -> int:
        """Sum the on-disk size of the HF cache directories belonging to the
        given repo ids. Robust to repos that haven't been touched yet.
        """
        from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore[import-not-found]

        cache_root = Path(HF_HUB_CACHE)
        if not cache_root.exists():
            return 0

        total = 0
        for repo_id in repo_ids:
            safe = "models--" + repo_id.replace("/", "--")
            repo_dir = cache_root / safe
            if not repo_dir.exists():
                continue
            # ``blobs/`` is where snapshot_download writes the actual file
            # bytes; ``snapshots/`` only contains symlinks pointing into
            # blobs. Walking the full repo dir would double-count. Sample
            # blobs only.
            blobs_dir = repo_dir / "blobs"
            if not blobs_dir.exists():
                continue
            try:
                for entry in blobs_dir.iterdir():
                    try:
                        st = entry.stat()
                    except (FileNotFoundError, OSError):
                        continue
                    total += int(getattr(st, "st_size", 0) or 0)
            except OSError:
                continue
        return total

    def sync_client_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = SyncClientStateRequest(**payload)
        with self._lock:
            return self.commit.sync_client_state(req)

    def apply_partial(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        utterance_id = payload.get("utterance_id") or "manual_preview"
        text = payload.get("text", "")
        with self._lock:
            from juno_v2.contracts.preview import PreviewEmission

            self.commit.apply_preview(
                PreviewEmission(
                    utterance_id=utterance_id,
                    text=text,
                    start_ms=0.0,
                    end_ms=0.0,
                    is_final=False,
                    backend_name="manual",
                )
            )
            return self.store.snapshot()

    def clear_partial(self) -> Dict[str, Any]:
        with self._lock:
            self.store.clear_partial()
            return self.store.snapshot()

    def set_final_candidate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        utterance_id = payload.get("utterance_id") or "manual_final"
        text = payload.get("text", "")
        with self._lock:
            from juno_v2.contracts.final import FinalTranscript

            self.commit.stage_final(
                FinalTranscript(
                    utterance_id=utterance_id,
                    text=text,
                    start_ms=0.0,
                    end_ms=0.0,
                    backend_name="manual",
                )
            )
            return self.store.snapshot()

    def clear_final_candidate(self) -> Dict[str, Any]:
        with self._lock:
            self.store.clear_final_candidate()
            self.store.state.pending_commit = False
            return self.store.snapshot()

    def commit_final(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.commit.active is not None:
                if payload.get("text"):
                    self.commit.active.final_text = str(payload["text"])
                    self.store.set_final_candidate(FinalCandidateRequest(text=self.commit.active.final_text))
                if payload.get("commit_mode"):
                    self.commit.active.anchor.commit_mode = CommitMode(payload["commit_mode"])
                    self.store.state.active_commit_mode = self.commit.active.anchor.commit_mode.value
                self.commit.commit_active()
                return self.store.snapshot()
            if "commit_mode" in payload and not isinstance(payload["commit_mode"], CommitMode):
                payload["commit_mode"] = CommitMode(payload["commit_mode"])
            req = FinalCommitRequest(**payload)
            return self.store.commit_final(req)

    def begin_utterance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        utterance_id = str(payload.get("utterance_id") or "manual_utterance")
        commit_mode = payload.get("commit_mode")
        mode = CommitMode(commit_mode) if commit_mode else None
        with self._lock:
            self.commit.begin_utterance(utterance_id, commit_mode=mode)
            return self.store.snapshot()

    def abort_utterance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        reason = str(payload.get("reason") or "manual_abort")
        with self._lock:
            self.commit.abort_active(reason)
            return self.store.snapshot()

    def reset(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        req = ResetRequest(**(payload or {}))
        with self._lock:
            self.commit.reset_all()
            return self.store.reset(keep_buffer=req.keep_buffer)

    # --- Juno Core v3 broker (thin surface; same session id as workbench trace) ---

    def broker_start_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_hints = payload.get("host_hints")
        hints = HostResourceHints.from_dict(raw_hints) if isinstance(raw_hints, dict) else None
        surface = payload.get("surface_id")
        sig = UserIntentSignals(
            has_selected_text=bool(payload.get("has_selected_text")),
            explicit_transform=bool(payload.get("explicit_transform")),
            explicit_insert=bool(payload.get("explicit_insert")),
            surface_id=str(surface) if surface else SurfaceId.WORKBENCH_DEV.value,
            host_hints=hints,
        )
        with self._lock:
            return self.broker.start_session(sig)


    def _broker_signals_from_payload(
        self,
        payload: Dict[str, Any],
        *,
        kind: str,
        default_surface_id: str | None = None,
    ) -> UserIntentSignals | None:
        """Build broker signals for an action payload.

        We only auto-start an action session when the caller supplied a real
        surface (or the server explicitly opts into a safe default such as the
        local workbench dev surface).  This prevents silent GOLD-session
        creation that bypasses surface policy.
        """
        raw_hints = payload.get("host_hints")
        hints = HostResourceHints.from_dict(raw_hints) if isinstance(raw_hints, dict) else None
        surface = payload.get("surface_id") or default_surface_id
        if not surface and hints is None:
            return None
        return UserIntentSignals(
            has_selected_text=bool(payload.get("has_selected_text")) or bool(payload.get("selected_text")),
            explicit_transform=(kind == "transform"),
            explicit_insert=(kind == "insert"),
            surface_id=str(surface) if surface else None,
            host_hints=hints,
        )

    def broker_ingest_recovery(self) -> Dict[str, Any]:
        with self._lock:
            return self.broker.ingest_recovery(self.store)

    def broker_paste_last(self) -> Dict[str, Any]:
        with self._lock:
            return self.broker.paste_last()

    def broker_retry_append(self) -> Dict[str, Any]:
        with self._lock:
            return self.broker.retry_append(self.store)

    def broker_recovery_history(self) -> Dict[str, Any]:
        with self._lock:
            return self.broker.recovery_history()

    def broker_utterance_history(
        self,
        *,
        limit: int = 50,
        before_updated_at_ms: int | None = None,
    ) -> Dict[str, Any]:
        """Persistent per-utterance history.

        Newest first, capped at ``limit``. Pass ``before_updated_at_ms`` to
        page backwards through older rows (UI infinite scroll). Each entry
        is decorated with a ``recovery`` blob describing what the History
        detail pane is allowed to offer (Insert again, Replay, Re-run,
        deep-links to System Settings) so the shell does not need to
        re-derive that from raw failure codes.
        """
        from juno_v2.observability.history_store import read_persistent_history

        entries = read_persistent_history(
            Path(self.config.log_dir),
            limit=limit,
            before_updated_at_ms=before_updated_at_ms,
        )
        for entry in entries:
            uid = str(entry.get("utterance_id") or "").strip()
            audio_present = False
            if entry.get("audio_path") and uid:
                audio_present = bool(self.broker_audio_replay_bytes(uid))
            if entry.get("audio_path") and uid and not audio_present:
                entry["replay_available"] = False
                entry["recording_status"] = "expired_or_deleted"
            entry["recovery"] = self._derive_recovery_hints(entry, audio_present=audio_present)
        next_cursor: int | None = None
        if entries:
            try:
                next_cursor = int(entries[-1].get("updated_at_ms") or entries[-1].get("ts_unix_ms") or 0)
            except Exception:
                next_cursor = None
            if next_cursor is not None and next_cursor <= 0:
                next_cursor = None
        return {
            "ok": True,
            "session_id": self.session_id,
            "entries": entries,
            "next_cursor_updated_at_ms": next_cursor,
            "page_size": int(limit),
            "has_more": bool(entries) and len(entries) >= int(limit),
        }

    # Failure-code → recovery shape. The keys mirror the codes the shell
    # already humanises in ``UtteranceHistoryEntry.displayFailureReason``.
    # ``actions`` is a closed set the shell switches on (no string parsing
    # of failure codes in the UI). ``severity`` drives the strip colour.
    _RECOVERY_BY_FAILURE: Dict[str, Dict[str, Any]] = {
        "paste_failed": {
            "severity": "warning",
            "actions": ["insert_again", "copy_transcript"],
            "category": "insertion",
        },
        "undo_safe_paste_failed": {
            "severity": "warning",
            "actions": ["insert_again", "copy_transcript"],
            "category": "insertion",
        },
        "no_active_text_field": {
            "severity": "info",
            "actions": ["copy_transcript"],
            "category": "insertion",
        },
        "paste_kind_none_with_text": {
            "severity": "info",
            "actions": ["copy_transcript"],
            "category": "insertion",
        },
        "ax_permission_missing": {
            "severity": "warning",
            "actions": ["grant_accessibility", "copy_transcript"],
            "category": "permission",
        },
        "empty_audio": {
            "severity": "info",
            "actions": [],
            "category": "capture",
        },
        "user_cancelled_hud": {
            "severity": "info",
            "actions": ["copy_transcript"],
            "category": "capture",
        },
        "broker_unreachable": {
            "severity": "danger",
            "actions": ["restart_engine"],
            "category": "engine",
        },
    }

    def _derive_recovery_hints(self, entry: Dict[str, Any], *, audio_present: bool) -> Dict[str, Any]:
        """Compute the recovery affordances the UI should show for a row.

        Pure function over ``entry`` + audio availability. The shell uses
        this to decide which inline buttons to render and which deep-links
        to wire up; raw failure codes stay out of the user-facing copy.
        """
        failure = str(entry.get("failure_reason") or "").strip()
        actions: list[str] = []
        severity = "info"
        category = "ok"
        if failure:
            if failure.startswith("capability_blocked"):
                spec = {
                    "severity": "warning",
                    "actions": ["allow_app", "copy_transcript"],
                    "category": "capability",
                }
            else:
                spec = self._RECOVERY_BY_FAILURE.get(
                    failure,
                    {"severity": "warning", "actions": ["copy_transcript"], "category": "unknown"},
                )
            actions = list(spec["actions"])
            severity = str(spec["severity"])
            category = str(spec["category"])
        # Audio-dependent affordances are always advertised when audio is
        # actually present (regardless of failure), so successful sessions
        # can also be replayed/re-run.
        if audio_present:
            for opt in ("replay_audio", "rerun_in_mode"):
                if opt not in actions:
                    actions.append(opt)
        # Insert-again only makes sense if we have a transcript to re-fire.
        has_text = bool(
            (str(entry.get("transcript") or "").strip())
            or (str(entry.get("committed_text") or "").strip())
        )
        if "insert_again" in actions and not has_text:
            actions = [a for a in actions if a != "insert_again"]
        return {
            "severity": severity,
            "category": category,
            "actions": actions,
            "audio_present": audio_present,
            "has_text": has_text,
            "failure_code": failure or None,
        }

    def broker_history_insert_again(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return the saved transcript text for *utterance_id* so the shell
        can re-paste it via the existing capability path.

        Body: ``{"utterance_id": "<uid>"}``. Response: ``{ok, text,
        app_bundle_id, app_name}``. The shell is responsible for putting
        ``text`` on the clipboard and triggering paste — keeping all
        keystroke-synthesis on the macOS side. We do *not* paste from the
        broker; doing so would require additional capability surface and
        skip the focus-drift diagnostics.
        """
        uid = str((payload or {}).get("utterance_id") or "").strip()
        if not uid:
            return {"ok": False, "error": "utterance_id_required"}
        try:
            from juno_v2.observability.product_history import get_product_history_store

            store = get_product_history_store(Path(self.config.log_dir))
            row = store.get_entry_text(uid)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": "lookup_failed", "detail": str(exc)}
        if row is None:
            return {"ok": False, "error": "not_found", "utterance_id": uid}
        text = str(row.get("transcript") or "").strip()
        if not text:
            text = str(row.get("committed_text") or "").strip()
        if not text:
            return {"ok": False, "error": "no_text_to_insert", "utterance_id": uid}
        self.recorder.record(
            TraceKind.SYSTEM,
            "history_insert_again_requested",
            {
                "utterance_id": uid,
                "text_len": len(text),
                "app_bundle_id": row.get("app_bundle_id") or None,
            },
        )
        return {
            "ok": True,
            "utterance_id": uid,
            "text": text,
            "app_bundle_id": row.get("app_bundle_id") or "",
            "app_name": row.get("app_name") or "",
            "window_title": row.get("window_title") or "",
        }

    def broker_history_delete(self, utterance_id: str) -> Dict[str, Any]:
        from juno_v2.observability.history_store import (
            delete_history_entry,
            resolve_history_utterance_id,
        )

        uid = (utterance_id or "").strip()
        if not uid:
            return {"ok": False, "error": "utterance_id_required"}
        resolved_uid = resolve_history_utterance_id(Path(self.config.log_dir), utterance_id=uid) or uid
        audio_removed = 0
        for p in self._candidate_wavs():
            if p.stem == resolved_uid:
                try:
                    p.unlink(missing_ok=True)
                    audio_removed += 1
                except Exception:
                    pass
        removed = delete_history_entry(Path(self.config.log_dir), utterance_id=resolved_uid)
        if not removed:
            return {"ok": False, "error": "not_found", "utterance_id": resolved_uid}
        return {"ok": True, "utterance_id": resolved_uid, "audio_removed": audio_removed}

    def broker_history_update_actions(
        self, utterance_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Persist macOS-side action execution results onto a history row.

        The macOS shell calls this after dispatching parsed actions to
        Reminders / Notes. Body shape:

            {"actions": [{"kind": "reminder", "status": "ok",
                          "sink_id": "...", "sink_url": "...",
                          "body_preview": "...", "when_iso": "...",
                          "error": null}]}

        Validation is intentionally permissive: the column is opaque JSON
        from Python's perspective, and the macOS contract lives in
        ``juno_core_v3/actions/contracts.py``. We only require that
        ``actions`` is a list (or null) so a malformed client request can
        never corrupt the schema.
        """

        from juno_v2.observability.product_history import get_product_history_store

        uid = (utterance_id or "").strip()
        if not uid:
            return {"ok": False, "error": "utterance_id_required"}
        actions = payload.get("actions") if isinstance(payload, dict) else None
        if actions is not None and not isinstance(actions, list):
            return {"ok": False, "error": "actions_must_be_list"}
        actions = self._actions_with_juno_ids(actions)
        store = get_product_history_store(Path(self.config.log_dir))
        # ``update_actions`` upserts: if the row does not exist yet (race
        # against the pipeline's history write), a stub row is created so
        # the action results aren't dropped on the floor. Pipeline upsert
        # later fills in the rest while the action results survive via
        # COALESCE. See ``ProductHistoryStore.update_actions`` for details.
        store.update_actions(uid, actions)
        if _env_bool("JUNO_ACTIONS_INDEX", False):
            self._upsert_actions_index_from_results(uid, actions, payload)
        return {"ok": True, "utterance_id": uid}

    @staticmethod
    def _actions_with_juno_ids(actions: Any) -> list[Any] | None:
        if actions is None:
            return None
        out: list[Any] = []
        for item in actions:
            if not isinstance(item, dict):
                out.append(item)
                continue
            copied = dict(item)
            raw = copied.get("juno_id") or copied.get("junoId")
            if not isinstance(raw, str) or not raw.strip():
                copied["juno_id"] = str(uuid.uuid4())
            elif "juno_id" not in copied:
                copied["juno_id"] = raw.strip()
            out.append(copied)
        return out

    def _upsert_actions_index_from_results(
        self,
        utterance_id: str,
        actions: list[Any] | None,
        payload: Dict[str, Any],
    ) -> None:
        if not actions:
            return
        try:
            from juno_v2.observability.actions_index import get_actions_index

            index = getattr(self, "actions_index", None) or get_actions_index(Path(self.config.log_dir))
        except Exception:
            logger.exception("actions_index_unavailable")
            return
        app_bundle_id = None
        if isinstance(payload, dict):
            raw_app = payload.get("app_bundle_id")
            app_bundle_id = raw_app.strip() if isinstance(raw_app, str) and raw_app.strip() else None
        for item in actions:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind not in {"note", "reminder", "alarm"}:
                continue
            juno_id = str(item.get("juno_id") or item.get("junoId") or "").strip()
            if not juno_id:
                continue
            body = str(item.get("body") or item.get("body_preview") or item.get("title") or "").strip()
            schedule = item.get("schedule") if isinstance(item.get("schedule"), dict) else None
            when_iso = item.get("when_iso") or item.get("whenIso")
            if schedule is None and isinstance(when_iso, str) and when_iso.strip():
                schedule = {"kind": "instant", "instant": {"iso": when_iso.strip()}}
            status = self._actions_index_status(item)
            try:
                index.upsert(
                    juno_id=juno_id,
                    utterance_id=utterance_id,
                    sink_kind=kind,
                    sink_id=str(item.get("sink_id") or item.get("sinkId") or "").strip() or None,
                    body=body,
                    schedule=schedule,
                    app_bundle_id=app_bundle_id,
                    status=status,
                    series_id=str(item.get("series_id") or item.get("seriesId") or "").strip() or None,
                    list_name=self._action_list_name(item),
                    last_seen_session=self.session_id,
                )
            except Exception:
                logger.exception("actions_index_upsert_failed")

    @staticmethod
    def _actions_index_status(action: dict[str, Any]) -> str:
        status = str(action.get("status") or "").strip().lower()
        if status in {"completed", "deleted", "failed"}:
            return status
        if status == "ok":
            operation = str(action.get("operation") or "").strip().lower()
            if operation == "complete":
                return "completed"
            if operation == "delete":
                return "deleted"
            return "active"
        return "failed"

    @staticmethod
    def _action_list_name(action: dict[str, Any]) -> str | None:
        raw = action.get("list_name") or action.get("listName")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        container = action.get("container")
        if isinstance(container, dict):
            raw = container.get("list_name") or container.get("listName")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return None

    def broker_history_cancel_draft(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a draft transcript when the macOS HUD is dismissed (Esc) without a broker row yet."""
        from juno_v2.observability.product_history import get_product_history_store

        uid = str(payload.get("utterance_id") or "").strip()
        transcript = str(payload.get("transcript") or "").strip()
        if not uid:
            return {"ok": False, "error": "utterance_id_required"}
        if not transcript:
            return {"ok": False, "error": "transcript_required"}

        app_bundle_id = str(payload.get("app_bundle_id") or "").strip() or None
        raw = str(payload.get("raw_transcript") or transcript).strip()
        frozen_in = payload.get("frozen_context")
        frozen_dict = frozen_in if isinstance(frozen_in, dict) else None
        _sanitized, privacy_summary = self._sanitize_frozen_context(frozen_dict, app_bundle_id=app_bundle_id)
        if not privacy_summary.get("save_history", True):
            return {"ok": True, "skipped": True, "reason": "save_history_disabled"}

        window_title = str(payload.get("window_title_hint") or "").strip() or None
        ctx: Dict[str, Any] = {
            "app_bundle_id": app_bundle_id,
            "window_title": window_title,
        }
        if isinstance(_sanitized, dict):
            for k, v in _sanitized.items():
                ctx.setdefault(k, v)

        wc = len([t for t in transcript.split() if t])
        record: Dict[str, Any] = {
            "utterance_id": uid,
            "ts_unix_ms": int(time.time() * 1000),
            "transcript": transcript,
            "raw_transcript": raw,
            "mode": None,
            "failure_reason": "user_cancelled_hud",
            "session_class": "insert",
            "paste_kind": "none",
            "words": wc,
            "processing_ms": 0.0,
            "language": None,
            "language_mode": str(payload.get("language_mode") or "").strip() or None,
            "context": ctx,
            "replay_available": False,
        }
        try:
            store = get_product_history_store(Path(self.config.log_dir))
            store.upsert_from_pipeline_record(record)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "utterance_id": uid}

    def broker_history_cancel_draft(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a draft transcript when the macOS HUD is dismissed (Esc) without a broker row yet."""
        from juno_v2.observability.product_history import get_product_history_store

        uid = str(payload.get("utterance_id") or "").strip()
        transcript = str(payload.get("transcript") or "").strip()
        if not uid:
            return {"ok": False, "error": "utterance_id_required"}
        if not transcript:
            return {"ok": False, "error": "transcript_required"}

        app_bundle_id = str(payload.get("app_bundle_id") or "").strip() or None
        raw = str(payload.get("raw_transcript") or transcript).strip()
        frozen_in = payload.get("frozen_context")
        frozen_dict = frozen_in if isinstance(frozen_in, dict) else None
        _sanitized, privacy_summary = self._sanitize_frozen_context(frozen_dict, app_bundle_id=app_bundle_id)
        if not privacy_summary.get("save_history", True):
            return {"ok": True, "skipped": True, "reason": "save_history_disabled"}

        window_title = str(payload.get("window_title_hint") or "").strip() or None
        ctx: Dict[str, Any] = {
            "app_bundle_id": app_bundle_id,
            "window_title": window_title,
        }
        if isinstance(_sanitized, dict):
            for k, v in _sanitized.items():
                ctx.setdefault(k, v)

        wc = len([t for t in transcript.split() if t])
        record: Dict[str, Any] = {
            "utterance_id": uid,
            "ts_unix_ms": int(time.time() * 1000),
            "transcript": transcript,
            "raw_transcript": raw,
            "mode": None,
            "failure_reason": "user_cancelled_hud",
            "session_class": "insert",
            "paste_kind": "none",
            "words": wc,
            "processing_ms": 0.0,
            "language": None,
            "language_mode": str(payload.get("language_mode") or "").strip() or None,
            "context": ctx,
            "replay_available": False,
        }
        try:
            store = get_product_history_store(Path(self.config.log_dir))
            store.upsert_from_pipeline_record(record)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "utterance_id": uid}

    # --- Personalization profile (P0) ---

    def broker_user_profile_get(self) -> Dict[str, Any]:
        memory_dir = Path(self.config.log_dir) / "memory"
        path = memory_dir / "user_profile.json"
        if not path.exists():
            return {"ok": True, "display_name": None, "language": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"ok": True, "display_name": None, "language": None}
        if not isinstance(data, dict):
            return {"ok": True, "display_name": None, "language": None}
        return {
            "ok": True,
            "display_name": data.get("display_name"),
            "language": data.get("language"),
        }

    def broker_user_profile_set(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from juno_v2.memory.term_policy import learned_term_allowed

        memory_dir = Path(self.config.log_dir) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        path = memory_dir / "user_profile.json"
        display_name = payload.get("display_name")
        language = payload.get("language")
        out: Dict[str, Any] = {"ok": True}
        record: Dict[str, Any] = {}
        if isinstance(display_name, str):
            cleaned_name = display_name.strip()
            if cleaned_name and not learned_term_allowed(cleaned_name):
                return {"ok": False, "error": "display_name_too_short", "min_chars": 3}
            record["display_name"] = cleaned_name or None
        if isinstance(language, str):
            record["language"] = language.strip() or None
        try:
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        out.update({"display_name": record.get("display_name"), "language": record.get("language")})
        return out

    def broker_writer_extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generic LLM extraction endpoint used by memory-add UIs.

        Request shape:
            {"text": str, "kind": "vocab"|"snippet"|"replacement", "limit": int?}

        Response shape:
            {"ok": True, "candidates": [...], "kind": "vocab"}

        Designed for the macOS shell's Save Phrase / Add Snippet / Add
        Replacement flows. Falls back to ``ok=False, error="unavailable"``
        when the writer backend is missing or doesn't expose extraction —
        the shell layers a deterministic regex fallback on top.
        """
        text = str(payload.get("text") or "").strip()
        kind = str(payload.get("kind") or "vocab").strip().lower()
        if kind not in {"vocab", "snippet", "replacement"}:
            return {"ok": False, "error": "invalid_kind"}
        if not text:
            return {"ok": True, "candidates": [], "kind": kind}
        try:
            limit = max(1, min(int(payload.get("limit") or 6), 12))
        except (TypeError, ValueError):
            limit = 6
        svc = getattr(self, "writer_service", None)
        extractor = getattr(svc, "extract_memory_candidates", None) if svc is not None else None
        if extractor is None:
            return {"ok": False, "error": "unavailable"}
        try:
            cands = extractor(text=text, kind=kind, limit=limit)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        if cands is None:
            return {"ok": False, "error": "extractor_returned_none"}
        return {"ok": True, "candidates": cands, "kind": kind}

    def broker_preview_warm(self) -> Dict[str, Any]:
        """Force the preview ASR backend to load its model into memory.

        Called fire-and-forget by the macOS shell after the engine reaches
        online state, so the very first dictation utterance pays the
        warm-up latency budget at app launch instead of at hotkey-down.
        Idempotent — the backend's own ``warm()`` short-circuits if
        already loaded.

        **Thread safety**: streaming MLX backends bind their weights to
        the importing thread (see ``_ensure_preview_decode_executor``).
        We MUST run ``warm()`` on the same dedicated decode worker that
        ``broker_dictation_preview_chunk`` uses, otherwise a shell-side
        prewarm landing on the JSON-RPC handler thread re-binds the
        model to the wrong thread and the next chunk crashes with
        ``RuntimeError: There is no Stream(gpu, 0) in current thread``.
        """
        runner = getattr(self, "dictation_runner", None)
        backend = getattr(runner, "preview_backend", None) if runner else None
        if backend is None:
            return {"ok": True, "already_warm": True, "backend": "none"}
        warm = getattr(backend, "warm", None)
        if not callable(warm):
            return {
                "ok": True,
                "already_warm": True,
                "backend": getattr(backend, "backend_name", None),
            }
        already = bool(getattr(backend, "_loaded", False))
        if not already:
            executor = self._ensure_preview_decode_executor()
            try:
                # Run warm on the dedicated worker. ``_ensure_preview_decode_executor``
                # already queued the on-worker rebuild as the first task; this
                # warm submission lands behind it and naturally no-ops because
                # the rebuild has already loaded the model on this thread.
                future = executor.submit(warm)
                future.result()
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "already_warm": already,
            "backend": getattr(backend, "backend_name", None),
        }

    def broker_dictation_preview_chunk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run one streaming-preview decode on a PCM chunk from the shell.

        The macOS shell streams ~120ms PCM frames to this endpoint while
        the user is dictating. Each call returns the cumulative-utterance
        partial text the preview backend produces for that chunk. The
        shell renders that text directly in the HUD — no Apple Speech
        intermediate.

        Stateful per ``utterance_id`` via ``decode_seq``; the first chunk
        sends ``decode_seq=0`` which resets decoder state. The last
        chunk sets ``is_final=true`` so the backend can flush.

        Audio: int16 mono PCM @ ``sample_rate_hz`` (16000 Hz default),
        base64-encoded under ``audio_b64``. We pick base64 over a raw
        binary POST so the existing JSON broker stack carries it.
        """
        from juno_v2.contracts.preview import PreviewDecodeRequest

        uid = str(payload.get("utterance_id") or "").strip()
        if not uid:
            return {"ok": False, "error": "missing_utterance_id"}
        root_uid = str(payload.get("root_utterance_id") or uid).strip() or uid
        if not bool(self._settings.get("live_caption_enabled", True)):
            return {
                "ok": True,
                "text": "",
                "is_final": bool(payload.get("is_final") or False),
                "disabled": True,
            }
        transport_is_final = bool(payload.get("is_final") or False)
        root_final = bool(payload.get("root_final") or payload.get("is_root_final") or False)
        if transport_is_final and uid == root_uid:
            root_final = True
        is_final = bool(transport_is_final or root_final)
        audio_b64 = str(payload.get("audio_b64") or "")
        if not audio_b64 and not transport_is_final:
            return {"ok": False, "error": "missing_audio"}

        runner = getattr(self, "dictation_runner", None)
        if runner is not None and getattr(runner, "preview_decode_enabled", True) is False:
            return {
                "ok": True,
                "text": "",
                "is_final": bool(payload.get("is_final") or False),
                "disabled": True,
            }
        backend = getattr(runner, "preview_backend", None) if runner else None
        if backend is None:
            return {"ok": False, "error": "preview_unavailable"}

        try:
            import base64
            import numpy as np
            pcm = base64.b64decode(audio_b64, validate=False)
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"audio_decode_failed: {exc}"}

        sr = int(payload.get("sample_rate_hz") or 16000)
        try:
            decode_seq = int(payload.get("decode_seq") or 0)
        except (TypeError, ValueError):
            decode_seq = 0
        state_uid = root_uid
        root_decode_seq = self._preview_root_decode_seq_next(
            root_uid,
            transport_decode_seq=decode_seq,
            release=root_final,
        )
        visible_text_hint = re.sub(
            r"\s+",
            " ",
            str(payload.get("visible_text_hint") or payload.get("fallback_visible_text") or "").strip(),
        )
        duration_ms = (len(audio) / sr) * 1000.0 if sr > 0 else 0.0
        preview_plan = self._preview_bias_plan_for_chunk(
            runner=runner,
            utterance_id=state_uid,
            decode_seq=root_decode_seq,
            is_final=root_final,
        )
        preview_language = None
        preview_allowed_languages: list[str] = []
        preview_language_policy = None
        preview_initial_prompt = None
        preview_bias_phrases: list[str] = []
        preview_context_payload: dict[str, Any] = {}
        if preview_plan is not None:
            language_decision = {}
            plan_metadata = getattr(preview_plan, "metadata", {})
            if not isinstance(plan_metadata, dict):
                plan_metadata = {}
            try:
                raw_language_decision = plan_metadata.get("language_decision", {})
                if isinstance(raw_language_decision, dict):
                    language_decision = raw_language_decision
            except Exception:
                language_decision = {}
            preview_config = getattr(runner, "preview_config", None)
            preview_language = language_decision.get("request_language") or getattr(preview_config, "language", None)
            raw_allowed = language_decision.get("allowed_languages", [])
            if isinstance(raw_allowed, list):
                preview_allowed_languages = [str(item) for item in raw_allowed if str(item).strip()]
            preview_language_policy = language_decision.get("policy_name")
            # Preview is acoustic-first. Personalization is still delivered to
            # the HUD path below, but as evidence-gated repair terms, not as a
            # broad Whisper prompt that can generate unspoken words on short
            # quiet segments.
            preview_initial_prompt = None
            preview_bias_phrases = []
            context = getattr(preview_plan, "context", None)
            to_dict = getattr(context, "to_dict", None)
            if callable(to_dict):
                try:
                    preview_context_payload = dict(to_dict())
                except Exception:
                    preview_context_payload = {}
            preview_context_payload["language_decision"] = language_decision
            # One bias decision layer for all lanes: the plan's phrases are
            # already screen-first + family-deduped. The HUD consumes them as
            # evidence-gated repair terms (never as a generative prompt).
            plan_phrases = list(getattr(preview_plan, "bias_phrases", None) or [])
            if plan_phrases:
                preview_context_payload["preview_personalization_terms"] = plan_phrases[:24]
            raw_memory_summary = plan_metadata.get("memory_packet_summary", {})
            preview_context_payload["memory_packet_summary"] = (
                dict(raw_memory_summary) if isinstance(raw_memory_summary, dict) else {}
            )
            try:
                from juno_v2.preview.personalization_repair import preview_personalization_terms_from_plan

                preview_context_payload["preview_personalization_terms"] = (
                    preview_personalization_terms_from_plan(preview_plan)
                )
            except Exception:
                logger.exception("preview_personalization_terms_failed")
                preview_context_payload["preview_personalization_terms"] = []
            preview_context_payload["preview_prompt_mode"] = "personalization_repair_only"
            preview_context_payload["preserve_state_on_decode_seq_zero"] = True
            preview_context_payload["preview_display_orthography"] = True
            if is_final and not root_final:
                preview_context_payload["retain_state_after_final"] = True
                preview_context_payload["preview_segment_final"] = True
        raw_payload_candidates: list[Any] = []
        explicit_payload_candidates = payload.get("candidate_entities") or payload.get("candidate_terms")
        if isinstance(explicit_payload_candidates, list):
            raw_payload_candidates.extend(explicit_payload_candidates[:24])
        raw_payload_candidates.extend(
            _preview_candidates_from_session_context_tape(payload.get("session_context_tape"))[:24]
        )
        if raw_payload_candidates:
            existing_candidates = preview_context_payload.get("candidate_entities")
            merged_candidates = list(existing_candidates) if isinstance(existing_candidates, list) else []
            seen_candidates = {str(item).casefold() for item in merged_candidates if str(item).strip()}
            for raw_candidate in raw_payload_candidates[:24]:
                candidate = str(raw_candidate or "").strip()
                if not candidate:
                    continue
                key = candidate.casefold()
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                merged_candidates.append(candidate)
            preview_context_payload["candidate_entities"] = merged_candidates[:40]
        preview_context_payload.setdefault("preview_display_orthography", True)
        if is_final and not root_final:
            preview_context_payload["retain_state_after_final"] = True
            preview_context_payload["preview_segment_final"] = True

        # Per-utterance lock: the streaming preview backend is stateful and
        # callers send chunks in strict decode_seq order; serializing here
        # is cheaper than coordinating across HTTP threads on the client.
        lock = self._preview_chunk_lock_for(state_uid)
        req = PreviewDecodeRequest(
            utterance_id=state_uid,
            audio=audio,
            sample_rate_hz=sr,
            start_ms=0.0,
            end_ms=duration_ms,
            is_final=is_final,
            language=preview_language,
            allowed_languages=preview_allowed_languages,
            language_policy=preview_language_policy,
            initial_prompt=preview_initial_prompt,
            decode_seq=root_decode_seq,
            reset_decoder_state=(root_decode_seq == 0),
            bias_phrases=preview_bias_phrases,
            context_payload=preview_context_payload,
        )

        # Stateful MLX preview backends carry per-utterance state that is
        # **thread-pinned** by their underlying MLX stream. Subsequent decode
        # calls must run on that same thread or MLX raises
        # ``RuntimeError: There is no Stream(gpu, N) in current thread``.
        #
        # The JSON-RPC server gives us a fresh socket-handler thread per
        # connection, and the Swift shell opens a fresh connection per
        # chunk, so without this dispatch every chunk lands on a different
        # thread and the streaming backend crashes from chunk 1 onward.
        # Queueing every preview decode through a single dedicated worker
        # thread makes the streaming context's thread invariant hold
        # across the whole utterance and across all utterances in the
        # process. The wrapping ``with lock:`` still serializes across
        # utterances and protects shared MLX state.
        executor = self._ensure_preview_decode_executor()
        try:
            future = executor.submit(self._run_preview_decode_locked, lock, backend, req)
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        finally:
            if root_final:
                self._preview_chunk_lock_release(state_uid)

        raw_meta = dict(getattr(result, "metadata", {}) or {})
        # LocalAgreement-2 contract: the preview-service emits a stable
        # ``committed_text`` (never shrinks) plus an unstable ``tail_text``.
        # ``text`` is the combined committed + tail string for legacy callers.
        #
        # Wake-phrase utterances keep streaming as ordinary dictation. The
        # earlier "Hey Juno: <kinds>" status takeover replaced the committed
        # lane and blanked the tail, which froze the HUD for the rest of the
        # utterance and leaked the synthetic status string into the live
        # transcript hint. The detected action kinds stay available to
        # observers as display metadata only.
        committed_text = str(raw_meta.get("committed_text") or "")
        tail_text = str(raw_meta.get("tail_text") or "")
        action_preview_text = _action_preview_display_text(committed_text, tail_text)
        if action_preview_text is not None:
            action_preview_text = self._remember_action_preview_display(
                state_uid,
                action_preview_text,
                final_chunk=bool(root_final),
            )
            raw_meta["display_override"] = {
                "kind": "action_status",
                "text": action_preview_text,
                "display_only": True,
            }
        elif root_final:
            self._clear_action_preview_display(state_uid)
        if committed_text and tail_text:
            text = f"{committed_text} {tail_text}"
        else:
            text = committed_text or tail_text or (result.text or "")

        audio_diag = analyze_audio_signal(audio, sample_rate_hz=sr)
        filter_reason: str | None = raw_meta.get("decode_skip_reason")

        out = {
            "ok": True,
            "text": text,
            "committed_text": committed_text,
            "tail_text": tail_text,
            "is_final": bool(transport_is_final),
            "decode_ms": float(result.decode_ms),
            "decode_seq": decode_seq,
            "root_decode_seq": root_decode_seq,
            "request_audio_ms": duration_ms,
            "root_utterance_id": root_uid,
            "preview_state_utterance_id": state_uid,
            "root_final": bool(root_final),
            "low_signal": bool(audio_diag.low_signal),
            "audio_diagnostics": audio_diag.to_dict(),
            "tail_display_suppress_reason": raw_meta.get("tail_display_suppress_reason"),
        }
        if raw_meta:
            out["preview_metadata"] = raw_meta
        for key in ("coalesced_chunk_count", "coalesced_audio_ms"):
            if key in payload:
                out[key] = payload.get(key)
        if filter_reason:
            out["filtered_reason"] = filter_reason
        # NOTE: live_transcript_delta was the old patch-envelope path used by
        # HUDTranscriptStore.applyLivePreviewDelta. That path was the source of
        # the HUD freeze (wholesale text replacement on every drift). It is
        # intentionally NOT emitted. The shell now reads committed_text /
        # tail_text directly.
        live_delta: dict[str, Any] | None = None
        raw_payload = raw_meta.get("raw") if isinstance(raw_meta.get("raw"), dict) else {}
        service_payload = raw_payload.get("service") if isinstance(raw_payload.get("service"), dict) else {}
        backend_config = getattr(backend, "config", None)
        configured_model_ref = getattr(backend_config, "model_path", None)
        raw_text = str(result.text or "")
        preview_model_ref = (
            raw_meta.get("mlx_whisper_model_ref")
            or raw_payload.get("mlx_whisper_model_ref")
            or raw_payload.get("service_model_path")
            or service_payload.get("model_path")
            or (str(configured_model_ref) if configured_model_ref is not None else None)
        )
        service_backend = (
            raw_payload.get("service_backend")
            or service_payload.get("backend")
            or getattr(backend, "backend_name", None)
            or getattr(result, "backend_name", None)
        )
        self.recorder.record(
            TraceKind.ASR_PREVIEW,
            "broker_preview_chunk_result",
            {
                "utterance_id": uid,
                "root_utterance_id": root_uid,
                "preview_state_utterance_id": state_uid,
                "preview_segment_id": uid if uid != root_uid else None,
                "decode_seq": decode_seq,
                "root_decode_seq": root_decode_seq,
                "is_final": bool(result.is_final),
                "transport_is_final": bool(transport_is_final),
                "root_final": bool(root_final),
                "backend": getattr(result, "backend_name", None),
                "service_backend": service_backend,
                "model_ref": preview_model_ref,
                "raw_text_preview": raw_text.replace("\n", " ")[:240],
                "raw_text_chars": len(raw_text),
                "committed_text_preview": committed_text.replace("\n", " ")[:240],
                "committed_text_chars": len(committed_text),
                "tail_text_preview": tail_text.replace("\n", " ")[:240],
                "tail_text_chars": len(tail_text),
                "decode_skip_reason": filter_reason,
                "commit_events": raw_meta.get("commit_events"),
                "segment_trim_events": raw_meta.get("segment_trim_events"),
                "force_trim_events": raw_meta.get("force_trim_events"),
                "decode_attempts": raw_meta.get("decode_attempts"),
                "decode_skipped_cadence": raw_meta.get("decode_skipped_cadence"),
                "decode_skipped_too_short": raw_meta.get("decode_skipped_too_short"),
                "decode_skipped_vad_silence": raw_meta.get("decode_skipped_vad_silence"),
                "decode_silence_confirmations": raw_meta.get("decode_silence_confirmations"),
                "decode_on_silence": raw_meta.get("decode_on_silence"),
                "vad_admit_reason": raw_meta.get("vad_admit_reason"),
                "buffer_audio_ms": raw_meta.get("buffer_audio_ms"),
                "buffer_start_t": raw_meta.get("buffer_start_t"),
                "decode_ms": float(result.decode_ms),
                "request_audio_ms": duration_ms,
                "cumulative_audio_ms": raw_meta.get("cumulative_audio_ms"),
                "fed_new_audio_ms": raw_meta.get("fed_new_audio_ms"),
                "visible_text_hint_chars": len(visible_text_hint),
                "low_signal": bool(audio_diag.low_signal),
                "audio_rms": audio_diag.rms,
                "audio_peak": audio_diag.peak,
                # Hallucination-defense visibility added round 6. Lets us
                # post-mortem WHY a segment was dropped (no_speech_prob too
                # high vs BoH phrase match) without re-instrumenting the
                # subprocess.
                "tail_suppressed_events": raw_meta.get("tail_suppressed_events"),
                "tail_suppress_reason": raw_meta.get("tail_suppress_reason"),
                "tail_display_suppress_reason": raw_meta.get("tail_display_suppress_reason"),
                "tail_quarantine_reason": raw_meta.get("tail_quarantine_reason"),
                "tail_commit_quarantine_events": raw_meta.get("tail_commit_quarantine_events"),
                "committed_replay_suppressed_events": raw_meta.get("committed_replay_suppressed_events"),
                "committed_replay_agreement_drops": raw_meta.get("committed_replay_agreement_drops"),
                "commit_draft_horizon_demotions": raw_meta.get("commit_draft_horizon_demotions", 0),
                "committed_boundary_letter_strips": raw_meta.get("committed_boundary_letter_strips"),
                "tail_no_speech_prob": raw_meta.get("tail_no_speech_prob"),
                "last_segment_no_speech_prob": raw_meta.get("last_segment_no_speech_prob"),
                "preview_repair_terms": raw_meta.get("preview_repair_terms"),
                "preview_repair_applied": raw_meta.get("preview_repair_applied"),
                "preview_repairs": raw_meta.get("preview_repairs"),
                "preview_orthography_applied": raw_meta.get("preview_orthography_applied"),
                "preview_orthography_committed_changed": raw_meta.get("preview_orthography_committed_changed"),
                "preview_orthography_tail_changed": raw_meta.get("preview_orthography_tail_changed"),
                "avg_logprob": raw_meta.get("avg_logprob"),
                "compression_ratio": raw_meta.get("compression_ratio"),
                "tail_final_promotion_status": raw_meta.get("tail_final_promotion_status"),
                "tail_final_promotion_reason": raw_meta.get("tail_final_promotion_reason"),
                "tail_final_promotion_blocked_events": raw_meta.get("tail_final_promotion_blocked_events"),
                "decodes_since_last_commit": raw_meta.get("decodes_since_last_commit"),
                "segments_dropped": raw_meta.get("segments_dropped"),
                "dropped_segments_preview": raw_meta.get("dropped_segments_preview"),
                "segment_count": raw_meta.get("segment_count"),
            },
        )
        # Per-utterance lifecycle counters so the user can verify "did the HUD
        # actually update?" from telemetry alone. Aggregated across the
        # utterance — the latest chunk's values are what gets persisted.
        if transport_is_final or decode_seq == 0 or (decode_seq % 8 == 0):
            self._record_utterance_lifecycle(
                uid,
                {
                    "event": "preview_telemetry",
                    "decode_seq": decode_seq,
                    "root_decode_seq": root_decode_seq,
                    "is_final": bool(result.is_final),
                    "transport_is_final": bool(transport_is_final),
                    "root_final": bool(root_final),
                    "preview_committed_chars": len(committed_text),
                    "preview_tail_chars": len(tail_text),
                    "preview_commit_events": int(raw_meta.get("commit_events") or 0),
                    "preview_segment_trim_events": int(raw_meta.get("segment_trim_events") or 0),
                    "preview_force_trim_events": int(raw_meta.get("force_trim_events") or 0),
                    "preview_decode_attempts": int(raw_meta.get("decode_attempts") or 0),
                    "preview_decode_skipped_cadence": int(raw_meta.get("decode_skipped_cadence") or 0),
                    "preview_decode_skipped_too_short": int(raw_meta.get("decode_skipped_too_short") or 0),
                    "preview_decode_skipped_vad_silence": int(raw_meta.get("decode_skipped_vad_silence") or 0),
                    "preview_decode_silence_confirmations": int(raw_meta.get("decode_silence_confirmations") or 0),
                    "preview_tail_suppressed_events": int(raw_meta.get("tail_suppressed_events") or 0),
                    "preview_tail_commit_quarantine_events": int(raw_meta.get("tail_commit_quarantine_events") or 0),
                    "preview_committed_replay_suppressed_events": int(raw_meta.get("committed_replay_suppressed_events") or 0),
                    "preview_committed_replay_agreement_drops": int(raw_meta.get("committed_replay_agreement_drops") or 0),
                    "preview_tail_display_suppress_reason": raw_meta.get("tail_display_suppress_reason"),
                    "preview_tail_quarantine_reason": raw_meta.get("tail_quarantine_reason"),
                    "preview_committed_boh_strips": int(raw_meta.get("committed_boh_strips") or 0),
                    "preview_committed_boundary_letter_strips": int(raw_meta.get("committed_boundary_letter_strips") or 0),
                    "preview_tail_no_speech_prob": raw_meta.get("tail_no_speech_prob"),
                    "preview_last_segment_no_speech_prob": raw_meta.get("last_segment_no_speech_prob"),
                    "preview_repair_terms": int(raw_meta.get("preview_repair_terms") or 0),
                    "preview_repair_applied": int(raw_meta.get("preview_repair_applied") or 0),
                    "preview_repairs": raw_meta.get("preview_repairs") or [],
                    "preview_orthography_applied": int(raw_meta.get("preview_orthography_applied") or 0),
                    "preview_tail_final_promotion_status": raw_meta.get("tail_final_promotion_status"),
                    "preview_tail_final_promotion_reason": raw_meta.get("tail_final_promotion_reason"),
                    "preview_tail_final_promotion_blocked_events": int(raw_meta.get("tail_final_promotion_blocked_events") or 0),
                    "preview_buffer_audio_ms": float(raw_meta.get("buffer_audio_ms") or 0.0),
                    "preview_buffer_start_t": float(raw_meta.get("buffer_start_t") or 0.0),
                    "preview_last_decode_ms": float(result.decode_ms),
                    # P3.2: on final, snapshot the full committed text so
                    # post-utterance audits don't need engine.log grepping.
                    **(
                        {"preview_committed_text_final": str(raw_meta.get("committed_text_final") or "")}
                        if bool(result.is_final)
                        else {}
                    ),
                },
            )
        return out

    def _preview_bias_plan_for_chunk(
        self,
        *,
        runner: Any,
        utterance_id: str,
        decode_seq: int,
        is_final: bool,
    ) -> Any | None:
        if runner is None:
            return None
        if decode_seq == 0:
            self._preview_bias_plan_release(utterance_id)
        cached = self._preview_bias_plan_get(utterance_id)
        if cached is not None:
            if is_final:
                self._preview_bias_plan_release(utterance_id)
            return cached
        build_plan = getattr(runner, "_plan_for_utterance", None)
        if not callable(build_plan):
            return None
        try:
            plan = build_plan(utterance_id)
        except Exception:
            logger.exception("preview_bias_plan_failed")
            return None
        if not is_final:
            self._preview_bias_plan_set(utterance_id, plan)
        return plan

    def _preview_bias_plan_get(self, uid: str) -> Any | None:
        if not hasattr(self, "_preview_bias_plan_by_uid"):
            self._preview_bias_plan_by_uid = {}
        return self._preview_bias_plan_by_uid.get(uid)

    def _preview_bias_plan_set(self, uid: str, plan: Any) -> None:
        if not hasattr(self, "_preview_bias_plan_by_uid"):
            self._preview_bias_plan_by_uid = {}
        self._preview_bias_plan_by_uid[uid] = plan

    def _preview_bias_plan_release(self, uid: str) -> None:
        if not hasattr(self, "_preview_bias_plan_by_uid"):
            return
        self._preview_bias_plan_by_uid.pop(uid, None)

    def _preview_root_decode_seq_next(
        self,
        uid: str,
        *,
        transport_decode_seq: int,
        release: bool = False,
    ) -> int:
        if not hasattr(self, "_preview_root_decode_seq_by_uid"):
            self._preview_root_decode_seq_by_uid = {}
        seqs = self._preview_root_decode_seq_by_uid
        if uid in seqs:
            seq = int(seqs.get(uid) or 0) + 1
        else:
            seq = max(0, int(transport_decode_seq or 0))
        if release:
            seqs.pop(uid, None)
        else:
            seqs[uid] = seq
        return seq

    def _preview_last_text_for(self, uid: str) -> str:
        if not hasattr(self, "_preview_last_text_by_uid"):
            self._preview_last_text_by_uid = {}
        return str(self._preview_last_text_by_uid.get(uid) or "")

    def _preview_text_state_set(self, uid: str, text: str) -> None:
        if not hasattr(self, "_preview_last_text_by_uid"):
            self._preview_last_text_by_uid = {}
        self._preview_last_text_by_uid[uid] = text

    def _preview_text_state_release(self, uid: str) -> None:
        if not hasattr(self, "_preview_last_text_by_uid"):
            return
        self._preview_last_text_by_uid.pop(uid, None)

    # _preview_live_state_manager removed: LiveTranscriptStateManager and the
    # live patch envelope path were deleted with the LocalAgreement-2 rewrite.
    # The remaining text-state helpers (_preview_last_text_*) are orphaned dead
    # code; they'll be removed in a separate cleanup pass once we confirm
    # nothing else depends on them.

    def _remember_action_preview_display(
        self,
        uid: str,
        current: str,
        *,
        final_chunk: bool,
    ) -> str:
        with self._action_preview_display_lock:
            previous = self._action_preview_display_by_utterance.get(uid)
            merged = _merge_action_preview_display_text(previous, current)
            if final_chunk:
                self._action_preview_display_by_utterance.pop(uid, None)
            else:
                self._action_preview_display_by_utterance[uid] = merged
            return merged

    def _clear_action_preview_display(self, uid: str) -> None:
        with self._action_preview_display_lock:
            self._action_preview_display_by_utterance.pop(uid, None)

    def _preview_chunk_lock_for(self, uid: str):
        import threading
        if not hasattr(self, "_preview_chunk_locks"):
            self._preview_chunk_locks = {}
            self._preview_chunk_locks_guard = threading.Lock()
        with self._preview_chunk_locks_guard:
            lock = self._preview_chunk_locks.get(uid)
            if lock is None:
                lock = threading.Lock()
                self._preview_chunk_locks[uid] = lock
            return lock

    def _preview_chunk_lock_release(self, uid: str) -> None:
        if not hasattr(self, "_preview_chunk_locks"):
            return
        with self._preview_chunk_locks_guard:
            self._preview_chunk_locks.pop(uid, None)

    def _ensure_preview_decode_executor(self) -> ThreadPoolExecutor:
        """Single-worker executor that owns *every* preview decode call.

        MLX models can bind arrays and stream state to the thread that warms
        or first evaluates them. Worker threads only see streams they created,
        so main-thread warmup can be unusable from a request handler thread.
        On first use we submit a worker-side rebuild task: it tears down any
        main-thread-warmed preview backend and re-warms it on the decode
        worker. Every subsequent decode runs on that same thread.

        Lazily created so unit tests that construct a stub WorkbenchApp
        without going through ``__init__`` still see a working pool the
        first time they call ``broker_dictation_preview_chunk``. We don't
        bother with shutdown — the executor lives for the broker process
        and Python's atexit cleans the daemon thread on interpreter exit.
        """
        executor = getattr(self, "_preview_decode_executor", None)
        if executor is not None:
            return executor
        # Guard creation so two concurrent callers don't race on first use.
        if not hasattr(self, "_preview_decode_executor_guard"):
            self._preview_decode_executor_guard = threading.Lock()
        with self._preview_decode_executor_guard:
            executor = getattr(self, "_preview_decode_executor", None)
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="juno-preview-decode",
                )
                # Queue the rebuild as the very first task so it runs
                # before any decode submission. ThreadPoolExecutor with
                # max_workers=1 is FIFO, so chunk submissions naturally
                # block behind this until the worker has its own copy of
                # the model. We do NOT join here — that would deadlock
                # if WorkbenchApp.__init__ ever called us. Decode
                # submissions await rebuild via the future.
                self._preview_rebuild_future = executor.submit(
                    self._rebuild_preview_backend_on_worker
                )
                self._preview_decode_executor = executor
        return executor

    def _rebuild_preview_backend_on_worker(self) -> None:
        """Re-warm the streaming preview backend on the calling thread.

        Runs exactly once on the dedicated decode worker. Drops any
        main-thread-loaded model state (so its main-thread MLX arrays
        get GC'd), re-imports + re-loads the model so its weights are
        bound to *this* thread's MLX stream, then **primes the streaming
        decode context** with a tiny audio chunk. The prime is critical:
        first decode on a fresh MLX model can trigger JIT compilation and
        first-pass graph setup. Without the prime here, that cost gets paid by
        the user's first real chunk and can exceed the shell's first-word
        freshness budget.

        Failures are logged and propagated as the future's exception so
        the first decode submission surfaces them; we don't silently
        leave the backend in a half-initialized state.
        """
        runner = getattr(self, "dictation_runner", None)
        backend = getattr(runner, "preview_backend", None) if runner else None
        if backend is None:
            return
        # Drop main-thread state. ``unload`` is the canonical reset; if
        # the backend doesn't expose it (test stubs) we fall back to
        # poking ``_model`` so the next ``warm()`` re-imports.
        unload = getattr(backend, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                logger.exception("preview_backend_unload_failed_on_rebuild")
        else:
            try:
                if hasattr(backend, "_model"):
                    backend._model = None
            except Exception:
                pass
        # Re-warm on this thread so MLX bindings are tied to it.
        lifecycle = getattr(self, "lifecycle_manager", None)
        if lifecycle is not None and hasattr(lifecycle, "warm_component"):
            lifecycle.warm_component("preview_asr", force=True)
        else:
            warm = getattr(backend, "warm", None)
            if callable(warm):
                warm()
        # Prime the streaming decode path. Use the backend's ``decode``
        # contract so any streaming preview backend is exercised the same way the
        # broker handler will exercise it. Failures are non-fatal — if
        # the primer can't construct a request the first real chunk
        # will pay the cold start, which is a regression but not a
        # crash.
        decode = getattr(backend, "decode", None)
        if callable(decode):
            try:
                from juno_v2.contracts.preview import PreviewDecodeRequest
                import numpy as _np

                # 250ms of near-silence is enough to compile common MLX
                # decode kernels. Real audio isn't required; kernels compile
                # on shape, not content.
                primer_audio = _np.zeros(4000, dtype=_np.float32)
                primer_uid = "__prewarm_streaming__"
                req = PreviewDecodeRequest(
                    utterance_id=primer_uid,
                    audio=primer_audio,
                    sample_rate_hz=16000,
                    start_ms=0.0,
                    end_ms=250.0,
                    is_final=True,
                    decode_seq=0,
                    reset_decoder_state=True,
                )
                decode(req)
            except Exception:
                logger.exception("preview_backend_streaming_prime_failed")

    def _run_preview_decode_locked(self, lock, backend, req):
        """Body executed on the dedicated preview-decode worker thread.

        Waits for the one-time worker-side rebuild to complete before
        the first decode runs. ``ThreadPoolExecutor`` with one worker
        already serializes us behind the rebuild submission, but we
        ``.result()`` defensively so a rebuild error surfaces here
        instead of silently producing a half-warmed decode call.
        """
        rebuild_future = getattr(self, "_preview_rebuild_future", None)
        if rebuild_future is not None and not rebuild_future.done():
            rebuild_future.result()
        elif rebuild_future is not None and rebuild_future.exception() is not None:
            raise rebuild_future.exception()  # type: ignore[misc]
        with lock:
            return backend.decode(req)

    def broker_writer_warm(self) -> Dict[str, Any]:
        svc = getattr(self, "writer_service", None)
        if svc is None or getattr(svc, "backend", None) is None:
            return {"ok": True, "already_warm": True, "backend": "none"}
        backend = getattr(svc, "backend", None)
        already = bool(getattr(backend, "_loaded", False))
        lifecycle = getattr(self, "lifecycle_manager", None)
        acquire = getattr(lifecycle, "acquire", None)
        release = getattr(lifecycle, "release", None)
        if callable(acquire) and callable(release):
            try:
                acquire("writer")
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            try:
                release("writer")
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        elif not already:
            try:
                svc.warm()
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "already_warm": already,
            "backend": getattr(backend, "backend_name", None),
        }

    def broker_engine_compatibility(self) -> Dict[str, Any]:
        """Mac shell / launcher identity handshake.

        Production-grade revamp (Phase 1, 2026-04-30): the response now
        carries enough identity for the shell to distinguish a live
        ``runtime.service`` engine from a developer's standalone
        ``python -m juno_v2.workbench.server`` impersonating it on a
        shared dev port. The Swift shell rejects any peer whose
        ``runtime_role`` is not ``juno_runtime_service``.
        """
        from juno_v2.runtime.shell_engine_contract import (
            SHELL_ENGINE_PROTOCOL_VERSION,
        )

        return {
            "ok": True,
            "product": "Juno",
            "engine": "juno-local-voice-engine",
            "engine_version": "0.2.0",
            "shell_engine_protocol_version": SHELL_ENGINE_PROTOCOL_VERSION,
            "min_shell_engine_protocol_version": SHELL_ENGINE_PROTOCOL_VERSION,
            "runtime_role": getattr(self, "runtime_role", "workbench_standalone"),
            "instance_id": getattr(self, "instance_id", ""),
            "bundle_id": getattr(self, "bundle_id", ""),
            "pid": getattr(self, "process_pid", 0),
            "started_at": getattr(self, "started_at", 0.0),
            "workbench_session_id": self.session_id,
            "deployment_profile": dict(getattr(self, "deployment_profile", {}) or {}),
            "auth_header": "X-Juno-Local-Token",
            "warm": self.warm_status(),
            "required_routes": [
                "/healthz",
                "/api/broker/engine/compatibility",
                "/api/broker/dictation/ingest_wav",
                "/api/broker/insertion/committed",
                "/api/broker/privacy/context_settings",
                "/api/broker/privacy/app_overrides",
                "/api/broker/settings/language_environment",
                "/api/broker/memory/vocab",
                "/api/broker/history",
                "/api/broker/history/cancel_draft",
            ],
        }

    def broker_stats_summary(self) -> Dict[str, Any]:
        from juno_v2.workbench.stats import StatsCache

        if not hasattr(self, "_stats_cache"):
            self._stats_cache = StatsCache()
        return self._stats_cache.get_or_compute(log_dir=Path(self.config.log_dir), ttl_s=60)

    def broker_storage_stats(self) -> Dict[str, Any]:
        history_path = Path(self.config.log_dir) / "history.jsonl"
        sql_entries = 0
        sql_bytes = 0
        try:
            from juno_v2.observability.product_history import get_product_history_store

            sql_entries, sql_bytes = get_product_history_store(Path(self.config.log_dir)).stats()
        except Exception:
            pass

        # Audio retention lives in multiple roots; use the shared walker so
        # stats match what replay/cleanup considers eligible.
        audio_files = 0
        audio_bytes = 0
        oldest_audio_ts: int | None = None
        for p in self._candidate_wavs():
            try:
                st = p.stat()
            except OSError:
                continue
            audio_files += 1
            audio_bytes += int(st.st_size)
            ts = int(st.st_mtime * 1000)
            oldest_audio_ts = ts if oldest_audio_ts is None else min(oldest_audio_ts, ts)

        history_entries = sql_entries
        history_bytes = sql_bytes
        if history_path.exists():
            try:
                history_bytes += int(history_path.stat().st_size)
            except OSError:
                pass
            try:
                with history_path.open("r", encoding="utf-8") as fh:
                    for _ in fh:
                        history_entries += 1
            except OSError:
                pass

        return {
            "ok": True,
            "log_dir": str(Path(self.config.log_dir).resolve()),
            "audio_files": audio_files,
            "audio_bytes": audio_bytes,
            "oldest_audio_ts": oldest_audio_ts,
            "history_entries": history_entries,
            "history_bytes": history_bytes,
            "product_history_sqlite_entries": sql_entries,
            "product_history_sqlite_bytes": sql_bytes,
        }

    def broker_storage_prune_all_audio(self) -> Dict[str, Any]:
        deleted = 0
        for p in self._candidate_wavs():
            try:
                p.unlink(missing_ok=True)
                deleted += 1
            except OSError:
                pass
        return {"ok": True, "deleted": deleted}

    def broker_audio_replay_bytes(self, utterance_id: str) -> bytes | None:
        pipeline = getattr(self, "oneshot_pipeline", None)
        if pipeline is None:
            return None
        fn = getattr(pipeline, "get_audio_for_rerun", None)
        if not callable(fn):
            return None
        try:
            return fn(str(utterance_id))
        except Exception:
            return None

    def _transcriber_from_registry_package(self, pkg: Any) -> DictationTranscriber:
        from juno_core_v3.dictation import FinalBackendTranscriber
        from juno_v2.final.config import FinalAsrConfig
        from juno_v2.runtime.backends import create_final_backend

        backend_name = str(pkg.manifest.backend.value)
        metadata = dict(getattr(pkg, "metadata", {}) or {})
        model_path = metadata.get("model_path") or metadata.get("hf_repo_id") or ""
        endpoint = metadata.get("local_http_endpoint") or metadata.get("endpoint")
        if backend_name in {"faster_whisper", "mlx_whisper"} and not model_path:
            raise ValueError("rerun_route_unbuildable")
        if backend_name == "local_http_json" and not endpoint:
            raise ValueError("rerun_route_unbuildable")
        cfg = FinalAsrConfig(
            model_path=model_path or ".",
            backend_name=backend_name,
            language=None,
            local_http_endpoint=endpoint,
            hf_repo_id=metadata.get("hf_repo_id"),
        )
        backend = create_final_backend(cfg)
        return FinalBackendTranscriber(backend=backend, language=None)

    def _build_rerun_transcriber(self, route: str | None) -> tuple[DictationTranscriber, Dict[str, Any]]:
        requested = str(route or "current").strip() or "current"
        if requested == "current":
            transcriber = self.oneshot_pipeline.transcriber
            return transcriber, {
                "requested": requested,
                "selected": "current",
                "backend": getattr(transcriber, "backend_name", "unknown"),
            }

        from juno_core_v3.model_registry.contracts import ModelSlot, SurfaceClass
        from juno_core_v3.model_registry.routing import RouteRequest

        pkg = None
        if requested == "default":
            chosen = self.broker.route_chooser.choose(
                RouteRequest(slot=ModelSlot.FINAL_ASR, surface=SurfaceClass.DESKTOP)
            )
            pkg = chosen.chosen
            if pkg is None:
                raise ValueError("rerun_route_unavailable")
        else:
            registry = getattr(self.broker.route_chooser, "registry", None)
            pkg = registry.get(requested) if registry is not None else None
            if pkg is None:
                raise ValueError("rerun_route_unknown")

        if pkg.manifest.slot != ModelSlot.FINAL_ASR:
            raise ValueError("rerun_route_not_final_asr")
        transcriber = self._transcriber_from_registry_package(pkg)
        return transcriber, {
            "requested": requested,
            "selected": pkg.package_id,
            "backend": pkg.manifest.backend.value,
            "model_ref": pkg.metadata.get("model_path")
            or pkg.metadata.get("hf_repo_id")
            or pkg.metadata.get("endpoint"),
        }

    def broker_replay_utterance(self, utterance_id: str, *, route: str | None = None) -> Dict[str, Any]:
        """Replay stored audio for utterance_id through the pipeline again."""
        if not utterance_id:
            return {"ok": False, "error": "utterance_id_required"}
        wav_bytes = self.oneshot_pipeline.get_audio_for_rerun(utterance_id)
        if wav_bytes is None:
            return {"ok": False, "error": "audio_not_retained", "utterance_id": utterance_id}
        try:
            transcriber, route_meta = self._build_rerun_transcriber(route)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "utterance_id": utterance_id}
        with self._lock:
            rerun_pipeline = replace(self.oneshot_pipeline, transcriber=transcriber)
            result = rerun_pipeline.run(wav_bytes, utterance_id=f"replay_{utterance_id[:20]}")
        out: Dict[str, Any] = {
            "ok": bool(result.ok),
            "original_utterance_id": utterance_id,
            "rerun_route": route_meta,
            "result": result.to_dict(),
        }
        if not result.ok:
            out["error"] = result.error
            out["error_code"] = result.error_code
        self.recorder.record(
            TraceKind.SYSTEM,
            "broker_replay_rerun",
            {
                "utterance_id": utterance_id,
                "replay_utterance_id": result.utterance_id,
                "ok": bool(result.ok),
                "requested_route": route_meta.get("requested"),
                "selected_route": route_meta.get("selected"),
                "backend": route_meta.get("backend"),
                "error_code": result.error_code,
            },
        )
        return out

    def broker_delete_replay_audio(self, utterance_id: str) -> Dict[str, Any]:
        if not utterance_id:
            return {"ok": False, "error": "utterance_id_required"}
        deleted = self.oneshot_pipeline.delete_audio_for_rerun(utterance_id)
        if not deleted:
            return {"ok": False, "error": "audio_not_retained", "utterance_id": utterance_id}
        self.recorder.record(
            TraceKind.SYSTEM,
            "broker_replay_audio_deleted",
            {"utterance_id": utterance_id},
        )
        return {"ok": True, "utterance_id": utterance_id, "deleted": True}

    def broker_history_reprocess(
        self,
        utterance_id: str,
        mode_name: str,
        *,
        is_custom: bool = False,
    ) -> Dict[str, Any]:
        """Re-transcribe stored audio with a different mode — preview only.

        Runs ASR + writer with the requested mode override. Never commits,
        pastes, or writes a history entry. The caller displays the returned
        transcript as a read-only preview card.
        """
        if not utterance_id:
            return {"ok": False, "error": "utterance_id_required"}
        if not mode_name:
            return {"ok": False, "error": "mode_required"}

        pipeline = getattr(self, "oneshot_pipeline", None)
        if pipeline is None:
            return {"ok": False, "error": "engine_not_ready"}

        wav_bytes = pipeline.get_audio_for_rerun(utterance_id)
        if wav_bytes is None:
            return {"ok": False, "error": "audio_not_retained", "utterance_id": utterance_id}

        manual_mode = None if is_custom else mode_name
        custom_mode = mode_name if is_custom else None

        try:
            result = pipeline.run(
                wav_bytes,
                utterance_id=f"reprocess_{utterance_id[:20]}",
                manual_writer_mode=manual_mode,
                custom_writer_mode=custom_mode,
                save_history=False,
                save_audio=False,
            )
        except Exception as exc:
            logger.warning("broker_history_reprocess error uid=%s: %s", utterance_id, exc)
            return {"ok": False, "error": str(exc), "utterance_id": utterance_id}

        if not result.ok:
            return {
                "ok": False,
                "error": result.error or "reprocess_failed",
                "error_code": result.error_code,
                "utterance_id": utterance_id,
            }

        self.recorder.record(
            TraceKind.SYSTEM,
            "broker_history_reprocess",
            {
                "utterance_id": utterance_id,
                "mode_name": mode_name,
                "is_custom": is_custom,
                "ok": True,
            },
        )
        return {
            "ok": True,
            "utterance_id": utterance_id,
            "mode_name": mode_name,
            "transcript": result.transcript,
            "raw_transcript": result.raw_transcript,
        }

    def broker_model_routes(self) -> Dict[str, Any]:
        with self._lock:
            return {"ok": True, "routes": self.broker.model_routes()}

    def broker_runtime_backends(self) -> Dict[str, Any]:
        """List currently-active final-ASR backend + registered alternatives.

        Returns ``{"ok": false, "error": "swap_not_available"}`` for
        standalone workbench runs (no engine loaded) — matches the
        existing convention used by other broker endpoints when their
        dependency is missing.
        """
        if self.final_swap is None:
            return {
                "ok": False,
                "error_code": "swap_not_available",
                "error": "no engine final backend wired into this workbench",
            }
        return {"ok": True, "final": self.final_swap.snapshot()}

    def broker_replay_all_finals(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run a previously-retained utterance through every registered
        final-ASR package and return all transcripts side-by-side.

        Lets users compare ASR models on the same audio without
        re-speaking. Sequential execution — each package is loaded,
        decoded, then dropped — so wall time is roughly the sum of
        per-model load + decode times. Cold loads can run 30s+ on
        first use; warm loads (model already in HF cache) are sub-second.

        ``payload`` keys:
            utterance_id  — explicit id; if omitted, picks the most
                            recently retained utterance from the
                            on-disk audio cache.
        """
        from juno_core_v3.model_registry.contracts import ModelSlot

        utterance_id_input = str(payload.get("utterance_id") or "").strip() or None
        if utterance_id_input is None:
            resolved = self._most_recent_retained_audio()
            if resolved is None:
                return {
                    "ok": False,
                    "error_code": "no_retained_audio",
                    "error": "no utterance audio is available for replay",
                }
            utterance_id, wav_bytes = resolved
        else:
            utterance_id = utterance_id_input
            wav_bytes = self._load_retained_audio(utterance_id)
            if wav_bytes is None:
                return {
                    "ok": False,
                    "error_code": "audio_not_retained",
                    "error": f"no retained audio for utterance {utterance_id!r}",
                    "utterance_id": utterance_id,
                }

        registry = getattr(self.broker.route_chooser, "registry", None)
        if registry is None:
            return {
                "ok": False,
                "error_code": "registry_unavailable",
                "error": "model registry not wired into this workbench",
            }
        packages = [
            p for p in registry._packages.values()
            if p.manifest.slot == ModelSlot.FINAL_ASR
        ]

        results: list[Dict[str, Any]] = []
        for pkg in packages:
            entry: Dict[str, Any] = {"package_id": pkg.package_id}
            try:
                t0 = time.monotonic()
                transcriber = self._transcriber_from_registry_package(pkg)
                load_ms = (time.monotonic() - t0) * 1000.0
                with self._lock:
                    rerun_pipeline = replace(self.oneshot_pipeline, transcriber=transcriber)
                    result = rerun_pipeline.run(
                        wav_bytes,
                        utterance_id=f"compareall_{pkg.package_id[:32]}_{utterance_id[:12]}",
                    )
                entry["ok"] = bool(result.ok)
                entry["transcript"] = result.transcript
                entry["raw_transcript"] = result.raw_transcript
                entry["backend_name"] = result.backend_name
                entry["language"] = result.language
                entry["load_ms"] = load_ms
                entry["decode_ms"] = result.decode_ms
                if not result.ok:
                    entry["error"] = result.error
                    entry["error_code"] = result.error_code
            except ValueError as exc:
                # _transcriber_from_registry_package raises this for
                # packages whose metadata is incomplete (no model_path
                # AND no endpoint). Skip these gracefully.
                entry["ok"] = False
                entry["error_code"] = "rerun_route_unbuildable"
                entry["error"] = str(exc)
            except Exception as exc:
                entry["ok"] = False
                entry["error_code"] = "load_or_decode_failed"
                entry["error"] = str(exc)
            results.append(entry)

        self.recorder.record(
            TraceKind.SYSTEM,
            "broker_replay_all_finals",
            {
                "utterance_id": utterance_id,
                "package_count": len(results),
                "ok_count": sum(1 for r in results if r.get("ok")),
            },
        )
        return {
            "ok": True,
            "utterance_id": utterance_id,
            "results": results,
        }

    def _retention_search_roots(self) -> list[Path]:
        """All directories that may contain retained per-utterance WAVs.

        Two paths exist for legitimate reasons: the one-shot HTTP path
        retains under ``oneshot_pipeline.audio_save_dir`` for the
        per-utterance "replay through different model" feature, and the
        live streaming session retains under
        ``<workbench log_dir>/../audio/<session>/`` for full-session
        replay and Mac-shell reruns. Compare-all needs to consider both
        so a fresh live-mic dictation isn't masked by an older one-shot.
        """
        roots: list[Path] = []
        oneshot_dir = getattr(self.oneshot_pipeline, "audio_save_dir", None)
        if oneshot_dir is not None:
            roots.append(Path(oneshot_dir))
        # Live streaming session audio sits next to the workbench log dir
        # rather than inside it. Walking from log_dir.parent picks up
        # every active and historical session's audio sub-tree.
        try:
            live_root = Path(self.config.log_dir).parent / "audio"
            if live_root.exists():
                roots.append(live_root)
        except Exception:
            pass
        return [r for r in roots if r.exists()]

    def _candidate_wavs(self) -> list[Path]:
        """Every retained ``utt_*.wav`` across known retention roots,
        excluding our own ``compareall_*`` replay artifacts. Walks
        recursively so live-session sub-directories are picked up."""
        results: list[Path] = []
        for root in self._retention_search_roots():
            for wav in root.rglob("*.wav"):
                if wav.stem.startswith("compareall_"):
                    continue
                results.append(wav)
        return results

    def _most_recent_retained_audio(self) -> tuple[str, bytes] | None:
        """Find the most recently saved utterance WAV across all retention
        paths. Returns ``(utterance_id, wav_bytes)`` or ``None``."""
        wavs = self._candidate_wavs()
        if not wavs:
            return None
        wavs.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        path = wavs[0]
        try:
            return path.stem, path.read_bytes()
        except OSError:
            return None

    def _load_retained_audio(self, utterance_id: str) -> bytes | None:
        """Return WAV bytes for *utterance_id* from any retention path.

        First tries the one-shot pipeline's API (covers HTTP-ingested
        audio), then falls back to scanning the live-session retention
        tree for a matching filename."""
        wav_bytes = self.oneshot_pipeline.get_audio_for_rerun(utterance_id)
        if wav_bytes is not None:
            return wav_bytes
        for path in self._candidate_wavs():
            if path.stem == utterance_id:
                try:
                    return path.read_bytes()
                except OSError:
                    return None
        return None

    def broker_runtime_swap_final(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Swap the active final-ASR backend to a registered package id.

        Returns the SwapResult shape directly so the UI can branch on
        ``ok`` and ``error_code`` without unwrapping.
        """
        if self.final_swap is None:
            return {
                "ok": False,
                "error_code": "swap_not_available",
                "error": "no engine final backend wired into this workbench",
            }
        package_id = str(payload.get("package_id") or "").strip()
        if not package_id:
            return {
                "ok": False,
                "error_code": "package_id_required",
                "error": "package_id is required in the request body",
            }
        result = self.final_swap.swap_to(package_id)
        return result.to_dict()

    def broker_surface_policy(self) -> Dict[str, Any]:
        """Return surface support tiers for all known surfaces."""
        from juno_core_v3.policy.surface_gate import SurfaceCapabilityGate

        gate = SurfaceCapabilityGate()
        return {"ok": True, "policy": gate.policy_map()}

    def broker_surface_active(self) -> Dict[str, Any]:
        """Return the broker's current best-effort active surface snapshot."""
        try:
            app_name = self.store.state.app_name
            window_title = self.store.state.window_title
            writer_mode_field = self.store.state.writer_mode
        except Exception:
            app_name = ""
            window_title = ""
            writer_mode_field = ""
        mode = ""
        try:
            cur = self.broker_modes_current()
            sel = cur.get("selection") or {}
            if isinstance(sel, dict):
                mode = str(sel.get("mode_name") or sel.get("mode") or "")
        except Exception:
            mode = ""
        return {
            "ok": True,
            "active": {
                "app_name": app_name,
                "window_title": window_title,
                "mode": mode,
                "writer_mode_field": writer_mode_field,
            },
        }

    def broker_modes_builtin_list(self) -> Dict[str, Any]:
        # Issue #5: hide non-manually-selectable built-ins (e.g. ``default_surface``,
        # which is the AUTO fallback policy and must not be clickable as a manual mode).
        items = [
            {"id": n, "policy": BUILTIN_MODES[n].to_dict()}
            for n in builtin_mode_names()
            if BUILTIN_MODES[n].manual_selectable
        ]
        return {"ok": True, "modes": items}

    def broker_modes_custom_list(self) -> Dict[str, Any]:
        return {"ok": True, "modes": [m.to_dict() for m in self.custom_mode_store.list_modes()]}

    def broker_modes_current(self) -> Dict[str, Any]:
        from juno_v2.context.app_classifier import classify_app_category

        manual = self.store.state.manual_writer_mode
        custom = self.store.state.custom_writer_mode
        rec = self.custom_mode_store.get(custom) if custom else None
        # Issue #10: thread bundle_id from WorkbenchState so bundle-keyed
        # surface presets (the user's "Cursor" / "VS Code" overrides)
        # resolve correctly. Pre-fix both args were hardcoded None and
        # bundle-only presets silently never fired.
        bundle_id = self.store.state.app_bundle_id
        surface_cat = classify_app_category(
            self.store.state.app_name,
            self.store.state.window_title,
            app_bundle_id=bundle_id,
        )
        sel, pol, _preset = resolve_mode_with_surface_presets(
            manual_mode_name=manual,
            custom_mode_name=custom,
            custom_record=rec,
            surface_hint=surface_cat,
            surface_bundle_id=bundle_id,
            preset_store=self.surface_preset_store,
            custom_mode_store=self.custom_mode_store,
        )
        return {
            "ok": True,
            "selection": sel.to_dict(),
            "policy": pol.to_dict(),
            "writer_mode_field": self.store.state.writer_mode,
            "app_category": surface_cat,
        }

    def broker_surface_editing_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Classify frontmost surface for macOS HUD (juno-capability-shaped JSON)."""
        from juno_v2.workbench.editing_profile import surface_editing_profile

        return surface_editing_profile(payload)

    def broker_modes_manual_set(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(payload.get("mode") or "").strip()
        if not mode:
            return {"ok": False, "error": "mode_required"}
        # Issue #5: reject non-manually-selectable mode names (e.g. ``default_surface``)
        # and unknown ids so RPC callers cannot pin a foot-gun manual override.
        policy = BUILTIN_MODES.get(mode)
        if policy is None or not policy.manual_selectable:
            return {"ok": False, "error": "mode_not_selectable"}
        with self._lock:
            self.store.set_manual_writer_mode(mode)
            return {"ok": True, "snapshot": self.store.snapshot()}

    def broker_modes_manual_clear(self) -> Dict[str, Any]:
        with self._lock:
            self.store.clear_manual_writer_mode()
            return {"ok": True, "snapshot": self.store.snapshot()}

    def broker_modes_custom_set(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "name_required"}
        # Issue #23: optional rename hint. The editor sheet captures the
        # mode's name when it opened; if the user renames the mode it
        # sends ``previous_name`` so we can delete the orphan row in the
        # same lock as the upsert. Without this, the store keys by name
        # and would leave both rows around.
        previous_name = str(payload.get("previous_name") or "").strip()
        rec = CustomModeRecord(
            name=name,
            base_mode=str(payload.get("base_mode") or "default_surface"),
            description=str(payload.get("description") or ""),
            prompt_prefix=str(payload.get("prompt_prefix") or ""),
            itn_override=payload.get("itn_override"),
            cleanup_override=payload.get("cleanup_override"),
            style_card_name=payload.get("style_card_name"),
            snippet_scope=payload.get("snippet_scope"),
            command_policy=payload.get("command_policy"),
            auto_transform_id=payload.get("auto_transform_id"),
            enabled=bool(payload.get("enabled", True)),
        )
        with self._lock:
            if previous_name and previous_name.casefold() != name.casefold():
                self.custom_mode_store.delete(previous_name)
                # Keep the active custom selection pointing at the new
                # name when the renamed mode was active — otherwise the
                # next snapshot dangles on a now-deleted name.
                current_active = (self.store.state.custom_writer_mode or "").strip()
                if current_active.casefold() == previous_name.casefold():
                    self.store.set_custom_writer_mode(name)
            self.custom_mode_store.upsert(rec)
            return {"ok": True, "mode": rec.to_dict()}

    def broker_modes_custom_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "name_required"}
        with self._lock:
            ok = self.custom_mode_store.delete(name)
            return {"ok": ok, "deleted": ok}

    def broker_modes_custom_activate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        with self._lock:
            if name:
                self.store.set_custom_writer_mode(name)
            else:
                self.store.clear_custom_writer_mode()
            return {"ok": True, "snapshot": self.store.snapshot()}

    def broker_surface_presets_user(self) -> Dict[str, Any]:
        return {"ok": True, "presets": [p.to_dict() for p in self.surface_preset_store.list_user_presets()]}

    def broker_surface_presets_merged(self) -> Dict[str, Any]:
        return {"ok": True, "presets": [p.to_dict() for p in self.surface_preset_store.list_presets_merged()]}

    def broker_surface_presets_upsert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = SurfacePresetRecord.from_dict(payload)
        if not rec.id.strip() or not rec.bundle_id.strip():
            return {"ok": False, "error": "id_and_bundle_id_required"}
        with self._lock:
            self.surface_preset_store.upsert(rec)
            return {"ok": True, "preset": rec.to_dict()}

    def broker_surface_presets_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(payload.get("id") or "").strip()
        if not pid:
            return {"ok": False, "error": "id_required"}
        with self._lock:
            ok = self.surface_preset_store.delete(pid)
        # "remove this app's override" is idempotent: if the row is already
        # gone the user's intent is satisfied. Returning ok=True silences the
        # toast on a refresh-race (where the row was removed via another path
        # but userRows still cached the id). Real failures (network down,
        # store unavailable) raise before reaching here.
        if ok:
            return {"ok": True, "deleted": True}
        return {"ok": True, "deleted": False, "note": "preset_not_found_treated_as_success"}

    def broker_transforms_builtin_list(self) -> Dict[str, Any]:
        return {"ok": True, "transforms": [t.to_dict() for t in BUILTIN_CATALOG.values()]}

    def broker_transforms_custom_list(self) -> Dict[str, Any]:
        return {"ok": True, "transforms": [t.to_dict() for t in self.custom_transform_store.list_all()]}

    def broker_transforms_custom_upsert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = CustomTransformRecord(
            name=str(payload.get("name") or ""),
            instruction=str(payload.get("instruction") or ""),
            base_transform_id=payload.get("base_transform_id"),
            mode_constraints=tuple(str(x) for x in (payload.get("mode_constraints") or ())),
            enabled=bool(payload.get("enabled", True)),
        )
        if not rec.name or not rec.instruction:
            return {"ok": False, "error": "name_and_instruction_required"}
        with self._lock:
            self.custom_transform_store.upsert(rec)
            return {"ok": True, "transform": rec.to_dict()}

    def broker_transforms_custom_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "name_required"}
        with self._lock:
            return {"ok": self.custom_transform_store.delete(name)}

    # --- Transform (broker session class) ---

    def broker_transform(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        selected = str(payload.get("selected_text") or "")
        hint = str(payload.get("hint") or payload.get("transform_id") or "polish")
        app_category = payload.get("app_category")
        metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {}
        if payload.get("transform_id"):
            metadata["transform_id"] = str(payload.get("transform_id"))
        if payload.get("custom_transform_instruction"):
            metadata["custom_transform_instruction"] = str(payload.get("custom_transform_instruction"))
        if payload.get("transform_source"):
            metadata["transform_source"] = str(payload.get("transform_source"))
        signals = self._broker_signals_from_payload(payload, kind="transform")
        with self._lock:
            return self.broker.run_transform(
                selected_text=selected,
                hint=hint,
                app_category=str(app_category) if app_category else None,
                metadata=metadata or None,
                signals=signals,
            )

    def broker_personalization_summary(self) -> Dict[str, Any]:
        """Return a stable, product-facing view of personalization state.

        The summary is intentionally lightweight: counts plus short previews
        for each inspectable layer. We preserve the historical ``vocab`` key
        for compatibility, but also expose the clearer ``vocabulary`` name and
        include session entities so the inspector reflects what the runtime can
        actually bias on.
        """
        summary: Dict[str, Any] = {"ok": True, "layers": {}}
        if self.memory is None:
            summary["layers"] = {"note": "no_memory_store"}
            return summary
        try:
            vocab = self.memory.vocabulary.list() if hasattr(self.memory, "vocabulary") else []
            replacements = self.memory.replacements.list() if hasattr(self.memory, "replacements") else []
            snippets = self.memory.snippets.list() if hasattr(self.memory, "snippets") else []
            corrections = self.memory.corrections.list() if hasattr(self.memory, "corrections") else []
            entities = self.memory.entities.list() if hasattr(self.memory, "entities") else []

            vocabulary_layer = {
                "count": len(vocab),
                "preview": [v.term for v in vocab[:5]],
            }
            summary["layers"] = {
                "vocabulary": vocabulary_layer,
                "vocab": dict(vocabulary_layer),
                "replacements": {
                    "count": len(replacements),
                    "preview": [f"{r.trigger}→{r.replacement}" for r in replacements[:5]],
                },
                "snippets": {
                    "count": len(snippets),
                    "preview": [s.trigger for s in snippets[:5]],
                },
                "corrections": {
                    "count": len(corrections),
                    "preview": [f"{c.observed}→{c.corrected}" for c in corrections[:5]],
                },
                "entities": {
                    "count": len(entities),
                    "preview": [e.value for e in entities[:5]],
                },
            }
        except Exception as exc:
            summary["ok"] = False
            summary["error"] = str(exc)
        return summary

    # --- One-shot dictation (Mac shell) ---

    def broker_capability(self) -> Dict[str, Any]:
        """Run the Mac capability probe and return ``{ok, reason, ...}``.

        The shell calls this right before opening the mic. A ``ok=False``
        result carries a stable ``reason`` code (``secure_field``,
        ``app_blocked``, ``ax_permission_missing``,
        ``helper_not_installed``) so the shell can branch on the kind
        of failure.

        This endpoint is the *refuse to paste into a password field*
        safety net. Skipping it because it "always says ok" is how a
        voice tool ends up transcribing someone's banking PIN into a
        browser history.

        The response also includes ``recognition_hints`` — a list of
        domain-specific terms (identifiers, filenames, lexicon entries)
        that the shell passes into the local preview/final pipeline so
        transcription is biased toward the user's current surface vocabulary.
        """
        decision = self.capability.decide()
        self.recorder.record(
            TraceKind.SYSTEM,
            "capability_probe",
            {
                "ok": decision.ok,
                "reason": decision.reason,
                "bundle_id": decision.report.frontmost_app_bundle_id,
            },
        )
        result = decision.to_dict()
        result["recognition_hints"] = self._build_recognition_hints(decision.report.raw)
        return result

    def broker_context_inspect(self) -> Dict[str, Any]:
        """Return the current context packet + provenance summary."""
        from juno_v2.contracts.context import TypedContextBundle

        with self._lock:
            snap = self.store.snapshot()

        bundle = TypedContextBundle(
            selected_text=snap.get("selected_text", ""),
            focused_text_before=snap.get("focused_text_before", ""),
            focused_text_after=snap.get("focused_text_after", ""),
            clipboard_text=snap.get("clipboard_text", ""),
            field_text_excerpt=snap.get("field_text_excerpt", ""),
            app_name=snap.get("app_name"),
            window_title=snap.get("window_title"),
        )
        return self.broker.context_inspect(bundle)

    def _build_recognition_hints(self, raw_probe: Dict[str, Any]) -> list[str]:
        """Derive recognition-hint strings from the capability probe payload + memory.

        The list is intentionally small (≤20 entries) and composed of:
        1. Technical identifiers extracted from the focused text / selected text.
        2. The user's top lexicon terms and session entities from memory.

        The local preview/final pipeline uses these strings as surface
        vocabulary evidence for domain-specific terms before final formatting.
        """
        from juno_v2.context.provider import _extract_candidates

        hints: list[str] = []
        seen: set[str] = set()

        def _add(val: str) -> None:
            v = (val or '').strip()
            if not v or len(v) > 80 or len(hints) >= 20:
                return
            key = v.casefold()
            if key in seen:
                return
            seen.add(key)
            hints.append(v)

        # Derive technical identifiers from the AX text fields.
        text_chunks = [
            str(raw_probe.get('selected_text') or ''),
            str(raw_probe.get('focused_text_before') or ''),
            str(raw_probe.get('focused_text_after') or ''),
            str(raw_probe.get('window_title') or ''),
        ]
        for candidate in _extract_candidates(text_chunks):
            _add(candidate)

        # Add the user's top memory terms when available.
        if self.memory is not None:
            try:
                snap = self.memory.snapshot()
                for entry in sorted(snap.lexicon, key=lambda e: -float(e.boost))[:8]:
                    _add(entry.canonical_form)
                for entity in sorted(snap.session_entities, key=lambda e: -int(e.count))[:6]:
                    _add(entity.value)
            except Exception:
                pass

        return hints

    def _run_oneshot_with_policy(
        self,
        wav_bytes: bytes,
        *,
        language: str | None,
        app_bundle_id: str | None,
        window_title_hint: str | None,
        utterance_id: str | None,
        policy: Dict[str, Any],
        frozen_context: Dict[str, Any] | None = None,
        transcript_stage: str | None = None,
        session_context_tape: Dict[str, Any] | list[Any] | None = None,
        transcript_hint: str | None = None,
        save_history: bool = True,
        save_audio: bool = True,
        language_mode: str | None = None,
        pause_sensitivity_seconds: float | None = None,
        broker_session_id: str | None = None,
    ):
        """Route the one-shot Insert path through the v3 ``InsertRunner``.

        B2 migration: the writer-disable policy lever and the pipeline
        invocation used to live inline here; both now belong to the
        runner. The HTTP / trace wire format is preserved because
        ``InsertResult.to_dict()`` is a passthrough of the underlying
        :class:`OneShotDictationResult.to_dict()`. Return value of this
        method stays the pipeline result (accessed via ``.raw``) so
        callers that reach into ``result.to_dict`` keep working.
        """
        degraded_features = list(policy.get("degraded_features") or [])
        reduce_optional_lanes = any(
            feature in {"reduce_optional_lanes", "writer_lane_budget_tight"}
            for feature in degraded_features
        )
        with self._lock:
            manual_wm = self.store.state.manual_writer_mode
            custom_wm = self.store.state.custom_writer_mode
        runner = InsertRunner(pipeline=self.oneshot_pipeline)
        result = runner.run(
            InsertRequest(
                wav_bytes=wav_bytes,
                language=language,
                app_bundle_id=app_bundle_id,
                window_title_hint=window_title_hint,
                utterance_id=utterance_id,
                manual_writer_mode=manual_wm,
                custom_writer_mode=custom_wm,
                reduce_optional_lanes=reduce_optional_lanes,
                frozen_context=frozen_context,
                transcript_stage=transcript_stage or "final_delivery",
                session_context_tape=session_context_tape,
                transcript_hint=transcript_hint,
                save_history=save_history,
                save_audio=save_audio,
                language_mode=language_mode,
                pause_sensitivity_seconds=pause_sensitivity_seconds,
                broker_session_id=broker_session_id,
            )
        )
        # Callers still consume ``result.to_dict()`` — keep handing them
        # the underlying pipeline result so the wire format and the
        # ``.utterance_id`` / ``.ok`` attribute access patterns work
        # unchanged.
        return result.raw

    def _cancelled_transcribe_response(
        self,
        utterance_id: str,
        *,
        live_adjudication: bool,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "utterance_id": utterance_id,
            "transcript": "",
            "stage": "live_adjudication" if live_adjudication else "final_delivery",
            "transcript_stage": "live_adjudication" if live_adjudication else "final_delivery",
            "error": reason or "inference_cancelled",
            "cancelled": True,
            "metadata": {
                "scheduler": {
                    "cancelled": True,
                    "reason": reason or "inference_cancelled",
                    **self._inference_scheduler.stats(),
                }
            },
        }

    def _record_utterance_lifecycle(self, utterance_id: str | None, payload: Dict[str, Any]) -> None:
        uid = str(utterance_id or "").strip()
        if not uid:
            return
        event = dict(payload)
        event.setdefault("utterance_id", uid)
        event.setdefault("recorded_at_ns", time.perf_counter_ns())
        try:
            self._utterance_lifecycle_executor.submit(
                self._write_utterance_lifecycle_event,
                uid,
                event,
            )
        except Exception:
            logger.exception("failed_to_submit_utterance_lifecycle")

    def _write_utterance_lifecycle_event(self, uid: str, event: Dict[str, Any]) -> None:
        safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)[:120]
        root = Path(self.config.runtime_dir) if self.config.runtime_dir is not None else Path(self.config.log_dir)
        lifecycle_dir = root / "utterance_lifecycle"
        try:
            with self._utterance_lifecycle_guard:
                lifecycle_dir.mkdir(parents=True, exist_ok=True)
                path = lifecycle_dir / f"{safe_uid}.json"
                doc: Dict[str, Any]
                if path.exists():
                    try:
                        loaded = json.loads(path.read_text(encoding="utf-8"))
                        doc = loaded if isinstance(loaded, dict) else {}
                    except Exception:
                        doc = {}
                else:
                    doc = {}
                events = doc.get("events")
                if not isinstance(events, list):
                    events = []
                events.append(event)
                doc["utterance_id"] = uid
                doc["updated_at_ns"] = event["recorded_at_ns"]
                doc["events"] = events[-200:]
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
                tmp.replace(path)
            self.recorder.record(TraceKind.SYSTEM, "utterance_lifecycle_v1", event)
        except Exception:
            logger.exception("failed_to_record_utterance_lifecycle")

    @staticmethod
    def _sanitize_shell_timeline(raw: Any) -> Dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        timeline: Dict[str, Any] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            safe_key = key.strip()
            if not safe_key or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", safe_key):
                continue
            if isinstance(value, bool):
                timeline[safe_key] = value
            elif isinstance(value, (int, float)):
                timeline[safe_key] = value
            elif isinstance(value, str) and len(value) <= 200:
                timeline[safe_key] = value
        return timeline or None

    def broker_dictation_transcribe(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        app_bundle_id: str | None = None,
        window_title_hint: str | None = None,
        utterance_id: str | None = None,
        surface_id: str | None = None,
        host_hints: HostResourceHints | None = None,
        frozen_context: Dict[str, Any] | None = None,
        transcript_stage: str | None = None,
        session_context_tape: Dict[str, Any] | list[Any] | None = None,
        transcript_hint: str | None = None,
        pause_sensitivity_seconds: float | None = None,
        shell_timeline: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """One-shot dictation endpoint driven by the full engine pipeline.

        Insert dictation must now be broker-governed. If there is already an
        active INSERT session we reuse it. Otherwise we start one from the
        caller-provided surface metadata. When ``surface_id`` is absent but
        ``app_bundle_id`` is present (normal Mac shell dictation), default to
        ``mac_overlay`` — not ``workbench_dev`` (product repair 2026-04).
        """
        request_received_ns = time.perf_counter_ns()
        resolved_surface = resolve_dictation_surface_id(surface_id, app_bundle_id)
        frozen_context, privacy_summary = self._sanitize_frozen_context(
            frozen_context,
            app_bundle_id=app_bundle_id,
        )
        stage = str(transcript_stage or "final_delivery").strip().lower() or "final_delivery"
        live_adjudication = stage in {"live", "adjudication", "live_correction", "live_adjudication"}
        shell_timeline = self._sanitize_shell_timeline(shell_timeline)
        language_mode = language or str(self._settings.get("language_mode") or "auto")
        if language_mode == "auto":
            # Keep Auto as a real ASR auto-detect request. The backends still
            # apply the existing silence / low-confidence guards, but forcing
            # English here erased Hindi or mixed-language source text before
            # History could show what the user actually spoke.
            language_for_asr = None
        elif language_mode == "keep_original":
            language_for_asr = None
        elif language_mode.startswith("pair:"):
            language_for_asr = None
        else:
            language_for_asr = language_mode
        signals = None
        if self.broker.broker_session is None or self.broker.broker_session.kind.value != "insert":
            signals = UserIntentSignals(
                explicit_insert=True,
                surface_id=resolved_surface,
                host_hints=host_hints,
            )
        environment_profile = str(self._settings.get("environment_profile") or "auto")
        with self._lock:
            signals = None
            if self.broker.broker_session is None or self.broker.broker_session.kind.value != "insert":
                signals = UserIntentSignals(
                    explicit_insert=True,
                    surface_id=resolved_surface,
                    host_hints=host_hints,
                )
            ensured = self.broker.ensure_session(kind=SessionKind.INSERT, signals=signals)
            if not ensured.get("ok"):
                return ensured
            policy = dict(ensured.get("policy") or {})
            broker_session_id = (
                self.broker.broker_session.session_id
                if self.broker.broker_session is not None
                else None
            )

        uid = str(utterance_id or uuid.uuid4().hex)
        scheduler_stage = "live_adjudication" if live_adjudication else "final_stop_delivery"
        scheduler_priority = (
            InferenceScheduler.LIVE_PRIORITY
            if live_adjudication
            else InferenceScheduler.FINAL_PRIORITY
        )

        def _run_pipeline():
            return self._run_oneshot_with_policy(
                wav_bytes,
                language=language_for_asr,
                app_bundle_id=app_bundle_id,
                window_title_hint=window_title_hint,
                utterance_id=uid,
                policy=policy,
                frozen_context=frozen_context,
                transcript_stage="live_adjudication" if live_adjudication else stage,
                session_context_tape=session_context_tape,
                transcript_hint=transcript_hint,
                save_history=False if live_adjudication else bool(privacy_summary.get("save_history", True)),
                save_audio=False if live_adjudication else bool(privacy_summary.get("save_audio", True)),
                language_mode=language_mode,
                pause_sensitivity_seconds=pause_sensitivity_seconds,
                broker_session_id=broker_session_id,
            )

        scheduler_future = self._inference_scheduler.submit(
            stage=scheduler_stage,
            utterance_id=uid,
            priority=scheduler_priority,
            fn=_run_pipeline,
        )
        try:
            scheduled = scheduler_future.result()
            result = scheduled.result
        except (InferenceJobCancelled, CancelledError) as exc:
            out = self._cancelled_transcribe_response(
                uid,
                live_adjudication=live_adjudication,
                reason=str(exc) or type(exc).__name__,
            )
            self._record_utterance_lifecycle(
                uid,
                {
                    "event": "broker_dictation_transcribe_cancelled",
                    "stage": scheduler_stage,
                    "request_received_ns": request_received_ns,
                    "shell_timeline": shell_timeline,
                    "reason": str(exc) or type(exc).__name__,
                    **self._inference_scheduler.stats(),
                },
            )
            return out
        out = result.to_dict()
        out.setdefault("privacy", privacy_summary)
        metadata_for_scheduler = out.setdefault("metadata", {})
        if not isinstance(metadata_for_scheduler, dict):
            metadata_for_scheduler = {}
            out["metadata"] = metadata_for_scheduler
        metadata_for_scheduler["scheduler"] = {
            "stage": scheduler_stage,
            "queue_wait_ms": scheduled.queue_wait_ms,
            "worker_service_ms": scheduled.worker_service_ms,
            "mlx_lock_wait_ms": scheduled.mlx_lock_wait_ms,
            "request_received_ns": request_received_ns,
            "scheduler_enqueued_ns": scheduled.scheduler_enqueued_ns,
            "scheduler_started_ns": scheduled.scheduler_started_ns,
            "scheduler_finished_ns": scheduled.scheduler_finished_ns,
            "shell_timeline": shell_timeline,
            **self._inference_scheduler.stats(),
        }
        if shell_timeline:
            metadata_for_scheduler["shell_timeline"] = shell_timeline
        self._record_utterance_lifecycle(
            uid,
            {
                "event": "broker_dictation_transcribe",
                "stage": scheduler_stage,
                "request_received_ns": request_received_ns,
                "shell_timeline": shell_timeline,
                "scheduler_enqueued_ns": scheduled.scheduler_enqueued_ns,
                "scheduler_started_ns": scheduled.scheduler_started_ns,
                "scheduler_finished_ns": scheduled.scheduler_finished_ns,
                "scheduler_queue_wait_ms": scheduled.queue_wait_ms,
                "scheduler_worker_service_ms": scheduled.worker_service_ms,
                "mlx_lock_wait_ms": scheduled.mlx_lock_wait_ms,
                "prompt_chars": metadata_for_scheduler.get("prompt_chars"),
                "output_tokens": metadata_for_scheduler.get("output_tokens")
                or metadata_for_scheduler.get("output_tokens_estimate"),
                **self._inference_scheduler.stats(),
            },
        )
        degraded_features = list((ensured.get("policy") or {}).get("degraded_features") or [])
        out.setdefault("broker_session", ensured.get("broker_session"))
        out.setdefault("policy", ensured.get("policy"))
        out.setdefault("support_level", (ensured.get("policy") or {}).get("support_level"))
        out["degraded"] = bool(degraded_features)
        if degraded_features:
            out["degraded_reason"] = degraded_features[0]
            out["degraded_features"] = degraded_features

        # Audit Issue #1 — preview→final reconciliation. When the shell sent
        # us a preview snapshot (``transcript_hint``) and we have a non-empty
        # final transcript, attach a final-stage ``transcript_patch_v1``
        # envelope so the HUD can reconcile in place via the existing patch
        # path instead of a hard cut. The legacy ``applyFinalText`` fallback
        # in the shell still handles responses without the envelope, so this
        # is fully backwards-compatible with older builds.
        if not live_adjudication:
            preview_snapshot = (transcript_hint or "").strip()
            final_transcript = str(out.get("transcript") or "")
            if preview_snapshot and final_transcript:
                from juno_v2.transcript.final_reconciliation import (
                    build_final_patch_envelope,
                )

                envelope = build_final_patch_envelope(
                    preview_text=preview_snapshot,
                    final_text=final_transcript,
                    utterance_id=str(out.get("utterance_id") or utterance_id or ""),
                )
                metadata_out = out.setdefault("metadata", {})
                if not isinstance(metadata_out, dict):
                    metadata_out = {}
                    out["metadata"] = metadata_out
                # The live-adjudication path may have already attached its
                # own ``transcript_patch`` to ``metadata`` earlier in the
                # pipeline. We deliberately do not run for live_adjudication
                # (guarded above) so there is no live-vs-final overwrite to
                # reason about here.
                metadata_out["transcript_patch"] = envelope.to_dict()

        return out

    def broker_dictation_live_correct(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        visible = str(
            payload.get("visible_text")
            or payload.get("transcript_hint")
            or payload.get("text")
            or ""
        ).strip()
        if not visible:
            return {
                "ok": False,
                "error": "empty_visible_text",
                "error_code": "empty_visible_text",
                "transcript": "",
                "raw_transcript": "",
                "metadata": {"live_correction": True, "rejected": True},
            }
        if not self._live_corrector_ready_for_live_request():
            return {
                "ok": True,
                "transcript": visible,
                "raw_transcript": visible,
                "metadata": {
                    "live_correction": True,
                    "live_correction_skipped": True,
                    "reason": "live_corrector_model_not_cached",
                },
            }
        host_hints = None
        if isinstance(payload.get("host_hints"), dict):
            host_hints = HostResourceHints.from_dict(payload["host_hints"])
        out = self.broker_dictation_transcribe(
            b"live-text-correction",
            language=payload.get("language") or payload.get("language_mode"),
            app_bundle_id=payload.get("app_bundle_id"),
            window_title_hint=payload.get("window_title_hint") or payload.get("window_title"),
            utterance_id=payload.get("utterance_id"),
            surface_id=payload.get("surface_id") or "mac_overlay",
            host_hints=host_hints,
            frozen_context=payload.get("frozen_context") if isinstance(payload.get("frozen_context"), dict) else None,
            transcript_stage="live_adjudication",
            session_context_tape=payload.get("session_context_tape")
            if isinstance(payload.get("session_context_tape"), (dict, list))
            else None,
            transcript_hint=visible,
            pause_sensitivity_seconds=payload.get("pause_sensitivity_seconds")
            if isinstance(payload.get("pause_sensitivity_seconds"), (int, float))
            else None,
            shell_timeline=payload.get("shell_timeline") if isinstance(payload.get("shell_timeline"), dict) else None,
        )
        if isinstance(out, dict):
            meta = out.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                out["metadata"] = meta
            meta["live_correction"] = True
            meta["live_correction_skipped"] = False
        return out

    def _live_corrector_ready_for_live_request(self) -> bool:
        svc = getattr(self, "live_corrector_service", None)
        if svc is None:
            return False
        backend_obj = getattr(svc, "backend", None)
        if backend_obj is not None and bool(getattr(backend_obj, "_loaded", False)):
            return True
        profile = dict(getattr(self, "deployment_profile", {}) or {})
        backend = str(profile.get("live_corrector_backend") or "").strip().lower()
        model_path = str(profile.get("live_corrector_model_path") or "").strip()
        if backend != "mlx_lm" or not model_path:
            return True
        try:
            from juno_v2.demo.models import is_hf_model_cached

            return bool(is_hf_model_cached(model_path))
        except Exception:
            return False

    # --- Post-paste correction learning (Mac textmon) ---

    def broker_insertion_committed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Record that the shell committed a transcript via undo-safe paste.

        This gives us a trace event that pairs with the preceding
        ``dictation_transcribed`` record, so when the text-monitor later
        reports a correction we can attribute it to an utterance. The
        payload is intentionally minimal — no transcript text (we
        already have it in the trace), just length and target PID plus
        whether the paste syscall returned ok.
        """
        transcript_len = int(payload.get("transcript_len") or 0)
        target_pid = int(payload.get("target_pid") or 0)
        ok = bool(payload.get("ok", True))
        transcript_text = payload.get("transcript") or payload.get("text")
        utterance_id = payload.get("utterance_id")
        if utterance_id is not None:
            utterance_id = str(utterance_id)
        paste_kind = payload.get("paste_kind")
        noop_reason = payload.get("noop_reason")
        paste_failure_reason = payload.get("paste_failure_reason") or payload.get("failure_reason")
        trigger_source = payload.get("trigger_source")
        shell_timeline = self._sanitize_shell_timeline(payload.get("shell_timeline"))
        ins_payload: Dict[str, Any] = {
            "transcript_len": transcript_len,
            "target_pid": target_pid,
            "ok": ok,
            "undo_safe": True,
            "utterance_id": utterance_id,
        }
        if shell_timeline:
            ins_payload["shell_timeline"] = shell_timeline
        if isinstance(paste_kind, str) and paste_kind.strip():
            ins_payload["paste_kind"] = paste_kind.strip()
        if isinstance(noop_reason, str) and noop_reason.strip():
            ins_payload["noop_reason"] = noop_reason.strip()
        if isinstance(paste_failure_reason, str) and paste_failure_reason.strip():
            ins_payload["paste_failure_reason"] = paste_failure_reason.strip()
        if isinstance(trigger_source, str) and trigger_source.strip():
            ins_payload["trigger_source"] = trigger_source.strip()
        for src_key, dst_key in (
            ("paste_attempted", "paste_attempted"),
            ("likely_paste_destination", "likely_paste_destination"),
        ):
            if src_key in payload:
                ins_payload[dst_key] = bool(payload.get(src_key))
        for src_key, dst_key in (
            ("target_bundle_id", "target_bundle_id"),
            ("target_app_bundle_id", "target_bundle_id"),
            ("app_bundle_id", "target_bundle_id"),
            ("target_app_name", "target_app_name"),
            ("app_name", "target_app_name"),
            ("target_window_title", "target_window_title"),
            ("window_title", "target_window_title"),
        ):
            value = payload.get(src_key)
            if isinstance(value, str) and value.strip() and dst_key not in ins_payload:
                ins_payload[dst_key] = value.strip()
        if ok and isinstance(transcript_text, str) and transcript_text.strip():
            ins_payload["transcript"] = transcript_text
        # Focus-drift diagnostic from the macOS shell (PR-B): when the
        # user switches apps between the hotkey press and the actual
        # ``juno-paste`` invocation, ``paste_frontmost_pid`` (sampled
        # immediately before the CGEvent.post) won't match
        # ``target_pid`` (recorded at hotkey time). juno-paste exits 0
        # in either case so this is currently the only post-hoc signal
        # that the keystroke probably landed in the wrong destination.
        paste_frontmost_pid = payload.get("paste_frontmost_pid")
        if isinstance(paste_frontmost_pid, (int, float)) and paste_frontmost_pid:
            ins_payload["paste_frontmost_pid"] = int(paste_frontmost_pid)
        paste_frontmost_drifted = bool(payload.get("paste_frontmost_drifted") or False)
        if paste_frontmost_drifted:
            ins_payload["paste_frontmost_drifted"] = True
            logger.warning(
                "insertion_committed paste_frontmost_drifted utterance_id=%s "
                "target_pid=%s paste_frontmost_pid=%s — keystroke likely landed "
                "in a different app than the one targeted at hotkey time",
                utterance_id,
                target_pid,
                int(paste_frontmost_pid) if isinstance(paste_frontmost_pid, (int, float)) else None,
            )
        self.recorder.record(TraceKind.SYSTEM, "insertion_committed", ins_payload)
        if utterance_id:
            self._record_utterance_lifecycle(
                utterance_id,
                {
                    "event": "shell_insertion_committed",
                    "ok": ok,
                    "paste_kind": ins_payload.get("paste_kind"),
                    "paste_attempted": ins_payload.get("paste_attempted"),
                    "paste_failure_reason": ins_payload.get("paste_failure_reason"),
                    "shell_timeline": shell_timeline,
                },
            )
            try:
                from juno_v2.observability.product_history import get_product_history_store

                history_context: Dict[str, Any] = {}
                for dst_key, *src_keys in (
                    ("app_bundle_id", "target_bundle_id", "target_app_bundle_id", "app_bundle_id"),
                    ("app_name", "target_app_name", "app_name"),
                    ("window_title", "target_window_title", "window_title"),
                ):
                    for src_key in src_keys:
                        value = payload.get(src_key)
                        if isinstance(value, str) and value.strip():
                            history_context[dst_key] = value.strip()
                            break
                get_product_history_store(Path(self.config.log_dir)).update_insertion_commit(
                    utterance_id=utterance_id,
                    committed_text=transcript_text if isinstance(transcript_text, str) else None,
                    ok=ok,
                    paste_kind=paste_kind if isinstance(paste_kind, str) else None,
                    failure_reason=paste_failure_reason if isinstance(paste_failure_reason, str) else None,
                    context=history_context,
                )
            except Exception as exc:  # noqa: BLE001
                self.recorder.record(
                    TraceKind.SYSTEM,
                    "insertion_committed_history_error",
                    {"error": str(exc), "utterance_id": utterance_id},
                )
        learn_summary: Dict[str, Any] = {"learned": False, "reason": "not_attempted"}
        # Feed the clipboard ring buffer: every undo-safe paste also
        # counts as a clipboard write from juno's perspective, so the
        # writer / tools can reference the most recent insertions as
        # context on the next turn. We only push when the shell
        # actually succeeded (ok=True) and forwarded the transcript.
        if ok and isinstance(transcript_text, str) and transcript_text.strip():
            try:
                self.clipboard_ring.push(transcript_text)
            except Exception:
                pass
            # Drive correction learning + session-entity upserts through
            # the one-shot pipeline. The pipeline matches the commit
            # against its retained utterance record (by ``utterance_id``
            # when the shell echoed it, otherwise "most recent") and
            # hits ``memory_store.record_correction`` /
            # ``upsert_session_entities`` so future decodes benefit.
            try:
                learn_summary = self.oneshot_pipeline.record_insertion(
                    utterance_id=utterance_id,
                    committed_text=transcript_text,
                )
            except Exception as exc:  # noqa: BLE001
                self.recorder.record(
                    TraceKind.MEMORY,
                    "insertion_committed_learn_error",
                    {"error": str(exc), "utterance_id": utterance_id},
                )
                learn_summary = {"learned": False, "reason": f"error:{exc}"}
        if utterance_id:
            paste_bundle = payload.get("paste_app_bundle_id")
            paste_name = payload.get("paste_app_name")
            paste_title = payload.get("paste_window_title")
            if (
                (isinstance(paste_bundle, str) and paste_bundle.strip())
                or (isinstance(paste_name, str) and paste_name.strip())
                or (isinstance(paste_title, str) and paste_title.strip())
            ):
                try:
                    from juno_v2.observability.product_history import get_product_history_store

                    store = get_product_history_store(Path(self.config.log_dir))
                    pb = (
                        str(paste_bundle).strip()
                        if isinstance(paste_bundle, str) and paste_bundle.strip()
                        else None
                    )
                    pn = (
                        str(paste_name).strip()
                        if isinstance(paste_name, str) and paste_name.strip()
                        else None
                    )
                    pt = (
                        str(paste_title).strip()
                        if isinstance(paste_title, str) and paste_title.strip()
                        else None
                    )
                    store.patch_paste_destination(
                        utterance_id,
                        app_bundle_id=pb,
                        app_name=pn,
                        window_title=pt,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.recorder.record(
                        TraceKind.SYSTEM,
                        "insertion_committed_history_patch_error",
                        {"error": str(exc), "utterance_id": utterance_id},
                    )
        return {"ok": True, "learning": learn_summary}

    def _normalize_correction_payload(self, payload: Dict[str, Any]) -> tuple[str, str]:
        """Normalize correction payload naming at the endpoint boundary.

        We accept the older ``expected`` naming for compatibility, but make the
        runtime semantics explicit internally: ``observed`` is what ASR produced
        and ``corrected`` is what the user intended / kept.
        """
        observed = str(
            payload.get("observed")
            or payload.get("raw")
            or payload.get("heard")
            or ""
        ).strip()
        corrected = str(
            payload.get("corrected")
            or payload.get("expected")
            or payload.get("committed")
            or payload.get("final")
            or ""
        ).strip()
        return observed, corrected

    def broker_observe_correction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        observed, corrected = self._normalize_correction_payload(payload)
        if not corrected or not observed or corrected == observed:
            return {"ok": False, "error": "nothing_to_learn"}
        # Issue #24: when the source is the post-paste AX monitor, the
        # text we receive is the *whole* focused-field value at the time
        # of the edit — surrounding greeting/signature included. Diff
        # against the actually-pasted segment instead so corrections
        # capture the user's edit, not unrelated context. Other sources
        # (live transcript adjudication, manual reprocess) already pass
        # tight observed/corrected pairs and skip this branch.
        source = str(payload.get("source") or "").strip().lower()
        if source == "mac_post_paste_monitor":
            from juno_v2.memory.correction_diff import diff_pasted_segment

            diffed = diff_pasted_segment(expected=observed, observed=corrected)
            if diffed is None:
                # Pasted segment isn't present in the post-edit field —
                # the user retyped or moved focus. Don't fabricate a
                # whole-field correction from unrelated content.
                return {"ok": False, "error": "paste_not_located"}
            observed, corrected = diffed
            if not corrected or not observed or corrected == observed:
                return {"ok": False, "error": "nothing_to_learn"}
        app_bundle_id = str(payload.get("app_bundle_id") or "").strip()
        if not self._app_override_bool(
            app_bundle_id,
            "learn",
            bool(self._settings.get("learn_from_corrections", True)),
        ):
            return {"ok": False, "error": "learning_disabled"}
        if self.memory is None:
            return {"ok": False, "error": "memory_not_configured"}
        with self._lock:
            # observed = what ASR produced (wrong), corrected = what user intended (correct)
            self.memory.record_correction(observed, corrected)
        self.recorder.record(
            TraceKind.SYSTEM,
            "correction_observed",
            {
                "source": str(payload.get("source") or "unknown"),
                "observed_chars": len(observed),
                "corrected_chars": len(corrected),
                # Backward-compatible alias for older tooling that still expects
                # the historical payload name.
                "expected_chars": len(corrected),
            },
        )
        uid = str(payload.get("utterance_id") or "").strip()
        if uid:
            try:
                from juno_v2.observability.product_history import increment_correction_count

                increment_correction_count(Path(self.config.log_dir), uid)
            except Exception:
                pass
        return {"ok": True, "observed": observed, "corrected": corrected}

    # --- Personalization memory (vocabulary / replacement / snippet /
    # style card / correction CRUD) -----------------------------------
    # Local clients need first-class read/write access to the user's memory. These
    # endpoints are JSON in, JSON out; every mutation returns the
    # updated list so callers can reflect the change without a second
    # GET. Every handler is a no-op when ``self.memory`` is ``None``
    # (standalone workbench without a memory store on disk).

    def _memory_required(self) -> Dict[str, Any] | None:
        if self.memory is None:
            return {"ok": False, "error": "memory_not_configured"}
        return None

    def broker_memory_snapshot(self) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        with self._lock:
            snap = self.memory.snapshot()
        return {
            "ok": True,
            "counts": {
                "lexicon": len(snap.lexicon),
                "replacements": len(snap.replacements),
                "corrections": len(snap.corrections),
                "session_entities": len(snap.session_entities),
                "snippets": len(self.memory.snippets.raw()),
            },
        }

    def broker_memory_clear_all(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        with self._lock:
            counts = self.memory.clear_all()
        return {"ok": True, **counts}

    def broker_memory_vocab_list(self) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        with self._lock:
            return {"ok": True, "entries": self.memory.vocabulary.raw()}

    def broker_memory_vocab_upsert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        from juno_v2.memory.fold import fold_key
        from juno_v2.memory.term_policy import learned_term_allowed

        term = str(payload.get("term") or "").strip()
        if not term:
            return {"ok": False, "error": "term_required"}
        canonical = str(payload.get("canonical_form") or term).strip() or term
        if not learned_term_allowed(term) or not learned_term_allowed(canonical):
            with self._lock:
                entries = self.memory.vocabulary.raw()
            return {
                "ok": False,
                "error": "term_too_short",
                "error_code": "term_too_short",
                "min_chars": 3,
                "entries": entries,
            }
        if fold_key(term) == "juno":
            if term != "Juno" or canonical != "Juno":
                with self._lock:
                    entries = self.memory.vocabulary.raw()
                return {
                    "ok": False,
                    "error": "protected_term",
                    "error_code": "protected_term",
                    "entries": entries,
                }
        # Conflict detection runs through fold_key so "Q.B.R", "Q B R", and
        # "qbr" all collide on the same row instead of stacking three
        # near-duplicate lexicon entries that fight in the bias plan.
        term_key = fold_key(term)
        canonical_key = fold_key(canonical)
        with self._lock:
            entries = self.memory.vocabulary.raw()
            for entry in entries:
                existing_canonical = str(
                    entry.get("canonical_form") or entry.get("term") or ""
                ).strip()
                existing_canonical_key = fold_key(existing_canonical)
                existing_terms = {
                    fold_key(str(entry.get("term") or "")),
                    existing_canonical_key,
                }
                existing_terms.update(
                    fold_key(str(alias))
                    for alias in (entry.get("aliases") or [])
                    if str(alias).strip()
                )
                existing_terms.discard("")
                if not term_key or not canonical_key:
                    continue
                if term_key not in existing_terms and canonical_key not in existing_terms:
                    continue
                if existing_canonical_key == canonical_key and term_key in existing_terms:
                    return {
                        "ok": True,
                        "skipped": True,
                        "reason": "already_known",
                        "entries": entries,
                    }
                if existing_canonical_key != canonical_key:
                    return {
                        "ok": False,
                        "error": "vocab_conflict",
                        "error_code": "vocab_conflict",
                        "existing": entry,
                        "entries": entries,
                    }
                break
        with self._lock:
            self.memory.add_lexicon_entry(
                term=term,
                canonical_form=canonical,
                aliases=list(payload.get("aliases") or []),
                pronunciation_hint=payload.get("pronunciation_hint"),
                boost=float(payload.get("boost") or 1.0),
                source=str(payload.get("source") or "user_edit"),
            )
            entries = self.memory.vocabulary.raw()
        return {"ok": True, "entries": entries}

    def broker_memory_vocab_remove(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        from juno_v2.memory.fold import fold_key

        term = str(payload.get("term") or "").strip()
        if not term:
            return {"ok": False, "error": "term_required"}
        if fold_key(term) == "juno":
            with self._lock:
                entries = self.memory.vocabulary.raw()
            return {
                "ok": False,
                "error": "protected_term",
                "error_code": "protected_term",
                "entries": entries,
            }
        with self._lock:
            removed = self.memory.remove_lexicon_entry(term)
            entries = self.memory.vocabulary.raw()
        return {"ok": True, "removed": removed, "entries": entries}

    def broker_shell_home_greeting(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Short writer-generated lines for the macOS shell Home hero."""
        if not bool(self._settings.get("writer_enabled", True)):
            return {"ok": False, "error": "writer_disabled", "error_code": "writer_disabled"}
        ws = getattr(self, "writer_service", None)
        raw_backend = getattr(ws, "backend", None) if ws is not None else None
        backend = ws if ws is not None and raw_backend is not None else None
        if backend is None:
            return {"ok": False, "error": "writer_unavailable", "error_code": "writer_unavailable"}
        if getattr(raw_backend, "_loaded", True) is False:
            return {"ok": False, "error": "writer_cold", "error_code": "writer_cold"}
        hour = int(payload.get("hour") or 12) % 24
        weekday = int(payload.get("weekday") or 1)
        name = str(payload.get("display_name") or "").strip()

        from juno_v2.contracts.writer import WriterMode, WriterTransformRequest

        instr = (
            "Return ONLY one JSON object (no markdown) with keys headline and subline (both strings). "
            "headline: short time-of-day greeting (max 42 chars). If a first name is given and it fits naturally, use it once. "
            "subline: one relaxed sentence (max 115 chars) about speaking and text appearing where they are typing; everything runs on this Mac. "
            "Avoid: em dashes, three-part lists, slogans like 'Private by design', 'no cloud', 'voice-to-text moment', or hype. "
            "Sound human and plain. "
            f"context_first_name: {json.dumps(name)}, local_hour: {hour}, weekday_1sun_7sat: {weekday}."
        )
        req = WriterTransformRequest(
            utterance_id=f"shell_greet_{uuid.uuid4().hex[:10]}",
            instruction=instr,
            source_text=".",
            mode=WriterMode.DEFAULT_SURFACE,
            context_payload={},
            metadata={"kind": "shell_home_greeting"},
        )

        def _rewrite():
            return backend.rewrite(req)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_rewrite)
                result = fut.result(timeout=5.0)
        except FuturesTimeout:
            return {"ok": False, "error": "timeout", "error_code": "timeout"}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "error_code": "writer_failed"}

        text = str(getattr(result, "text", "") or "").strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if "\n" in text:
                text = text.split("\n", 1)[1].strip()
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
        except Exception:
            return {"ok": False, "error": "bad_greeting_json", "error_code": "bad_greeting_json"}
        if not isinstance(obj, dict):
            return {"ok": False, "error": "bad_greeting_json", "error_code": "bad_greeting_json"}
        hl = str(obj.get("headline") or "").strip()
        sl = str(obj.get("subline") or "").strip()
        if not hl or not sl:
            return {"ok": False, "error": "empty_greeting_fields", "error_code": "empty_greeting_fields"}
        hl = re.sub(r"\s+", " ", hl)[:80]
        sl = re.sub(r"\s+", " ", sl)[:200]
        return {"ok": True, "headline": hl, "subline": sl, "source": "writer"}

    def broker_memory_replacement_list(self) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        with self._lock:
            return {"ok": True, "entries": self.memory.replacements.raw()}

    def broker_memory_replacement_upsert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        trigger = str(payload.get("trigger") or "").strip()
        replacement = str(payload.get("replacement") or "")
        if not trigger or not replacement:
            return {"ok": False, "error": "trigger_and_replacement_required"}
        with self._lock:
            self.memory.add_replacement(
                trigger=trigger,
                replacement=replacement,
                scope=str(payload.get("scope") or "global"),
                case_sensitive=bool(payload.get("case_sensitive") or False),
                source=str(payload.get("source") or "user_edit"),
            )
            entries = self.memory.replacements.raw()
        return {"ok": True, "entries": entries}

    def broker_memory_replacement_remove(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        trigger = str(payload.get("trigger") or "").strip()
        if not trigger:
            return {"ok": False, "error": "trigger_required"}
        with self._lock:
            removed = self.memory.remove_replacement(
                trigger,
                scope=str(payload.get("scope") or "global"),
                case_sensitive=bool(payload.get("case_sensitive") or False),
            )
            entries = self.memory.replacements.raw()
        return {"ok": True, "removed": removed, "entries": entries}

    def broker_memory_snippet_list(self) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        with self._lock:
            return {"ok": True, "entries": self.memory.snippets.raw()}

    def broker_memory_snippet_upsert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        trigger = str(payload.get("trigger") or "").strip()
        body = str(payload.get("body") or "")
        if not trigger or not body:
            return {"ok": False, "error": "trigger_and_body_required"}
        with self._lock:
            self.memory.add_snippet(
                trigger=trigger,
                body=body,
                scope=str(payload.get("scope") or "global"),
                case_sensitive=bool(payload.get("case_sensitive") or False),
                source=str(payload.get("source") or "user_edit"),
                description=str(payload.get("description") or ""),
            )
            entries = self.memory.snippets.raw()
        return {"ok": True, "entries": entries}

    def broker_memory_snippet_remove(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        trigger = str(payload.get("trigger") or "").strip()
        if not trigger:
            return {"ok": False, "error": "trigger_required"}
        with self._lock:
            removed = self.memory.remove_snippet(
                trigger, scope=str(payload.get("scope") or "global")
            )
            entries = self.memory.snippets.raw()
        return {"ok": True, "removed": removed, "entries": entries}

    def broker_memory_correction_list(self) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        with self._lock:
            return {"ok": True, "entries": self.memory.corrections.raw()}

    def broker_memory_correction_remove(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        err = self._memory_required()
        if err:
            return err
        observed = str(payload.get("observed") or "").strip()
        corrected_raw = payload.get("corrected")
        corrected = str(corrected_raw).strip() if corrected_raw else None
        if not observed:
            return {"ok": False, "error": "observed_required"}
        with self._lock:
            removed = self.memory.remove_correction(observed, corrected)
            entries = self.memory.corrections.raw()
        return {"ok": True, "removed": removed, "entries": entries}

    def broker_clipboard_push(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            return {"ok": False, "error": "text_required"}
        self.clipboard_ring.push(text)
        return {"ok": True, "size": len(self.clipboard_ring.recent(limit=1_000_000))}

    def broker_clipboard_recent(self, limit: int = 5) -> Dict[str, Any]:
        try:
            n = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            n = 5
        entries = self.clipboard_ring.recent(limit=n)
        return {
            "ok": True,
            "entries": [
                {"text": e.text, "ts_unix_ms": e.ts_unix_ms, "redacted": e.redacted}
                for e in entries
            ],
        }

    def broker_tools_list(self, *, read_only_only: bool = False) -> Dict[str, Any]:
        return {"ok": True, "tools": self.tools.list_tools(read_only_only=read_only_only)}

    def broker_tool_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "missing_tool_name"}
        args = payload.get("arguments") or {}
        if not isinstance(args, dict):
            return {"ok": False, "error": "arguments_must_be_object"}
        with self._lock:
            result = self.tools.call(name, args)
        out = result.to_dict()
        out["tool"] = name
        return out


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    server_version = "JunoWorkbench/0.1"

    def __init__(self, *args: Any, app: WorkbenchApp, **kwargs: Any) -> None:
        self.app = app
        super().__init__(*args, **kwargs)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8")) if body else {}

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(
        self,
        status: int,
        text: str,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, relative_path: str) -> None:
        safe_path = (STATIC_DIR / relative_path).resolve()
        if (
            not str(safe_path).startswith(str(STATIC_DIR.resolve()))
            or not safe_path.exists()
            or not safe_path.is_file()
        ):
            self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = "text/plain; charset=utf-8"
        if safe_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif safe_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif safe_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self._send_text(
            HTTPStatus.OK,
            safe_path.read_text(encoding="utf-8"),
            content_type=content_type,
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        self.app.recorder.record(TraceKind.API, "http_get", {"path": path})
        routes = {
            "/api/state": self.app.state,
            "/api/runtime": self.app.runtime_snapshot,
            "/healthz": lambda: {
                "ok": True,
                "session_id": self.app.session_id,
                "warm": self.app.warm_status(),
            },
            "/api/broker/engine/compatibility": self.app.broker_engine_compatibility,
            "/api/broker/context/inspect": self.app.broker_context_inspect,
            "/api/broker/surface/policy": self.app.broker_surface_policy,
            "/api/broker/surface/active": self.app.broker_surface_active,
            "/api/broker/model/routes": self.app.broker_model_routes,
            "/api/broker/setup/status": self.app.broker_setup_status,
            "/api/broker/stats/summary": self.app.broker_stats_summary,
            "/api/broker/storage/stats": self.app.broker_storage_stats,
            "/api/broker/runtime/backends": self.app.broker_runtime_backends,
            "/api/broker/modes/builtin": self.app.broker_modes_builtin_list,
            "/api/broker/modes/custom": self.app.broker_modes_custom_list,
            "/api/broker/modes/current": self.app.broker_modes_current,
            "/api/broker/transforms/builtin": self.app.broker_transforms_builtin_list,
            "/api/broker/transforms/custom": self.app.broker_transforms_custom_list,
            "/api/broker/recovery/history": self.app.broker_recovery_history,
            "/api/broker/settings": self.app.broker_settings_get,
            "/api/broker/personalization/summary": self.app.broker_personalization_summary,
        }
        if path == "/":
            return self._serve_static("index.html")
        if path == "/app.js":
            return self._serve_static("app.js")
        if path == "/styles.css":
            return self._serve_static("styles.css")
        if path == "/api/broker/history":
            qs = parse_qs(parsed.query)
            try:
                limit = int((qs.get("limit") or ["50"])[0])
            except ValueError:
                limit = 50
            return self._send_json(HTTPStatus.OK, self.app.broker_utterance_history(limit=limit))
        if path == "/api/broker/export/data.zip":
            return self._send_bytes(
                HTTPStatus.OK,
                self.app.broker_export_data_zip_bytes(),
                "application/zip",
            )
        if path.startswith("/api/broker/audio/") and path.endswith("/replay"):
            uid = path.split("/api/broker/audio/", 1)[1].rsplit("/replay", 1)[0].strip("/")
            data = self.app.broker_audio_replay_bytes(uid)
            if data is None:
                return self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "not_found", "error_code": "not_found"},
                )
            return self._send_bytes(HTTPStatus.OK, data, "audio/wav")
        if path == "/api/broker/recovery/replay":
            qs = parse_qs(parsed.query)
            uid = (qs.get("utterance_id") or [""])[0]
            route = (qs.get("route") or [None])[0]
            return self._send_json(
                HTTPStatus.OK,
                self.app.broker_replay_utterance(uid, route=route),
            )
        handler = routes.get(path)
        if handler is not None:
            return self._send_json(HTTPStatus.OK, handler())
        return self._send_text(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        self.app.recorder.record(TraceKind.API, "http_post", {"path": path})
        payload = self._read_json()
        routes = {
            "/api/broker/session/start": self.app.broker_start_session,
            "/api/broker/session/transform": self.app.broker_transform,
            "/api/broker/modes/manual/set": self.app.broker_modes_manual_set,
            "/api/broker/modes/custom/upsert": self.app.broker_modes_custom_set,
            "/api/broker/modes/custom/delete": self.app.broker_modes_custom_delete,
            "/api/broker/modes/custom/activate": self.app.broker_modes_custom_activate,
            "/api/broker/transforms/custom/upsert": self.app.broker_transforms_custom_upsert,
            "/api/broker/transforms/custom/delete": self.app.broker_transforms_custom_delete,
            "/api/broker/insertion/committed": self.app.broker_insertion_committed,
            "/api/broker/learning/observe_correction": self.app.broker_observe_correction,
            "/api/broker/shell/home_greeting": self.app.broker_shell_home_greeting,
            "/api/broker/writer/extract": self.app.broker_writer_extract,
        }
        if path == "/api/broker/setup/install":
            return self._send_json(HTTPStatus.OK, self.app.broker_setup_install(payload, force=False))
        if path == "/api/broker/setup/repair":
            return self._send_json(HTTPStatus.OK, self.app.broker_setup_install(payload, force=True))
        if path == "/api/broker/dictation/live_correct":
            return self._send_json(HTTPStatus.OK, self.app.broker_dictation_live_correct(payload))
        if path == "/api/broker/modes/manual/clear":
            return self._send_json(HTTPStatus.OK, self.app.broker_modes_manual_clear())
        if path == "/api/broker/storage/audio/prune_all":
            return self._send_json(HTTPStatus.OK, self.app.broker_storage_prune_all_audio())
        if path == "/api/broker/history/clear_all":
            return self._send_json(HTTPStatus.OK, self.app.broker_history_clear_all())
        if path == "/api/broker/memory/clear_all":
            return self._send_json(HTTPStatus.OK, self.app.broker_memory_clear_all())
        if path == "/api/broker/history/cancel_draft":
            return self._send_json(HTTPStatus.OK, self.app.broker_history_cancel_draft(payload))
        if path == "/api/broker/history/reprocess":
            return self._send_json(
                HTTPStatus.OK,
                self.app.broker_history_reprocess(
                    str(payload.get("utterance_id") or ""),
                    str(payload.get("mode_name") or ""),
                    is_custom=bool(payload.get("is_custom") or False),
                ),
            )
        if path.startswith("/api/broker/history/") and path.endswith("/actions"):
            uid = path[len("/api/broker/history/"): -len("/actions")].strip("/")
            return self._send_json(
                HTTPStatus.OK,
                self.app.broker_history_update_actions(uid, payload),
            )
        handler = routes.get(path)
        if handler is not None:
            return self._send_json(HTTPStatus.OK, handler(payload))
        return self._send_text(HTTPStatus.NOT_FOUND, "Not found")

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        self.app.recorder.record(TraceKind.API, "http_delete", {"path": path})
        if path.startswith("/api/broker/history/"):
            out = self.app.broker_history_delete(path.split("/api/broker/history/", 1)[1].strip("/"))
            status = HTTPStatus.OK if out.get("ok") else HTTPStatus.NOT_FOUND
            return self._send_json(status, out)
        return self._send_text(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        if self.app.config.debug:
            super().log_message(format, *args)


def start_http_server(app: WorkbenchApp) -> tuple[ThreadingHTTPServer, threading.Thread]:
    handler = partial(WorkbenchRequestHandler, app=app)
    httpd = ThreadingHTTPServer((app.config.host, app.config.port), handler)
    thread = threading.Thread(target=httpd.serve_forever, name="juno-v2-workbench", daemon=True)
    thread.start()
    app.recorder.record(
        TraceKind.SYSTEM,
        "server_started",
        {"host": app.config.host, "port": app.config.port},
    )
    return httpd, thread


def stop_http_server(app: WorkbenchApp, httpd: ThreadingHTTPServer) -> None:
    app.recorder.record(TraceKind.SYSTEM, "server_stopped", {})
    httpd.shutdown()
    httpd.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Juno workbench server.")
    parser.add_argument("--host", default=os.environ.get("JUNO_V2_WORKBENCH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("JUNO_V2_WORKBENCH_PORT", "8765")))
    parser.add_argument("--log-dir", default=os.environ.get("JUNO_V2_WORKBENCH_LOG_DIR", ".juno_v2_logs/workbench"))
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = WorkbenchRuntimeConfig(
        host=args.host,
        port=args.port,
        log_dir=Path(args.log_dir),
        debug=bool(args.debug),
    )
    app = WorkbenchApp(config)
    httpd, thread = start_http_server(app)
    url = f"http://{config.host}:{config.port}"
    print(f"Juno workbench running at {url}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Juno workbench.", flush=True)
    finally:
        stop_http_server(app, httpd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
