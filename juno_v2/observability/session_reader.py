"""
Session log reader and summarizer.

Reads a Juno v2 session JSONL trace file and produces a human-readable
timeline or structured JSON summary. Events like vad_frame_decision and
frame_processed are filtered by default; use --verbose to include them.

Usage:
    python -m juno_v2.observability.session_reader [FILE_OR_SESSION_ID]
    python -m juno_v2.observability.session_reader --list [--log-dir DIR]
    python -m juno_v2.observability.session_reader FILE --format json
    python -m juno_v2.observability.session_reader FILE --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_LOG_DIRS = [
    Path(".juno_v2_demo/logs/service"),
    Path(".juno_v2_logs"),
    Path(".juno_v2_logs/service"),
    Path(".juno_v2_logs/workbench"),
    Path(".juno_v2_logs/engine"),
]

_NOISY_EVENTS = {
    "vad_frame_decision",
    "frame_processed",
    "http_get",
    "partial_updated",
    "preview_applied_to_commit_session",
    "final_candidate_updated",
    "final_candidate_cleared",
    "partial_cleared",
    "language_state_updated",
    "live_metrics_updated",
    "runtime_phase_updated",
    "runtime_status_updated",
    "writer_mode_updated",
    "writer_action_updated",
    "workbench_initialized",
    "server_started",
    "server_stopped",
}

_UTTERANCE_SIGNAL_EVENTS = {
    "speech_started",
    "speech_confirmed",
    "speech_paused",
    "speech_resumed",
    "speech_ended",
    "session_aborted",
    "utterance_context_planned",
    "preview_decode_started",
    "preview_decode_completed",
    "preview_emitted",
    "preview_emission_dropped",
    "final_decode_started",
    "final_decode_completed",
    "final_transcript_emitted",
    "writer_intent_parsed",
    "writer_outcome",
    "commit_completed",
    "final_committed",
    "memory_updated_from_commit",
    "speech_start_deferred_due_to_inflight_terminal_stage",
    "editable_sync_applied",
    "editable_sync_failed",
}


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def find_session_file(id_or_path: str, log_dir: Path | None = None) -> Path:
    p = Path(id_or_path)
    if p.exists():
        return p
    search_dirs = [log_dir] if log_dir else _DEFAULT_LOG_DIRS
    for d in search_dirs:
        if d.exists():
            for f in d.rglob("*.jsonl"):
                if id_or_path in f.name or id_or_path == f.stem:
                    return f
    raise FileNotFoundError(f"Session not found: {id_or_path!r}")


def list_sessions(log_dir: Path | None = None) -> list[dict]:
    search_dirs = [log_dir] if log_dir else _DEFAULT_LOG_DIRS
    results = []
    seen: set[Path] = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.jsonl")):
            if f in seen:
                continue
            seen.add(f)
            try:
                meta = _session_quick_meta(f)
                results.append(meta)
            except Exception:
                pass
    results.sort(key=lambda x: x.get("started_at_unix", 0), reverse=True)
    return results


def _session_quick_meta(path: Path) -> dict:
    first: dict | None = None
    last: dict | None = None
    count = 0
    utterance_ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            if first is None:
                first = e
            last = e
            uid = e.get("payload", {}).get("utterance_id")
            if uid and uid.startswith("utt_"):
                utterance_ids.add(uid)
    if first is None:
        raise ValueError("empty file")
    started_ms = first.get("ts_unix_ms", 0)
    ended_ms = last.get("ts_unix_ms", 0) if last else started_ms
    duration_s = max(0, (ended_ms - started_ms) / 1000)
    return {
        "session_id": first.get("session_id", path.stem),
        "path": str(path),
        "started_at_unix": started_ms / 1000,
        "started_at": _fmt_ts(started_ms),
        "duration_s": duration_s,
        "event_count": count,
        "utterance_count": len(utterance_ids),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse(events: list[dict]) -> dict:
    session_id = events[0].get("session_id", "unknown") if events else "unknown"
    started_ms = events[0].get("ts_unix_ms", 0) if events else 0
    ended_ms = events[-1].get("ts_unix_ms", 0) if events else 0

    session_meta: dict = {}
    utterances: dict[str, dict] = {}
    global_errors: list[dict] = []
    writer_actions: dict[str, int] = defaultdict(int)
    language_stats: dict[str, dict] = {
        "requested": defaultdict(int),
        "observed": defaultdict(int),
        "policy": defaultdict(int),
    }
    memory_events: list[dict] = []

    def _utt(uid: str) -> dict:
        if uid not in utterances:
            utterances[uid] = {
                "utterance_id": uid,
                "speech_start_ms": None,
                "speech_end_ms": None,
                "previews": [],
                "bias_plan": None,
                "final_raw": None,
                "final_normalized": None,
                "final_decode_ms": None,
                "normalization_applied": [],
                "language_requested": None,
                "language_observed": None,
                "language_policy": None,
                "writer_intent": None,
                "writer_action": None,
                "writer_output": None,
                "writer_mode": None,
                "committed": None,
                "committed_text": None,
                "conflict_reason": None,
                "commit_mode": None,
                "speech_end_to_final_ms": None,
                "speech_end_to_commit_ms": None,
                "ttft_ms": None,
                "entities_learned": [],
                "errors": [],
                "deferred": False,
                "app_name": None,
                "window_title": None,
                "context_text_before": None,
                "initial_prompt": None,
                "bias_phrases": [],
            }
        return utterances[uid]

    for e in events:
        name = e.get("name", "")
        p = e.get("payload", {})
        ts = e.get("ts_unix_ms", 0)
        uid = p.get("utterance_id")

        # ── Session-level ──────────────────────────────────────────────────
        if name == "dictation_session_started":
            session_meta = {
                "engine_mode": p.get("engine_mode"),
                "preview_backend": p.get("preview_backend"),
                "final_backend": p.get("final_backend"),
                "memory_enabled": p.get("memory_enabled"),
                "context_enabled": p.get("context_enabled"),
                "writer_enabled": p.get("writer_enabled"),
                "allow_interrupt": p.get("allow_interrupt"),
            }
        elif name in ("dictation_runtime_metrics", "dictation_session_completed"):
            if "utterance_count" in p:
                session_meta.update({
                    "utterance_count": p.get("utterance_count"),
                    "committed_count": p.get("committed_count"),
                    "conflict_count": p.get("conflict_count"),
                    "preview_emit_count": p.get("preview_emit_count"),
                    "final_decode_count": p.get("final_decode_count"),
                    "normalization_change_count": p.get("normalization_change_count"),
                })
        elif name == "language_state_updated":
            rl = p.get("requested_language")
            ol = p.get("observed_language")
            lp = p.get("language_policy")
            if rl:
                language_stats["requested"][rl] += 1
            if ol:
                language_stats["observed"][ol] += 1
            if lp:
                language_stats["policy"][lp] += 1

        # ── Utterance-level ───────────────────────────────────────────────
        if not uid:
            continue

        u = _utt(uid)

        if name == "speech_started":
            u["speech_start_ms"] = ts
        elif name == "speech_ended":
            u["speech_end_ms"] = ts
        elif name == "speech_start_deferred_due_to_inflight_terminal_stage":
            u["deferred"] = True

        elif name == "utterance_context_planned":
            ctx = p.get("context", {})
            u["app_name"] = ctx.get("app_name")
            u["window_title"] = ctx.get("window_title")
            u["context_text_before"] = ctx.get("focused_text_before", "")
            u["initial_prompt"] = p.get("initial_prompt")
            u["bias_phrases"] = p.get("bias_phrases", [])
            u["bias_plan"] = {
                "initial_prompt": p.get("initial_prompt"),
                "bias_phrases": p.get("bias_phrases", []),
                "candidate_count": p.get("metadata", {}).get("candidate_count", 0),
                "context_app": ctx.get("app_name"),
                "context_window": ctx.get("window_title"),
                "context_before": (ctx.get("focused_text_before") or "")[-80:],
                "selected_text": ctx.get("selected_text"),
            }

        elif name == "preview_decode_completed":
            preview_text = p.get("text", "")
            norm = p.get("normalization", {})
            decode_ms = p.get("decode_ms")
            if preview_text:
                u["previews"].append({
                    "text": preview_text,
                    "decode_ms": decode_ms,
                    "is_final": p.get("is_final"),
                    "language": p.get("language"),
                    "normalization_applied": norm.get("applied", []),
                })
            # TTFT = time from speech_start to first preview returned
            if u["speech_start_ms"] and u["ttft_ms"] is None and preview_text:
                u["ttft_ms"] = ts - u["speech_start_ms"]

        elif name == "final_decode_completed":
            norm = p.get("normalization", {})
            u["final_raw"] = norm.get("raw_text") or p.get("text", "")
            u["final_normalized"] = norm.get("normalized_text") or p.get("text", "")
            u["final_decode_ms"] = p.get("decode_ms")
            u["normalization_applied"] = norm.get("applied", [])
            u["language_requested"] = p.get("requested_language")
            u["language_observed"] = p.get("language")
            u["language_policy"] = p.get("language_policy")
            if u["speech_end_ms"]:
                u["speech_end_to_final_ms"] = ts - u["speech_end_ms"]

        elif name == "writer_intent_parsed":
            intent = p.get("intent", {})
            u["writer_intent"] = intent.get("kind")
            u["writer_mode"] = p.get("active_mode")

        elif name == "writer_outcome":
            u["writer_action"] = p.get("action")
            u["writer_output"] = p.get("output_text")
            u["writer_mode"] = p.get("writer_mode")
            writer_actions[p.get("action", "unknown")] += 1

        elif name == "commit_completed":
            u["committed"] = p.get("committed")
            u["committed_text"] = p.get("committed_text")
            u["conflict_reason"] = p.get("conflict_reason")
            u["commit_mode"] = p.get("commit_mode")
            if u["speech_end_ms"] and p.get("committed"):
                u["speech_end_to_commit_ms"] = ts - u["speech_end_ms"]

        elif name == "memory_updated_from_commit":
            u["entities_learned"] = p.get("entities", [])
            memory_events.append({"utterance_id": uid, "raw": p.get("raw_text"), "committed": p.get("committed_text"), "entities": p.get("entities", [])})

        elif name == "editable_sync_failed":
            u["errors"].append({"kind": "editable_sync_failed", "error": p.get("error")})
            global_errors.append({"utterance_id": uid, "kind": name, "payload": p})

    # ── Latency summaries ─────────────────────────────────────────────────
    ttfts = [u["ttft_ms"] for u in utterances.values() if u["ttft_ms"] is not None]
    finals = [u["speech_end_to_final_ms"] for u in utterances.values() if u["speech_end_to_final_ms"] is not None]
    commits = [u["speech_end_to_commit_ms"] for u in utterances.values() if u["speech_end_to_commit_ms"] is not None]
    decode_ms_list = [u["final_decode_ms"] for u in utterances.values() if u["final_decode_ms"] is not None]

    latency = {
        "ttft_ms": _dist(ttfts),
        "speech_end_to_final_ms": _dist(finals),
        "speech_end_to_commit_ms": _dist(commits),
        "final_decode_ms": _dist(decode_ms_list),
    }

    # Utterances ordered by speech start time
    ordered_utts = sorted(
        utterances.values(),
        key=lambda u: u["speech_start_ms"] or 0,
    )

    return {
        "session_id": session_id,
        "started_at": _fmt_ts(started_ms),
        "ended_at": _fmt_ts(ended_ms),
        "duration_s": max(0, (ended_ms - started_ms) / 1000),
        "event_count": len(events),
        "session_meta": session_meta,
        "utterances": ordered_utts,
        "latency": latency,
        "language_stats": {k: dict(v) for k, v in language_stats.items()},
        "writer_actions": dict(writer_actions),
        "memory_events": memory_events,
        "errors": global_errors,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_text(data: dict) -> str:
    lines: list[str] = []
    w = lines.append

    m = data["session_meta"]
    w("═" * 72)
    w(f"SESSION  {data['session_id']}")
    w(f"Started  {data['started_at']}   Duration  {_fmt_dur(data['duration_s'])}")
    w(f"Events   {data['event_count']:,}  (raw log lines)")

    if m:
        w("")
        w(f"Engine   {m.get('engine_mode', '?')}")
        w(f"Preview  {m.get('preview_backend', '?')}   Final  {m.get('final_backend', '?')}")
        flags = []
        if m.get("writer_enabled"):
            flags.append("writer=on")
        if m.get("memory_enabled"):
            flags.append("memory=on")
        if m.get("context_enabled"):
            flags.append("context=on")
        if m.get("allow_interrupt"):
            flags.append("interrupt=on")
        if flags:
            w(f"Flags    {' | '.join(flags)}")

    # Language
    ls = data["language_stats"]
    if ls.get("observed"):
        top_lang = sorted(ls["observed"].items(), key=lambda x: -x[1])
        w(f"Language {' '.join(f'{l}×{c}' for l, c in top_lang)}"
          + (f"  (policy={list(ls['policy'].keys())[0]})" if ls.get("policy") else ""))

    utts = data["utterances"]
    committed_utts = [u for u in utts if u.get("committed")]
    conflict_utts = [u for u in utts if u.get("conflict_reason")]
    w("")
    w(f"UTTERANCES  {len(utts)} total  |  {len(committed_utts)} committed  |  {len(conflict_utts)} conflict")

    if not utts:
        w("  (no utterances recorded)")
    else:
        w("─" * 72)
        for i, u in enumerate(utts, 1):
            _fmt_utterance(u, i, lines)

    # Latency
    lat = data["latency"]
    w("")
    w("LATENCY SUMMARY")
    w("─" * 72)
    for key, label in [
        ("ttft_ms", "Time-to-first-token (preview)"),
        ("speech_end_to_final_ms", "Speech end → final ready"),
        ("speech_end_to_commit_ms", "Speech end → committed"),
        ("final_decode_ms", "Final ASR decode time"),
    ]:
        d = lat.get(key, {})
        if d.get("count", 0) > 0:
            w(f"  {label:<38} n={d['count']:2}  "
              f"p50={d.get('p50', 0):.0f}ms  p95={d.get('p95', 0):.0f}ms  "
              f"min={d.get('min', 0):.0f}ms  max={d.get('max', 0):.0f}ms")

    # Writer
    wa = data.get("writer_actions", {})
    if wa:
        w("")
        w("WRITER ACTIONS")
        w("─" * 72)
        for action, count in sorted(wa.items(), key=lambda x: -x[1]):
            w(f"  {action:<40} {count}")

    # Memory
    me = data.get("memory_events", [])
    if me:
        w("")
        w("MEMORY LEARNING (per commit)")
        w("─" * 72)
        for ev in me:
            raw = ev.get("raw", "") or ""
            committed = ev.get("committed", "") or ""
            entities = ev.get("entities", [])
            if raw != committed:
                w(f"  correction: {raw!r} → {committed!r}")
            if entities:
                w(f"  entities learned: {', '.join(entities[:10])}")

    # Errors
    errs = data.get("errors", [])
    if errs:
        w("")
        w(f"ERRORS / ANOMALIES  ({len(errs)} events)")
        w("─" * 72)
        for ev in errs:
            w(f"  [{ev.get('utterance_id', '?')}]  {ev.get('kind', '?')}  {ev.get('payload', {})}")

    w("═" * 72)
    return "\n".join(lines)


def _fmt_utterance(u: dict, idx: int, lines: list[str]) -> None:
    w = lines.append

    uid_short = u["utterance_id"][-12:] if len(u["utterance_id"]) > 12 else u["utterance_id"]
    speech_start = u.get("speech_start_ms")
    speech_end = u.get("speech_end_ms")
    dur_label = ""
    if speech_start and speech_end:
        dur = (speech_end - speech_start) / 1000
        dur_label = f"  ({dur:.1f}s speech)"

    committed_mark = "✓" if u.get("committed") else ("✗ conflict" if u.get("conflict_reason") else "–")
    deferred_note = " [deferred start]" if u.get("deferred") else ""
    w(f"[{idx:02d}] {uid_short}{deferred_note}  {committed_mark}{dur_label}")

    # Context / App
    app = u.get("app_name")
    win = u.get("window_title")
    if app or win:
        w(f"      app={app or '?'}  window={win or '?'}")

    # Bias plan / prompt
    prompt = u.get("initial_prompt")
    phrases = u.get("bias_phrases", [])
    if prompt:
        p_display = prompt if len(prompt) <= 120 else prompt[:117] + "..."
        w(f"      prompt: {p_display!r}")
    if phrases:
        truncated = [p[:40] + "…" if len(p) > 40 else p for p in phrases[:8]]
        w(f"      bias phrases ({len(phrases)}): {', '.join(truncated)}" + (" ..." if len(phrases) > 8 else ""))

    # Preview stream
    previews = u.get("previews", [])
    if previews:
        preview_texts = [p["text"] for p in previews]
        last_preview = preview_texts[-1] if preview_texts else ""
        decode_ms_list = [p["decode_ms"] for p in previews if p.get("decode_ms")]
        avg_decode = mean(decode_ms_list) if decode_ms_list else None
        ttft = u.get("ttft_ms")
        timing = f"ttft={ttft:.0f}ms" if ttft else ""
        if avg_decode:
            timing += f"  avg_decode={avg_decode:.0f}ms"
        w(f"      preview ({len(previews)} emissions): {last_preview!r:.60}  [{timing}]")

    # Final transcript
    final_raw = u.get("final_raw", "")
    final_norm = u.get("final_normalized", "")
    final_decode = u.get("final_decode_ms")
    lang = u.get("language_observed") or u.get("language_requested") or "?"
    if final_norm or final_raw:
        decode_note = f"  decode={final_decode:.0f}ms" if final_decode else ""
        w(f"      final [{lang}]: {final_norm!r:.80}{decode_note}")
        if final_raw and final_raw != final_norm:
            w(f"      raw:          {final_raw!r:.80}")

    # Normalization changes
    norm_applied = u.get("normalization_applied", [])
    if norm_applied:
        for change in norm_applied[:4]:
            kind = change.get("kind", "?")
            before = change.get("before", "")
            after = change.get("after", "")
            source = change.get("source", "")
            w(f"      norm [{kind}/{source}]: {before!r} → {after!r}")
        if len(norm_applied) > 4:
            w(f"      norm: ... +{len(norm_applied) - 4} more changes")

    # Writer
    writer_intent = u.get("writer_intent")
    writer_action = u.get("writer_action")
    writer_output = u.get("writer_output")
    writer_mode = u.get("writer_mode")
    if writer_intent or writer_action:
        parts = [f"intent={writer_intent}", f"action={writer_action}"]
        if writer_mode:
            parts.append(f"mode={writer_mode}")
        w(f"      writer: {' | '.join(p for p in parts if p)}")
        if writer_output and writer_output != final_norm:
            w(f"      writer output: {writer_output!r:.80}")

    # Commit
    if u.get("committed"):
        ctext = u.get("committed_text", "")
        s2c = u.get("speech_end_to_commit_ms")
        commit_timing = f"  speech_end→commit={s2c:.0f}ms" if s2c else ""
        w(f"      committed: {ctext!r:.80}{commit_timing}")
    elif u.get("conflict_reason"):
        w(f"      conflict: {u['conflict_reason']}")

    # Entities learned
    entities = u.get("entities_learned", [])
    if entities:
        w(f"      learned entities: {', '.join(entities[:8])}")

    # Errors
    for err in u.get("errors", []):
        w(f"      ⚠ {err.get('kind')}: {err.get('error')}")

    w("")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_ts(unix_ms: int) -> str:
    if not unix_ms:
        return "?"
    try:
        dt = datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OSError, OverflowError, ValueError):
        return str(unix_ms)


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m{s:02d}s"


def _dist(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    return {
        "count": len(vals),
        "min": vals[0],
        "max": vals[-1],
        "mean": mean(vals),
        "p50": _pct(vals, 0.50),
        "p95": _pct(vals, 0.95),
    }


def _pct(sorted_vals: list[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Read and summarize a Juno v2 session trace log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("session", nargs="?", help="Session JSONL file path or session ID")
    p.add_argument("--list", action="store_true", help="List all recorded sessions")
    p.add_argument("--log-dir", type=Path, default=None, help="Root log directory (default: .juno_v2_logs)")
    p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Include noisy low-level events (VAD frames, http_get, etc.)",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.list or not args.session:
        sessions = list_sessions(args.log_dir)
        if not sessions:
            print("No session logs found.", file=sys.stderr)
            sys.exit(0)
        if args.format == "json":
            print(json.dumps(sessions, indent=2, ensure_ascii=False))
        else:
            print(f"{'SESSION ID':<46} {'STARTED':<22} {'DUR':>7} {'EVENTS':>8} {'UTTS':>6}")
            print("─" * 96)
            for s in sessions:
                print(
                    f"{s['session_id']:<46} {s['started_at']:<22} "
                    f"{_fmt_dur(s['duration_s']):>7} {s['event_count']:>8,} {s['utterance_count']:>6}"
                )
        return

    try:
        path = find_session_file(args.session, args.log_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    events = load_events(path)

    if not args.verbose:
        events = [e for e in events if e.get("name") not in _NOISY_EVENTS]

    data = analyse(events)

    if args.format == "json":
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_text(data))


if __name__ == "__main__":
    main()
