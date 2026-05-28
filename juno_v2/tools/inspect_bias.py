"""Inspect the screen-context bias plan Juno would build right now.

Reads the live AX snapshot via the bundled ``juno-capability`` helper (or a
caller-supplied simulated screen), walks the same context pipeline the live
engine uses, and prints the resulting bias plan: candidate entities, compiled
terms (with their priority and protected flag), and the ``initial_prompt`` /
``bias_phrases`` that would land on Whisper and the Qwen adjudicator.

This is a read-only probe. No models are loaded, no audio is recorded,
nothing is committed. Use it any time to verify that a screen with custom
identifiers actually surfaces them to the bias pipeline.

Usage::

    # Live frontmost app (requires Accessibility permission for juno-capability)
    python -m juno_v2.tools.inspect_bias

    # Simulated screen — useful for verifying custom-token extraction
    python -m juno_v2.tools.inspect_bias --simulate \
        --focused-before "ping alice42 about cosmos1" \
        --window-title "Slack — eng"

    # JSON output (for piping into other tools or tests)
    python -m juno_v2.tools.inspect_bias --json
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

from juno_v2.context.compiler import compile_context
from juno_v2.context.macos_desktop import (
    MacOSDesktopContextProvider,
    MacOSDesktopContextProviderConfig,
)
from juno_v2.context.provider import StaticContextProvider
from juno_v2.contracts.context import TypedContextBundle
from juno_v2.contracts.memory import MemorySnapshot
from juno_v2.contracts.modes import ModePolicy, ModeSelection, ModeSource
from juno_v2.memory.bias import RecognitionBiasEngine


DEFAULT_HELPER = "/Applications/Juno.app/Contents/MacOS/juno-capability"


def _empty_mode_selection() -> ModeSelection:
    return ModeSelection(
        effective_mode="default",
        mode_source=ModeSource.AUTO,
        manual_mode_name=None,
        custom_mode_name=None,
        resolved_from_surface=None,
    )


def _neutral_policy() -> ModePolicy:
    return ModePolicy(
        mode_name="default",
        base_mode="default",
        manual_selectable=True,
        writer_behavior="passthrough",
        transform_behavior="passthrough",
        command_behavior="off",
        itn_policy="standard",
        punctuation_policy="standard",
        cleanup_policy="standard",
        snippet_scope_policy="global",
        style_scope_policy="global",
        allow_auto_transform=False,
        allow_model_insert_rewrite=False,
        allow_inline_commands=False,
        allow_recent_target_commands=False,
        allow_selection_commands=False,
        command_ambiguity_policy="reject",
        degradation_behavior="passthrough",
        prompt_prefix="",
    )


def _empty_memory() -> MemorySnapshot:
    return MemorySnapshot(
        schema_version=1,
        lexicon=[],
        replacements=[],
        corrections=[],
        session_entities=[],
    )


def _snapshot_bundle(args: argparse.Namespace) -> tuple[TypedContextBundle, str]:
    """Return (bundle, mode_label) — mode_label is for the human header."""
    if args.simulate:
        provider = StaticContextProvider(
            app_name=args.app_name or "TextEdit",
            window_title=args.window_title or "Untitled — TextEdit",
            focused_text_before=args.focused_before or "",
            focused_text_after=args.focused_after or "",
            selected_text=args.selected_text or "",
            clipboard_text=args.clipboard or "",
        )
        return provider.snapshot(), "SIMULATED"
    provider = MacOSDesktopContextProvider(
        config=MacOSDesktopContextProviderConfig(helper_command=args.helper),
    )
    return provider.snapshot(), "LIVE"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    bundle, mode_label = _snapshot_bundle(args)
    snapshot = _empty_memory()
    utterance_id = str(uuid.uuid4())
    compiled = compile_context(
        utterance_id=utterance_id,
        context=bundle,
        memory_snapshot=snapshot,
        mode_selection=_empty_mode_selection(),
        mode_policy=_neutral_policy(),
        transcript_hint=None,
        session_terms=None,
        language="en",
        stage="final",
    )
    engine = RecognitionBiasEngine()
    plan = engine.build_plan(
        utterance_id=utterance_id,
        snapshot=snapshot,
        context=bundle,
        base_prompt=None,
        memory_packet=compiled.memory_packet,
        mode_policy=_neutral_policy(),
        effective_mode="default",
    )
    asr_packet = compiled.asr_bias_packet()
    return {
        "mode": mode_label,
        "capture": {
            "app_name": bundle.app_name,
            "app_category": bundle.app_category,
            "window_title": bundle.window_title,
            "selected_text": bundle.selected_text,
            "focused_text_before": bundle.focused_text_before,
            "focused_text_after": bundle.focused_text_after,
            "clipboard_text": bundle.clipboard_text,
            "focused_file_path": bundle.focused_file_path,
            "symbol_under_cursor": bundle.symbol_under_cursor,
            "candidate_entities": list(bundle.candidate_entities),
        },
        "compiled_terms": [
            {
                "text": t.text,
                "source": t.source,
                "priority": t.priority,
                "protected": t.protected,
            }
            for t in compiled.terms
        ],
        "asr_bias_packet": {
            "initial_prompt": asr_packet.initial_prompt,
            "bias_phrases": list(asr_packet.bias_phrases),
            "max_prompt_chars": asr_packet.max_prompt_chars,
        },
        "recognition_bias_plan": {
            "initial_prompt": plan.initial_prompt,
            "bias_phrases": list(plan.bias_phrases),
            "prefer_forms_phrases_embedded": plan.metadata.get(
                "prefer_forms_phrases_embedded"
            ),
        },
        "protected_terms_seen_by_qwen": [
            (t.canonical or t.text) for t in compiled.terms if t.protected
        ][:32],
    }


def _print_human(report: dict[str, Any]) -> None:
    cap = report["capture"]
    rule = "=" * 72
    print(rule)
    print(f"[mode] {report['mode']}")
    print(rule)
    print("Capture")
    print(rule)
    for key in (
        "app_name",
        "app_category",
        "window_title",
        "selected_text",
        "focused_text_before",
        "focused_text_after",
        "clipboard_text",
        "focused_file_path",
        "symbol_under_cursor",
    ):
        print(f"  {key:<22} {cap.get(key)!r}")
    print()
    print(rule)
    print("Rare-word seed (provider._extract_candidates)")
    print(rule)
    print(f"  {cap['candidate_entities']!r}")
    print()
    print(rule)
    print("Compiled terms (compiler._compile_terms)")
    print(rule)
    if not report["compiled_terms"]:
        print("  (no terms compiled)")
    for t in report["compiled_terms"]:
        flag = "PROT" if t["protected"] else "    "
        print(
            f"  [{t['source']:>10}] prio={t['priority']:6.1f} {flag} {t['text']!r}"
        )
    print()
    print(rule)
    print("Whisper bias (RecognitionBiasPlan → live engine path)")
    print(rule)
    plan = report["recognition_bias_plan"]
    print(f"  initial_prompt = {plan['initial_prompt']!r}")
    print(f"  bias_phrases   = {plan['bias_phrases']!r}")
    print()
    print(rule)
    print("Qwen adjudicator: protected_terms (preserve exact spelling)")
    print(rule)
    print(f"  {report['protected_terms_seen_by_qwen']!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inspect_bias",
        description="Dry-run Juno's screen-context bias pipeline.",
    )
    parser.add_argument(
        "--helper",
        default=DEFAULT_HELPER,
        help="Path to the juno-capability binary (default: installed app).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Skip the live AX capture and use --focused-before / --window-title.",
    )
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--window-title", default=None)
    parser.add_argument("--focused-before", default=None)
    parser.add_argument("--focused-after", default=None)
    parser.add_argument("--selected-text", default=None)
    parser.add_argument("--clipboard", default=None)
    parser.add_argument(
        "--json", action="store_true", help="Emit the report as JSON instead of text."
    )
    args = parser.parse_args(argv)
    report = build_report(args)
    if args.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
