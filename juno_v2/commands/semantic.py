from __future__ import annotations

from juno_v2.contracts.commands import CommandTargetClass, SemanticCommandIntent
from juno_v2.contracts.modes import ModePolicy
from juno_v2.contracts.writer import WriterMode


def interpret_semantic_command(
    text: str,
    *,
    mode_policy: ModePolicy | None,
    active_mode: WriterMode,
    target_class: CommandTargetClass,
    target_text: str | None,
) -> SemanticCommandIntent | None:
    """Lightweight template interpreter; returns None when no semantic op."""
    t = (text or "").strip().casefold()
    if not t or target_class == CommandTargetClass.NONE:
        return None
    allow = mode_policy is None or mode_policy.allow_model_insert_rewrite or active_mode == WriterMode.COMMAND_MODE
    if not allow:
        return SemanticCommandIntent(
            intent_name="declined_semantic",
            target_class=target_class,
            target_confidence=0.2,
            rewrite_instruction="",
            requires_confirmation=True,
            ambiguity_reason="mode_disallows_model_semantics",
        )
    def _intent(name: str, instruction: str, confidence: float = 0.75) -> SemanticCommandIntent:
        return SemanticCommandIntent(
            intent_name=name,
            target_class=target_class,
            target_confidence=confidence,
            rewrite_instruction=instruction,
            requires_confirmation=False,
            ambiguity_reason=None,
        )

    if any(token in t for token in ("concise", "shorter", "brief")) and any(token in t for token in ("that", "this", "it")):
        return _intent("make_shorter", "Make the text more concise. Preserve meaning.", 0.78)
    if any(token in t for token in ("clearer", "clear", "clarity")) and any(token in t for token in ("that", "this", "it")):
        return _intent("make_clearer", "Improve clarity. Preserve meaning.", 0.74)
    if any(token in t for token in ("formal", "professional", "polished")) and any(token in t for token in ("that", "this", "it")):
        return _intent("make_formal", "Rewrite in a formal, professional tone. Preserve meaning.", 0.77)
    if any(token in t for token in ("casual", "friendly", "conversational", "informal")) and any(token in t for token in ("that", "this", "it")):
        return _intent("make_casual", "Rewrite in a casual, friendly tone. Preserve meaning.", 0.77)
    if any(token in t for token in ("grammar", "spelling", "punctuation", "typos")) and any(token in t for token in ("fix", "clean", "correct")):
        return _intent("fix_grammar", "Fix grammar, spelling, and punctuation. Preserve meaning exactly.", 0.8)
    if any(token in t for token in ("summarize", "summarise", "summary")) and any(token in t for token in ("that", "this", "it")):
        return _intent("summarize", "Summarize into concise key points. Preserve core meaning.", 0.76)
    if "bullet" in t and any(token in t for token in ("that", "this", "it")):
        return _intent("bullets", "Convert the text into clear bullet points. Preserve meaning.", 0.75)
    if ("numbered" in t or "number" in t) and "list" in t and any(token in t for token in ("that", "this", "it")):
        return _intent("numbered", "Convert the text into a numbered list. Preserve meaning.", 0.75)
    if any(token in t for token in ("expand", "longer", "detailed", "elaborate")) and any(token in t for token in ("that", "this", "it")):
        return _intent("expand", "Expand with useful detail while preserving meaning.", 0.73)
    if any(token in t for token in ("simplify", "simpler", "easier to read")) and any(token in t for token in ("that", "this", "it")):
        return _intent("simplify", "Simplify the text. Preserve meaning.", 0.74)
    if "shorter" in t and "that" in t:
        return SemanticCommandIntent(
            intent_name="make_shorter",
            target_class=target_class,
            target_confidence=0.75,
            rewrite_instruction="Make the text more concise. Preserve meaning.",
            requires_confirmation=False,
            ambiguity_reason=None,
        )
    if "clearer" in t and "that" in t:
        return SemanticCommandIntent(
            intent_name="make_clearer",
            target_class=target_class,
            target_confidence=0.72,
            rewrite_instruction="Improve clarity. Preserve meaning.",
            requires_confirmation=False,
            ambiguity_reason=None,
        )
    return None
