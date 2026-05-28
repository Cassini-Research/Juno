"""Built-in broker tools.

Tools here are the *only* surface LLM agents / external MCP clients can
use to drive Juno. They wrap higher-level subsystems (personalization
memory, broker facade) in a stable JSON-schema interface
and return :class:`ToolResult` values.

Design rules:

- Tools never import IO libraries directly. They reach in through the
  objects they're given, so a test can pass fakes.
- Tools are small: one intent per tool, and errors become
  ``ToolResult(ok=False, ...)`` instead of exceptions.
- Write tools must state that in ``read_only=False``; read-only tools
  must never mutate on disk.

The set below is intentionally small — just the ones the Mac shell,
tests, and upcoming MCP transport need. Larger catalogues belong in
phase-specific modules so this file stays readable.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from juno_core_v3.broker.tools.registry import (
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)


class _MemoryLike(Protocol):
    def add_lexicon_entry(self, *, term: str, canonical_form: str | None, boost: float, source: str) -> None: ...
    def add_replacement(self, *, trigger: str, replacement: str, scope: str, case_sensitive: bool, source: str) -> None: ...
    def add_snippet(self, *, trigger: str, body: str, scope: str, case_sensitive: bool, source: str, description: str) -> None: ...
    def record_correction(self, observed: str, corrected: str) -> None: ...


# --------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------- #

def _str(description: str, *, enum: list[str] | None = None) -> dict[str, Any]:
    s: dict[str, Any] = {"type": "string", "description": description}
    if enum is not None:
        s["enum"] = enum
    return s


def _bool(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


# --------------------------------------------------------------------- #
# Personalization tools
# --------------------------------------------------------------------- #

def _tool_learn_vocabulary(memory: _MemoryLike) -> ToolDefinition:
    def handler(args: Mapping[str, Any]) -> ToolResult:
        term = str(args.get("term", "")).strip()
        if not term:
            return ToolResult(ok=False, display_text="term is empty", error="invalid_argument")
        canonical = args.get("canonical_form")
        boost = float(args.get("boost", 1.0))
        memory.add_lexicon_entry(
            term=term,
            canonical_form=canonical,
            boost=boost,
            source="tool",
        )
        return ToolResult(
            ok=True,
            display_text=f"Learned vocabulary {term!r}",
            data={"term": term},
        )

    return ToolDefinition(
        name="memory.learn_vocabulary",
        description="Add a vocabulary item so ASR and writer bias toward this term.",
        parameters={
            "term": _str("The word or phrase to remember."),
            "canonical_form": _str("Preferred spelling / casing of the term."),
            "boost": {"type": "number", "description": "Recognition bias (default 1.0)."},
        },
        required=("term",),
        handler=handler,
        read_only=False,
        tags=("personalization",),
    )


def _tool_learn_replacement(memory: _MemoryLike) -> ToolDefinition:
    def handler(args: Mapping[str, Any]) -> ToolResult:
        trigger = str(args.get("trigger", "")).strip()
        replacement = str(args.get("replacement", ""))
        if not trigger or not replacement:
            return ToolResult(
                ok=False,
                display_text="trigger and replacement are required",
                error="invalid_argument",
            )
        memory.add_replacement(
            trigger=trigger,
            replacement=replacement,
            scope=str(args.get("scope", "global")),
            case_sensitive=bool(args.get("case_sensitive", False)),
            source="tool",
        )
        return ToolResult(
            ok=True,
            display_text=f"Added replacement {trigger!r} -> {replacement!r}",
        )

    return ToolDefinition(
        name="memory.learn_replacement",
        description="Teach the writer to substitute trigger with replacement.",
        parameters={
            "trigger": _str("Text the user tends to say."),
            "replacement": _str("Text to write in its place."),
            "scope": _str("'global' or an app-category scope."),
            "case_sensitive": _bool("Whether to match case-sensitively."),
        },
        required=("trigger", "replacement"),
        handler=handler,
        read_only=False,
        tags=("personalization",),
    )


def _tool_learn_snippet(memory: _MemoryLike) -> ToolDefinition:
    def handler(args: Mapping[str, Any]) -> ToolResult:
        trigger = str(args.get("trigger", "")).strip()
        body = str(args.get("body", ""))
        if not trigger or not body:
            return ToolResult(
                ok=False,
                display_text="trigger and body are required",
                error="invalid_argument",
            )
        memory.add_snippet(
            trigger=trigger,
            body=body,
            scope=str(args.get("scope", "global")),
            case_sensitive=bool(args.get("case_sensitive", False)),
            source="tool",
            description=str(args.get("description", "")),
        )
        return ToolResult(
            ok=True,
            display_text=f"Added snippet {trigger!r}",
        )

    return ToolDefinition(
        name="memory.learn_snippet",
        description="Register a reusable snippet that expands when the user says the trigger.",
        parameters={
            "trigger": _str("Short phrase the user says to expand the snippet."),
            "body": _str("Full text to insert."),
            "scope": _str("'global' or an app-category scope."),
            "case_sensitive": _bool("Match the trigger case-sensitively."),
            "description": _str("Optional human-readable description."),
        },
        required=("trigger", "body"),
        handler=handler,
        read_only=False,
        tags=("personalization", "snippet"),
    )


def _tool_record_correction(memory: _MemoryLike) -> ToolDefinition:
    def handler(args: Mapping[str, Any]) -> ToolResult:
        observed = str(args.get("observed", ""))
        corrected = str(args.get("corrected", ""))
        if not observed or not corrected:
            return ToolResult(
                ok=False,
                display_text="observed and corrected are required",
                error="invalid_argument",
            )
        memory.record_correction(observed, corrected)
        return ToolResult(ok=True, display_text="correction recorded")

    return ToolDefinition(
        name="memory.record_correction",
        description="Record a user correction (observed -> corrected) for the post-edit learning loop.",
        parameters={
            "observed": _str("What Juno wrote."),
            "corrected": _str("What the user actually wanted."),
        },
        required=("observed", "corrected"),
        handler=handler,
        read_only=False,
        tags=("personalization",),
    )


# --------------------------------------------------------------------- #
# Public wiring
# --------------------------------------------------------------------- #

def register_builtin_tools(
    registry: ToolRegistry,
    *,
    memory: _MemoryLike | None = None,
) -> ToolRegistry:
    """Register the default tool catalogue.

    ``memory`` is the personalization store; when omitted, personalization
    tools are skipped so read-only deployments still work.
    """
    if memory is not None:
        registry.register(_tool_learn_vocabulary(memory))
        registry.register(_tool_learn_replacement(memory))
        registry.register(_tool_learn_snippet(memory))
        registry.register(_tool_record_correction(memory))
    return registry


__all__ = ["register_builtin_tools"]
