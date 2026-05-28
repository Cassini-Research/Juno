"""Tool registry — the broker's action surface.

Tools are the broker's only sanctioned side-effect boundary. Contrast
with :mod:`juno_core_v3.broker.runners`, which are internal session
executors. A tool:

- is discoverable (``list_tools``) and self-describing (JSON schema);
- is either read-only (no user-visible changes) or write (mutates state
  like notes, personalization, clipboard);
- is synchronous from the caller's POV — long-running flows are
  sessions, not tools.

We deliberately keep the argument type as ``dict[str, Any]`` because the
transport (HTTP JSON, MCP, in-process SDK) always speaks JSON. Each
handler is responsible for validating its own args — the registry only
enforces that required keys are present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


class ToolNotFoundError(KeyError):
    """Raised when :meth:`ToolRegistry.call` is asked for an unknown tool."""


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a tool invocation.

    ``ok`` is False for *expected* failures (e.g., note-not-found). For
    programmer errors — bad schema, missing handler — we raise instead.
    """

    ok: bool
    display_text: str
    data: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "display_text": self.display_text,
            "data": dict(self.data),
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class ToolDefinition:
    """Schema + handler for a single tool."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], ToolResult]
    read_only: bool = False
    required: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        """Return an MCP / OpenAI-function-calling compatible descriptor."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": dict(self.parameters),
                "required": list(self.required),
            },
            "read_only": self.read_only,
            "tags": list(self.tags),
        }


class ToolRegistry:
    """Thread-unsafe mapping of tool name → :class:`ToolDefinition`.

    The broker is assumed to own a single registry instance and serialize
    calls onto its own event loop, so we don't lock here.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def list_tools(self, *, read_only_only: bool = False) -> list[dict[str, Any]]:
        tools = self._tools.values()
        if read_only_only:
            tools = [t for t in tools if t.read_only]
        return [t.describe() for t in tools]

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def call(self, name: str, args: Mapping[str, Any] | None = None) -> ToolResult:
        tool = self.get(name)
        args = args or {}
        missing = [k for k in tool.required if k not in args]
        if missing:
            return ToolResult(
                ok=False,
                display_text=f"missing required argument(s): {', '.join(missing)}",
                error="missing_arguments",
                data={"missing": missing},
            )
        try:
            return tool.handler(args)
        except Exception as exc:  # handlers translate to ToolResult; this is the safety net
            return ToolResult(
                ok=False,
                display_text=f"tool {name!r} raised {type(exc).__name__}: {exc}",
                error="handler_exception",
            )
