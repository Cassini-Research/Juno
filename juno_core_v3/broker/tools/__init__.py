"""Broker tool registry.

The broker exposes *tools* — strongly-typed, side-effecting actions — so
both local AI agents and external MCP clients can drive Juno
programmatically. The contract mirrors common tool-use conventions:

- Every tool declares a ``name``, ``description``, JSON-schema
  ``parameters``, and a ``read_only`` flag.
- Tools return a :class:`ToolResult` with a machine-usable payload plus a
  human-readable ``display_text`` for UI surfaces.
- Read-only tools are safe to call any number of times. Write tools can
  be gated by broker policy (``BrokerPolicyEngine``) later.

This package intentionally avoids any LLM SDK coupling. The Python
dictionaries it produces are compatible with JSON-schema tool definitions,
so wrapping for a concrete transport is a one-file task.
"""

from juno_core_v3.broker.tools.builtins import register_builtin_tools
from juno_core_v3.broker.tools.registry import (
    ToolDefinition,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)

__all__ = [
    "ToolDefinition",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
    "register_builtin_tools",
]
