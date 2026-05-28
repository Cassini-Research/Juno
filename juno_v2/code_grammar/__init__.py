"""Code-output grammar layer.

Deterministic transforms for code-context voice writing:
- case style conversion (snake_case, camelCase, PascalCase, kebab-case)
- file-reference tagging (@file.ts style for code-chat surfaces)
- code-safe symbol rendering

Only activated when app_category == 'code' or an explicit code-grammar
overlay is requested. Never leaks into prose/messaging/email flows.
"""
from juno_v2.code_grammar.engine import CodeGrammarEngine, CodeGrammarMode, CodeGrammarResult

__all__ = ["CodeGrammarEngine", "CodeGrammarMode", "CodeGrammarResult"]
