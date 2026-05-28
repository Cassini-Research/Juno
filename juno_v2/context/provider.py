from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from juno_v2.context.app_classifier import classify_app_category
from juno_v2.context.redaction import ContextRedactor
from juno_v2.contracts.context import ContextWindowSource, RedactionSummary, TypedContextBundle


class ContextProvider(Protocol):
    def snapshot(self) -> TypedContextBundle: ...


@dataclass(slots=True)
class StaticContextProvider:
    app_name: str | None = None
    window_title: str | None = None
    clipboard_text: str = ""
    selected_text: str = ""
    focused_text_before: str = ""
    focused_text_after: str = ""
    max_field_chars: int = 1600
    redactor: ContextRedactor = field(default_factory=ContextRedactor)

    def snapshot(self) -> TypedContextBundle:
        return _build_bundle(
            app_name=self.app_name,
            window_title=self.window_title,
            app_bundle_id=None,
            selected_text=self.selected_text,
            focused_text_before=self.focused_text_before,
            focused_text_after=self.focused_text_after,
            clipboard_text=self.clipboard_text,
            max_field_chars=self.max_field_chars,
            redactor=self.redactor,
            last_committed_text='',
            last_committed_start=None,
            last_committed_end=None,
            last_committed_utterance_id=None,
        )


@dataclass(slots=True)
class WorkbenchContextProvider:
    store: ContextWindowSource
    max_field_chars: int = 1600
    redactor: ContextRedactor = field(default_factory=ContextRedactor)

    def snapshot(self) -> TypedContextBundle:
        window = self.store.context_window(max_field_chars=self.max_field_chars)
        return _build_bundle(
            app_name=window['app_name'],
            window_title=window['window_title'],
            app_bundle_id=window.get('app_bundle_id'),
            selected_text=window['selected_text'],
            focused_text_before=window['focused_text_before'],
            focused_text_after=window['focused_text_after'],
            clipboard_text=window['clipboard_text'],
            max_field_chars=self.max_field_chars,
            redactor=self.redactor,
            last_committed_text=window.get('last_committed_text', ''),
            last_committed_start=window.get('last_committed_start'),
            last_committed_end=window.get('last_committed_end'),
            last_committed_utterance_id=window.get('last_committed_utterance_id'),
        )


def _build_bundle(
    *,
    app_name: str | None,
    window_title: str | None,
    app_bundle_id: str | None = None,
    selected_text: str,
    focused_text_before: str,
    focused_text_after: str,
    clipboard_text: str,
    max_field_chars: int,
    redactor: ContextRedactor,
    last_committed_text: str,
    last_committed_start: int | None,
    last_committed_end: int | None,
    last_committed_utterance_id: str | None,
) -> TypedContextBundle:
    redaction = RedactionSummary()

    def merge(summary: RedactionSummary) -> None:
        redaction.emails += summary.emails
        redaction.urls += summary.urls
        redaction.digit_sequences += summary.digit_sequences
        redaction.secrets += summary.secrets

    selected_text, r = redactor.redact(selected_text)
    merge(r)
    focused_text_before, r = redactor.redact(focused_text_before[-max_field_chars:])
    merge(r)
    focused_text_after, r = redactor.redact(focused_text_after[:max_field_chars])
    merge(r)
    clipboard_text, r = redactor.redact((clipboard_text or '')[:max_field_chars])
    merge(r)

    field_excerpt = (
        focused_text_before + ((" |SEL| " + selected_text) if selected_text else '') + focused_text_after
    ).strip()
    candidates = _extract_candidates(
        [selected_text, focused_text_before, focused_text_after, clipboard_text, window_title or '', app_name or '']
    )
    app_category = classify_app_category(app_name, window_title, app_bundle_id=app_bundle_id)
    meta: dict = {
        'max_field_chars': max_field_chars,
        'last_committed_text': last_committed_text,
        'last_committed_start': last_committed_start,
        'last_committed_end': last_committed_end,
        'last_committed_utterance_id': last_committed_utterance_id,
    }
    if app_bundle_id and str(app_bundle_id).strip():
        meta['app_bundle_id'] = str(app_bundle_id).strip()
    return TypedContextBundle(
        app_name=app_name,
        window_title=window_title,
        selected_text=selected_text,
        focused_text_before=focused_text_before,
        focused_text_after=focused_text_after,
        clipboard_text=clipboard_text,
        field_text_excerpt=field_excerpt[: max_field_chars * 2],
        candidate_entities=candidates,
        redaction=redaction,
        app_category=app_category,
        metadata=meta,
    )


# Matches snake_case and kebab-case identifiers (at least two parts).
_TECH_IDENT_RE = re.compile(
    r'(?<!\w)'
    r'(?:[a-z][a-z0-9]*(?:(?:_[a-z0-9]+)+|(?:-[a-z][a-z0-9]*)+))'
    r'(?!\w)',
    re.ASCII,
)

# File extensions that strongly imply a technical filename token.
_CODE_EXTENSIONS: frozenset[str] = frozenset({
    'sh', 'bash', 'zsh', 'fish',
    'py', 'pyi', 'ipynb',
    'swift', 'kt', 'java', 'cs', 'go', 'rs', 'rb', 'php',
    'c', 'h', 'cpp', 'cc', 'cxx', 'mm', 'm',
    'ts', 'tsx', 'js', 'jsx', 'mjs', 'cjs',
    'json', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf', 'env',
    'sql', 'md', 'rst', 'dockerfile', 'makefile',
    'html', 'css', 'scss', 'less',
    'proto', 'graphql', 'tf', 'tfvars',
})


def _extract_candidates(chunks: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> bool:
        v = token.strip(" ,.!?;:()[]{}<>\"'`")
        if len(v) < 2 or len(v) > 80:
            return False
        key = v.casefold()
        if key in seen:
            return False
        seen.add(key)
        out.append(v)
        return len(out) >= 32

    # Pass 1: title-case and mixed-case tokens (proper nouns, acronyms, camelCase).
    for chunk in chunks:
        for token in chunk.replace("\n", " ").split():
            clean = token.strip(" ,.!?;:()[]{}<>\"'`")
            if len(clean) < 2:
                continue
            # Skip tokens with no alphabetic character (pure numbers, symbols).
            if not any(c.isalpha() for c in clean):
                continue
            # Require uppercase at start or somewhere inside (catches camelCase too).
            if not (clean[:1].isupper() or any(c.isupper() for c in clean[1:])):
                continue
            if _add(clean):
                return out

    # Pass 2: technical identifiers — snake_case, kebab-case, filename.ext.
    for chunk in chunks:
        for m in _TECH_IDENT_RE.finditer(chunk):
            if _add(m.group(0)):
                return out
        for token in chunk.replace("\n", " ").split():
            clean = token.strip(" ,.!?;:()[]{}<>\"'`")
            # For path-like tokens (containing /), extract the basename.
            candidate = clean.rsplit('/', 1)[-1] if '/' in clean else clean
            if '.' in candidate and not candidate.startswith('.'):
                _, _, ext = candidate.rpartition('.')
                if ext.lower() in _CODE_EXTENSIONS:
                    if _add(candidate):
                        return out

    # Pass 3: lowercase letter+digit mixed tokens. Catches user-coined
    # handles, IDs, and product codes (alpha42, cosmos1, iso8601,
    # chrome120) that have no uppercase or separator and so escape
    # Pass 1 and Pass 2. These are almost always intentional custom
    # tokens — a regular English word doesn't have a digit in it — so
    # they're high-value bias hints. Bounds keep version tags ("v1"),
    # time suffixes ("3pm"), and long hashes out: require >=4 chars,
    # >=2 letters, >=1 digit, <=19 chars, and pure alnum.
    for chunk in chunks:
        for token in chunk.replace("\n", " ").split():
            clean = token.strip(" ,.!?;:()[]{}<>\"'`")
            if len(clean) < 4 or len(clean) > 19:
                continue
            if not clean.isalnum():
                continue
            # Already covered by Pass 1 (uppercase) — skip to avoid
            # double-counting the same token at lower priority.
            if any(c.isupper() for c in clean):
                continue
            letters = sum(1 for c in clean if c.isalpha())
            digits = sum(1 for c in clean if c.isdigit())
            if letters < 2 or digits < 1:
                continue
            if _add(clean):
                return out

    return out
