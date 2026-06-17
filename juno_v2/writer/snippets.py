from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from juno_v2.memory.fold import fold_key
from juno_v2.writer.deterministic import SnippetResolver, expand_snippets


_SNIPPET_INVOKE_RE = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+)?"
    r"(?:insert|paste|use|add)\s+(?:the\s+)?(?P<trigger>.+?)\s*[.!?]?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_SNIPPET_DESTINATION_TAIL_RE = re.compile(
    r"\s+(?:into|in|inside|to|onto|for)\s+"
    r"(?:(?:the|this|my|current)\s+)?"
    r"(?:email|mail|message|reply|chat|document|doc|note|field|text\s+field|window|app)\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SnippetTextCandidate:
    source: str
    text: str
    allow_bare: bool = True


@dataclass(frozen=True, slots=True)
class SnippetInvocation:
    trigger: str
    spoken_trigger: str
    body: str
    stored_scope: str
    requested_scope: str
    source: str
    match_kind: str
    case_sensitive: bool

    @property
    def body_chars(self) -> int:
        return len(self.body)

    @property
    def trigger_key_len(self) -> int:
        return len(fold_key(self.trigger) or self.trigger)

    def metadata(self) -> dict[str, object]:
        return {
            "trigger": self.trigger,
            "spoken_trigger": self.spoken_trigger,
            "scope": self.stored_scope,
            "requested_scope": self.requested_scope,
            "source": self.source,
            "match_kind": self.match_kind,
            "body_chars": self.body_chars,
            "case_sensitive": self.case_sensitive,
        }


def looks_like_explicit_snippet_invocation(text: str) -> bool:
    match = _SNIPPET_INVOKE_RE.match(text or "")
    return bool(match is not None and _snippet_word_present(match.group("trigger") or ""))


def resolve_snippet_invocation(
    candidates: Sequence[SnippetTextCandidate],
    *,
    resolver: SnippetResolver | None,
    scopes: Sequence[str],
) -> SnippetInvocation | None:
    """Resolve a spoken snippet invocation across final/raw ASR candidates.

    Product behavior:
    - Explicit commands such as "use customer intro snippet" are high
      confidence and win over model/formatting lanes when the snippet exists.
    - Bare trigger names are allowed only when the caller marks that source as
      safe; they are ranked by specificity so a longer raw trigger can recover
      from wake-word tail stripping.
    - Scoped snippets beat global snippets; global remains the fallback for all
      non-raw surfaces.
    """
    if resolver is None:
        return None
    normalized_scopes = _normalize_scopes(scopes)
    if not normalized_scopes:
        return None

    matches: list[SnippetInvocation] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        text = _clean_spoken_trigger(candidate.text)
        if not text:
            continue
        explicit = _parse_explicit_snippet_trigger(text)
        if explicit:
            hit = _resolve_trigger(
                explicit,
                resolver=resolver,
                scopes=normalized_scopes,
                source=candidate.source,
                match_kind="explicit",
            )
            if hit is not None:
                key = (hit.source, hit.match_kind, hit.trigger.casefold())
                if key not in seen:
                    seen.add(key)
                    matches.append(hit)
        if candidate.allow_bare:
            hit = _resolve_trigger(
                text,
                resolver=resolver,
                scopes=normalized_scopes,
                source=candidate.source,
                match_kind="bare_name",
            )
            if hit is not None:
                key = (hit.source, hit.match_kind, hit.trigger.casefold())
                if key not in seen:
                    seen.add(key)
                    matches.append(hit)
    if not matches:
        return None
    source_rank = {"final": 0, "raw": 1}
    matches.sort(
        key=lambda hit: (
            0 if hit.match_kind == "explicit" else 1,
            -hit.trigger_key_len,
            source_rank.get(hit.source, 9),
            _scope_rank(hit.stored_scope, normalized_scopes),
        )
    )
    return matches[0]


def snippet_bodies_present_in_text(
    text: str,
    *,
    resolver: SnippetResolver | None,
    scopes: Sequence[str],
) -> list[str]:
    list_fn = getattr(resolver, "list", None) if resolver is not None else None
    if not text or not callable(list_fn):
        return []
    allowed_scopes = set(_normalize_scopes(scopes))
    allowed_scopes.add("global")
    out: list[str] = []
    seen: set[str] = set()
    try:
        snippets = list(list_fn())
    except Exception:  # noqa: BLE001
        return []
    for snippet in snippets:
        scope = _snippet_scope(snippet)
        if scope not in allowed_scopes:
            continue
        body = str(getattr(snippet, "body", "") or "")
        if not body or body not in text or body in seen:
            continue
        seen.add(body)
        out.append(body)
        if len(out) >= 16:
            break
    return out


@dataclass(slots=True)
class SnippetBodyProtection:
    replacements: list[tuple[str, str]]

    @classmethod
    def from_bodies(cls, text: str, bodies: Sequence[str]) -> "SnippetBodyProtection":
        replacements: list[tuple[str, str]] = []
        protected = text or ""
        for idx, body in enumerate(bodies):
            if not body or body not in protected:
                continue
            placeholder = f"SNIPPETPROTECT{idx}TOKEN"
            protected = protected.replace(body, placeholder, 1)
            replacements.append((placeholder, body))
        return cls(replacements)

    @property
    def active(self) -> bool:
        return bool(self.replacements)

    def protect(self, text: str) -> str:
        protected = text or ""
        for placeholder, body in self.replacements:
            protected = protected.replace(body, placeholder, 1)
        return protected

    def restore(self, text: str) -> str:
        restored = text or ""
        for placeholder, body in self.replacements:
            restored = restored.replace(placeholder, body)
        return restored

    def placeholders_preserved(self, text: str) -> bool:
        current = text or ""
        return all(placeholder in current for placeholder, _body in self.replacements)


def _parse_explicit_snippet_trigger(text: str) -> str | None:
    match = _SNIPPET_INVOKE_RE.match(text or "")
    if match is None:
        return None
    trigger = _clean_spoken_trigger(match.group("trigger") or "")
    if not trigger:
        return None
    trigger = _SNIPPET_DESTINATION_TAIL_RE.sub("", trigger)
    trigger = re.sub(
        r"^(?:snippet|text\s+snippet)\s+(?:called|named)\s+",
        "",
        trigger,
        flags=re.IGNORECASE,
    )
    return _clean_spoken_trigger(trigger)


def _snippet_word_present(trigger: str) -> bool:
    return bool(re.search(r"\bsnippet\b", trigger or "", flags=re.IGNORECASE))


def _trigger_candidates(raw_trigger: str) -> list[str]:
    trigger = _clean_spoken_trigger(raw_trigger)
    if not trigger:
        return []
    candidates: list[str] = []

    def add(value: str) -> None:
        cleaned = _clean_spoken_trigger(value)
        if cleaned and cleaned.casefold() not in {item.casefold() for item in candidates}:
            candidates.append(cleaned)

    add(trigger)
    add(re.sub(r"(?i)^the\s+", "", trigger))
    if _snippet_word_present(trigger):
        add(re.sub(r"(?i)^snippet\s+", "", trigger))
        add(re.sub(r"(?i)\s+snippet$", "", trigger))
        add(re.sub(r"(?i)^the\s+", "", re.sub(r"(?i)\s+snippet$", "", trigger)))
    return candidates


def _resolve_trigger(
    raw_trigger: str,
    *,
    resolver: SnippetResolver,
    scopes: Sequence[str],
    source: str,
    match_kind: str,
) -> SnippetInvocation | None:
    global_fallback: SnippetInvocation | None = None
    for trigger in _trigger_candidates(raw_trigger):
        for requested_scope in scopes:
            try:
                snippet = resolver.resolve(trigger, scope=requested_scope)
            except Exception:  # noqa: BLE001
                snippet = None
            if snippet is None:
                continue
            body = str(getattr(snippet, "body", "") or "")
            if not body:
                continue
            stored_scope = _snippet_scope(snippet)
            invocation = SnippetInvocation(
                trigger=str(getattr(snippet, "trigger", "") or trigger),
                spoken_trigger=raw_trigger,
                body=body,
                stored_scope=stored_scope,
                requested_scope=requested_scope,
                source=source,
                match_kind=match_kind,
                case_sensitive=bool(getattr(snippet, "case_sensitive", False)),
            )
            if stored_scope == requested_scope:
                return invocation
            if stored_scope == "global" and global_fallback is None:
                global_fallback = invocation
    return global_fallback


def _normalize_scopes(scopes: Sequence[str]) -> list[str]:
    out: list[str] = []
    for scope in scopes:
        normalized = (scope or "global").strip().lower() or "global"
        if normalized not in out:
            out.append(normalized)
    return out


def _scope_rank(scope: str, scopes: Sequence[str]) -> int:
    normalized = (scope or "global").strip().lower() or "global"
    try:
        return list(scopes).index(normalized)
    except ValueError:
        return 999


def _snippet_scope(snippet: object) -> str:
    return (str(getattr(snippet, "scope", "global") or "global").strip().lower() or "global")


def _clean_spoken_trigger(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" \t\r\n.,!?:;\"'")).strip()


__all__ = [
    "SnippetBodyProtection",
    "SnippetInvocation",
    "SnippetResolver",
    "SnippetTextCandidate",
    "expand_snippets",
    "looks_like_explicit_snippet_invocation",
    "resolve_snippet_invocation",
    "snippet_bodies_present_in_text",
]
