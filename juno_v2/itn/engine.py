"""ITN engine — routes text through the right profile(s).

Profile selection by app category (from context) and session class.
All transforms are deterministic and traceable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from juno_v2.itn.format_policy import ITNFormatPolicy
from juno_v2.itn.rules import (
    apply_code_identifiers,
    apply_currency,
    apply_dates,
    apply_email_url,
    apply_numeric,
    apply_spoken_punctuation,
    apply_terminal_ops,
    apply_times,
)


class ITNProfile(str, Enum):
    """Named ITN profile sets. Each selects which rules run."""

    PROSE = "prose"           # plain writing: numbers, dates, times, currency
    NUMERIC = "numeric"       # same as prose
    EMAIL_URL = "email_url"   # email + URL forms
    CODE = "code"             # code identifiers only (dot, underscore, dash, slash)
    TERMINAL = "terminal"     # terminal operators + code identifiers
    FULL = "full"             # all profiles combined
    NONE = "none"             # pass-through


@dataclass(slots=True)
class ITNResult:
    """Output of ITNEngine.run()."""

    text: str
    original_text: str
    profile: str
    rules_applied: list[str]
    changed: bool
    format_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "original_text": self.original_text,
            "profile": self.profile,
            "rules_applied": list(self.rules_applied),
            "changed": self.changed,
            "format": dict(self.format_snapshot),
        }


# Mapping from app_category string → recommended ITN profile
_CATEGORY_TO_PROFILE: dict[str, ITNProfile] = {
    "messaging": ITNProfile.PROSE,
    "email": ITNProfile.FULL,
    "docs": ITNProfile.FULL,
    "forms": ITNProfile.PROSE,
    "code": ITNProfile.CODE,
    "terminal": ITNProfile.TERMINAL,
    "unknown": ITNProfile.PROSE,
    "": ITNProfile.PROSE,
}


class ITNEngine:
    """Route text through ITN rules by profile.

    Usage::

        engine = ITNEngine()
        result = engine.run(text, profile=ITNProfile.FULL)
    """

    def profile_for_category(self, app_category: str | None) -> ITNProfile:
        """Choose the appropriate profile for an app category string."""
        return _CATEGORY_TO_PROFILE.get((app_category or "").lower(), ITNProfile.PROSE)

    def run(
        self,
        text: str,
        *,
        profile: ITNProfile | str = ITNProfile.PROSE,
        format_policy: ITNFormatPolicy | None = None,
    ) -> ITNResult:
        if isinstance(profile, str):
            try:
                profile = ITNProfile(profile)
            except ValueError:
                profile = ITNProfile.PROSE

        fmt = format_policy or ITNFormatPolicy.default()
        snap = fmt.to_summary_dict()

        if profile == ITNProfile.NONE or not text:
            return ITNResult(
                text=text,
                original_text=text,
                profile=profile.value if isinstance(profile, ITNProfile) else str(profile),
                rules_applied=[],
                changed=False,
                format_snapshot=snap,
            )

        out = text
        all_rules: list[str] = []

        if profile in (ITNProfile.CODE, ITNProfile.TERMINAL):
            if profile == ITNProfile.TERMINAL:
                out, r = apply_terminal_ops(out)
                all_rules.extend(r)
            out, r = apply_code_identifiers(out)
            all_rules.extend(r)
            # Terminal/code dictation needs spoken punctuation MORE than
            # prose ("quote fix colon preserve metadata quote" must become
            # "fix: preserve metadata"). Runs after the terminal/code passes
            # so their multi-word operator phrases are consumed first.
            out, r = apply_spoken_punctuation(out)
            all_rules.extend(r)
        else:
            # PROSE / NUMERIC / EMAIL_URL / FULL all run the numeric stack
            out, r = apply_currency(out, fmt)
            all_rules.extend(r)
            out, r = apply_dates(out, fmt)
            all_rules.extend(r)
            out, r = apply_times(out, fmt)
            all_rules.extend(r)
            out, r = apply_numeric(out)
            all_rules.extend(r)
            # Spoken punctuation runs after numeric/currency (so "two comma five"
            # stays a number) but before code-identifier collapse.
            out, r = apply_spoken_punctuation(out)
            all_rules.extend(r)

            if profile in (ITNProfile.EMAIL_URL, ITNProfile.FULL):
                out, r = apply_email_url(out)
                all_rules.extend(r)

            if profile == ITNProfile.FULL:
                out, r = apply_code_identifiers(out)
                all_rules.extend(r)

        return ITNResult(
            text=out,
            original_text=text,
            profile=profile.value,
            rules_applied=all_rules,
            changed=(out != text),
            format_snapshot=snap,
        )
