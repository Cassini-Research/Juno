from __future__ import annotations

import re
from dataclasses import replace

from juno_v2.contracts.context import RedactionSummary

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_DIGIT_SEQUENCE_RE = re.compile(r"\b\d{4,}\b")
_SECRET_RE = re.compile(r"\b(?:otp|passcode|password|secret|pin)\s*[:=-]?\s*\S+", re.I)


class ContextRedactor:
    def redact(self, text: str) -> tuple[str, RedactionSummary]:
        summary = RedactionSummary()
        result = text or ""
        result, n = _EMAIL_RE.subn("<email>", result)
        summary = replace(summary, emails=n)
        result, n = _URL_RE.subn("<url>", result)
        summary = replace(summary, urls=n, emails=summary.emails)
        result, n = _SECRET_RE.subn("<secret>", result)
        summary = replace(summary, urls=summary.urls, emails=summary.emails, secrets=n)
        result, n = _DIGIT_SEQUENCE_RE.subn("<digits>", result)
        summary = replace(
            summary,
            urls=summary.urls,
            emails=summary.emails,
            secrets=summary.secrets,
            digit_sequences=n,
        )
        return result, summary
