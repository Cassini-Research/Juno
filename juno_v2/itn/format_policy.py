"""ITN display policy for dates, times, and currency symbols.

Surfaces may set ``TypedContextBundle.metadata["locale_identifier"]`` (BCP-47
style, e.g. ``en_GB``) and/or ``metadata["itn_format"]`` with explicit keys.
Explicit ``itn_format`` always wins over locale-derived defaults.

All resolution is deterministic and offline — no network or OS calls on the
Python side beyond parsing strings already on the context bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from juno_v2.contracts.context import TypedContextBundle

DateStyle = Literal["us_medium", "iso", "dmy_slash", "mdy_slash", "dmy_long"]
ClockStyle = Literal["12h", "24h"]
CurrencyDecimalStyle = Literal["period", "comma"]

_DATE_STYLES: dict[str, DateStyle] = {
    "us_medium": "us_medium",
    "iso": "iso",
    "dmy_slash": "dmy_slash",
    "mdy_slash": "mdy_slash",
    "dmy_long": "dmy_long",
}
_CLOCK_STYLES: dict[str, ClockStyle] = {"12h": "12h", "24h": "24h"}
_CURRENCY_DECIMAL_STYLES: dict[str, CurrencyDecimalStyle] = {"period": "period", "comma": "comma"}


@dataclass(frozen=True, slots=True)
class ITNFormatPolicy:
    """Controls how ITN renders spoken dates/times/currency amounts."""

    date_style: DateStyle = "us_medium"
    clock: ClockStyle = "12h"
    currency_decimal: CurrencyDecimalStyle = "period"

    @staticmethod
    def default() -> ITNFormatPolicy:
        """Legacy Juno behaviour (US-centric prose dates, 12h clock, ``.`` decimals)."""
        return ITNFormatPolicy()

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "date_style": self.date_style,
            "clock": self.clock,
            "currency_decimal": self.currency_decimal,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> ITNFormatPolicy:
        """Build from a JSON-safe dict; unknown keys are ignored."""
        if not data:
            return cls.default()
        ds = str(data.get("date_style") or "").strip().lower()
        ck = str(data.get("clock") or "").strip().lower()
        cd = str(data.get("currency_decimal") or "").strip().lower()
        date_style = _DATE_STYLES.get(ds, "us_medium")
        clock = _CLOCK_STYLES.get(ck, "12h")
        currency_decimal = _CURRENCY_DECIMAL_STYLES.get(cd, "period")
        return cls(date_style=date_style, clock=clock, currency_decimal=currency_decimal)

    @classmethod
    def from_locale_identifier(cls, locale_id: str | None) -> ITNFormatPolicy:
        """Derive a reasonable policy from a BCP-47-ish identifier (``en_GB``, ``de-DE``)."""
        if not locale_id or not str(locale_id).strip():
            return cls.default()
        tag = str(locale_id).strip().replace("-", "_").lower()
        parts = tag.split("_")
        lang = parts[0] if parts else ""
        region = parts[-1] if len(parts) >= 2 else ""

        # European regions: day-first dates, 24h clock, comma decimals for currency.
        eu_regions = {
            "at", "be", "bg", "ch", "cz", "de", "dk", "ee", "es", "fi", "fr", "gr",
            "hr", "hu", "ie", "is", "it", "lt", "lu", "lv", "nl", "no", "pl", "pt",
            "ro", "se", "si", "sk",
        }
        if region in eu_regions or lang in ("de", "fr", "nl", "it", "es", "pt", "pl", "cs", "sk", "sl"):
            return cls(date_style="dmy_slash", clock="24h", currency_decimal="comma")

        # UK, AU, NZ, IN: day-first; keep period for £ — common in UI strings.
        if region in ("gb", "au", "nz", "in", "za", "sg", "my") or tag in ("en_gb", "en_au", "en_nz", "en_in"):
            return cls(date_style="dmy_long", clock="12h", currency_decimal="period")

        # Japan: ISO dates and 24h are typical in written output.
        if region == "jp" or lang == "ja":
            return cls(date_style="iso", clock="24h", currency_decimal="period")

        # US and unknown: preserve historical Juno defaults.
        return cls.default()


def resolve_itn_format_policy(context: TypedContextBundle | dict[str, Any] | None) -> ITNFormatPolicy:
    """Resolve ITN formatting from a :class:`~juno_v2.contracts.context.TypedContextBundle` or dict-like."""
    if context is None:
        return ITNFormatPolicy.default()
    meta: dict[str, Any] | None = None
    if hasattr(context, "metadata"):
        raw = getattr(context, "metadata")
        meta = dict(raw) if isinstance(raw, dict) else None
    elif isinstance(context, dict):
        meta = dict(context.get("metadata") or {})
    if not meta:
        return ITNFormatPolicy.default()
    explicit = meta.get("itn_format")
    if isinstance(explicit, dict) and explicit:
        return ITNFormatPolicy.from_mapping(explicit)
    loc = meta.get("locale_identifier")
    if isinstance(loc, str) and loc.strip():
        return ITNFormatPolicy.from_locale_identifier(loc.strip())
    return ITNFormatPolicy.default()
