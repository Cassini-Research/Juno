from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    code: str
    name: str
    scripts: tuple[str, ...]


LANGUAGE_CATALOG: dict[str, LanguageSpec] = {
    'en': LanguageSpec('en', 'English', ('latin',)),
    'zh': LanguageSpec('zh', 'Mandarin Chinese', ('han',)),
    'es': LanguageSpec('es', 'Spanish', ('latin',)),
    'ja': LanguageSpec('ja', 'Japanese', ('han', 'kana')),
    'hi': LanguageSpec('hi', 'Hindi', ('devanagari',)),
    'ko': LanguageSpec('ko', 'Korean', ('hangul',)),
    'fr': LanguageSpec('fr', 'French', ('latin',)),
    'de': LanguageSpec('de', 'German', ('latin',)),
    'pt': LanguageSpec('pt', 'Portuguese', ('latin',)),
    'ar': LanguageSpec('ar', 'Arabic', ('arabic',)),
    'th': LanguageSpec('th', 'Thai', ('thai',)),
    'id': LanguageSpec('id', 'Indonesian', ('latin',)),
}

DEFAULT_SUPPORTED_LANGUAGES: tuple[str, ...] = (
    'en', 'zh', 'es', 'ja', 'hi', 'ko', 'fr', 'de', 'pt', 'ar', 'th', 'id'
)

DEFAULT_PAIR_PRESETS: dict[str, tuple[str, str]] = {
    'en_hi': ('en', 'hi'),
    'en_zh': ('en', 'zh'),
    'en_es': ('en', 'es'),
    'en_ja': ('en', 'ja'),
    'en_ko': ('en', 'ko'),
    'en_ar': ('en', 'ar'),
}

LEGACY_POLICY_ALIASES: dict[str, tuple[str, tuple[str, str] | None]] = {
    'code_switch_en_hi': ('pair', ('en', 'hi')),
    'code_switch_en_zh': ('pair', ('en', 'zh')),
    'code_switch_en_es': ('pair', ('en', 'es')),
    'code_switch_en_ja': ('pair', ('en', 'ja')),
    'code_switch_en_ko': ('pair', ('en', 'ko')),
    'code_switch_en_ar': ('pair', ('en', 'ar')),
}


def get_language_name(code: str | None) -> str:
    if not code:
        return 'Unknown language'
    spec = LANGUAGE_CATALOG.get(code)
    return spec.name if spec else code


def normalize_supported_languages(values: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        code = (raw or '').strip().lower()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def normalize_pair_languages(values: tuple[str, str] | list[str] | None) -> tuple[str, str] | None:
    if not values:
        return None
    cleaned = normalize_supported_languages(list(values))
    if len(cleaned) != 2:
        return None
    return cleaned[0], cleaned[1]


def parse_pair_policy_string(
    raw: str,
    *,
    supported: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, str]:
    """Strictly parse a ``pair:<a>,<b>`` language policy string.

    Raises ``ValueError`` on any malformed input so the product fails fast at
    startup instead of silently falling back to ``(en, hi)`` at runtime.

    The ``supported`` argument, when provided, restricts the accepted codes to
    that set; leave it ``None`` to accept any code registered in
    ``LANGUAGE_CATALOG``.
    """
    if not isinstance(raw, str):
        raise ValueError(f"pair policy must be a string, got {type(raw).__name__}")
    stripped = raw.strip().lower()
    if not stripped.startswith('pair:'):
        raise ValueError(f"pair policy must start with 'pair:', got {raw!r}")
    body = stripped.split(':', 1)[1]
    if ',' not in body:
        raise ValueError(
            f"pair policy must contain exactly one comma, got {raw!r} "
            "(example: 'pair:en,hi')"
        )
    parts = [item.strip() for item in body.split(',')]
    if len(parts) != 2:
        raise ValueError(
            f"pair policy must contain exactly two language codes, got {raw!r}"
        )
    code_a, code_b = parts
    if not code_a or not code_b:
        raise ValueError(
            f"pair policy language codes must be non-empty, got {raw!r}"
        )
    if code_a == code_b:
        raise ValueError(
            f"pair policy language codes must differ, got {raw!r}"
        )
    allowed_codes = (
        set(normalize_supported_languages(list(supported)))
        if supported is not None
        else set(LANGUAGE_CATALOG.keys())
    )
    missing = [c for c in (code_a, code_b) if c not in allowed_codes]
    if missing:
        raise ValueError(
            f"pair policy language codes not in supported set {sorted(allowed_codes)}: "
            f"{missing} (from {raw!r})"
        )
    return code_a, code_b
