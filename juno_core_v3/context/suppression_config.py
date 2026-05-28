from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SuppressionConfig:
    blocklist_bundle_ids: frozenset[str]
    warnlist_bundle_ids: frozenset[str]
    blocklist_window_title_patterns: tuple[re.Pattern[str], ...]

    @classmethod
    def default(cls) -> "SuppressionConfig":
        return cls(
            blocklist_bundle_ids=frozenset(),
            warnlist_bundle_ids=frozenset(),
            blocklist_window_title_patterns=(),
        )

    @classmethod
    def load(cls, path: Path | str) -> "SuppressionConfig":
        """Load from a JSON file. Raises FileNotFoundError if missing,
        ValueError on malformed JSON or unknown `version`."""
        p = Path(path)
        try:
            raw = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"suppression_config_malformed_json: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("suppression_config_malformed_json: root must be an object")

        version = data.get("version")
        if version != 1:
            raise ValueError(f"suppression_config_unsupported_version: {version}")

        block_raw = data.get("blocklist_bundle_ids", [])
        warn_raw = data.get("warnlist_bundle_ids", [])
        patterns_raw = data.get("blocklist_window_title_patterns", [])

        if not isinstance(block_raw, list):
            raise ValueError("suppression_config_malformed_json: blocklist_bundle_ids must be a list")
        if not isinstance(warn_raw, list):
            raise ValueError("suppression_config_malformed_json: warnlist_bundle_ids must be a list")
        if not isinstance(patterns_raw, list):
            raise ValueError(
                "suppression_config_malformed_json: blocklist_window_title_patterns must be a list"
            )

        blocklist_bundle_ids = frozenset(str(x).lower() for x in block_raw)
        warnlist_bundle_ids = frozenset(str(x).lower() for x in warn_raw)

        compiled: list[re.Pattern[str]] = []
        for pat in patterns_raw:
            pstr = str(pat)
            try:
                compiled.append(re.compile(pstr))
            except re.error as exc:
                raise ValueError(f"suppression_config_bad_regex: {pstr}: {exc}") from exc

        return cls(
            blocklist_bundle_ids=blocklist_bundle_ids,
            warnlist_bundle_ids=warnlist_bundle_ids,
            blocklist_window_title_patterns=tuple(compiled),
        )


__all__ = ["SuppressionConfig"]
