from __future__ import annotations

from typing import Protocol

from juno_v2.contracts.final import FinalDecodeRequest, FinalDecodeResult


def effective_decode_language(req: FinalDecodeRequest, config_language: str | None) -> str | None:
    """Resolve the language hint for one ASR request.

    A request with no language and an auto-style policy means "let the ASR
    model detect it"; it must not silently fall back to a process-level English
    default, otherwise mixed Hindi/English speech is translated before History
    can preserve what was spoken. Fixed-language callers still pass the
    explicit code and keep the old deterministic behavior.
    """

    requested = (req.language or "").strip()
    if requested:
        lowered = requested.lower()
        if lowered in {"auto", "keep_original"} or lowered.startswith("pair:"):
            return None
        return requested

    policy = (req.language_policy or "").strip().lower()
    if policy in {"auto", "auto_supported", "pair", "keep_original"}:
        return None
    return config_language


class FinalAsrBackend(Protocol):
    backend_name: str

    def warm(self) -> None:
        raise NotImplementedError

    def decode(self, req: FinalDecodeRequest) -> FinalDecodeResult:
        raise NotImplementedError
