from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from juno_v2.personalization.seed.models import StructuredBiasBundle

_LOG = logging.getLogger(__name__)


class AsrBiasBackendCapabilities(Protocol):
    """Narrow capability probe for dictation backends."""

    supports_bias_phrases: bool
    supports_structured_term_hints: bool


@dataclass(slots=True)
class AsrBiasDelivery:
    """What the pipeline hands to ``transcribe_wav`` after adapter shaping."""

    initial_prompt: str | None
    bias_phrases: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_prompt": self.initial_prompt,
            "bias_phrases": list(self.bias_phrases),
            "metadata": dict(self.metadata),
        }


def adapt_structured_bundle_for_asr(
    *,
    base_initial_prompt: str | None,
    base_bias_phrases: list[str],
    structured: StructuredBiasBundle | None,
    capabilities: AsrBiasBackendCapabilities | None = None,
) -> AsrBiasDelivery:
    """Adapter boundary: structured hints vs string list vs no-op.

    When ``capabilities`` is ``None``, bias phrases are merged conservatively
    (prefix seed phrases before existing memory-derived phrases).
    """
    meta: dict[str, Any] = {
        "adapter": "juno_seed_asr_bias_adapter",
        "structured_present": structured is not None,
    }
    if structured is None:
        return AsrBiasDelivery(
            initial_prompt=base_initial_prompt,
            bias_phrases=list(base_bias_phrases),
            metadata={**meta, "mode": "no_structured_input"},
        )

    structured_dict = structured.to_dict()
    supports_terms = bool(getattr(capabilities, "supports_structured_term_hints", False))
    supports_phrases = bool(getattr(capabilities, "supports_bias_phrases", True))

    if supports_terms:
        meta["mode"] = "structured_terms_plus_phrases"
        meta["structured_bundle"] = structured_dict
        merged = list(structured.flattened_bias_phrases) + [p for p in base_bias_phrases if p]
        if not supports_phrases:
            _LOG.info("juno_asr_bias_adapter: backend lacks bias_phrases; structured only in metadata")
            return AsrBiasDelivery(
                initial_prompt=base_initial_prompt,
                bias_phrases=[],
                metadata={**meta, "bias_phrases_noop": True},
            )
        return AsrBiasDelivery(
            initial_prompt=base_initial_prompt,
            bias_phrases=merged,
            metadata=meta,
        )

    if not supports_phrases:
        _LOG.info("juno_asr_bias_adapter: backend unsupported for bias_phrases; no-op bias list")
        return AsrBiasDelivery(
            initial_prompt=base_initial_prompt,
            bias_phrases=[],
            metadata={**meta, "mode": "noop_unsupported", "clipping": structured.clipped},
        )

    merged = list(structured.flattened_bias_phrases) + [p for p in base_bias_phrases if p]
    meta["mode"] = "flattened_phrases_only"
    meta["clipping"] = structured.clipped
    meta["clip_reason"] = structured.clip_reason
    if structured.clipped:
        _LOG.info(
            "juno_asr_bias_adapter: bias bundle clipped reason=%s flat_len=%s",
            structured.clip_reason,
            len(structured.flattened_bias_phrases),
        )
    return AsrBiasDelivery(
        initial_prompt=base_initial_prompt,
        bias_phrases=merged,
        metadata=meta,
    )
