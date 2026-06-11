"""Curated AI/ML rare-spelling glossary used by the protected-term reconciler.

This list exists so the deterministic protected-term reconciler can repair
common ASR near-misses against AI/LLM domain vocabulary (Qwen, Llama,
Mixtral, MLX, GGUF, …) without the user having to add every term to their
personal memory. The previous gate hardcoded shape heuristics (e.g.
``q``-not-``u``) so it would match exactly one term (Qwen); this module
replaces those heuristics with an explicit, auditable list.

Curation rules:

- Domain-only — model names, frameworks, formats, techniques. No personal
  names. No company-specific identifiers that change quickly.
- Add a term only when (a) it has a distinctive spelling, and (b) ASR is
  known to misspell it. ``GPT`` is not in the list because plain Whisper
  rarely mishears it.
- Keep entries short: ≥3 chars, ≤24 chars. Longer phrases belong in a
  bias-prompt builder, not here.

The reconciler treats this glossary identically to user memory subterms.
"""

from __future__ import annotations

AI_GLOSSARY: frozenset[str] = frozenset(
    {
        # LLM family names
        "Qwen",
        "Kimi",
        "Llama",
        "Gemma",
        "Mistral",
        "Mixtral",
        "DeepSeek",
        "Phi",
        "Grok",
        "Falcon",
        "Vicuna",
        "Mamba",
        # "Yi" deliberately absent: 2 chars violates the ≥3 curation rule
        # above and makes near-miss repair too eager on short tokens.
        "Solar",
        "Zephyr",
        "Orca",
        "Hermes",
        "Marco",
        # Anthropic + OpenAI model surface (the rare-spelling ones)
        "Sonnet",
        "Opus",
        "Haiku",
        "Claude",
        # ML/MLX runtime + format names
        "MLX",
        "ONNX",
        "GGUF",
        "GGML",
        "vLLM",
        "TensorRT",
        "PyTorch",
        # ML techniques and abbreviations
        "LoRA",
        "QLoRA",
        "RAG",
        "RLHF",
        "DPO",
        "PPO",
        "MoE",
        "RoPE",
        "FlashAttention",
        "KVCache",
        # ASR / audio stack vocabulary
        "Whisper",
        "Silero",
        "Conformer",
        "Wav2Vec",
        "NeMo",
        # Vendors with distinctive spellings
        "Anthropic",
        "Moonshot",
        "Mistral",
    }
)


def is_ai_glossary_term(token: str) -> bool:
    """True if ``token`` matches an AI glossary entry case-insensitively."""

    if not token:
        return False
    folded = token.strip().casefold()
    return any(entry.casefold() == folded for entry in AI_GLOSSARY)
