from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

from juno_v2.writer.backends.mlx_lm import (
    MlxLmWriterBackend,
    _MAX_PREFIX_CACHE_ENTRIES,
    _cache_token_offset,
)
from juno_v2.writer.config import WriterConfig


class _DummyArraysCache:
    """Simulates mlx-lm ArraysCache on hybrid SSM/linear-attention layers (no .offset)."""

    def __init__(self) -> None:
        self.steps = 0


class _DummyKVCache:
    """Simulates mlx-lm KVCache tracking token .offset."""

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset


def _install_fake_mlx_lm(
    monkeypatch: Any,
    *,
    hybrid_layer0: bool = False,
    trimmable: bool = True,
    broken_trim: bool = False,
    token_delay_s: float = 0.0,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "prefill_closed": 0,
        "decode_closed": 0,
        "trim_calls": [],
    }

    def make_prompt_cache(_model: Any) -> list[Any]:
        if hybrid_layer0:
            return [_DummyArraysCache(), _DummyKVCache(0)]
        return [_DummyKVCache(0)]

    def can_trim_prompt_cache(_cache: list[Any]) -> bool:
        return trimmable

    def trim_prompt_cache(cache: list[Any], num_to_trim: int) -> int:
        state["trim_calls"].append(num_to_trim)
        if broken_trim:
            return 0
        for layer in cache:
            if hasattr(layer, "offset"):
                layer.offset = max(0, int(layer.offset) - int(num_to_trim))
        return num_to_trim

    def stream_generate(
        _model: Any,
        _tok: Any,
        *,
        prompt: list[int],
        max_tokens: int,
        prompt_cache: list[Any],
        sampler: Any = None,
    ):
        del _model, _tok, sampler
        is_prefill = max_tokens in (0, 1)
        for layer in prompt_cache:
            if hasattr(layer, "offset"):
                layer.offset += len(prompt)
            if isinstance(layer, _DummyArraysCache):
                layer.steps += len(prompt)

        def _gen():
            import time

            try:
                for i in range(max_tokens):
                    for layer in prompt_cache:
                        if hasattr(layer, "offset"):
                            layer.offset += 1
                        if isinstance(layer, _DummyArraysCache):
                            layer.steps += 1
                    if token_delay_s > 0:
                        time.sleep(token_delay_s)
                    yield SimpleNamespace(text=f"tok{i}")
            finally:
                if is_prefill:
                    state["prefill_closed"] += 1
                else:
                    state["decode_closed"] += 1

        return _gen()

    mlx_lm_mod = types.ModuleType("mlx_lm")
    mlx_lm_mod.stream_generate = stream_generate  # type: ignore[attr-defined]

    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = make_prompt_cache  # type: ignore[attr-defined]
    cache_mod.can_trim_prompt_cache = can_trim_prompt_cache  # type: ignore[attr-defined]
    cache_mod.trim_prompt_cache = trim_prompt_cache  # type: ignore[attr-defined]

    sample_mod = types.ModuleType("mlx_lm.sample_utils")
    sample_mod.make_sampler = lambda **_kw: object()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_mod)
    return state


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


def test_cache_token_offset_handles_hybrid_layer0() -> None:
    cache = [_DummyArraysCache(), _DummyKVCache(42)]
    assert _cache_token_offset(cache) == 42
    assert _cache_token_offset([_DummyArraysCache()]) is None


def test_prefix_cache_rewinds_on_hybrid_model_and_closes_streams(monkeypatch: Any) -> None:
    state = _install_fake_mlx_lm(monkeypatch, hybrid_layer0=True)
    backend = MlxLmWriterBackend(WriterConfig(model_path="dummy"))
    backend._loaded = True
    backend._model = object()
    backend._tokenizer = _FakeTokenizer()

    out = backend._generate_with_prefix_cache(
        full_prompt="SYS:hello",
        prefix_str="SYS:",
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        deadline_s=None,
    )
    assert out is not None
    text, meta = out
    assert text == "tok0tok1tok2"
    assert meta["prompt_cache_hit"] is True
    assert meta["cached_prefix_tokens"] == 4
    assert meta["suffix_tokens"] == 5
    assert state["prefill_closed"] == 1
    assert state["decode_closed"] == 1

    # Cache offset on the attention layer must be rewound to exactly len("SYS:") == 4.
    entry = next(iter(backend._kv_caches.values()))
    assert _cache_token_offset(entry["cache"]) == 4


def test_prefix_cache_rewinds_on_deadline_break(monkeypatch: Any) -> None:
    state = _install_fake_mlx_lm(monkeypatch, hybrid_layer0=True, token_delay_s=0.01)
    backend = MlxLmWriterBackend(WriterConfig(model_path="dummy"))
    backend._loaded = True
    backend._model = object()
    backend._tokenizer = _FakeTokenizer()

    out = backend._generate_with_prefix_cache(
        full_prompt="SYS:world",
        prefix_str="SYS:",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        deadline_s=0.001,
    )
    assert out is not None
    _text, meta = out
    assert meta["timed_out"] is True
    assert state["decode_closed"] == 1
    entry = next(iter(backend._kv_caches.values()))
    assert _cache_token_offset(entry["cache"]) == 4


def test_prefix_cache_evicts_when_trim_cannot_restore_offset(monkeypatch: Any) -> None:
    _install_fake_mlx_lm(monkeypatch, broken_trim=True)
    backend = MlxLmWriterBackend(WriterConfig(model_path="dummy"))
    backend._loaded = True
    backend._model = object()
    backend._tokenizer = _FakeTokenizer()

    out = backend._generate_with_prefix_cache(
        full_prompt="SYS:world",
        prefix_str="SYS:",
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        deadline_s=None,
    )
    assert out is None
    assert len(backend._kv_caches) == 0


def test_prefix_cache_bounds_max_entries_and_clears_on_unload(monkeypatch: Any) -> None:
    _install_fake_mlx_lm(monkeypatch)
    backend = MlxLmWriterBackend(WriterConfig(model_path="dummy"))
    backend._loaded = True
    backend._model = object()
    backend._tokenizer = _FakeTokenizer()
    backend._split_prefix_memo["k"] = "v"

    for i in range(_MAX_PREFIX_CACHE_ENTRIES + 2):
        prefix = f"P{i}:"
        out = backend._generate_with_prefix_cache(
            full_prompt=f"{prefix}body",
            prefix_str=prefix,
            max_tokens=2,
            temperature=0.0,
            top_p=1.0,
            deadline_s=None,
        )
        assert out is not None

    assert len(backend._kv_caches) == _MAX_PREFIX_CACHE_ENTRIES
    backend.unload()
    assert len(backend._kv_caches) == 0
    assert len(backend._split_prefix_memo) == 0


def test_prefix_cache_snapshots_non_trimmable_hybrid_arrays_cache(monkeypatch: Any) -> None:
    state = _install_fake_mlx_lm(monkeypatch, hybrid_layer0=True, trimmable=False)
    backend = MlxLmWriterBackend(WriterConfig(model_path="dummy"))
    backend._loaded = True
    backend._model = object()
    backend._tokenizer = _FakeTokenizer()

    for suffix in ("hello", "world"):
        out = backend._generate_with_prefix_cache(
            full_prompt=f"SYS:{suffix}",
            prefix_str="SYS:",
            max_tokens=3,
            temperature=0.0,
            top_p=1.0,
            deadline_s=None,
        )
        assert out is not None
        text, meta = out
        assert text == "tok0tok1tok2"
        assert meta["prompt_cache_hit"] is True

    # Non-trimmable hybrid cache should never call trim_prompt_cache and its
    # stored prefix snapshot must retain both KVCache.offset == 4 and
    # ArraysCache.steps == 4 across repeated requests.
    assert state["trim_calls"] == []
    entry = next(iter(backend._kv_caches.values()))
    assert entry["trimmable"] is False
    assert _cache_token_offset(entry["cache"]) == 4
    assert entry["cache"][0].steps == 4

