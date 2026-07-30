"""
examples/synthetic_model.py
============================
A tiny, dependency-free (numpy only) decoder-only transformer, used to
validate the rest of this repo end-to-end *in this sandbox* -- where
installing PyTorch's PyPI wheel (which bundles multi-GB CUDA dependencies
even for CPU-only use, against ~10GB free disk here) was judged not worth
the risk. See README.md, "What was actually run vs. what needs your infra".

This is a REAL forward pass with real causal self-attention
(softmax(QK^T/sqrt(d))V) and a real, growing KV cache -- not mocked or
pre-baked data; every number `examples/run_demo.py` reports from this model
comes from actually running it. What is NOT real: the weights are random
(untrained), so attention patterns and generated token IDs carry no
semantic meaning. This validates the *architecture* end to end (hooking,
statistics, compression, debugging, storage), not model *quality*, and
absolute latency numbers are a toy model's, not Gemma's.

`NumpyKVCache.update(key_states, value_states, layer_idx, cache_kwargs)`
matches the interface `core.hooks.CacheTap` is written to wrap (the same
shape as HuggingFace `transformers.cache_utils.Cache`). `run_demo.py` wraps
a `NumpyKVCache` instance in a real `CacheTap`, so `core/hooks.py`'s actual
class is exercised here -- not a mirror of its pattern. Attention-weight
capture (entropy) still needs a separate hook, since CacheTap only sees
cache appends, not attention weights -- see `register_post_attn_hook`
below, which plays the same role `core.hooks.KVHookManager`'s forward hook
plays on a real `torch.nn.Module`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class LayerWeights:
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray


class NumpyKVCache:
    """Minimal stand-in for HF `transformers.cache_utils.Cache`:
    `update(key_states, value_states, layer_idx, cache_kwargs=None) ->
    (all_keys, all_values)`. This is the object `core.hooks.CacheTap`
    wraps; matching its interface here means CacheTap runs for real in
    this demo."""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        z = lambda: np.zeros((num_heads, 0, head_dim), dtype=np.float32)
        self.k = [z() for _ in range(num_layers)]
        self.v = [z() for _ in range(num_layers)]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.k[layer_idx] = np.concatenate([self.k[layer_idx], key_states], axis=1)
        self.v[layer_idx] = np.concatenate([self.v[layer_idx], value_states], axis=1)
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.k[layer_idx].shape[1]


class SyntheticTransformer:
    """`num_layers` layers, each causal multi-head self-attention plus a
    residual add (no MLP/LayerNorm -- the point is exercising the KV-cache
    path, not modeling quality)."""

    def __init__(self, num_layers=4, num_heads=4, head_dim=16, vocab_size=1000, seed=0):
        self.num_layers, self.num_heads, self.head_dim = num_layers, num_heads, head_dim
        self.hidden_dim = num_heads * head_dim
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self.embed = rng.normal(scale=0.02, size=(vocab_size, self.hidden_dim)).astype(np.float32)
        self.unembed = rng.normal(scale=0.02, size=(self.hidden_dim, vocab_size)).astype(np.float32)
        self.layers = [
            LayerWeights(*(
                rng.normal(scale=0.02, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32)
                for _ in range(4)
            ))
            for _ in range(num_layers)
        ]

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        seq = x.shape[0]
        return x.reshape(seq, self.num_heads, self.head_dim).transpose(1, 0, 2)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        heads, seq, hd = x.shape
        return x.transpose(1, 0, 2).reshape(seq, heads * hd)

    def project_qkv(self, layer_idx: int, x_new: np.ndarray):
        w = self.layers[layer_idx]
        return self._split_heads(x_new @ w.Wq), self._split_heads(x_new @ w.Wk), self._split_heads(x_new @ w.Wv)

    def attend(self, layer_idx: int, q: np.ndarray, k_all: np.ndarray, v_all: np.ndarray, causal_offset: int):
        n_new, seq_total = q.shape[1], k_all.shape[1]
        scores = np.einsum("hnd,hsd->hns", q, k_all) / np.sqrt(self.head_dim)
        query_pos = causal_offset + np.arange(n_new)[:, None]
        key_pos = np.arange(seq_total)[None, :]
        scores = np.where((key_pos <= query_pos)[None, :, :], scores, -1e9)
        attn = softmax(scores, axis=-1)
        out = np.einsum("hns,hsd->hnd", attn, v_all)
        out = self._merge_heads(out) @ self.layers[layer_idx].Wo
        return out, attn


class SyntheticHookableModel:
    """Wraps SyntheticTransformer with an injectable KV cache object (see
    NumpyKVCache above) so `core.hooks.CacheTap` can wrap *that* for real
    KV-append capture, plus a post-attention hook site for attention-weight
    derived metrics (entropy) that CacheTap alone can't see."""

    def __init__(self, transformer: SyntheticTransformer, cache=None):
        self.t = transformer
        self.cache = cache or NumpyKVCache(transformer.num_layers, transformer.num_heads, transformer.head_dim)
        self.seq_len = 0
        self._post_attn_hooks: list[Callable] = []

    def register_post_attn_hook(self, fn: Callable) -> None:
        self._post_attn_hooks.append(fn)

    def step(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: [n_new] int array (n_new > 1 => prefill, n_new == 1
        => one decode step). Returns logits [n_new, vocab_size]."""
        x = self.t.embed[np.asarray(token_ids)]
        causal_offset = self.seq_len
        for layer_idx in range(self.t.num_layers):
            q, k_new, v_new = self.t.project_qkv(layer_idx, x)
            t0 = time.perf_counter()
            k_all, v_all = self.cache.update(k_new, v_new, layer_idx)   # <- real CacheTap call site
            out, attn = self.t.attend(layer_idx, q, k_all, v_all, causal_offset)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            x = x + out
            for fn in self._post_attn_hooks:
                fn(layer_idx=layer_idx, attn_weights=attn, latency_ms=latency_ms,
                   causal_offset=causal_offset, n_new=len(token_ids))
        self.seq_len += len(token_ids)
        return x @ self.t.unembed

    def generate(self, prompt_ids, n_new_tokens: int, temperature: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> list[int]:
        rng = rng or np.random.default_rng(0)
        logits = self.step(np.asarray(prompt_ids, dtype=np.int64))[-1]
        generated = []
        for _ in range(n_new_tokens):
            probs = softmax(logits / temperature)
            next_id = int(rng.choice(self.t.vocab_size, p=probs))
            generated.append(next_id)
            logits = self.step(np.array([next_id], dtype=np.int64))[-1]
        return generated


def build_demo_model(seed: int = 0, cache=None) -> SyntheticHookableModel:
    """4 layers x 4 heads x 16 head_dim (hidden=64), vocab 1000 -- small
    enough to run in milliseconds on CPU, large enough that heads/layers/
    tokens are all meaningfully distinct for the debugger to show."""
    t = SyntheticTransformer(num_layers=4, num_heads=4, head_dim=16, vocab_size=1000, seed=seed)
    return SyntheticHookableModel(t, cache=cache)
