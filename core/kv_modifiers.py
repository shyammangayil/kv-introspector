"""
core/kv_modifiers.py
=====================
Live KV cache interventions -- turns generation from something this
framework only *observes* into something it can *causally experiment on*.

Everything in `compression/plugins.py` benchmarks compress()/decompress()
against a captured snapshot -- reconstruction error, never touching actual
generation. A `KVModifier` is different: wired into `core.hooks.CacheTap`
(`CacheTap(cache, sink, modifier=...)`), it runs INSIDE the live
`Cache.update()` call, and its output is what the model's attention
actually reads next. "Zero token 4's KV, does the model still say
Ulaanbaatar" is a causal question about the model; "what's the max
reconstruction error of a rank-16 SVD of this KV block" is a question
about an offline buffer. Both are useful; only the first is what
mechanistic-interpretability-style intervention experiments need.

Design contract, matching what every implementation below relies on:
  - `modify(layer_idx, key, value, metadata)` receives the FULL current
    key/value for this layer -- shape [batch, kv_heads, seq_len,
    head_dim] -- AFTER this call's new tokens have already been appended,
    not just the newly-added slice. Modifiers that only care about new
    tokens can use `metadata["new_token_start"]` to slice; modifiers that
    need to touch history (swap two old tokens, decay old memories) can
    just index directly.
  - Returns `(key, value)` -- same shape, same framework (numpy in, numpy
    out; torch in, torch out). Every modifier here uses only indexing,
    multiplication, and addition, which behave identically on both, so
    none of them need to know or care which one they got.
  - `CacheTap` writes the returned tensors back into the cache's own
    storage (via `core.cache_adapters.set_layer_kv`), so a change persists
    across every future step, not just the current forward pass -- this
    is what makes `MemoryDecayModifier` (needs to keep compounding) and
    "zero a token, does that affect everything downstream" (needs the
    zeroing to still be there 5 tokens later) both work correctly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np


def _is_torch_tensor(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _clone(x):
    return x.clone() if _is_torch_tensor(x) else np.array(x, copy=True)


def _randn_like(x, rng: Optional[np.random.Generator] = None):
    if _is_torch_tensor(x):
        import torch
        return torch.randn(tuple(x.shape), dtype=x.dtype, device=x.device)
    r = rng or np.random.default_rng()
    return r.standard_normal(x.shape).astype(x.dtype)


class KVModifier(ABC):
    """Base class for live KV interventions. `name` shows up in
    `MutationDB` records (core/storage.py) -- override it per subclass."""
    name: str = "base"

    @abstractmethod
    def modify(self, layer_idx: int, key: Any, value: Any, metadata: dict) -> tuple[Any, Any]:
        """metadata keys guaranteed present: run_id, layer_idx,
        is_prefill, new_token_start, new_token_count, total_seq_len."""
        ...

    def applies_to_layer(self, layer_idx: int) -> bool:
        """Override to scope to specific layers; base class applies
        everywhere CacheTap's wrapped cache has a layer."""
        return True

    def describe(self) -> dict:
        """JSON-able summary of this modifier's config, for MutationDB
        (`core/storage.py::MutationDB.log`) -- override if a subclass adds
        fields beyond __dict__'s plain values."""
        return {"name": self.name, **{k: v for k, v in vars(self).items() if not k.startswith("_")}}


class NullModifier(KVModifier):
    """No-op -- useful as an explicit "baseline" run for
    MutationRunner.compare (see harness/mutation_runner.py) rather than
    special-casing `modifier=None` everywhere."""
    name = "null"

    def modify(self, layer_idx, key, value, metadata):
        return key, value


class ZeroModifier(KVModifier):
    """Zero specific token position(s) and/or head(s) -- the simplest
    causal-tracing-style intervention: does removing this token's/head's
    memory change generation? `token_positions`/`head_indices` are
    absolute indices into the FULL cache (not "how many steps ago")."""
    name = "zero"

    def __init__(
        self,
        token_positions: Optional[Sequence[int]] = None,
        head_indices: Optional[Sequence[int]] = None,
        layer_indices: Optional[Sequence[int]] = None,
    ):
        self.token_positions = list(token_positions) if token_positions is not None else None
        self.head_indices = list(head_indices) if head_indices is not None else None
        self.layer_indices = list(layer_indices) if layer_indices is not None else None

    def applies_to_layer(self, layer_idx):
        return self.layer_indices is None or layer_idx in self.layer_indices

    def modify(self, layer_idx, key, value, metadata):
        if not self.applies_to_layer(layer_idx):
            return key, value
        k, v = _clone(key), _clone(value)
        seq_len = k.shape[2]
        if self.head_indices is not None:
            for h in self.head_indices:
                k[:, h, :, :] = 0
                v[:, h, :, :] = 0
        if self.token_positions is not None:
            for t in self.token_positions:
                if 0 <= t < seq_len:
                    k[:, :, t, :] = 0
                    v[:, :, t, :] = 0
        return k, v


class NoiseModifier(KVModifier):
    """Add Gaussian noise -- sweep `sigma` to find the corruption level
    where output/perplexity starts to degrade. `sigma` is in the KV
    values' own units: real Gemma keys run roughly 20-40 in per-token L2
    norm across a 256-dim head (see op.txt), so sigma=0.01 is negligible
    there and sigma=1-5+ starts being meaningful; calibrate per model."""
    name = "noise"

    def __init__(
        self, sigma: float = 0.1, target: str = "both",
        layer_indices: Optional[Sequence[int]] = None, seed: Optional[int] = None,
    ):
        if target not in ("key", "value", "both"):
            raise ValueError(f"target must be 'key'/'value'/'both', got {target!r}")
        self.sigma = sigma
        self.target = target
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def applies_to_layer(self, layer_idx):
        return self.layer_indices is None or layer_idx in self.layer_indices

    def modify(self, layer_idx, key, value, metadata):
        if not self.applies_to_layer(layer_idx):
            return key, value
        k = key + _randn_like(key, self._rng) * self.sigma if self.target in ("key", "both") else key
        v = value + _randn_like(value, self._rng) * self.sigma if self.target in ("value", "both") else value
        return k, v


class SwapTokensModifier(KVModifier):
    """Swap two token positions' KV entries -- the kind of intervention
    used for causal-tracing experiments (does the model's output track
    the *position* or the *content* that was there?)."""
    name = "swap_tokens"

    def __init__(self, pos_a: int, pos_b: int, layer_indices: Optional[Sequence[int]] = None):
        self.pos_a, self.pos_b = pos_a, pos_b
        self.layer_indices = list(layer_indices) if layer_indices is not None else None

    def applies_to_layer(self, layer_idx):
        return self.layer_indices is None or layer_idx in self.layer_indices

    def modify(self, layer_idx, key, value, metadata):
        if not self.applies_to_layer(layer_idx):
            return key, value
        seq_len = key.shape[2]
        if self.pos_a >= seq_len or self.pos_b >= seq_len or self.pos_a == self.pos_b:
            return key, value  # target(s) not in the cache yet, or no-op swap -- skip rather than error
        k, v = _clone(key), _clone(value)
        idx = [self.pos_a, self.pos_b]
        idx_swapped = [self.pos_b, self.pos_a]
        k[:, :, idx, :] = k[:, :, idx_swapped, :]
        v[:, :, idx, :] = v[:, :, idx_swapped, :]
        return k, v


class HeadMaskModifier(KVModifier):
    """Scale specific head(s) by a factor: 0.0 drops a head entirely,
    1.0 is a no-op, >1.0 amplifies it. Defaults to scaling `value` only
    (matching the intuition that scaling V changes how much a head's
    attended-to content contributes to the residual stream, without also
    shifting *what* it attends to the way scaling K would -- a different,
    also-valid intervention, available via target="key"/"both")."""
    name = "head_mask"

    def __init__(
        self, head_indices: Sequence[int], scale: float, target: str = "value",
        layer_indices: Optional[Sequence[int]] = None,
    ):
        if target not in ("key", "value", "both"):
            raise ValueError(f"target must be 'key'/'value'/'both', got {target!r}")
        self.head_indices = list(head_indices)
        self.scale = scale
        self.target = target
        self.layer_indices = list(layer_indices) if layer_indices is not None else None

    def applies_to_layer(self, layer_idx):
        return self.layer_indices is None or layer_idx in self.layer_indices

    def modify(self, layer_idx, key, value, metadata):
        if not self.applies_to_layer(layer_idx):
            return key, value
        k, v = _clone(key), _clone(value)
        for h in self.head_indices:
            if self.target in ("key", "both"):
                k[:, h, :, :] = k[:, h, :, :] * self.scale
            if self.target in ("value", "both"):
                v[:, h, :, :] = v[:, h, :, :] * self.scale
        return k, v


class MemoryDecayModifier(KVModifier):
    """Progressively weakens older tokens' KV magnitude each step --
    approximates forgetting. Stateful by construction: every call decays
    whatever already existed (`metadata["new_token_start"]` tokens) by
    exactly one factor of `decay_per_step`, so a token ages through
    `decay_per_step ** n` after n further generation steps, compounding
    correctly WITHOUT needing to store pristine originals separately --
    each call only ever multiplies by the incremental per-step factor,
    never recomputes from scratch (which would double-decay already-
    decayed history)."""
    name = "memory_decay"

    def __init__(self, decay_per_step: float = 0.98, layer_indices: Optional[Sequence[int]] = None):
        if not (0.0 < decay_per_step <= 1.0):
            raise ValueError(f"decay_per_step should be in (0, 1], got {decay_per_step}")
        self.decay_per_step = decay_per_step
        self.layer_indices = list(layer_indices) if layer_indices is not None else None

    def applies_to_layer(self, layer_idx):
        return self.layer_indices is None or layer_idx in self.layer_indices

    def modify(self, layer_idx, key, value, metadata):
        if not self.applies_to_layer(layer_idx):
            return key, value
        pre_existing = metadata.get("new_token_start", 0)
        if pre_existing == 0:
            return key, value  # nothing existed before this call -- nothing to decay yet
        k, v = _clone(key), _clone(value)
        k[:, :, :pre_existing, :] = k[:, :, :pre_existing, :] * self.decay_per_step
        v[:, :, :pre_existing, :] = v[:, :, :pre_existing, :] * self.decay_per_step
        return k, v


class LowRankModifier(KVModifier):
    """Applies `compression.plugins.LowRankCompression`'s compress+
    decompress round-trip LIVE, every step, feeding the lossy
    reconstruction back into generation -- "compression in the loop"
    rather than benchmarking reconstruction error against a static
    snapshot. Reuses the existing plugin rather than reimplementing SVD
    truncation; converts to/from numpy at this boundary specifically
    (LowRankCompression is numpy-only) since this is a genuine
    framework-crossing point, not something worth forcing into the
    torch-agnostic style the other modifiers manage."""
    name = "low_rank_live"

    def __init__(self, rank: int = 16, layer_indices: Optional[Sequence[int]] = None):
        self.rank = rank
        self.layer_indices = list(layer_indices) if layer_indices is not None else None

    def applies_to_layer(self, layer_idx):
        return self.layer_indices is None or layer_idx in self.layer_indices

    def modify(self, layer_idx, key, value, metadata):
        if not self.applies_to_layer(layer_idx):
            return key, value
        from compression.plugins import LowRankCompression
        plugin = LowRankCompression()

        was_torch = _is_torch_tensor(key)
        k_np = key.detach().float().cpu().numpy()[0] if was_torch else np.asarray(key)[0]
        v_np = value.detach().float().cpu().numpy()[0] if was_torch else np.asarray(value)[0]

        k_recon = plugin.decompress(plugin.run(k_np, rank=self.rank))[None, ...]
        v_recon = plugin.decompress(plugin.run(v_np, rank=self.rank))[None, ...]

        if was_torch:
            import torch
            k_recon = torch.as_tensor(k_recon, dtype=key.dtype, device=key.device)
            v_recon = torch.as_tensor(v_recon, dtype=value.dtype, device=value.device)
        return k_recon, v_recon


MODIFIER_REGISTRY: dict[str, type] = {
    "null": NullModifier,
    "zero": ZeroModifier,
    "noise": NoiseModifier,
    "swap_tokens": SwapTokensModifier,
    "head_mask": HeadMaskModifier,
    "memory_decay": MemoryDecayModifier,
    "low_rank_live": LowRankModifier,
}
