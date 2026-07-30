"""
core/cache_adapters.py
========================
Subproblem 1/5 extension -- the "Normalization" layer TODO.txt asked for.

Normalizes whatever Cache representation the installed `transformers`
version (or, longer-term, another runtime entirely) hands back into one
stable shape the rest of this repo already expects: a
[batch, kv_heads, seq_len, head_dim] tensor pair per layer, plus a
`LayerMeta` record of whatever richer metadata is available.

Why this exists: transformers' Cache API changed mid-project. The legacy
tuple-of-tuples format `((layer0_k, layer0_v), (layer1_k, layer1_v), ...)`
was replaced by a `DynamicCache` object exposing `.layers[i].keys` /
`.values` -- confirmed by actually running this against real Gemma 3 4B
(see CHANGELOG.md). That same object also exposes real per-layer metadata
this repo didn't have before: `.sliding_window`, `.cumulative_length`,
`.is_initialized`, and a layer *class*
(`DynamicSlidingWindowLayer` vs `DynamicLayer`) identifying which
attention pattern that layer uses -- Gemma 3 4B alternates 5
sliding-window layers per 1 global layer (confirmed: layers 5, 11, 17, 23,
29 of 34 are global). This module is the one place that knows about any
of this, so `core/inspector.py` and `compression/plugins.py` never have
to -- they only ever see a plain numpy array.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LayerMeta:
    """Whatever metadata is available for one cache layer, beyond the raw
    K/V tensors. Fields are None/"unknown" when the source cache format
    doesn't expose them (e.g. the legacy tuple format has none of this) --
    this is best-effort observability, not a required input."""
    layer_idx: int
    layer_type: str                        # "global" | "sliding_window" | "unknown"
    seq_len: int
    window_size: Optional[int] = None
    is_initialized: Optional[bool] = None


def get_layer_kv(cache: Any, layer_idx: int):
    """Returns (keys, values) as [batch, kv_heads, seq_len, head_dim]
    tensors for one layer, regardless of which Cache representation
    `cache` is. Batch dimension is deliberately NOT dropped here --
    callers wanting [kv_heads, seq_len, head_dim] should index [0]
    themselves, since batch>1 callers exist and shouldn't have that
    silently assumed away in a shared utility.

    Tries, in order (richest / most-current first):
      1. Legacy tuple: `((k0, v0), (k1, v1), ...)` -- pre-refactor
         transformers, or the output of `.to_legacy_cache()`.
      2. `.layers[i].keys` / `.values` -- current `DynamicCache`, confirmed
         against a real Gemma 3 4B run (transformers version current as of
         July 2026). Richest path: also where `get_layer_meta` reads from.
      3. `.key_cache[i]` / `.value_cache[i]` -- the `DynamicCache` shape
         from one refactor step before #2. Not exercised against a real
         checkpoint (transformers had already moved to `.layers` by the
         time this was written against a real model) -- included because
         it cost nothing to also handle and was a real public attribute
         for a while.
      4. `.to_legacy_cache()` -- last-resort flattening for any other
         Cache subclass that supports it.
    """
    if isinstance(cache, tuple):
        return cache[layer_idx][0], cache[layer_idx][1]

    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    if hasattr(cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()
        if legacy is not None:
            return legacy[layer_idx][0], legacy[layer_idx][1]

    raise TypeError(
        f"Don't know how to extract K/V from {type(cache).__name__} -- none of "
        "tuple / .layers[i] / .key_cache[i] / .to_legacy_cache() matched. This is "
        "exactly the kind of thing this adapter exists to isolate -- add a branch "
        "here rather than in core/inspector.py or compression/plugins.py."
    )


def get_layer_meta(cache: Any, layer_idx: int) -> LayerMeta:
    """Best-effort metadata for one layer. Never raises -- falls back to
    "unknown"/None fields rather than failing, since this is observability
    on top of get_layer_kv, not a required input to the rest of the
    pipeline."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        cls_name = type(layer).__name__
        if "SlidingWindow" in cls_name:
            layer_type = "sliding_window"
        elif "Layer" in cls_name:
            layer_type = "global"
        else:
            layer_type = "unknown"
        keys = getattr(layer, "keys", None)
        seq_len = int(keys.shape[-2]) if keys is not None else int(getattr(layer, "cumulative_length", 0))
        return LayerMeta(
            layer_idx=layer_idx, layer_type=layer_type, seq_len=seq_len,
            window_size=getattr(layer, "sliding_window", None),
            is_initialized=getattr(layer, "is_initialized", None),
        )

    k, _ = get_layer_kv(cache, layer_idx)
    return LayerMeta(layer_idx=layer_idx, layer_type="unknown", seq_len=int(k.shape[-2]))


def describe_cache(cache: Any, n_layers: int) -> str:
    """Human-readable per-layer summary -- the format TODO.txt asked for:

        Layer 00
          Type           : sliding_window
          Window Size    : 1024
          Current Length : 10
    """
    lines = []
    for i in range(n_layers):
        m = get_layer_meta(cache, i)
        block = [f"Layer {i:02d}", f"  Type           : {m.layer_type}"]
        if m.window_size is not None:
            block.append(f"  Window Size    : {m.window_size}")
        block.append(f"  Current Length : {m.seq_len}")
        lines.append("\n".join(block))
    return "\n".join(lines)
