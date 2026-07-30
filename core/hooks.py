"""
core/hooks.py
=============
Model-agnostic runtime capture layer (Subproblem 1).

As of 2026-07-29 this module has a SINGLE source of truth for KVEvents:
`CacheTap`. `KVHookManager` (forward hooks on attention modules) is
demoted to optional, storage-decoupled instrumentation -- it collects
latency/entropy observations in memory for whoever wants to inspect them,
but it no longer writes to any sink. This split exists because of two real
problems found running this against actual Gemma 3 4B (full account in
CHANGELOG.md):

  1. Module discovery (`looks_like_attention_module` + `named_modules()`)
     is fundamentally fragile against a real model tree: it over-matched
     children of attention blocks at first (fixed), and -- more
     fundamentally -- `google/gemma-3-4b-it` is a MULTIMODAL checkpoint (a
     SigLIP vision tower bundled with the language model). Even correct
     attention-module discovery finds 27 vision-encoder attention blocks
     (`SiglipAttention`) alongside the 34 real language-model decoder
     layers (`Gemma3Attention`) -- 61 total, none of which is a "bug" in
     the matching logic; they're all genuinely attention modules. The
     vision ones just aren't what a KV-cache-for-text-generation tool
     cares about, and there's no name-pattern fix for that distinction in
     general (a different multimodal model could name things differently).
  2. `Cache.update(key_states, value_states, layer_idx, cache_kwargs)` is
     the actual, narrow interface every layer's KV append goes through --
     it isn't discovered by searching the module tree at all, so it never
     sees the vision tower's attention in the first place (SigLIP doesn't
     use this text-generation KV cache; there's nothing to filter out).

`CacheTap` is therefore the ground truth this repo is built around: wrap a
cache, hand it to `generate(past_key_values=...)`, and every real KV
append -- and only real KV appends -- produces a KVEvent. See
`verify_capture_integrity` at the bottom of this file for how to prove
that end to end against a real cache's own `get_seq_length()`.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from core.storage import EventSink, KVEvent

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only when torch is absent
    torch = None  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Pure helpers (torch-independent -> directly unit-testable)
# ---------------------------------------------------------------------------

_ATTENTION_CLASS_SUFFIXES = ("attention", "attn")


def looks_like_attention_module(name: str, module: Any) -> bool:
    """Predicate deciding whether a named submodule is an attention block.

    Three independent signals, any one sufficient:
    (a) the CLASS NAME ends with "attention"/"attn" -- covers
        GemmaAttention, Gemma2Attention, LlamaAttention, SiglipAttention.
    (b) the LEAF segment of the dotted path (`name.rsplit(".", 1)[-1]`,
        i.e. "self_attn" out of "model.layers.0.self_attn") ends with
        "attention"/"attn" -- catches the common case of a
        conventionally-named `self_attn` attribute wrapped in an
        otherwise generically-named class. Deliberately the LEAF segment
        only, not a substring search over the full path: an earlier
        version matched "attn"/"attention" *anywhere* in the full dotted
        path, which also matched every child of an attention block
        ("self_attn.q_proj" contains "attn" -- but its leaf, "q_proj",
        doesn't) and unrelated modules whose name happens to contain
        "attention" (e.g. "post_attention_layernorm" -- a standard name
        across the whole Llama/Gemma/Mistral/Qwen family; its leaf is
        "post_attention_layernorm" itself, which doesn't end with
        "attn"/"attention" either, so this still correctly rejects it).
    (c) duck-typed shape: exposes a heads-count attribute AND a
        q/k/v-projection attribute (covers cases where neither the class
        name nor the leaf path segment gives it away).

    Still matches vision-encoder attention on a multimodal checkpoint
    (SiglipAttention genuinely is an attention module) -- see
    `include_prefixes`/`exclude_prefixes` on `KVHookManager` for scoping
    to one part of a multimodal model tree.
    """
    cls_name = type(module).__name__.lower()
    if cls_name.endswith(_ATTENTION_CLASS_SUFFIXES):
        return True
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf.endswith(_ATTENTION_CLASS_SUFFIXES):
        return True
    has_head_attr = any(
        hasattr(module, attr)
        for attr in ("num_heads", "num_attention_heads", "num_key_value_heads")
    )
    has_qkv_attr = any(
        hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "qkv_proj")
    )
    return has_head_attr and has_qkv_attr


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor, duck-typed (no import needed)
        x = x.detach()
        if hasattr(x, "float"):
            x = x.float()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
    return np.asarray(x)


def compute_attn_entropy(attn_weights) -> Optional[float]:
    """Mean Shannon entropy (nats) of an attention distribution over its
    last axis, averaged over every other axis (batch, heads, query pos).
    Accepts numpy arrays or torch tensors. Returns None for None input."""
    if attn_weights is None:
        return None
    arr = _to_numpy(attn_weights)
    arr = np.clip(arr, 1e-12, 1.0)
    ent = -np.sum(arr * np.log(arr), axis=-1)
    return float(np.mean(ent))


def tensor_bytes(x) -> int:
    """Byte footprint of a tensor/array, torch-or-numpy agnostic."""
    if hasattr(x, "element_size") and hasattr(x, "nelement"):
        return int(x.element_size() * x.nelement())
    arr = _to_numpy(x)
    return int(arr.nbytes)


def l2_norm_per_token(x) -> np.ndarray:
    """L2 norm along the last (head_dim) axis for a [..., seq, head_dim]
    array, returning one norm per token position (averaged over any
    leading batch/head axes)."""
    arr = _to_numpy(x)
    norms = np.linalg.norm(arr, axis=-1)          # [..., seq]
    while norms.ndim > 1:
        norms = norms.mean(axis=0)
    return np.atleast_1d(norms)


# ---------------------------------------------------------------------------
# Instrumentation only -- NOT the ground truth. See module docstring.
# ---------------------------------------------------------------------------

@dataclass
class AttnObservation:
    """One forward-hook firing, kept in memory only. Deliberately NOT a
    KVEvent and NOT written to any sink -- CacheTap owns that."""
    layer_idx: int
    layer_name: str
    call_index: int
    latency_ms: float
    seq_len: int
    attn_entropy: Optional[float] = None


class KVHookManager:
    """Storage-decoupled instrumentation: forward hooks on attention
    modules, collecting latency + (optionally) attention entropy in
    memory. Does not write to a sink and does not produce KVEvents --
    `CacheTap` is the ground truth for those. Useful for:
      - Cross-checking CacheTap: did every attention module fire the same
        number of times as its layer's cache updated? (`call_counts`)
      - Attention entropy, which CacheTap can't see (it only observes
        cache appends, not attention weights).

    Use as a context manager so hooks are always removed:

        with KVHookManager(model, exclude_prefixes=["vision_tower"]) as mgr:
            model.generate(**inputs, output_attentions=True)
        print(mgr.call_counts)      # {layer_idx: n_forward_calls}
        print(mgr.observations)     # list[AttnObservation]

    `include_prefixes`/`exclude_prefixes` scope discovery to part of the
    module tree -- e.g. on a multimodal checkpoint,
    `include_prefixes=["language_model"]` (or `exclude_prefixes=
    ["vision_tower"]`) keeps this to the text decoder, matching what
    `CacheTap` naturally sees by construction.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        run_id: Optional[str] = None,
        capture_attn_weights: bool = True,
        prefill_len: Optional[int] = None,
        include_prefixes: Optional[list[str]] = None,
        exclude_prefixes: Optional[list[str]] = None,
    ):
        # No hard torch import-check here on purpose: nothing in this class
        # calls torch directly -- it only calls `.named_modules()` /
        # `.register_forward_pre_hook()` / `.register_forward_hook()` on
        # whatever `model` was passed in. A real torch.nn.Module provides
        # that API; so does the duck-typed fake in
        # tests/test_core.py::_FakeTorchAttnModule.
        self.model = model
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.capture_attn_weights = capture_attn_weights
        self.prefill_len = prefill_len
        self.include_prefixes = include_prefixes
        self.exclude_prefixes = exclude_prefixes or []
        self._handles: list = []
        self._layer_index: dict[str, int] = {}
        self._t0_by_module: dict[int, float] = {}
        self._call_count_by_layer: dict[int, int] = {}
        self.observations: list[AttnObservation] = []

    def _find_attention_modules(self) -> list[tuple[str, Any]]:
        candidates = [
            (name, module)
            for name, module in self.model.named_modules()
            if looks_like_attention_module(name, module)
        ]
        if self.include_prefixes:
            candidates = [(n, m) for n, m in candidates
                          if any(n.startswith(p) for p in self.include_prefixes)]
        if self.exclude_prefixes:
            candidates = [(n, m) for n, m in candidates
                          if not any(n.startswith(p) for p in self.exclude_prefixes)]

        # Guard against over-matching on nested paths: "self_attn.q_proj"
        # would match (a) too if self_attn itself didn't happen to also
        # match -- keep only outermost matches per subtree.
        names = {n for n, _ in candidates}

        def _is_descendant(n: str) -> bool:
            return any(n != other and n.startswith(other + ".") for other in names)

        return [(n, m) for n, m in candidates if not _is_descendant(n)]

    def attach(self) -> "KVHookManager":
        self._call_count_by_layer.clear()
        self.observations.clear()
        for idx, (name, module) in enumerate(self._find_attention_modules()):
            self._layer_index[name] = idx
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(module)))
            self._handles.append(module.register_forward_hook(self._post_hook(name, idx)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def hooked_layers(self) -> dict[str, int]:
        """{qualified_module_name: layer_idx} for everything currently
        hooked. `len(mgr.hooked_layers)` should equal the count of
        attention blocks you actually meant to instrument -- on a
        multimodal model, remember that's not necessarily
        `config.num_hidden_layers` unless you've scoped with
        include_prefixes/exclude_prefixes."""
        return dict(self._layer_index)

    @property
    def call_counts(self) -> dict[int, int]:
        """{layer_idx: number of forward-hook firings}. Compare against
        CacheTap's per-layer update() call count as a cross-check."""
        return dict(self._call_count_by_layer)

    def __enter__(self) -> "KVHookManager":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    def _pre_hook(self, module):
        def hook(mod, args, kwargs=None):
            self._t0_by_module[id(module)] = time.perf_counter()
        return hook

    def _post_hook(self, name: str, layer_idx: int):
        def hook(module, args, output):
            t0 = self._t0_by_module.pop(id(module), time.perf_counter())
            latency_ms = (time.perf_counter() - t0) * 1000.0

            attn_weights = None
            if self.capture_attn_weights and isinstance(output, tuple) and len(output) > 1:
                candidate = output[1]
                if candidate is not None and hasattr(candidate, "shape"):
                    attn_weights = candidate

            seq_len = int(attn_weights.shape[-2]) if attn_weights is not None else 1
            entropy = compute_attn_entropy(attn_weights) if attn_weights is not None else None
            self._call_count_by_layer[layer_idx] = self._call_count_by_layer.get(layer_idx, 0) + 1
            self.observations.append(AttnObservation(
                layer_idx=layer_idx, layer_name=name,
                call_index=self._call_count_by_layer[layer_idx],
                latency_ms=latency_ms, seq_len=seq_len, attn_entropy=entropy,
            ))
        return hook


# ---------------------------------------------------------------------------
# Ground truth: CacheTap
# ---------------------------------------------------------------------------

def _write_update_events(
    sink: EventSink, run_id: str, layer_idx: int, key_states, value_states,
    latency_ms: float, step_by_layer: dict,
) -> None:
    n_new = int(key_states.shape[-2])
    is_prefill = n_new > 1
    norms = l2_norm_per_token(key_states)
    kv_bytes_total = tensor_bytes(key_states) + tensor_bytes(value_states)
    bytes_per_token = kv_bytes_total // max(n_new, 1)
    base = step_by_layer.get(int(layer_idx), 0)
    for i in range(n_new):
        sink.write(KVEvent(
            run_id=run_id, ts=time.time(), layer_idx=int(layer_idx),
            token_pos=base + i, token_source="prompt" if is_prefill else "generated",
            kv_bytes=bytes_per_token, latency_ms=latency_ms / n_new,
            l2_norm=float(norms[i]) if i < len(norms) else float(np.mean(norms)),
        ))
    step_by_layer[int(layer_idx)] = base + n_new


class CacheTap:
    """The ground truth for KVEvents (see module docstring).

    `CacheTap(cache, sink, run_id=...)` does NOT return a `CacheTap`
    instance -- it mutates `cache` IN PLACE (reassigning its `__class__`
    to a dynamically-created subclass of `type(cache)` that overrides
    `update()`) and returns that same object. Deliberate, not an artifact:

    The original design (composition: a separate wrapper object
    delegating everything except `update` via `__getattr__`) does not
    survive `isinstance(x, Cache)` checks that `transformers.generate()`
    may perform internally on `past_key_values` -- wrapping doesn't change
    an object's real type, so `isinstance(wrapper, DynamicCache)` is False
    even though every call transparently reaches a real `DynamicCache`
    underneath. Retroactive subclassing doesn't have this problem: the
    returned object genuinely IS an instance of (a subclass of) the
    original cache's class, so every isinstance check and every inherited
    method (`get_seq_length`, `reorder_cache`, `crop`, ...) behaves
    exactly as on the untouched original -- there's nothing to delegate,
    because nothing was wrapped. Confirmed with a standalone reproduction
    of exactly this pattern before relying on it here; see CHANGELOG.md.

    Usage:
        from transformers import DynamicCache
        cache = CacheTap(DynamicCache(), sink)           # cache IS-A DynamicCache
        model.generate(**inputs, past_key_values=cache)   # every update() writes real KVEvents
    """

    def __new__(cls, cache: Any, sink: EventSink, run_id: Optional[str] = None):
        if not hasattr(cache, "update"):
            raise TypeError(
                f"CacheTap needs an object with an `update(key_states, value_states, "
                f"layer_idx, cache_kwargs=None)` method -- {type(cache).__name__} doesn't have one."
            )
        run_id_ = run_id or uuid.uuid4().hex[:8]
        step_by_layer: dict[int, int] = {}
        call_count_by_layer: dict[int, int] = {}
        base_cls = type(cache)

        class _Traced(base_cls):
            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                t0 = time.perf_counter()
                out = super().update(key_states, value_states, layer_idx, cache_kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                call_count_by_layer[int(layer_idx)] = call_count_by_layer.get(int(layer_idx), 0) + 1
                _write_update_events(sink, run_id_, layer_idx, key_states, value_states,
                                      latency_ms, step_by_layer)
                return out

        _Traced.__name__ = f"Traced{base_cls.__name__}"
        _Traced.__qualname__ = _Traced.__name__
        cache.__class__ = _Traced
        cache._kv_introspector_run_id = run_id_             # for callers that want it back
        cache._kv_introspector_call_counts = call_count_by_layer  # live dict, updates as generation proceeds
        return cache


def verify_capture_integrity(cache: Any, sink: EventSink, run_id: str, n_layers: int) -> dict:
    """Milestone-2-style proof: for each layer, does the highest
    `token_pos` this run wrote (+1) match that layer's own
    `get_seq_length(layer_idx)`? If CacheTap is capturing correctly, these
    can never disagree -- they're reading the same underlying growth from
    two different angles (events written vs. the cache's own accounting).

    Returns {"per_layer": {idx: {"events": int, "cache_reports": int,
    "match": bool}}, "all_match": bool}. Does not require torch -- `cache`
    just needs `get_seq_length(layer_idx)`; `events` just needs `layer_idx`
    and `token_pos` keys, so this works against the InMemoryEventSink-based
    tests too.
    """
    events = sink.read_all(run_id=run_id)
    per_layer = {}
    all_match = True
    for layer_idx in range(n_layers):
        layer_events = [e for e in events if e["layer_idx"] == layer_idx]
        events_count = (max((e["token_pos"] for e in layer_events), default=-1)) + 1
        try:
            cache_reports = int(cache.get_seq_length(layer_idx))
        except TypeError:
            cache_reports = int(cache.get_seq_length())  # some Cache versions take no arg
        match = events_count == cache_reports
        all_match = all_match and match
        per_layer[layer_idx] = {
            "events": events_count, "cache_reports": cache_reports, "match": match,
        }
    return {"per_layer": per_layer, "all_match": all_match}
