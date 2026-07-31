"""
tests/test_core.py
===================
Fast, dependency-light unit tests -- `pytest tests/ -v` from the repo root.
None of these need torch (the hook/entropy logic under test is
torch-independent by design -- see core/hooks.py) or a GPU.
`examples/run_demo.py` is the complementary end-to-end integration check;
these are the isolated per-function/class checks.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.hooks import (
    looks_like_attention_module, compute_attn_entropy, tensor_bytes,
    CacheTap, KVHookManager, verify_capture_integrity,
)
from core.cache_adapters import get_layer_kv, set_layer_kv, get_layer_meta, describe_cache
from core.inspector import KVCacheInspector
from core.storage import InMemoryEventSink, ExperienceDB, KnowledgeGraphDB
from compression.plugins import PLUGIN_REGISTRY, get_plugin
from harness.retry import RetryPolicy
from harness.budget import BudgetEnforcer, BudgetExceeded
from harness.evaluator import Evaluator


class _FakeAttentionModule:  # duck-typed torch.nn.Module stand-in -- no torch import needed
    num_heads = 4
    q_proj = k_proj = v_proj = object()


class _FakeLinearModule:
    pass


class _FakeHandle:
    def __init__(self, owner_list, fn):
        self._owner_list, self._fn = owner_list, fn

    def remove(self):
        if self._fn in self._owner_list:
            self._owner_list.remove(self._fn)


class _FakeTorchAttnModule:
    """Duck-typed stand-in for a torch.nn.Module attention block: just
    enough of register_forward_pre_hook / register_forward_hook /
    named_modules to drive KVHookManager.attach() and fire its hooks in the
    same order torch would, without needing torch installed. `fire()`
    simulates one forward() call."""

    def __init__(self, name: str = "self_attn"):
        self._name = name
        self._pre_hooks: list = []
        self._post_hooks: list = []

    def named_modules(self):
        yield self._name, self

    def register_forward_pre_hook(self, fn):
        self._pre_hooks.append(fn)
        return _FakeHandle(self._pre_hooks, fn)

    def register_forward_hook(self, fn):
        self._post_hooks.append(fn)
        return _FakeHandle(self._post_hooks, fn)

    def fire(self, output):
        for fn in list(self._pre_hooks):
            fn(self, (), None)
        for fn in list(self._post_hooks):
            fn(self, (), output)


# --------------------------------------------------------------------- #
# core/hooks.py -- pure, torch-independent logic
# --------------------------------------------------------------------- #

def test_looks_like_attention_module_by_class_name():
    class GemmaAttention(_FakeLinearModule):
        pass
    assert looks_like_attention_module("model.layers.0.self_attn", GemmaAttention())


def test_looks_like_attention_module_by_leaf_path_segment():
    """A generically-named class, but the conventional 'self_attn' leaf
    attribute name -- should still match via the leaf-segment check even
    though neither the class name nor duck-typing gives it away."""
    assert looks_like_attention_module("model.layers.0.self_attn", _FakeLinearModule())


def test_looks_like_attention_module_by_duck_typing():
    assert looks_like_attention_module("weird_name_123", _FakeAttentionModule())


def test_looks_like_attention_module_rejects_unrelated():
    assert not looks_like_attention_module("model.layers.0.mlp", _FakeLinearModule())


def test_looks_like_attention_module_rejects_nested_projection_children():
    """The leaf of "model.layers.0.self_attn.q_proj" is "q_proj", not
    "self_attn" -- should NOT match on its own (descendant-filtering in
    _find_attention_modules is the second line of defense; this is
    checking the predicate itself doesn't even need it for this case)."""
    assert not looks_like_attention_module("model.layers.0.self_attn.q_proj", _FakeLinearModule())


def test_looks_like_attention_module_rejects_post_attention_layernorm():
    """Regression guard: "post_attention_layernorm" contains "attention"
    as a substring (a standard name across the whole Llama/Gemma/Mistral/
    Qwen family) but its leaf segment doesn't END with "attn"/"attention",
    so the leaf-segment check correctly rejects it -- unlike a naive
    substring-anywhere-in-the-path check would."""
    assert not looks_like_attention_module(
        "model.layers.0.post_attention_layernorm", _FakeLinearModule()
    )


def test_find_attention_modules_ignores_nested_children_of_a_match():
    """Regression test for a real bug found running against real Gemma 3:
    an earlier matching rule matched on the full dotted path, so
    "model.layers.0.self_attn.q_proj" matched too (it contains "attn").
    Unguarded, every child of an attention block gets hooked alongside the
    block itself. Simulates the real Gemma3Attention shape: one attention
    block with 4 nested projection children."""
    class _Child(_FakeLinearModule):
        pass

    class _Model:
        def named_modules(self):
            attn = _FakeTorchAttnModule("model.layers.0.self_attn")
            yield "model.layers.0.self_attn", attn
            for child in ("q_proj", "k_proj", "v_proj", "o_proj"):
                yield f"model.layers.0.self_attn.{child}", _Child()
            yield "model.layers.0.mlp", _FakeLinearModule()

    mgr = KVHookManager(_Model(), run_id="t")
    found = mgr._find_attention_modules()
    assert [name for name, _ in found] == ["model.layers.0.self_attn"], (
        f"expected only the attention block itself, got: {[n for n, _ in found]}"
    )


def test_find_attention_modules_scoping_excludes_vision_tower():
    """Regression test for the actual real-world finding: google/gemma-3-4b-it
    is multimodal -- 27 SiglipAttention (vision tower) + 34 Gemma3Attention
    (language model) = 61 modules match by name/class alone, correctly
    (they ARE all attention modules). exclude_prefixes scopes discovery
    to just the part of the tree that matters for text-generation KV
    cache introspection."""
    class _Model:
        def named_modules(self):
            for i in range(3):
                yield f"model.vision_tower.encoder.layers.{i}.self_attn", _FakeTorchAttnModule()
            for i in range(2):
                yield f"model.language_model.layers.{i}.self_attn", _FakeTorchAttnModule()

    mgr = KVHookManager(_Model(), exclude_prefixes=["model.vision_tower"])
    found = [n for n, _ in mgr._find_attention_modules()]
    assert len(found) == 2
    assert all("language_model" in n for n in found)

    mgr2 = KVHookManager(_Model(), include_prefixes=["model.language_model"])
    found2 = [n for n, _ in mgr2._find_attention_modules()]
    assert found == found2


def test_compute_attn_entropy_uniform_beats_peaky():
    seq = 8
    uniform = np.ones((4, seq, seq)) / seq
    peaky = np.zeros((4, seq, seq)); peaky[..., 0] = 1.0
    assert compute_attn_entropy(uniform) > compute_attn_entropy(peaky)


def test_compute_attn_entropy_none_is_none():
    assert compute_attn_entropy(None) is None


def test_tensor_bytes_matches_numpy_nbytes():
    arr = np.zeros((4, 8, 16), dtype=np.float32)
    assert tensor_bytes(arr) == arr.nbytes


# --------------------------------------------------------------------- #
# CacheTap -- ground truth for KVEvents (isinstance-safe by design; see
# core/hooks.py's CacheTap docstring for why this matters)
# --------------------------------------------------------------------- #

class _DummyCache:
    def __init__(self):
        self.k = np.zeros((2, 0, 4), dtype=np.float32)

    def update(self, k, v, layer_idx, cache_kwargs=None):
        self.k = np.concatenate([self.k, k], axis=1)
        return self.k, self.k

    def get_seq_length(self, layer_idx=0):
        return self.k.shape[1]


def test_cache_tap_returns_the_same_object_mutated_not_a_wrapper():
    backing = _DummyCache()
    tapped = CacheTap(backing, InMemoryEventSink(), run_id="t")
    assert tapped is backing, "CacheTap must mutate in place, not return a separate wrapper object"


def test_cache_tap_survives_isinstance_checks():
    """The entire reason CacheTap was redesigned around retroactive
    subclassing instead of a __getattr__-delegating wrapper: code that
    does `isinstance(past_key_values, SomeCache)` (as transformers'
    generate() may, internally) must still see a real SomeCache instance."""
    backing = _DummyCache()
    tapped = CacheTap(backing, InMemoryEventSink(), run_id="t")
    assert isinstance(tapped, _DummyCache)
    assert type(tapped) is not _DummyCache          # it's a dynamic subclass...
    assert type(tapped).__mro__[1] is _DummyCache    # ...whose parent is the original class


def test_cache_tap_inherited_methods_still_work():
    backing = _DummyCache()
    tapped = CacheTap(backing, InMemoryEventSink(), run_id="t")
    k_new = np.random.default_rng(0).normal(size=(2, 3, 4)).astype(np.float32)
    tapped.update(k_new, k_new, 0)
    assert tapped.get_seq_length() == 3              # inherited, unmodified method


def test_cache_tap_writes_one_event_per_new_token():
    sink = InMemoryEventSink()
    tap = CacheTap(_DummyCache(), sink, run_id="t")
    k_new = np.random.default_rng(0).normal(size=(2, 5, 4)).astype(np.float32)
    tap.update(k_new, k_new, layer_idx=3)
    rows = sink.read_all()
    assert len(rows) == 5
    assert all(r["layer_idx"] == 3 for r in rows)
    assert [r["token_pos"] for r in rows] == [0, 1, 2, 3, 4]


def test_cache_tap_tracks_absolute_position_across_prefill_and_decode():
    sink = InMemoryEventSink()
    tap = CacheTap(_DummyCache(), sink, run_id="t")
    rng = np.random.default_rng(0)
    prefill = rng.normal(size=(2, 4, 4)).astype(np.float32)   # 4-token prefill
    decode1 = rng.normal(size=(2, 1, 4)).astype(np.float32)   # 1-token decode step
    decode2 = rng.normal(size=(2, 1, 4)).astype(np.float32)
    tap.update(prefill, prefill, layer_idx=0)
    tap.update(decode1, decode1, layer_idx=0)
    tap.update(decode2, decode2, layer_idx=0)

    rows = sink.read_all()
    positions = sorted(r["token_pos"] for r in rows)
    assert positions == [0, 1, 2, 3, 4, 5]
    sources = {r["token_pos"]: r["token_source"] for r in rows}
    assert sources[0] == sources[3] == "prompt"
    assert sources[4] == sources[5] == "generated"


def test_cache_tap_rejects_objects_without_update():
    class NotACache:
        pass
    with pytest.raises(TypeError, match="needs an object with an `update"):
        CacheTap(NotACache(), InMemoryEventSink())


def test_cache_tap_exposes_call_counts_for_cross_checking():
    """CacheTap tracks per-layer *call* counts (not just token/event
    counts) as a live attribute, specifically so it can be cross-checked
    against KVHookManager.call_counts -- calls vs calls, not tokens vs
    tokens, which is the comparison that actually catches "this layer's
    attention module fired but its cache never updated" or vice versa."""
    sink = InMemoryEventSink()
    cache = CacheTap(_DummyCache(), sink, run_id="t")
    cache.update(np.zeros((2, 4, 4), dtype=np.float32), np.zeros((2, 4, 4), dtype=np.float32), 0)
    cache.update(np.zeros((2, 1, 4), dtype=np.float32), np.zeros((2, 1, 4), dtype=np.float32), 0)
    cache.update(np.zeros((2, 1, 4), dtype=np.float32), np.zeros((2, 1, 4), dtype=np.float32), 1)
    assert cache._kv_introspector_call_counts == {0: 2, 1: 1}


class _Dummy4DCache:
    """Like _DummyCache but with a real heads axis -- [batch, heads, seq,
    head_dim] -- matching real transformers Cache tensors, for testing
    CacheTap's modifier wiring specifically (core/kv_modifiers.py's
    modifiers are written against this 4D shape)."""
    def __init__(self, heads=2, dim=4):
        self.k = np.zeros((1, heads, 0, dim), dtype=np.float32)
        self.v = np.zeros((1, heads, 0, dim), dtype=np.float32)

    def update(self, k, v, layer_idx, cache_kwargs=None):
        self.k = np.concatenate([self.k, k], axis=2)
        self.v = np.concatenate([self.v, v], axis=2)
        return self.k, self.v

    def get_seq_length(self, layer_idx=0):
        return self.k.shape[2]


def test_cache_tap_with_modifier_changes_what_gets_stored():
    from core.kv_modifiers import ZeroModifier
    sink = InMemoryEventSink()
    modifier = ZeroModifier(token_positions=[1])
    cache = CacheTap(_Dummy4DCache(), sink, run_id="t", modifier=modifier)

    k1 = np.random.default_rng(0).normal(size=(1, 2, 3, 4)).astype(np.float32)  # 3-token prefill
    k_out, v_out = cache.update(k1, k1, layer_idx=0)
    assert np.all(k_out[:, :, 1, :] == 0)              # position 1 was zeroed by the modifier
    assert not np.all(k_out[:, :, 0, :] == 0)            # position 0 untouched


def test_cache_tap_with_modifier_persists_across_future_calls():
    """The whole point of writing back into the cache's own storage: a
    zeroed token must STAY zeroed after further generation steps, not
    just affect the one forward pass where it was zeroed."""
    from core.kv_modifiers import ZeroModifier
    sink = InMemoryEventSink()
    modifier = ZeroModifier(token_positions=[0])
    cache = CacheTap(_Dummy4DCache(), sink, run_id="t", modifier=modifier)

    prefill = np.random.default_rng(0).normal(size=(1, 2, 3, 4)).astype(np.float32)
    cache.update(prefill, prefill, layer_idx=0)
    assert np.all(cache.k[:, :, 0, :] == 0)   # zeroed on the call where it was added

    decode1 = np.random.default_rng(1).normal(size=(1, 2, 1, 4)).astype(np.float32)
    k_out, _ = cache.update(decode1, decode1, layer_idx=0)
    # a LATER call, that never touched position 0 directly -- it must
    # still be zero, proving the write-back persisted rather than only
    # affecting the return value of the call where modify() ran.
    assert np.all(k_out[:, :, 0, :] == 0)
    assert np.all(cache.k[:, :, 0, :] == 0)


def test_cache_tap_without_modifier_is_unaffected():
    """No modifier passed -> behaves exactly as before (regression guard
    against the modifier wiring accidentally changing the no-modifier path)."""
    sink = InMemoryEventSink()
    cache = CacheTap(_Dummy4DCache(), sink, run_id="t")  # no modifier=
    k1 = np.random.default_rng(0).normal(size=(1, 2, 3, 4)).astype(np.float32)
    k_out, v_out = cache.update(k1, k1, layer_idx=0)
    np.testing.assert_array_equal(k_out, k1)
    rows = sink.read_all()
    assert len(rows) == 3


def test_verify_capture_integrity_matches_when_correct():
    sink = InMemoryEventSink()
    cache = CacheTap(_DummyCache(), sink, run_id="t")
    cache.update(np.zeros((2, 4, 4), dtype=np.float32), np.zeros((2, 4, 4), dtype=np.float32), 0)
    cache.update(np.zeros((2, 1, 4), dtype=np.float32), np.zeros((2, 1, 4), dtype=np.float32), 0)
    result = verify_capture_integrity(cache, sink, "t", n_layers=1)
    assert result["all_match"] is True
    assert result["per_layer"][0] == {"events": 5, "cache_reports": 5, "match": True}


def test_verify_capture_integrity_catches_a_mismatch():
    """If something wrote events that don't match the cache's own
    accounting (e.g. a stale/duplicate write, or a second write path
    active at the same time), this must catch it rather than pass green."""
    sink = InMemoryEventSink()
    cache = CacheTap(_DummyCache(), sink, run_id="t")
    cache.update(np.zeros((2, 4, 4), dtype=np.float32), np.zeros((2, 4, 4), dtype=np.float32), 0)
    # Simulate a rogue second writer duplicating an event for this run_id.
    from core.storage import KVEvent
    sink.write(KVEvent(run_id="t", ts=0.0, layer_idx=0, token_pos=99, token_source="generated",
                        kv_bytes=0, latency_ms=0.0))
    result = verify_capture_integrity(cache, sink, "t", n_layers=1)
    assert result["all_match"] is False
    assert result["per_layer"][0]["match"] is False


# --------------------------------------------------------------------- #
# KVHookManager -- storage-decoupled instrumentation only (no KVEvents;
# see core/hooks.py's module docstring for why CacheTap owns those now)
# --------------------------------------------------------------------- #

def test_kv_hook_manager_does_not_touch_storage():
    """KVHookManager has no sink parameter at all anymore -- this is
    intentional (Milestone 3: forward hooks become pure instrumentation),
    not an oversight. Confirms the constructor doesn't even accept one."""
    import inspect
    params = inspect.signature(KVHookManager.__init__).parameters
    assert "sink" not in params


def test_kv_hook_manager_collects_observations_in_memory():
    module = _FakeTorchAttnModule()

    class _FakeModel:
        def named_modules(self):
            return module.named_modules()

    mgr = KVHookManager(_FakeModel(), run_id="t", capture_attn_weights=True, prefill_len=4)
    mgr.attach()

    rng = np.random.default_rng(0)
    prefill_attn = rng.dirichlet(np.ones(4), size=(2, 4)).reshape(2, 4, 4)  # heads=2, q=4, kv=4
    decode_attn = rng.dirichlet(np.ones(5), size=(2, 1)).reshape(2, 1, 5)    # heads=2, q=1, kv=5
    module.fire((None, prefill_attn))
    module.fire((None, decode_attn))
    mgr.detach()

    assert len(mgr.observations) == 2
    assert mgr.observations[0].seq_len == 4
    assert mgr.observations[1].seq_len == 1
    assert mgr.call_counts == {0: 2}
    assert all(obs.attn_entropy is not None for obs in mgr.observations)


def test_kv_hook_manager_detach_removes_hooks():
    module = _FakeTorchAttnModule()

    class _FakeModel:
        def named_modules(self):
            return module.named_modules()

    mgr = KVHookManager(_FakeModel(), run_id="t")
    mgr.attach()
    assert module._pre_hooks and module._post_hooks
    mgr.detach()
    assert not module._pre_hooks and not module._post_hooks


# --------------------------------------------------------------------- #
# core/inspector.py
# --------------------------------------------------------------------- #

def test_inspector_l2_norms_shape_and_positivity():
    kv = np.random.default_rng(0).normal(size=(4, 10, 16)).astype(np.float32)
    norms = KVCacheInspector.l2_norms(kv)
    assert norms.shape == (10,)
    assert (norms > 0).all()


def test_inspector_cosine_similarity_diagonal_is_one():
    kv = np.random.default_rng(0).normal(size=(2, 5, 8)).astype(np.float32)
    sim = KVCacheInspector.cosine_similarity_matrix(kv)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)


def test_inspector_duplicate_clusters_finds_exact_duplicate():
    kv = np.random.default_rng(0).normal(size=(1, 5, 8)).astype(np.float32)
    kv[0, 1] = kv[0, 0]
    clusters = KVCacheInspector.duplicate_clusters(kv, threshold=0.999)
    assert any(set(c) == {0, 1} for c in clusters)


def test_inspector_token_source_split():
    sources = KVCacheInspector.token_source(list(range(6)), prefill_len=3)
    assert sources == ["prompt", "prompt", "prompt", "generated", "generated", "generated"]


def test_inspector_head_importance_shape():
    attn = np.random.default_rng(0).dirichlet(np.ones(6), size=(3, 6)).astype(np.float32)  # [heads=3,seq=6,seq=6]-ish
    imp = KVCacheInspector.head_importance(attn.reshape(3, 6, 6))
    assert imp.shape == (3,)


# --------------------------------------------------------------------- #
# compression/plugins.py
# --------------------------------------------------------------------- #

# Plugins that reconstruct every position (approximately) rather than
# dropping some outright -- decompress() should return the original shape.
_SHAPE_PRESERVING = {"fp8", "int4", "low_rank"}


@pytest.mark.parametrize("plugin_name", list(PLUGIN_REGISTRY))
def test_plugin_round_trip_shape(plugin_name):
    kv = np.random.default_rng(0).normal(size=(4, 12, 16)).astype(np.float32)
    kwargs = {
        "low_rank": {"rank": 4}, "sliding_window": {"window": 6},
        "token_eviction": {"keep_n": 6}, "head_pruning": {"drop": False},
    }.get(plugin_name, {})
    plugin = get_plugin(plugin_name)
    result = plugin.run(kv, **kwargs)
    assert result.compression_ratio > 0
    assert result.latency_ms >= 0
    recon = plugin.decompress(result)

    if plugin_name in _SHAPE_PRESERVING or plugin_name == "head_pruning":
        # head_pruning is configured with drop=False above, which zeroes
        # heads in place rather than slicing them out -- shape preserved.
        assert recon.shape == kv.shape
    else:
        # sliding_window / token_eviction drop whole tokens outright.
        # decompress() returns exactly what survived, NOT zero-padded back
        # to the original shape -- an earlier version did zero-pad, which
        # made "reconstruction error" actually measure the magnitude of
        # what was thrown away (misleading; caught running against real
        # Gemma). See each plugin's decompress() docstring.
        assert recon.shape[1] < kv.shape[1]
        assert recon.shape[0] == kv.shape[0] and recon.shape[2] == kv.shape[2]


def test_sliding_window_and_token_eviction_kept_values_are_exact():
    """Values that survive sliding_window/token_eviction are exact copies
    -- only whole tokens are dropped, never lossily transformed -- so
    decompress() should be right-valued, not just right-shaped."""
    kv = np.random.default_rng(0).normal(size=(2, 10, 8)).astype(np.float32)
    cases = [("sliding_window", {"window": 4}), ("token_eviction", {"keep_n": 4, "policy": "fifo"})]
    for name, kwargs in cases:
        plugin = get_plugin(name)
        result = plugin.run(kv, **kwargs)
        recon = plugin.decompress(result)
        keep_idx = result.data["keep_idx"]
        np.testing.assert_array_equal(recon, kv[:, keep_idx, :])


def test_head_pruning_drop_true_returns_kept_heads_only():
    kv = np.random.default_rng(0).normal(size=(4, 6, 8)).astype(np.float32)
    attn = np.random.default_rng(1).dirichlet(np.ones(6), size=(4, 6)).reshape(4, 6, 6)
    result = get_plugin("head_pruning").run(kv, attn_weights=attn, entropy_threshold=-1.0, drop=True)
    recon = get_plugin("head_pruning").decompress(result)
    assert recon.shape[0] == len(result.data["keep_idx"])
    np.testing.assert_array_equal(recon, kv[result.data["keep_idx"]])


def test_fp8_reconstruction_error_bounded():
    kv = np.random.default_rng(0).normal(size=(2, 5, 8)).astype(np.float32)
    result = get_plugin("fp8").run(kv)
    assert result.metadata["max_abs_err"] < 0.5  # loose bound -- fp8's mantissa is coarse by design


def test_int4_compression_ratio_plausible():
    kv = np.random.default_rng(0).normal(size=(4, 64, 32)).astype(np.float32)  # big enough that group overhead is small
    result = get_plugin("int4").run(kv)
    assert 3.0 < result.compression_ratio < 9.0  # ~8x theoretical (32bit->4bit) minus group overhead


def test_sliding_window_keeps_exactly_window_tokens():
    kv = np.random.default_rng(0).normal(size=(2, 20, 8)).astype(np.float32)
    result = get_plugin("sliding_window").run(kv, window=5, sink_tokens=0)
    assert result.data["kept"].shape[1] == 5


def test_head_pruning_never_drops_every_head():
    kv = np.random.default_rng(0).normal(size=(4, 6, 8)).astype(np.float32)
    attn = np.zeros((4, 6, 6), dtype=np.float32); attn[..., 0] = 1.0  # all heads maximally peaky (entropy 0)
    result = get_plugin("head_pruning").run(kv, attn_weights=attn, entropy_threshold=999.0)
    assert len(result.data["keep_idx"]) >= 1


# --------------------------------------------------------------------- #
# harness/retry.py + harness/budget.py
# --------------------------------------------------------------------- #

def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert RetryPolicy(max_attempts=3, backoff_seconds=0).run(flaky) == "ok"
    assert calls["n"] == 3


def test_retry_falls_back_after_exhausting_attempts():
    def always_fails():
        raise RuntimeError("persistent")

    result = RetryPolicy(max_attempts=3, backoff_seconds=0).run(always_fails, fallback=lambda: "fallback")
    assert result == "fallback"


def test_retry_reraises_without_fallback():
    def always_fails():
        raise RuntimeError("persistent")

    with pytest.raises(RuntimeError):
        RetryPolicy(max_attempts=2, backoff_seconds=0).run(always_fails)


def test_budget_enforcer_rejects_oversized_request():
    b = BudgetEnforcer(max_kv_bytes=100)
    with pytest.raises(BudgetExceeded):
        b.reserve(200)


def test_budget_enforcer_tracks_usage_and_rejects_cumulative_overflow():
    b = BudgetEnforcer(max_kv_bytes=100)
    b.reserve(60)
    assert b.utilization["bytes_used"] == 60
    with pytest.raises(BudgetExceeded):
        b.reserve(60)  # 60 + 60 > 100


# --------------------------------------------------------------------- #
# core/storage.py
# --------------------------------------------------------------------- #

def test_experience_db_round_trip(tmp_path):
    db = ExperienceDB(tmp_path / "exp.db")
    db.log("exp1", "fp8", {"a": 1}, 2.0, 0.3, 1.2, True, "lesson text")
    rows = db.recent(5)
    assert len(rows) == 1
    assert rows[0]["plugin"] == "fp8"
    assert rows[0]["lesson"] == "lesson text"


def test_mutation_db_round_trip(tmp_path):
    from core.storage import MutationDB
    db = MutationDB(tmp_path / "mut.db")
    db.log(run_id="r1", mutation_type="zero", config={"token_positions": [1]},
           prompt="[1,2,3]", baseline_output="[4,5]", mutated_output="[4,6]",
           latency_ms=1.2, integrity_passed=True)
    rows = db.recent(5)
    assert len(rows) == 1
    assert rows[0]["mutation_type"] == "zero"
    assert rows[0]["output_changed"] == 1
    assert rows[0]["integrity_passed"] == 1
    assert json.loads(rows[0]["config_json"]) == {"token_positions": [1]}


def test_mutation_db_detects_unchanged_output(tmp_path):
    from core.storage import MutationDB
    db = MutationDB(tmp_path / "mut2.db")
    db.log(run_id="r2", mutation_type="null", config={}, prompt="p",
           baseline_output="same", mutated_output="same", latency_ms=0.1, integrity_passed=True)
    assert db.recent(1)[0]["output_changed"] == 0


def test_knowledge_graph_entities_and_relationships(tmp_path):
    kg = KnowledgeGraphDB(tmp_path / "kg.db")
    kg.upsert_entity("Token", "token:0", position=0)
    kg.upsert_entity("Token", "token:1", position=1)
    kg.add_relationship("token:0", "token:1", "ATTENDED_BY", weight=0.9)
    assert kg.entity_count() == 2
    assert kg.relationship_count() == 1
    assert kg.neighbors("token:0")[0]["dst_key"] == "token:1"


# --------------------------------------------------------------------- #
# harness/evaluator.py
# --------------------------------------------------------------------- #

def test_evaluator_reports_na_not_misleading_error_for_shape_changing_plugins():
    """Regression test: a real run against Gemma 3 reported
    reconstruction_max_abs_err of 40-47 for sliding_window/token_eviction,
    which looked like broken math but was really just "the magnitude of
    the tokens that got dropped, compared against invented zeros". Large,
    real-scale values (Gemma's KV values run into the tens/hundreds) make
    this especially misleading. Confirms the fix: err is None, explained."""
    # Large-magnitude values, like real Gemma KV -- the old bug was easy to
    # miss on the synthetic model's tiny (~0.01-0.05) values but glaring here.
    kv = np.random.default_rng(0).normal(scale=30.0, size=(4, 10, 16)).astype(np.float32)
    evaluator = Evaluator(min_compression_ratio=1.3)

    for name, kwargs in [("sliding_window", {"window": 4}), ("token_eviction", {"keep_n": 4})]:
        report = evaluator.benchmark_plugin(get_plugin(name), kv, **kwargs)
        assert report.reconstruction_max_abs_err is None
        assert any("N/A" in n for n in report.notes)


def test_evaluator_still_reports_real_error_for_approximating_plugins():
    kv = np.random.default_rng(0).normal(scale=30.0, size=(4, 10, 16)).astype(np.float32)
    evaluator = Evaluator(min_compression_ratio=1.3)
    report = evaluator.benchmark_plugin(get_plugin("fp8"), kv)
    assert report.reconstruction_max_abs_err is not None
    assert report.reconstruction_max_abs_err >= 0


# --------------------------------------------------------------------- #
# core/cache_adapters.py -- normalizing across transformers Cache versions
# --------------------------------------------------------------------- #

def test_get_layer_kv_legacy_tuple_format():
    k = np.zeros((1, 4, 10, 256)); v = np.ones((1, 4, 10, 256))
    legacy_cache = ((k, v), (k * 2, v * 2))  # 2 layers
    got_k, got_v = get_layer_kv(legacy_cache, 1)
    np.testing.assert_array_equal(got_k, k * 2)
    np.testing.assert_array_equal(got_v, v * 2)


def test_get_layer_kv_dynamic_cache_layers_format():
    """Mirrors the real shape discovered running against Gemma 3 4B:
    cache.layers[i].keys / .values, plus sliding-window metadata."""
    class DynamicSlidingWindowLayer:
        def __init__(self):
            self.keys = np.zeros((1, 4, 10, 256))
            self.values = np.ones((1, 4, 10, 256))
            self.sliding_window = 1024
            self.cumulative_length = 10
            self.is_initialized = True

    class DynamicLayer:
        def __init__(self):
            self.keys = np.zeros((1, 4, 10, 256)) + 5
            self.values = np.ones((1, 4, 10, 256)) + 5
            self.cumulative_length = 10
            self.is_initialized = True
            # no .sliding_window attribute -- global layers don't have one

    class _FakeDynamicCache:
        def __init__(self):
            self.layers = [DynamicSlidingWindowLayer(), DynamicLayer()]

    cache = _FakeDynamicCache()

    k0, v0 = get_layer_kv(cache, 0)
    np.testing.assert_array_equal(k0, cache.layers[0].keys)
    k1, _ = get_layer_kv(cache, 1)
    np.testing.assert_array_equal(k1, cache.layers[1].keys)

    meta0 = get_layer_meta(cache, 0)
    assert meta0.layer_type == "sliding_window"
    assert meta0.window_size == 1024
    assert meta0.seq_len == 10

    meta1 = get_layer_meta(cache, 1)
    assert meta1.layer_type == "global"
    assert meta1.window_size is None


def test_get_layer_kv_simple_k_v_list_convention():
    """This repo's own convention (examples/synthetic_model.py's
    NumpyKVCache): cache.k / cache.v as lists indexed by layer_idx."""
    class _SimpleCache:
        def __init__(self):
            self.k = [np.zeros((2, 3, 4)), np.ones((2, 3, 4))]
            self.v = [np.zeros((2, 3, 4)), np.ones((2, 3, 4)) * 2]

    cache = _SimpleCache()
    k, v = get_layer_kv(cache, 1)
    np.testing.assert_array_equal(k, cache.k[1])
    np.testing.assert_array_equal(v, cache.v[1])


def test_set_layer_kv_simple_k_v_list_convention_round_trips():
    class _SimpleCache:
        def __init__(self):
            self.k = [np.zeros((2, 3, 4))]
            self.v = [np.zeros((2, 3, 4))]

    cache = _SimpleCache()
    new_k, new_v = np.ones((2, 3, 4)), np.ones((2, 3, 4)) * 5
    set_layer_kv(cache, 0, new_k, new_v)
    got_k, got_v = get_layer_kv(cache, 0)
    np.testing.assert_array_equal(got_k, new_k)
    np.testing.assert_array_equal(got_v, new_v)


def test_get_layer_kv_key_value_cache_list_format():
    class _FakeOlderDynamicCache:
        def __init__(self):
            self.key_cache = [np.zeros((1, 4, 5, 64)), np.ones((1, 4, 5, 64))]
            self.value_cache = [np.zeros((1, 4, 5, 64)), np.ones((1, 4, 5, 64))]

    cache = _FakeOlderDynamicCache()
    k, v = get_layer_kv(cache, 1)
    np.testing.assert_array_equal(k, cache.key_cache[1])
    np.testing.assert_array_equal(v, cache.value_cache[1])


def test_get_layer_kv_raises_clear_error_for_unknown_format():
    class _Mystery:
        pass

    with pytest.raises(TypeError, match="Don't know how to extract"):
        get_layer_kv(_Mystery(), 0)


def test_describe_cache_format():
    class DynamicSlidingWindowLayer:
        def __init__(self):
            self.keys = np.zeros((1, 2, 3, 8))
            self.sliding_window = 1024
            self.cumulative_length = 3

    class DynamicLayer:
        def __init__(self):
            self.keys = np.zeros((1, 2, 3, 8))
            self.cumulative_length = 3

    class _FakeCache:
        layers = [DynamicSlidingWindowLayer(), DynamicLayer()]

    desc = describe_cache(_FakeCache(), n_layers=2)
    assert "Layer 00" in desc and "Layer 01" in desc
    assert "Current Length : 3" in desc
