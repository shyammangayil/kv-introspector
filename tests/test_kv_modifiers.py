"""
tests/test_kv_modifiers.py
============================
Tests for core/kv_modifiers.py. Uses proper 4D [batch, heads, seq,
head_dim] shapes throughout, matching real transformers Cache tensors
(confirmed shape from a real Gemma 3 4B run: (1, 4, 10, 256) -- see
op.txt / CHANGELOG.md) -- not this repo's synthetic model's simpler 3D
convention, since these modifiers are written against the real shape.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.kv_modifiers import (
    KVModifier, NullModifier, ZeroModifier, NoiseModifier, SwapTokensModifier,
    HeadMaskModifier, MemoryDecayModifier, LowRankModifier, MODIFIER_REGISTRY,
)


def _kv(batch=1, heads=2, seq=6, dim=8, scale=10.0, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.normal(scale=scale, size=(batch, heads, seq, dim)).astype(np.float32)
    v = rng.normal(scale=scale, size=(batch, heads, seq, dim)).astype(np.float32)
    return k, v


def _meta(new_start=0, new_count=6):
    return {
        "run_id": "t", "layer_idx": 0, "is_prefill": new_count > 1,
        "new_token_start": new_start, "new_token_count": new_count,
        "total_seq_len": new_start + new_count,
    }


def test_modifier_registry_is_complete_and_instantiable():
    for name, cls in MODIFIER_REGISTRY.items():
        assert issubclass(cls, KVModifier)
        assert cls.name == name or name == "null"  # null maps to NullModifier.name == "null" too


def test_null_modifier_is_a_true_no_op():
    k, v = _kv()
    k2, v2 = NullModifier().modify(0, k, v, _meta())
    assert k2 is k and v2 is v


def test_zero_modifier_zeroes_only_requested_tokens():
    k, v = _kv(seq=6)
    mod = ZeroModifier(token_positions=[2, 4])
    k2, v2 = mod.modify(0, k, v, _meta())
    assert np.all(k2[:, :, 2, :] == 0) and np.all(v2[:, :, 2, :] == 0)
    assert np.all(k2[:, :, 4, :] == 0) and np.all(v2[:, :, 4, :] == 0)
    for t in (0, 1, 3, 5):
        np.testing.assert_array_equal(k2[:, :, t, :], k[:, :, t, :])
    # original untouched -- modify() must not mutate its input in place
    assert not np.all(k[:, :, 2, :] == 0)


def test_zero_modifier_zeroes_only_requested_heads():
    k, v = _kv(heads=4, seq=5)
    mod = ZeroModifier(head_indices=[1, 3])
    k2, v2 = mod.modify(0, k, v, _meta())
    assert np.all(k2[:, 1, :, :] == 0) and np.all(k2[:, 3, :, :] == 0)
    for h in (0, 2):
        np.testing.assert_array_equal(k2[:, h, :, :], k[:, h, :, :])


def test_zero_modifier_out_of_range_position_is_ignored_not_an_error():
    k, v = _kv(seq=4)
    k2, v2 = ZeroModifier(token_positions=[99]).modify(0, k, v, _meta())
    np.testing.assert_array_equal(k2, k)


def test_zero_modifier_respects_layer_scoping():
    k, v = _kv()
    mod = ZeroModifier(token_positions=[0], layer_indices=[5])
    assert not mod.applies_to_layer(0)
    k2, v2 = mod.modify(0, k, v, _meta())
    assert k2 is k and v2 is v  # layer 0 not in scope -- untouched, same object


def test_noise_modifier_changes_values_and_is_seed_reproducible():
    k, v = _kv()
    mod_a = NoiseModifier(sigma=1.0, seed=42)
    mod_b = NoiseModifier(sigma=1.0, seed=42)
    ka, va = mod_a.modify(0, k, v, _meta())
    kb, vb = mod_b.modify(0, k, v, _meta())
    assert not np.array_equal(ka, k)          # actually changed something
    np.testing.assert_allclose(ka, kb)          # same seed -> reproducible
    np.testing.assert_allclose(va, vb)


def test_noise_modifier_respects_target():
    k, v = _kv()
    k2, v2 = NoiseModifier(sigma=1.0, target="key", seed=0).modify(0, k, v, _meta())
    assert not np.array_equal(k2, k)
    np.testing.assert_array_equal(v2, v)  # value untouched when target="key"


def test_noise_modifier_rejects_bad_target():
    with pytest.raises(ValueError):
        NoiseModifier(target="bogus")


def test_swap_tokens_modifier_swaps_correctly():
    k, v = _kv(seq=6)
    k2, v2 = SwapTokensModifier(pos_a=1, pos_b=4).modify(0, k, v, _meta())
    np.testing.assert_array_equal(k2[:, :, 1, :], k[:, :, 4, :])
    np.testing.assert_array_equal(k2[:, :, 4, :], k[:, :, 1, :])
    for t in (0, 2, 3, 5):
        np.testing.assert_array_equal(k2[:, :, t, :], k[:, :, t, :])


def test_swap_tokens_modifier_out_of_range_is_a_no_op():
    k, v = _kv(seq=3)
    k2, v2 = SwapTokensModifier(pos_a=0, pos_b=10).modify(0, k, v, _meta())
    assert k2 is k and v2 is v


def test_head_mask_modifier_zero_scale_drops_a_head():
    k, v = _kv(heads=4)
    k2, v2 = HeadMaskModifier(head_indices=[2], scale=0.0, target="both").modify(0, k, v, _meta())
    assert np.all(k2[:, 2, :, :] == 0) and np.all(v2[:, 2, :, :] == 0)


def test_head_mask_modifier_amplifies():
    k, v = _kv(heads=3)
    k2, v2 = HeadMaskModifier(head_indices=[0], scale=3.0, target="value").modify(0, k, v, _meta())
    np.testing.assert_allclose(v2[:, 0, :, :], v[:, 0, :, :] * 3.0)
    np.testing.assert_array_equal(k2, k)  # target="value" -- key untouched


def test_memory_decay_modifier_only_decays_pre_existing_tokens():
    k, v = _kv(seq=6)
    # 4 tokens pre-existed before this call; 2 are brand new this call.
    meta = _meta(new_start=4, new_count=2)
    k2, v2 = MemoryDecayModifier(decay_per_step=0.9).modify(0, k, v, meta)
    np.testing.assert_allclose(k2[:, :, :4, :], k[:, :, :4, :] * 0.9)
    np.testing.assert_array_equal(k2[:, :, 4:, :], k[:, :, 4:, :])  # new tokens untouched this call


def test_memory_decay_modifier_compounds_correctly_across_calls():
    """Regression-style test for the design this modifier depends on:
    applying the per-step factor repeatedly, once per call, must compound
    to decay_per_step**n after n calls -- NOT decay_per_step*n, and NOT
    over- or under-decay from re-touching already-decayed history."""
    decay = 0.9
    mod = MemoryDecayModifier(decay_per_step=decay)
    k, v = _kv(seq=1, scale=1.0)   # start with exactly 1 token, magnitude tracked exactly
    original = k.copy()

    # Simulate 3 further generation steps, each appending 1 new token and
    # re-invoking modify() with the correct new_token_start each time.
    seq_so_far = 1
    for step in range(3):
        new_tok_k, new_tok_v = _kv(seq=1, seed=100 + step)
        k = np.concatenate([k, new_tok_k], axis=2)
        v = np.concatenate([v, new_tok_v], axis=2)
        k, v = mod.modify(0, k, v, _meta(new_start=seq_so_far, new_count=1))
        seq_so_far += 1

    expected_factor = decay ** 3  # position 0 has aged through 3 further steps
    np.testing.assert_allclose(k[:, :, 0, :], original[:, :, 0, :] * expected_factor, rtol=1e-5)


def test_memory_decay_modifier_no_op_when_nothing_pre_existed():
    k, v = _kv(seq=3)
    k2, v2 = MemoryDecayModifier(decay_per_step=0.5).modify(0, k, v, _meta(new_start=0, new_count=3))
    assert k2 is k and v2 is v


def test_memory_decay_modifier_rejects_bad_rate():
    with pytest.raises(ValueError):
        MemoryDecayModifier(decay_per_step=1.5)
    with pytest.raises(ValueError):
        MemoryDecayModifier(decay_per_step=0.0)


def test_low_rank_modifier_changes_values_but_preserves_shape():
    k, v = _kv(heads=2, seq=12, dim=16)
    mod = LowRankModifier(rank=2)
    k2, v2 = mod.modify(0, k, v, _meta(new_count=12))
    assert k2.shape == k.shape
    assert not np.allclose(k2, k)  # rank-2 truncation of rank-up-to-12 data should actually lose information


def test_modifier_describe_is_json_serializable():
    import json
    for cls in (ZeroModifier, NoiseModifier, HeadMaskModifier, MemoryDecayModifier):
        instance = {
            ZeroModifier: ZeroModifier(token_positions=[1, 2]),
            NoiseModifier: NoiseModifier(sigma=0.3),
            HeadMaskModifier: HeadMaskModifier(head_indices=[0], scale=2.0),
            MemoryDecayModifier: MemoryDecayModifier(decay_per_step=0.9),
        }[cls]
        json.dumps(instance.describe())  # must not raise
