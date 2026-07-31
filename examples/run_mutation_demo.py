"""
examples/run_mutation_demo.py
===============================
Proves core.kv_modifiers.KVModifier causally affects generation -- not
just that it's structurally wired into CacheTap, but that zeroing/
swapping/scaling/decaying real KV entries actually changes what a model
generates, compared to an identical run with no modifier. Fully
executable in this sandbox (pure numpy, no torch) -- see the module
docstring in `examples/synthetic_model.py` for why this repo validates
architecture this way when GPU/HF access isn't available.

Deliberately a SEPARATE tiny model from examples/synthetic_model.py, not
a reuse of it: that one uses a 3D [heads, seq, head_dim] cache convention
(no batch dim) to keep run_demo.py's already-validated pipeline simple.
core.kv_modifiers's modifiers are written against the REAL transformers
shape -- 4D, [batch, heads, seq, head_dim] (confirmed against actual
Gemma 3 4B: op.txt shows (1, 4, 10, 256)) -- so this demo uses that same
4D convention throughout, specifically so what gets proven here transfers
to the real-Gemma case in examples/run_gemma.py's --modifier flag,
without touching or risking run_demo.py's existing, validated pipeline.

Usage:
    python -m examples.run_mutation_demo
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.storage import InMemoryEventSink, MutationDB
from core.hooks import CacheTap, verify_capture_integrity
from core.kv_modifiers import (
    KVModifier, ZeroModifier, NoiseModifier, SwapTokensModifier,
    HeadMaskModifier, MemoryDecayModifier, LowRankModifier,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class Simple4DCache:
    """Minimal cache using this repo's own `.k`/`.v` list convention
    (core.cache_adapters.get_layer_kv/set_layer_kv's 4th path) -- but with
    a real batch axis, [1, heads, seq, head_dim], matching transformers'
    actual shape rather than examples/synthetic_model.py's simpler
    no-batch convention."""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.k = [np.zeros((1, num_heads, 0, head_dim), dtype=np.float32) for _ in range(num_layers)]
        self.v = [np.zeros((1, num_heads, 0, head_dim), dtype=np.float32) for _ in range(num_layers)]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.k[layer_idx] = np.concatenate([self.k[layer_idx], key_states], axis=2)
        self.v[layer_idx] = np.concatenate([self.v[layer_idx], value_states], axis=2)
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.k[layer_idx].shape[2]


class Tiny4DModel:
    """Real causal self-attention, real growing KV cache, deliberately
    tiny and untrained (random weights) -- see examples/synthetic_model.py
    for the full rationale on what a toy model like this does and doesn't
    prove. Greedy (argmax) decoding throughout: fully deterministic given
    the same cache/modifier, so ANY difference between two runs is
    attributable to the modifier, not sampling randomness."""

    def __init__(self, num_layers=4, num_heads=3, head_dim=8, vocab_size=30, seed=0):
        self.num_layers, self.num_heads, self.head_dim = num_layers, num_heads, head_dim
        self.hidden_dim = num_heads * head_dim
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self.embed = rng.normal(scale=0.5, size=(vocab_size, self.hidden_dim)).astype(np.float32)
        self.unembed = rng.normal(scale=0.5, size=(self.hidden_dim, vocab_size)).astype(np.float32)
        self.Wq = [rng.normal(scale=0.3, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32) for _ in range(num_layers)]
        self.Wk = [rng.normal(scale=0.3, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32) for _ in range(num_layers)]
        self.Wv = [rng.normal(scale=0.3, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32) for _ in range(num_layers)]
        self.Wo = [rng.normal(scale=0.3, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32) for _ in range(num_layers)]

    def _split_heads(self, x):
        seq = x.shape[0]
        return x.reshape(seq, self.num_heads, self.head_dim).transpose(1, 0, 2)

    def _merge_heads(self, x):
        h, seq, d = x.shape
        return x.transpose(1, 0, 2).reshape(seq, h * d)

    def _forward(self, x, cache, causal_offset):
        for layer_idx in range(self.num_layers):
            q = self._split_heads(x @ self.Wq[layer_idx])
            k_new = self._split_heads(x @ self.Wk[layer_idx])[None, ...]  # [1, heads, n, dim]
            v_new = self._split_heads(x @ self.Wv[layer_idx])[None, ...]
            k_all, v_all = cache.update(k_new, v_new, layer_idx)
            k_all_s, v_all_s = k_all[0], v_all[0]  # drop batch -> [heads, seq_total, dim]

            n_new, seq_total = q.shape[1], k_all_s.shape[1]
            scores = np.einsum("hnd,hsd->hns", q, k_all_s) / np.sqrt(self.head_dim)
            query_pos = causal_offset + np.arange(n_new)[:, None]
            key_pos = np.arange(seq_total)[None, :]
            scores = np.where((key_pos <= query_pos)[None, :, :], scores, -1e9)
            attn = _softmax(scores, axis=-1)
            out = np.einsum("hns,hsd->hnd", attn, v_all_s)
            out = self._merge_heads(out) @ self.Wo[layer_idx]
            x = x + out
        return x @ self.unembed

    def generate(self, prompt_ids, n_new: int, cache) -> list[int]:
        x = self.embed[np.asarray(prompt_ids)]
        logits = self._forward(x, cache, causal_offset=0)
        generated = []
        for _ in range(n_new):
            next_id = int(np.argmax(logits[-1]))
            generated.append(next_id)
            causal_offset = cache.get_seq_length(0)
            x = self.embed[np.asarray([next_id])]
            logits = self._forward(x, cache, causal_offset)
        return generated


@dataclass
class MutationExperiment:
    name: str
    modifier: KVModifier
    description: str


def run_one(model: Tiny4DModel, prompt: list[int], n_new: int, modifier=None, run_id="x"):
    sink = InMemoryEventSink()
    raw_cache = Simple4DCache(model.num_layers, model.num_heads, model.head_dim)
    cache = CacheTap(raw_cache, sink, run_id=run_id, modifier=modifier)
    t0 = time.perf_counter()
    output = model.generate(prompt, n_new, cache)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    integrity = verify_capture_integrity(cache, sink, run_id, n_layers=model.num_layers)
    return output, latency_ms, integrity["all_match"], cache


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = Tiny4DModel(num_layers=4, num_heads=3, head_dim=8, vocab_size=30, seed=0)
    prompt = [3, 7, 1, 9, 2, 5]   # 6 "prompt tokens" (meaningless IDs -- untrained model)
    n_new = 8
    mutation_db = MutationDB(DATA_DIR / "mutations_demo.db")

    print("=" * 78)
    print(f"Model: {model.num_layers} layers x {model.num_heads} heads x {model.head_dim} head_dim "
          f"(untrained, random weights -- real attention math, real growing 4D KV cache)")
    print(f"Prompt: {prompt} | generating {n_new} new tokens, greedy (deterministic)")
    print("=" * 78)

    baseline_out, baseline_ms, baseline_ok, _ = run_one(model, prompt, n_new, modifier=None, run_id="baseline")
    print(f"\nBASELINE (no modifier): {baseline_out}")
    print(f"  latency {baseline_ms:.3f}ms | integrity check: {'PASS' if baseline_ok else 'FAIL'}")

    experiments = [
        MutationExperiment("zero_first_prompt_token", ZeroModifier(token_positions=[0]),
                            "Zero token 0's KV in every layer -- does the first prompt token matter?"),
        MutationExperiment("zero_all_prompt_tokens", ZeroModifier(token_positions=list(range(len(prompt)))),
                            "Zero every prompt token's KV -- generation should be driven by ~nothing but position."),
        MutationExperiment("swap_first_and_last_prompt", SwapTokensModifier(pos_a=0, pos_b=len(prompt) - 1),
                            "Swap token 0 and token 5's KV -- causal-tracing-style intervention."),
        MutationExperiment("noise_sigma_2", NoiseModifier(sigma=2.0, seed=0),
                            "Add N(0, 2.0) noise to every K/V -- large relative to this model's ~0.3-scale weights."),
        MutationExperiment("drop_head_1_all_layers", HeadMaskModifier(head_indices=[1], scale=0.0),
                            "Zero out head 1's contribution (value only) in every layer -- does that head matter?"),
        MutationExperiment("amplify_head_0", HeadMaskModifier(head_indices=[0], scale=5.0),
                            "Amplify head 0's value contribution 5x in every layer."),
        MutationExperiment("memory_decay_fast", MemoryDecayModifier(decay_per_step=0.7),
                            "Aggressively decay older tokens each step -- approximates fast forgetting."),
        MutationExperiment("low_rank_live_rank1", LowRankModifier(rank=1),
                            "Feed rank-1-reconstructed KV back into generation every step (Example 7: "
                            "compression IN the loop, not just reconstruction-error benchmarking)."),
    ]

    print(f"\n{'Experiment':<28}{'Output':<28}{'Changed?':<10}{'Integrity':<10}Latency")
    print("-" * 90)
    changed_count = 0
    for exp in experiments:
        out, latency_ms, integrity_ok, _ = run_one(model, prompt, n_new, modifier=exp.modifier, run_id=exp.name)
        changed = out != baseline_out
        changed_count += int(changed)
        print(f"{exp.name:<28}{str(out):<28}{'YES' if changed else 'no':<10}"
              f"{'PASS' if integrity_ok else 'FAIL':<10}{latency_ms:.3f}ms")
        mutation_db.log(
            run_id=exp.name, mutation_type=exp.modifier.name, config=exp.modifier.describe(),
            prompt=str(prompt), baseline_output=str(baseline_out), mutated_output=str(out),
            latency_ms=latency_ms, integrity_passed=integrity_ok,
        )

    print("-" * 90)
    print(f"\n{changed_count}/{len(experiments)} interventions changed the greedy output vs. baseline.")
    print("(Not forced to 8/8 or 0/8 by design -- honestly reporting which interventions had a "
          "causal effect on THIS random-weight model is the actual result, not something to force "
          "either way. See CHANGELOG.md for what happened when this ran.)")

    print(f"\nMemory-decay compounding check: running a longer generation and confirming position 0's "
          f"key really does shrink by decay_per_step**n over n further steps (not just once).")
    long_cache = Simple4DCache(model.num_layers, model.num_heads, model.head_dim)
    long_sink = InMemoryEventSink()
    decay_mod = MemoryDecayModifier(decay_per_step=0.9)

    # Capture token 0's ORIGINAL (pre-decay) key by running an undecayed
    # cache alongside, same prompt/weights -- so this checks a real ratio
    # against a real baseline, not just prints the expected multiplier.
    undecayed_cache = Simple4DCache(model.num_layers, model.num_heads, model.head_dim)
    model.generate(prompt, 0, undecayed_cache)  # prefill only, no modifier, no decode steps yet
    original_k0_norm = float(np.linalg.norm(undecayed_cache.k[0][0, 0, 0, :]))

    tapped = CacheTap(long_cache, long_sink, run_id="decay_check", modifier=decay_mod)
    model.generate(prompt, 6, tapped)
    final_k0_norm = float(np.linalg.norm(long_cache.k[0][0, 0, 0, :]))
    actual_ratio = final_k0_norm / original_k0_norm
    expected_ratio = decay_mod.decay_per_step ** 6
    print(f"  Layer 0, head 0, token 0's key L2 norm: {original_k0_norm:.4f} (undecayed) -> "
          f"{final_k0_norm:.4f} (after 6 further decode steps)")
    print(f"  Actual ratio: {actual_ratio:.4f} | Expected (0.9**6): {expected_ratio:.4f} | "
          f"{'MATCH' if abs(actual_ratio - expected_ratio) < 1e-4 else 'MISMATCH -- see the math in MemoryDecayModifier'}")

    print(f"\nDone. {len(experiments) + 1} experiments recorded in {DATA_DIR / 'mutations_demo.db'} "
          f"(MutationDB) -- query it directly, or extend integrations/autoresearch_adapter.py's "
          f"Reflector pattern to summarize these the same way it does compression experience.")


if __name__ == "__main__":
    main()
