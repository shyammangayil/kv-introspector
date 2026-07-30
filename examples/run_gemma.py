"""
examples/run_gemma.py
======================
Real-Gemma counterpart to examples/run_demo.py: same downstream pipeline
(inspector -> compression -> storage -> debugger data), against an actual
downloaded Gemma checkpoint instead of the synthetic model.

ARCHITECTURE (2026-07-29): `CacheTap` wrapping a real `DynamicCache` is now
the SOLE source of KVEvents -- passed directly into
`generate(past_key_values=...)`. `KVHookManager` is optional,
storage-decoupled cross-check instrumentation (--cross-check), scoped to
just the language model. This replaces an earlier design that hooked
attention modules to produce KVEvents directly, which turned out to be
fragile in a way that only running against a real (and, it turned out,
multimodal) checkpoint exposed -- see CHANGELOG.md and
core/hooks.py's module docstring for the full account of why.

Before running, see README.md "Running this against real Gemma" for
getting HF access, picking a model size for your hardware, and installing
torch/transformers/accelerate.

Usage:
    python -m examples.run_gemma
    python -m examples.run_gemma --model google/gemma-3-4b-it --prompt "The capital of Mongolia is" --max-new-tokens 5
    python -m examples.run_gemma --cross-check   # also runs KVHookManager alongside, for the extra proof (slower)
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from core.storage import SQLiteEventSink, KnowledgeGraphDB
from core.hooks import CacheTap, KVHookManager, verify_capture_integrity
from core.cache_adapters import get_layer_kv, get_layer_meta, describe_cache
from core.inspector import KVCacheInspector
from core import visualize
from compression.plugins import PLUGIN_REGISTRY, get_plugin
from harness.evaluator import Evaluator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-3-270m-it",
                     help="HF model id. Defaults to the smallest real Gemma 3 checkpoint "
                          "(270M params) so a first run doesn't need a serious GPU. Bump up to "
                          "google/gemma-3-1b-it / -4b-it / -12b-it / -27b-it once this works. "
                          "-4b-it confirmed working (CPU) on 2026-07-28/29 -- note it's "
                          "multimodal (a bundled vision tower); see CHANGELOG.md.")
    ap.add_argument("--prompt", default="The capital of Mongolia is")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--inspect-layer", type=int, default=0,
                     help="Which transformer layer to run the Inspector/Compression sections on.")
    ap.add_argument("--device", default=None,
                     help="Force a device ('cpu', 'cuda:0', ...) instead of device_map='auto'. "
                          "'auto' will silently disk-offload layers that don't fit in GPU VRAM + "
                          "system RAM -- not just slow, it can also run generation on a layer "
                          "whose weights haven't finished swapping in. Force a device if loading "
                          "is unexpectedly slow, so it fails loudly on OOM instead.")
    ap.add_argument("--cross-check", action="store_true", default=False,
                     help="Also attach KVHookManager (scoped to the language model only, "
                          "excluding the vision tower) as an independent witness alongside "
                          "CacheTap, comparing per-layer call counts and adding attention "
                          "entropy. Needs eager attention + output_attentions=True, which is "
                          "much slower (~35s/generated-token observed for -4b-it on CPU) -- off "
                          "by default so a first run is fast; turn on once you want the proof.")
    args = ap.parse_args()

    # Imported here (not at module level) so `pytest` / `run_demo.py` keep working
    # without torch installed -- see requirements.txt.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    # Fresh data dir every run -- NOT accumulated across invocations. A real
    # bug lived here: `events_gemma.db` was never cleared between runs, and
    # `SQLiteEventSink.read_all()` never filtered by run_id, so every past
    # run's rows silently piled up in the same file. Several "event count
    # mismatch" investigations earlier in this project were partly chasing
    # that contamination without knowing it. Fixed on both ends now
    # (read_all(run_id=...) exists too, for when accumulation IS wanted --
    # e.g. comparing runs later) but this script defaults to a clean slate.
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    attn_impl = "eager" if args.cross_check else "sdpa"
    print(f"Loading {args.model} (attn_implementation={attn_impl}) ...")
    print("First run downloads weights from huggingface.co -- needs `huggingface-cli login` "
          "done and the model's license accepted first. See README.md.")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map=(args.device or "auto"),
        attn_implementation=attn_impl,
    )
    model.eval()

    # -it checkpoints load a composite Gemma3Config (text + vision
    # sub-configs) -- text-architecture fields live under get_text_config(),
    # not model.config directly. Version-robust: returns the nested config
    # on a composite checkpoint, model.config unchanged on a plain text model.
    text_cfg = model.config.get_text_config()
    n_layers = text_cfg.num_hidden_layers
    print(f"Loaded on: {model.device} | dtype: {model.dtype} | language-model layers: {n_layers} | "
          f"heads: {text_cfg.num_attention_heads} | "
          f"kv_heads: {getattr(text_cfg, 'num_key_value_heads', text_cfg.num_attention_heads)}")
    if hasattr(model.config, "vision_config"):
        print("Note: this checkpoint is multimodal (bundles a vision tower). CacheTap only ever "
              "sees the language model's cache updates -- the vision tower doesn't route through "
              "generate()'s KV cache at all for a text-only prompt, so it never enters the picture. "
              "(--cross-check's KVHookManager is explicitly scoped away from it too, in case you "
              "add image input later.)")

    run_id = f"gemma_{int(time.time())}"
    sink = SQLiteEventSink(DATA_DIR / "events_gemma.db")

    # ------------------------------------------------------------------ #
    # Milestone 1: CacheTap wrapping a real DynamicCache, passed directly
    # into generate(past_key_values=...). This is the ground truth.
    # ------------------------------------------------------------------ #
    # HuggingFace's own "Cache strategies" doc uses exactly this pattern:
    # `DynamicCache(config=model.config)` then passed straight into
    # `generate(past_key_values=...)`. Tried first, matching documented
    # usage; bare `DynamicCache()` as fallback in case a pinned older
    # transformers version doesn't accept `config=`. Neither path has been
    # run against a real checkpoint by this repo's own testing (no GPU/HF
    # access here) -- if this fails, transformers' own error message will
    # be far more current and specific than anything guessed in advance.
    try:
        base_cache = DynamicCache(config=text_cfg)
    except TypeError:
        base_cache = DynamicCache()
    cache = CacheTap(base_cache, sink, run_id=run_id)
    print(f"\nCacheTap ready: cache is now a {type(cache).__name__}, "
          f"isinstance(cache, DynamicCache) = {isinstance(cache, DynamicCache)} "
          "(must be True -- this is the whole reason CacheTap is built the way it is; "
          "see core/hooks.py's CacheTap docstring).")

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    hook_mgr = None
    if args.cross_check:
        hook_mgr = KVHookManager(
            model, run_id=run_id, capture_attn_weights=True, prefill_len=prompt_len,
            exclude_prefixes=["vision_tower", "model.vision_tower"],
        ).attach()
        print(f"Cross-check: KVHookManager hooked {len(hook_mgr.hooked_layers)} modules "
              f"(expect {n_layers}, language-model only).")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens, do_sample=False,
        return_dict_in_generate=True, past_key_values=cache,
    )
    if args.cross_check:
        gen_kwargs["output_attentions"] = True

    t0 = time.perf_counter()
    gen_out = model.generate(**inputs, **gen_kwargs)
    total_ms = (time.perf_counter() - t0) * 1000.0

    if hook_mgr is not None:
        hook_mgr.detach()

    text = tok.decode(gen_out.sequences[0], skip_special_tokens=True)
    generated_len = gen_out.sequences.shape[1] - prompt_len
    print(f"\nPrompt ({prompt_len} tok) + generated ({generated_len} tok): {text!r}")
    print(f"Wall-clock for generate(): {total_ms:.1f}ms")

    # ------------------------------------------------------------------ #
    # Milestone 2: prove the capture is correct -- not by eyeballing an
    # event count against a hand-derived formula (which is exactly what
    # went wrong before: the formula assumed every hooked "layer" behaves
    # identically, which broke the moment vision-tower modules and
    # projection-layer children were mixed into the same count), but by
    # checking every layer's captured span against that layer's OWN
    # `get_seq_length()` accounting.
    # ------------------------------------------------------------------ #
    events = sink.read_all(run_id=run_id)   # scoped to THIS run -- see the accumulation-bug note above
    print(f"\nCacheTap wrote {len(events)} KVEvents this run (run_id={run_id}).")
    integrity = verify_capture_integrity(cache, sink, run_id, n_layers=n_layers)
    print("Integrity check (events captured vs. cache.get_seq_length() per layer): "
          + ("ALL MATCH" if integrity["all_match"] else "MISMATCH -- see below"))
    if not integrity["all_match"]:
        for idx, info in integrity["per_layer"].items():
            if not info["match"]:
                print(f"  layer {idx}: {info}")

    if hook_mgr is not None:
        cache_calls = cache._kv_introspector_call_counts
        hook_calls = hook_mgr.call_counts
        # KVHookManager's layer_idx is its own enumeration order (0..n_layers-1
        # over whatever it hooked); CacheTap's layer_idx is the model's real
        # layer_idx argument. With --cross-check correctly scoped to just the
        # language model, these should be the same integers 0..n_layers-1 --
        # if they're not, that's worth knowing, not silently reconciled.
        agree = cache_calls == hook_calls
        print(f"\nCross-check -- CacheTap per-layer call counts: {cache_calls}")
        print(f"Cross-check -- KVHookManager per-layer call counts: {hook_calls}")
        print(f"Cross-check: {'AGREE' if agree else 'DISAGREE -- see counts above'}")

    # ------------------------------------------------------------------ #
    # Real captured KV tensors + rich per-layer metadata, both through
    # core.cache_adapters -- unchanged; still works identically whether
    # the cache came from CacheTap or anywhere else, since CacheTap
    # mutates in place rather than wrapping (cache IS the real DynamicCache).
    # ------------------------------------------------------------------ #
    layer_idx = min(args.inspect_layer, n_layers - 1)
    k_layer, v_layer = get_layer_kv(cache, layer_idx)
    k_layer = k_layer[0].detach().float().cpu().numpy()  # drop batch dim -> [kv_heads, seq, head_dim]

    print(f"\nCaptured real KV tensor, layer {layer_idx}: shape {k_layer.shape} "
          f"({k_layer.nbytes} bytes fp32) -- this is what core.inspector / compression.plugins run on next.")
    if getattr(text_cfg, "num_key_value_heads", None) and \
            text_cfg.num_key_value_heads != text_cfg.num_attention_heads:
        print(f"Note: Gemma uses grouped-query attention -- {text_cfg.num_key_value_heads} KV heads "
              f"shared across {text_cfg.num_attention_heads} query heads, so 'heads' here means KV heads.")

    print(f"\nPer-layer cache metadata (from the real DynamicCache CacheTap wrapped):")
    print(describe_cache(cache, n_layers))

    kg = KnowledgeGraphDB(DATA_DIR / "knowledge_graph_gemma.db")
    for i in range(n_layers):
        meta = get_layer_meta(cache, i)
        kg.upsert_entity("Layer", f"layer:{i}", layer_type=meta.layer_type,
                          window_size=meta.window_size, seq_len=meta.seq_len)
    kg.close()

    # ------------------------------------------------------------------ #
    # Inspector (Subproblem 2) -- unmodified from run_demo.py's synthetic path.
    # ------------------------------------------------------------------ #
    n_tokens = k_layer.shape[1]
    report = KVCacheInspector.full_report(k_layer, list(range(n_tokens)), prefill_len=prompt_len, current_step=n_tokens)
    print(f"\nLayer {layer_idx} per-token stats (first 10 of {n_tokens}, real Gemma keys):")
    for r in report[:10]:
        print(f"  token {r.token_pos:3d} [{r.token_source:>9}]  L2={r.l2_norm:7.3f}  var={r.variance:9.5f}  "
              f"nearest=tok{r.nearest_neighbor} (cos_sim={r.nearest_neighbor_sim:.3f})")

    pca_path = OUTPUT_DIR / f"gemma_pca_layer{layer_idx}.png"
    visualize.save(visualize.plot_pca_projection(k_layer, title=f"Gemma layer {layer_idx} key-vector PCA (real)"), str(pca_path))
    print(f"Saved: {pca_path}")

    # ------------------------------------------------------------------ #
    # Compression (Subproblem 3) -- unmodified; same 6 plugins, real Gemma KV.
    # ------------------------------------------------------------------ #
    evaluator = Evaluator(min_compression_ratio=1.3)
    print(f"\nCompression benchmark, layer {layer_idx}, real Gemma KV block {k_layer.shape}:")
    for name in PLUGIN_REGISTRY:
        kwargs = {"low_rank": {"rank": min(16, n_tokens - 1)}, "sliding_window": {"window": max(4, n_tokens // 2)},
                  "token_eviction": {"keep_n": max(4, n_tokens // 2)}}.get(name, {})
        rep = evaluator.benchmark_plugin(get_plugin(name), k_layer, **kwargs)
        err = rep.reconstruction_max_abs_err
        err_str = f"{err:8.5f}" if err is not None else "     N/A"
        print(f"  {name:>14}: {rep.compression_ratio:5.2f}x   latency {rep.latency_ms:7.4f}ms   "
              f"max err {err_str}   [{'PASS' if rep.passed else 'below threshold'}]")

    print(f"\nDone. events_gemma.db has {len(events)} real rows (this run only) -- point the debugger at it:\n"
          f"  streamlit run debugger/app_streamlit.py   (sidebar -> events.db path -> "
          f"data/events_gemma.db)\n"
          f"GEPA/Autoresearch integration (integrations/*.py) reads ExperienceDB/KnowledgeGraphDB "
          f"the same way regardless of where the numbers came from -- see run_demo.py Section 5 "
          f"for the wiring, unchanged here.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------- #
# Notes (read if something errors)
# ---------------------------------------------------------------------- #
# - `dtype=` replaces the older `torch_dtype=` kwarg on recent transformers;
#   if you're on an older pinned version and get a TypeError, swap it back.
# - `return_legacy_cache=True` was removed from `generate()` on current
#   transformers -- not needed anymore anyway, since CacheTap mutates the
#   real DynamicCache in place (`cache IS gen_out.past_key_values`
#   identity-wise), so core/cache_adapters.py::get_layer_kv reads it
#   directly rather than needing generate() to hand back a converted copy.
# - `DynamicCache(config=text_cfg)` is tried first -- this exact pattern
#   (`DynamicCache(config=model.config)` passed straight into
#   `generate(past_key_values=...)`) is HuggingFace's own documented usage
#   (https://huggingface.co/docs/transformers/en/kv_cache, "Cache
#   strategies"), not a guess. Bare `DynamicCache()` is the fallback, for
#   an older pinned transformers version that might not accept `config=`.
#   Neither path has been confirmed against a real checkpoint by this
#   repo's own testing (no GPU/HF access here) -- if it fails, transformers'
#   own error message will be far more current and specific than anything
#   guessed here in advance.
# - Gemma 3 alternates ~5 local sliding-window attention layers for every
#   1 global attention layer (confirmed on -4b-it: layers 5, 11, 17, 23, 29
#   of 34 are global, rest sliding-window with window=1024). Sliding-window
#   layers' cache stops growing once context exceeds the window; global
#   layers keep growing -- expect the Timeline Debugger to show different
#   per-layer growth curves than the synthetic model's uniform growth.
#   `describe_cache()` above prints this per layer, straight from the real
#   DynamicCache object CacheTap wrapped -- no separate bookkeeping needed.
# - `google/gemma-3-4b-it` (and presumably every -it size) is multimodal --
#   confirmed by hooking 61 attention modules (27 SiglipAttention + 34
#   Gemma3Attention) with an earlier, module-discovery-based version of
#   this script. CacheTap sidesteps this by construction (see
#   core/hooks.py's module docstring); --cross-check's KVHookManager needs
#   explicit `exclude_prefixes` to avoid it, which this script sets.
# - `--cross-check` is slow: ~35s/generated token observed for -4b-it on
#   CPU with eager attention + output_attentions=True. Expected (eager
#   attention materializes the full attention matrix, and
#   output_attentions disables some fused-kernel fast paths) -- without
#   --cross-check, CacheTap alone needs neither, so plain `sdpa` (the
#   default, fast backend) is used instead.
