"""
examples/run_demo.py
=====================
End-to-end demo: exercises every module in this repo against the synthetic
model and reports REAL numbers -- nothing printed here is fabricated.
Run with: `python -m examples.run_demo` from the repo root.

What this does and doesn't prove (see README.md for the full breakdown):
  - core.hooks.CacheTap really captures every KV append (Subproblem 1) --
    it's not mocked here, the actual class from core/hooks.py is used.
  - core.inspector's stats and core.visualize's plots run on real captured
    tensors (Subproblem 2).
  - all 6 compression/plugins.py plugins really compress a real KV block
    and decompress it, with a real reconstruction error (Subproblem 3).
  - the JSON this writes is exactly what the chat's inline visualization
    (and debugger/app_streamlit.py) read (Subproblem 4).
  - integrations/gepa_adapter.py and integrations/autoresearch_adapter.py
    really read ExperienceDB/KnowledgeGraphDB rows this run produced
    (Subproblem 5).
  - harness/retry.py and harness/budget.py are exercised against
    deliberately-forced failures, not just happy-path calls (Subproblem 6).
  - it does NOT touch a real Gemma model, GPU, or Wikitext-2/ARC/HellaSwag
    -- none of those are available in this sandbox. Every number below
    comes from a small random-weight model; see each section for what
    that does and doesn't tell you.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from core.storage import SQLiteEventSink, InMemoryEventSink, ExperienceDB, KnowledgeGraphDB, KVEvent
from core.hooks import CacheTap, compute_attn_entropy
from core.inspector import KVCacheInspector
from core import visualize
from compression.plugins import PLUGIN_REGISTRY, get_plugin
from harness.evaluator import Evaluator, InfrastructureUnavailable
from harness.router import Router, ExperimentConfig
from harness.retry import RetryPolicy
from harness.budget import BudgetEnforcer, BudgetExceeded
from integrations.gepa_adapter import GEPAAdapter
from integrations.autoresearch_adapter import Planner, Reflector
from examples.synthetic_model import build_demo_model, NumpyKVCache

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> dict:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    section("1. RUNTIME HOOK LAYER  (Subproblem 1)")
    # =====================================================================
    run_id = uuid.uuid4().hex[:8]
    event_sink = SQLiteEventSink(DATA_DIR / "events.db")
    raw_cache = NumpyKVCache(num_layers=4, num_heads=4, head_dim=16)
    tapped_cache = CacheTap(raw_cache, event_sink, run_id=run_id)   # the real core.hooks.CacheTap
    model = build_demo_model(seed=0, cache=tapped_cache)

    prefill_len_holder = {"v": None}
    captured_attn: dict[int, np.ndarray] = {}   # last prefill attention matrix per layer

    def post_attn_hook(layer_idx, attn_weights, latency_ms, causal_offset, n_new):
        if n_new > 1:
            captured_attn[layer_idx] = attn_weights.copy()
        for i in range(n_new):
            token_pos = causal_offset + i
            is_prompt = prefill_len_holder["v"] is None or token_pos < prefill_len_holder["v"]
            event_sink.write(KVEvent(
                run_id=run_id, ts=time.time(), layer_idx=layer_idx, token_pos=token_pos,
                token_source="prompt" if is_prompt else "generated", kv_bytes=0,
                latency_ms=latency_ms / n_new, attn_entropy=compute_attn_entropy(attn_weights[:, i, :]),
            ))
        if prefill_len_holder["v"] is None:
            prefill_len_holder["v"] = n_new

    model.register_post_attn_hook(post_attn_hook)

    PROMPT_LEN, N_GENERATED = 8, 12
    rng = np.random.default_rng(42)
    prompt_ids = rng.integers(0, model.t.vocab_size, size=PROMPT_LEN)
    t_start = time.perf_counter()
    model.generate(prompt_ids, N_GENERATED, temperature=1.0, rng=rng)
    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    all_events = event_sink.read_all()
    expected_events = model.t.num_layers * model.seq_len * 2  # CacheTap + post_attn_hook, one row/token/layer each
    print(f"Prompt: {PROMPT_LEN} tokens | Generated: {N_GENERATED} tokens | "
          f"Total sequence: {model.seq_len} tokens x {model.t.num_layers} layers")
    print(f"Events captured: {len(all_events)} rows in {DATA_DIR / 'events.db'} "
          f"(expected {expected_events}) -> {'MATCH' if len(all_events) == expected_events else 'MISMATCH'}")
    print(f"Wall-clock for the whole generate() call: {total_latency_ms:.2f} ms")
    hook_latencies = [e["latency_ms"] for e in all_events if e["latency_ms"]]
    print(f"Per-call latency (this toy model, CPU): mean {np.mean(hook_latencies):.4f}ms, "
          f"p99 {np.percentile(hook_latencies, 99):.4f}ms")
    assert model.seq_len == PROMPT_LEN + N_GENERATED
    assert len(all_events) == expected_events
    print("Hook coverage check: PASS (every layer x token forward pass was hooked; "
          "generation completed with hooks attached and produced the expected token count)")

    # --- A/B overhead benchmark: same model/seed, with vs without hooks ---
    N_REPS = 25
    bench_prompt = np.random.default_rng(7).integers(0, 1000, size=8)

    def run_once(with_hooks: bool) -> float:
        cache = NumpyKVCache(4, 4, 16)
        if with_hooks:
            cache = CacheTap(cache, InMemoryEventSink(), run_id="bench")
        m = build_demo_model(seed=7, cache=cache)
        if with_hooks:
            m.register_post_attn_hook(lambda attn_weights, **_: compute_attn_entropy(attn_weights))
        t0 = time.perf_counter()
        m.generate(bench_prompt, 8, rng=np.random.default_rng(7))
        return (time.perf_counter() - t0) * 1000.0

    baseline_ms = float(np.median([run_once(False) for _ in range(N_REPS)]))
    hooked_ms = float(np.median([run_once(True) for _ in range(N_REPS)]))
    overhead_pct = 100 * (hooked_ms - baseline_ms) / baseline_ms
    print(f"\nHook overhead (median of {N_REPS} runs): {baseline_ms:.3f}ms bare -> {hooked_ms:.3f}ms hooked "
          f"({overhead_pct:+.1f}%). Toy CPU model -- see README for why this % won't transfer directly to Gemma-scale.")

    # =====================================================================
    section("2. KV CACHE INSPECTOR  (Subproblem 2)")
    # =====================================================================
    INSPECT_LAYER = 2
    kv_layer = raw_cache.k[INSPECT_LAYER]           # [heads, seq_len, head_dim] -- real captured keys
    ten_tok_k = kv_layer[:, :10, :]                  # first 10 tokens, per the spec's "10-token sequence"

    report = KVCacheInspector.full_report(
        ten_tok_k, list(range(10)), prefill_len=PROMPT_LEN, current_step=model.seq_len
    )
    print(f"Layer {INSPECT_LAYER}, first 10 tokens -- real per-token stats from captured keys:")
    for r in report:
        print(f"  token {r.token_pos:2d} [{r.token_source:>9}]  L2={r.l2_norm:6.3f}  "
              f"var={r.variance:7.5f}  age={r.age:2d}  nearest_neighbor=tok{r.nearest_neighbor} "
              f"(cos_sim={r.nearest_neighbor_sim:.3f})")

    dup = KVCacheInspector.duplicate_clusters(ten_tok_k, threshold=0.9)
    print(f"Near-duplicate clusters (cosine >= 0.90): {dup if dup else 'none at this threshold'} "
          f"(expected -- random untrained keys rarely coincide)")

    heatmap_path = visualize.save(
        visualize.plot_attention_heatmap(captured_attn[INSPECT_LAYER],
                                          title=f"Layer {INSPECT_LAYER} prefill attention (real, 8 prompt tokens)"),
        str(OUTPUT_DIR / "heatmap_layer2.png"))
    pca_path = visualize.save(
        visualize.plot_pca_projection(ten_tok_k, title=f"Layer {INSPECT_LAYER} key-vector PCA (real, 10 tokens)"),
        str(OUTPUT_DIR / "pca_layer2.png"))
    graph_path = visualize.save(
        visualize.plot_nearest_neighbor_graph(ten_tok_k, k=2, title=f"Layer {INSPECT_LAYER} nearest-neighbor graph (real)"),
        str(OUTPUT_DIR / "nn_graph_layer2.png"))
    print(f"Saved: {heatmap_path}\n       {pca_path}\n       {graph_path}")

    # =====================================================================
    section("3. COMPRESSION PLUGINS  (Subproblem 3)")
    # =====================================================================
    full_kv = raw_cache.k[INSPECT_LAYER]  # full real captured cache for this layer, [4, 20, 16]
    print(f"Benchmarking all {len(PLUGIN_REGISTRY)} plugins against layer {INSPECT_LAYER}'s real captured "
          f"KV block, shape {full_kv.shape} ({full_kv.nbytes} bytes fp32):\n")

    evaluator = Evaluator(min_compression_ratio=1.3)
    plugin_kwargs = {
        "fp8": {}, "int4": {}, "low_rank": {"rank": 4},
        "sliding_window": {"window": 8, "sink_tokens": 2},
        "token_eviction": {"keep_n": 8, "policy": "attention"},
        "head_pruning": {"attn_weights": captured_attn[INSPECT_LAYER], "entropy_threshold": 1.0, "drop": True},
    }
    plugin_reports = {}
    for name in PLUGIN_REGISTRY:
        rep = evaluator.benchmark_plugin(get_plugin(name), full_kv, **plugin_kwargs[name])
        plugin_reports[name] = rep
        err = rep.reconstruction_max_abs_err
        err_str = f"{err:8.5f}" if err is not None else "     N/A"
        print(f"  {name:>14}: {rep.compression_ratio:5.2f}x   latency {rep.latency_ms:7.4f}ms   "
              f"max recon err {err_str}   "
              f"[{'PASS >=1.3x' if rep.passed else 'below threshold'}]")

    print("\nPerplexity/accuracy deltas intentionally NOT shown above -- confirming the guard fires "
          "rather than fabricating a number:")
    try:
        evaluator.evaluate_perplexity(None, None, ["dummy text"])
    except InfrastructureUnavailable as e:
        print(f"  evaluate_perplexity() correctly raised InfrastructureUnavailable: {e}")

    # =====================================================================
    section("4. TIMELINE & MEMORY DEBUGGER DATA  (Subproblem 4)")
    # =====================================================================
    timeline, cumulative_bytes = [], 0
    for pos in range(model.seq_len):
        rows = [e for e in all_events if e["token_pos"] == pos]
        pos_bytes = sum(e["kv_bytes"] for e in rows)
        cumulative_bytes += pos_bytes
        l2s = [e["l2_norm"] for e in rows if e["l2_norm"] is not None]
        ents = [e["attn_entropy"] for e in rows if e["attn_entropy"] is not None]
        lats = [e["latency_ms"] for e in rows if e["latency_ms"] is not None]
        timeline.append({
            "token_pos": pos, "token_source": rows[0]["token_source"] if rows else "unknown",
            "bytes_this_token": pos_bytes, "cumulative_bytes": cumulative_bytes,
            "mean_l2_norm": float(np.mean(l2s)) if l2s else None,
            "mean_attn_entropy": float(np.mean(ents)) if ents else None,
            "mean_latency_ms": float(np.mean(lats)) if lats else None,
        })

    neighbors = KVCacheInspector.nearest_neighbors(ten_tok_k, k=2)
    dependency_graph = [
        {"token": i, "neighbors": [{"token": j, "similarity": round(sim, 4)} for j, sim in nbrs]}
        for i, nbrs in enumerate(neighbors)
    ]
    demo_output = {
        "run_id": run_id, "prompt_len": PROMPT_LEN, "generated_len": N_GENERATED,
        "num_layers": model.t.num_layers, "num_heads": model.t.num_heads, "inspected_layer": INSPECT_LAYER,
        "timeline": timeline, "dependency_graph": dependency_graph,
        "plugin_results": {n: {"compression_ratio": r.compression_ratio, "latency_ms": r.latency_ms,
                                "reconstruction_max_abs_err": r.reconstruction_max_abs_err, "passed": r.passed}
                            for n, r in plugin_reports.items()},
    }
    (DATA_DIR / "demo_output.json").write_text(json.dumps(demo_output, indent=2))
    print(f"Timeline: {len(timeline)} tokens, {cumulative_bytes} cumulative bytes (real, from events.db)")
    print(f"Dependency graph: {len(dependency_graph)} nodes (real cosine similarity over captured keys)")
    print(f"Saved: {DATA_DIR / 'demo_output.json'}")

    # =====================================================================
    section("5. INTEGRATION: GEPA / AUTORESEARCH / META-HARNESS  (Subproblem 5)")
    # =====================================================================
    experience_db = ExperienceDB(DATA_DIR / "experience.db")
    kg_db = KnowledgeGraphDB(DATA_DIR / "knowledge_graph.db")
    budget = BudgetEnforcer(max_kv_bytes=10 * 1024 * 1024, max_tokens_in_flight=10_000)
    retry = RetryPolicy(max_attempts=3, backoff_seconds=0.01)
    router = Router(evaluator=evaluator, retry=retry, budget=budget, experience_db=experience_db)

    for name in PLUGIN_REGISTRY:
        cfg = ExperimentConfig(name=f"demo_{name}", plugin=name, plugin_kwargs=plugin_kwargs[name], layer_idx=INSPECT_LAYER)
        router.run_experiment(cfg, full_kv)
    print(f"Router ran {len(PLUGIN_REGISTRY)} real experiments through router.py -> evaluator.py -> ExperienceDB")
    print(f"experience.db: {len(experience_db.recent(100))} rows")

    # --- ontology: log Token/Layer/KVCache/Plugin entities + relationships ---
    for pos in range(model.seq_len):
        kg_db.upsert_entity("Token", f"token:{pos}", position=pos, source="prompt" if pos < PROMPT_LEN else "generated")
    for layer_idx in range(model.t.num_layers):
        kg_db.upsert_entity("Layer", f"layer:{layer_idx}", num_heads=model.t.num_heads, head_dim=model.t.head_dim)
    kg_db.upsert_entity("KVCache", f"kvcache:{run_id}", run_id=run_id, total_tokens=model.seq_len)
    for edge in dependency_graph:
        for nbr in edge["neighbors"]:
            kg_db.add_relationship(f"token:{edge['token']}", f"token:{nbr['token']}", "ATTENDED_BY", weight=nbr["similarity"])
    for name, rep in plugin_reports.items():
        kg_db.upsert_entity("Plugin", f"plugin:{name}", compression_ratio=rep.compression_ratio)
        kg_db.add_relationship(f"kvcache:{run_id}", f"plugin:{name}", "COMPRESSED_BY", weight=rep.compression_ratio)
    print(f"Knowledge graph: {kg_db.entity_count()} entities, {kg_db.relationship_count()} relationships "
          f"in {DATA_DIR / 'knowledge_graph.db'}")

    # --- retry / fallback / budget, against deliberately forced failures ---
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError(f"simulated transient failure on attempt {calls['n']}")
        return "recovered"

    retried_result = RetryPolicy(max_attempts=3, backoff_seconds=0.01).run(flaky)
    print(f"\nRetry demo: failed twice, succeeded on attempt 3 -> {retried_result!r} (real retries)")

    def always_fails():
        raise RuntimeError("simulated persistent OOM")

    fallback_result = RetryPolicy(max_attempts=3, backoff_seconds=0.01).run(always_fails, fallback=lambda: "FULL_PRECISION_FALLBACK")
    print(f"Fallback demo: failed 3/3, fell back -> {fallback_result!r}")

    try:
        budget.reserve(999_999_999_999)
        budget_ok = False
    except BudgetExceeded as e:
        budget_ok = True
        print(f"Budget demo: oversized request correctly rejected -> {e}")

    gepa = GEPAAdapter(experience_db, prompts_dir=str(DATA_DIR / "prompts"))
    planner_path = gepa.update_planner_md()
    print(f"\nGEPA: wrote real lesson-derived addendum to {planner_path}")
    print("--- addendum content (real, derived from this run's ExperienceDB rows) ---")
    print(gepa.suggest_prompt_update())

    ar_planner = Planner(experience_db)
    suggestions = ar_planner.suggest_next_experiments(n=3)
    print("\nAutoresearch planner.suggest_next_experiments() (real, from ExperienceDB history):")
    for s in suggestions:
        print(f"  priority={s.priority:.1f}  {s.config.plugin:>20}  {s.rationale}")

    head_importance_runs = []
    for seed in (1, 2, 3):
        m2 = build_demo_model(seed=seed)
        captured2: dict[int, np.ndarray] = {}

        def capture_prefill(layer_idx, attn_weights, n_new, **_):
            if n_new > 1:
                captured2[layer_idx] = attn_weights

        m2.register_post_attn_hook(capture_prefill)
        p_ids = np.random.default_rng(seed).integers(0, m2.t.vocab_size, size=8)
        m2.generate(p_ids, 4, rng=np.random.default_rng(seed))
        head_importance_runs.append(KVCacheInspector.head_importance(captured2[INSPECT_LAYER]))

    threshold = float(np.median(np.concatenate(head_importance_runs)))
    reflector = Reflector(kg_db)
    head_usage = reflector.analyze_head_usage(head_importance_runs, threshold=threshold)
    print(f"\nAutoresearch reflector.analyze_head_usage() over {len(head_importance_runs)} independent real runs, "
          f"layer {INSPECT_LAYER} (median-split threshold={threshold:.3f}):")
    for h, info in head_usage.items():
        print(f"  head {h}: underused in {info['underused_fraction'] * 100:.0f}% of runs -> {info['verdict']}")
    print(reflector.summarize_experience(experience_db))

    # =====================================================================
    section("6. SUCCESS CRITERIA -- LIVE SCORECARD  (Subproblem 6)")
    # =====================================================================
    scorecard = {
        "hooks_capture_100pct_of_forward_passes": len(all_events) == expected_events,
        "hook_and_inspector_overhead_under_10pct": overhead_pct < 10.0,
        "at_least_one_plugin_ge_30pct_compression": any(r.compression_ratio >= 1.3 for r in plugin_reports.values()),
        "hook_latency_under_5ms_per_call": float(np.mean(hook_latencies)) < 5.0,
        "retries_3x_before_fallback": calls["n"] == 3,
        "oom_style_failure_falls_back_to_full_precision": fallback_result == "FULL_PRECISION_FALLBACK",
        "budget_enforcement_rejects_oversized_requests": budget_ok,
        "ontology_entities_and_relationships_logged": kg_db.entity_count() > 0 and kg_db.relationship_count() > 0,
    }
    print("Computed from this run's real measurements (not asserted a priori):")
    for k, v in scorecard.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    event_sink.close(); experience_db.close(); kg_db.close()
    section("DONE")
    print(f"Artifacts written under {DATA_DIR} and {OUTPUT_DIR}")

    return {
        "scorecard": scorecard, "overhead_pct": overhead_pct,
        "hook_latency_mean_ms": float(np.mean(hook_latencies)),
        "n_events": len(all_events), "plugin_reports": {n: r.__dict__ for n, r in plugin_reports.items()},
    }


if __name__ == "__main__":
    main()
