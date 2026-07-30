# Success Criteria (Subproblem 6)

Every ✅/❌ below is the actual output of `python -m examples.run_demo`
against the synthetic model (run log reproduced in the main chat response
and in `data/` after you run it yourself) — not an a-priori assertion. Where
a criterion needs infrastructure this sandbox doesn't have (real Gemma, GPU,
Wikitext-2/ARC/HellaSwag), it's marked ⏳ **not measurable here** with what
would be needed, instead of a fabricated pass.

## Functional

| # | Criterion | Threshold | Result |
|---|---|---|---|
| F1 | Hooks capture forward passes without crashing | 100% | ✅ **160/160** events captured (`4 layers × 20 tokens × 2 sources`), exact match, asserted in code (`run_demo.py` section 1) |
| F2 | Inspector computes stats for all layers/heads | <10% overhead | ❌ **Measured +65-90% overhead** on the toy model, consistent across repeated runs (see Performance, P1) — a real fail, not rounded away. See "why" below. |
| F3 | Compression plugins reduce KV size ≥30% with <5% PPL drop | ≥1.3x size **and** <5% PPL | ⚠️ **Partially measurable.** Size reduction: 5/6 plugins hit ≥1.3x (fp8 4.0x, int4 5.33x, low_rank 2.16x, sliding_window 2.0x, token_eviction 2.5x; head_pruning 1.0x — see note). PPL delta: ⏳ not measurable without a real trained model (untrained weights produce no meaningful PPL signal either way). |

**Why F2 fails here:** the hook call's fixed bookkeeping cost (dataclass
construction, a couple of numpy reductions, a sink write) is tens of
microseconds in absolute terms. Against this toy model's sub-millisecond
attention op, that's a large relative overhead. Against a real Gemma
attention op (milliseconds, on GPU), the same absolute overhead would be a
much smaller fraction — but that's an expectation this sandbox cannot
verify, not a measured fact. Treat F2 as **failed on the only model
available to test it on**, with a documented reason to expect (not assume)
a better number on the real target.

**Note on head_pruning in F3:** it legitimately hit 1.0x (no reduction) on
this run because `entropy_threshold=1.0` didn't clear any heads — the
untrained model's attention is close to uniform (high entropy) for every
head, so nothing looked prunable. That's the plugin working correctly
given the input, not a bug; a trained model's real head-specialization
would very likely give this plugin something to actually prune. See
`data/demo_output.json` → `plugin_results.head_pruning`.

**Note on sliding_window / token_eviction's `reconstruction_max_abs_err`:**
these two report `N/A`, not a number — and that's a fix, not a gap. An
earlier version zero-padded their output back to the original shape so it
could be diffed like the other plugins', which meant the "error" it
reported was actually just the magnitude of the *discarded* tokens
compared against invented zeros. Invisible on this repo's tiny synthetic
values, but a real run against Gemma 3 (whose KV values run into the tens)
reported errors like 47.0 that looked broken but weren't computing
anything meaningful either way. Fixed: `decompress()` for these two now
returns exactly what survived (kept tokens are exact copies — 0 error by
construction, not worth reporting as a number), and `Evaluator` explains
why in `BenchmarkReport.notes` instead of printing a misleading value.

## Performance

| # | Criterion | Threshold | Result |
|---|---|---|---|
| P1 | Hook latency per token | <5ms | ✅ mean **~2-3ms** (varies 1.8-2.6ms run to run on shared sandbox CPU), p99 **~7-10ms** across 160 calls (p99 has occasional scheduling-noise spikes above 5ms; mean comfortably under) |
| P2 | Visualizations render in <1s for 1K tokens | <1s @ 1K tokens | ⏳ Not directly tested — this run used 20 tokens (heatmap/PCA/graph rendered in well under 1s each, timed informally via wall clock). matplotlib/plotly rendering is O(tokens²) for the heatmap and nearest-neighbor graph specifically (pairwise cosine similarity), so 1K tokens should be profiled before relying on this — flagged, not assumed. |

## Resilience

| # | Criterion | Threshold | Result |
|---|---|---|---|
| R1 | Falls back to full-precision KV on OOM-style errors | required | ✅ `RetryPolicy` + `Router._fallback_report`: forced 3/3 failures → real fallback returned (`FULL_PRECISION_FALLBACK` / a `passed=False` full-precision report), demonstrated live in `run_demo.py` §5 |
| R2 | Retries failed hooks 3x before aborting | 3 attempts | ✅ forced 2 transient failures, succeeded on attempt 3, confirmed via a real attempt counter (not a mocked return value) |
| R3 | Budget enforcement | reject over-limit requests | ✅ a deliberately oversized reserve request (`999,999,999,999` bytes against a 10MB budget) was correctly rejected with `BudgetExceeded` |

## Ontology

| # | Criterion | Threshold | Result |
|---|---|---|---|
| O1 | Token/Layer/Head/KVCache/Plugin entities logged | all present | ✅ **31 entities** written to `knowledge_graph.db` this run: 20 Token, 4 Layer, 1 KVCache, 6 Plugin (head-level entities are written when `Reflector.analyze_head_usage` runs — see O2) |
| O2 | Relationships logged | ATTENDED_BY / COMPRESSED_BY etc. | ✅ **26 relationships**: 20 `ATTENDED_BY` (token↔nearest-neighbor edges) + 6 `COMPRESSED_BY` (cache↔plugin) |

## Research

| # | Criterion | Threshold | Result |
|---|---|---|---|
| Q1 | "Which heads are unused?" (the spec's own example) | answerable | ✅ `Reflector.analyze_head_usage()` answers this for real, from 3 independent real generation runs — see `run_demo.py` §5 output |
| Q2–Q8 | The other 7 of "8 predefined questions" | ≥90% accuracy | ⚠️ **Genuinely unresolved — see below, not glossed over.** |

**On Q2–Q8:** the source spec says "Answers all 8 predefined questions
(e.g., 'Which heads are unused?') with ≥90% accuracy" but never lists the
other 7, and doesn't say what "accuracy" means for an open-ended research
question (accuracy against what ground truth?). Rather than silently
inventing 7 questions and grading myself against them, here's a candidate
set this framework *can* answer given what's already built — treat this as
a proposal for you to confirm or edit, not a completed criterion:

1. Which heads are unused? — `Reflector.analyze_head_usage` ✅ implemented
2. Which tokens have near-duplicate KV entries? — `KVCacheInspector.duplicate_clusters` ✅ implemented
3. Which layer contributes the most to cache growth? — sum `kv_bytes` by `layer_idx` in `events.db` ✅ implemented (one query)
4. Which compression plugin gives the best ratio-for-a-given-error-budget? — `Evaluator.benchmark_plugin` sweep + sort ✅ implemented
5. Does attention entropy trend up or down over a generation? — `timeline[i]["mean_attn_entropy"]` in `demo_output.json` ✅ implemented
6. Which tokens are the least "similar to anything else" (candidates to keep, not evict)? — invert `nearest_neighbors` ✅ implemented
7. How does hook overhead scale with sequence length? — needs a sweep over sequence lengths, not just one run — **not implemented**, would take ~10 lines added to `run_demo.py`'s §1 benchmark loop
8. Does compression ratio vs. reconstruction error trade off consistently across layers, not just the one layer this demo inspected? — needs the demo to loop over layers, not just `INSPECT_LAYER=2` — **not implemented**

"≥90% accuracy" isn't scored anywhere above because 6 of these 8 are
deterministic computations over captured data (no "accuracy" concept
applies — they're either computed correctly or they're a bug, and the test
suite checks correctness, not accuracy-against-ground-truth). If the
original intent was closer to "an LLM answering these questions in natural
language, graded against a human-labeled rubric," that's a different,
larger piece of work (an eval harness for the *research* layer itself) that
isn't built here — flagged honestly rather than assumed.
