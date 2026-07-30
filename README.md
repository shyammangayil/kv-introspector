# KV Cache Introspector

A model-agnostic KV cache observability, compression-experimentation, and
research framework for LLMs (Gemma and others), with integration points for
GEPA, Autoresearch, and Meta-Harness.

This README covers **local setup first** (skip to "Quick start" if that's
all you need), then the architecture, and an honest account of what was and
wasn't validated in the sandbox this repo was built in.

---

## Quick start (local setup)

Requires **Python 3.9+** (developed and tested on 3.12).

**1. Unzip and enter the project:**
```bash
unzip kv_introspector.zip
cd kv_introspector
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```
This installs everything needed for the demo, tests, and debugger (numpy,
pandas, matplotlib, scikit-learn, networkx, PyYAML, plotly, streamlit,
pytest). It does **not** install torch/transformers — see "Running this
against real Gemma" below for that.

**4. Run the end-to-end demo:**
```bash
python -m examples.run_demo
```
This runs a small synthetic transformer through the entire pipeline — hooks,
inspector, all 6 compression plugins, GEPA/Autoresearch/Meta-Harness
integration, retry/budget resilience — and prints real, measured results
(not fabricated) to your terminal. It writes:
- `data/events.db`, `data/experience.db`, `data/knowledge_graph.db` (SQLite)
- `data/demo_output.json` (timeline + dependency-graph snapshot)
- `data/prompts/planner.md` (GEPA's real generated prompt update)
- `outputs/*.png` (attention heatmap, PCA projection, nearest-neighbor graph)

**5. Run the test suite:**
```bash
pytest tests/ -v
```
54 tests, all torch-independent, run in well under a second.

**6. Launch the Timeline & Memory Debugger:**
```bash
streamlit run debugger/app_streamlit.py
```
Open the URL Streamlit prints (usually `http://localhost:8501`). Run step 4
first so there's data to load.

That's the whole loop. Everything below is reference material.

---

## Repository layout

```
kv_introspector/
├── core/
│   ├── hooks.py            # Subproblem 1: runtime hook layer (KVHookManager, CacheTap)
│   ├── cache_adapters.py    # Normalizes transformers Cache API differences (legacy/DynamicCache/...)
│   ├── storage.py           # SQLite schemas: events, experience, knowledge graph
│   ├── inspector.py          # Subproblem 2: L2 norm / cosine sim / variance / age / duplicates
│   └── visualize.py           # Subproblem 2: heatmap / PCA / UMAP / nearest-neighbor plots
├── compression/
│   ├── base.py               # Plugin contract (CompressionPlugin, CompressedResult)
│   └── plugins.py             # Subproblem 3: fp8, int4, low-rank, sliding window, eviction, head pruning
├── harness/
│   ├── evaluator.py           # Subproblem 3/5: benchmark pipeline + guarded PPL/accuracy
│   ├── router.py                # Subproblem 5: dispatches experiments to plugins
│   ├── retry.py                  # Subproblem 5/6: retry + fallback
│   └── budget.py                  # Subproblem 5/6: memory/token limits
├── integrations/
│   ├── gepa_adapter.py           # Subproblem 5: GEPA (real published framework) integration
│   └── autoresearch_adapter.py    # Subproblem 5: planner + reflector reference implementations
├── debugger/
│   └── app_streamlit.py           # Subproblem 4: Timeline Debugger + Memory Debugger
├── examples/
│   ├── synthetic_model.py         # NumPy transformer used for real in-sandbox validation
│   ├── run_demo.py                 # End-to-end demo -- source of every number in this README
│   └── run_gemma.py                 # Same pipeline against a real downloaded Gemma checkpoint
├── config/
│   ├── models.yaml                 # Model/hook/storage config (synthetic + real Gemma targets)
│   └── budgets.yaml                 # Resource limits + pass/fail thresholds
├── tests/
│   └── test_core.py                 # 54 unit tests, no torch required
├── success_criteria.md              # Subproblem 6: criteria + this run's real scorecard
├── CHANGELOG.md                      # Real-world validation milestones and the bugs they found
└── requirements.txt
```

---

## What was actually run vs. what needs your infra

Being upfront about this because the alternative — silently presenting
Gemma-scale benchmark numbers that were never measured — would be worse
than not having them.

**Actually run, in the sandbox this repo was built in:**
- Every module imports and runs correctly (`pytest tests/`: **54/54 pass**).
- `examples/run_demo.py` runs the entire pipeline against a real, small
  numpy-only transformer (4 layers, 4 heads, head_dim 16, untrained/random
  weights) with a **real causal-attention forward pass** and a **real,
  growing KV cache** — not mocked or pre-baked data.
- `core.hooks.CacheTap` — the actual production class, not a stand-in —
  wraps that model's cache object and captured **160/160** expected events
  with zero drops.
- All 6 compression plugins really compressed and decompressed a real
  captured KV block, with real measured compression ratios (1.0x–5.33x
  depending on plugin/settings) and real reconstruction error.
- `harness/retry.py` and `harness/budget.py` were exercised against
  **deliberately forced failures** (not just happy-path calls): 2 simulated
  transient failures then success, 3/3 failures then fallback, and a
  budget-overflow rejection — all real, all caught on camera in the demo's
  stdout.
- `integrations/gepa_adapter.py` and `integrations/autoresearch_adapter.py`
  read real `ExperienceDB`/`KnowledgeGraphDB` rows this run produced and
  generated real (not scripted) suggestions from them.
- `debugger/app_streamlit.py` was boot-tested with Streamlit's own
  `AppTest` harness against the real data above: zero exceptions, both
  tabs render, both selectboxes populate.
- **`core.hooks.KVHookManager` (forward hooks) had two real bugs**, found
  once this was run against a real model (this repo's own synthetic-model
  testing couldn't have caught either — different problem classes,
  detailed below):
  1. Decode-step events all wrote `token_pos=-1` instead of the true
     absolute position (found writing `examples/run_gemma.py`, before it
     ever ran against real Gemma).
  2. Over-broad module matching: hooked every child of an attention block
     (q_proj, k_proj, ...) alongside the block itself (found *from* a real
     Gemma run — the hook count didn't divide evenly by the layer count).
  Both fixed at the time — but a second real run exposed something bigger
  than either: **`google/gemma-3-4b-it` is multimodal** (a SigLIP vision
  tower bundled with the language model), so even *correct* attention-
  module discovery finds 27 vision-encoder attention blocks alongside the
  34 real language-model layers — not a bug, just not what a KV-cache
  tool for text generation cares about, and not fixable by better regex.
  **Architectural response, not another patch:** `core.hooks.CacheTap` —
  which hooks the *cache object itself* via `Cache.update()`, never
  searching the module tree — is now the sole source of `KVEvent`s.
  `KVHookManager` is demoted to optional, storage-decoupled
  instrumentation (no sink, no KVEvents; cross-checking / entropy only).
  Full account, including a related data-integrity bug this also
  surfaced (`events_gemma.db` silently accumulating rows across every
  past run, since nothing cleared it and reads weren't scoped by
  `run_id`) in `CHANGELOG.md`.

**Independently run against a real model, outside this sandbox, twice:**
2026-07-28, `examples/run_gemma.py` ran end-to-end against a real,
locally-downloaded `google/gemma-3-4b-it` (CPU) — model load, generation,
hook capture, KV extraction, the full inspector + compression pipeline,
against real weights and real captured tensors. Better evidence than
anything this sandbox could produce itself (no GPU/HF access here — see
below), precisely because it's independent: nobody involved in writing the
code ran it. 2026-07-29, a second independent run (with the module-
matching fix applied) surfaced the multimodal/accumulation findings above,
prompting the CacheTap redesign. **Not yet independently re-confirmed
after the redesign** — `core/hooks.py`'s new `CacheTap` (isinstance-safe
retroactive subclassing) and `examples/run_gemma.py`'s new
`DynamicCache(config=...)` + `generate(past_key_values=...)` wiring are
unit-tested in this sandbox with duck-typed fakes (54/54 pass), but nobody
has run this exact version against a real checkpoint yet. If you're the
one running it: that's the next useful data point, and the biggest
remaining unknown either way.

**NOT run in this sandbox specifically**, because the infra isn't
available here (no GPU, no network access to huggingface.co for
weights/datasets, and only ~10GB free disk — too little to safely install
PyTorch's PyPI wheel, which bundles multi-GB CUDA dependencies even for
CPU-only use). This sandbox has not independently re-confirmed the
post-fix state against real Gemma (see above), and specifically has not run:
- `harness/evaluator.py`'s `evaluate_perplexity` / `evaluate_accuracy`
  against Wikitext-2 / ARC / HellaSwag — these raise
  `InfrastructureUnavailable` with a specific, actionable message rather
  than return a fabricated number. `run_demo.py` deliberately calls
  `evaluate_perplexity(None, None, ...)` and prints the resulting error as
  a live confirmation that the guard actually fires.
- `core.hooks.KVHookManager`'s `register_forward_hook` path against a real
  `torch.nn.Module` — it's ordinary, correct PyTorch API usage, but
  literally executing it needs torch installed.
- Full JAX/Flax hook support — not implemented (see `success_criteria.md`
  "Uncertainties").

Absolute latency/overhead numbers from the synthetic model are a **toy CPU
model's** numbers, not Gemma's — useful for validating the pipeline's
correctness and the *pattern* of measurement, not for capacity planning.
One concrete example: the measured hook overhead came out to roughly
+65–90% on this toy model, consistently across repeated runs (see
`success_criteria.md`) — that FAILS the
"<10% overhead" target, honestly reported rather than rounded away. It's
almost certainly a toy-model artifact (fixed per-call bookkeeping cost is
tens of microseconds in absolute terms, which is small next to a real
Gemma attention op on GPU but large next to this tiny model's sub-
millisecond op) — but that's an expectation, not something this sandbox
could verify. Re-run the benchmark in `run_demo.py`'s Section 1 against a
real model to get a number you can trust for capacity planning.

---

## Running this against real Gemma

**1. Get Hugging Face access to Gemma** (it's gated — free, but not automatic):
   1. Create an HF account if needed: https://huggingface.co/join
   2. Open the model page for the size you want (e.g.
      https://huggingface.co/google/gemma-3-4b-it) while logged in, and
      accept Google's usage license — approval is instant.
   3. Create a read-access token: https://huggingface.co/settings/tokens
   4. Authenticate locally: `pip install huggingface_hub[cli]` then
      `huggingface-cli login` and paste the token (or `export
      HF_TOKEN=hf_xxx`, or pass `token="hf_xxx"` to `from_pretrained`).

**2. Pick a size that fits your hardware:**

| Model | Params | bf16 weight size | Practical minimum |
|---|---|---|---|
| `google/gemma-3-270m-it` | 270M | ~0.5GB | any laptop CPU, or any GPU — **start here** |
| `google/gemma-3-1b-it` | 1B | ~2GB | modest GPU (4-6GB) or slow CPU |
| `google/gemma-3-4b-it` | 4B | ~8GB | 12-16GB GPU (T4, RTX 3060+), or 4-bit quant on less |
| `google/gemma-3-12b-it` | 12B | ~24GB | 24GB+ GPU (A10, RTX 3090/4090), or quantized |
| `google/gemma-3-27b-it` | 27B | ~54GB | multi-GPU / A100-80GB, or heavy quantization |

No local GPU that fits your target size → a free Colab/Kaggle T4 covers up
to ~4B comfortably; RunPod/Lambda/Vast.ai rent bigger ones by the hour. To
fit a bigger model on a smaller GPU, load with
`quantization_config=BitsAndBytesConfig(load_in_4bit=True)` (needs
`pip install bitsandbytes`) — this is the real-world version of what this
repo's own `compression/plugins.py::INT4Quantization` simulates in numpy.

**3. Install the extra dependencies** (same venv as before):
```bash
pip install torch transformers accelerate
pip install bitsandbytes   # optional, only needed for 4-bit/8-bit loading
```

**4. Run it — architecture updated 2026-07-29 (see `CHANGELOG.md`); not yet
independently re-confirmed against real Gemma since that update:**
```bash
python -m examples.run_gemma                             # 270M model, fast smoke test
python -m examples.run_gemma --model google/gemma-3-4b-it --prompt "The capital of Mongolia is" --max-new-tokens 8
python -m examples.run_gemma --cross-check                # + independent verification, slower
```
`examples/run_gemma.py` is the real-Gemma counterpart to
`examples/run_demo.py` — same downstream pipeline (inspector → compression
→ storage), against a real downloaded checkpoint. `core.hooks.CacheTap`
wraps a real `DynamicCache` and is passed straight into
`generate(past_key_values=...)` — every real KV append writes a `KVEvent`;
nothing else does. By default this uses the fast `sdpa` attention backend
(CacheTap needs neither eager attention nor `output_attentions`, unlike
the module-hooking approach this replaced). Prints real per-token stats, a
real per-layer cache-type table (global vs. sliding-window, straight from
the real `DynamicCache`), and real compression ratios on Gemma's actual KV
cache; writes `data/events_gemma.db` + `outputs/gemma_*.png` — fresh every
run, not accumulated (see technical notes). Add `--device cuda:0` (or
`cpu`, to fail loudly instead of silently disk-offloading) to override
`device_map="auto"`.

**5. Point the debugger at the new data:**
```bash
streamlit run debugger/app_streamlit.py
```
In the sidebar, change the events.db path to `data/events_gemma.db`.

**6. GEPA/Autoresearch/Meta-Harness need no changes** — `integrations/*.py`
and `harness/*.py` only ever consume `ExperienceDB`/`KnowledgeGraphDB` rows
and numpy KV tensors; they don't know or care whether those came from the
synthetic model or real Gemma. Wire `harness.router.Router.run_experiment`
into `run_gemma.py`'s KV block the same way `run_demo.py`'s Section 5 does,
and everything downstream — GEPA's lesson generation, Autoresearch's
planner/reflector — works unmodified.

**Technical notes, current as of the 2026-07-29 architecture (full story
in `CHANGELOG.md`):**
- **`google/gemma-3-4b-it` is multimodal** — it bundles a SigLIP vision
  tower alongside the language model. `CacheTap` sidesteps this by
  construction (it hooks the cache object, never searches the module
  tree, so the vision tower's attention never enters the picture for a
  text-only prompt). If you use `--cross-check` (which does hook attention
  modules, for entropy/latency cross-verification), it's explicitly scoped
  away from `vision_tower` already — but this is the reason that scoping
  exists, not a hypothetical.
- **`data/events_gemma.db` is recreated fresh every run**, not appended
  to. Earlier versions of this script didn't do this, and
  `SQLiteEventSink.read_all()` didn't filter by `run_id` either (it does
  now, as an optional argument) — meaning event counts from earlier
  debugging sessions were quietly counting rows from every past run mixed
  together. If you want cross-run history instead of a clean slate, call
  `sink.read_all()` with no `run_id` filter, or `sink.distinct_run_ids()`
  to see what's accumulated.
- `DynamicCache(config=text_cfg)` is how the cache is constructed —
  matching HuggingFace's own documented usage
  (huggingface.co/docs/transformers/en/kv_cache, "Cache strategies"), not
  a guess. Falls back to bare `DynamicCache()` on `TypeError` for older
  transformers versions. **This specific construction has not been
  confirmed against a real checkpoint yet** — it's the most important
  thing to watch on your first run after this update.
- `-it` checkpoints load a *composite* `Gemma3Config` — architecture
  fields like `num_hidden_layers` live under
  `model.config.get_text_config()`, not `model.config` directly.
  `run_gemma.py` handles this; `get_text_config()` is the version-robust
  way to reach these fields elsewhere too (returns the nested config on a
  composite checkpoint, `model.config` unchanged on a plain text model).
- Gemma 3 alternates ~5 local sliding-window attention layers for every 1
  global attention layer (confirmed on `-4b-it`: layers 5, 11, 17, 23, 29
  of 34 are global, window=1024 elsewhere). Sliding-window layers' cache
  stops growing once context exceeds the window; global layers keep
  growing — expect the Timeline Debugger to show different per-layer
  growth curves than the synthetic model's uniform growth.
- Gemma uses grouped-query attention (fewer KV heads than query heads,
  shared across groups) — `compression/plugins.py::HeadPruning` operates
  on KV heads, the right level for real memory savings, but "head" means
  something slightly different here than in a vanilla multi-head-attention
  diagram.
- `--cross-check`'s eager attention + `output_attentions=True` is slow:
  ~35s/generated token was observed for `-4b-it` on CPU (expected — eager
  attention materializes the full attention matrix, and
  `output_attentions` disables some fused-kernel fast paths). Without
  `--cross-check`, CacheTap alone needs neither, so the default `sdpa`
  backend is used and this shouldn't apply.
- First download is several GB — budget time/disk accordingly, and note
  `HF_HOME` (defaults to `~/.cache/huggingface`) if you want it elsewhere.

For perplexity/accuracy against Wikitext-2/ARC/HellaSwag, pass your loaded
`model`/`tokenizer` into `Evaluator.evaluate_perplexity` /
`.evaluate_accuracy` instead of `None` (needs `pip install lm-eval` too for
loading the actual datasets).

---

## Key design choices worth knowing about

- **`CacheTap` is the sole source of `KVEvent`s** (`core/hooks.py`) —
  hooks the cache *object*'s `update()` directly, never searches the
  module tree, so it's unaffected by a model's internal naming and (a real
  finding, not a hypothetical) never sees a multimodal checkpoint's vision
  tower. Built via *retroactive subclassing* — `CacheTap(cache, sink)`
  reassigns `cache.__class__` to a dynamic subclass and returns the same
  object — specifically so `isinstance(result, DynamicCache)` stays `True`
  when passed into `generate(past_key_values=...)`, which a simpler
  `__getattr__`-delegating wrapper would not survive. `KVHookManager`
  (forward hooks on attention modules) is demoted to optional,
  storage-decoupled instrumentation — cross-checking / attention entropy
  only, no longer a KVEvent source. Full account in `CHANGELOG.md`.
- **A dedicated normalization layer** (`core/cache_adapters.py`) isolates
  every place this repo needs to know which shape `transformers`' Cache
  API happens to be this release (legacy tuple / `DynamicCache.layers` /
  older `.key_cache`) into one module, so `core/inspector.py` and
  `compression/plugins.py` only ever see a plain numpy array regardless.
  Added after a real run against Gemma 3 hit exactly this problem — see
  `CHANGELOG.md`.
- **Compression plugins are pure numpy**, not tied to bitsandbytes/CUDA —
  the *algorithms* (quantization grid, SVD truncation, eviction policies)
  are the real thing; hardware-specific bit-packing is simulated, and each
  plugin's docstring says exactly what the real GPU-kernel swap looks like.
- **The GEPA integration is grounded in the real published framework**
  (Agrawal et al., 2025; github.com/gepa-ai/gepa, also available as
  `dspy.GEPA`) — see `integrations/gepa_adapter.py`'s docstring for the
  trajectory+feedback shape and exactly where a real `gepa.optimize()` call
  would plug in.
- **"Autoresearch" and "Meta-Harness" adapters are reference
  implementations** against the interface the original spec described
  (`agents/planner.py`, `agents/reflector.py`, `harness/router.py`, etc.).
  This repo has no access to your actual Autoresearch/Meta-Harness
  codebases, so these are real, tested, standalone implementations of that
  interface for your actual agents to call — not verified integrations
  with your specific repos.

---

## License / attribution

No external code is copied in. GEPA is referenced, not vendored — see
https://github.com/gepa-ai/gepa for its own license if you integrate it
directly.
