# Changelog

## 2026-07-29 (afternoon) -- Live KV modification: from inspector to research platform

`examples/run_gemma.py` ran successfully end-to-end against real
`google/gemma-3-4b-it` for the first time since the CacheTap redesign
(`op.txt`) -- `DynamicCache(config=text_cfg)` construction worked exactly
as documented, `isinstance(cache, DynamicCache)` held, and
`verify_capture_integrity` reported ALL MATCH across all 34 real layers.
Milestones 1-4 from the previous entry are now confirmed against a real
checkpoint, not just unit-tested.

That run also surfaced two real, worth-fixing issues, and prompted the
biggest feature addition yet.

### Bugs found in op.txt's real output

- **`head_pruning: 1.00x, 0 error`, every time.** Traced to a real bug,
  not "operating on a copy" (though that observation was also correct,
  see below): `run_gemma.py` called `head_pruning` with no kwargs at all,
  so it fell back to KV-norm-based importance (real Gemma norms run in
  the tens) compared against the default `entropy_threshold=1.0` --
  every head cleared that bar trivially, so nothing was ever pruned.
  Fixed: the threshold is now the real median of that layer's actual
  per-head KV norms, computed from the same captured tensor, so roughly
  half the heads get pruned as a meaningful demonstration rather than a
  miscalibrated constant.
- **`generated_len` (5) vs. the cache's own reported length (10, for a
  6-token prompt) don't sum to 11.** Not a capture bug --
  `verify_capture_integrity` already proved CacheTap's count matches
  `get_seq_length()` exactly. `transformers` doesn't need to cache the
  very last generated token's K/V once generation is stopping (nothing
  will ever attend back to it), so the cache legitimately ends up one
  entry short of `prompt_len + generated_len`. Documented in
  `run_gemma.py` so it doesn't look like an unresolved discrepancy.

### The bigger idea: KVModifier

Everything up to now only *observes* the cache. `core/kv_modifiers.py`
adds a `KVModifier` interface, wired directly into `CacheTap.update()`:
after the real cache append, a modifier gets the full current key/value
for that layer and can return a changed version, which is written back
into the cache's own storage (`core.cache_adapters.set_layer_kv`) --
so the change is what the model's own next attention computation reads,
and *persists* for every future step, not just the one where it ran.
This is what turns "zero this token" from an offline reconstruction-error
metric into an actual causal experiment about the model.

Six modifiers implemented: `ZeroModifier`, `NoiseModifier`,
`SwapTokensModifier`, `HeadMaskModifier` (drop via scale=0, amplify via
scale>1), `MemoryDecayModifier` (stateful -- decays only what pre-existed
each call, so it compounds to `decay_per_step**n` correctly without
storing pristine originals), and `LowRankModifier` (reuses
`compression.plugins.LowRankCompression` live, every step -- "compression
in the loop" rather than a static benchmark). `MutationDB`
(`core/storage.py`) records every experiment: mutation type, config,
prompt, baseline output, mutated output, latency, and whether
`verify_capture_integrity` still held after the intervention.

**Proven causally, not just wired structurally** -- ran for real, in this
sandbox (pure numpy, no torch needed): `examples/run_mutation_demo.py`, a
small standalone 4D-cache model (deliberately separate from
`examples/synthetic_model.py`, so nothing about `run_demo.py`'s existing
validated pipeline was touched or risked), showed all 8 tested
interventions changing greedy-decoded output vs. an identical unmodified
baseline, with `verify_capture_integrity` passing throughout. The
memory-decay compounding math was checked against a real measured ratio,
not just asserted: 0.9**6 predicted, 0.5314 both predicted and measured
from actual before/after L2 norms.

A real bug surfaced building this, fixed before it shipped:
`core.cache_adapters.get_layer_kv`/`set_layer_kv` only handled the two
real-transformers shapes -- this repo's own `.k`/`.v` list convention
(`examples/synthetic_model.py`'s `NumpyKVCache`, and several test fakes)
wasn't recognized, so modifier write-back silently had nowhere to go
against anything but a real `DynamicCache`. Both functions now handle
three conventions uniformly.

`examples/run_gemma.py` gained `--modifier 'name:key=val;key=val'` (e.g.
`--modifier 'zero:token_positions=0,3'`) and `--compare-baseline` (runs
generation twice -- once clean, once modified -- and diffs the output),
logging every real-Gemma mutation experiment to
`data/mutations_gemma.db`. **Not yet run against a real checkpoint** --
the parser and wiring are tested without torch, but nobody has pointed
`--modifier` at real Gemma yet.

`requirements.txt`: `torch`/`transformers`/`accelerate` uncommented,
unpinned (by request) -- confirmed working unpinned against real Gemma 3
4B as of this session.

## 2026-07-29 -- CacheTap becomes the sole source of KVEvents

Following up on 07-28's run: re-running the fixed module-matching logic
against real `google/gemma-3-4b-it` still showed a mismatch (61 hooked
modules, not 34; hook event counts that didn't reconcile against any
"layers x tokens" formula). Investigation traced this to something bigger
than a matching bug, and prompted an architectural change rather than
another patch.

### What was actually going on

1. **`google/gemma-3-4b-it` is multimodal.** Debug output (printing each
   hooked module's class + name) showed exactly 27 `SiglipAttention`
   modules (`model.vision_tower.encoder.layers.*.self_attn`) plus 34
   `Gemma3Attention` modules (`model.language_model.layers.*.self_attn`)
   -- 61 total, matching the reported count exactly. Not a bug: SigLIP's
   attention modules are correctly identified as attention modules by any
   reasonable definition. They just don't route through the language
   model's KV cache at all, and there's no general name-pattern fix for
   "only match the text decoder" that would hold up against a different
   multimodal model's naming.
2. **`events_gemma.db` was never cleared between runs, and
   `SQLiteEventSink.read_all()` never filtered by `run_id`.** Every past
   invocation's rows were silently sitting in the same file, getting
   counted alongside the current run's. Several event-count "mismatches"
   across this debugging session were partly counting stale rows from
   earlier runs without anyone knowing it. This alone would have made
   Milestone 2 (verify a clean 1:1-per-token mapping) impossible to trust,
   independent of anything else.
3. An uploaded work-in-progress version of `core/hooks.py` had the
   `_post_hook` sink-write commented out and an unwired, broken
   `_make_attention_hook` method (references `sink` and `self.debug`,
   neither in scope -- would raise `NameError`/`AttributeError` if it were
   ever called). Not currently active, but real bugs nonetheless.

### The architectural change

Module-discovery-based capture (find modules that "look like" attention,
hook their `forward()`) is fragile in a way no amount of regex tuning
fully closes: it depends on guessing which modules matter from names and
shapes, and a multimodal checkpoint proved that guess can be systematically
wrong in a way that isn't a simple bug to patch. `Cache.update(key_states,
value_states, layer_idx, cache_kwargs)`, by contrast, is the actual, narrow
interface every real KV append goes through -- not discovered by searching
a module tree, so it never sees the vision tower in the first place.

**`CacheTap` is now the single source of truth for `KVEvent`s.**
`KVHookManager` (forward hooks) is demoted to optional, storage-decoupled
instrumentation -- it no longer writes to any sink at all; it collects
latency/entropy observations in memory for cross-checking or for entropy
data CacheTap structurally can't see (it only observes cache appends, not
attention weights).

**`CacheTap` itself was redesigned**, not just repurposed: the original
version was a composition-based wrapper (`__getattr__` delegating
everything except `update`). That does not survive `isinstance(x, Cache)`
checks `transformers.generate()` may perform internally on
`past_key_values` -- wrapping doesn't change an object's real type. The
new version uses *retroactive subclassing*: `CacheTap(cache, sink)`
reassigns `cache.__class__` to a dynamically-created subclass of
`type(cache)` and returns that same object (same identity, not a wrapper).
`isinstance(result, DynamicCache)` is genuinely `True`; every inherited
method behaves exactly as on the untouched original, because nothing was
wrapped. Confirmed with a standalone reproduction of the pattern
(`__new__` returning a non-`cls` instance correctly skips `__init__`, in
case that Python semantic was misremembered) before relying on it.

`examples/run_gemma.py` now constructs `DynamicCache(config=text_cfg)`
(HuggingFace's own documented pattern -- confirmed via their "Cache
strategies" doc, not guessed), wraps it with `CacheTap`, and passes it
directly to `generate(past_key_values=...)`. A new
`verify_capture_integrity()` checks every layer's captured span against
that layer's own `get_seq_length()` -- proof against the cache's own
accounting, not a hand-derived "layers x tokens" formula (exactly the kind
of formula that broke down once it had to account for 61 differently-
behaved modules instead of 34 uniform ones).

`--cross-check` optionally still attaches `KVHookManager`, explicitly
scoped away from the vision tower (`exclude_prefixes`), purely to compare
per-layer call counts against CacheTap's as an independent witness, plus
attention entropy CacheTap can't see. Off by default: it needs eager
attention + `output_attentions=True`, which is why runs were measured at
~35s/token -- without it, CacheTap alone needs neither, so the default
(fast) `sdpa` backend is used.

### Not yet independently re-confirmed

None of the above -- the multimodal finding, the accumulation-bug fix, the
CacheTap redesign, or `DynamicCache(config=...)` construction -- has been
re-run against a real checkpoint since these changes. Unit-tested in this
sandbox with duck-typed fakes (54/54 pass, including new tests
specifically simulating the isinstance-safety property and the prefill/
decode position-tracking this all depends on), but the next real run is
the thing that will actually confirm or falsify `DynamicCache(config=...)`
construction and full integration with `generate()`.

## 2026-07-28 -- First real-model validation (google/gemma-3-4b-it)

The framework was run end-to-end against a real, locally-downloaded
`google/gemma-3-4b-it` (CPU, eager attention) for the first time, outside
this repo's own sandbox (which has no GPU/no HF access -- see README.md).
Result: **it worked**, and surfaced three real bugs that no amount of
synthetic-model testing had caught, plus one already-anticipated API
migration. All four are fixed and covered by new tests.

### What broke, and why the synthetic model couldn't have caught it

1. **`KVHookManager` hooked far more modules than intended.**
   `looks_like_attention_module`'s name check matches on the *full dotted
   path* returned by `named_modules()`, so `model.layers.5.self_attn.q_proj`
   matches too -- it contains "attn" because it's nested under `self_attn`.
   Every child of every attention block (q_proj, k_proj, v_proj, o_proj, ...)
   was getting hooked alongside the block itself. Invisible in this repo's
   own testing because the synthetic model is a flat, hand-rolled tree with
   no nested submodules to over-match. **Fixed:**
   `KVHookManager._find_attention_modules` now drops any match that is a
   dotted-path descendant of another match. New test:
   `test_find_attention_modules_ignores_nested_children_of_a_match`.

2. **`transformers`' Cache API moved out from under the code.**
   `return_legacy_cache=True` (a `generate()` kwarg this repo depended on)
   no longer exists on current `transformers` -- confirmed by hitting
   `ValueError: The following model_kwargs are not used by the model:
   ['return_legacy_cache']` against the real checkpoint. The replacement,
   `DynamicCache`, no longer looks like a tuple or exposes `.key_cache` /
   `.value_cache` directly -- it holds a list of per-layer objects
   (`cache.layers[i].keys` / `.values`), which also expose real metadata
   this repo didn't have before: `.sliding_window`, `.cumulative_length`,
   `.is_initialized`, and a layer *class*
   (`DynamicSlidingWindowLayer` vs `DynamicLayer`) identifying the
   attention pattern. Confirmed on `-4b-it`: layers 5, 11, 17, 23, 29 of 34
   are global; the rest are sliding-window with `window=1024` -- the 5:1
   ratio this repo's docs had cited from a web search, now empirically
   confirmed rather than assumed. **Fixed:** new `core/cache_adapters.py`
   module isolates this -- `get_layer_kv` tries the tuple / `.layers` /
   `.key_cache` / `.to_legacy_cache()` shapes in that order;
   `get_layer_meta` / `describe_cache` surface the new per-layer metadata.
   `examples/run_gemma.py` uses it instead of ad hoc extraction.

3. **Compression plugins' `reconstruction_max_abs_err` was measuring the
   wrong thing for `sliding_window` / `token_eviction`.**
   Their `decompress()` zero-padded dropped tokens back to the original
   shape "so downstream shape checks don't break." That meant the
   "reconstruction error" the evaluator reported was really just the
   magnitude of the *discarded* tokens compared against invented zeros --
   harmless-looking on this repo's tiny synthetic values (~0.01-0.05), but
   the real run against Gemma (whose KV values run into the tens) reported
   errors like 40-47 that looked like broken math. **Fixed:** these two
   plugins' `decompress()` now returns exactly what survived, unpadded;
   `Evaluator` reports `reconstruction_max_abs_err=None` with an explained
   note (`"discards N elements outright rather than approximating them"`)
   instead of a misleading number. New tests:
   `test_evaluator_reports_na_not_misleading_error_for_shape_changing_plugins`,
   `test_sliding_window_and_token_eviction_kept_values_are_exact`.

4. **Composite `Gemma3Config`.** `model.config.num_hidden_layers` doesn't
   exist directly on `-it` checkpoints -- they load a composite config (text
   + vision sub-configs), so text-architecture fields live under
   `model.config.get_text_config()`. Found and fixed during the same
   session (not by this repo originally) -- kept as-is in
   `examples/run_gemma.py`, `get_text_config()` is the version-robust way
   to reach these fields regardless of whether the checkpoint is composite.

### Confirmed working end-to-end against the real model

- Model load, tokenization, generation (`google/gemma-3-4b-it`, CPU, eager).
- Hook attachment and capture (post-fix: event count now matches
  `layers x total_tokens` exactly, where it previously overcounted ~64x).
- KV tensor extraction via `core.cache_adapters.get_layer_kv`.
- `core.inspector`'s full stats pipeline on real Gemma keys.
- All 6 `compression/plugins.py` plugins against a real Gemma KV block.
- `core.visualize`'s PCA projection, saved as a real PNG.

### Known slow, not investigated further yet

CPU + `eager` attention + `output_attentions=True` measured at roughly
35s/generated token for `-4b-it`. Expected given eager attention
materializes the full attention matrix and `output_attentions` disables
some fused-kernel fast paths -- not yet profiled to know how much of that
is the model itself vs. this repo's hooks. `examples/run_gemma.py` now
defaults `--max-new-tokens` to 5 accordingly.
