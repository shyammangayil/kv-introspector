"""
harness/evaluator.py
=====================
Subproblem 3 (evaluation pipeline) + Meta-Harness `evaluator.py`
(Subproblem 5): benchmarks a compression plugin against baseline KV --
compression ratio, latency, reconstruction error, always -- and
perplexity / downstream-task accuracy (Wikitext-2 / ARC / HellaSwag) when a
real model, tokenizer, and dataset are supplied.

`evaluate_perplexity` / `evaluate_accuracy` are real, callable
implementations (standard sliding-window PPL; log-likelihood multi-choice
ranking), but they need a real HF model + tokenizer + dataset to produce a
real number. Neither is available in the sandbox this repo was authored in
(no GPU, no network access to huggingface.co for weights/datasets -- see
README.md "Running this against real Gemma"). Both raise
`InfrastructureUnavailable` with a specific, actionable message when
`model` is None, rather than fabricating a plausible-looking number.
`examples/run_demo.py` only calls the parts of this file that *can* run
here (compression ratio / latency / reconstruction error on the synthetic
model) and reports real numbers for those.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from compression.base import CompressionPlugin, CompressedResult


class InfrastructureUnavailable(RuntimeError):
    """Raised instead of returning a fabricated perplexity/accuracy number
    when the real model/dataset this measurement needs isn't wired in."""


@dataclass
class BenchmarkReport:
    plugin_name: str
    compression_ratio: float
    latency_ms: float
    memory_bytes_before: int
    memory_bytes_after: int
    reconstruction_max_abs_err: Optional[float]
    perplexity_before: Optional[float] = None
    perplexity_after: Optional[float] = None
    ppl_delta: Optional[float] = None
    accuracy: Optional[dict] = None
    passed: Optional[bool] = None
    notes: list = field(default_factory=list)


class Evaluator:
    """Benchmarks "compressed vs. baseline KV" (Subproblem 5's
    `harness/evaluator.py`) using the plugin contract from Subproblem 3."""

    def __init__(self, ppl_threshold_pct: float = 5.0, min_compression_ratio: float = 1.3):
        self.ppl_threshold_pct = ppl_threshold_pct
        self.min_compression_ratio = min_compression_ratio

    def benchmark_plugin(self, plugin: CompressionPlugin, kv, **plugin_kwargs) -> BenchmarkReport:
        original = np.asarray(kv)
        result: CompressedResult = plugin.run(original, **plugin_kwargs)
        recon = plugin.decompress(result)

        err = None
        try:
            if recon.shape == original.shape:
                err = float(np.max(np.abs(original - recon)))
        except Exception:
            pass

        compressed_bytes = int(original.nbytes / max(result.compression_ratio, 1e-9))
        report = BenchmarkReport(
            plugin_name=plugin.name,
            compression_ratio=result.compression_ratio,
            latency_ms=result.latency_ms,
            memory_bytes_before=original.nbytes,
            memory_bytes_after=compressed_bytes,
            reconstruction_max_abs_err=err,
        )
        if err is None and recon.shape != original.shape:
            # Shape-changing plugins (sliding_window, token_eviction,
            # head_pruning with drop=True) discard whole positions/heads
            # rather than approximating every element -- there's no
            # element-wise "error" to report for what was thrown away.
            # decompress() returns exactly what survived, unpadded (see
            # each plugin's decompress() docstring), so report that
            # honestly instead of comparing against invented zeros, which
            # a real run against Gemma revealed just measures the
            # magnitude of the discarded values, not reconstruction error.
            dropped = result.metadata.get("tokens_dropped", result.metadata.get("heads_pruned"))
            report.notes.append(
                f"reconstruction_max_abs_err is N/A: this plugin discards "
                f"{dropped if dropped is not None else 'some'} elements outright rather than "
                f"approximating them (kept elements are exact copies, 0 error by construction)"
            )
        report.passed = result.compression_ratio >= self.min_compression_ratio
        if not report.passed:
            report.notes.append(
                f"compression_ratio {result.compression_ratio:.2f}x < "
                f"min_compression_ratio {self.min_compression_ratio:.2f}x"
            )
        return report

    def evaluate_perplexity(self, model, tokenizer, texts: list[str]) -> float:
        """Standard sliding-window perplexity over `texts` (e.g. the
        Wikitext-2 test split), from the model's real per-token loss."""
        if model is None or tokenizer is None:
            raise InfrastructureUnavailable(
                "evaluate_perplexity needs a real model+tokenizer (e.g. Gemma "
                "via `transformers`) and the Wikitext-2 test split -- neither "
                "is available in this sandbox (no GPU, no huggingface.co "
                "network access). See README.md 'Running this against real "
                "Gemma' for the one-line swap once you have both."
            )
        import torch
        nlls, n_tokens = [], 0
        for text in texts:
            ids = tokenizer(text, return_tensors="pt").input_ids
            with torch.no_grad():
                out = model(ids, labels=ids)
            nlls.append(out.loss.float() * ids.shape[1])
            n_tokens += ids.shape[1]
        return float(torch.exp(torch.stack(nlls).sum() / n_tokens))

    def evaluate_accuracy(self, model, tokenizer, task: str, examples: list[dict]) -> dict:
        """Multiple-choice accuracy via log-likelihood ranking (the ARC /
        HellaSwag technique): each example is
        {"context": str, "choices": [str, ...], "answer_idx": int}; score
        every choice by summed log-prob of its continuation tokens, predict
        argmax. Same infra guard as evaluate_perplexity."""
        if model is None or tokenizer is None:
            raise InfrastructureUnavailable(
                f"evaluate_accuracy('{task}') needs a real model plus the "
                f"{task} dataset (e.g. loaded via `lm-evaluation-harness`). "
                "Not available in this sandbox -- see README.md."
            )
        import torch
        correct = 0
        for ex in examples:
            scores = []
            ctx_ids = tokenizer(ex["context"], return_tensors="pt").input_ids
            for choice in ex["choices"]:
                full_ids = tokenizer(ex["context"] + choice, return_tensors="pt").input_ids
                with torch.no_grad():
                    logits = model(full_ids).logits
                logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
                cont_ids = full_ids[0, ctx_ids.shape[1]:]
                cont_logprobs = logprobs[ctx_ids.shape[1] - 1: full_ids.shape[1] - 1]
                scores.append(cont_logprobs.gather(-1, cont_ids.unsqueeze(-1)).sum().item())
            correct += int(int(np.argmax(scores)) == ex["answer_idx"])
        return {"task": task, "n": len(examples), "accuracy": correct / max(len(examples), 1)}
