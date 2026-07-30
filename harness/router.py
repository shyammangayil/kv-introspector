"""
harness/router.py
==================
Subproblem 5 (Meta-Harness `router.py`): routes experiment configs to
compression plugins and the Evaluator, through `harness/retry.py` and
`harness/budget.py`, logging every result to `core.storage.ExperienceDB`.
This is the piece Autoresearch's planner
(`integrations/autoresearch_adapter.py`) calls to actually run an
experiment it decided on, and what GEPA's adapter
(`integrations/gepa_adapter.py`) reads lessons from afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from compression.plugins import get_plugin, PLUGIN_REGISTRY
from harness.evaluator import Evaluator, BenchmarkReport
from harness.retry import RetryPolicy
from harness.budget import BudgetEnforcer, BudgetExceeded
from core.storage import ExperienceDB


@dataclass
class ExperimentConfig:
    name: str
    plugin: str
    plugin_kwargs: dict = field(default_factory=dict)
    layer_idx: Optional[int] = None


class Router:
    def __init__(
        self,
        evaluator: Optional[Evaluator] = None,
        retry: Optional[RetryPolicy] = None,
        budget: Optional[BudgetEnforcer] = None,
        experience_db: Optional[ExperienceDB] = None,
    ):
        self.evaluator = evaluator or Evaluator()
        self.retry = retry or RetryPolicy()
        self.budget = budget
        self.experience_db = experience_db

    def run_experiment(self, config: ExperimentConfig, kv) -> BenchmarkReport:
        if config.plugin not in PLUGIN_REGISTRY:
            raise KeyError(f"Unknown plugin {config.plugin!r}. Available: {list(PLUGIN_REGISTRY)}")

        kv_arr = np.asarray(kv)
        if self.budget is not None:
            self.budget.check(kv_arr.nbytes)  # raises BudgetExceeded before doing any work

        def _attempt() -> BenchmarkReport:
            plugin = get_plugin(config.plugin)
            return self.evaluator.benchmark_plugin(plugin, kv_arr, **config.plugin_kwargs)

        report = self.retry.run(_attempt, fallback=lambda: self._fallback_report(config))

        if self.experience_db is not None:
            self.experience_db.log(
                experiment_name=config.name, plugin=config.plugin,
                config=config.plugin_kwargs, compression_ratio=report.compression_ratio,
                ppl_delta=report.ppl_delta, latency_ms=report.latency_ms,
                passed=bool(report.passed), lesson=self._lesson(config, report),
            )
        return report

    def run_sweep(self, plugins: list[str], kv, **shared_kwargs) -> list[BenchmarkReport]:
        """Runs every plugin in `plugins` against the same KV block --
        what Autoresearch's planner calls for a broad first pass before
        drilling into a specific plugin/layer."""
        return [
            self.run_experiment(
                ExperimentConfig(name=f"sweep_{p}", plugin=p, plugin_kwargs=dict(shared_kwargs)), kv
            )
            for p in plugins
        ]

    @staticmethod
    def _fallback_report(config: ExperimentConfig) -> BenchmarkReport:
        """What retry.py falls back to once retries are exhausted: a
        full-precision (uncompressed), explicitly-failed report -- the
        "falls back to full-precision KV" resilience criterion."""
        return BenchmarkReport(
            plugin_name=config.plugin, compression_ratio=1.0, latency_ms=0.0,
            memory_bytes_before=0, memory_bytes_after=0,
            reconstruction_max_abs_err=None, passed=False,
            notes=[f"FALLBACK: {config.plugin} failed after retries; kept full-precision KV"],
        )

    @staticmethod
    def _lesson(config: ExperimentConfig, report: BenchmarkReport) -> str:
        """Rule-based one-line lesson. This exact string is what
        integrations/gepa_adapter.py folds into planner.md / reflection.md
        -- see that file for the full reflective-update loop."""
        if report.ppl_delta is not None and report.ppl_delta < 1.0 and report.compression_ratio > 2:
            return (f"{config.plugin} loses <1 PPL at {report.compression_ratio:.1f}x "
                     "-> try more aggressive settings")
        if not report.passed:
            return (f"{config.plugin} under-compressed ({report.compression_ratio:.2f}x) "
                     f"with config {config.plugin_kwargs} -> avoid this config")
        return f"{config.plugin} reached {report.compression_ratio:.2f}x compression, {report.latency_ms:.2f}ms latency"
