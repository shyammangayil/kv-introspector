"""
integrations/autoresearch_adapter.py
======================================
Subproblem 5: Autoresearch integration.

Reference implementations of `agents/planner.py` ("prioritize experiments,
e.g. 'Test head pruning on Layer 16'") and `agents/reflector.py` ("analyze
results, e.g. 'Head 7 is unused in 90% of cases'") against this project's
ExperienceDB + KnowledgeGraphDB.

Deliberately rule-based, with no LLM call inside this module -- these are
real, deterministic, testable functions that produce exactly the *inputs*
an LLM-driven planner/reflector would reason over (recent experiment
history; per-head importance stats), rather than a mocked LLM call standing
in for a real one and inventing results. Point your actual Autoresearch
agents at `Planner.suggest_next_experiments` / `Reflector.analyze_head_usage`
as their tool functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from compression.plugins import PLUGIN_REGISTRY
from core.storage import ExperienceDB, KnowledgeGraphDB
from harness.router import ExperimentConfig


@dataclass
class ExperimentSuggestion:
    config: ExperimentConfig
    rationale: str
    priority: float  # higher = more promising; used to sort a work queue


class Planner:
    """`agents/planner.py`: decides what to try next, from ExperienceDB
    history alone (no model access needed -- this can run between
    experiment batches, offline)."""

    def __init__(self, experience_db: ExperienceDB):
        self.experience_db = experience_db

    def suggest_next_experiments(
        self, n: int = 3, layer_idx: Optional[int] = None
    ) -> list[ExperimentSuggestion]:
        history = self.experience_db.recent(50)
        tried_plugins = {r["plugin"] for r in history}
        untried = [p for p in PLUGIN_REGISTRY if p not in tried_plugins]

        suggestions: list[ExperimentSuggestion] = [
            ExperimentSuggestion(
                config=ExperimentConfig(name=f"explore_{p}", plugin=p, layer_idx=layer_idx),
                rationale=f"'{p}' has not been tried yet -- exploration priority.",
                priority=1.0,
            )
            for p in untried
        ]

        passed = [r for r in history if r["passed"]]
        if passed:
            best = max(passed, key=lambda r: r["compression_ratio"] or 0)
            suggestions.append(ExperimentSuggestion(
                config=ExperimentConfig(
                    name=f"exploit_{best['plugin']}_more_aggressive",
                    plugin=best["plugin"], layer_idx=layer_idx,
                ),
                rationale=(f"'{best['plugin']}' is the best passing plugin so far "
                           f"({best['compression_ratio']:.2f}x) -- push it further."),
                priority=0.8,
            ))

        for p in sorted({r["plugin"] for r in history if not r["passed"]}):
            suggestions.append(ExperimentSuggestion(
                config=ExperimentConfig(name=f"retry_{p}_gentler", plugin=p, layer_idx=layer_idx),
                rationale=f"'{p}' failed previously -- retry with a gentler (less aggressive) config.",
                priority=0.4,
            ))

        suggestions.sort(key=lambda s: -s.priority)
        return suggestions[:n]


class Reflector:
    """`agents/reflector.py`: analyzes results after the fact."""

    def __init__(self, knowledge_graph: KnowledgeGraphDB):
        self.kg = knowledge_graph

    def analyze_head_usage(
        self, head_importance_by_run: list[np.ndarray], threshold: float = 0.5
    ) -> dict:
        """head_importance_by_run: list of [num_heads] arrays, one per
        run/prompt (e.g. from `KVCacheInspector.head_importance`). Reports,
        per head, the fraction of runs it fell below `threshold` in --
        directly answers the spec's example research question, "Which
        heads are unused?", and writes each head's verdict into the
        knowledge graph (Ontology criterion)."""
        if not head_importance_by_run:
            return {}
        stacked = np.stack(head_importance_by_run)             # [n_runs, n_heads]
        underused_frac = (stacked < threshold).mean(axis=0)      # [n_heads]
        result = {
            int(h): {
                "underused_fraction": float(underused_frac[h]),
                "mean_importance": float(stacked[:, h].mean()),
                "verdict": "likely prunable" if underused_frac[h] >= 0.9 else "keep",
            }
            for h in range(stacked.shape[1])
        }
        for h, info in result.items():
            self.kg.upsert_entity("Head", f"head:{h}", **info)
        return result

    def summarize_experience(self, experience_db: ExperienceDB, n: int = 50) -> str:
        rows = experience_db.recent(n)
        if not rows:
            return "No experiments logged yet."
        n_passed = sum(1 for r in rows if r["passed"])
        by_plugin: dict[str, list[float]] = {}
        for r in rows:
            by_plugin.setdefault(r["plugin"], []).append(r["compression_ratio"] or 0)
        lines = [f"{n_passed}/{len(rows)} recent experiments passed."]
        for plugin, ratios in sorted(by_plugin.items()):
            lines.append(f"  - {plugin}: mean {np.mean(ratios):.2f}x over {len(ratios)} runs")
        return "\n".join(lines)
