"""
integrations/gepa_adapter.py
=============================
Subproblem 5: GEPA integration.

GEPA ("Genetic-Pareto") is a real, published reflective-prompt-optimization
framework (Agrawal et al., "GEPA: Reflective Prompt Evolution Can
Outperform Reinforcement Learning", 2025; code at
https://github.com/gepa-ai/gepa; also available as `dspy.GEPA`). Its core
idea: sample execution trajectories, reflect on them in natural language,
and mutate prompts based on the lessons learned, keeping a Pareto-diverse
set of high performers rather than collapsing everything to one scalar
reward.

This adapter builds the piece specific to this project -- turning
`harness.router.Router` experiment results into the trajectory+feedback
shape a reflective optimizer consumes, and maintaining the local
`experience.db` / `prompts/*.md` files the spec describes. It does not
depend on the `gepa` PyPI package (not installed here, and not needed for
the adapter logic below). `to_gepa_feedback()` is the integration seam:
feed its output to `gepa.optimize(...)` / `dspy.GEPA(...)` if you wire in
the real optimizer; `suggest_prompt_update()` is a zero-dependency
rule-based stand-in for the reflection step, useful on its own and as a
reference for what the real reflector should be able to derive from the
same rows.
"""
from __future__ import annotations

import time
from pathlib import Path

from core.storage import ExperienceDB
from harness.evaluator import BenchmarkReport
from harness.router import ExperimentConfig


def to_gepa_feedback(config: ExperimentConfig, report: BenchmarkReport) -> dict:
    """Formats one experiment as the {trajectory, feedback, score} shape
    GEPA's reflection step expects: a natural-language account of what was
    tried and what happened, not just a scalar -- textual feedback over
    scalar reward is GEPA's whole premise."""
    verdict = "passed" if report.passed else "failed"
    feedback_text = (
        f"Plugin '{config.plugin}' with config {config.plugin_kwargs} on layer "
        f"{config.layer_idx} {verdict}: {report.compression_ratio:.2f}x compression, "
        f"{report.latency_ms:.2f}ms latency"
    )
    if report.ppl_delta is not None:
        feedback_text += f", {report.ppl_delta:+.2f} PPL delta"
    if report.notes:
        feedback_text += f". Notes: {'; '.join(report.notes)}"
    return {
        "trajectory": {"plugin": config.plugin, "config": config.plugin_kwargs, "layer_idx": config.layer_idx},
        "feedback": feedback_text,
        "score": report.compression_ratio if report.passed else 0.0,
    }


class GEPAAdapter:
    """Reads ExperienceDB rows, derives lessons, and rewrites
    `prompts/planner.md` / `prompts/reflection.md` -- the spec's "Update
    prompts/planner.md, reflection.md, etc. based on memory/experience.db
    lessons" requirement."""

    def __init__(self, experience_db: ExperienceDB, prompts_dir: str = "prompts"):
        self.experience_db = experience_db
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, config: ExperimentConfig, report: BenchmarkReport) -> int:
        return self.experience_db.log(
            experiment_name=config.name, plugin=config.plugin, config=config.plugin_kwargs,
            compression_ratio=report.compression_ratio, ppl_delta=report.ppl_delta,
            latency_ms=report.latency_ms, passed=bool(report.passed),
            lesson=to_gepa_feedback(config, report)["feedback"],
        )

    def suggest_prompt_update(self, n_recent: int = 10) -> str:
        """Rule-based stand-in for a GEPA reflection pass: turns the most
        recent lessons into a short instructional addendum. Swap this
        method's body for a real `gepa.optimize(...)` / `dspy.GEPA(...)`
        call to get genuine LLM-driven reflective mutation -- the
        ExperienceDB rows are already shaped for that (see
        `to_gepa_feedback`)."""
        rows = self.experience_db.recent(n_recent)
        if not rows:
            return "(no experience yet -- run some experiments first)"

        best = max(rows, key=lambda r: (r["passed"], r["compression_ratio"] or 0))
        failures = [r for r in rows if not r["passed"]]

        lines = [
            f"## Auto-generated addendum (from experience.db, {len(rows)} recent "
            f"experiments, {time.strftime('%Y-%m-%d %H:%M')})",
            "",
            f"- Best so far: `{best['plugin']}` ({best['compression_ratio']:.2f}x, "
            f"{best['latency_ms']:.2f}ms) -- {best['lesson']}",
        ]
        if failures:
            plugins = sorted({r["plugin"] for r in failures})
            lines.append(f"- Avoid without changes: {', '.join(plugins)} "
                          f"(failed {len(failures)}/{len(rows)} recent runs)")
        lines.append(
            f"- Next experiment to try: push `{best['plugin']}` more aggressively while "
            "PPL-delta budget remains (success criterion: >=30% size reduction, <5% PPL increase)."
        )
        return "\n".join(lines)

    def update_planner_md(self, n_recent: int = 10) -> Path:
        addendum = self.suggest_prompt_update(n_recent)
        path = self.prompts_dir / "planner.md"
        existing = path.read_text() if path.exists() else "# Planner\n\nBase planning instructions go here.\n"
        path.write_text(existing.rstrip() + "\n\n" + addendum + "\n")
        return path
