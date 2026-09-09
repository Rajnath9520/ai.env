"""Phase 21 -- Research Experiments (Section 37).

Runs Experiments A-D (random / fixed / generic_adaptive / failure_driven)
under IDENTICAL algorithm, training budget, evaluation procedure, and seed
policy -- the only thing that differs between runs is the curriculum
strategy itself, which is what makes the resulting comparison meaningful.

Multiple seeds are used per strategy (Section 37: "Where possible use
multiple seeds") so single-seed noise doesn't get mistaken for a real
effect. Statistical significance testing (confidence intervals, t-tests)
is explicitly NOT done here -- that is Phase 24's job, and with only a
couple of seeds per strategy there isn't enough data for a meaningful test
yet. This script reports descriptive means only, and says so.

Usage:
    python -m scripts.run_research_experiments
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.manager import ExperimentManager

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_research_experiments")

RESULTS_DIR = Path("experiments/results/research")

SHARED_KWARGS = dict(
    agent_mode="demo_ppo",
    grid_size=8,
    obstacle_density=0.15,
    dead_end_probability=0.15,
    path_complexity=0.15,
    max_steps_per_episode=40,
    n_curriculum_steps=8,
    timesteps_per_step=8_000,
    eval_episodes=10,
    n_candidates=4,
    trajectories_per_analysis=4,
    run_generalization=True,
    generalization_envs_per_split=6,
    generalization_episodes_per_config=4,
)

STRATEGIES = ["random", "fixed", "generic_adaptive", "failure_driven"]
SEEDS = [7, 13]


def run_all() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict]] = {s: [] for s in STRATEGIES}

    for strategy in STRATEGIES:
        for seed in SEEDS:
            run_one(strategy, seed, summary)

    return summary


def run_one(strategy: str, seed: int, summary: dict | None = None) -> dict:
    """Runs a single (strategy, seed) pair. Split out from run_all() so
    each run can also be invoked individually -- useful for resuming after
    an interruption without re-running everything that already completed.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{strategy}_seed{seed}"
    logger.info("run_started name=%s", name)
    config = ExperimentConfig(name=name, curriculum_type=strategy, seed=seed, **SHARED_KWARGS)
    result = ExperimentManager(config).run()
    result.save(RESULTS_DIR / f"{name}.json")

    final_step = result.curriculum_history[-1] if result.curriculum_history else None
    entry = {
        "seed": seed,
        "final_success_rate": final_step.success_rate if final_step else None,
        "final_average_reward": final_step.average_reward if final_step else None,
        "generalization": result.generalization_results,
    }
    if summary is not None:
        summary.setdefault(strategy, []).append(entry)
    logger.info("run_completed name=%s final_success_rate=%.3f", name, entry["final_success_rate"] or 0.0)
    return entry


def summarize(summary: dict) -> dict:
    report = {}
    for strategy, runs in summary.items():
        success_rates = [r["final_success_rate"] for r in runs if r["final_success_rate"] is not None]

        def _split_rates(split: str) -> list[float]:
            return [
                r["generalization"][split]["success_rate"]
                for r in runs
                if r["generalization"] and split in r["generalization"]
            ]

        val_rates = _split_rates("validation")
        test_rates = _split_rates("test")
        ood_rates = _split_rates("ood")

        report[strategy] = {
            "seeds": [r["seed"] for r in runs],
            "final_success_rate_per_seed": success_rates,
            "final_success_rate_mean": statistics.mean(success_rates) if success_rates else None,
            "validation_success_rate_per_seed": val_rates,
            "validation_success_rate_mean": statistics.mean(val_rates) if val_rates else None,
            "test_success_rate_per_seed": test_rates,
            "test_success_rate_mean": statistics.mean(test_rates) if test_rates else None,
            "ood_success_rate_per_seed": ood_rates,
            "ood_success_rate_mean": statistics.mean(ood_rates) if ood_rates else None,
        }
    return report


def main() -> None:
    summary = run_all()
    report = summarize(summary)

    output_path = RESULTS_DIR / "summary.json"
    output_path.write_text(json.dumps(report, indent=2))

    logger.info("=" * 60)
    logger.info("RESEARCH EXPERIMENT SUMMARY (descriptive only, n=%d seeds/strategy)", len(SEEDS))
    for strategy, stats in report.items():
        logger.info(
            "%-18s final_success_rate=%s (mean=%.3f)  ood_success_rate=%s (mean=%.3f)",
            strategy,
            [round(v, 3) for v in stats["final_success_rate_per_seed"]],
            stats["final_success_rate_mean"] or 0.0,
            [round(v, 3) for v in stats["ood_success_rate_per_seed"]],
            stats["ood_success_rate_mean"] or 0.0,
        )
    logger.info("results_saved path=%s", output_path)


if __name__ == "__main__":
    main()
