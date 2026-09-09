"""Phase 22 -- run the ablation study (Section 38).

Runs Variants A-D under IDENTICAL budget/seed conditions -- same pattern as
Phase 21's research experiments, extended to isolate which component of the
failure-driven method actually contributes. Descriptive results only, same
honesty constraints as Phase 21 (no significance testing at this seed
count -- Phase 24's job).
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from app.ml.curriculum.ablation import ABLATION_VARIANTS, build_ablation_runner
from app.ml.environments.config import GridWorldConfig
from app.ml.training.trainer import AgentTrainer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_ablation_study")

RESULTS_DIR = Path("experiments/results/ablation")

SHARED_KWARGS = dict(
    timesteps_per_step=8_000,
    eval_episodes=10,
    n_candidates=4,
    trajectories_per_analysis=4,
)
N_STEPS = 8
SEEDS = [7, 13]


def base_config(seed: int) -> GridWorldConfig:
    return GridWorldConfig(
        grid_size=8,
        obstacle_density=0.15,
        dead_end_probability=0.15,
        path_complexity=0.15,
        max_steps=40,
        seed=seed,
    )


def run_one(variant_key: str, seed: int, summary: dict | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{variant_key}_seed{seed}"
    logger.info("run_started name=%s", name)

    trainer = AgentTrainer(seed=seed)
    runner = build_ablation_runner(variant_key, trainer, base_config(seed), **SHARED_KWARGS)
    result = runner.run(N_STEPS)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(result.to_dict(), indent=2))

    final_step = result.history[-1] if result.history else None
    entry = {
        "seed": seed,
        "final_success_rate": final_step.success_rate if final_step else None,
        "final_average_reward": final_step.average_reward if final_step else None,
    }
    if summary is not None:
        summary.setdefault(variant_key, []).append(entry)
    logger.info("run_completed name=%s final_success_rate=%.3f", name, entry["final_success_rate"] or 0.0)
    return entry


def run_all() -> dict:
    summary: dict[str, list[dict]] = {v: [] for v in ABLATION_VARIANTS}
    for variant_key in ABLATION_VARIANTS:
        for seed in SEEDS:
            run_one(variant_key, seed, summary)
    return summary


def summarize(summary: dict) -> dict:
    report = {}
    for variant_key, runs in summary.items():
        rates = [r["final_success_rate"] for r in runs if r["final_success_rate"] is not None]
        report[variant_key] = {
            "label": ABLATION_VARIANTS[variant_key].label if variant_key in ABLATION_VARIANTS else variant_key,
            "seeds": [r["seed"] for r in runs],
            "final_success_rate_per_seed": rates,
            "final_success_rate_mean": statistics.mean(rates) if rates else None,
        }
    return report


def main() -> None:
    summary = run_all()
    report = summarize(summary)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, indent=2))

    logger.info("=" * 60)
    logger.info("ABLATION STUDY SUMMARY (descriptive only, n=%d seeds/variant)", len(SEEDS))
    for variant_key, stats in report.items():
        logger.info(
            "%-28s final_success_rate=%s (mean=%.3f)",
            variant_key,
            [round(v, 3) for v in stats["final_success_rate_per_seed"]],
            stats["final_success_rate_mean"] or 0.0,
        )


if __name__ == "__main__":
    main()
