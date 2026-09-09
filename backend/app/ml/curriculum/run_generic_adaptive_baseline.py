"""CLI for Phase 8: `python -m app.ml.curriculum.run_generic_adaptive_baseline`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.ml.curriculum.generic_adaptive import GenericAdaptiveCurriculum
from app.ml.curriculum.runner import CurriculumRunner
from app.ml.training.trainer import AgentTrainer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_generic_adaptive_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Baseline C: generic adaptive curriculum.")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps-per-episode", type=int, default=60)
    parser.add_argument("--n-curriculum-steps", type=int, default=20)
    parser.add_argument("--timesteps-per-step", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=str,
        default="experiments/results/generic_adaptive_baseline.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategy = GenericAdaptiveCurriculum(
        grid_size=args.grid_size,
        max_steps=args.max_steps_per_episode,
        seed=args.seed,
    )
    trainer = AgentTrainer(seed=args.seed)
    runner = CurriculumRunner(
        trainer=trainer,
        strategy=strategy,
        timesteps_per_step=args.timesteps_per_step,
        eval_episodes=args.eval_episodes,
    )

    logger.info("experiment_started curriculum=generic_adaptive n_steps=%d", args.n_curriculum_steps)
    history = runner.run(args.n_curriculum_steps)

    for step in history:
        difficulty = strategy.current_difficulty(history[: step.step_index])
        logger.info(
            "environment_selected id=%s difficulty_before_step=%.2f actual_difficulty=%.2f "
            "success_rate=%.2f avg_reward=%.2f",
            step.environment_id,
            difficulty,
            step.difficulty,
            step.success_rate,
            step.average_reward,
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([s.to_dict() for s in history], indent=2))
    logger.info("results_saved path=%s steps=%d", output_path, len(history))


if __name__ == "__main__":
    main()
