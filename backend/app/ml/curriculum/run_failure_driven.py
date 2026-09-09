"""CLI for Phase 13: `python -m app.ml.curriculum.run_failure_driven`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.ml.curriculum.failure_driven import AdaptiveCurriculumRunner
from app.ml.environments.config import GridWorldConfig
from app.ml.training.trainer import AgentTrainer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_failure_driven")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the proposed method: failure-driven adaptive curriculum.")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps-per-episode", type=int, default=60)
    parser.add_argument("--n-curriculum-steps", type=int, default=20)
    parser.add_argument("--timesteps-per-step", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--trajectories-per-analysis", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=str,
        default="experiments/results/failure_driven.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_config = GridWorldConfig(
        grid_size=args.grid_size,
        obstacle_density=0.15,
        dead_end_probability=0.15,
        path_complexity=0.15,
        max_steps=args.max_steps_per_episode,
        seed=args.seed,
    )
    trainer = AgentTrainer(seed=args.seed)
    runner = AdaptiveCurriculumRunner(
        trainer=trainer,
        base_config=base_config,
        timesteps_per_step=args.timesteps_per_step,
        eval_episodes=args.eval_episodes,
        n_candidates=args.n_candidates,
        trajectories_per_analysis=args.trajectories_per_analysis,
    )

    logger.info("experiment_started curriculum=failure_driven n_steps=%d", args.n_curriculum_steps)
    result = runner.run(args.n_curriculum_steps)

    for step, weakness in zip(result.history, result.weakness_history):
        logger.info(
            "environment_selected id=%s target_weakness=%s difficulty=%.2f "
            "success_rate=%.2f learning_progress=%s | next_primary_weakness=%s",
            step.environment_id,
            step.target_weakness,
            step.difficulty,
            step.success_rate,
            f"{step.learning_progress:+.3f}" if step.learning_progress is not None else "--",
            weakness.primary_weakness,
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("results_saved path=%s steps=%d", output_path, len(result.history))


if __name__ == "__main__":
    main()
