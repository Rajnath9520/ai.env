"""CLI command for Phase 2: `python -m app.ml.training.train_baseline`.

Trains a plain PPO agent on a single fixed GridWorldConfig with no
curriculum at all. This must work — and be shown to work — before any
adaptive curriculum logic is introduced (Section 9: "The baseline must work
BEFORE any adaptive curriculum is implemented.").
"""

from __future__ import annotations

import argparse
import logging

from app.ml.environments.config import GridWorldConfig
from app.ml.training.trainer import AgentTrainer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("train_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PPO GridWorld baseline agent.")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--obstacle-density", type=float, default=0.25)
    parser.add_argument("--dead-end-probability", type=float, default=0.2)
    parser.add_argument("--path-complexity", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="experiments/results/checkpoints/baseline_ppo.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = GridWorldConfig(
        grid_size=args.grid_size,
        obstacle_density=args.obstacle_density,
        dead_end_probability=args.dead_end_probability,
        path_complexity=args.path_complexity,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    logger.info("experiment_started config=%s", config.to_dict())

    trainer = AgentTrainer(seed=args.seed)
    result = trainer.train(
        config=config,
        total_timesteps=args.total_timesteps,
        environment_id="baseline_fixed_env",
    )
    logger.info(
        "training_completed episodes=%d final_success_rate=%.3f final_average_reward=%.3f",
        len(result.episodes),
        result.final_success_rate,
        result.final_average_reward,
    )

    evaluation = trainer.evaluate(
        config=config,
        episodes=args.eval_episodes,
        deterministic=True,
        environment_id="baseline_fixed_env",
    )
    logger.info(
        "evaluation_completed episodes=%d success_rate=%.3f average_reward=%.3f "
        "average_steps=%.1f collision_rate=%.3f",
        evaluation.episodes,
        evaluation.success_rate,
        evaluation.average_reward,
        evaluation.average_steps,
        evaluation.collision_rate,
    )

    trainer.save_checkpoint(args.checkpoint_path)
    logger.info("checkpoint_saved path=%s", args.checkpoint_path)


if __name__ == "__main__":
    main()
