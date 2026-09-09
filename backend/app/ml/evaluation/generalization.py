"""Phase 14 -- Generalization Testing (Section 21).

Problem solved: every prior phase evaluates the agent on environments
drawn from the same distribution (often literally the same config) it just
trained on. That answers "did the agent memorize this," not "did it learn
something that transfers." This module enforces a hard separation between
training, validation, test, and out-of-distribution (OOD) environment
pools, and a `GeneralizationEvaluator` that only ever *evaluates* (never
trains) on non-training splits -- structurally, not just by convention,
since `AgentTrainer.evaluate()` never calls `train()` internally.

Split construction:
- train / validation / test: same parameter *distribution* as the base
  config, but drawn from disjoint, non-overlapping seed ranges, so no
  concrete environment can appear in more than one split even by chance.
- ood: deliberately shifted parameters (larger grid, higher obstacle
  density / dead-end probability / path complexity) on top of a disjoint
  seed range, per Section 21's "different obstacle layouts, different goal
  locations, different grid sizes, different parameter combinations."

Limitation: uniqueness of concrete *layouts* across splits rests on (a)
disjoint seed ranges and (b) the generator being a deterministic function
of its seed -- this makes any accidental overlap vanishingly unlikely but
is not a formal proof of zero collision probability. What Section 21 most
directly requires -- the agent never being *trained* on an environment
config drawn from the validation/test/ood seed ranges -- is guaranteed by
construction: nothing in this module calls `AgentTrainer.train()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.environments.config import GridWorldConfig
from app.ml.training.trainer import AgentTrainer

SplitName = str  # "train" | "validation" | "test" | "ood"

# Disjoint integer seed ranges, one per split. Non-overlapping by
# construction -- the invariant this whole module exists to protect.
SEED_RANGES: dict[SplitName, tuple[int, int]] = {
    "train": (0, 100_000),
    "validation": (100_000, 200_000),
    "test": (200_000, 300_000),
    "ood": (300_000, 400_000),
}

MIN_GRID_SIZE = 4
MAX_GRID_SIZE = 24

# How much OOD configs shift each parameter relative to the base config,
# on top of using the disjoint "ood" seed range.
OOD_GRID_SIZE_DELTA = 4
OOD_OBSTACLE_DENSITY_SHIFT = 0.2
OOD_DEAD_END_PROBABILITY_SHIFT = 0.2
OOD_PATH_COMPLEXITY_SHIFT = 0.2


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class EnvironmentSplitGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample_seed(self, split: SplitName) -> int:
        lo, hi = SEED_RANGES[split]
        return int(self._rng.integers(lo, hi))

    def in_distribution_configs(
        self, base_config: GridWorldConfig, split: SplitName, n: int
    ) -> list[GridWorldConfig]:
        if split == "ood":
            raise ValueError("use ood_configs() for the ood split")
        return [
            GridWorldConfig(**{**base_config.to_dict(), "seed": self.sample_seed(split)})
            for _ in range(n)
        ]

    def ood_configs(self, base_config: GridWorldConfig, n: int) -> list[GridWorldConfig]:
        grid_size = int(
            _clamp(base_config.grid_size + OOD_GRID_SIZE_DELTA, MIN_GRID_SIZE, MAX_GRID_SIZE)
        )
        obstacle_density = _clamp(
            base_config.obstacle_density + OOD_OBSTACLE_DENSITY_SHIFT, 0.0, 0.9
        )
        dead_end_probability = _clamp(
            base_config.dead_end_probability + OOD_DEAD_END_PROBABILITY_SHIFT, 0.0, 1.0
        )
        path_complexity = _clamp(
            base_config.path_complexity + OOD_PATH_COMPLEXITY_SHIFT, 0.0, 1.0
        )
        return [
            GridWorldConfig(
                grid_size=grid_size,
                obstacle_density=obstacle_density,
                goal_distance=None,
                dead_end_probability=dead_end_probability,
                path_complexity=path_complexity,
                max_steps=base_config.max_steps,
                seed=self.sample_seed("ood"),
            )
            for _ in range(n)
        ]

    def build_all_splits(
        self, base_config: GridWorldConfig, n_per_split: int
    ) -> dict[SplitName, list[GridWorldConfig]]:
        return {
            "train": self.in_distribution_configs(base_config, "train", n_per_split),
            "validation": self.in_distribution_configs(base_config, "validation", n_per_split),
            "test": self.in_distribution_configs(base_config, "test", n_per_split),
            "ood": self.ood_configs(base_config, n_per_split),
        }

    @staticmethod
    def assert_no_seed_overlap(splits: dict[SplitName, list[GridWorldConfig]]) -> None:
        """Raises AssertionError if any two splits share a config seed --
        the concrete, checkable form of "never train on test/ood
        environments." Intended for use in tests and experiment sanity
        checks, not hot paths.
        """
        seed_sets = {name: {c.seed for c in configs} for name, configs in splits.items()}
        names = list(seed_sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = seed_sets[names[i]] & seed_sets[names[j]]
                assert not overlap, f"seed overlap between {names[i]!r} and {names[j]!r}: {overlap}"


@dataclass(frozen=True)
class GeneralizationResult:
    split: SplitName
    n_environments: int
    episodes: int
    success_rate: float
    average_reward: float
    average_steps: float
    collision_rate: float

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "n_environments": self.n_environments,
            "episodes": self.episodes,
            "success_rate": round(self.success_rate, 4),
            "average_reward": round(self.average_reward, 4),
            "average_steps": round(self.average_steps, 4),
            "collision_rate": round(self.collision_rate, 4),
        }


class GeneralizationEvaluator:
    """Evaluates a trained policy against pre-built environment splits.
    Deliberately has no method that trains anything -- generalization
    evaluation and training are different operations on purpose.
    """

    def evaluate_split(
        self,
        trainer: AgentTrainer,
        configs: list[GridWorldConfig],
        split: SplitName,
        episodes_per_config: int = 5,
    ) -> GeneralizationResult:
        if not configs:
            return GeneralizationResult(split, 0, 0, 0.0, 0.0, 0.0, 0.0)

        all_success: list[bool] = []
        all_rewards: list[float] = []
        all_steps: list[float] = []
        all_collision_flags: list[float] = []

        for config in configs:
            evaluation = trainer.evaluate(
                config=config,
                episodes=episodes_per_config,
                deterministic=True,
                environment_id=f"{split}_{config.seed}",
            )
            all_success.extend(evaluation.episode_successes)
            all_rewards.extend(evaluation.episode_rewards)
            all_steps.append(evaluation.average_steps)
            all_collision_flags.append(evaluation.collision_rate)

        total_episodes = len(all_success)
        return GeneralizationResult(
            split=split,
            n_environments=len(configs),
            episodes=total_episodes,
            success_rate=float(np.mean(all_success)) if all_success else 0.0,
            average_reward=float(np.mean(all_rewards)) if all_rewards else 0.0,
            average_steps=float(np.mean(all_steps)) if all_steps else 0.0,
            collision_rate=float(np.mean(all_collision_flags)) if all_collision_flags else 0.0,
        )

    def evaluate_all_splits(
        self,
        trainer: AgentTrainer,
        splits: dict[SplitName, list[GridWorldConfig]],
        episodes_per_config: int = 5,
    ) -> dict[SplitName, GeneralizationResult]:
        return {
            name: self.evaluate_split(trainer, configs, name, episodes_per_config)
            for name, configs in splits.items()
            if name != "train"
        }
