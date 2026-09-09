"""Baseline A (Section 13): the simplest possible environment-generation
strategy — sample each environment parameter uniformly at random, within
configurable ranges, every step, regardless of agent performance.

This does not "cheat" toward invalid environments: parameter *ranges* are
randomized here, but Phase 1's GridWorldGenerator still guarantees the
resulting concrete layout is solvable, so validity is never in question —
only how interesting/targeted the resulting environment is, which is
exactly what this baseline deliberately does not optimize for.
"""

from __future__ import annotations

import numpy as np

from app.ml.curriculum.base import CurriculumStep
from app.ml.environments.config import GridWorldConfig


class RandomCurriculum:
    name = "random"

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 60,
        obstacle_density_range: tuple[float, float] = (0.0, 0.5),
        dead_end_probability_range: tuple[float, float] = (0.0, 0.6),
        path_complexity_range: tuple[float, float] = (0.0, 0.6),
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obstacle_density_range = obstacle_density_range
        self.dead_end_probability_range = dead_end_probability_range
        self.path_complexity_range = path_complexity_range
        self._rng = np.random.default_rng(seed)

    def next_environment(self, history: list[CurriculumStep]) -> GridWorldConfig:
        return GridWorldConfig(
            grid_size=self.grid_size,
            obstacle_density=float(self._rng.uniform(*self.obstacle_density_range)),
            goal_distance=None,  # random goal placement, no explicit distance target
            dead_end_probability=float(self._rng.uniform(*self.dead_end_probability_range)),
            path_complexity=float(self._rng.uniform(*self.path_complexity_range)),
            max_steps=self.max_steps,
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )
