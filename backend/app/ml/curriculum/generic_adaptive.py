"""Baseline C (Section 15): adapts difficulty based on *overall* success
rate only — no failure-type detail at all. This is the last baseline before
the proposed method (Phase 9+), and the most important one to distinguish
from it: both this and the failure-driven method "adapt," but this one only
ever asks "is the agent doing well or poorly overall," never "doing poorly
*at what, specifically*."

Rule (Section 15, applied verbatim):
    success_rate > 0.85  -> increase difficulty
    success_rate < 0.40  -> decrease difficulty
    otherwise            -> maintain difficulty

Design choice: rather than storing mutable difficulty state on the
strategy object, `next_environment` recomputes the current difficulty by
replaying the increase/decrease/maintain rule over the full history each
call. This keeps the strategy a pure function of `history` — consistent
with RandomCurriculum/FixedCurriculum and much easier to unit test (no
hidden internal state to reset between test cases) — at the cost of O(n)
recomputation per call, which is negligible at curriculum lengths this
project uses (tens to low hundreds of steps).
"""

from __future__ import annotations

import numpy as np

from app.ml.curriculum.base import CurriculumStep
from app.ml.environments.config import GridWorldConfig

# Maximum parameter values reached at difficulty=1.0 — matches the upper
# bounds FixedCurriculum's "very_hard" stage uses, so a difficulty=1.0
# environment here is comparable in intent to a very_hard fixed-curriculum
# environment.
MAX_OBSTACLE_DENSITY = 0.60
MAX_DEAD_END_PROBABILITY = 0.75
MAX_PATH_COMPLEXITY = 0.75
MIN_GOAL_DISTANCE_FRACTION = 0.2
MAX_GOAL_DISTANCE_FRACTION = 0.9


class GenericAdaptiveCurriculum:
    name = "generic_adaptive"

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 60,
        initial_difficulty: float = 0.15,
        difficulty_step: float = 0.10,
        increase_threshold: float = 0.85,
        decrease_threshold: float = 0.40,
        seed: int | None = None,
    ) -> None:
        if not (0.0 <= initial_difficulty <= 1.0):
            raise ValueError("initial_difficulty must be in [0.0, 1.0]")
        if decrease_threshold >= increase_threshold:
            raise ValueError("decrease_threshold must be < increase_threshold")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.initial_difficulty = initial_difficulty
        self.difficulty_step = difficulty_step
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self._rng = np.random.default_rng(seed)

    def current_difficulty(self, history: list[CurriculumStep]) -> float:
        difficulty = self.initial_difficulty
        for step in history:
            if step.success_rate > self.increase_threshold:
                difficulty = min(1.0, difficulty + self.difficulty_step)
            elif step.success_rate < self.decrease_threshold:
                difficulty = max(0.0, difficulty - self.difficulty_step)
            # else: within the "maintain" band — difficulty unchanged
        return difficulty

    def next_environment(self, history: list[CurriculumStep]) -> GridWorldConfig:
        difficulty = self.current_difficulty(history)

        goal_distance_fraction = (
            MIN_GOAL_DISTANCE_FRACTION
            + difficulty * (MAX_GOAL_DISTANCE_FRACTION - MIN_GOAL_DISTANCE_FRACTION)
        )

        return GridWorldConfig(
            grid_size=self.grid_size,
            obstacle_density=difficulty * MAX_OBSTACLE_DENSITY,
            goal_distance=max(1, round(goal_distance_fraction * self.grid_size)),
            dead_end_probability=difficulty * MAX_DEAD_END_PROBABILITY,
            path_complexity=difficulty * MAX_PATH_COMPLEXITY,
            max_steps=self.max_steps,
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )
