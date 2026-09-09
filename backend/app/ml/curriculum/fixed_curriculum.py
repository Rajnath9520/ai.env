"""Baseline B (Section 14): a fixed, predetermined sequence of difficulty
stages (easy -> moderate -> hard -> very_hard), advancing purely on step
count — never on measured agent performance. That's the defining contrast
with Baseline C (Phase 8, which reacts to success rate) and the
failure-driven method (Phase 9+, which reacts to *why* the agent failed).

Each stage is a range of explicit environment parameters, matching
Section 14's instruction that "difficulty should be based on explicit
environment parameters," combined via the same difficulty_estimate()
formula already used elsewhere (app/ml/environments/config.py) so
"difficulty" means the same thing across every baseline for a fair
comparison later.
"""

from __future__ import annotations

import numpy as np

from app.ml.curriculum.base import CurriculumStep
from app.ml.environments.config import GridWorldConfig

# Stage name -> (obstacle_density, dead_end_probability, path_complexity)
# ranges, plus a goal_distance fraction of grid_size. Ranges deliberately
# overlap slightly at their edges rather than having hard gaps, so stage
# transitions aren't a discontinuous jump in practice.
STAGE_PRESETS: dict[str, dict] = {
    "easy": {
        "obstacle_density": (0.0, 0.15),
        "dead_end_probability": (0.0, 0.15),
        "path_complexity": (0.0, 0.15),
        "goal_distance_fraction": 0.3,
    },
    "moderate": {
        "obstacle_density": (0.15, 0.30),
        "dead_end_probability": (0.15, 0.35),
        "path_complexity": (0.15, 0.35),
        "goal_distance_fraction": 0.5,
    },
    "hard": {
        "obstacle_density": (0.30, 0.45),
        "dead_end_probability": (0.35, 0.55),
        "path_complexity": (0.35, 0.55),
        "goal_distance_fraction": 0.7,
    },
    "very_hard": {
        "obstacle_density": (0.45, 0.60),
        "dead_end_probability": (0.55, 0.75),
        "path_complexity": (0.55, 0.75),
        "goal_distance_fraction": 0.9,
    },
}
STAGE_ORDER: tuple[str, ...] = ("easy", "moderate", "hard", "very_hard")


class FixedCurriculum:
    name = "fixed"

    def __init__(
        self,
        total_steps: int,
        grid_size: int = 8,
        max_steps: int = 60,
        seed: int | None = None,
    ) -> None:
        if total_steps < len(STAGE_ORDER):
            raise ValueError(
                f"total_steps must be >= {len(STAGE_ORDER)} to cover every stage at least once"
            )
        self.total_steps = total_steps
        self.grid_size = grid_size
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

    def stage_for_step(self, step_index: int) -> str:
        stage_idx = min(
            len(STAGE_ORDER) - 1, (step_index * len(STAGE_ORDER)) // self.total_steps
        )
        return STAGE_ORDER[stage_idx]

    def next_environment(self, history: list[CurriculumStep]) -> GridWorldConfig:
        step_index = len(history)  # advances by step count only, never by performance
        stage = self.stage_for_step(step_index)
        preset = STAGE_PRESETS[stage]

        goal_distance = max(1, round(preset["goal_distance_fraction"] * self.grid_size))

        return GridWorldConfig(
            grid_size=self.grid_size,
            obstacle_density=float(self._rng.uniform(*preset["obstacle_density"])),
            goal_distance=goal_distance,
            dead_end_probability=float(self._rng.uniform(*preset["dead_end_probability"])),
            path_complexity=float(self._rng.uniform(*preset["path_complexity"])),
            max_steps=self.max_steps,
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )
