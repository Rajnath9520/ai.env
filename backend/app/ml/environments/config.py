"""Configuration for procedurally generated GridWorld environments.

Kept as a plain, validated dataclass so it can be constructed directly in
Python, loaded from an experiment.yaml (Phase 15), or produced by the
environment synthesizer (Phase 9) without any of those callers needing to
know about Gymnasium internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GridWorldConfig:
    """Parameters controlling procedural GridWorld generation.

    All "density"/"probability"/"complexity" knobs are normalized to [0, 1]
    so the environment synthesizer (Phase 9) can reason about them uniformly
    regardless of grid size.
    """

    grid_size: int = 10
    obstacle_density: float = 0.2
    goal_distance: int | None = None
    """Desired minimum Manhattan distance between start and goal.
    None means "no preference" (goal placed anywhere reachable)."""
    dead_end_probability: float = 0.2
    """Probability weight used when carving extra branches that terminate
    in dead ends, rather than only carving the shortest connecting path."""
    path_complexity: float = 0.3
    """Probability weight used when carving extra branches that create
    additional valid (non-dead-end) routes between start and goal."""
    max_steps: int = 200
    seed: int | None = None

    # Internal generation controls, not exposed to the synthesizer directly.
    max_generation_attempts: int = 200
    field_names: tuple[str, ...] = field(
        default=(
            "grid_size",
            "obstacle_density",
            "goal_distance",
            "dead_end_probability",
            "path_complexity",
            "max_steps",
        ),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.grid_size < 4:
            raise ValueError("grid_size must be >= 4 to fit start, goal, and a path")
        if not (0.0 <= self.obstacle_density <= 0.9):
            raise ValueError("obstacle_density must be in [0.0, 0.9]")
        if not (0.0 <= self.dead_end_probability <= 1.0):
            raise ValueError("dead_end_probability must be in [0.0, 1.0]")
        if not (0.0 <= self.path_complexity <= 1.0):
            raise ValueError("path_complexity must be in [0.0, 1.0]")
        if self.goal_distance is not None and self.goal_distance < 1:
            raise ValueError("goal_distance must be >= 1 when specified")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.field_names} | {
            "seed": self.seed
        }

    def difficulty_estimate(self) -> float:
        """Cheap heuristic difficulty score in [0, 1], used only for display
        and for the fixed-curriculum baseline (Phase 7). The real difficulty
        signal for research purposes is measured agent performance, not this
        estimate — this must never be treated as ground truth difficulty.
        """
        size_term = min(self.grid_size / 20.0, 1.0)
        distance_term = min((self.goal_distance or self.grid_size) / (2 * self.grid_size), 1.0)
        score = (
            0.30 * self.obstacle_density
            + 0.25 * distance_term
            + 0.20 * self.dead_end_probability
            + 0.15 * self.path_complexity
            + 0.10 * size_term
        )
        return max(0.0, min(1.0, score))
