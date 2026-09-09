"""Phase 28 (Section 7/13) -- EnvironmentAnalyzer.

Problem solved: environment-only mode (Section 3) needs real, inspectable
characteristics of a generated environment -- grid dimensions, obstacle
density, path length, reachable cells, dead ends, branching factor,
connectivity, a difficulty estimate, and an overall generation score -- and
those characteristics must never be confused with agent-training metrics
(reward, episode, success rate). This module computes exactly the former
and nothing that depends on an agent ever having run.

Every number here is derived from the environment's own generated layout
(via the existing GridWorldGenerator/EnvironmentValidator/dead-end analysis
already used elsewhere in the project) -- nothing is estimated, guessed, or
faked.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.analysis.dead_ends import dead_end_zone_cells, reachable_cells
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GenerationReport
from app.ml.environments.validator import GridWorldLayout


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class EnvironmentMetrics:
    """Structural, agent-free characteristics of one generated environment
    (Section 13). None of these fields describe agent behavior."""

    grid_size: int
    obstacle_density_actual: float
    dead_end_probability_config: float
    path_complexity_config: float
    path_length: int | None
    reachable_cells: int
    total_cells: int
    dead_end_cells: int
    branching_factor: float
    connectivity: float
    is_valid: bool
    validation_reasons: tuple[str, ...]
    difficulty_estimate: float
    generation_score: float

    def to_dict(self) -> dict:
        return {
            "grid_size": self.grid_size,
            "obstacle_density_actual": round(self.obstacle_density_actual, 4),
            "dead_end_probability_config": round(self.dead_end_probability_config, 4),
            "path_complexity_config": round(self.path_complexity_config, 4),
            "path_length": self.path_length,
            "reachable_cells": self.reachable_cells,
            "total_cells": self.total_cells,
            "dead_end_cells": self.dead_end_cells,
            "branching_factor": round(self.branching_factor, 4),
            "connectivity": round(self.connectivity, 4),
            "is_valid": self.is_valid,
            "validation_reasons": list(self.validation_reasons),
            "difficulty_estimate": round(self.difficulty_estimate, 4),
            "generation_score": round(self.generation_score, 4),
        }


class EnvironmentAnalyzer:
    """Stateless: every method is a pure function of a generated layout /
    report, independent of any agent or training run (Section 7)."""

    def characterize(
        self,
        config: GridWorldConfig,
        report: GenerationReport,
        diversity_score: float = 1.0,
    ) -> EnvironmentMetrics:
        layout: GridWorldLayout = report.layout
        validation = report.validation
        total_cells = layout.grid_size * layout.grid_size

        reachable = reachable_cells(layout)
        dead_end_cells = dead_end_zone_cells(layout, reachable)
        actual_obstacle_density = len(layout.obstacles) / total_cells if total_cells else 0.0
        connectivity = len(reachable) / total_cells if total_cells else 0.0

        # Branching factor: average number of free neighbors per reachable
        # cell -- a corridor-only layout is close to 2.0, a highly branched
        # layout (many alternate routes) is higher.
        branching_factor = self._average_branching(layout, reachable)

        difficulty = config.difficulty_estimate()
        generation_score = self._generation_score(
            validity=1.0 if validation.is_valid else 0.0,
            diversity=diversity_score,
            difficulty=difficulty,
        )

        return EnvironmentMetrics(
            grid_size=layout.grid_size,
            obstacle_density_actual=actual_obstacle_density,
            dead_end_probability_config=config.dead_end_probability,
            path_complexity_config=config.path_complexity,
            path_length=validation.shortest_path_length,
            reachable_cells=len(reachable),
            total_cells=total_cells,
            dead_end_cells=len(dead_end_cells),
            branching_factor=branching_factor,
            connectivity=connectivity,
            is_valid=validation.is_valid,
            validation_reasons=validation.reasons,
            difficulty_estimate=difficulty,
            generation_score=generation_score,
        )

    @staticmethod
    def _average_branching(layout: GridWorldLayout, reachable: set[tuple[int, int]]) -> float:
        if not reachable:
            return 0.0
        moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
        total_degree = 0
        for pos in reachable:
            total_degree += sum(
                1 for dx, dy in moves if (pos[0] + dx, pos[1] + dy) in reachable
            )
        return total_degree / len(reachable)

    @staticmethod
    def _generation_score(validity: float, diversity: float, difficulty: float) -> float:
        """A simple, auditable combination -- not a claim of optimality
        (documented per the project's research-mode rule, same as
        CurriculumController's scoring): a candidate must be valid first,
        then diversity and a moderate difficulty (neither trivial nor
        unsolvable-feeling) are rewarded equally.
        """
        if validity <= 0.0:
            return 0.0
        moderate_difficulty_bonus = 1.0 - abs(difficulty - 0.5) * 2  # peaks at difficulty=0.5
        return _clamp(0.5 * diversity + 0.5 * _clamp(moderate_difficulty_bonus))
