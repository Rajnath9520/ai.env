"""Layout representation and validity checking for GridWorld environments.

The generator (generator.py) must never hand a GridWorldEnv an unsolvable
layout (Section 8: "The environment generator must never produce invalid
environments."). This module is the single source of truth for what
"valid" means, so both the generator and its tests use the same definition.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

Position = tuple[int, int]

_MOVES: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right


@dataclass(frozen=True)
class GridWorldLayout:
    """A concrete, fully-specified environment instance."""

    grid_size: int
    agent_start: Position
    goal: Position
    obstacles: frozenset[Position]

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_free(self, pos: Position) -> bool:
        return self.in_bounds(pos) and pos not in self.obstacles


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reasons: tuple[str, ...]
    shortest_path_length: int | None
    dead_end_count: int
    reachable_cell_count: int


class EnvironmentValidator:
    """Deterministic, side-effect-free validity checks for a layout."""

    def validate(self, layout: GridWorldLayout) -> ValidationResult:
        reasons: list[str] = []

        if not layout.in_bounds(layout.agent_start):
            reasons.append("agent_start out of bounds")
        if not layout.in_bounds(layout.goal):
            reasons.append("goal out of bounds")
        if layout.agent_start in layout.obstacles:
            reasons.append("agent_start overlaps an obstacle")
        if layout.goal in layout.obstacles:
            reasons.append("goal overlaps an obstacle")
        if layout.agent_start == layout.goal:
            reasons.append("agent_start equals goal")

        shortest_path_length: int | None = None
        reachable_cell_count = 0
        dead_end_count = 0

        if not reasons:
            shortest_path_length, reachable = self._bfs(layout)
            reachable_cell_count = len(reachable)
            dead_end_count = self._count_dead_ends(layout, reachable)
            if shortest_path_length is None:
                reasons.append("goal is not reachable from agent_start")

        return ValidationResult(
            is_valid=not reasons,
            reasons=tuple(reasons),
            shortest_path_length=shortest_path_length,
            dead_end_count=dead_end_count,
            reachable_cell_count=reachable_cell_count,
        )

    def _bfs(self, layout: GridWorldLayout) -> tuple[int | None, set[Position]]:
        start, goal = layout.agent_start, layout.goal
        visited: set[Position] = {start}
        queue: deque[tuple[Position, int]] = deque([(start, 0)])
        distance_to_goal: int | None = None

        while queue:
            pos, dist = queue.popleft()
            if pos == goal:
                distance_to_goal = dist
                break
            for dx, dy in _MOVES:
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt not in visited and layout.is_free(nxt):
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))

        return distance_to_goal, visited

    def _count_dead_ends(self, layout: GridWorldLayout, reachable: set[Position]) -> int:
        """A dead end is a reachable, non-goal, non-start free cell with
        exactly one free neighbor (i.e. a corridor terminus)."""
        count = 0
        for pos in reachable:
            if pos in (layout.agent_start, layout.goal):
                continue
            free_neighbors = sum(
                1
                for dx, dy in _MOVES
                if layout.is_free((pos[0] + dx, pos[1] + dy))
            )
            if free_neighbors == 1:
                count += 1
        return count
