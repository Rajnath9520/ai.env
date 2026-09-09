"""Procedural GridWorld generation.

Design choice (documented per the project's "research mode" rule):

Problem solved: turn a GridWorldConfig (density/complexity knobs) into a
concrete, always-solvable layout.

Why needed: Phase 1 requires "the environment generator must never produce
invalid environments" — a reject-and-retry approach (sample randomly,
validate, retry) can never *guarantee* this and can loop for a long time at
high obstacle densities.

Approach: guarantee-by-construction. A connecting path between start and
goal is carved first and marked "protected" (always free); only the
remaining cells are subject to random obstacle placement. This makes every
generated layout solvable by construction, and the validator (validator.py)
is then used as a cheap confirmation check + diagnostic source, not as the
primary correctness mechanism.

Alternative approaches considered: (a) reject-and-retry random placement —
simple but not guaranteed to terminate and wasteful at high density; (b) a
full randomized-Prim's/Kruskal's maze over the whole grid — produces very
uniform maze structure and makes obstacle_density hard to control
independently of path structure. The carve-then-scatter approach keeps the
knobs closer to their names at the cost of not being a "true" uniform maze.

Limitations: branch placement (for path_complexity / dead_end_probability)
is a local greedy random walk, not a global optimizer, so extreme parameter
combinations (e.g. very high path_complexity on a very small grid) may
produce fewer branches than requested simply because there is no room. This
is surfaced via the returned GenerationReport rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.environments.config import GridWorldConfig
from app.ml.environments.validator import (
    EnvironmentValidator,
    GridWorldLayout,
    Position,
    ValidationResult,
)

_MOVES: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class GenerationReport:
    """Diagnostics about a single generation, independent of the layout
    itself — useful for the frontend's "Environment Parameters" panel and
    for the environment synthesizer's diversity scoring (Phase 10)."""

    layout: GridWorldLayout
    validation: ValidationResult
    branches_requested: int
    branches_placed: int
    attempts_used: int


class GenerationError(RuntimeError):
    """Raised only if construction produces a layout the validator still
    rejects — treated as a bug in the generator, not an expected outcome,
    since the whole point of this module is to make that impossible."""


class GridWorldGenerator:
    def __init__(self, validator: EnvironmentValidator | None = None) -> None:
        self._validator = validator or EnvironmentValidator()

    def generate(self, config: GridWorldConfig) -> GenerationReport:
        rng = np.random.default_rng(config.seed)

        for attempt in range(1, config.max_generation_attempts + 1):
            start, goal = self._sample_start_goal(config, rng)
            protected = self._carve_primary_path(start, goal, config.grid_size, rng)

            branches_requested, branches_placed, protected = self._carve_branches(
                protected, config, rng
            )

            obstacles = self._scatter_obstacles(protected, config, rng)

            layout = GridWorldLayout(
                grid_size=config.grid_size,
                agent_start=start,
                goal=goal,
                obstacles=frozenset(obstacles),
            )
            validation = self._validator.validate(layout)

            if validation.is_valid:
                return GenerationReport(
                    layout=layout,
                    validation=validation,
                    branches_requested=branches_requested,
                    branches_placed=branches_placed,
                    attempts_used=attempt,
                )
            # Construction should make this unreachable; if it happens,
            # retry with a fresh random draw rather than silently accepting
            # an invalid environment.

        raise GenerationError(
            f"Failed to construct a valid GridWorld after "
            f"{config.max_generation_attempts} attempts with config={config.to_dict()}"
        )

    # -- start / goal placement -------------------------------------------------

    def _sample_start_goal(
        self, config: GridWorldConfig, rng: np.random.Generator
    ) -> tuple[Position, Position]:
        n = config.grid_size
        all_cells = [(x, y) for x in range(n) for y in range(n)]

        start = tuple(all_cells[rng.integers(len(all_cells))])

        if config.goal_distance is None:
            candidates = [c for c in all_cells if c != start]
        else:
            candidates = [
                c
                for c in all_cells
                if c != start
                and abs(c[0] - start[0]) + abs(c[1] - start[1]) >= config.goal_distance
            ]
            if not candidates:
                # Requested distance too large for this grid — fall back to
                # the pair of cells with maximum achievable distance rather
                # than silently ignoring the request.
                candidates = sorted(
                    (c for c in all_cells if c != start),
                    key=lambda c: abs(c[0] - start[0]) + abs(c[1] - start[1]),
                )[-1:]

        goal = tuple(candidates[rng.integers(len(candidates))])
        return start, goal

    # -- primary connecting path --------------------------------------------

    def _carve_primary_path(
        self,
        start: Position,
        goal: Position,
        grid_size: int,
        rng: np.random.Generator,
        bias: float = 0.75,
        max_iterations_factor: int = 6,
    ) -> set[Position]:
        """Biased random walk toward the goal; falls back to a deterministic
        L-shaped path if the walk doesn't converge in a reasonable number of
        steps, so a connecting path always exists."""
        path: set[Position] = {start}
        current = start
        max_iterations = grid_size * grid_size * max_iterations_factor

        for _ in range(max_iterations):
            if current == goal:
                return path

            dx = goal[0] - current[0]
            dy = goal[1] - current[1]
            moves_toward = [
                (mx, my)
                for mx, my in _MOVES
                if (mx != 0 and (mx > 0) == (dx > 0) and dx != 0)
                or (my != 0 and (my > 0) == (dy > 0) and dy != 0)
            ]
            moves_away = [m for m in _MOVES if m not in moves_toward]

            use_toward = moves_toward and rng.random() < bias
            pool = moves_toward if use_toward else (moves_away or list(_MOVES))
            mx, my = pool[rng.integers(len(pool))]
            nxt = (current[0] + mx, current[1] + my)

            if 0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size:
                current = nxt
                path.add(current)

        if current != goal:
            # Deterministic fallback: guarantees connectivity regardless of
            # how the random walk went.
            path |= self._l_shaped_path(current, goal)
            path |= self._l_shaped_path(start, goal)

        return path

    @staticmethod
    def _l_shaped_path(a: Position, b: Position) -> set[Position]:
        cells: set[Position] = set()
        x, y = a
        step_x = 1 if b[0] >= x else -1
        while x != b[0]:
            cells.add((x, y))
            x += step_x
        step_y = 1 if b[1] >= y else -1
        while y != b[1]:
            cells.add((x, y))
            y += step_y
        cells.add(b)
        return cells

    # -- branches (path_complexity / dead_end_probability) ------------------

    def _carve_branches(
        self,
        protected: set[Position],
        config: GridWorldConfig,
        rng: np.random.Generator,
    ) -> tuple[int, int, set[Position]]:
        n = config.grid_size
        protected = set(protected)
        path_cells = list(protected)

        # Requested branch counts scale with grid size and the two
        # complexity knobs; capped so tiny grids don't request more
        # branches than they can physically host.
        max_possible = max(1, len(path_cells) // 2)
        loop_branches_requested = round(config.path_complexity * max_possible * 0.5)
        dead_end_branches_requested = round(config.dead_end_probability * max_possible * 0.5)
        branches_requested = loop_branches_requested + dead_end_branches_requested
        branches_placed = 0

        for _ in range(loop_branches_requested):
            if self._carve_one_branch(protected, path_cells, n, rng, reconnect=True):
                branches_placed += 1

        for _ in range(dead_end_branches_requested):
            if self._carve_one_branch(protected, path_cells, n, rng, reconnect=False):
                branches_placed += 1

        return branches_requested, branches_placed, protected

    def _carve_one_branch(
        self,
        protected: set[Position],
        path_cells: list[Position],
        grid_size: int,
        rng: np.random.Generator,
        reconnect: bool,
        min_len: int = 2,
        max_len: int = 5,
    ) -> bool:
        origin = path_cells[rng.integers(len(path_cells))]
        length = int(rng.integers(min_len, max_len + 1))
        current = origin
        branch_cells: list[Position] = []

        for _ in range(length):
            options = [
                (current[0] + dx, current[1] + dy)
                for dx, dy in _MOVES
                if 0 <= current[0] + dx < grid_size and 0 <= current[1] + dy < grid_size
            ]
            options = [c for c in options if c not in protected and c not in branch_cells]
            if not options:
                break
            current = options[rng.integers(len(options))]
            branch_cells.append(current)

        if len(branch_cells) < min_len:
            return False

        protected.update(branch_cells)
        if reconnect:
            # Try to connect the branch's tail back into the path so it
            # forms an alternate route rather than a dead end.
            tail = branch_cells[-1]
            for dx, dy in _MOVES:
                candidate = (tail[0] + dx, tail[1] + dy)
                if candidate in protected and candidate != branch_cells[-2 if len(branch_cells) > 1 else 0]:
                    return True
            # No reconnection found — it still adds free space, just not a
            # guaranteed loop; counted as placed either way since it was
            # successfully carved.
        return True

    # -- obstacle scattering --------------------------------------------------

    def _scatter_obstacles(
        self,
        protected: set[Position],
        config: GridWorldConfig,
        rng: np.random.Generator,
    ) -> set[Position]:
        n = config.grid_size
        all_cells = [(x, y) for x in range(n) for y in range(n)]
        free_candidates = [c for c in all_cells if c not in protected]
        obstacle_mask = rng.random(len(free_candidates)) < config.obstacle_density
        return {c for c, is_obstacle in zip(free_candidates, obstacle_mask) if is_obstacle}
