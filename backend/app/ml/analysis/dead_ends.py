"""Identifies which reachable free cells belong to a "dead-end zone" —
branches of the map that terminate rather than looping back into the main
route between start and goal.

Problem solved: the validator's dead-end *count* (validator.py) only counts
immediate degree-1 cells. Failure analysis needs the *full* zone (an entire
spur corridor, however long) so it can tell whether a trajectory wandered
into and out of a dead end, not just whether it touched the exact tip.

Approach: iterative leaf-pruning ("peeling"). Repeatedly remove degree-1
cells (excluding start/goal) from the reachable free-cell graph; whatever
gets removed is, by construction, not part of any cycle and not on any
route between two non-removed cells — i.e. a dead-end zone. Cells that sit
on an alternate loop (created by path_complexity branching, Phase 1) never
reach degree 1 and are correctly left out.

This is equivalent to computing the graph's 2-core relative to {start,
goal} and is O(V) for a grid graph — cheap enough to run per analyzed
trajectory.
"""

from __future__ import annotations

from collections import deque

from app.ml.environments.validator import GridWorldLayout, Position

_MOVES: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def reachable_cells(layout: GridWorldLayout) -> set[Position]:
    """All free cells reachable from agent_start (ignores whether the goal
    itself is reachable — used purely as the universe of cells to analyze)."""
    start = layout.agent_start
    visited: set[Position] = {start}
    queue: deque[Position] = deque([start])
    while queue:
        pos = queue.popleft()
        for dx, dy in _MOVES:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in visited and layout.is_free(nxt):
                visited.add(nxt)
                queue.append(nxt)
    return visited


def dead_end_zone_cells(
    layout: GridWorldLayout, reachable: set[Position] | None = None
) -> frozenset[Position]:
    reachable = reachable if reachable is not None else reachable_cells(layout)
    protected = {layout.agent_start, layout.goal}

    neighbors: dict[Position, set[Position]] = {
        pos: {
            (pos[0] + dx, pos[1] + dy)
            for dx, dy in _MOVES
            if (pos[0] + dx, pos[1] + dy) in reachable
        }
        for pos in reachable
    }
    degree = {pos: len(neighbors[pos]) for pos in reachable}

    queue: deque[Position] = deque(
        pos for pos in reachable if degree[pos] == 1 and pos not in protected
    )
    removed: set[Position] = set()

    while queue:
        cell = queue.popleft()
        if cell in removed or cell in protected:
            continue
        removed.add(cell)
        for neighbor in neighbors[cell]:
            if neighbor in removed:
                continue
            degree[neighbor] -= 1
            if degree[neighbor] == 1 and neighbor not in protected:
                queue.append(neighbor)

    return frozenset(removed)
