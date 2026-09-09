"""Trajectory data model and derived statistics.

Problem solved: Phase 4's failure analyzer needs per-episode behavioral
signals (dead-end entries, repeated states, direction reversals, path
efficiency, ...) that aren't visible in a scalar reward. This module is the
single place that turns a raw step-by-step trajectory into both (a) the raw
record and (b) a compact, storable summary of those signals.

Design: `TrajectoryStep` mirrors exactly what Section 10 asks for (state,
action, reward, next_state, done, timestamp, step_number). `summarize()` is
a pure function over a `Trajectory` so it can be unit-tested against
hand-built synthetic trajectories independent of the environment or agent —
important since these derived signals are what failure detection (Phase 4)
will be built on, and their correctness has to be verifiable directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

Position = tuple[int, int]


@dataclass(frozen=True)
class TrajectoryStep:
    step_number: int
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float
    agent_pos: Position
    next_agent_pos: Position
    collided: bool


@dataclass
class Trajectory:
    episode_id: str
    environment_id: str | None
    goal_pos: Position
    steps: list[TrajectoryStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class TrajectorySummary:
    episode_id: str
    environment_id: str | None
    trajectory_length: int
    total_reward: float
    success: bool
    collision_count: int
    unique_states_visited: int
    distance_to_goal: int
    repeated_states: int
    direction_reversals: int


# Actions that are direct opposites of one another (UP/DOWN, LEFT/RIGHT),
# matching app.ml.environments.gridworld.Action's ordering (0..3).
_OPPOSITE_ACTIONS: dict[int, int] = {0: 1, 1: 0, 2: 3, 3: 2}


def summarize(trajectory: Trajectory) -> TrajectorySummary:
    steps = trajectory.steps
    if not steps:
        return TrajectorySummary(
            episode_id=trajectory.episode_id,
            environment_id=trajectory.environment_id,
            trajectory_length=0,
            total_reward=0.0,
            success=False,
            collision_count=0,
            unique_states_visited=0,
            distance_to_goal=_manhattan(
                trajectory.goal_pos, trajectory.goal_pos
            ),
            repeated_states=0,
            direction_reversals=0,
        )

    visited_positions: list[Position] = [steps[0].agent_pos] + [
        s.next_agent_pos for s in steps
    ]
    unique_positions = set(visited_positions)

    seen: set[Position] = set()
    repeated_states = 0
    for pos in visited_positions:
        if pos in seen:
            repeated_states += 1
        seen.add(pos)

    direction_reversals = 0
    for prev_step, curr_step in zip(steps, steps[1:]):
        if _OPPOSITE_ACTIONS.get(prev_step.action) == curr_step.action:
            direction_reversals += 1

    final_pos = steps[-1].next_agent_pos
    total_reward = float(sum(s.reward for s in steps))
    collision_count = sum(1 for s in steps if s.collided)
    success = bool(steps[-1].done and final_pos == trajectory.goal_pos)

    return TrajectorySummary(
        episode_id=trajectory.episode_id,
        environment_id=trajectory.environment_id,
        trajectory_length=len(steps),
        total_reward=total_reward,
        success=success,
        collision_count=collision_count,
        unique_states_visited=len(unique_positions),
        distance_to_goal=_manhattan(final_pos, trajectory.goal_pos),
        repeated_states=repeated_states,
        direction_reversals=direction_reversals,
    )


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
