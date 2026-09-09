"""Gymnasium-compatible 2D GridWorld environment.

This is the environment referenced everywhere else in the pipeline
(trajectory logging, failure analysis, curriculum generation). It is
intentionally minimal for Phase 1: four discrete moves, a dense-ish
distance-shaped reward, and a step limit — matching Section 8 exactly.
"""

from __future__ import annotations

import enum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GenerationReport, GridWorldGenerator
from app.ml.environments.validator import EnvironmentValidator, Position

REWARD_GOAL = 1.0
REWARD_COLLISION = -0.05
REWARD_STEP = -0.01


class Action(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


_ACTION_DELTAS: dict[Action, Position] = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


class GridWorldEnv(gym.Env):
    """A single-agent, fully-observable 2D grid navigation task."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: GridWorldConfig | None = None,
        render_mode: str | None = None,
        generator: GridWorldGenerator | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GridWorldConfig()
        self.render_mode = render_mode
        self._generator = generator or GridWorldGenerator(EnvironmentValidator())

        n = self.config.grid_size
        self.action_space = spaces.Discrete(len(Action))
        # Observation: [agent_x, agent_y, goal_x, goal_y] normalized to
        # [0, 1], plus a local 3x3 obstacle occupancy patch centered on the
        # agent (flattened) — enough for the agent to both navigate toward
        # the goal and react to nearby obstacles without needing the full
        # grid, which would not scale to larger grid sizes later.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4 + 9,), dtype=np.float32
        )

        self._layout = None
        self._generation_report: GenerationReport | None = None
        self._agent_pos: Position = (0, 0)
        self._steps_taken = 0
        self._visited: set[Position] = set()

    # -- Gymnasium API --------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        config = self.config
        if seed is not None:
            config = GridWorldConfig(**{**config.to_dict(), "seed": seed})
        report = self._generator.generate(config)

        self._layout = report.layout
        self._generation_report = report
        self._agent_pos = report.layout.agent_start
        self._steps_taken = 0
        self._visited = {self._agent_pos}

        observation = self._build_observation()
        info = self._build_info(collided=False, reached_goal=False)
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._layout is None:
            raise RuntimeError("step() called before reset()")

        action_enum = Action(int(action))
        dx, dy = _ACTION_DELTAS[action_enum]
        candidate = (self._agent_pos[0] + dx, self._agent_pos[1] + dy)

        collided = not self._layout.is_free(candidate)
        if not collided:
            self._agent_pos = candidate
        self._visited.add(self._agent_pos)
        self._steps_taken += 1

        reached_goal = self._agent_pos == self._layout.goal
        terminated = reached_goal
        truncated = self._steps_taken >= self.config.max_steps and not terminated

        reward = REWARD_STEP
        if collided:
            reward += REWARD_COLLISION
        if reached_goal:
            reward += REWARD_GOAL

        observation = self._build_observation()
        info = self._build_info(collided=collided, reached_goal=reached_goal)
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._layout is None:
            return None
        n = self._layout.grid_size
        rows = []
        for y in range(n):
            row = []
            for x in range(n):
                pos = (x, y)
                if pos == self._agent_pos:
                    row.append("A")
                elif pos == self._layout.goal:
                    row.append("G")
                elif pos in self._layout.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            rows.append("".join(row))
        rendered = "\n".join(rows)
        if self.render_mode == "human":
            print(rendered)
            return None
        return rendered

    # -- helpers --------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        layout = self._layout
        n = layout.grid_size
        ax, ay = self._agent_pos
        gx, gy = layout.goal

        local_patch = np.zeros(9, dtype=np.float32)
        for i, (dx, dy) in enumerate(
            [(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)]
        ):
            neighbor = (ax + dx, ay + dy)
            local_patch[i] = 0.0 if layout.is_free(neighbor) else 1.0

        obs = np.array(
            [ax / (n - 1), ay / (n - 1), gx / (n - 1), gy / (n - 1)],
            dtype=np.float32,
        )
        return np.concatenate([obs, local_patch]).astype(np.float32)

    def _build_info(self, *, collided: bool, reached_goal: bool) -> dict[str, Any]:
        layout = self._layout
        return {
            "agent_pos": self._agent_pos,
            "goal_pos": layout.goal if layout else None,
            "steps_taken": self._steps_taken,
            "collided": collided,
            "reached_goal": reached_goal,
            "unique_states_visited": len(self._visited),
            "generation_attempts_used": (
                self._generation_report.attempts_used if self._generation_report else None
            ),
        }
