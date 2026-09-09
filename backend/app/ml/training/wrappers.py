"""Gymnasium wrapper that accumulates per-episode statistics GridWorldEnv
only exposes per-step (collisions, whether the goal was reached, steps
taken), and attaches them to `info` at episode end so both the SB3 callback
and later the trajectory logger (Phase 3) can read one consistent summary.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class GridWorldEpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, environment_id: str | None = None):
        super().__init__(env)
        self.environment_id = environment_id
        self._collision_count = 0

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        self._collision_count = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("collided"):
            self._collision_count += 1

        if terminated or truncated:
            info = {
                **info,
                "success": bool(info.get("reached_goal", False)),
                "collision_count": self._collision_count,
                "steps_to_goal": info.get("steps_taken") if info.get("reached_goal") else None,
                "environment_id": self.environment_id,
            }
        return obs, reward, terminated, truncated, info
