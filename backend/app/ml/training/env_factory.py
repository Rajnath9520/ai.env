"""Builds the concrete Gymnasium env SB3 trains against: GridWorldEnv
wrapped with episode-stats tracking and SB3's Monitor (for reward/length
bookkeeping SB3's own tooling expects).
"""

from __future__ import annotations

from stable_baselines3.common.monitor import Monitor

from app.ml.environments.config import GridWorldConfig
from app.ml.environments.gridworld import GridWorldEnv
from app.ml.training.wrappers import GridWorldEpisodeStatsWrapper


def build_env(config: GridWorldConfig, environment_id: str | None = None) -> Monitor:
    env = GridWorldEnv(config=config)
    env = GridWorldEpisodeStatsWrapper(env, environment_id=environment_id)
    # info_keywords must be listed explicitly for Monitor to forward them
    # through to the callback via the episode-end info dict.
    return Monitor(env, info_keywords=("success", "collision_count", "steps_to_goal", "environment_id"))
