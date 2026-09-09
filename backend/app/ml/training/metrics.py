"""Per-episode metrics collection during training.

Problem solved: Section 9 requires tracking episode reward, success,
episode length, collisions, steps to goal, training episode number, and
environment ID for every training episode — SB3 only gives you rollout-level
aggregates by default.

Approach: a BaseCallback that inspects the `infos` dict SB3 passes on every
`_on_step`, and appends one EpisodeMetrics record whenever an episode ends
(SB3/Monitor signals this via `info["episode"]`). This keeps metrics
collection decoupled from the trainer itself, so the same callback can be
reused by every curriculum strategy later (Phases 6-9), not just the
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class EpisodeMetrics:
    training_episode: int
    environment_id: str | None
    episode_reward: float
    episode_length: int
    success: bool
    collision_count: int
    steps_to_goal: int | None


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0, on_episode: "Callable[[TrainingMetricsCallback], None] | None" = None):
        super().__init__(verbose=verbose)
        self.episodes: list[EpisodeMetrics] = []
        self._episode_counter = 0
        self._on_episode = on_episode

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            self._episode_counter += 1
            self.episodes.append(
                EpisodeMetrics(
                    training_episode=self._episode_counter,
                    environment_id=info.get("environment_id"),
                    episode_reward=float(episode_info["r"]),
                    episode_length=int(episode_info["l"]),
                    success=bool(info.get("success", False)),
                    collision_count=int(info.get("collision_count", 0)),
                    steps_to_goal=info.get("steps_to_goal"),
                )
            )
            if self._on_episode is not None:
                self._on_episode(self)
        return True

    def success_rate(self, last_n: int | None = None) -> float:
        episodes = self.episodes[-last_n:] if last_n else self.episodes
        if not episodes:
            return 0.0
        return sum(e.success for e in episodes) / len(episodes)

    def average_reward(self, last_n: int | None = None) -> float:
        episodes = self.episodes[-last_n:] if last_n else self.episodes
        if not episodes:
            return 0.0
        return sum(e.episode_reward for e in episodes) / len(episodes)
