"""Runs single episodes step-by-step and records full trajectories, plus a
bounded store that keeps raw trajectories only up to a cap while retaining
every summary (Section 10: "Do not store enormous unnecessary data
permanently... Support raw trajectory / summarized trajectory statistics.").
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Callable, Protocol

import numpy as np

from app.ml.analysis.trajectory import Trajectory, TrajectoryStep, TrajectorySummary, summarize

PolicyFn = Callable[[np.ndarray], int]


class EpisodicEnv(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...


class TrajectoryLogger:
    """Stateless recorder: each call to `run_episode` produces one
    independent `Trajectory` by driving the given env with the given
    policy function. Kept independent of AgentTrainer/SB3 so it can also
    replay/record trajectories for saved checkpoints, scripted policies,
    or unit tests without touching PPO at all.
    """

    def run_episode(
        self,
        env: EpisodicEnv,
        policy_fn: PolicyFn,
        seed: int | None = None,
        episode_id: str | None = None,
        environment_id: str | None = None,
        max_steps: int | None = None,
    ) -> Trajectory:
        obs, info = env.reset(seed=seed)
        goal_pos = info["goal_pos"]
        agent_pos = info["agent_pos"]

        steps: list[TrajectoryStep] = []
        step_number = 0

        while True:
            action = int(policy_fn(obs))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_agent_pos = info["agent_pos"]

            steps.append(
                TrajectoryStep(
                    step_number=step_number,
                    state=obs,
                    action=action,
                    reward=float(reward),
                    next_state=next_obs,
                    done=done,
                    timestamp=time.time(),
                    agent_pos=agent_pos,
                    next_agent_pos=next_agent_pos,
                    collided=bool(info.get("collided", False)),
                )
            )

            obs = next_obs
            agent_pos = next_agent_pos
            step_number += 1

            if done or (max_steps is not None and step_number >= max_steps):
                break

        return Trajectory(
            episode_id=episode_id or str(uuid.uuid4()),
            environment_id=environment_id,
            goal_pos=goal_pos,
            steps=steps,
        )


class TrajectoryStore:
    """In-memory trajectory storage with a hard cap on raw trajectories
    retained (oldest evicted first). Summaries are cheap (a handful of
    numbers) and are always retained in full, since downstream consumers
    (failure analysis, weakness vectors, curriculum decisions, the
    dashboard) rely on summaries far more than raw step data — raw data is
    mainly useful for the Live Training grid replay of the *current*
    episode.
    """

    def __init__(self, max_raw_trajectories: int = 200):
        if max_raw_trajectories < 0:
            raise ValueError("max_raw_trajectories must be >= 0")
        self.max_raw_trajectories = max_raw_trajectories
        self._raw: "OrderedDict[str, Trajectory]" = OrderedDict()
        self._summaries: "OrderedDict[str, TrajectorySummary]" = OrderedDict()

    def add(self, trajectory: Trajectory) -> TrajectorySummary:
        summary = summarize(trajectory)
        self._summaries[trajectory.episode_id] = summary

        if self.max_raw_trajectories > 0:
            self._raw[trajectory.episode_id] = trajectory
            while len(self._raw) > self.max_raw_trajectories:
                self._raw.popitem(last=False)

        return summary

    def get_raw(self, episode_id: str) -> Trajectory | None:
        return self._raw.get(episode_id)

    def get_summary(self, episode_id: str) -> TrajectorySummary | None:
        return self._summaries.get(episode_id)

    def all_summaries(self) -> list[TrajectorySummary]:
        return list(self._summaries.values())

    def raw_count(self) -> int:
        return len(self._raw)

    def summary_count(self) -> int:
        return len(self._summaries)
