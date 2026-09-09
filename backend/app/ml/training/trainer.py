"""PPO agent training/evaluation, wrapping Stable-Baselines3.

Problem solved: give every curriculum strategy (Phases 6-9) and the
experiment manager (Phase 15) one consistent way to train, checkpoint, and
evaluate an agent against a GridWorldConfig, without any of those callers
touching SB3 directly.

Why PPO / SB3: Section 4 and Section 9 specify PPO via Stable-Baselines3 as
the starting algorithm — a well-tested, on-policy algorithm that handles
this environment's small discrete action space comfortably, so the research
question can be about environment synthesis, not RL-algorithm tuning.

Assumptions: single-environment (non-vectorized) training for Phase 2,
since the curriculum swaps environments frequently (one config per
curriculum step) — vectorized multi-env training would blur which
environment produced which episode's failures. This can be revisited if
sample efficiency becomes a bottleneck later.

Limitations: no hyperparameter search is performed; SB3 defaults plus a
handful of commonly-adjusted PPO knobs are used as a reasonable starting
point, documented in `DEFAULT_PPO_KWARGS` below rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO

from app.ml.environments.config import GridWorldConfig
from app.ml.training.env_factory import build_env
from app.ml.training.metrics import EpisodeMetrics, TrainingMetricsCallback

DEFAULT_PPO_KWARGS: dict = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # slightly favors exploration in a sparse-reward gridworld
    "verbose": 0,
}


@dataclass
class EvaluationResult:
    environment_id: str | None
    episodes: int
    success_rate: float
    average_reward: float
    average_steps: float
    collision_rate: float
    episode_rewards: list[float] = field(default_factory=list)
    episode_successes: list[bool] = field(default_factory=list)


@dataclass
class TrainingResult:
    total_timesteps: int
    episodes: list[EpisodeMetrics]
    final_success_rate: float
    final_average_reward: float


class AgentTrainer:
    """Owns exactly one PPO policy across its lifetime. Callers that need
    to train across a sequence of different environments (i.e. every
    curriculum strategy) call `train()` repeatedly with different configs —
    the underlying policy persists between calls, which is what makes the
    curriculum "adaptive" rather than "retrain from scratch every step".
    """

    def __init__(
        self,
        ppo_kwargs: dict | None = None,
        seed: int | None = None,
    ) -> None:
        self.ppo_kwargs = {**DEFAULT_PPO_KWARGS, **(ppo_kwargs or {})}
        self.seed = seed
        self.model: PPO | None = None

    # -- lifecycle --------------------------------------------------------

    def _ensure_model(self, env) -> PPO:
        if self.model is None:
            self.model = PPO(env=env, seed=self.seed, **self.ppo_kwargs)
        else:
            self.model.set_env(env)
        return self.model

    def train(
        self,
        config: GridWorldConfig,
        total_timesteps: int,
        environment_id: str | None = None,
        on_episode: "Callable[[TrainingMetricsCallback], None] | None" = None,
    ) -> TrainingResult:
        env = build_env(config, environment_id=environment_id)
        model = self._ensure_model(env)

        callback = TrainingMetricsCallback(on_episode=on_episode)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )

        return TrainingResult(
            total_timesteps=total_timesteps,
            episodes=callback.episodes,
            final_success_rate=callback.success_rate(last_n=20),
            final_average_reward=callback.average_reward(last_n=20),
        )

    def evaluate(
        self,
        config: GridWorldConfig,
        episodes: int = 20,
        deterministic: bool = True,
        environment_id: str | None = None,
        eval_seed_offset: int = 10_000,
    ) -> EvaluationResult:
        """Evaluate the current policy. Uses a seed range disjoint from any
        seed likely used for training (offset by `eval_seed_offset`) so
        evaluation environments aren't accidentally identical to recently
        trained-on ones, even when the caller reuses the same base seed.
        """
        if self.model is None:
            raise RuntimeError("evaluate() called before any train() call")

        env = build_env(config, environment_id=environment_id)
        rewards: list[float] = []
        successes: list[bool] = []
        steps: list[int] = []
        collisions: list[int] = []

        base_seed = (config.seed or 0) + eval_seed_offset
        for i in range(episodes):
            obs, info = env.reset(seed=base_seed + i)
            terminated = truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_reward += reward
            rewards.append(episode_reward)
            successes.append(bool(info.get("success", False)))
            steps.append(int(info.get("episode", {}).get("l", info.get("steps_taken", 0))))
            collisions.append(int(info.get("collision_count", 0)))

        return EvaluationResult(
            environment_id=environment_id,
            episodes=episodes,
            success_rate=float(np.mean(successes)) if successes else 0.0,
            average_reward=float(np.mean(rewards)) if rewards else 0.0,
            average_steps=float(np.mean(steps)) if steps else 0.0,
            collision_rate=float(np.mean([c > 0 for c in collisions])) if collisions else 0.0,
            episode_rewards=rewards,
            episode_successes=successes,
        )

    # -- checkpointing ------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("save_checkpoint() called before any train() call")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load_checkpoint(self, path: str | Path, config: GridWorldConfig) -> None:
        """SB3 needs an env to infer observation/action spaces on load, so
        a representative config is required — it does not need to be the
        exact config the checkpoint was trained on."""
        env = build_env(config)
        self.model = PPO.load(str(path), env=env)
