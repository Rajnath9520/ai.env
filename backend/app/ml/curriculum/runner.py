"""Drives any CurriculumStrategy (random, fixed, generic-adaptive — and,
once built, failure-driven) through a fixed number of train+evaluate steps
against one continuously-improving AgentTrainer, recording one
CurriculumStep per iteration.

Kept deliberately simple for Phases 6-8: strategies here only need the
success-rate/reward history the runner records, not full trajectories or
failure analysis. The failure-driven method (Phase 9+) needs richer
per-step inputs (weakness vectors) and will be driven by the more elaborate
AdaptiveCurriculumRunner built in Phase 13 — this runner is not meant to be
stretched to cover that case, to keep both loops easy to reason about
independently.

`progress_callback` (Phase 18): an optional plain callable, `dict -> None`,
invoked once per training episode with a small summarized tick (episode
number, reward, rolling success rate, environment id, training status), and
once more per curriculum step when that step finishes. Deliberately just a
callable — this module has no import of FastAPI, WebSockets, or any
broadcast infrastructure, keeping the ML engine independent of the web
application (Phase 0's architecture decision). The web layer turns this
callable into something that publishes over a websocket.
"""

from __future__ import annotations

from typing import Callable

from app.ml.curriculum.base import CurriculumStep, CurriculumStrategy
from app.ml.training.metrics import TrainingMetricsCallback
from app.ml.training.trainer import AgentTrainer

ROLLING_WINDOW = 20


class CurriculumRunner:
    def __init__(
        self,
        trainer: AgentTrainer,
        strategy: CurriculumStrategy,
        timesteps_per_step: int = 5_000,
        eval_episodes: int = 20,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self.trainer = trainer
        self.strategy = strategy
        self.timesteps_per_step = timesteps_per_step
        self.eval_episodes = eval_episodes
        self.progress_callback = progress_callback

    def run(self, n_steps: int) -> list[CurriculumStep]:
        history: list[CurriculumStep] = []

        for i in range(n_steps):
            config = self.strategy.next_environment(history)
            environment_id = f"{self.strategy.name}_env_{i:04d}"

            def on_episode(callback: TrainingMetricsCallback, environment_id=environment_id) -> None:
                if self.progress_callback is None:
                    return
                latest = callback.episodes[-1]
                self.progress_callback(
                    {
                        "episode": latest.training_episode,
                        "reward": latest.episode_reward,
                        "success_rate": callback.success_rate(last_n=ROLLING_WINDOW),
                        "environment_id": environment_id,
                        "primary_weakness": None,
                        "learning_progress": None,
                        "training_status": "training",
                    }
                )

            self.trainer.train(
                config=config,
                total_timesteps=self.timesteps_per_step,
                environment_id=environment_id,
                on_episode=on_episode,
            )
            evaluation = self.trainer.evaluate(
                config=config,
                episodes=self.eval_episodes,
                deterministic=True,
                environment_id=environment_id,
            )

            step = CurriculumStep(
                step_index=i,
                environment_id=environment_id,
                config=config,
                difficulty=config.difficulty_estimate(),
                success_rate=evaluation.success_rate,
                average_reward=evaluation.average_reward,
                average_steps=evaluation.average_steps,
                collision_rate=evaluation.collision_rate,
            )
            history.append(step)

            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "episode": None,
                        "reward": evaluation.average_reward,
                        "success_rate": evaluation.success_rate,
                        "environment_id": environment_id,
                        "primary_weakness": None,
                        "learning_progress": None,
                        "training_status": "step_completed",
                    }
                )

        return history
