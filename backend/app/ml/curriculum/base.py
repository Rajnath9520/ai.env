"""Shared types every curriculum strategy (random, fixed, generic-adaptive,
failure-driven) and the runner that drives them are built around.

Keeping this interface identical across Phases 6-9 is what makes the later
comparison (Section 27/33) fair: every strategy is judged by the same
runner, calling the same AgentTrainer, producing the same CurriculumStep
records — only `next_environment()`'s internals differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.ml.environments.config import GridWorldConfig


@dataclass(frozen=True)
class CurriculumStep:
    step_index: int
    environment_id: str
    config: GridWorldConfig
    difficulty: float
    success_rate: float
    average_reward: float
    target_weakness: str | None = None
    learning_progress: float | None = None
    average_steps: float | None = None
    collision_rate: float | None = None

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "environment_id": self.environment_id,
            "config": self.config.to_dict(),
            "difficulty": round(self.difficulty, 4),
            "success_rate": round(self.success_rate, 4),
            "average_reward": round(self.average_reward, 4),
            "target_weakness": self.target_weakness,
            "learning_progress": (
                round(self.learning_progress, 4) if self.learning_progress is not None else None
            ),
            "average_steps": round(self.average_steps, 4) if self.average_steps is not None else None,
            "collision_rate": round(self.collision_rate, 4) if self.collision_rate is not None else None,
        }


class CurriculumStrategy(Protocol):
    """Every curriculum strategy implements exactly this: given the history
    of steps so far, decide the next environment's configuration. Nothing
    about training, evaluation, or storage belongs here — that's the
    runner's job — so a strategy can be tested in complete isolation from
    PPO/SB3.
    """

    name: str

    def next_environment(self, history: list[CurriculumStep]) -> GridWorldConfig: ...
