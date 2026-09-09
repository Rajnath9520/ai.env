"""Phase 11 — Learning Progress (Section 18).

LP(E) = Performance_after(E) - Performance_before(E)

This is the signal the curriculum controller (Phase 12) will eventually use
alongside weakness alignment and diversity to decide which environment
actually helped, rather than just which one looked well-targeted on paper.

Design: kept as a thin, metric-pluggable wrapper around AgentTrainer's
existing evaluate()/train() rather than a new evaluation code path, so
"performance" always means exactly what AgentTrainer.evaluate() measures —
one source of truth. Section 18 explicitly scopes v1 to success rate, with
reward/path-efficiency/failure-reduction as later options; `METRICS` below
is deliberately a small, extensible registry rather than a single hardcoded
formula, so adding those later doesn't require touching this module's
control flow.

Assumption made explicit: if a policy hasn't been trained on anything yet
(AgentTrainer.model is None), performance_before is treated as 0.0 (an
untrained policy has no measured success) rather than raising — this is a
reasonable default for the very first curriculum step, not a general claim
that 0.0 is always the right baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.ml.environments.config import GridWorldConfig
from app.ml.training.trainer import AgentTrainer, EvaluationResult

MetricFn = Callable[[EvaluationResult], float]

# Registry of supported performance metrics. Section 18 scopes v1 to
# success_rate; average_reward is included since AgentTrainer.evaluate()
# already computes it cheaply. path_efficiency/failure_reduction are not
# yet wired in — they need trajectory-level data evaluate() doesn't
# currently return — and are left as a documented limitation rather than
# implemented partially.
METRICS: dict[str, MetricFn] = {
    "success_rate": lambda evaluation: evaluation.success_rate,
    "average_reward": lambda evaluation: evaluation.average_reward,
}


@dataclass(frozen=True)
class LearningProgressRecord:
    environment_id: str
    weakness_target: str | None
    metric: str
    performance_before: float
    performance_after: float
    learning_progress: float

    def to_dict(self) -> dict:
        return {
            "environment_id": self.environment_id,
            "weakness_target": self.weakness_target,
            "metric": self.metric,
            "performance_before": round(self.performance_before, 4),
            "performance_after": round(self.performance_after, 4),
            "learning_progress": round(self.learning_progress, 4),
        }


class LearningProgressEvaluator:
    def __init__(self, metric: str = "success_rate") -> None:
        if metric not in METRICS:
            raise ValueError(f"Unknown metric {metric!r}; expected one of {list(METRICS)}")
        self.metric = metric

    def evaluate_performance(
        self,
        trainer: AgentTrainer,
        config: GridWorldConfig,
        eval_episodes: int = 20,
        environment_id: str | None = None,
    ) -> float:
        if trainer.model is None:
            return 0.0  # see module docstring: untrained-policy baseline
        evaluation = trainer.evaluate(
            config=config, episodes=eval_episodes, deterministic=True, environment_id=environment_id
        )
        return METRICS[self.metric](evaluation)

    def compute(
        self,
        environment_id: str,
        weakness_target: str | None,
        performance_before: float,
        performance_after: float,
    ) -> LearningProgressRecord:
        return LearningProgressRecord(
            environment_id=environment_id,
            weakness_target=weakness_target,
            metric=self.metric,
            performance_before=performance_before,
            performance_after=performance_after,
            learning_progress=performance_after - performance_before,
        )

    def measure_around_training(
        self,
        trainer: AgentTrainer,
        config: GridWorldConfig,
        total_timesteps: int,
        environment_id: str,
        weakness_target: str | None = None,
        eval_episodes: int = 20,
    ) -> LearningProgressRecord:
        """Convenience method covering the full before -> train -> after
        flow in one call, matching Section 18's example exactly (measure,
        train, measure again, compute the delta)."""
        performance_before = self.evaluate_performance(
            trainer, config, eval_episodes, environment_id=environment_id
        )
        trainer.train(config=config, total_timesteps=total_timesteps, environment_id=environment_id)
        performance_after = self.evaluate_performance(
            trainer, config, eval_episodes, environment_id=environment_id
        )
        return self.compute(environment_id, weakness_target, performance_before, performance_after)
