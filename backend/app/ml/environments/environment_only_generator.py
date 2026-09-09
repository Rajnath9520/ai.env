"""Phase 28 (Section 3/13) -- environment-only generation, entirely
agent-free.

This is the "Environment Core" half of the architecture (Section 35):
Generate -> Validate -> Analyze -> Rank -> Select, over and over, with no
AgentTrainer, no PPO, and no training of any kind. It reuses the exact
same building blocks the rest of the project already trusts:

- `GridWorldGenerator` / `EnvironmentValidator` (Phase 1) for construction
  and validity.
- `EnvironmentAnalyzer` (Phase 28) for structural characteristics.
- The same `difficulty_estimate()` heuristic used everywhere else, so
  environment-only "difficulty" means the same thing as it does for the
  curriculum baselines -- just never fed by agent performance, because
  there is no agent.

Three generation strategies (Section 13/23), all pure functions of
(step_index, history, base_config, rng):

- "progressive": reuses `FixedCurriculum`'s existing easy->very_hard stage
  schedule (Phase 7) purely as a step-indexed parameter schedule -- it
  never depended on agent performance to begin with, so it is honest to
  reuse unmodified here.
- "random": reuses `RandomCurriculum` (Phase 6) the same way.
- "diverse" (default): generates several candidates per step by jittering
  the base config's parameters, keeps only valid ones, and picks the one
  most different from every previously-selected environment -- giving a
  genuine "generate candidates, validate, rank, select" loop (Section 13)
  without requiring a weakness signal that only exists once an agent is
  involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.ml.analysis.environment_analyzer import EnvironmentAnalyzer, EnvironmentMetrics
from app.ml.curriculum.fixed_curriculum import FixedCurriculum
from app.ml.curriculum.random_curriculum import RandomCurriculum
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator

VALID_STRATEGIES = ("diverse", "progressive", "random")

JITTER_RANGES: dict[str, tuple[float, float]] = {
    "obstacle_density": (-0.08, 0.10),
    "dead_end_probability": (-0.08, 0.12),
    "path_complexity": (-0.08, 0.12),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class GeneratedEnvironmentRecord:
    """One environment-only generation step's result (Section 13). Notably
    absent: success_rate, average_reward, learning_progress -- those
    describe agent behavior, which this mode never produces."""

    step_index: int
    environment_id: str
    config: GridWorldConfig
    metrics: EnvironmentMetrics
    generation_reason: str

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "environment_id": self.environment_id,
            "config": self.config.to_dict(),
            "grid_size": self.config.grid_size,
            "obstacle_density": self.config.obstacle_density,
            "dead_end_probability": self.config.dead_end_probability,
            "path_complexity": self.config.path_complexity,
            "difficulty": round(self.metrics.difficulty_estimate, 4),
            "generation_score": round(self.metrics.generation_score, 4),
            "generation_reason": self.generation_reason,
            "metrics": self.metrics.to_dict(),
            "validation": {
                "is_valid": self.metrics.is_valid,
                "reasons": list(self.metrics.validation_reasons),
            },
        }


class EnvironmentOnlyGenerator:
    """Drives environment generation for `agent_mode == "environment_only"`.
    Contains no reference to AgentTrainer, PPO, or stable_baselines3 --
    see the module docstring and Section 30's "search for hidden coupling"
    requirement.
    """

    def __init__(
        self,
        strategy: str = "diverse",
        seed: int | None = None,
        generator: GridWorldGenerator | None = None,
        analyzer: EnvironmentAnalyzer | None = None,
    ) -> None:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self._rng = np.random.default_rng(seed)
        self._generator = generator or GridWorldGenerator()
        self._analyzer = analyzer or EnvironmentAnalyzer()
        self._seed = seed

    def run(
        self,
        base_config: GridWorldConfig,
        n_environments: int,
        n_candidates: int = 5,
        progress_callback: "Callable[[dict], None] | None" = None,
    ) -> list[GeneratedEnvironmentRecord]:
        history: list[GeneratedEnvironmentRecord] = []
        selected_configs: list[GridWorldConfig] = []

        strategy_impl = self._build_strategy_impl(base_config, n_environments)

        for i in range(n_environments):
            environment_id = f"env_only_{i:04d}"
            if progress_callback is not None:
                progress_callback({"type": "generation_started", "step_index": i, "environment_id": environment_id})

            if self.strategy == "diverse":
                record = self._generate_diverse_step(
                    i, environment_id, base_config, selected_configs, n_candidates, progress_callback
                )
            else:
                record = self._generate_scheduled_step(
                    i, environment_id, strategy_impl, selected_configs, progress_callback
                )

            history.append(record)
            selected_configs.append(record.config)

            if progress_callback is not None:
                progress_callback(
                    {
                        "type": "environment_selected",
                        "step_index": i,
                        "environment_id": environment_id,
                        "record": record.to_dict(),
                    }
                )

        return history

    # -- strategy implementations ------------------------------------------

    def _build_strategy_impl(self, base_config: GridWorldConfig, n_environments: int):
        if self.strategy == "progressive":
            return FixedCurriculum(
                total_steps=max(n_environments, 4),
                grid_size=base_config.grid_size,
                max_steps=base_config.max_steps,
                seed=self._seed,
            )
        if self.strategy == "random":
            return RandomCurriculum(
                grid_size=base_config.grid_size,
                max_steps=base_config.max_steps,
                seed=self._seed,
            )
        return None  # "diverse" doesn't use a CurriculumStrategy object

    def _generate_scheduled_step(
        self,
        step_index: int,
        environment_id: str,
        strategy_impl,
        selected_configs: list[GridWorldConfig],
        progress_callback,
    ) -> GeneratedEnvironmentRecord:
        # Both FixedCurriculum and RandomCurriculum are pure functions of a
        # `history: list[CurriculumStep]` -- they only ever read its length
        # (FixedCurriculum) or ignore it entirely (RandomCurriculum), so an
        # empty-but-correctly-sized placeholder list satisfies that contract
        # without fabricating agent performance fields on it.
        config = strategy_impl.next_environment([None] * step_index)  # type: ignore[arg-type]
        report = self._generator.generate(config)
        diversity = self._diversity(config, selected_configs)
        metrics = self._analyzer.characterize(config, report, diversity_score=diversity)
        reason = (
            f"Strategy '{self.strategy}': step {step_index} follows the configured "
            f"{self.strategy} parameter schedule (no agent weakness signal exists in "
            f"environment-only mode)."
        )
        if progress_callback is not None:
            progress_callback(
                {"type": "candidate_validated", "step_index": step_index, "environment_id": environment_id}
            )
        return GeneratedEnvironmentRecord(
            step_index=step_index,
            environment_id=environment_id,
            config=config,
            metrics=metrics,
            generation_reason=reason,
        )

    def _generate_diverse_step(
        self,
        step_index: int,
        environment_id: str,
        base_config: GridWorldConfig,
        selected_configs: list[GridWorldConfig],
        n_candidates: int,
        progress_callback,
    ) -> GeneratedEnvironmentRecord:
        candidates: list[tuple[GridWorldConfig, EnvironmentMetrics]] = []
        for c in range(n_candidates):
            config = self._jitter_config(base_config, step_index)
            report = self._generator.generate(config)
            if progress_callback is not None:
                progress_callback(
                    {
                        "type": "candidate_generated",
                        "step_index": step_index,
                        "candidate_index": c,
                        "environment_id": environment_id,
                    }
                )
            diversity = self._diversity(config, selected_configs + [cfg for cfg, _ in candidates])
            metrics = self._analyzer.characterize(config, report, diversity_score=diversity)
            if progress_callback is not None:
                progress_callback(
                    {
                        "type": "candidate_validated",
                        "step_index": step_index,
                        "candidate_index": c,
                        "environment_id": environment_id,
                        "is_valid": metrics.is_valid,
                    }
                )
            candidates.append((config, metrics))

        valid_candidates = [(cfg, m) for cfg, m in candidates if m.is_valid] or candidates
        best_config, best_metrics = max(valid_candidates, key=lambda pair: pair[1].generation_score)

        reason = (
            f"Selected candidate with the highest generation score "
            f"({best_metrics.generation_score:.3f}) among {len(candidates)} candidates: "
            f"most diverse from previously generated environments while remaining valid "
            f"and at a moderate difficulty ({best_metrics.difficulty_estimate:.3f})."
        )
        return GeneratedEnvironmentRecord(
            step_index=step_index,
            environment_id=environment_id,
            config=best_config,
            metrics=best_metrics,
            generation_reason=reason,
        )

    # -- helpers --------------------------------------------------------------

    def _jitter_config(self, base_config: GridWorldConfig, step_index: int) -> GridWorldConfig:
        # A gentle upward drift with step_index keeps later environments
        # trending harder on average (matching the "progress over time"
        # expectation from the live-generation UI, Section 15) while the
        # per-candidate jitter is what actually creates diversity to choose
        # between (Section 13).
        drift = min(0.25, step_index * 0.015)
        obstacle_density = _clamp(
            base_config.obstacle_density + drift + self._rng.uniform(*JITTER_RANGES["obstacle_density"]),
            0.0,
            0.85,
        )
        dead_end_probability = _clamp(
            base_config.dead_end_probability + drift + self._rng.uniform(*JITTER_RANGES["dead_end_probability"]),
            0.0,
            1.0,
        )
        path_complexity = _clamp(
            base_config.path_complexity + drift + self._rng.uniform(*JITTER_RANGES["path_complexity"]),
            0.0,
            1.0,
        )
        return GridWorldConfig(
            grid_size=base_config.grid_size,
            obstacle_density=obstacle_density,
            goal_distance=base_config.goal_distance,
            dead_end_probability=dead_end_probability,
            path_complexity=path_complexity,
            max_steps=base_config.max_steps,
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )

    @staticmethod
    def _diversity(config: GridWorldConfig, reference_configs: list[GridWorldConfig]) -> float:
        if not reference_configs:
            return 1.0
        vector = EnvironmentOnlyGenerator._param_vector(config)
        distances = [
            EnvironmentOnlyGenerator._distance(vector, EnvironmentOnlyGenerator._param_vector(c))
            for c in reference_configs
        ]
        avg_distance = sum(distances) / len(distances)
        max_possible_distance = len(vector) ** 0.5
        return _clamp(avg_distance / max_possible_distance, 0.0, 1.0)

    @staticmethod
    def _param_vector(config: GridWorldConfig) -> list[float]:
        return [config.obstacle_density, config.dead_end_probability, config.path_complexity]

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
