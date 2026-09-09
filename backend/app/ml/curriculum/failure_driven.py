"""Phase 13 -- the full adaptive loop (Section 20), wiring together every
previous phase into the proposed method end to end:

    train -> collect trajectories -> analyze failures -> weakness vector
    -> generate candidates -> score -> select -> train -> measure LP -> repeat

Design: step 0 has no trained policy and therefore no trajectories to
analyze, so it trains on `base_config` directly with `target_weakness=None`
(an explicit warm-up step, not a hidden special case). From step 1 onward,
each step's environment is chosen by scoring candidates generated from the
weakness vector measured *after* the previous step's training -- i.e. the
loop always acts on the most recently observed weakness, exactly as
Section 20 specifies.

Trajectory collection for failure analysis (step 2 of the loop) happens
right after training, using several fresh same-parameter, different-seed
variants of the just-trained-on environment (not the exact single seeded
layout again) -- this gives the failure analyzer more than one trajectory
to work from per step while still reflecting "what kind of environment did
the agent just train on," matching Section 10/Phase 3's original recording
approach.

This runner is intentionally a distinct code path from Phases 6-8's
`CurriculumRunner` (see runner.py's docstring) since the failure-driven
method needs trajectories and weakness vectors that the simpler baselines
never touch.

`progress_callback` (Phase 18): optional plain callable, `dict -> None`,
invoked per training episode and once more per curriculum step -- see
runner.py's docstring for the same pattern; this module stays free of any
web/broadcast import for the same "keep the ML engine independent of the
web application" reason.

`use_failure_analysis` (Phase 22): when False, candidate targeting uses a
content-blind weakness vector (see `_blind_weakness_vector`) instead of the
real one -- this is how ablation B ("learning progress without detailed
failure analysis") is implemented without a separate code path. The real
weakness vector is still computed and recorded either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.ml.analysis.failure_analyzer import FailureAnalyzer
from app.ml.analysis.trajectory_logger import TrajectoryLogger, TrajectoryStore
from app.ml.analysis.weakness_vector import WEAKNESS_DIMENSIONS, WeaknessVector, WeaknessVectorBuilder
from app.ml.curriculum.base import CurriculumStep
from app.ml.curriculum.controller import CurriculumController
from app.ml.environments.candidate_generator import CandidateGenerator
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator
from app.ml.environments.validator import EnvironmentValidator
from app.ml.evaluation.learning_progress import LearningProgressEvaluator, LearningProgressRecord
from app.ml.training.env_factory import build_env
from app.ml.training.metrics import TrainingMetricsCallback
from app.ml.training.trainer import AgentTrainer

DEFAULT_TRAJECTORIES_PER_ANALYSIS = 8
TRAJECTORY_SEED_OFFSET = 5_000
ROLLING_WINDOW = 20


def _blind_weakness_vector(episodes_analyzed: int = 0) -> WeaknessVector:
    """Phase 22 (Section 22, ablation B): a weakness vector that carries no
    real diagnostic signal -- every dimension is set to the same value, so
    candidate targeting (which reads `primary_weakness`/`secondary_weaknesses`)
    cycles through all six weaknesses in a fixed, content-blind rotation
    rather than being informed by actually-detected severity. Used only for
    the *targeting* decision in ablation B ("learning progress without
    detailed failure analysis") -- the real weakness vector from actual
    failure analysis is still computed and recorded every step regardless,
    so the ablation study can compare "what we blindly targeted" against
    "what was actually wrong."
    """
    values = {dim: 0.5 for dim in WEAKNESS_DIMENSIONS}
    return WeaknessVector(
        **values,
        primary_weakness=WEAKNESS_DIMENSIONS[0],
        secondary_weaknesses=tuple(WEAKNESS_DIMENSIONS[1:]),
        episodes_analyzed=episodes_analyzed,
    )


@dataclass
class AdaptiveLoopResult:
    history: list[CurriculumStep] = field(default_factory=list)
    weakness_history: list[WeaknessVector] = field(default_factory=list)
    learning_progress_history: list[LearningProgressRecord] = field(default_factory=list)
    failure_reports_history: list[list[dict]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "history": [s.to_dict() for s in self.history],
            "weakness_history": [w.to_dict() for w in self.weakness_history],
            "learning_progress_history": [lp.to_dict() for lp in self.learning_progress_history],
            "failure_reports_history": self.failure_reports_history,
        }


class AdaptiveCurriculumRunner:
    name = "failure_driven"

    def __init__(
        self,
        trainer: AgentTrainer,
        base_config: GridWorldConfig,
        timesteps_per_step: int = 10_000,
        eval_episodes: int = 20,
        n_candidates: int = 5,
        trajectories_per_analysis: int = DEFAULT_TRAJECTORIES_PER_ANALYSIS,
        validator: EnvironmentValidator | None = None,
        generator: GridWorldGenerator | None = None,
        trajectory_logger: TrajectoryLogger | None = None,
        trajectory_store: TrajectoryStore | None = None,
        failure_analyzer: FailureAnalyzer | None = None,
        weakness_builder: WeaknessVectorBuilder | None = None,
        candidate_generator: CandidateGenerator | None = None,
        controller: CurriculumController | None = None,
        lp_evaluator: LearningProgressEvaluator | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        use_failure_analysis: bool = True,
    ) -> None:
        self.trainer = trainer
        self.base_config = base_config
        self.timesteps_per_step = timesteps_per_step
        self.eval_episodes = eval_episodes
        self.n_candidates = n_candidates
        self.trajectories_per_analysis = trajectories_per_analysis
        self.progress_callback = progress_callback
        self.use_failure_analysis = use_failure_analysis

        self.validator = validator or EnvironmentValidator()
        self.generator = generator or GridWorldGenerator(self.validator)
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()
        self.trajectory_store = trajectory_store or TrajectoryStore()
        self.failure_analyzer = failure_analyzer or FailureAnalyzer(self.validator)
        self.weakness_builder = weakness_builder or WeaknessVectorBuilder()
        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.controller = controller or CurriculumController()
        self.lp_evaluator = lp_evaluator or LearningProgressEvaluator()

    def run(self, n_steps: int) -> AdaptiveLoopResult:
        result = AdaptiveLoopResult()
        weakness: WeaknessVector | None = None

        for i in range(n_steps):
            environment_id = f"env_{i:04d}"

            if i == 0 or weakness is None:
                config = self.base_config
                target_weakness = None
            else:
                weakness_for_targeting = weakness if self.use_failure_analysis else _blind_weakness_vector(
                    weakness.episodes_analyzed
                )
                candidates = self.candidate_generator.generate(
                    weakness_for_targeting,
                    self.base_config,
                    previous_configs=[s.config for s in result.history],
                    n_candidates=self.n_candidates,
                )
                recent_success_rate = result.history[-1].success_rate
                ranked = self.controller.select(
                    candidates,
                    weakness_for_targeting,
                    lp_history=result.learning_progress_history,
                    recent_success_rate=recent_success_rate,
                )
                best = ranked[0]
                config = best.candidate.config
                target_weakness = best.candidate.target_weakness

            performance_before = self.lp_evaluator.evaluate_performance(
                self.trainer, config, self.eval_episodes, environment_id=environment_id
            )

            current_primary_weakness = weakness.primary_weakness if weakness is not None else None

            def on_episode(
                callback: TrainingMetricsCallback,
                environment_id=environment_id,
                current_primary_weakness=current_primary_weakness,
            ) -> None:
                if self.progress_callback is None:
                    return
                latest = callback.episodes[-1]
                self.progress_callback(
                    {
                        "episode": latest.training_episode,
                        "reward": latest.episode_reward,
                        "success_rate": callback.success_rate(last_n=ROLLING_WINDOW),
                        "environment_id": environment_id,
                        "primary_weakness": current_primary_weakness,
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
            performance_after = evaluation.success_rate  # matches default metric="success_rate"
            lp_record = self.lp_evaluator.compute(
                environment_id, target_weakness, performance_before, performance_after
            )
            result.learning_progress_history.append(lp_record)

            step = CurriculumStep(
                step_index=i,
                environment_id=environment_id,
                config=config,
                difficulty=config.difficulty_estimate(),
                success_rate=evaluation.success_rate,
                average_reward=evaluation.average_reward,
                target_weakness=target_weakness,
                learning_progress=lp_record.learning_progress,
                average_steps=evaluation.average_steps,
                collision_rate=evaluation.collision_rate,
            )
            result.history.append(step)

            weakness, failure_reports = self._analyze_post_training_weakness(config, environment_id)
            result.weakness_history.append(weakness)
            result.failure_reports_history.append([r.to_dict() for r in failure_reports])

            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "episode": None,
                        "reward": evaluation.average_reward,
                        "success_rate": evaluation.success_rate,
                        "environment_id": environment_id,
                        "primary_weakness": weakness.primary_weakness,
                        "learning_progress": lp_record.learning_progress,
                        "training_status": "step_completed",
                    }
                )

        return result

    # -- helpers --------------------------------------------------------------

    def _analyze_post_training_weakness(
        self, config: GridWorldConfig, environment_id: str
    ):
        policy_fn = lambda obs: int(self.trainer.model.predict(obs, deterministic=True)[0])
        failure_reports = []

        for j in range(self.trajectories_per_analysis):
            variant_seed = (config.seed or 0) + TRAJECTORY_SEED_OFFSET + j
            variant_config = GridWorldConfig(**{**config.to_dict(), "seed": variant_seed})
            generation = self.generator.generate(variant_config)

            env = build_env(variant_config, environment_id=environment_id)
            trajectory = self.trajectory_logger.run_episode(
                env,
                policy_fn,
                seed=variant_seed,
                episode_id=f"{environment_id}_traj_{j:02d}",
                environment_id=environment_id,
            )
            summary = self.trajectory_store.add(trajectory)
            failure_reports.extend(
                self.failure_analyzer.analyze(
                    trajectory, summary, generation.layout, generation.validation
                )
            )

        weakness = self.weakness_builder.build(failure_reports)
        return weakness, failure_reports
