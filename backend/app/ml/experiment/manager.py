"""Phase 15 -- ExperimentManager (Section 22).
Phase 28 (Section 5/6/7) -- now agent_mode-aware: `run()` no longer
unconditionally builds an AgentTrainer/PPO. It reads `config.agent_mode`
first and dispatches:

- "environment_only" (default) -- generates/validates/analyzes a sequence
  of environments via `EnvironmentOnlyGenerator`. No AgentTrainer, no PPO,
  no training, no training-shaped events.
- "demo_ppo" -- exactly the original curriculum + AgentTrainer + PPO +
  generalization-evaluation flow, unchanged, but only reachable when the
  user explicitly selected Demo Agent -- PPO.
- "external" -- `run()` is not used for this mode at all; external-agent
  experiments are driven interactively through
  `app/ml/agents/external_session.py` via the reset/step HTTP endpoints.
  `ExperimentService.start()` is the layer that makes this split (see its
  docstring) so a long-running background task is never scheduled for a
  mode that has no "run to completion" concept.

Reproducibility (Section 22's explicit requirement): every sub-component
that draws randomness is seeded from `config.seed` here, explicitly --
`AgentTrainer(seed=...)` already makes PPO/SB3 training itself deterministic
(Phase 2); this manager's job is making sure the *curriculum*/generation
layer (candidate generation, environment synthesis, environment split
sampling) is equally deterministic rather than silently falling back to OS
entropy via an unseeded `None`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.ml.curriculum.base import CurriculumStep
from app.ml.curriculum.controller import CurriculumController
from app.ml.curriculum.failure_driven import AdaptiveCurriculumRunner
from app.ml.curriculum.fixed_curriculum import FixedCurriculum
from app.ml.curriculum.generic_adaptive import GenericAdaptiveCurriculum
from app.ml.curriculum.random_curriculum import RandomCurriculum
from app.ml.curriculum.runner import CurriculumRunner
from app.ml.environments.candidate_generator import CandidateGenerator
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.environment_only_generator import EnvironmentOnlyGenerator, GeneratedEnvironmentRecord
from app.ml.evaluation.generalization import EnvironmentSplitGenerator, GeneralizationEvaluator
from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.reproducibility import ManifestTimer, ReproducibilityManifest, get_git_info

logger = logging.getLogger("experiment_manager")


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    curriculum_history: list[CurriculumStep] = field(default_factory=list)
    weakness_history: list[dict] | None = None
    learning_progress_history: list[dict] | None = None
    failure_reports_history: list[list[dict]] | None = None
    generalization_results: dict[str, dict] | None = None
    reproducibility: dict | None = None
    # Phase 28 (Section 13) -- populated only for agent_mode="environment_only".
    # Deliberately separate from curriculum_history (which carries
    # success_rate/average_reward -- agent metrics that do not exist here).
    generated_environments: list[dict] | None = None

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "curriculum_history": [s.to_dict() for s in self.curriculum_history],
            "weakness_history": self.weakness_history,
            "learning_progress_history": self.learning_progress_history,
            "failure_reports_history": self.failure_reports_history,
            "generalization_results": self.generalization_results,
            "reproducibility": self.reproducibility,
            "generated_environments": self.generated_environments,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


class ExperimentManager:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self, progress_callback: Callable[[dict], None] | None = None, save_checkpoint: bool = True) -> ExperimentResult:
        config = self.config
        if config.agent_mode == "environment_only":
            return self._run_environment_only(progress_callback)
        if config.agent_mode == "external":
            raise ValueError(
                "ExperimentManager.run() does not run external-agent experiments -- "
                "they are driven interactively via app/ml/agents/external_session.py's "
                "reset()/step() through the /agent/reset and /agent/step endpoints, "
                "never as a long background task. See ExperimentService.start()."
            )
        if config.agent_mode != "demo_ppo":
            raise ValueError(f"Unhandled agent_mode: {config.agent_mode!r}")
        return self._run_demo_ppo(progress_callback, save_checkpoint)

    # -- Mode A: environment-only (Section 3/13), no agent involved ----------

    def _run_environment_only(
        self, progress_callback: Callable[[dict], None] | None = None
    ) -> ExperimentResult:
        config = self.config
        logger.info(
            "environment_generation_started name=%s strategy=%s seed=%d",
            config.name, config.generation_strategy, config.seed,
        )

        if progress_callback is not None:
            progress_callback({"type": "experiment_started", "mode": "environment_only"})

        base_config = GridWorldConfig(
            grid_size=config.grid_size,
            obstacle_density=config.obstacle_density,
            dead_end_probability=config.dead_end_probability,
            path_complexity=config.path_complexity,
            max_steps=config.max_steps_per_episode,
            seed=config.seed,
        )
        generator = EnvironmentOnlyGenerator(strategy=config.generation_strategy, seed=config.seed)
        records: list[GeneratedEnvironmentRecord] = generator.run(
            base_config,
            n_environments=config.n_curriculum_steps,
            n_candidates=config.n_candidates,
            progress_callback=progress_callback,
        )

        if progress_callback is not None:
            progress_callback({"type": "generation_completed", "n_environments": len(records)})
            progress_callback({"type": "experiment_completed", "mode": "environment_only"})

        logger.info(
            "environment_generation_completed name=%s n_environments=%d",
            config.name, len(records),
        )

        return ExperimentResult(
            config=config,
            curriculum_history=[],
            generated_environments=[r.to_dict() for r in records],
        )

    # -- Mode C: Demo Agent -- PPO (Section 4), explicit opt-in only ---------

    def _run_demo_ppo(
        self,
        progress_callback: Callable[[dict], None] | None = None,
        save_checkpoint: bool = True,
    ) -> ExperimentResult:
        # Imported lazily so that importing this module (and therefore every
        # environment_only/external code path) never pulls in
        # stable_baselines3 -- Section 30's "no hidden coupling" requirement,
        # made structural rather than just documented.
        from app.ml.training.trainer import AgentTrainer

        config = self.config
        logger.info(
            "experiment_started name=%s curriculum=%s seed=%d agent_mode=demo_ppo",
            config.name, config.curriculum_type, config.seed,
        )

        with ManifestTimer() as timer:
            base_config = GridWorldConfig(
                grid_size=config.grid_size,
                obstacle_density=config.obstacle_density,
                dead_end_probability=config.dead_end_probability,
                path_complexity=config.path_complexity,
                max_steps=config.max_steps_per_episode,
                seed=config.seed,
            )
            trainer: AgentTrainer = AgentTrainer(seed=config.seed)

            if config.curriculum_type == "failure_driven":
                curriculum_history, weakness_history, lp_history, failure_reports_history = self._run_failure_driven(
                    trainer, base_config, progress_callback
                )
            else:
                curriculum_history = self._run_baseline(trainer, base_config, progress_callback)
                weakness_history = None
                lp_history = None
                failure_reports_history = None

            generalization_results = None
            if config.run_generalization:
                generalization_results = self._run_generalization(trainer, base_config)

            checkpoint_path: str | None = None
            if save_checkpoint and trainer.model is not None:
                checkpoint_path = f"experiments/results/checkpoints/{config.name}.zip"
                try:
                    trainer.save_checkpoint(checkpoint_path)
                except Exception:
                    logger.exception("checkpoint_save_failed name=%s", config.name)
                    checkpoint_path = None

        manifest = ReproducibilityManifest(
            experiment_name=config.name,
            config=config.to_dict(),
            git_info=get_git_info(),
            started_at=timer.started_at,
            ended_at=timer.ended_at,
            checkpoint_path=checkpoint_path,
        )
        manifest.save(f"experiments/results/manifests/{config.name}.json")

        result = ExperimentResult(
            config=config,
            curriculum_history=curriculum_history,
            weakness_history=weakness_history,
            learning_progress_history=lp_history,
            failure_reports_history=failure_reports_history,
            generalization_results=generalization_results,
            reproducibility=manifest.to_dict(),
        )
        logger.info(
            "experiment_completed name=%s steps=%d duration_seconds=%.1f",
            config.name, len(curriculum_history), manifest.duration_seconds,
        )
        return result

    # -- curriculum construction ------------------------------------------

    def _run_failure_driven(
        self,
        trainer: AgentTrainer,
        base_config: GridWorldConfig,
        progress_callback: Callable[[dict], None] | None = None,
    ):
        config = self.config
        candidate_generator = CandidateGenerator(seed=config.seed)
        controller = CurriculumController()

        runner = AdaptiveCurriculumRunner(
            trainer=trainer,
            base_config=base_config,
            timesteps_per_step=config.timesteps_per_step,
            eval_episodes=config.eval_episodes,
            n_candidates=config.n_candidates,
            trajectories_per_analysis=config.trajectories_per_analysis,
            candidate_generator=candidate_generator,
            controller=controller,
            progress_callback=progress_callback,
        )
        result = runner.run(config.n_curriculum_steps)
        return (
            result.history,
            [w.to_dict() for w in result.weakness_history],
            [lp.to_dict() for lp in result.learning_progress_history],
            result.failure_reports_history,
        )

    def _run_baseline(
        self,
        trainer: AgentTrainer,
        base_config: GridWorldConfig,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> list[CurriculumStep]:
        config = self.config
        strategy = self._build_baseline_strategy(base_config)
        runner = CurriculumRunner(
            trainer=trainer,
            strategy=strategy,
            timesteps_per_step=config.timesteps_per_step,
            eval_episodes=config.eval_episodes,
            progress_callback=progress_callback,
        )
        return runner.run(config.n_curriculum_steps)

    def _build_baseline_strategy(self, base_config: GridWorldConfig):
        config = self.config
        if config.curriculum_type == "random":
            return RandomCurriculum(
                grid_size=base_config.grid_size, max_steps=base_config.max_steps, seed=config.seed
            )
        if config.curriculum_type == "fixed":
            return FixedCurriculum(
                total_steps=config.n_curriculum_steps,
                grid_size=base_config.grid_size,
                max_steps=base_config.max_steps,
                seed=config.seed,
            )
        if config.curriculum_type == "generic_adaptive":
            return GenericAdaptiveCurriculum(
                grid_size=base_config.grid_size, max_steps=base_config.max_steps, seed=config.seed
            )
        raise ValueError(f"Unhandled baseline curriculum_type: {config.curriculum_type!r}")

    # -- generalization -----------------------------------------------------

    def _run_generalization(self, trainer: AgentTrainer, base_config: GridWorldConfig) -> dict[str, dict]:
        config = self.config
        split_generator = EnvironmentSplitGenerator(seed=config.seed)
        splits = split_generator.build_all_splits(
            base_config, n_per_split=config.generalization_envs_per_split
        )
        EnvironmentSplitGenerator.assert_no_seed_overlap(splits)

        evaluator = GeneralizationEvaluator()
        results = evaluator.evaluate_all_splits(
            trainer, splits, episodes_per_config=config.generalization_episodes_per_config
        )
        return {name: r.to_dict() for name, r in results.items()}
