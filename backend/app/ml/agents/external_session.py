

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.ml.analysis.environment_analyzer import EnvironmentAnalyzer
from app.ml.analysis.failure_analyzer import FailureAnalyzer, FailureReport
from app.ml.analysis.trajectory import Trajectory, TrajectoryStep, summarize
from app.ml.analysis.weakness_vector import WeaknessVector, WeaknessVectorBuilder
from app.ml.environments.candidate_generator import CandidateGenerator
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator
from app.ml.environments.gridworld import GridWorldEnv
from app.ml.environments.validator import EnvironmentValidator
from app.ml.experiment.config import ExperimentConfig
from app.models.orm import AgentWeakness, Environment, Episode, Experiment, Failure, TrainingRun, Trajectory as TrajectoryRow
from app.workers.broadcaster import broadcaster


class ExternalAgentError(Exception):
    pass


@dataclass
class _EpisodeBuffer:
    episode_id: str
    environment_id: str
    goal_pos: tuple[int, int]
    steps: list[TrajectoryStep] = field(default_factory=list)


@dataclass
class ExternalAgentSession:
    experiment_id: int
    config: ExperimentConfig
    seed: int | None = None

    connected: bool = False
    status: str = "waiting_for_agent"

    _env: GridWorldEnv | None = field(default=None, repr=False)
    _env_config: GridWorldConfig | None = field(default=None, repr=False)
    environment_key: str | None = None
    environment_db_id: int | None = None
    episode_index: int = 0
    step_index: int = 0
    last_reward: float | None = None
    last_terminated: bool | None = None
    last_truncated: bool | None = None

    _episode_buffer: "_EpisodeBuffer | None" = field(default=None, repr=False)
    _pending_failure_reports: list[FailureReport] = field(default_factory=list, repr=False)
    _episodes_since_analysis: int = 0
    _selected_configs: list[GridWorldConfig] = field(default_factory=list, repr=False)
    trajectories_recorded: int = 0
    latest_weakness: WeaknessVector | None = None
    training_run_id: int | None = None

    _generator: GridWorldGenerator = field(default_factory=GridWorldGenerator, repr=False)
    _validator: EnvironmentValidator = field(default_factory=EnvironmentValidator, repr=False)
    _analyzer: EnvironmentAnalyzer = field(default_factory=EnvironmentAnalyzer, repr=False)
    _failure_analyzer: FailureAnalyzer = field(default_factory=FailureAnalyzer, repr=False)
    _weakness_builder: WeaknessVectorBuilder = field(default_factory=WeaknessVectorBuilder, repr=False)
    _candidate_generator: CandidateGenerator | None = field(default=None, repr=False)
    _rng: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._candidate_generator = CandidateGenerator(seed=self.seed)

    # -- lifecycle ----------------------------------------------------------

    def base_config(self) -> GridWorldConfig:
        c = self.config
        return GridWorldConfig(
            grid_size=c.grid_size,
            obstacle_density=c.obstacle_density,
            dead_end_probability=c.dead_end_probability,
            path_complexity=c.path_complexity,
            max_steps=c.max_steps_per_episode,
            seed=c.seed,
        )

    def prepare(self, db: Session) -> None:
        """Generates the initial environment and persists it, without
        requiring an agent to be connected yet (Section 9: "prepare
        environment/agent interface"). Does not instantiate any model."""
        if self._env_config is not None:
            return
        config = self.base_config()
        self._set_environment(db, config, step_index=0, reason="Initial environment for the experiment's base configuration.")
        self._set_status(db, "waiting_for_agent")

    def reset(self, db: Session) -> tuple[np.ndarray, dict[str, Any]]:
        if self._env_config is None:
            self.prepare(db)
        if not self.connected:
            self.connected = True
            self._set_status(db, "agent_connected")
            self._publish({"type": "agent_connected", "experiment_id": self.experiment_id})

        self._env = GridWorldEnv(config=self._env_config, generator=self._generator)
        obs, info = self._env.reset()
        self.episode_index += 1
        self.step_index = 0
        self._episode_buffer = _EpisodeBuffer(
            episode_id=f"ext_ep_{self.episode_index:05d}",
            environment_id=self.environment_key or "unknown",
            goal_pos=info["goal_pos"],
        )
        self._set_status(db, "interacting")
        self._publish(
            {
                "type": "environment_reset",
                "experiment_id": self.experiment_id,
                "environment_id": self.environment_key,
                "episode": self.episode_index,
            }
        )
        self._ensure_training_run(db)
        return obs, info

    def step(self, db: Session, action: int) -> dict[str, Any]:
        if self._env is None or self._episode_buffer is None:
            raise ExternalAgentError("step() called before reset(); call reset() first")

        agent_pos_before = self._env._agent_pos  # type: ignore[attr-defined]
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.step_index += 1
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)

        self._episode_buffer.steps.append(
            TrajectoryStep(
                step_number=self.step_index - 1,
                state=np.zeros(0),  # raw states are not retained long-term (Section 10)
                action=int(action),
                reward=float(reward),
                next_state=np.zeros(0),
                done=bool(terminated or truncated),
                timestamp=0.0,
                agent_pos=agent_pos_before,
                next_agent_pos=info["agent_pos"],
                collided=bool(info.get("collided", False)),
            )
        )

        self._publish(
            {
                "type": "step_completed",
                "experiment_id": self.experiment_id,
                "environment_id": self.environment_key,
                "episode": self.episode_index,
                "step": self.step_index,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        result: dict[str, Any] = {
            "observation": obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
            "episode_summary": None,
            "weakness": None,
            "next_environment": None,
        }

        if terminated or truncated:
            episode_summary, weakness, next_env_record = self._complete_episode(db)
            result["episode_summary"] = episode_summary
            result["weakness"] = weakness
            result["next_environment"] = next_env_record
            self._set_status(db, "agent_connected")

        return result

    # -- episode completion / trajectory -> failure -> weakness -> adaptive --

    def _complete_episode(self, db: Session) -> tuple[dict, dict | None, dict | None]:
        assert self._episode_buffer is not None
        buffer = self._episode_buffer
        trajectory = Trajectory(
            episode_id=buffer.episode_id,
            environment_id=buffer.environment_id,
            goal_pos=buffer.goal_pos,
            steps=buffer.steps,
        )
        summary = summarize(trajectory)
        layout = self._env._layout  # type: ignore[attr-defined]
        validation = self._validator.validate(layout)

        self._set_status(db, "analyzing_trajectory")
        self._publish(
            {
                "type": "failure_analysis_started",
                "experiment_id": self.experiment_id,
                "episode": self.episode_index,
            }
        )
        reports = self._failure_analyzer.analyze(trajectory, summary, layout, validation)
        self._persist_episode(db, summary, reports)
        self._pending_failure_reports.extend(reports)
        self._episodes_since_analysis += 1
        self.trajectories_recorded += 1
        self._publish(
            {
                "type": "failure_analysis_completed",
                "experiment_id": self.experiment_id,
                "episode": self.episode_index,
                "failures": [r.to_dict() for r in reports],
            }
        )

        episode_summary = {
            "episode": self.episode_index,
            "trajectory_length": summary.trajectory_length,
            "total_reward": summary.total_reward,
            "success": summary.success,
            "collision_count": summary.collision_count,
            "failures": [r.to_dict() for r in reports],
        }

        weakness_dict: dict | None = None
        next_env_dict: dict | None = None
        threshold = max(1, self.config.trajectories_per_analysis)
        if self._episodes_since_analysis >= threshold:
            weakness = self._weakness_builder.build(
                self._pending_failure_reports, episodes_analyzed=self._episodes_since_analysis
            )
            self.latest_weakness = weakness
            self._persist_weakness(db, weakness)
            weakness_dict = weakness.to_dict()
            self._publish(
                {
                    "type": "weakness_detected",
                    "experiment_id": self.experiment_id,
                    "weakness": weakness_dict,
                }
            )

            self._set_status(db, "adapting")
            self._publish({"type": "adaptive_generation_started", "experiment_id": self.experiment_id})
            next_env_dict = self._adapt_environment(db, weakness)

            self._pending_failure_reports = []
            self._episodes_since_analysis = 0

        return episode_summary, weakness_dict, next_env_dict

    def _adapt_environment(self, db: Session, weakness: WeaknessVector) -> dict:
        candidates = self._candidate_generator.generate(
            weakness,
            self.base_config(),
            previous_configs=self._selected_configs,
            n_candidates=max(1, self.config.n_candidates),
        )
        best = max(candidates, key=lambda c: (c.validity_score, c.diversity_score))
        step_index = len(self._selected_configs)
        reason = (
            f"Targeting '{best.target_weakness}' (weakness score "
            f"{getattr(weakness, best.target_weakness):.3f}), diversity "
            f"{best.diversity_score:.3f} from prior environments."
        )
        self._set_environment(db, best.config, step_index=step_index, reason=reason)
        self._publish(
            {
                "type": "next_environment_ready",
                "experiment_id": self.experiment_id,
                "environment_id": self.environment_key,
                "reason": reason,
            }
        )
        record = self._environment_record_dict(step_index, best.config, reason)
        return record

    # -- persistence ----------------------------------------------------------

    def _ensure_training_run(self, db: Session) -> None:
        if self.training_run_id is not None:
            return
        run = TrainingRun(experiment_id=self.experiment_id, environment_id=self.environment_db_id)
        db.add(run)
        db.flush()
        self.training_run_id = run.id
        db.commit()

    def _persist_episode(self, db: Session, summary, reports: list[FailureReport]) -> None:
        self._ensure_training_run(db)
        episode_row = Episode(
            training_run_id=self.training_run_id,
            episode_number=self.episode_index,
            reward=summary.total_reward,
            steps=summary.trajectory_length,
            success=summary.success,
            collisions=summary.collision_count,
        )
        db.add(episode_row)
        db.flush()

        db.add(
            TrajectoryRow(
                episode_id=episode_row.id,
                trajectory_length=summary.trajectory_length,
                unique_states_visited=summary.unique_states_visited,
                repeated_states=summary.repeated_states,
                direction_reversals=summary.direction_reversals,
                distance_to_goal=summary.distance_to_goal,
            )
        )
        for report in reports:
            db.add(
                Failure(
                    episode_id=episode_row.id,
                    failure_type=report.failure_type,
                    score=report.score,
                    confidence=report.confidence,
                    features=report.features,
                )
            )
        db.commit()

    def _persist_weakness(self, db: Session, weakness: WeaknessVector) -> None:
        self._ensure_training_run(db)
        db.add(
            AgentWeakness(
                training_run_id=self.training_run_id,
                obstacle_avoidance=weakness.obstacle_avoidance,
                exploration=weakness.exploration,
                dead_end_recovery=weakness.dead_end_recovery,
                path_efficiency=weakness.path_efficiency,
                short_horizon_planning=weakness.short_horizon_planning,
                long_horizon_planning=weakness.long_horizon_planning,
                primary_weakness=weakness.primary_weakness,
                secondary_weaknesses=list(weakness.secondary_weaknesses),
            )
        )
        db.commit()

    def _set_environment(self, db: Session, config: GridWorldConfig, step_index: int, reason: str) -> None:
        report = self._generator.generate(config)
        diversity = 1.0
        if self._selected_configs:
            diversity = self._candidate_generator._diversity_score(config, self._selected_configs)
        metrics = self._analyzer.characterize(config, report, diversity_score=diversity)

        environment_key = f"external_env_{step_index:04d}"
        env_row = Environment(
            experiment_id=self.experiment_id,
            environment_key=environment_key,
            params=config.to_dict(),
            difficulty=metrics.difficulty_estimate,
            target_weakness=(self.latest_weakness.primary_weakness if self.latest_weakness else None),
            split="external",
        )
        db.add(env_row)
        db.flush()
        db.commit()

        self._env_config = config
        self.environment_key = environment_key
        self.environment_db_id = env_row.id
        self._selected_configs.append(config)
        self.training_run_id = None  # a new TrainingRun links to the new environment

    def _environment_record_dict(self, step_index: int, config: GridWorldConfig, reason: str) -> dict:
        report = self._generator.generate(config)
        metrics = self._analyzer.characterize(config, report)
        return {
            "step_index": step_index,
            "environment_id": self.environment_key,
            "grid_size": config.grid_size,
            "obstacle_density": config.obstacle_density,
            "dead_end_probability": config.dead_end_probability,
            "path_complexity": config.path_complexity,
            "difficulty": round(metrics.difficulty_estimate, 4),
            "generation_score": round(metrics.generation_score, 4),
            "generation_reason": reason,
            "metrics": metrics.to_dict(),
            "validation": {"is_valid": metrics.is_valid, "reasons": list(metrics.validation_reasons)},
        }

    def _set_status(self, db: Session, status: str) -> None:
        self.status = status
        experiment = db.get(Experiment, self.experiment_id)
        if experiment is not None:
            experiment.status = status
            db.commit()

    def _publish(self, event: dict) -> None:
        broadcaster.publish(self.experiment_id, event)

    # -- read-only state ------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "agent_mode": "external",
            "connected": self.connected,
            "status": self.status,
            "environment_id": self.environment_key,
            "episode": self.episode_index or None,
            "step": self.step_index or None,
            "last_reward": self.last_reward,
            "last_terminated": self.last_terminated,
            "last_truncated": self.last_truncated,
            "trajectories_recorded": self.trajectories_recorded,
            "latest_weakness": self.latest_weakness.to_dict() if self.latest_weakness else None,
        }


class ExternalAgentManager:
    """Process-wide registry of one `ExternalAgentSession` per experiment_id
    (same in-memory, single-process scope documented for the cancellation
    flags and broadcaster -- see module docstring)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[int, ExternalAgentSession] = {}

    def get_or_create(self, experiment_id: int, config: ExperimentConfig) -> ExternalAgentSession:
        with self._lock:
            session = self._sessions.get(experiment_id)
            if session is None:
                session = ExternalAgentSession(experiment_id=experiment_id, config=config, seed=config.seed)
                self._sessions[experiment_id] = session
            return session

    def get(self, experiment_id: int) -> ExternalAgentSession | None:
        with self._lock:
            return self._sessions.get(experiment_id)

    def remove(self, experiment_id: int) -> None:
        with self._lock:
            self._sessions.pop(experiment_id, None)


external_agent_manager = ExternalAgentManager()
