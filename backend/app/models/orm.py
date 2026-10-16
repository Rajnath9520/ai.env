"""Phase 16 -- Database schema (Section 23), as SQLAlchemy 2.0 ORM models.

Targets PostgreSQL specifically (per Section 4's technology decision, not
a generic SQL-portable schema) -- JSONB columns are used deliberately for
flexible/derived data (environment params, failure features, secondary
weaknesses) rather than approximated with a portable JSON type.

Design notes:
- Raw per-step trajectory data is intentionally NOT a table here. Section
  10/54 both say not to store enormous unnecessary data permanently; only
  the *summarized* trajectory statistics (Phase 3's TrajectorySummary)
  have a table (`trajectories`). Full step-by-step logs stay in the
  bounded in-memory TrajectoryStore (Phase 3) for the current episode's
  live-replay use case, not in Postgres.
- `environment_parameters` exists per Section 23's table list, but is a
  secondary/normalized view -- `environments.params` (JSONB) is the
  primary, always-populated source of truth, since a normalized key/value
  table is rarely queried directly and mainly useful for ad-hoc parameter
  analytics later.
- Cascades: deleting an experiment cascades to everything that only makes
  sense in that experiment's context (environments, training runs,
  evaluations, and transitively episodes/failures/etc.) so there's no way
  to end up with orphaned rows pointing at a deleted experiment.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Agent(Base):
    """Section 5 -- a registered Local Agent: a separate process
    (`agent_app/main.py`, typically on the user's own machine) that ai.env
    talks to over HTTP instead of touching that filesystem or running
    commands itself. `token` is the shared bearer secret that agent was
    started with (`AI_ENV_AGENT_TOKEN`). It has to be stored recoverably,
    since the backend must present it on every call -- so it sits in the
    database in plaintext, protected only by whatever protects that
    database, and is never returned by any API schema (`has_token` is all
    the API exposes). `status` reflects the last check
    (`POST /api/agents/{id}/ping`), not a live connection: `connected`,
    `unreachable`, or `unauthorized` (reachable, but the token is missing
    or wrong)."""

    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    base_url: Mapped[str] = mapped_column(String(500))
    token: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # connected | unreachable | unauthorized | unknown
    status: Mapped[str] = mapped_column(String(20), default="unknown")
    last_seen_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    projects: Mapped[list["Project"]] = relationship(back_populates="agent")

    @property
    def has_token(self) -> bool:
        return bool(self.token)


class Project(Base):
    """Mode B (master prompt Section 4/7/8) -- a registered, user-owned AI
    project. Deliberately separate from `Experiment`: a `Project` is a
    durable pointer at *the user's own code* (their `root_path`, scanned
    and reconfirmed over time), while an `Experiment` is one ai.env-owned
    run of it. GridWorld/PPO experiments have no `Project` row at all
    (`experiments.project_id` stays NULL) -- they are the "built-in"
    source per Section 46's project-card example, not a registered one.

    `detected_config` is the last raw `ProjectScanner` result (Section 7) --
    always suggestions, never trusted directly. `config_yaml` is only
    populated once the user has confirmed/edited that suggestion into a
    complete `AIEnvConfig` (Section 8); routes must never let
    `detected_config` substitute for it.
    """

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    root_path: Mapped[str] = mapped_column(String(1000))
    # "local" (Local Agent-scanned path) | "upload" (Section 4's ZIP
    # alternative) | "built_in" (reserved; GridWorld itself has no row).
    source: Mapped[str] = mapped_column(String(20), default="local")
    # NULL -- executed directly by the backend process (Section 61's
    # "Initial implementation: LOCAL execution", i.e. project and backend
    # share a filesystem). Set -- every scan/validate/train/evaluate for
    # this project is instead proxied to that registered Agent over HTTP
    # (`app/services/agent_client.py`), which is what actually lets
    # `root_path` refer to a path on a *different* machine than the
    # backend (Section 5).
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    language: Mapped[str | None] = mapped_column(String(30), nullable=True)
    framework: Mapped[str | None] = mapped_column(String(50), nullable=True)
    task: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # draft -> scanned -> configured -> validated | validation_failed
    status: Mapped[str] = mapped_column(String(30), default="draft")
    capabilities: Mapped[list] = mapped_column(JSONB, default=list)
    detected_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    config_yaml: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    experiments: Mapped[list["Experiment"]] = relationship(back_populates="project")
    runs: Mapped[list["ProjectRun"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="ProjectRun.run_number"
    )
    agent: Mapped["Agent | None"] = relationship(back_populates="projects")


class ProjectRun(Base):
    """Mode B -- one train(+evaluate) cycle for a `Project`, executed by
    `ProjectService`'s background worker via `CommandAdapter` (Section 19/
    Rule 11: never inline in the request). Deliberately its own table
    rather than reusing `TrainingRun`/`Evaluation` -- those are shaped
    around GridWorld/PPO specifically (hardcoded `success_rate`,
    `average_reward`, etc., which Section 23 explicitly says not to
    impose on every task), so treating them as "equivalent existing
    models" (Section 16) would mean forcing irrelevant columns onto a
    face-recognition or object-detection run. `train_result`/
    `evaluation_result` instead store `TrainResult.to_dict()`/
    `EvaluationResult.to_dict()` directly -- arbitrary per-task metrics
    live in `evaluation_result["metrics"]` as JSONB (Section 23/58).

    `run_number` is per-project and starts at 1 (Section 18: "Never
    silently overwrite previous versions"); `managed_artifact_path` is the
    ai.env-archived copy from `archive_run_artifact`, distinct from
    `train_result["artifact"]["path"]`, which is the live path in the
    user's own project that the *next* run's script may overwrite.
    """

    __tablename__ = "project_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    run_number: Mapped[int]
    # queued -> running_training -> running_evaluation -> completed | failed | cancelled
    status: Mapped[str] = mapped_column(String(30), default="queued")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)
    # Set only when this run executed via a registered Agent (Section 5)
    # rather than directly by the backend -- the id *that agent* uses for
    # this run, needed to poll/cancel/download from it. Meaningless
    # without also knowing `project.agent_id`, which is why this lives on
    # the run (agents can change) rather than being assumed globally.
    agent_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    train_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    evaluation_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    managed_artifact_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="runs")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Mode B -- NULL for every built-in GridWorld/PPO experiment (all of
    # them, until a future phase lets those be registered as a Project
    # too). Never required, never backfilled: existing rows are valid as
    # NULL and stay that way.
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255))
    config_yaml: Mapped[str] = mapped_column(Text)
    curriculum_type: Mapped[str] = mapped_column(String(50))
    # Phase 28 (Section 6/21) -- the explicit switch between environment-only
    # (default), external-agent, and demo-PPO modes. Persisted directly
    # (not left implicit inside config_yaml only) so it's trivially queryable
    # and so old rows can be given an honest default via migration/backfill.
    agent_mode: Mapped[str] = mapped_column(String(20), default="environment_only")
    seed: Mapped[int]
    # Widened from 20 -> 32 (Section 18) to comfortably fit the larger set of
    # mode-aware status values (e.g. "analyzing_trajectory",
    # "waiting_for_agent") alongside the original training-oriented ones.
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    project: Mapped["Project | None"] = relationship(back_populates="experiments")
    environments: Mapped[list["Environment"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class Environment(Base):
    __tablename__ = "environments"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    environment_key: Mapped[str] = mapped_column(String(64))  # app-level id, e.g. "env_0004"
    params: Mapped[dict] = mapped_column(JSONB)
    difficulty: Mapped[float]
    target_weakness: Mapped[str | None] = mapped_column(String(50), nullable=True)
    split: Mapped[str] = mapped_column(String(20), default="train")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    experiment: Mapped["Experiment"] = relationship(back_populates="environments")
    parameters: Mapped[list["EnvironmentParameter"]] = relationship(
        back_populates="environment", cascade="all, delete-orphan"
    )
    training_runs: Mapped[list["TrainingRun"]] = relationship(back_populates="environment")
    learning_progress_records: Mapped[list["LearningProgress"]] = relationship(
        back_populates="environment", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("experiment_id", "environment_key"),)


class EnvironmentParameter(Base):
    """Secondary, normalized key/value view of an environment's params --
    see module docstring. `environments.params` (JSONB) is authoritative."""

    __tablename__ = "environment_parameters"

    id: Mapped[int] = mapped_column(primary_key=True)
    environment_id: Mapped[int] = mapped_column(ForeignKey("environments.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(String(255))

    environment: Mapped["Environment"] = relationship(back_populates="parameters")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    environment_id: Mapped[int | None] = mapped_column(
        ForeignKey("environments.id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)

    experiment: Mapped["Experiment"] = relationship(back_populates="training_runs")
    environment: Mapped["Environment | None"] = relationship(back_populates="training_runs")
    episodes: Mapped[list["Episode"]] = relationship(
        back_populates="training_run", cascade="all, delete-orphan"
    )
    weaknesses: Mapped[list["AgentWeakness"]] = relationship(
        back_populates="training_run", cascade="all, delete-orphan"
    )
    checkpoints: Mapped[list["ModelCheckpoint"]] = relationship(
        back_populates="training_run", cascade="all, delete-orphan"
    )


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"))
    episode_number: Mapped[int]
    reward: Mapped[float]
    steps: Mapped[int]
    success: Mapped[bool]
    collisions: Mapped[int] = mapped_column(default=0)

    training_run: Mapped["TrainingRun"] = relationship(back_populates="episodes")
    trajectory: Mapped["Trajectory | None"] = relationship(
        back_populates="episode", uselist=False, cascade="all, delete-orphan"
    )
    failures: Mapped[list["Failure"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("training_run_id", "episode_number"),)


class Trajectory(Base):
    """Summarized trajectory statistics only (Phase 3's TrajectorySummary)
    -- deliberately not raw per-step data; see module docstring."""

    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_id: Mapped[int] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"), unique=True
    )
    trajectory_length: Mapped[int]
    unique_states_visited: Mapped[int]
    repeated_states: Mapped[int]
    direction_reversals: Mapped[int]
    distance_to_goal: Mapped[int]

    episode: Mapped["Episode"] = relationship(back_populates="trajectory")


class Failure(Base):
    __tablename__ = "failures"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"))
    failure_type: Mapped[str] = mapped_column(String(50))
    score: Mapped[float]
    confidence: Mapped[float]
    features: Mapped[dict] = mapped_column(JSONB)

    episode: Mapped["Episode"] = relationship(back_populates="failures")


class AgentWeakness(Base):
    __tablename__ = "agent_weaknesses"

    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"))
    obstacle_avoidance: Mapped[float]
    exploration: Mapped[float]
    dead_end_recovery: Mapped[float]
    path_efficiency: Mapped[float]
    short_horizon_planning: Mapped[float]
    long_horizon_planning: Mapped[float]
    primary_weakness: Mapped[str] = mapped_column(String(50))
    secondary_weaknesses: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    training_run: Mapped["TrainingRun"] = relationship(back_populates="weaknesses")


class LearningProgress(Base):
    __tablename__ = "learning_progress"

    id: Mapped[int] = mapped_column(primary_key=True)
    environment_id: Mapped[int] = mapped_column(ForeignKey("environments.id", ondelete="CASCADE"))
    metric: Mapped[str] = mapped_column(String(30), default="success_rate")
    performance_before: Mapped[float]
    performance_after: Mapped[float]
    learning_progress: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    environment: Mapped["Environment"] = relationship(back_populates="learning_progress_records")


class Evaluation(Base):
    """Generalization evaluation results (Phase 14) per split."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    split: Mapped[str] = mapped_column(String(20))
    success_rate: Mapped[float]
    average_reward: Mapped[float]
    average_steps: Mapped[float]
    collision_rate: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    experiment: Mapped["Experiment"] = relationship(back_populates="evaluations")


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"))
    path: Mapped[str] = mapped_column(String(500))
    episode: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    training_run: Mapped["TrainingRun"] = relationship(back_populates="checkpoints")
