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


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
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
