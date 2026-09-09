"""Phase 17 -- Pydantic request/response schemas (Section 24).

These are the only place FastAPI routes touch data shape; routes never
construct dicts by hand and never import ORM models directly into a
response.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CurriculumType = Literal["random", "fixed", "generic_adaptive", "failure_driven"]

# Phase 28 (Section 6) -- the three explicit agent modes. Default is
# "environment_only": creating an experiment must never implicitly create or
# require an AI model (Section 2).
AgentMode = Literal["environment_only", "external", "demo_ppo"]
GenerationStrategy = Literal["diverse", "progressive", "random"]


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    agent_mode: AgentMode = "environment_only"
    curriculum_type: CurriculumType = "failure_driven"
    seed: int = 42

    grid_size: int = Field(default=10, ge=4, le=24)
    obstacle_density: float = Field(default=0.15, ge=0.0, le=0.9)
    dead_end_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    path_complexity: float = Field(default=0.15, ge=0.0, le=1.0)
    max_steps_per_episode: int = Field(default=60, ge=1)

    n_curriculum_steps: int = Field(default=20, ge=1)
    timesteps_per_step: int = Field(default=10_000, ge=1)
    eval_episodes: int = Field(default=20, ge=1)

    n_candidates: int = Field(default=5, ge=1)
    trajectories_per_analysis: int = Field(default=8, ge=1)

    generation_strategy: GenerationStrategy = "diverse"

    run_generalization: bool = True
    generalization_envs_per_split: int = Field(default=10, ge=1)
    generalization_episodes_per_config: int = Field(default=5, ge=1)


class ExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    agent_mode: AgentMode
    curriculum_type: str
    seed: int
    status: str
    created_at: datetime


class ExperimentStatusRead(BaseModel):
    id: int
    status: str
    message: str | None = None


class EnvironmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    environment_key: str
    params: dict
    difficulty: float
    target_weakness: str | None
    split: str


class EpisodeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    training_run_id: int
    episode_number: int
    reward: float
    steps: int
    success: bool
    collisions: int


class TrajectoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    episode_id: int
    trajectory_length: int
    unique_states_visited: int
    repeated_states: int
    direction_reversals: int
    distance_to_goal: int


class FailureRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    episode_id: int
    failure_type: str
    score: float
    confidence: float
    features: dict


class WeaknessRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    training_run_id: int
    obstacle_avoidance: float
    exploration: float
    dead_end_recovery: float
    path_efficiency: float
    short_horizon_planning: float
    long_horizon_planning: float
    primary_weakness: str
    secondary_weaknesses: list


class LearningProgressRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    environment_id: int
    metric: str
    performance_before: float
    performance_after: float
    learning_progress: float


class GeneralizationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    split: str
    success_rate: float
    average_reward: float
    average_steps: float
    collision_rate: float


class EnvironmentLayoutRead(BaseModel):
    """Phase 20 -- the concrete, regenerated maze for one environment.
    Not stored in the DB (Phase 16 only persists generative parameters);
    regenerated on demand since the generator is a deterministic function
    of (grid_size, seed, density knobs) -- see the route docstring.
    """

    environment_id: int
    environment_key: str
    grid_size: int
    agent_start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: list[tuple[int, int]]


class MetricsRead(BaseModel):
    experiment_id: int
    agent_mode: AgentMode
    status: str
    n_environments_trained: int
    latest_success_rate: float | None
    latest_difficulty: float | None
    latest_primary_weakness: str | None
    latest_learning_progress: float | None


class WeaknessVectorRead(BaseModel):
    """Phase 28 -- the raw `WeaknessVector.to_dict()` shape (no DB id /
    training_run_id yet, unlike `WeaknessRead` which mirrors a persisted
    `AgentWeakness` row). Used for external-agent responses where a
    weakness has just been computed and may or may not have been persisted
    yet within the same response.
    """

    obstacle_avoidance: float
    exploration: float
    dead_end_recovery: float
    path_efficiency: float
    short_horizon_planning: float
    long_horizon_planning: float
    primary_weakness: str
    secondary_weaknesses: list[str]
    episodes_analyzed: int


class GeneratedEnvironmentRead(BaseModel):
    """Phase 28 (Section 13) -- one environment produced by environment-only
    generation. Deliberately has no reward/episode/success-rate fields:
    those describe agent behavior, which does not exist in this mode.
    """

    step_index: int
    environment_id: str
    grid_size: int
    obstacle_density: float
    dead_end_probability: float
    path_complexity: float
    difficulty: float
    generation_score: float
    generation_reason: str
    metrics: dict
    validation: dict


class AgentStateRead(BaseModel):
    """Phase 28 (Section 16) -- the honest, current state of the
    external-agent connection for one experiment. `connected=False` must
    always be reflected as "Waiting for connection" in the UI, never as
    training/interacting.
    """

    experiment_id: int
    agent_mode: AgentMode
    connected: bool
    status: str
    environment_id: str | None = None
    episode: int | None = None
    step: int | None = None
    last_reward: float | None = None
    last_terminated: bool | None = None
    last_truncated: bool | None = None
    trajectories_recorded: int = 0
    latest_weakness: WeaknessVectorRead | None = None


class AgentResetResponse(BaseModel):
    experiment_id: int
    environment_id: str
    episode: int
    observation: list[float]
    info: dict


class AgentStepRequest(BaseModel):
    action: int = Field(ge=0, le=3)


class AgentStepResponse(BaseModel):
    experiment_id: int
    environment_id: str
    episode: int
    step: int
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    # Populated only once an episode ends and enough trajectories have been
    # analyzed to produce a fresh weakness vector (Section 10/16) -- None
    # otherwise, never a fabricated placeholder.
    episode_summary: dict | None = None
    weakness: WeaknessVectorRead | None = None
    next_environment: GeneratedEnvironmentRead | None = None


class ComparisonEntry(BaseModel):
    experiment_id: int
    name: str
    agent_mode: AgentMode
    curriculum_type: str
    status: str
    final_success_rate: float | None
    generalization: dict[str, GeneralizationRead] = {}
