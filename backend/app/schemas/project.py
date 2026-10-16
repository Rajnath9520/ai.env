"""Mode B -- Pydantic request/response schemas for `/api/projects/*`.
Follows `app/schemas/experiment.py`'s convention: routes only ever see
these shapes, never the ORM model or the `app.ml.project` dataclasses
directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.experiment import CurriculumType

ProjectSource = Literal["local", "upload", "built_in"]
ProjectStatus = Literal["draft", "scanned", "configured", "validated", "validation_failed"]
MetricsSource = Literal["file", "stdout"]


class ProjectRegisterRequest(BaseModel):
    """Section 4's "Add Project -> Local Project" step. Registering does
    not scan by itself (Section 47 lists Scan as a separate, explicit
    action) -- it only creates the `Project` row so its id exists for
    subsequent scan/configure/validate calls."""

    root_path: str = Field(min_length=1)
    name: str | None = None
    # Section 5 -- if set, every scan/validate/run for this project goes
    # through that registered Agent instead of the backend executing
    # directly. Can also be set/changed later via PUT /{id}/agent.
    agent_id: int | None = None


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    root_path: str
    source: ProjectSource
    agent_id: int | None
    language: str | None
    framework: str | None
    task: str | None
    status: ProjectStatus
    capabilities: list[str]
    created_at: datetime
    updated_at: datetime


class ScanResultRead(BaseModel):
    """Section 7's scan-result shape, unchanged -- every field here is a
    *suggestion*; the UI must label it "Detected / Suggested" (Section 7)
    and never treat it as already confirmed."""

    root: str
    language: str | None
    framework: str | None
    python_version: str | None
    gpu: dict
    training_candidates: list[str]
    evaluation_candidates: list[str]
    dataset_candidates: list[str]
    artifact_candidates: list[str]
    dependency_files: list[str]


class ProjectDetailRead(ProjectRead):
    detected_config: dict | None
    config_yaml: str | None
    validation_result: dict | None


class ProjectAgentAssignment(BaseModel):
    """Body for `PUT /api/projects/{id}/agent` -- `None` switches the
    project back to direct, same-machine backend execution (Section 61)."""

    agent_id: int | None


# -- ai-env.yaml contract (Section 8), as a nested request schema --------


class ProjectMetaSchema(BaseModel):
    task: str | None = None
    framework: str | None = None


class TrainingConfigSchema(BaseModel):
    command: str = Field(min_length=1)
    entrypoint: str | None = None
    working_directory: str = "."
    timeout_seconds: int = Field(default=86400, ge=1)


class EvaluationConfigSchema(BaseModel):
    command: str | None = None
    entrypoint: str | None = None
    working_directory: str | None = None
    timeout_seconds: int | None = Field(default=None, ge=1)
    metrics: list[str] = []
    metrics_file: str | None = None
    metrics_source: MetricsSource | None = None


class ArtifactConfigSchema(BaseModel):
    path: str | None = None
    type: str | None = None


class DatasetConfigSchema(BaseModel):
    path: str | None = None


class CurriculumConfigSchema(BaseModel):
    strategy: CurriculumType = "failure_driven"


class OptimizationConfigSchema(BaseModel):
    max_iterations: int = Field(default=10, ge=1)
    target_metric: str | None = None
    target_value: float | None = None
    minimum_improvement: float = Field(default=0.005, ge=0.0)


class AdapterConfigSchema(BaseModel):
    """Section 11/12 -- which `TrainingAdapter` runs this project.
    `"custom"` means `entrypoint` (default `.ai-env/adapter.py`) is a
    Python module defining `get_adapter()`, run in its own subprocess by
    `python` -- see docs/custom-adapters.md."""

    type: Literal["command", "custom"] = "command"
    entrypoint: str | None = None
    python: str = "python"


class ProjectConfigUpdate(BaseModel):
    """Section 47's "Edit Configuration" action / Section 49's wizard
    steps 2-6, submitted as one payload. `name` is required since it also
    updates `Project.name` (a project may be renamed independently of its
    `root_path`)."""

    name: str = Field(min_length=1, max_length=255)
    project: ProjectMetaSchema = ProjectMetaSchema()
    training: TrainingConfigSchema
    evaluation: EvaluationConfigSchema = EvaluationConfigSchema()
    artifact: ArtifactConfigSchema = ArtifactConfigSchema()
    dataset: DatasetConfigSchema = DatasetConfigSchema()
    curriculum: CurriculumConfigSchema = CurriculumConfigSchema()
    optimization: OptimizationConfigSchema = OptimizationConfigSchema()
    adapter: AdapterConfigSchema = AdapterConfigSchema()


class ProjectConfigRead(BaseModel):
    """The confirmed contract, parsed back out of `config_yaml` for
    display/editing (Section 47's "Edit Configuration" needs to show the
    current values, not just accept new ones)."""

    project: ProjectMetaSchema
    training: TrainingConfigSchema
    evaluation: EvaluationConfigSchema
    artifact: ArtifactConfigSchema
    dataset: DatasetConfigSchema
    curriculum: CurriculumConfigSchema
    optimization: OptimizationConfigSchema
    adapter: AdapterConfigSchema


class ValidationIssueRead(BaseModel):
    field: str
    message: str
    severity: Literal["error", "warning"]


class ProjectValidationRead(BaseModel):
    ok: bool
    issues: list[ValidationIssueRead]


# -- runs (train + evaluate cycles, Section 19/Rule 11) -------------------

ProjectRunStatus = Literal[
    "queued", "running_training", "running_evaluation", "completed", "failed", "cancelled"
]


class ArtifactInfoRead(BaseModel):
    path: str
    exists: bool
    size_bytes: int | None = None
    type: str | None = None


class TrainResultRead(BaseModel):
    status: str
    command: str
    exit_code: int | None
    started_at: str
    ended_at: str
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str
    artifact: ArtifactInfoRead | None = None


class EvaluationResultRead(BaseModel):
    status: str
    metrics: dict
    samples: int | None = None
    errors: int | None = None
    artifact: str | None = None
    warnings: list[str] = []


class ProjectRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    run_number: int
    status: ProjectRunStatus
    started_at: datetime
    ended_at: datetime | None
    train_result: TrainResultRead | None
    evaluation_result: EvaluationResultRead | None
    managed_artifact_path: str | None
    error: str | None


# -- progress (Section 41, generalized) ------------------------------------


class MetricTrendPointRead(BaseModel):
    run_number: int
    value: float


class MetricTrendRead(BaseModel):
    metric: str
    points: list[MetricTrendPointRead]
    latest_value: float
    previous_value: float | None
    delta_from_previous: float | None


class ProjectProgressRead(BaseModel):
    metrics: list[MetricTrendRead]
    completed_run_count: int
    target_metric: str | None
    target_value: float | None
    latest_target_value: float | None
    distance_to_target: float | None
    moved_closer_to_target: bool | None
