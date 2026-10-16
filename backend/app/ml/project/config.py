"""Mode B (master prompt Section 8) -- the `ai-env.yaml` training contract.

This is the one artifact that lets ai.env orchestrate a user's own AI
project without ever reading or rewriting that project's source code
(Section 3's "most important architectural principle"). It mirrors the
nested YAML shape from Section 8's example exactly (`project:`,
`training:`, `evaluation:`, `artifact:`, `dataset:`, `curriculum:`,
`optimization:`) rather than the flat shape `ExperimentConfig` uses for
GridWorld -- the two configs describe different things (a static
per-project contract vs. one GridWorld experiment's numeric knobs) and
Section 16 only asks that *equivalent* models be reused, not that every
config in the project share one shape.

Follows the same dataclass conventions as `app.ml.experiment.config`
(`__post_init__` validation, `to_dict`/`from_dict`, `save`/`load`) rather
than pydantic, since this lives in `app/ml` which stays free of any web
framework -- the FastAPI-facing shape is a separate, thin schema in
`app/schemas/project.py` that converts to/from this.

Every field the user hasn't confirmed yet is a *suggestion* (Section 6/7):
nothing here is ever populated automatically and then silently trusted --
callers (the scanner, the project service) are responsible for keeping
"detected" and "confirmed" data visibly separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from app.ml.experiment.config import VALID_CURRICULUM_TYPES

# Section 10 -- the fixed set of `{name}` placeholders a training/evaluation
# command may reference. Kept here (not in the adapter module) since it is
# part of the contract itself, not an execution detail.
SUPPORTED_VARIABLES = ("project", "dataset", "experiment", "run", "checkpoint", "output", "config")

VALID_METRICS_SOURCES = (None, "file", "stdout")


@dataclass
class ProjectMeta:
    name: str
    task: str | None = None
    framework: str | None = None


@dataclass
class TrainingConfig:
    command: str
    entrypoint: str | None = None
    working_directory: str = "."
    timeout_seconds: int = 86400

    def __post_init__(self) -> None:
        if not self.command or not self.command.strip():
            raise ValueError("training.command must not be empty")
        if self.timeout_seconds < 1:
            raise ValueError("training.timeout_seconds must be >= 1")


@dataclass
class EvaluationConfig:
    command: str | None = None
    entrypoint: str | None = None
    working_directory: str | None = None
    timeout_seconds: int | None = None
    metrics: list[str] = field(default_factory=list)
    # Section 24 -- exactly one of these decides how metrics are collected,
    # and it must be explicit. Leaving both unset is valid (the command
    # still runs) but yields no metrics rather than a guess.
    metrics_file: str | None = None
    metrics_source: str | None = None  # "file" | "stdout" | None

    def __post_init__(self) -> None:
        if self.metrics_source not in VALID_METRICS_SOURCES:
            raise ValueError(
                f"evaluation.metrics_source must be one of {VALID_METRICS_SOURCES}, "
                f"got {self.metrics_source!r}"
            )
        if self.metrics_source == "file" and not self.metrics_file:
            raise ValueError("evaluation.metrics_source='file' requires evaluation.metrics_file")
        if self.timeout_seconds is not None and self.timeout_seconds < 1:
            raise ValueError("evaluation.timeout_seconds must be >= 1")


@dataclass
class ArtifactConfig:
    path: str | None = None
    type: str | None = None


@dataclass
class DatasetConfig:
    path: str | None = None


@dataclass
class CurriculumConfig:
    strategy: str = "failure_driven"

    def __post_init__(self) -> None:
        if self.strategy not in VALID_CURRICULUM_TYPES:
            raise ValueError(
                f"curriculum.strategy must be one of {VALID_CURRICULUM_TYPES}, got {self.strategy!r}"
            )


VALID_ADAPTER_TYPES = ("command", "custom")


@dataclass
class AdapterConfig:
    """Section 11/12 -- which `TrainingAdapter` implementation runs this
    project. `"command"` (the default) is `CommandAdapter` (Section 10):
    `training`/`evaluation` above are shell commands. `"custom"` is
    `CustomAdapter`: `entrypoint` (default `.ai-env/adapter.py`) is a
    Python module defining `get_adapter()`, run in its own subprocess
    (never imported in-process -- see `custom_adapter.py`'s docstring for
    why) via `python`.

    Known limitation: `training.command`/`training.entrypoint` are still
    required, non-empty fields on this contract even when `type ==
    "custom"`, since a pure custom-adapter project has no shell command at
    all. See docs/custom-adapters.md for why that wasn't relaxed."""

    type: str = "command"
    entrypoint: str | None = None
    python: str = "python"

    def __post_init__(self) -> None:
        if self.type not in VALID_ADAPTER_TYPES:
            raise ValueError(f"adapter.type must be one of {VALID_ADAPTER_TYPES}, got {self.type!r}")


@dataclass
class OptimizationConfig:
    max_iterations: int = 10
    target_metric: str | None = None
    target_value: float | None = None
    minimum_improvement: float = 0.005

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("optimization.max_iterations must be >= 1")
        if self.minimum_improvement < 0:
            raise ValueError("optimization.minimum_improvement must be >= 0")


@dataclass
class AIEnvConfig:
    """The full contract persisted as `ai-env.yaml` in the user's project
    (Section 8), and mirrored (never duplicated) in `Project.config_yaml`
    once the user confirms it (Section 15's storage-structure principle:
    the user's project is only ever written to when they explicitly ask
    for it -- `save()` below is opt-in, not called by the scan/confirm
    flow itself)."""

    project: ProjectMeta
    training: TrainingConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AIEnvConfig":
        data = data or {}
        project = data.get("project") or {}
        if "name" not in project:
            raise ValueError("project.name is required")
        training = data.get("training") or {}
        if "command" not in training:
            raise ValueError("training.command is required")

        return cls(
            project=ProjectMeta(**project),
            training=TrainingConfig(**training),
            evaluation=EvaluationConfig(**(data.get("evaluation") or {})),
            artifact=ArtifactConfig(**(data.get("artifact") or {})),
            dataset=DatasetConfig(**(data.get("dataset") or {})),
            curriculum=CurriculumConfig(**(data.get("curriculum") or {})),
            optimization=OptimizationConfig(**(data.get("optimization") or {})),
            adapter=AdapterConfig(**(data.get("adapter") or {})),
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "AIEnvConfig":
        return cls.from_dict(yaml.safe_load(text))

    def save(self, path: str | Path) -> None:
        """Write `ai-env.yaml` to `path`. Callers decide *whether* and
        *where* to call this -- see Section 15/73: ai.env never writes into
        a user's project without an explicit action."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())

    @classmethod
    def load(cls, path: str | Path) -> "AIEnvConfig":
        return cls.from_yaml(Path(path).read_text())
