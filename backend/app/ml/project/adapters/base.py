"""Section 9 -- the universal `TrainingAdapter` abstraction, plus the
handful of small value objects every adapter implementation shares
(`RunContext`, `ValidationResult`, `TrainResult`, `EvaluationResult`,
`ArtifactInfo`).

Only `capabilities` is a genuine abstract requirement. Every other method
has a base implementation that checks the capability and raises
`CapabilityNotSupported` when the subclass hasn't declared it, or
`NotImplementedError` when it *has* declared it but forgot to override the
method -- so a half-finished adapter fails loudly rather than silently
behaving like a fully-featured one. This is what Section 9's "Do NOT
require every adapter to implement every method" means in practice: a
`CommandAdapter` that only has a training command is a completely valid
adapter that simply never claims `Capability.EVALUATE`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
from typing import Mapping

from app.ml.project.capabilities import Capability
from app.ml.project.config import SUPPORTED_VARIABLES


class CapabilityNotSupported(Exception):
    def __init__(self, adapter: "TrainingAdapter", capability: Capability):
        super().__init__(f"{type(adapter).__name__} does not declare capability {capability.value!r}")
        self.capability = capability


def substitute_variables(command: str, variables: Mapping[str, str]) -> str:
    """Render a configured command string's `{name}` placeholders
    (Section 10). All seven supported variables always get a value --
    empty string for ones not yet known in this context (e.g.
    `{checkpoint}` before any run has produced one) -- so a valid command
    never fails to render just because one placeholder isn't relevant yet.
    Referencing any other `{name}` is a configuration error and raises
    immediately rather than executing a mangled command.
    """
    full = {var: variables.get(var, "") for var in SUPPORTED_VARIABLES}
    try:
        return command.format_map(full)
    except KeyError as exc:
        raise ValueError(
            f"command {command!r} references unsupported variable {{{exc.args[0]}}}; "
            f"supported variables are {SUPPORTED_VARIABLES}"
        ) from exc


@dataclass(frozen=True)
class RunContext:
    """Everything a training/evaluation command's `{name}` placeholders
    might need to resolve to, for one concrete run. `output_dir` is the
    ai.env-managed experiment storage directory (Section 15), never the
    user's own project directory."""

    project_root: Path
    experiment_id: str
    run_id: str
    dataset_path: str | None = None
    checkpoint_path: str | None = None
    output_dir: str | None = None
    config_path: str | None = None

    def variables(self) -> dict[str, str]:
        return {
            "project": str(self.project_root),
            "dataset": self.dataset_path or "",
            "experiment": self.experiment_id,
            "run": self.run_id,
            "checkpoint": self.checkpoint_path or "",
            "output": self.output_dir or "",
            "config": self.config_path or "",
        }


@dataclass
class ValidationIssue:
    field: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "issues": [asdict(i) for i in self.issues]}


@dataclass
class ArtifactInfo:
    path: str
    exists: bool
    size_bytes: int | None = None
    type: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    status: str  # "success" | "failed" | "timeout"
    command: str
    exit_code: int | None
    started_at: str
    ended_at: str
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str
    artifact: ArtifactInfo | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["artifact"] = self.artifact.to_dict() if self.artifact else None
        return d


@dataclass
class EvaluationResult:
    """Section 24's evaluation contract shape, plus `warnings` for the
    honest "no metrics collected" case (Section 88 -- never fabricate a
    metrics value in place of one that wasn't actually produced)."""

    status: str  # "success" | "failed"
    metrics: dict = field(default_factory=dict)
    samples: int | None = None
    errors: int | None = None
    artifact: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TrainingAdapter(ABC):
    """Section 9's universal contract. `capabilities` is the only thing a
    concrete adapter must define; it is a promise, so declaring a
    capability without overriding its method is a bug, not a supported
    configuration -- the base method deliberately raises
    `NotImplementedError` (not silently no-ops) in that case."""

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[Capability]: ...

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def _guard(self, capability: Capability) -> None:
        if not self.supports(capability):
            raise CapabilityNotSupported(self, capability)

    def validate_project(self) -> ValidationResult:
        return ValidationResult(ok=True, issues=[])

    def prepare_environment(self) -> None:
        """Optional hook (e.g. creating a virtualenv, installing
        dependencies). No-op by default -- most CommandAdapter projects
        already have a working environment and don't need this."""
        return None

    def train(self, ctx: RunContext, cancel_event: Event | None = None) -> TrainResult:
        self._guard(Capability.TRAIN)
        raise NotImplementedError(f"{type(self).__name__} declares TRAIN but has not implemented train()")

    def evaluate(self, ctx: RunContext, cancel_event: Event | None = None) -> EvaluationResult:
        self._guard(Capability.EVALUATE)
        raise NotImplementedError(f"{type(self).__name__} declares EVALUATE but has not implemented evaluate()")

    def get_metrics(self) -> dict | None:
        """The most recently collected evaluation metrics, if any. `None`
        (not `{}`) means no evaluation has run yet -- callers must not
        treat that as "all metrics are zero" (Section 88)."""
        return None

    def get_artifact(self) -> ArtifactInfo:
        self._guard(Capability.CHECKPOINT)
        raise NotImplementedError(f"{type(self).__name__} declares CHECKPOINT but has not implemented get_artifact()")

    def load_artifact(self):
        self._guard(Capability.CHECKPOINT)
        raise NotImplementedError(f"{type(self).__name__} declares CHECKPOINT but has not implemented load_artifact()")

    def predict(self, input_data):
        self._guard(Capability.PREDICT)
        raise NotImplementedError(f"{type(self).__name__} declares PREDICT but has not implemented predict()")

    def cleanup(self) -> None:
        return None
