"""Section 10 -- `CommandAdapter`, the first (and, for this phase, only)
universal `TrainingAdapter` implementation. It runs an existing project's
own training/evaluation commands as subprocesses per the confirmed
`AIEnvConfig`, so a project never has to be rewritten to fit ai.env
(Section 3).

Execution model / Rule 11: `train()`/`evaluate()` block the calling thread
until the subprocess exits, times out, or is cancelled. This class has no
opinion about *when* it is called -- it is intentionally as dumb as a
synchronous function call. The one place Rule 11 ("NEVER run long
training synchronously inside a FastAPI HTTP request") actually gets
enforced is the caller: `ProjectService`'s background worker
(`app/services/project_service.py`) invokes these off the request thread,
the same way `ExperimentService._run_experiment_in_background` already
does for the built-in GridWorld/PPO path.

Cancellation (Section 20) is delegated to
`app/ml/project/adapters/subprocess_runner.py`'s `run_subprocess` -- real,
not aspirational: it polls the subprocess via `Popen.communicate(timeout=...)`
rather than blocking on a single `subprocess.run(timeout=...)`, so a
caller-supplied `threading.Event` can interrupt a run in flight, and it
kills the whole process group on cancel/timeout, not just the top-level
process. `CustomAdapter` (Section 11) shares the same execution primitive.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from app.ml.project.adapters.base import (
    ArtifactInfo,
    CapabilityNotSupported,
    EvaluationResult,
    RunContext,
    TrainingAdapter,
    TrainResult,
    ValidationIssue,
    ValidationResult,
    substitute_variables,
)
from app.ml.project.adapters.subprocess_runner import SubprocessResult, run_subprocess, tail
from app.ml.project.capabilities import Capability
from app.ml.project.config import AIEnvConfig
from app.ml.project.security import PathTraversalError, resolve_within


def _extract_trailing_json(stdout: str) -> dict:
    """Section 24: "If stdout parsing is used, it must be explicitly
    configured. Do not silently perform unreliable log parsing." -- so the
    one thing this looks for is a JSON object on the *last* non-empty
    line, nothing fuzzier (no brace-scanning, no "find anything that looks
    like JSON anywhere in the output")."""
    lines = [line for line in stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("evaluation.metrics_source='stdout' but the command produced no output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "evaluation.metrics_source='stdout' but the last line of output was not valid JSON: "
            f"{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("evaluation.metrics_source='stdout' expects a JSON object, got a different JSON type")
    return payload


class CommandAdapter(TrainingAdapter):
    def __init__(self, config: AIEnvConfig, project_root: str | Path):
        self.config = config
        self.project_root = Path(project_root)
        self._last_evaluation: EvaluationResult | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        caps: set[Capability] = set()
        if self.config.training.command:
            caps.add(Capability.TRAIN)
        if self.config.evaluation.command:
            caps.add(Capability.EVALUATE)
        if self.config.artifact.path:
            caps.add(Capability.CHECKPOINT)
        return frozenset(caps)

    # -- validation (Section 74, cheap/side-effect-free) -----------------

    def validate_project(self) -> ValidationResult:
        issues: list[ValidationIssue] = []

        if not self.project_root.is_dir():
            return ValidationResult(
                ok=False,
                issues=[ValidationIssue("root_path", f"{self.project_root} is not a directory", "error")],
            )

        try:
            working_directory = self._working_directory(self.config.training.working_directory)
            if not working_directory.is_dir():
                issues.append(
                    ValidationIssue(
                        "training.working_directory", f"{working_directory} does not exist", "error"
                    )
                )
        except PathTraversalError as exc:
            issues.append(ValidationIssue("training.working_directory", str(exc), "error"))

        if self.config.training.entrypoint:
            entry = self.project_root / self.config.training.entrypoint
            if not entry.is_file():
                issues.append(ValidationIssue("training.entrypoint", f"{entry} not found", "warning"))

        if self.supports(Capability.EVALUATE):
            if self.config.evaluation.entrypoint:
                entry = self.project_root / self.config.evaluation.entrypoint
                if not entry.is_file():
                    issues.append(ValidationIssue("evaluation.entrypoint", f"{entry} not found", "warning"))
            if not self.config.evaluation.metrics_file and not self.config.evaluation.metrics_source:
                issues.append(
                    ValidationIssue(
                        "evaluation.metrics_source",
                        "no metrics_file or metrics_source configured -- evaluation will run but "
                        "produce no metrics",
                        "warning",
                    )
                )

        if self.config.dataset.path:
            dataset_dir = self.project_root / self.config.dataset.path
            if not dataset_dir.exists():
                issues.append(ValidationIssue("dataset.path", f"{dataset_dir} not found", "warning"))

        if self.supports(Capability.CHECKPOINT):
            try:
                self._resolve_artifact_path()
            except PathTraversalError as exc:
                issues.append(ValidationIssue("artifact.path", str(exc), "error"))

        ok = not any(issue.severity == "error" for issue in issues)
        return ValidationResult(ok=ok, issues=issues)

    # -- command building (Section 10) -----------------------------------

    def build_train_command(self, ctx: RunContext) -> str:
        return substitute_variables(self.config.training.command, ctx.variables())

    def build_evaluate_command(self, ctx: RunContext) -> str:
        if not self.config.evaluation.command:
            raise CapabilityNotSupported(self, Capability.EVALUATE)
        return substitute_variables(self.config.evaluation.command, ctx.variables())

    # -- execution --------------------------------------------------------

    def train(self, ctx: RunContext, cancel_event: threading.Event | None = None) -> TrainResult:
        self._guard(Capability.TRAIN)
        command = self.build_train_command(ctx)
        working_directory = self._working_directory(self.config.training.working_directory)
        raw = run_subprocess(command, working_directory, self.config.training.timeout_seconds, cancel_event)

        artifact = self.get_artifact() if self.supports(Capability.CHECKPOINT) else None
        return TrainResult(
            status=raw.status,
            command=raw.command,
            exit_code=raw.exit_code,
            started_at=raw.started_at,
            ended_at=raw.ended_at,
            duration_seconds=raw.duration_seconds,
            stdout_tail=tail(raw.stdout),
            stderr_tail=tail(raw.stderr),
            artifact=artifact,
        )

    def evaluate(self, ctx: RunContext, cancel_event: threading.Event | None = None) -> EvaluationResult:
        self._guard(Capability.EVALUATE)
        command = self.build_evaluate_command(ctx)
        working_directory = self._working_directory(
            self.config.evaluation.working_directory or self.config.training.working_directory
        )
        timeout_seconds = self.config.evaluation.timeout_seconds or self.config.training.timeout_seconds
        raw = run_subprocess(command, working_directory, timeout_seconds, cancel_event)

        result = self._parse_evaluation(raw, working_directory)
        self._last_evaluation = result
        return result

    def get_metrics(self) -> dict | None:
        return self._last_evaluation.metrics if self._last_evaluation else None

    def get_artifact(self) -> ArtifactInfo:
        self._guard(Capability.CHECKPOINT)
        path = self._resolve_artifact_path()
        exists = path.is_file()
        return ArtifactInfo(
            path=str(path),
            exists=exists,
            size_bytes=path.stat().st_size if exists else None,
            type=self.config.artifact.type,
        )

    # -- internals ----------------------------------------------------------

    def _working_directory(self, relative: str | None) -> Path:
        return resolve_within(self.project_root, relative or ".")

    def _resolve_artifact_path(self) -> Path:
        return resolve_within(self.project_root, self.config.artifact.path)

    def _parse_evaluation(self, raw: SubprocessResult, working_directory: Path) -> EvaluationResult:
        if raw.status != "success":
            return EvaluationResult(
                status="failed",
                warnings=[f"evaluation command exited with status {raw.status!r} (exit_code={raw.exit_code})"],
            )

        ec = self.config.evaluation
        source = ec.metrics_source or ("file" if ec.metrics_file else None)

        if source == "file":
            metrics_path = working_directory / ec.metrics_file
            if not metrics_path.is_file():
                return EvaluationResult(
                    status="failed", warnings=[f"metrics file {metrics_path} not found after evaluation"]
                )
            try:
                payload = json.loads(metrics_path.read_text())
            except json.JSONDecodeError as exc:
                return EvaluationResult(status="failed", warnings=[f"metrics file is not valid JSON: {exc}"])
            return self._result_from_payload(payload)

        if source == "stdout":
            try:
                payload = _extract_trailing_json(raw.stdout)
            except ValueError as exc:
                return EvaluationResult(status="failed", warnings=[str(exc)])
            return self._result_from_payload(payload)

        # Neither configured -- the command still ran (useful for its exit
        # code / side effects) but no metrics were collected. See Section
        # 88: never invent a metrics value here.
        return EvaluationResult(
            status="success",
            metrics={},
            warnings=[
                "no evaluation.metrics_file or evaluation.metrics_source configured; "
                "the command ran but no metrics were collected"
            ],
        )

    def _result_from_payload(self, payload: dict) -> EvaluationResult:
        return EvaluationResult(
            status=payload.get("status", "success"),
            metrics=payload.get("metrics", {}) or {},
            samples=payload.get("samples"),
            errors=payload.get("errors"),
            artifact=payload.get("artifact"),
        )
