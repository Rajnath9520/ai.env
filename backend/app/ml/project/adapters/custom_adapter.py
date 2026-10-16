"""Section 11 -- `CustomAdapter`, the second `TrainingAdapter`
implementation (alongside `CommandAdapter`, Section 10). Where
`CommandAdapter` runs a project's own train.py/evaluate.py as configured
shell commands, `CustomAdapter` runs a project's own `.ai-env/adapter.py`:
a small Python object exposing `train(config)`/`evaluate(config)`/
`predict(config)` methods.

Deliberately NOT imported in-process. Importlib-loading and calling
arbitrary user code inside the same process as the backend or an agent
would mean a bug (or something malicious) in that code could crash, hang,
or read/corrupt state belonging to ai.env itself -- exactly what Section
59's "Executing user-provided code is dangerous... Implement safeguards"
is about, and a materially larger trust boundary than `CommandAdapter`
already accepts by running the user's own train.py as a *subprocess*. So
`CustomAdapter` runs `.ai-env/adapter.py` the same way: as a subprocess
(`_custom_adapter_runner.py`, a small stdlib-only script this repo owns),
through the identical `run_subprocess` primitive `CommandAdapter` uses --
same real cancellation, same process-group kill, same secret redaction of
anything the adapter prints. Even *inspecting* which methods the user's
object defines happens in a subprocess (the `describe` action) rather
than importing it here to check -- there is no code path in this class
that loads the user's module in-process, not even for introspection.

Contract for `.ai-env/adapter.py`, deliberately smaller than the master
prompt's own sketch (which imports ai.env's own `TrainingAdapter` base
class into the user's project -- tight coupling this avoids so a custom
adapter stays plain, dependency-free Python): the module must define
`get_adapter()` returning an object. Whichever of `train(config)`,
`evaluate(config)`, `predict(config)` methods that object defines are this
adapter's capabilities. `config` is a plain JSON-serializable dict:
`{"variables": {...RunContext fields...}, "config": {...the full
AIEnvConfig...}}`. Each method's return value must itself be JSON
serializable; `train`/`evaluate` results are expected to loosely follow
`TrainResult`/`EvaluationResult`'s own shapes (e.g. an `evaluate()` that
returns `{"metrics": {...}}`), but nothing enforces that beyond JSON
serializability -- this adapter reports whatever came back.

See docs/custom-adapters.md for the full design writeup, worked examples,
and known limitations (no configurable working directory, no streamed
progress, `training.command` staying a required-but-unused field).
"""

from __future__ import annotations

import json
import shlex
import tempfile
import threading
from pathlib import Path

from app.ml.project.adapters.base import (
    ArtifactInfo,
    EvaluationResult,
    RunContext,
    TrainingAdapter,
    TrainResult,
    ValidationIssue,
    ValidationResult,
)
from app.ml.project.adapters.subprocess_runner import SubprocessResult, run_subprocess, tail
from app.ml.project.capabilities import Capability
from app.ml.project.config import AIEnvConfig
from app.ml.project.security import PathTraversalError, resolve_within

_RUNNER_SCRIPT = Path(__file__).parent / "_custom_adapter_runner.py"
_DESCRIBE_TIMEOUT_SECONDS = 30.0
_DEFAULT_ADAPTER_ENTRYPOINT = ".ai-env/adapter.py"


class CustomAdapter(TrainingAdapter):
    def __init__(self, config: AIEnvConfig, project_root: str | Path):
        self.config = config
        self.project_root = Path(project_root)
        self._last_evaluation: EvaluationResult | None = None
        self._describe_cache: dict | None = None

    # -- capabilities (discovered, not declared -- see module docstring) --

    @property
    def capabilities(self) -> frozenset[Capability]:
        described = self._describe()
        caps: set[Capability] = set()
        if "capabilities" in described:
            if described["capabilities"].get("train"):
                caps.add(Capability.TRAIN)
            if described["capabilities"].get("evaluate"):
                caps.add(Capability.EVALUATE)
            if described["capabilities"].get("predict"):
                caps.add(Capability.PREDICT)
        if self.config.artifact.path:
            caps.add(Capability.CHECKPOINT)
        return frozenset(caps)

    def _describe(self) -> dict:
        """Runs the `describe` action once per instance and caches the
        result -- `capabilities` is read repeatedly (every `supports()`
        check), and re-spawning a subprocess on each access would be both
        slow and pointless: the module on disk isn't expected to change
        mid-run."""
        if self._describe_cache is not None:
            return self._describe_cache

        try:
            adapter_path = self._adapter_path()
        except PathTraversalError as exc:
            self._describe_cache = {"status": "error", "message": str(exc)}
            return self._describe_cache
        if not adapter_path.is_file():
            self._describe_cache = {"status": "error", "message": f"{adapter_path} not found"}
            return self._describe_cache

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "describe.json"
            self._invoke_runner("describe", adapter_path, output_path, None, _DESCRIBE_TIMEOUT_SECONDS)
            self._describe_cache = self._read_output(output_path)
        return self._describe_cache

    # -- validation (Section 74, cheap) -----------------------------------

    def validate_project(self) -> ValidationResult:
        issues: list[ValidationIssue] = []

        if not self.project_root.is_dir():
            return ValidationResult(
                ok=False,
                issues=[ValidationIssue("root_path", f"{self.project_root} is not a directory", "error")],
            )

        try:
            adapter_path = self._adapter_path()
        except PathTraversalError as exc:
            issues.append(ValidationIssue("adapter.entrypoint", str(exc), "error"))
            adapter_path = None

        if adapter_path is not None:
            if not adapter_path.is_file():
                issues.append(ValidationIssue("adapter.entrypoint", f"{adapter_path} not found", "error"))
            else:
                described = self._describe()
                if described.get("status") == "error":
                    issues.append(ValidationIssue("adapter.entrypoint", described["message"], "error"))
                elif not any(described.get("capabilities", {}).values()):
                    issues.append(
                        ValidationIssue(
                            "adapter.entrypoint",
                            "get_adapter() returned an object with none of train()/evaluate()/"
                            "predict() defined -- this adapter cannot do anything",
                            "warning",
                        )
                    )

        if self.config.artifact.path:
            try:
                self._resolve_artifact_path()
            except PathTraversalError as exc:
                issues.append(ValidationIssue("artifact.path", str(exc), "error"))

        ok = not any(issue.severity == "error" for issue in issues)
        return ValidationResult(ok=ok, issues=issues)

    # -- execution ----------------------------------------------------------

    def train(self, ctx: RunContext, cancel_event: threading.Event | None = None) -> TrainResult:
        self._guard(Capability.TRAIN)
        payload, raw = self._run_action("train", ctx, cancel_event, self.config.training.timeout_seconds)

        if raw.status != "success" or payload.get("status") != "success":
            return TrainResult(
                status=raw.status if raw.status != "success" else "failed",
                command=raw.command,
                exit_code=raw.exit_code,
                started_at=raw.started_at,
                ended_at=raw.ended_at,
                duration_seconds=raw.duration_seconds,
                stdout_tail=tail(raw.stdout),
                stderr_tail=tail(_with_runner_error(raw.stderr, payload)),
            )

        artifact = self.get_artifact() if self.supports(Capability.CHECKPOINT) else None
        return TrainResult(
            status="success",
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
        timeout_seconds = self.config.evaluation.timeout_seconds or self.config.training.timeout_seconds
        payload, raw = self._run_action("evaluate", ctx, cancel_event, timeout_seconds)

        if raw.status != "success":
            result = EvaluationResult(
                status="failed",
                warnings=[f"evaluate() subprocess exited with status {raw.status!r} (exit_code={raw.exit_code})"],
            )
        elif payload.get("status") != "success":
            result = EvaluationResult(status="failed", warnings=[payload.get("message", "evaluate() failed")])
        else:
            value = payload.get("result")
            if not isinstance(value, dict):
                result = EvaluationResult(
                    status="failed",
                    warnings=[f"evaluate() must return a JSON object with a 'metrics' key, got {type(value).__name__}"],
                )
            else:
                result = EvaluationResult(
                    status=value.get("status", "success"),
                    metrics=value.get("metrics", {}) or {},
                    samples=value.get("samples"),
                    errors=value.get("errors"),
                    artifact=value.get("artifact"),
                )

        self._last_evaluation = result
        return result

    def predict(self, input_data):
        self._guard(Capability.PREDICT)
        payload, raw = self._run_action_with_config("predict", {"input": input_data}, None, _DESCRIBE_TIMEOUT_SECONDS)
        if raw.status != "success" or payload.get("status") != "success":
            message = payload.get("message") or f"predict() subprocess exited with status {raw.status!r}"
            raise RuntimeError(message)
        return payload.get("result")

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

    def _adapter_path(self) -> Path:
        entrypoint = self.config.adapter.entrypoint or _DEFAULT_ADAPTER_ENTRYPOINT
        return resolve_within(self.project_root, entrypoint)

    def _resolve_artifact_path(self) -> Path:
        return resolve_within(self.project_root, self.config.artifact.path)

    def _run_action(
        self, action: str, ctx: RunContext, cancel_event: threading.Event | None, timeout_seconds: float
    ) -> tuple[dict, SubprocessResult]:
        config = {"variables": ctx.variables(), "config": self.config.to_dict()}
        return self._run_action_with_config(action, config, cancel_event, timeout_seconds)

    def _run_action_with_config(
        self, action: str, config: dict, cancel_event: threading.Event | None, timeout_seconds: float
    ) -> tuple[dict, SubprocessResult]:
        adapter_path = self._adapter_path()
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(config))
            raw = self._invoke_runner(action, adapter_path, output_path, config_path, timeout_seconds, cancel_event)
            payload = self._read_output(output_path)
        return payload, raw

    def _invoke_runner(
        self,
        action: str,
        adapter_path: Path,
        output_path: Path,
        config_path: Path | None,
        timeout_seconds: float,
        cancel_event: threading.Event | None = None,
    ) -> SubprocessResult:
        args = [self.config.adapter.python, str(_RUNNER_SCRIPT), str(adapter_path), action, str(output_path)]
        if config_path is not None:
            args.append(str(config_path))
        command = " ".join(shlex.quote(arg) for arg in args)
        return run_subprocess(command, self.project_root, timeout_seconds, cancel_event)

    def _read_output(self, output_path: Path) -> dict:
        if not output_path.is_file():
            return {"status": "error", "message": "the runner produced no output (it may have been killed)"}
        try:
            return json.loads(output_path.read_text())
        except json.JSONDecodeError as exc:
            return {"status": "error", "message": f"the runner's output was not valid JSON: {exc}"}


def _with_runner_error(stderr: str, payload: dict) -> str:
    message = payload.get("message")
    return f"{stderr}\n{message}" if message else stderr
