
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.ml.project.adapters.base import RunContext
from app.ml.project.adapters.registry import build_adapter
from app.ml.project.capabilities import Capability
from app.ml.project.config import AIEnvConfig
from app.ml.project.security import resolve_within


@dataclass
class AgentRun:
    id: str
    status: str = "running_training"  # running_training | running_evaluation | completed | failed | cancelled
    train_result: dict | None = None
    evaluation_result: dict | None = None
    error: str | None = None
    artifact_path: str | None = None  # local to the agent's own machine
    cancel_event: threading.Event = field(default_factory=threading.Event)


class RunManager:
    """One instance per agent process, held for its lifetime."""

    def __init__(self, output_root: str | Path):
        self._runs: dict[str, AgentRun] = {}
        self._lock = threading.Lock()
        self.output_root = Path(output_root)

    def start_run(self, root_path: str, config_dict: dict, run_number: int) -> str:
        run_id = uuid.uuid4().hex
        run = AgentRun(id=run_id)
        with self._lock:
            self._runs[run_id] = run

        thread = threading.Thread(
            target=self._execute,
            args=(run, root_path, config_dict, run_number),
            daemon=True,
        )
        thread.start()
        return run_id

    def get(self, run_id: str) -> AgentRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        run = self.get(run_id)
        if run is None:
            return False
        run.cancel_event.set()
        return True

    def _execute(self, run: AgentRun, root_path: str, config_dict: dict, run_number: int) -> None:
        try:
            config = AIEnvConfig.from_dict(config_dict)
            adapter = build_adapter(config, root_path)

            # {dataset} must be resolved HERE, on the agent's own machine
            # -- the backend, talking to a genuinely remote agent, may
            # have no filesystem access to `root_path` at all, so it
            # cannot resolve this itself the way it does for a same-machine
            # (no-agent) run.
            dataset_path = (
                str(resolve_within(Path(root_path), config.dataset.path)) if config.dataset.path else None
            )

            # This agent's own local storage, independent of the backend's
            # AI_ENV_STORAGE_ROOT (a different machine entirely) -- just
            # somewhere for `{output}` to point at during the run itself.
            output_dir = self.output_root / "runs" / str(run_number)
            output_dir.mkdir(parents=True, exist_ok=True)

            ctx = RunContext(
                project_root=Path(root_path),
                experiment_id="agent",
                run_id=str(run_number),
                dataset_path=dataset_path,
                output_dir=str(output_dir),
            )

            train_result = adapter.train(ctx, cancel_event=run.cancel_event)
            run.train_result = train_result.to_dict()

            if train_result.status != "success":
                run.status = "cancelled" if run.cancel_event.is_set() else "failed"
                return

            if train_result.artifact is not None and train_result.artifact.exists:
                run.artifact_path = train_result.artifact.path

            if run.cancel_event.is_set():
                run.status = "cancelled"
                return

            if not adapter.supports(Capability.EVALUATE):
                run.status = "completed"
                return

            run.status = "running_evaluation"
            evaluation_result = adapter.evaluate(ctx, cancel_event=run.cancel_event)
            run.evaluation_result = evaluation_result.to_dict()

            if run.cancel_event.is_set():
                run.status = "cancelled"
            elif evaluation_result.status == "success":
                run.status = "completed"
            else:
                run.status = "failed"
        except Exception as exc:  # defensive: never leave a run stuck "running" on a bug here
            run.status = "failed"
            run.error = str(exc)
