"""Mode B -- service layer for `/api/projects/*` (Section 24's "no ML/
business logic in routes", applied the same way `ExperimentService`
applies it to `/api/experiments/*`).

`validate()` calls `TrainingAdapter.validate_project()`, which Section
74/75 both describe as cheap and side-effect-free (existence checks only,
no subprocess). Actually running the user's training/evaluation command
happens in `start_run()`/`_run_project_run_in_background` below, using the
exact same FastAPI `BackgroundTasks` pattern
`ExperimentService._run_experiment_in_background` already uses for
GridWorld/PPO (Rule 11: never inline in the request; the route only ever
sets status to `queued`/`running_training` and hands off).

Section 5 (Local Agent): every method that touches the project's actual
filesystem or runs a command -- `scan`, `validate`, and the run execution
inside `_run_project_run_in_background` -- branches on `project.agent_id`.
`None` means what it always has (Section 61's "local execution": the
backend does it itself via `ProjectScanner`/`build_adapter` directly, same
machine, same filesystem). Set means every one of those operations is
instead proxied over HTTP to that registered Agent
(`app/services/agent_client.py`, wrapping `agent_app/main.py`) -- the only
way `root_path` can refer to a path on a genuinely different machine than
this backend. Callers of `ProjectService` never need to know which case
they're in; the return shapes (`ScanResult`, `ValidationResult`,
`ProjectRun`) are identical either way.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.session import SessionLocal
from app.ml.project.adapters.base import RunContext, ValidationIssue, ValidationResult
from app.ml.project.adapters.registry import build_adapter
from app.ml.project.capabilities import Capability
from app.ml.project.config import (
    AdapterConfig,
    AIEnvConfig,
    ArtifactConfig,
    CurriculumConfig,
    DatasetConfig,
    EvaluationConfig,
    OptimizationConfig,
    ProjectMeta,
    TrainingConfig,
)
from app.ml.project.progress import ProjectProgress, compute_progress
from app.ml.project.scanner import ProjectScanner, ScanResult
from app.ml.project.security import resolve_within
from app.ml.project.storage import archive_downloaded_artifact, archive_run_artifact
from app.models.orm import Agent, Project, ProjectRun
from app.schemas.project import ProjectConfigUpdate, ProjectRegisterRequest
from app.services.agent_client import AgentError, AgentUnreachable
from app.services.agent_service import AgentNotFound, client_for_agent

logger = logging.getLogger(__name__)

# Mirrors `experiment_service._cancel_flags`: an in-process, best-effort
# cancellation signal per run id. Like that one, this does not survive a
# backend restart and does not coordinate across multiple backend
# processes -- a documented limitation, not an oversight (see
# docs/projects-and-adapters.md). For an agent-backed run this flag is
# what tells `_execute_via_agent`'s poll loop to forward the cancellation
# on to the agent -- the agent's own execution is a different process
# entirely and has no way to see this flag directly.
_cancel_flags: dict[int, threading.Event] = {}

_AGENT_POLL_INTERVAL_SECONDS = 0.5


class ProjectNotFound(Exception):
    def __init__(self, project_id: int):
        super().__init__(f"Project {project_id} not found")
        self.project_id = project_id


class ProjectRunNotFound(Exception):
    def __init__(self, project_id: int, run_id: int):
        super().__init__(f"Run {run_id} not found for project {project_id}")
        self.project_id = project_id
        self.run_id = run_id


class ProjectStateError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def _scan_result_from_dict(data: dict) -> ScanResult:
    """Reconstructs a `ScanResult` from an Agent's `/scan` response, which
    is `ScanResult.to_dict()`'s exact shape (the agent calls the same
    `ProjectScanner`). Keeps `ProjectService.scan()`'s return type uniform
    regardless of which execution path produced it."""
    return ScanResult(
        root=data["root"],
        language=data["language"],
        framework=data["framework"],
        python_version=data["python_version"],
        gpu_available=data["gpu"]["available"],
        gpu_name=data["gpu"]["name"],
        training_candidates=data["training_candidates"],
        evaluation_candidates=data["evaluation_candidates"],
        dataset_candidates=data["dataset_candidates"],
        artifact_candidates=data["artifact_candidates"],
        dependency_files=data["dependency_files"],
    )


def _validation_result_from_dict(data: dict) -> ValidationResult:
    return ValidationResult(ok=data["ok"], issues=[ValidationIssue(**issue) for issue in data["issues"]])


class ProjectService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def register(self, payload: ProjectRegisterRequest) -> Project:
        root = Path(payload.root_path)
        project = Project(
            name=payload.name or root.name or payload.root_path,
            root_path=str(root),
            source="local",
            status="draft",
            capabilities=[],
            agent_id=payload.agent_id,
        )
        self.session.add(project)
        self.session.commit()
        self.session.refresh(project)
        return project

    def list_all(self) -> list[Project]:
        return self.session.query(Project).order_by(Project.id).all()

    def get(self, project_id: int) -> Project:
        project = self.session.get(Project, project_id)
        if project is None:
            raise ProjectNotFound(project_id)
        return project

    def _require_agent(self, project: Project) -> Agent:
        agent = self.session.get(Agent, project.agent_id)
        if agent is None:
            raise ProjectStateError(f"project's assigned agent (id={project.agent_id}) no longer exists")
        return agent

    def assign_agent(self, project_id: int, agent_id: int | None) -> Project:
        """Section 5's "Install/connect Local Agent" step, applied to an
        already-registered project: switches which machine future scan/
        validate/run calls happen on. Does not touch `config_yaml`,
        `detected_config`, or run history -- those stay whatever they were,
        even though a rescan against the new agent may now disagree with
        them (Section 6: detected values are always just suggestions)."""
        project = self.get(project_id)
        if agent_id is not None and self.session.get(Agent, agent_id) is None:
            raise AgentNotFound(agent_id)
        project.agent_id = agent_id
        self.session.commit()
        self.session.refresh(project)
        return project

    def scan(self, project_id: int) -> ScanResult:
        """Section 47's "Scan"/"Rescan" action. Only ever updates
        `language`/`framework`/`detected_config` -- never touches
        `config_yaml`, since a rescan must not silently change a
        configuration the user already confirmed (Section 6: detected
        values are suggestions, and re-detecting them doesn't make them
        any less so)."""
        project = self.get(project_id)

        if project.agent_id is not None:
            agent = self._require_agent(project)
            try:
                result = _scan_result_from_dict(client_for_agent(agent).scan(project.root_path))
            except AgentUnreachable as exc:
                raise ProjectStateError(f"agent {agent.name!r} is unreachable: {exc}") from exc
            except AgentError as exc:
                raise ProjectStateError(f"agent {agent.name!r} could not scan the project: {exc.detail}") from exc
        else:
            result = ProjectScanner().scan(project.root_path)  # NotADirectoryError propagates to the route

        project.language = result.language
        project.framework = result.framework
        project.detected_config = result.to_dict()
        if project.status == "draft":
            project.status = "scanned"
        self.session.commit()
        self.session.refresh(project)
        return result

    def update_config(self, project_id: int, payload: ProjectConfigUpdate) -> Project:
        """Section 47's "Edit Configuration" / Section 49's wizard,
        submitted as one confirmed contract. Building `AIEnvConfig` here
        re-runs every dataclass `__post_init__` check (Section 8), so an
        invalid payload raises `ValueError` and nothing is persisted --
        never a half-valid config silently saved."""
        project = self.get(project_id)

        config = AIEnvConfig(
            project=ProjectMeta(
                name=payload.name, task=payload.project.task, framework=payload.project.framework
            ),
            training=TrainingConfig(**payload.training.model_dump()),
            evaluation=EvaluationConfig(**payload.evaluation.model_dump()),
            artifact=ArtifactConfig(**payload.artifact.model_dump()),
            dataset=DatasetConfig(**payload.dataset.model_dump()),
            curriculum=CurriculumConfig(**payload.curriculum.model_dump()),
            optimization=OptimizationConfig(**payload.optimization.model_dump()),
            adapter=AdapterConfig(**payload.adapter.model_dump()),
        )

        # CommandAdapter's capabilities are a pure function of the config
        # itself (which sections have a command/path set) -- never of
        # filesystem reality -- so computing them here, on the backend, is
        # correct regardless of whether this project ultimately executes
        # locally or through an agent. CustomAdapter's are NOT: they come
        # from importing the user's actual `.ai-env/adapter.py` (Section
        # 11), which may only exist on an assigned agent's filesystem, not
        # this backend's. So for a custom adapter, `Project.capabilities`
        # is deliberately left empty here rather than risk inspecting the
        # wrong machine -- `validate()` (which already correctly delegates
        # the whole check to wherever the project actually lives) is where
        # a custom adapter's real capabilities surface, as validation
        # issues.
        if config.adapter.type == "command":
            adapter = build_adapter(config, project.root_path)
            project.capabilities = [capability.value for capability in adapter.capabilities]
        else:
            project.capabilities = []

        project.name = payload.name
        project.task = payload.project.task
        project.framework = payload.project.framework or project.framework
        project.config_yaml = config.to_yaml()
        project.status = "configured"
        project.validation_result = None  # stale as of any config edit -- must be re-validated
        self.session.commit()
        self.session.refresh(project)
        return project

    def get_config(self, project_id: int) -> AIEnvConfig:
        project = self.get(project_id)
        if not project.config_yaml:
            raise ProjectStateError(
                f"project {project_id} has no confirmed configuration yet -- call "
                "PUT /api/projects/{id}/config first"
            )
        return AIEnvConfig.from_yaml(project.config_yaml)

    def validate(self, project_id: int) -> ValidationResult:
        """Section 74 -- cheap, side-effect-free checks only (paths,
        commands configured, python available). Never executes the
        project's training/evaluation command; see module docstring."""
        project = self.get(project_id)
        config = self.get_config(project_id)

        if project.agent_id is not None:
            agent = self._require_agent(project)
            try:
                result = _validation_result_from_dict(
                    client_for_agent(agent).validate(project.root_path, config.to_dict())
                )
            except AgentUnreachable as exc:
                raise ProjectStateError(f"agent {agent.name!r} is unreachable: {exc}") from exc
            except AgentError as exc:
                raise ProjectStateError(f"agent {agent.name!r} could not validate the project: {exc.detail}") from exc
        else:
            adapter = build_adapter(config, project.root_path)
            result = adapter.validate_project()

        project.status = "validated" if result.ok else "validation_failed"
        project.validation_result = result.to_dict()
        self.session.commit()
        self.session.refresh(project)
        return result

    # -- runs (Section 19: training/evaluation, off the request thread) --

    def start_run(self, project_id: int, background_tasks) -> ProjectRun:
        project = self.get(project_id)
        config = self.get_config(project_id)  # raises ProjectStateError if unconfigured
        # CommandAdapter's TRAIN capability is a pure config check (same
        # reasoning as update_config() above), so pre-flighting it here is
        # safe and gives a fast, clear 409 instead of a doomed run.
        # CustomAdapter's is not -- it needs the real `.ai-env/adapter.py`,
        # which may only be reachable through an assigned agent -- so for
        # a custom adapter this check is skipped here and left to actual
        # execution (`_execute_locally`/`_execute_via_agent`, which build
        # the adapter correctly on whichever machine actually has the
        # project), where an unsupported capability still surfaces
        # clearly, just as a failed run rather than a rejected request.
        if config.adapter.type == "command" and not build_adapter(config, project.root_path).supports(
            Capability.TRAIN
        ):
            raise ProjectStateError(f"project {project_id}'s configuration has no training command")
        if project.agent_id is not None:
            self._require_agent(project)  # fail fast rather than queuing a run doomed to error immediately

        last_run_number = (
            self.session.query(func.max(ProjectRun.run_number))
            .filter(ProjectRun.project_id == project_id)
            .scalar()
            or 0
        )

        run = ProjectRun(project_id=project_id, run_number=last_run_number + 1, status="running_training")
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)

        _cancel_flags[run.id] = threading.Event()
        background_tasks.add_task(_run_project_run_in_background, run.id)
        return run

    def list_runs(self, project_id: int) -> list[ProjectRun]:
        self.get(project_id)  # 404s if the project itself doesn't exist
        return (
            self.session.query(ProjectRun)
            .filter(ProjectRun.project_id == project_id)
            .order_by(ProjectRun.run_number.desc())
            .all()
        )

    def get_run(self, project_id: int, run_id: int) -> ProjectRun:
        run = self.session.get(ProjectRun, run_id)
        if run is None or run.project_id != project_id:
            raise ProjectRunNotFound(project_id, run_id)
        return run

    def cancel_run(self, project_id: int, run_id: int) -> ProjectRun:
        """Best-effort, same honesty as `ExperimentService.stop()`: this
        only sets the flag. For a local run, the background thread notices
        it on its next ~0.2s poll (`CommandAdapter._run`); for an
        agent-backed run, `_execute_via_agent`'s poll loop notices it on
        its next ~0.5s cycle and forwards a cancel call to the agent.
        Either way, `status` may still read `running_training`/
        `running_evaluation` for a moment after this returns."""
        run = self.get_run(project_id, run_id)
        flag = _cancel_flags.get(run_id)
        if flag is not None:
            flag.set()
        return run

    def get_progress(self, project_id: int) -> ProjectProgress:
        """Section 41, generalized: real run-over-run metric trends, never
        a fabricated weakness score (Section 88) -- see
        `app/ml/project/progress.py`'s module docstring for why this stops
        short of a full per-condition weakness vector. Lenient about an
        unconfigured project (no runs can exist yet anyway) rather than
        raising, matching Section 89's "show real state, empty if there is
        none" empty-state philosophy."""
        project = self.get(project_id)
        runs = self.list_runs(project_id)
        optimization = self.get_config(project_id).optimization if project.config_yaml else OptimizationConfig()
        return compute_progress(runs, optimization)


def _run_project_run_in_background(run_id: int) -> None:
    session = SessionLocal()
    cancel_event = _cancel_flags.get(run_id, threading.Event())
    try:
        run = session.get(ProjectRun, run_id)
        if run is None:
            return

        project = session.get(Project, run.project_id)
        if project is None:
            run.status = "failed"
            run.error = "the project was deleted while this run was queued"
            run.ended_at = datetime.now(timezone.utc)
            session.commit()
            return

        try:
            if project.agent_id is not None:
                _execute_via_agent(session, project, run, cancel_event)
            else:
                _execute_locally(session, project, run, cancel_event)
        except Exception as exc:
            logger.exception("project_run_errored run_id=%s", run_id)
            run.status = "failed"
            run.error = str(exc)
            run.ended_at = datetime.now(timezone.utc)
            session.commit()
    finally:
        session.close()
        _cancel_flags.pop(run_id, None)


def _execute_locally(session: Session, project: Project, run: ProjectRun, cancel_event: threading.Event) -> None:
    config = AIEnvConfig.from_yaml(project.config_yaml)
    adapter = build_adapter(config, project.root_path)

    # {output} -- an ai.env-managed, per-run directory the command can
    # write into if it wants to, created up front so it's guaranteed to
    # exist. {dataset} -- resolved to an actual filesystem path (not left
    # as the bare, unresolved config value), since a relative path
    # substituted verbatim would depend on the command's cwd matching the
    # project root exactly. `{checkpoint}`/`{config}` have no meaningful
    # value yet in this phase (no "previous checkpoint" or generated
    # per-run config file concept implemented) and stay empty -- see
    # docs/projects-and-adapters.md.
    run_storage_dir = Path(settings.ai_env_storage_root) / "projects" / str(project.id) / "runs" / str(run.run_number)
    run_storage_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = str(resolve_within(Path(project.root_path), config.dataset.path)) if config.dataset.path else None

    ctx = RunContext(
        project_root=Path(project.root_path),
        experiment_id=str(project.id),
        run_id=str(run.run_number),
        dataset_path=dataset_path,
        output_dir=str(run_storage_dir),
    )

    logger.info("project_run_started project_id=%s run_number=%s mode=local", project.id, run.run_number)

    train_result = adapter.train(ctx, cancel_event=cancel_event)
    run.train_result = train_result.to_dict()

    if train_result.status != "success":
        run.status = "cancelled" if cancel_event.is_set() else "failed"
        run.ended_at = datetime.now(timezone.utc)
        session.commit()
        return

    if train_result.artifact is not None and train_result.artifact.exists:
        run.managed_artifact_path = archive_run_artifact(
            settings.ai_env_storage_root, project.id, run.run_number, Path(train_result.artifact.path)
        )

    if cancel_event.is_set():
        run.status = "cancelled"
        run.ended_at = datetime.now(timezone.utc)
        session.commit()
        return

    if not adapter.supports(Capability.EVALUATE):
        run.status = "completed"
        run.ended_at = datetime.now(timezone.utc)
        session.commit()
        return

    run.status = "running_evaluation"
    session.commit()

    evaluation_result = adapter.evaluate(ctx, cancel_event=cancel_event)
    run.evaluation_result = evaluation_result.to_dict()

    if cancel_event.is_set():
        run.status = "cancelled"
    elif evaluation_result.status == "success":
        run.status = "completed"
    else:
        run.status = "failed"
    run.ended_at = datetime.now(timezone.utc)
    session.commit()
    logger.info(
        "project_run_finished project_id=%s run_number=%s status=%s", project.id, run.run_number, run.status
    )


def _execute_via_agent(session: Session, project: Project, run: ProjectRun, cancel_event: threading.Event) -> None:
    """Section 5's whole point: `root_path` may not even exist on this
    machine, so every filesystem/execution step -- not just command
    execution -- has to happen on the agent's side. This function's job is
    reduced to orchestration: start the run, poll it, forward cancellation,
    and pull the finished artifact's bytes back once there's something to
    archive. See `agent_app/run_manager.py` for the actual execution."""
    agent = session.get(Agent, project.agent_id)
    if agent is None:
        run.status = "failed"
        run.error = f"project's assigned agent (id={project.agent_id}) no longer exists"
        run.ended_at = datetime.now(timezone.utc)
        session.commit()
        return

    client = client_for_agent(agent)
    config_dict = AIEnvConfig.from_yaml(project.config_yaml).to_dict()

    logger.info(
        "project_run_started project_id=%s run_number=%s mode=agent agent_id=%s",
        project.id,
        run.run_number,
        agent.id,
    )

    try:
        agent_run_id = client.start_run(project.root_path, config_dict, run.run_number)
    except (AgentUnreachable, AgentError) as exc:
        run.status = "failed"
        run.error = f"could not start run on agent {agent.name!r}: {exc}"
        run.ended_at = datetime.now(timezone.utc)
        session.commit()
        return

    run.agent_run_id = agent_run_id
    session.commit()

    agent_run: dict = {}
    while True:
        if cancel_event.is_set():
            try:
                client.cancel_run(agent_run_id)
            except (AgentUnreachable, AgentError):
                pass  # best-effort, same as the local cancellation path

        try:
            agent_run = client.get_run(agent_run_id)
        except (AgentUnreachable, AgentError) as exc:
            run.status = "failed"
            run.error = f"lost contact with agent {agent.name!r} while this run was in progress: {exc}"
            run.ended_at = datetime.now(timezone.utc)
            session.commit()
            return

        run.status = agent_run["status"]
        run.train_result = agent_run["train_result"]
        run.evaluation_result = agent_run["evaluation_result"]
        if agent_run.get("error"):
            run.error = agent_run["error"]
        session.commit()

        if run.status not in ("running_training", "running_evaluation"):
            break
        time.sleep(_AGENT_POLL_INTERVAL_SECONDS)

    if agent_run.get("has_artifact"):
        try:
            downloaded = client.download_artifact(agent_run_id)
        except (AgentUnreachable, AgentError):
            downloaded = None
        if downloaded is not None:
            content, filename = downloaded
            run.managed_artifact_path = archive_downloaded_artifact(
                settings.ai_env_storage_root, project.id, run.run_number, filename, content
            )
            session.commit()

    run.ended_at = datetime.now(timezone.utc)
    session.commit()
    logger.info(
        "project_run_finished project_id=%s run_number=%s status=%s mode=agent",
        project.id,
        run.run_number,
        run.status,
    )
