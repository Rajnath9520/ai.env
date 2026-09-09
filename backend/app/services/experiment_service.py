"""Phase 17 -- service layer (Section 24: "Do not put ML logic directly
inside API routes. Use service classes.").

`ExperimentService` is the only thing routes talk to. It owns experiment
CRUD, starting/stopping a run, and querying results -- routes translate
HTTP <-> this service, nothing more.

Background execution: environment_only generation and demo_ppo training can
both take a while, so `start()` never runs either in the request/response
cycle for those modes. It flips status to "running" and schedules
`_run_experiment_in_background` via FastAPI's BackgroundTasks (the
in-process choice Phase 0's plan called out for the prototype, with
Redis/a real queue as a documented future upgrade). The background task
opens its *own* DB session -- the request's session is closed once the
response is sent, so reusing it would be unsafe.

Phase 28 (Section 5/9) -- `agent_mode == "external"` is handled differently
on purpose: there is no "run to completion" for an external agent
experiment (Section 3's diagram has no terminal step until the external
agent disconnects) -- `start()` only generates the initial environment
(fast, synchronous, no model of any kind) and marks the experiment
`waiting_for_agent`. All subsequent progress happens through
`app/ml/agents/external_session.py`'s reset()/step(), driven by the
external agent's own HTTP calls, not by this service's background task.

Cancellation (Section 53's "experiment cancellation"): guaranteed only for
experiments that haven't started yet ("pending" -> "stopped" is immediate
and certain). For a "running" experiment, `stop()` sets a per-experiment
threading.Event as a cooperative cancellation signal, but the current
curriculum runners (Phases 6-13) execute their step loop synchronously to
completion and don't yet poll this flag between steps -- so stopping a
genuinely in-progress run is best-effort, not guaranteed, and that
limitation is stated here rather than silently implied to work.

Live progress (Phase 18): the background task also publishes small
summarized ticks to `app.workers.broadcaster.broadcaster` as training
proceeds, and calls `broadcaster.close()` when the run ends (success or
failure) so any subscribed WebSocket connections know to stop waiting.
"""

from __future__ import annotations

import logging
import threading

import yaml
from sqlalchemy.orm import Session

from app.database.repository import attach_environment_only_results, attach_results_to_experiment
from app.database.session import SessionLocal
from app.ml.agents.external_session import external_agent_manager
from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.manager import ExperimentManager
from app.models.orm import Experiment
from app.schemas.experiment import ExperimentCreate
from app.workers.broadcaster import broadcaster

logger = logging.getLogger("experiment_service")

# In-memory, per-process cancellation signals. Documented limitation: this
# does not span multiple worker processes -- a real deployment with
# multiple API workers would need a shared store (e.g. Redis) instead.
_cancel_flags: dict[int, threading.Event] = {}


class ExperimentNotFound(Exception):
    def __init__(self, experiment_id: int):
        super().__init__(f"Experiment {experiment_id} not found")
        self.experiment_id = experiment_id


class ExperimentStateError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ExperimentService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: ExperimentCreate) -> Experiment:
        config = ExperimentConfig(**payload.model_dump())
        experiment = Experiment(
            name=config.name,
            config_yaml=yaml.safe_dump(config.to_dict(), sort_keys=False),
            curriculum_type=config.curriculum_type,
            agent_mode=config.agent_mode,
            seed=config.seed,
            status="pending",
        )
        self.session.add(experiment)
        self.session.commit()
        self.session.refresh(experiment)
        return experiment

    def list_all(self) -> list[Experiment]:
        return self.session.query(Experiment).order_by(Experiment.id).all()

    def get(self, experiment_id: int) -> Experiment:
        experiment = self.session.get(Experiment, experiment_id)
        if experiment is None:
            raise ExperimentNotFound(experiment_id)
        return experiment

    def start(self, experiment_id: int, background_tasks) -> Experiment:
        experiment = self.get(experiment_id)
        if experiment.status != "pending":
            raise ExperimentStateError(
                f"cannot start experiment {experiment_id} in status {experiment.status!r}; "
                "only a 'pending' experiment can be started"
            )

        if experiment.agent_mode == "external":
            # No long background run exists for this mode (see module
            # docstring) -- just prepare the initial environment, fast and
            # synchronously, and wait for the external agent to connect.
            config = ExperimentConfig.from_dict(yaml.safe_load(experiment.config_yaml))
            session = external_agent_manager.get_or_create(experiment_id, config)
            session.prepare(self.session)
            self.session.refresh(experiment)
            return experiment

        experiment.status = "running"
        self.session.commit()
        self.session.refresh(experiment)

        _cancel_flags[experiment_id] = threading.Event()
        background_tasks.add_task(_run_experiment_in_background, experiment_id)
        return experiment

    def stop(self, experiment_id: int) -> Experiment:
        experiment = self.get(experiment_id)

        if experiment.status == "pending":
            experiment.status = "stopped"
            self.session.commit()
            self.session.refresh(experiment)
            return experiment

        if experiment.status == "running":
            flag = _cancel_flags.get(experiment_id)
            if flag is not None:
                flag.set()
            # Best-effort only -- see module docstring. Status is left as
            # "running" here since we cannot honestly claim it stopped.
            return experiment

        return experiment  # completed/failed/stopped already -- no-op


def _run_experiment_in_background(experiment_id: int) -> None:
    session = SessionLocal()
    try:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            return

        config = ExperimentConfig.from_dict(yaml.safe_load(experiment.config_yaml))
        try:
            logger.info("experiment_started id=%s name=%s", experiment_id, config.name)

            def progress_callback(tick: dict) -> None:
                broadcaster.publish(experiment_id, tick)

            result = ExperimentManager(config).run(progress_callback=progress_callback)
            if config.agent_mode == "environment_only":
                attach_environment_only_results(session, experiment, result)
            else:
                attach_results_to_experiment(session, experiment, result)
            experiment.status = "completed"
            logger.info("experiment_completed id=%s", experiment_id)
        except Exception:
            logger.exception("experiment_failed id=%s", experiment_id)
            experiment.status = "failed"
        session.commit()
    finally:
        session.close()
        _cancel_flags.pop(experiment_id, None)
        broadcaster.close(experiment_id)
