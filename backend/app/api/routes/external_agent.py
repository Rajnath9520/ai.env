

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.ml.agents.external_session import ExternalAgentError, external_agent_manager
from app.ml.experiment.config import ExperimentConfig
from app.models.orm import Experiment
from app.schemas.experiment import (
    AgentResetResponse,
    AgentStateRead,
    AgentStepRequest,
    AgentStepResponse,
)

import yaml

router = APIRouter(prefix="/api/experiments", tags=["external-agent"])


def _get_experiment_or_404(db: Session, experiment_id: int) -> Experiment:
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Experiment {experiment_id} not found")
    return experiment


def _require_external_mode(experiment: Experiment) -> None:
    if experiment.agent_mode != "external":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"Experiment {experiment.id} has agent_mode={experiment.agent_mode!r}; "
                "the /agent/* endpoints are only available for agent_mode='external'"
            ),
        )


@router.post("/{experiment_id}/agent/reset", response_model=AgentResetResponse)
def reset_agent(experiment_id: int, db: Session = Depends(get_db)):
    experiment = _get_experiment_or_404(db, experiment_id)
    _require_external_mode(experiment)

    config = ExperimentConfig.from_dict(yaml.safe_load(experiment.config_yaml))
    session = external_agent_manager.get_or_create(experiment_id, config)
    obs, info = session.reset(db)

    return AgentResetResponse(
        experiment_id=experiment_id,
        environment_id=session.environment_key or "",
        episode=session.episode_index,
        observation=obs.tolist(),
        info={k: v for k, v in info.items() if k != "generation_attempts_used" or v is not None},
    )


@router.post("/{experiment_id}/agent/step", response_model=AgentStepResponse)
def step_agent(experiment_id: int, payload: AgentStepRequest, db: Session = Depends(get_db)):
    experiment = _get_experiment_or_404(db, experiment_id)
    _require_external_mode(experiment)

    config = ExperimentConfig.from_dict(yaml.safe_load(experiment.config_yaml))
    session = external_agent_manager.get_or_create(experiment_id, config)
    try:
        result = session.step(db, payload.action)
    except ExternalAgentError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return AgentStepResponse(
        experiment_id=experiment_id,
        environment_id=session.environment_key or "",
        episode=session.episode_index,
        step=session.step_index,
        observation=result["observation"].tolist(),
        reward=result["reward"],
        terminated=result["terminated"],
        truncated=result["truncated"],
        info=result["info"],
        episode_summary=result["episode_summary"],
        weakness=result["weakness"],
        next_environment=result["next_environment"],
    )


@router.get("/{experiment_id}/agent/state", response_model=AgentStateRead)
def get_agent_state(experiment_id: int, db: Session = Depends(get_db)):
    experiment = _get_experiment_or_404(db, experiment_id)
    _require_external_mode(experiment)

    session = external_agent_manager.get(experiment_id)
    if session is None:
        return AgentStateRead(
            experiment_id=experiment_id,
            agent_mode="external",
            connected=False,
            status=experiment.status,
        )
    return AgentStateRead(**session.state_dict())
