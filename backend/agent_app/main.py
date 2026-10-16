

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import Response

from agent_app.auth import auth_disabled_explicitly, configured_token, require_auth
from agent_app.run_manager import RunManager
from agent_app.schemas import (
    AgentRunRead,
    CancelRead,
    HealthRead,
    ScanRequest,
    StartRunRead,
    StartRunRequest,
    ValidateRequest,
)
from app.ml.project.adapters.base import ValidationResult
from app.ml.project.adapters.registry import build_adapter
from app.ml.project.config import AIEnvConfig
from app.ml.project.scanner import ProjectScanner

AGENT_NAME = os.environ.get("AI_ENV_AGENT_NAME", "local-agent")
AGENT_VERSION = "0.1.0"
AGENT_OUTPUT_ROOT = os.environ.get("AI_ENV_AGENT_OUTPUT_ROOT", "./agent-data")

app = FastAPI(title="ai-env-agent", version=AGENT_VERSION)
run_manager = RunManager(AGENT_OUTPUT_ROOT)


@app.get("/health", response_model=HealthRead)
def health() -> dict:
    """Deliberately unauthenticated (see `agent_app/auth.py`): it's how a
    backend tells "unreachable" apart from "reachable, wrong token"."""
    return {
        "status": "ok",
        "agent_name": AGENT_NAME,
        "version": AGENT_VERSION,
        "auth_required": not auth_disabled_explicitly(),
        "token_configured": configured_token() is not None,
    }


@app.get("/auth-check", dependencies=[Depends(require_auth)], response_model=CancelRead)
def auth_check() -> dict:
    """Succeeds only if the caller's token is valid -- lets the backend
    verify credentials without scanning or running anything."""
    return {"status": "ok"}


@app.post("/scan", response_model=dict, dependencies=[Depends(require_auth)])
def scan(payload: ScanRequest) -> dict:
    try:
        result = ProjectScanner().scan(payload.root_path)
    except NotADirectoryError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/validate", response_model=dict, dependencies=[Depends(require_auth)])
def validate(payload: ValidateRequest) -> dict:
    try:
        config = AIEnvConfig.from_dict(payload.config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    adapter = build_adapter(config, payload.root_path)
    result: ValidationResult = adapter.validate_project()
    return result.to_dict()


@app.post("/runs", response_model=StartRunRead, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_auth)])
def start_run(payload: StartRunRequest) -> dict:
    try:
        AIEnvConfig.from_dict(payload.config)  # fail fast on a malformed contract, before spawning a thread
    except (ValueError, TypeError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    agent_run_id = run_manager.start_run(
        root_path=payload.root_path,
        config_dict=payload.config,
        run_number=payload.run_number,
    )
    return {"agent_run_id": agent_run_id, "status": "running_training"}


@app.get("/runs/{run_id}", response_model=AgentRunRead, dependencies=[Depends(require_auth)])
def get_run(run_id: str) -> dict:
    run = run_manager.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"unknown run {run_id!r}")
    return {
        "status": run.status,
        "train_result": run.train_result,
        "evaluation_result": run.evaluation_result,
        "error": run.error,
        "has_artifact": run.artifact_path is not None,
    }


@app.post("/runs/{run_id}/cancel", response_model=CancelRead, dependencies=[Depends(require_auth)])
def cancel_run(run_id: str) -> dict:
    if not run_manager.cancel(run_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"unknown run {run_id!r}")
    return {"status": "ok"}


@app.get("/runs/{run_id}/artifact", dependencies=[Depends(require_auth)])
def get_artifact(run_id: str) -> Response:
    run = run_manager.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"unknown run {run_id!r}")
    if run.artifact_path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="this run has no artifact (yet, or ever)")
    path = Path(run.artifact_path)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="the artifact no longer exists on disk")
    return Response(
        content=path.read_bytes(),
        media_type="application/octet-stream",
        headers={"X-Artifact-Filename": path.name},
    )
