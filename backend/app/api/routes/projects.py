

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.orm import Project, ProjectRun
from app.schemas.project import (
    ProjectAgentAssignment,
    ProjectConfigRead,
    ProjectConfigUpdate,
    ProjectDetailRead,
    ProjectProgressRead,
    ProjectRead,
    ProjectRegisterRequest,
    ProjectRunRead,
    ProjectValidationRead,
    ScanResultRead,
)
from app.services.agent_service import AgentNotFound
from app.services.project_service import ProjectNotFound, ProjectRunNotFound, ProjectService, ProjectStateError

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _service(db: Session = Depends(get_db)) -> ProjectService:
    return ProjectService(db)


def _get_project_or_404(service: ProjectService, project_id: int) -> Project:
    try:
        return service.get(project_id)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/register", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def register_project(payload: ProjectRegisterRequest, service: ProjectService = Depends(_service)):
    return service.register(payload)


@router.get("", response_model=list[ProjectRead])
def list_projects(service: ProjectService = Depends(_service)):
    return service.list_all()


@router.get("/{project_id}", response_model=ProjectDetailRead)
def get_project(project_id: int, service: ProjectService = Depends(_service)):
    return _get_project_or_404(service, project_id)


@router.put("/{project_id}/agent", response_model=ProjectRead)
def assign_agent(project_id: int, payload: ProjectAgentAssignment, service: ProjectService = Depends(_service)):
    try:
        return service.assign_agent(project_id, payload.agent_id)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AgentNotFound as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/{project_id}/scan", response_model=ScanResultRead)
def scan_project(project_id: int, service: ProjectService = Depends(_service)):
    try:
        result = service.scan(project_id)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ProjectStateError as exc:
        # An assigned agent that's unreachable, rejects our token, or
        # can't scan the path -- a conflict with the project's current
        # state, not a server fault.
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return result.to_dict()


@router.get("/{project_id}/config", response_model=ProjectConfigRead)
def get_project_config(project_id: int, service: ProjectService = Depends(_service)):
    try:
        config = service.get_config(project_id)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ProjectStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return config.to_dict()


@router.put("/{project_id}/config", response_model=ProjectRead)
def update_project_config(
    project_id: int, payload: ProjectConfigUpdate, service: ProjectService = Depends(_service)
):
    try:
        return service.update_config(project_id, payload)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/{project_id}/validate", response_model=ProjectValidationRead)
def validate_project(project_id: int, service: ProjectService = Depends(_service)):
    try:
        result = service.validate(project_id)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ProjectStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return result.to_dict()


def _get_run_or_404(service: ProjectService, project_id: int, run_id: int) -> ProjectRun:
    try:
        return service.get_run(project_id, run_id)
    except ProjectRunNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{project_id}/runs", response_model=ProjectRunRead, status_code=status.HTTP_201_CREATED)
def start_run(project_id: int, background_tasks: BackgroundTasks, service: ProjectService = Depends(_service)):
    try:
        return service.start_run(project_id, background_tasks)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ProjectStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/{project_id}/runs", response_model=list[ProjectRunRead])
def list_runs(project_id: int, service: ProjectService = Depends(_service)):
    try:
        return service.list_runs(project_id)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{project_id}/runs/{run_id}", response_model=ProjectRunRead)
def get_run(project_id: int, run_id: int, service: ProjectService = Depends(_service)):
    return _get_run_or_404(service, project_id, run_id)


@router.post("/{project_id}/runs/{run_id}/cancel", response_model=ProjectRunRead)
def cancel_run(project_id: int, run_id: int, service: ProjectService = Depends(_service)):
    try:
        return service.cancel_run(project_id, run_id)
    except ProjectRunNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{project_id}/progress", response_model=ProjectProgressRead)
def get_progress(project_id: int, service: ProjectService = Depends(_service)):
    try:
        return service.get_progress(project_id)
    except ProjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
