

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.agent import AgentRead, AgentRegisterRequest, AgentTokenUpdate
from app.services.agent_service import AgentNotFound, AgentService

router = APIRouter(prefix="/api/agents", tags=["agents"])


def _service(db: Session = Depends(get_db)) -> AgentService:
    return AgentService(db)


@router.post("/register", response_model=AgentRead, status_code=status.HTTP_201_CREATED)
def register_agent(payload: AgentRegisterRequest, service: AgentService = Depends(_service)):
    token = payload.token.get_secret_value() if payload.token else None
    return service.register(payload.name, payload.base_url, token)


@router.get("", response_model=list[AgentRead])
def list_agents(service: AgentService = Depends(_service)):
    return service.list_all()


@router.get("/{agent_id}", response_model=AgentRead)
def get_agent(agent_id: int, service: AgentService = Depends(_service)):
    try:
        return service.get(agent_id)
    except AgentNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{agent_id}/ping", response_model=AgentRead)
def ping_agent(agent_id: int, service: AgentService = Depends(_service)):
    try:
        return service.ping(agent_id)
    except AgentNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put("/{agent_id}/token", response_model=AgentRead)
def update_agent_token(agent_id: int, payload: AgentTokenUpdate, service: AgentService = Depends(_service)):
    """Mirror a token rotated on the agent side (or clear it, with null)."""
    try:
        token = payload.token.get_secret_value() if payload.token else None
        return service.update_token(agent_id, token)
    except AgentNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
