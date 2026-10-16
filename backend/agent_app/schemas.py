from __future__ import annotations

from pydantic import BaseModel


class HealthRead(BaseModel):
    status: str
    agent_name: str
    version: str
    auth_required: bool
    token_configured: bool


class ScanRequest(BaseModel):
    root_path: str


class ValidateRequest(BaseModel):
    root_path: str
    config: dict  # the ai-env.yaml contract as a dict -- validated by AIEnvConfig.from_dict itself


class StartRunRequest(BaseModel):
    root_path: str
    config: dict
    run_number: int


class StartRunRead(BaseModel):
    agent_run_id: str
    status: str


class AgentRunRead(BaseModel):
    status: str
    train_result: dict | None
    evaluation_result: dict | None
    error: str | None
    has_artifact: bool


class CancelRead(BaseModel):
    status: str
