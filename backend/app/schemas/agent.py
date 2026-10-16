"""Section 5 -- request/response schemas for `/api/agents/*`."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

AgentStatus = Literal["connected", "unreachable", "unauthorized", "unknown"]


class AgentRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    base_url: str = Field(min_length=1, description="e.g. http://192.168.1.42:8765")
    # SecretStr so the token can't leak through repr()/logging of this
    # request object; it's unwrapped exactly once, in the route.
    token: SecretStr | None = Field(
        default=None, description="the agent's AI_ENV_AGENT_TOKEN; write-only, never returned"
    )


class AgentTokenUpdate(BaseModel):
    token: SecretStr | None = None


class AgentRead(BaseModel):
    """Deliberately has no `token` field -- only whether one is stored."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    base_url: str
    has_token: bool
    status: AgentStatus
    last_seen_at: datetime | None
    created_at: datetime
