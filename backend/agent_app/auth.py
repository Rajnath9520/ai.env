

from __future__ import annotations

from fastapi import Header

from app.core import bearer_auth

TOKEN_ENV = "AI_ENV_AGENT_TOKEN"
OPT_OUT_ENV = "AI_ENV_AGENT_ALLOW_NO_AUTH"


def configured_token() -> str | None:
    return bearer_auth.configured_token(TOKEN_ENV)


def auth_disabled_explicitly() -> bool:
    return bearer_auth.auth_disabled_explicitly(OPT_OUT_ENV)


def require_auth(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency guarding every agent route except `/health`."""
    bearer_auth.verify_bearer(
        authorization, token_env=TOKEN_ENV, opt_out_env=OPT_OUT_ENV, subject="this agent"
    )
