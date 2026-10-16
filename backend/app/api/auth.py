

from __future__ import annotations

from fastapi import Header

from app.core import bearer_auth



TOKEN_ENV = "AI_ENV_API_TOKEN"
OPT_OUT_ENV = "AI_ENV_API_ALLOW_NO_AUTH"


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    bearer_auth.verify_bearer(
        authorization, token_env=TOKEN_ENV, opt_out_env=OPT_OUT_ENV, subject="this API"
    )
