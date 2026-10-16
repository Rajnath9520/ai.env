

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, status


def configured_token(token_env: str) -> str | None:
    return os.environ.get(token_env, "") or None


def auth_disabled_explicitly(opt_out_env: str) -> bool:
    return os.environ.get(opt_out_env) == "1"


def verify_bearer(authorization: str | None, *, token_env: str, opt_out_env: str, subject: str) -> None:
    """Raises `HTTPException` unless `authorization` carries the right
    bearer token. `subject` names what's being protected, for the 503
    message ("this agent" / "this API")."""
    if auth_disabled_explicitly(opt_out_env):
        return

    expected = configured_token(token_env)
    if expected is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"{subject} has no {token_env} configured and refuses to run unauthenticated; "
                f"set a token (or {opt_out_env}=1 to opt out explicitly)"
            ),
        )

    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
