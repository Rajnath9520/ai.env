"""Phase 17 -- FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.experiments import router as experiments_router
from app.api.routes.external_agent import router as external_agent_router
from app.api.routes.websocket import router as websocket_router

app = FastAPI(
    title="ai.env -- AI Environment Generator",
    description=(
        "Environment-generation and environment-analysis platform. Supports "
        "environment-only generation (default, no agent), external-agent "
        "interaction (agent-owned model, ai.env owns the environment), and "
        "an optional Demo Agent -- PPO for demonstration purposes."
    ),
    version="0.28.0",
)

app.include_router(experiments_router)
app.include_router(external_agent_router)
app.include_router(websocket_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
