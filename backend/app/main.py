

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import require_api_token
from app.api.routes.agents import router as agents_router
from app.api.routes.experiments import router as experiments_router
from app.api.routes.external_agent import router as external_agent_router
from app.api.routes.projects import router as projects_router
from app.api.routes.websocket import router as websocket_router

app = FastAPI(
    title="ai.env -- AI Environment Generator",
    description=(
        "Environment-generation and environment-analysis platform. Supports "
        "environment-only generation (default, no agent), external-agent "
        "interaction (agent-owned model, ai.env owns the environment), and "
        "an optional Demo Agent -- PPO for demonstration purposes. Also "
        "supports registering a user's own AI project (Mode B) and "
        "orchestrating it through the ai-env.yaml training contract, "
        "either directly or through a registered Local Agent (Section 5). "
        "Mode B's command-executing routes (/api/projects, /api/agents) "
        "require a bearer token (AI_ENV_API_TOKEN); Mode A's GridWorld/PPO "
        "routes do not -- see docs/deployment.md."
    ),
    version="0.31.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router, dependencies=[Depends(require_api_token)])
app.include_router(experiments_router)
app.include_router(external_agent_router)
app.include_router(projects_router, dependencies=[Depends(require_api_token)])
app.include_router(websocket_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
