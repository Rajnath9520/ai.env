"""Phase 28 (Section 9) -- ExternalAgentAdapter contract (documentation
module).

Unlike `DemoPPOAgent`, ai.env does NOT implement an "ExternalAgentAdapter"
class that wraps a live external model -- that would mean ai.env owns or
drives the external agent's policy, which Section 9 explicitly forbids:
"the external agent, not ai.env, owns the model."

Instead, the external-agent path is a thin HTTP contract
(`app/api/routes/external_agent.py`, backed by `external_session.py`) that
any external agent process can call directly:

    POST /api/experiments/{id}/agent/reset  -> {observation, info, ...}
    POST /api/experiments/{id}/agent/step   -> {observation, reward,
                                                 terminated, truncated,
                                                 info, ...}
    GET  /api/experiments/{id}/agent/state  -> current connection/episode
                                                state, for the frontend

This module exists to document that contract in one place (mirroring the
shape of `AgentInterface.reset`/`act` for consistency, even though nothing
here is instantiated) rather than leaving it implicit in route code.
"""

from __future__ import annotations

EXTERNAL_AGENT_HTTP_CONTRACT = {
    "reset": "POST /api/experiments/{experiment_id}/agent/reset -> observation, info",
    "step": "POST /api/experiments/{experiment_id}/agent/step {action} -> observation, reward, terminated, truncated, info",
    "state": "GET /api/experiments/{experiment_id}/agent/state -> connection/episode/weakness state",
}
