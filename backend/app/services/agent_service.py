"""Section 5 -- registering and health-checking Local Agents
(`agent_app/main.py` processes). Deliberately separate from
`ProjectService`: an `Agent` is a machine ai.env can talk to, independent
of any particular project, and several projects may share one.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.orm import Agent
from app.services.agent_client import AgentClient, AgentError, AgentUnreachable


class AgentNotFound(Exception):
    def __init__(self, agent_id: int):
        super().__init__(f"Agent {agent_id} not found")
        self.agent_id = agent_id


def client_for_agent(agent: Agent, timeout: float = 30.0) -> AgentClient:
    """The one place an `Agent` row becomes an `AgentClient`, so every
    caller presents the stored token the same way."""
    return AgentClient(agent.base_url, token=agent.token, timeout=timeout)


class AgentService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def register(self, name: str, base_url: str, token: str | None = None) -> Agent:
        """Registering always creates the row, even if the agent turns out
        to be unreachable or rejects the token right now (a machine that's
        briefly offline is still worth remembering) -- `status` honestly
        reflects whichever of those actually happened, via an immediate
        check."""
        agent = Agent(name=name, base_url=base_url, token=token or None, status="unknown")
        self.session.add(agent)
        self.session.commit()
        self.session.refresh(agent)
        self._ping(agent)
        return agent

    def list_all(self) -> list[Agent]:
        return self.session.query(Agent).order_by(Agent.id).all()

    def get(self, agent_id: int) -> Agent:
        agent = self.session.get(Agent, agent_id)
        if agent is None:
            raise AgentNotFound(agent_id)
        return agent

    def ping(self, agent_id: int) -> Agent:
        agent = self.get(agent_id)
        self._ping(agent)
        return agent

    def update_token(self, agent_id: int, token: str | None) -> Agent:
        """Rotating a token on the agent side must be mirrored here, or
        every later call 401s -- so this exists, and re-checks immediately."""
        agent = self.get(agent_id)
        agent.token = token or None
        self.session.commit()
        self._ping(agent)
        return agent

    def _ping(self, agent: Agent) -> None:
        """Two steps, because `/health` is deliberately unauthenticated:
        it proves the agent is reachable, but only `/auth-check` proves the
        stored token is actually accepted."""
        client = client_for_agent(agent, timeout=5.0)
        try:
            client.health()
        except (AgentUnreachable, AgentError):
            agent.status = "unreachable"
        else:
            try:
                client.check_auth()
            except AgentUnreachable:
                agent.status = "unreachable"
            except AgentError:
                # 401 (wrong/missing token), 403, or 503 (the agent itself
                # has no token configured and refuses everything).
                agent.status = "unauthorized"
            else:
                agent.status = "connected"
                agent.last_seen_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(agent)
