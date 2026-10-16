"""Section 5 -- the backend's side of talking to a registered Local Agent
(`agent_app/main.py`) over plain HTTP. `ProjectService` uses this instead
of calling `ProjectScanner`/`CommandAdapter` directly whenever a project's
`agent_id` is set; see `project_service.py`'s module docstring for how the
branch is made.

Authentication is a shared bearer token (`AI_ENV_AGENT_TOKEN` on the
agent, `Agent.token` here), sent on every request. There is no TLS: over
plain HTTP the token itself crosses the network in the clear, so anything
beyond a trusted network needs TLS terminated in front of the agent
(docs/local-agent.md).
"""

from __future__ import annotations

import httpx


class AgentUnreachable(Exception):
    """Could not even reach the agent (network error, timeout, connection
    refused) -- as opposed to `AgentError`, where the agent responded but
    with a failure."""


class AgentError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"agent returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class AgentClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = httpx.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, headers=headers, **kwargs
            )
        except httpx.HTTPError as exc:
            raise AgentUnreachable(f"could not reach agent at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            raise AgentError(response.status_code, detail)
        return response

    def health(self) -> dict:
        return self._request("GET", "/health").json()

    def check_auth(self) -> None:
        """Raises `AgentError` (401/403/503) if this client's token isn't
        accepted -- `health()` alone can't tell, since it's unauthenticated."""
        self._request("GET", "/auth-check")

    def scan(self, root_path: str) -> dict:
        return self._request("POST", "/scan", json={"root_path": root_path}).json()

    def validate(self, root_path: str, config: dict) -> dict:
        return self._request("POST", "/validate", json={"root_path": root_path, "config": config}).json()

    def start_run(self, root_path: str, config: dict, run_number: int) -> str:
        body = {"root_path": root_path, "config": config, "run_number": run_number}
        return self._request("POST", "/runs", json=body).json()["agent_run_id"]

    def get_run(self, agent_run_id: str) -> dict:
        return self._request("GET", f"/runs/{agent_run_id}").json()

    def cancel_run(self, agent_run_id: str) -> None:
        self._request("POST", f"/runs/{agent_run_id}/cancel")

    def download_artifact(self, agent_run_id: str) -> tuple[bytes, str] | None:
        """Returns `(content, filename)`, or `None` if this run produced
        no artifact (not an error -- see `agent_app`'s `/artifact` route)."""
        try:
            response = self._request("GET", f"/runs/{agent_run_id}/artifact")
        except AgentError as exc:
            if exc.status_code == 404:
                return None
            raise
        filename = response.headers.get("X-Artifact-Filename", "artifact.bin")
        return response.content, filename
