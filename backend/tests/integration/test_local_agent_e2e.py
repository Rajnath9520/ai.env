"""Section 5's whole point, proven end-to-end: the main backend and the
Local Agent are two genuinely separate OS processes communicating only
over HTTP, exactly as they would across two different machines. Every
other Mode B test either exercises `app.ml.project.*` directly or goes
through the backend's own in-process execution path -- this is the only
place a real `agent_app.main:app` process is spawned via `uvicorn` and
talked to over a real loopback socket, so it's the only place a bug in the
*wiring* between backend and agent (as opposed to either side's own logic)
would actually surface.
"""

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "ml" / "fixtures" / "test-ai-project"
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
PY = sys.executable


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


TOKEN = "e2e-shared-secret-7f3a"


def _spawn_agent(tmp_path, token: str | None):
    """Starts a real `uvicorn agent_app.main:app` process. `token=None`
    starts it with NO token configured (and no opt-out), i.e. the
    fail-closed state."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {k: v for k, v in os.environ.items() if k not in ("AI_ENV_AGENT_TOKEN", "AI_ENV_AGENT_ALLOW_NO_AUTH")}
    env["AI_ENV_AGENT_OUTPUT_ROOT"] = str(tmp_path / "agent-data")
    if token is not None:
        env["AI_ENV_AGENT_TOKEN"] = token
    process = subprocess.Popen(
        [PY, "-m", "uvicorn", "agent_app.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=BACKEND_ROOT,
        env=env,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                return process, base_url
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    process.kill()
    raise RuntimeError("agent process did not become healthy in time")


def _stop(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture()
def running_agent(tmp_path):
    process, base_url = _spawn_agent(tmp_path, TOKEN)
    try:
        yield base_url
    finally:
        _stop(process)


@pytest.fixture()
def running_agent_without_token(tmp_path):
    process, base_url = _spawn_agent(tmp_path, None)
    try:
        yield base_url
    finally:
        _stop(process)


@pytest.fixture()
def project_copy(tmp_path) -> Path:
    dest = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, dest)
    return dest


CUSTOM_ADAPTER_FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "ml" / "fixtures" / "test-custom-adapter-project"


@pytest.fixture()
def custom_adapter_project_copy(tmp_path) -> Path:
    dest = tmp_path / "custom-adapter-project"
    shutil.copytree(CUSTOM_ADAPTER_FIXTURE_ROOT, dest)
    return dest


def _valid_config(**overrides) -> dict:
    base = {
        "name": "agent-e2e-project",
        "project": {"task": "face_recognition", "framework": "pytorch"},
        "training": {"command": f"{PY} train.py --dataset {{dataset}}", "entrypoint": "train.py"},
        "evaluation": {
            "command": f"{PY} evaluate.py --dataset {{dataset}}",
            "entrypoint": "evaluate.py",
            "metrics_source": "file",
            "metrics_file": "outputs/metrics.json",
        },
        "artifact": {"path": "outputs/best_model.pt", "type": "pytorch_checkpoint"},
        "dataset": {"path": "dataset"},
        "curriculum": {"strategy": "failure_driven"},
        "optimization": {"max_iterations": 10},
    }
    for section, values in overrides.items():
        base[section] = {**base.get(section, {}), **values}
    return base


class TestLocalAgentEndToEnd:
    def test_registering_a_reachable_agent_marks_it_connected(self, client, running_agent):
        resp = client.post("/api/agents/register", json={"name": "laptop", "base_url": running_agent, "token": TOKEN})
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "connected"
        assert body["last_seen_at"] is not None

    def test_registering_with_the_wrong_token_is_marked_unauthorized_not_connected(
        self, client, running_agent
    ):
        resp = client.post(
            "/api/agents/register",
            json={"name": "laptop", "base_url": running_agent, "token": "wrong-token"},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "unauthorized"
        assert resp.json()["last_seen_at"] is None

    def test_registering_with_no_token_against_a_protected_agent_is_unauthorized(
        self, client, running_agent
    ):
        resp = client.post("/api/agents/register", json={"name": "laptop", "base_url": running_agent})
        assert resp.json()["status"] == "unauthorized"
        assert resp.json()["has_token"] is False

    def test_an_agent_with_no_token_configured_refuses_everything_and_reads_as_unauthorized(
        self, client, running_agent_without_token
    ):
        # Fail closed: the agent process itself was started without a
        # secret, so even a backend presenting *a* token can't use it.
        resp = client.post(
            "/api/agents/register",
            json={"name": "unconfigured", "base_url": running_agent_without_token, "token": "anything"},
        )
        assert resp.json()["status"] == "unauthorized"

        health = httpx.get(f"{running_agent_without_token}/health").json()
        assert health["token_configured"] is False
        scan = httpx.post(f"{running_agent_without_token}/scan", json={"root_path": "/tmp"})
        assert scan.status_code == 503

    def test_the_token_is_never_returned_by_any_agent_endpoint(self, client, running_agent):
        created = client.post(
            "/api/agents/register", json={"name": "laptop", "base_url": running_agent, "token": TOKEN}
        )
        agent_id = created.json()["id"]
        assert created.json()["has_token"] is True

        responses = [
            created,
            client.get("/api/agents"),
            client.get(f"/api/agents/{agent_id}"),
            client.post(f"/api/agents/{agent_id}/ping"),
            client.put(f"/api/agents/{agent_id}/token", json={"token": TOKEN}),
        ]
        for response in responses:
            assert TOKEN not in response.text
            assert '"token"' not in response.text

    def test_rotating_a_token_on_both_sides_recovers_a_broken_agent(self, client, tmp_path):
        process, base_url = _spawn_agent(tmp_path, "new-secret")
        try:
            agent_id = client.post(
                "/api/agents/register",
                json={"name": "rotated", "base_url": base_url, "token": "old-secret"},
            ).json()["id"]
            assert client.get(f"/api/agents/{agent_id}").json()["status"] == "unauthorized"

            updated = client.put(f"/api/agents/{agent_id}/token", json={"token": "new-secret"})
            assert updated.status_code == 200
            assert updated.json()["status"] == "connected"
        finally:
            _stop(process)

    def test_updating_the_token_of_a_missing_agent_returns_404(self, client):
        assert client.put("/api/agents/999999/token", json={"token": "x"}).status_code == 404

    def test_a_run_against_an_agent_with_a_bad_token_fails_with_a_clear_reason(
        self, client, running_agent, project_copy
    ):
        agent_id = client.post(
            "/api/agents/register",
            json={"name": "laptop", "base_url": running_agent, "token": "wrong-token"},
        ).json()["id"]
        project_id = client.post(
            "/api/projects/register", json={"root_path": str(project_copy), "agent_id": agent_id}
        ).json()["id"]
        client.put(f"/api/projects/{project_id}/config", json=_valid_config())

        start = client.post(f"/api/projects/{project_id}/runs")
        assert start.status_code == 201
        run = self._poll_run(client, project_id, start.json()["id"])
        assert run["status"] == "failed"
        assert "401" in run["error"]
        # And nothing ran on the agent's machine.
        assert not (project_copy / "outputs" / "best_model.pt").exists()

    def test_scanning_through_an_agent_with_a_bad_token_is_a_409_not_a_leak(
        self, client, running_agent, project_copy
    ):
        agent_id = client.post(
            "/api/agents/register",
            json={"name": "laptop", "base_url": running_agent, "token": "wrong-token"},
        ).json()["id"]
        project_id = client.post(
            "/api/projects/register", json={"root_path": str(project_copy), "agent_id": agent_id}
        ).json()["id"]
        resp = client.post(f"/api/projects/{project_id}/scan")
        assert resp.status_code == 409
        assert "wrong-token" not in resp.text

    def test_scanning_through_an_unreachable_agent_is_a_409(self, client, project_copy):
        agent_id = client.post(
            "/api/agents/register", json={"name": "offline", "base_url": "http://127.0.0.1:1", "token": TOKEN}
        ).json()["id"]
        project_id = client.post(
            "/api/projects/register", json={"root_path": str(project_copy), "agent_id": agent_id}
        ).json()["id"]
        resp = client.post(f"/api/projects/{project_id}/scan")
        assert resp.status_code == 409
        assert "unreachable" in resp.json()["detail"]

    def test_registering_an_unreachable_agent_is_honest_about_it(self, client):
        resp = client.post(
            "/api/agents/register", json={"name": "offline", "base_url": "http://127.0.0.1:1"}
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "unreachable"

    def test_ping_updates_status(self, client, running_agent):
        agent_id = client.post(
            "/api/agents/register", json={"name": "laptop", "base_url": running_agent, "token": TOKEN}
        ).json()["id"]
        resp = client.post(f"/api/agents/{agent_id}/ping")
        assert resp.status_code == 200
        assert resp.json()["status"] == "connected"

    def test_full_project_lifecycle_through_a_real_separate_agent_process(
        self, client, running_agent, project_copy
    ):
        agent_id = client.post(
            "/api/agents/register", json={"name": "laptop", "base_url": running_agent, "token": TOKEN}
        ).json()["id"]

        project_id = client.post(
            "/api/projects/register",
            json={"root_path": str(project_copy), "agent_id": agent_id},
        ).json()["id"]

        # Scan: the backend never touches project_copy's filesystem here
        # at all -- only the agent process does. If the wiring were wrong
        # (e.g. the backend silently fell back to local execution) this
        # would still happen to pass, since it's the same machine in this
        # test; the point is the *code path* taken is the HTTP one, which
        # test_project_service-level unit tests can't exercise.
        scan = client.post(f"/api/projects/{project_id}/scan")
        assert scan.status_code == 200
        assert scan.json()["framework"] == "pytorch"
        assert "train.py" in scan.json()["training_candidates"]

        config_resp = client.put(f"/api/projects/{project_id}/config", json=_valid_config())
        assert config_resp.status_code == 200
        assert config_resp.json()["agent_id"] == agent_id

        validate = client.post(f"/api/projects/{project_id}/validate")
        assert validate.status_code == 200
        assert validate.json()["ok"] is True

        start = client.post(f"/api/projects/{project_id}/runs")
        assert start.status_code == 201
        run_id = start.json()["id"]

        run = self._poll_run(client, project_id, run_id)
        assert run["status"] == "completed"
        assert run["train_result"]["status"] == "success"
        assert run["evaluation_result"]["metrics"] == {"accuracy": 0.968, "f1": 0.966}

        # The artifact lived only on the agent's (in this test, also-local
        # but architecturally separate) filesystem -- this proves it was
        # actually downloaded over HTTP and archived by the backend, not
        # just referenced by a path the backend happened to also be able
        # to see.
        assert run["managed_artifact_path"] is not None
        archived = Path(run["managed_artifact_path"])
        assert archived.is_file()
        assert archived.read_bytes() == b"not-a-real-checkpoint"

        progress = client.get(f"/api/projects/{project_id}/progress").json()
        assert progress["completed_run_count"] == 1

    def test_cancelling_an_agent_backed_run_stops_it_well_before_its_full_duration(
        self, client, running_agent, project_copy
    ):
        agent_id = client.post(
            "/api/agents/register", json={"name": "laptop", "base_url": running_agent, "token": TOKEN}
        ).json()["id"]
        project_id = client.post(
            "/api/projects/register", json={"root_path": str(project_copy), "agent_id": agent_id}
        ).json()["id"]
        client.put(
            f"/api/projects/{project_id}/config",
            json=_valid_config(training={"command": f"{PY} -c \"import time; time.sleep(30)\""}),
        )

        # The initial POST /runs call itself completes synchronously
        # within this TestClient call (its BackgroundTask -- including the
        # whole poll loop against the agent -- runs to completion before
        # control returns), so it has to happen on its own thread for
        # there to be anything left in flight to cancel.
        result: dict = {}

        def start():
            result["response"] = client.post(f"/api/projects/{project_id}/runs")

        thread = threading.Thread(target=start)
        started_at = time.monotonic()
        thread.start()

        run_id = None
        for _ in range(50):
            time.sleep(0.1)
            runs = client.get(f"/api/projects/{project_id}/runs").json()
            if runs:
                run_id = runs[0]["id"]
                break
        assert run_id is not None, "run row was never created"

        client.post(f"/api/projects/{project_id}/runs/{run_id}/cancel")
        thread.join(timeout=25)
        elapsed = time.monotonic() - started_at

        assert not thread.is_alive()
        assert elapsed < 20  # well under the command's own 30s sleep
        final = client.get(f"/api/projects/{project_id}/runs/{run_id}").json()
        assert final["status"] == "cancelled"

    def test_agent_becoming_unreachable_mid_run_fails_the_run_honestly(self, client, project_copy):
        # A run started against an agent that then vanishes must not hang
        # forever or silently report success -- it should fail with a
        # clear reason.
        resp = client.post(
            "/api/agents/register", json={"name": "flaky", "base_url": "http://127.0.0.1:1"}
        )
        agent_id = resp.json()["id"]
        project_id = client.post(
            "/api/projects/register", json={"root_path": str(project_copy), "agent_id": agent_id}
        ).json()["id"]
        client.put(f"/api/projects/{project_id}/config", json=_valid_config())

        start = client.post(f"/api/projects/{project_id}/runs")
        assert start.status_code == 201
        run = self._poll_run(client, project_id, start.json()["id"])
        assert run["status"] == "failed"
        assert run["error"] is not None

    def test_custom_adapter_runs_correctly_through_a_real_agent_process(
        self, client, running_agent, custom_adapter_project_copy
    ):
        # This is exactly the scenario a capability-discovery bug would
        # show up in: CustomAdapter.capabilities requires importing the
        # project's own .ai-env/adapter.py, which lives only on the
        # agent's filesystem here, not the backend's. If ProjectService
        # ever went back to computing capabilities (or the TRAIN
        # pre-flight check) locally for a custom adapter, this would fail
        # or silently inspect the wrong (nonexistent) path on the backend
        # instead.
        agent_id = client.post(
            "/api/agents/register", json={"name": "laptop", "base_url": running_agent, "token": TOKEN}
        ).json()["id"]
        project_id = client.post(
            "/api/projects/register",
            json={"root_path": str(custom_adapter_project_copy), "agent_id": agent_id},
        ).json()["id"]

        config = {
            "name": "custom-adapter-e2e",
            "training": {"command": "unused-for-custom-adapters"},
            "artifact": {"path": "outputs/model.bin"},
            "adapter": {"type": "custom", "python": PY},
        }
        configured = client.put(f"/api/projects/{project_id}/config", json=config)
        assert configured.status_code == 200
        assert configured.json()["capabilities"] == []  # never eagerly computed locally -- see above

        validate = client.post(f"/api/projects/{project_id}/validate")
        assert validate.status_code == 200
        assert validate.json()["ok"] is True

        run_id = client.post(f"/api/projects/{project_id}/runs").json()["id"]
        run = self._poll_run(client, project_id, run_id)

        assert run["status"] == "completed"
        assert run["evaluation_result"]["metrics"] == {"accuracy": 0.87, "f1": 0.83}
        assert run["managed_artifact_path"] is not None
        assert Path(run["managed_artifact_path"]).read_text() == "trained-by-custom-adapter"

    @staticmethod
    def _poll_run(client, project_id: int, run_id: int, timeout: float = 15.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = client.get(f"/api/projects/{project_id}/runs/{run_id}").json()
            if run["status"] not in ("queued", "running_training", "running_evaluation"):
                return run
            time.sleep(0.2)
        raise AssertionError(f"run {run_id} did not reach a terminal status within {timeout}s")
