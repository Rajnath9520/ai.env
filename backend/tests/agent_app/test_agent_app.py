import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_app.main import app

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "ml" / "fixtures" / "test-ai-project"
PY = sys.executable

TOKEN = "test-token-abc123"

# Every existing test below talks to the agent the way a real backend
# does: with the right bearer token. Auth itself is covered by TestAuth.
client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
anonymous_client = TestClient(app)


@pytest.fixture(autouse=True)
def _agent_token(monkeypatch):
    monkeypatch.setenv("AI_ENV_AGENT_TOKEN", TOKEN)
    monkeypatch.delenv("AI_ENV_AGENT_ALLOW_NO_AUTH", raising=False)


@pytest.fixture()
def project(tmp_path) -> Path:
    dest = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, dest)
    return dest


def _config(**overrides) -> dict:
    base = {
        "project": {"name": "agent-test"},
        "training": {"command": f"{PY} train.py --dataset {{dataset}}"},
        "evaluation": {
            "command": f"{PY} evaluate.py --dataset {{dataset}}",
            "metrics_source": "file",
            "metrics_file": "outputs/metrics.json",
        },
        "artifact": {"path": "outputs/best_model.pt"},
        "dataset": {"path": "dataset"},
    }
    for section, values in overrides.items():
        base[section] = {**base.get(section, {}), **values}
    return base


def _poll_until_terminal(run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] not in ("running_training", "running_evaluation"):
            return run
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not reach a terminal status within {timeout}s")


class TestHealth:
    def test_health_reports_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "agent_name" in body
        assert "version" in body
        assert body["auth_required"] is True
        assert body["token_configured"] is True


class TestScan:
    def test_scan_detects_framework_and_candidates(self, project):
        resp = client.post("/scan", json={"root_path": str(project)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["framework"] == "pytorch"
        assert "train.py" in body["training_candidates"]

    def test_scan_of_missing_path_returns_422(self, tmp_path):
        resp = client.post("/scan", json={"root_path": str(tmp_path / "nope")})
        assert resp.status_code == 422


class TestValidate:
    def test_validate_reports_ok_for_a_correct_config(self, project):
        resp = client.post("/validate", json={"root_path": str(project), "config": _config()})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_validate_flags_path_traversal(self, project):
        resp = client.post(
            "/validate",
            json={"root_path": str(project), "config": _config(artifact={"path": "../../outside.pt"})},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert any(i["field"] == "artifact.path" for i in body["issues"])

    def test_validate_with_a_malformed_config_returns_422(self, project):
        resp = client.post(
            "/validate", json={"root_path": str(project), "config": {"training": {"command": "x"}}}
        )
        assert resp.status_code == 422  # missing project.name


class TestRuns:
    def test_full_run_trains_and_evaluates(self, project):
        start = client.post("/runs", json={"root_path": str(project), "config": _config(), "run_number": 1})
        assert start.status_code == 201
        run_id = start.json()["agent_run_id"]
        assert start.json()["status"] == "running_training"

        final = _poll_until_terminal(run_id)
        assert final["status"] == "completed"
        assert final["train_result"]["status"] == "success"
        assert final["evaluation_result"]["metrics"] == {"accuracy": 0.968, "f1": 0.966}
        assert final["has_artifact"] is True

    def test_malformed_config_is_rejected_before_starting_a_thread(self, project):
        resp = client.post(
            "/runs", json={"root_path": str(project), "config": {"training": {"command": "x"}}, "run_number": 1}
        )
        assert resp.status_code == 422

    def test_get_unknown_run_returns_404(self):
        assert client.get("/runs/does-not-exist").status_code == 404

    def test_cancel_unknown_run_returns_404(self):
        assert client.post("/runs/does-not-exist/cancel").status_code == 404

    def test_cancelling_a_slow_run_stops_it_early(self, project):
        config = _config(training={"command": f"{PY} -c \"import time; time.sleep(20)\""})
        start = client.post("/runs", json={"root_path": str(project), "config": config, "run_number": 1})
        run_id = start.json()["agent_run_id"]

        time.sleep(0.3)
        cancel = client.post(f"/runs/{run_id}/cancel")
        assert cancel.status_code == 200

        final = _poll_until_terminal(run_id, timeout=10.0)
        assert final["status"] == "cancelled"


class TestArtifactDownload:
    def test_downloads_the_produced_artifact_with_a_filename_header(self, project):
        start = client.post("/runs", json={"root_path": str(project), "config": _config(), "run_number": 1})
        run_id = start.json()["agent_run_id"]
        _poll_until_terminal(run_id)

        resp = client.get(f"/runs/{run_id}/artifact")
        assert resp.status_code == 200
        assert resp.content == b"not-a-real-checkpoint"
        assert resp.headers["x-artifact-filename"] == "best_model.pt"

    def test_downloading_before_a_run_finishes_or_for_an_artifact_less_config_returns_404(self, project):
        config = _config(artifact={"path": None})
        start = client.post("/runs", json={"root_path": str(project), "config": config, "run_number": 1})
        run_id = start.json()["agent_run_id"]
        _poll_until_terminal(run_id)

        assert client.get(f"/runs/{run_id}/artifact").status_code == 404

    def test_downloading_for_an_unknown_run_returns_404(self):
        assert client.get("/runs/does-not-exist/artifact").status_code == 404


PROTECTED_ROUTES = [
    ("GET", "/auth-check", None),
    ("POST", "/scan", {"root_path": "/tmp"}),
    ("POST", "/validate", {"root_path": "/tmp", "config": {}}),
    ("POST", "/runs", {"root_path": "/tmp", "config": {}, "run_number": 1}),
    ("GET", "/runs/anything", None),
    ("POST", "/runs/anything/cancel", None),
    ("GET", "/runs/anything/artifact", None),
]


def _call(c, method, path, body):
    return c.request(method, path, json=body) if body is not None else c.request(method, path)


class TestAuth:
    def test_health_is_open_without_a_token(self):
        assert anonymous_client.get("/health").status_code == 200

    @pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES)
    def test_every_protected_route_rejects_a_missing_token(self, method, path, body):
        resp = _call(anonymous_client, method, path, body)
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"

    @pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES)
    def test_every_protected_route_rejects_a_wrong_token(self, method, path, body):
        resp = _call(
            TestClient(app, headers={"Authorization": "Bearer not-the-token"}), method, path, body
        )
        assert resp.status_code == 401

    def test_a_non_bearer_scheme_is_rejected(self):
        resp = TestClient(app, headers={"Authorization": f"Basic {TOKEN}"}).get("/auth-check")
        assert resp.status_code == 401

    def test_the_right_token_is_accepted(self):
        assert client.get("/auth-check").status_code == 200

    def test_a_token_that_is_only_a_prefix_of_the_real_one_is_rejected(self):
        resp = TestClient(app, headers={"Authorization": f"Bearer {TOKEN[:-1]}"}).get("/auth-check")
        assert resp.status_code == 401

    def test_fails_closed_when_no_token_is_configured(self, monkeypatch):
        monkeypatch.delenv("AI_ENV_AGENT_TOKEN")
        resp = client.post("/scan", json={"root_path": "/tmp"})
        assert resp.status_code == 503
        assert "AI_ENV_AGENT_TOKEN" in resp.json()["detail"]

        health = anonymous_client.get("/health").json()
        assert health["token_configured"] is False
        assert health["auth_required"] is True

    def test_an_empty_token_counts_as_not_configured(self, monkeypatch):
        monkeypatch.setenv("AI_ENV_AGENT_TOKEN", "")
        assert client.get("/auth-check").status_code == 503

    def test_explicit_opt_out_allows_unauthenticated_access(self, monkeypatch):
        monkeypatch.delenv("AI_ENV_AGENT_TOKEN")
        monkeypatch.setenv("AI_ENV_AGENT_ALLOW_NO_AUTH", "1")
        assert anonymous_client.get("/auth-check").status_code == 200
        assert anonymous_client.get("/health").json()["auth_required"] is False

    def test_opt_out_requires_exactly_1(self, monkeypatch):
        monkeypatch.delenv("AI_ENV_AGENT_TOKEN")
        monkeypatch.setenv("AI_ENV_AGENT_ALLOW_NO_AUTH", "true")
        assert anonymous_client.get("/auth-check").status_code == 503

    def test_a_rejected_request_never_reaches_the_command_runner(self, tmp_path):
        marker = tmp_path / "ran"
        config = _config(training={"command": f"{PY} -c \"open({str(marker)!r}, 'w').close()\""})
        resp = anonymous_client.post(
            "/runs", json={"root_path": str(tmp_path), "config": config, "run_number": 1}
        )
        assert resp.status_code == 401
        time.sleep(0.5)
        assert not marker.exists()
