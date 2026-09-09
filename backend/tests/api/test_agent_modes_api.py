"""Phase 28 (Section 24/27) -- API-level tests for agent_mode.

Exercises the actual HTTP surface a frontend would call: creating
experiments in each mode, starting them, and (for external mode) driving
the reset/step/state endpoints -- verifying the same PPO-never-instantiated
guarantee holds end-to-end through the API layer, not just at the
ExperimentManager/session level (already covered by
tests/ml/test_agent_modes.py and tests/db/test_external_agent.py).
"""

from __future__ import annotations

from tests.api.conftest import TINY_PAYLOAD

ENV_ONLY_PAYLOAD = {**TINY_PAYLOAD, "agent_mode": "environment_only"}
EXTERNAL_PAYLOAD = {**TINY_PAYLOAD, "agent_mode": "external"}
DEMO_PPO_PAYLOAD = {**TINY_PAYLOAD, "agent_mode": "demo_ppo"}


class TestAgentModeDefaultsAndCreation:
    def test_create_without_agent_mode_defaults_to_environment_only(self, client):
        payload = {k: v for k, v in TINY_PAYLOAD.items() if k != "agent_mode"}
        resp = client.post("/api/experiments", json={"name": "api_default_mode", "curriculum_type": "random", **payload})
        assert resp.status_code == 201
        assert resp.json()["agent_mode"] == "environment_only"

    def test_rejects_unknown_agent_mode(self, client):
        resp = client.post(
            "/api/experiments",
            json={"name": "api_bad_mode", "curriculum_type": "random", **{**TINY_PAYLOAD, "agent_mode": "auto_train"}},
        )
        assert resp.status_code == 422


class TestEnvironmentOnlyModeAPI:
    def test_start_generates_environments_with_no_training_rows(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_env_only", "curriculum_type": "random", **ENV_ONLY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]

        start_resp = client.post(f"/api/experiments/{exp_id}/start")
        assert start_resp.status_code == 200

        status_resp = client.get(f"/api/experiments/{exp_id}/status")
        assert status_resp.json()["status"] == "completed"

        environments = client.get(f"/api/experiments/{exp_id}/environments").json()
        assert len(environments) == ENV_ONLY_PAYLOAD["n_curriculum_steps"]
        for env in environments:
            assert env["target_weakness"] is None

        # No agent ever ran: these must all be empty, not fabricated.
        assert client.get(f"/api/experiments/{exp_id}/episodes").json() == []
        assert client.get(f"/api/experiments/{exp_id}/weaknesses").json() == []
        assert client.get(f"/api/experiments/{exp_id}/generalization").json() == []

    def test_metrics_endpoint_reports_environment_only_mode(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_env_only_metrics", "curriculum_type": "random", **ENV_ONLY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/api/experiments/{exp_id}/start")

        metrics = client.get(f"/api/experiments/{exp_id}/metrics").json()
        assert metrics["agent_mode"] == "environment_only"
        assert metrics["latest_primary_weakness"] is None


class TestExternalAgentModeAPI:
    def test_start_prepares_environment_and_waits_for_agent(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_external_start", "curriculum_type": "random", **EXTERNAL_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]

        start_resp = client.post(f"/api/experiments/{exp_id}/start")
        assert start_resp.status_code == 200
        assert start_resp.json()["status"] == "waiting_for_agent"

        state = client.get(f"/api/experiments/{exp_id}/agent/state").json()
        assert state["connected"] is False
        assert state["status"] == "waiting_for_agent"

    def test_reset_and_step_drive_real_interaction(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_external_interact", "curriculum_type": "random", **EXTERNAL_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/api/experiments/{exp_id}/start")

        reset_resp = client.post(f"/api/experiments/{exp_id}/agent/reset")
        assert reset_resp.status_code == 200
        reset_body = reset_resp.json()
        assert reset_body["episode"] == 1
        assert isinstance(reset_body["observation"], list)

        step_resp = client.post(f"/api/experiments/{exp_id}/agent/step", json={"action": 0})
        assert step_resp.status_code == 200
        step_body = step_resp.json()
        assert "reward" in step_body
        assert "terminated" in step_body

        state = client.get(f"/api/experiments/{exp_id}/agent/state").json()
        assert state["connected"] is True
        assert state["status"] in ("interacting", "agent_connected")

    def test_agent_routes_reject_wrong_mode(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_wrong_mode_agent", "curriculum_type": "random", **ENV_ONLY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        resp = client.post(f"/api/experiments/{exp_id}/agent/reset")
        assert resp.status_code == 409

    def test_step_action_out_of_range_is_rejected(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_external_bad_action", "curriculum_type": "random", **EXTERNAL_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/api/experiments/{exp_id}/start")
        client.post(f"/api/experiments/{exp_id}/agent/reset")

        resp = client.post(f"/api/experiments/{exp_id}/agent/step", json={"action": 99})
        assert resp.status_code == 422


class TestDemoPPOModeAPI:
    def test_start_trains_and_produces_metrics(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_demo_ppo", "curriculum_type": "failure_driven", **DEMO_PPO_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]

        start_resp = client.post(f"/api/experiments/{exp_id}/start")
        assert start_resp.status_code == 200

        status_resp = client.get(f"/api/experiments/{exp_id}/status")
        assert status_resp.json()["status"] == "completed"

        environments = client.get(f"/api/experiments/{exp_id}/environments").json()
        assert len(environments) == DEMO_PPO_PAYLOAD["n_curriculum_steps"]

        learning_progress = client.get(f"/api/experiments/{exp_id}/learning-progress").json()
        assert len(learning_progress) == DEMO_PPO_PAYLOAD["n_curriculum_steps"]

        metrics = client.get(f"/api/experiments/{exp_id}/metrics").json()
        assert metrics["agent_mode"] == "demo_ppo"
