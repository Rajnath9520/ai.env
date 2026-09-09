from tests.api.conftest import TINY_PAYLOAD


class TestExperimentCRUD:
    def test_create_experiment_returns_pending(self, client):
        resp = client.post("/api/experiments", json={"name": "api_test_create", "curriculum_type": "random", **TINY_PAYLOAD})
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "api_test_create"
        assert body["status"] == "pending"
        assert "id" in body

    def test_create_experiment_rejects_invalid_curriculum_type(self, client):
        resp = client.post(
            "/api/experiments", json={"name": "bad", "curriculum_type": "not_real", **TINY_PAYLOAD}
        )
        assert resp.status_code == 422

    def test_create_experiment_rejects_out_of_range_params(self, client):
        resp = client.post(
            "/api/experiments",
            json={"name": "bad2", "curriculum_type": "random", **{**TINY_PAYLOAD, "grid_size": 2}},
        )
        assert resp.status_code == 422

    def test_list_experiments_includes_created(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_test_list", "curriculum_type": "random", **TINY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        list_resp = client.get("/api/experiments")
        assert list_resp.status_code == 200
        assert any(e["id"] == exp_id for e in list_resp.json())

    def test_get_single_experiment(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_test_get", "curriculum_type": "random", **TINY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        resp = client.get(f"/api/experiments/{exp_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "api_test_get"

    def test_get_missing_experiment_returns_404(self, client):
        resp = client.get("/api/experiments/999999")
        assert resp.status_code == 404


class TestExperimentLifecycle:
    def test_start_pending_experiment_runs_to_completion(self, client):
        create_resp = client.post(
            "/api/experiments",
            json={"name": "api_test_lifecycle", "curriculum_type": "failure_driven", **TINY_PAYLOAD},
        )
        exp_id = create_resp.json()["id"]

        start_resp = client.post(f"/api/experiments/{exp_id}/start")
        assert start_resp.status_code == 200
        assert start_resp.json()["status"] == "running"

        status_resp = client.get(f"/api/experiments/{exp_id}/status")
        assert status_resp.json()["status"] == "completed"

    def test_starting_already_running_experiment_returns_409(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_test_conflict", "curriculum_type": "random", **TINY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/api/experiments/{exp_id}/start")  # completes synchronously in TestClient

        second_start = client.post(f"/api/experiments/{exp_id}/start")
        assert second_start.status_code == 409

    def test_starting_missing_experiment_returns_404(self, client):
        resp = client.post("/api/experiments/999999/start")
        assert resp.status_code == 404

    def test_stop_pending_experiment_marks_stopped(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "api_test_stop", "curriculum_type": "random", **TINY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        stop_resp = client.post(f"/api/experiments/{exp_id}/stop")
        assert stop_resp.status_code == 200
        assert stop_resp.json()["status"] == "stopped"


class TestExperimentResultsEndpoints:
    def _run_failure_driven_experiment(self, client) -> int:
        create_resp = client.post(
            "/api/experiments",
            json={"name": "api_test_results", "curriculum_type": "failure_driven", **TINY_PAYLOAD},
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/api/experiments/{exp_id}/start")
        return exp_id

    def test_environments_endpoint(self, client):
        exp_id = self._run_failure_driven_experiment(client)
        resp = client.get(f"/api/experiments/{exp_id}/environments")
        assert resp.status_code == 200
        assert len(resp.json()) == TINY_PAYLOAD["n_curriculum_steps"]

    def test_weaknesses_endpoint(self, client):
        exp_id = self._run_failure_driven_experiment(client)
        resp = client.get(f"/api/experiments/{exp_id}/weaknesses")
        assert resp.status_code == 200
        assert len(resp.json()) == TINY_PAYLOAD["n_curriculum_steps"]
        assert "primary_weakness" in resp.json()[0]

    def test_learning_progress_endpoint(self, client):
        exp_id = self._run_failure_driven_experiment(client)
        resp = client.get(f"/api/experiments/{exp_id}/learning-progress")
        assert resp.status_code == 200
        assert len(resp.json()) == TINY_PAYLOAD["n_curriculum_steps"]

    def test_generalization_endpoint(self, client):
        exp_id = self._run_failure_driven_experiment(client)
        resp = client.get(f"/api/experiments/{exp_id}/generalization")
        assert resp.status_code == 200
        splits = {r["split"] for r in resp.json()}
        assert splits == {"validation", "test", "ood"}

    def test_episodes_and_failures_endpoints_have_real_data(self, client):
        exp_id = self._run_failure_driven_experiment(client)
        episodes_resp = client.get(f"/api/experiments/{exp_id}/episodes")
        assert episodes_resp.status_code == 200
        assert len(episodes_resp.json()) == TINY_PAYLOAD["n_curriculum_steps"]

        failures_resp = client.get(f"/api/experiments/{exp_id}/failures")
        assert failures_resp.status_code == 200
        assert len(failures_resp.json()) > 0
        assert failures_resp.json()[0]["failure_type"]

    def test_trajectories_endpoint_returns_empty_list_not_error(self, client):
        exp_id = self._run_failure_driven_experiment(client)
        resp = client.get(f"/api/experiments/{exp_id}/trajectories")
        assert resp.status_code == 200
        assert resp.json() == []  # documented: not yet persisted, see repository docstring

    def test_metrics_endpoint(self, client):
        exp_id = self._run_failure_driven_experiment(client)
        resp = client.get(f"/api/experiments/{exp_id}/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["n_environments_trained"] == TINY_PAYLOAD["n_curriculum_steps"]
        assert body["latest_primary_weakness"] is not None


class TestComparison:
    def test_comparison_includes_multiple_experiments(self, client):
        id_a = self._run(client, "cmp_a", "random")
        id_b = self._run(client, "cmp_b", "fixed", n_curriculum_steps=4)

        resp = client.get(f"/api/experiments/{id_a}/comparison?compare_to={id_a},{id_b}")
        assert resp.status_code == 200
        ids_returned = {e["experiment_id"] for e in resp.json()}
        assert ids_returned == {id_a, id_b}

    def test_comparison_missing_base_experiment_returns_404(self, client):
        resp = client.get("/api/experiments/999999/comparison")
        assert resp.status_code == 404

    def _run(self, client, name, curriculum_type, **overrides):
        payload = {**TINY_PAYLOAD, **overrides}
        create_resp = client.post(
            "/api/experiments", json={"name": name, "curriculum_type": curriculum_type, **payload}
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/api/experiments/{exp_id}/start")
        return exp_id
