from tests.api.conftest import TINY_PAYLOAD


class TestEnvironmentLayoutEndpoint:
    def _run_experiment(self, client) -> int:
        create_resp = client.post(
            "/api/experiments", json={"name": "layout_test", "curriculum_type": "random", **TINY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/api/experiments/{exp_id}/start")
        return exp_id

    def test_returns_valid_layout_matching_stored_params(self, client):
        exp_id = self._run_experiment(client)
        envs = client.get(f"/api/experiments/{exp_id}/environments").json()
        assert len(envs) > 0
        env = envs[0]

        layout_resp = client.get(f"/api/experiments/{exp_id}/environments/{env['id']}/layout")
        assert layout_resp.status_code == 200
        layout = layout_resp.json()

        assert layout["environment_id"] == env["id"]
        assert layout["grid_size"] == env["params"]["grid_size"]
        assert layout["agent_start"] != layout["goal"]
        assert isinstance(layout["obstacles"], list)
        obstacle_set = {tuple(o) for o in layout["obstacles"]}
        assert tuple(layout["agent_start"]) not in obstacle_set
        assert tuple(layout["goal"]) not in obstacle_set

    def test_same_environment_returns_identical_layout_on_repeat_calls(self, client):
        exp_id = self._run_experiment(client)
        env_id = client.get(f"/api/experiments/{exp_id}/environments").json()[0]["id"]

        first = client.get(f"/api/experiments/{exp_id}/environments/{env_id}/layout").json()
        second = client.get(f"/api/experiments/{exp_id}/environments/{env_id}/layout").json()
        assert first == second

    def test_missing_environment_returns_404(self, client):
        exp_id = self._run_experiment(client)
        resp = client.get(f"/api/experiments/{exp_id}/environments/999999/layout")
        assert resp.status_code == 404

    def test_environment_belonging_to_other_experiment_returns_404(self, client):
        exp_id_a = self._run_experiment(client)
        exp_id_b = self._run_experiment(client)
        env_from_a = client.get(f"/api/experiments/{exp_id_a}/environments").json()[0]

        resp = client.get(f"/api/experiments/{exp_id_b}/environments/{env_from_a['id']}/layout")
        assert resp.status_code == 404
