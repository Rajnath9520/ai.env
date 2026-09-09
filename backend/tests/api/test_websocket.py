from app.workers.broadcaster import _SENTINEL, broadcaster
from tests.api.conftest import TINY_PAYLOAD


class TestBroadcastIntegration:
    """Exercises the real ExperimentManager -> AdaptiveCurriculumRunner ->
    progress_callback -> broadcaster.publish pipeline end to end by
    subscribing to the broadcaster directly (plain Python, no WebSocket)
    before starting the experiment through the real API. TestClient runs
    the background task synchronously, so every tick published during that
    run is already sitting in our queue once start() returns.

    The real WebSocket route (which just reads from this same broadcaster)
    is additionally exercised against a live uvicorn server as part of
    manual verification (see the PHASE report) -- driving both a WebSocket
    connection and a concurrent POST through the *same* TestClient instance
    deadlocks (TestClient's ASGI transport isn't designed for that
    nesting), so that combination is intentionally not what this test does.
    """

    def test_failure_driven_run_publishes_real_ticks(self, client):
        create_resp = client.post(
            "/api/experiments",
            json={"name": "broadcast_test", "curriculum_type": "failure_driven", **TINY_PAYLOAD},
        )
        exp_id = create_resp.json()["id"]

        q = broadcaster.subscribe(exp_id)
        start_resp = client.post(f"/api/experiments/{exp_id}/start")
        assert start_resp.status_code == 200

        ticks = []
        while True:
            item = q.get_nowait()
            if item is _SENTINEL:
                break
            ticks.append(item)

        assert len(ticks) > 0
        for tick in ticks:
            assert tick["training_status"] in ("training", "step_completed")
            assert "reward" in tick and "success_rate" in tick and "environment_id" in tick

        step_completed = [t for t in ticks if t["training_status"] == "step_completed"]
        assert len(step_completed) == TINY_PAYLOAD["n_curriculum_steps"]
        assert any(t["primary_weakness"] is not None for t in step_completed)

        final_status = client.get(f"/api/experiments/{exp_id}/status").json()["status"]
        assert final_status == "completed"

    def test_random_curriculum_ticks_never_carry_a_weakness_label(self, client):
        create_resp = client.post(
            "/api/experiments", json={"name": "broadcast_random", "curriculum_type": "random", **TINY_PAYLOAD}
        )
        exp_id = create_resp.json()["id"]

        q = broadcaster.subscribe(exp_id)
        client.post(f"/api/experiments/{exp_id}/start")

        ticks = []
        while True:
            item = q.get_nowait()
            if item is _SENTINEL:
                break
            ticks.append(item)

        assert len(ticks) > 0
        assert all(t["primary_weakness"] is None for t in ticks)
