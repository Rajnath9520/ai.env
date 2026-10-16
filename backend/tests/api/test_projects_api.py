import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "ml" / "fixtures" / "test-ai-project"
PY = sys.executable


def _valid_config_payload() -> dict:
    return {
        "name": "api-test-project",
        "project": {"task": "face_recognition", "framework": "pytorch"},
        "training": {"command": "python train.py --dataset {dataset}", "entrypoint": "train.py"},
        "evaluation": {
            "command": "python evaluate.py --dataset {dataset}",
            "entrypoint": "evaluate.py",
            "metrics_source": "file",
            "metrics_file": "outputs/metrics.json",
        },
        "artifact": {"path": "outputs/best_model.pt", "type": "pytorch_checkpoint"},
        "dataset": {"path": "dataset"},
        "curriculum": {"strategy": "failure_driven"},
        "optimization": {"max_iterations": 10, "target_metric": "f1", "target_value": 0.95},
    }


def _register(client, **overrides) -> int:
    payload = {"root_path": str(FIXTURE_ROOT), **overrides}
    return client.post("/api/projects/register", json=payload).json()["id"]


@pytest.fixture()
def project_copy(tmp_path) -> Path:
    """A run actually executes train.py/evaluate.py, which writes into
    outputs/ -- run-lifecycle tests get their own isolated copy rather
    than sharing FIXTURE_ROOT with every other test in this module."""
    dest = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, dest)
    return dest


def _register_and_configure(client, root_path: Path, **config_overrides) -> int:
    project_id = client.post("/api/projects/register", json={"root_path": str(root_path)}).json()["id"]
    payload = _valid_config_payload()
    payload["training"]["command"] = f"{PY} train.py --dataset {{dataset}}"
    payload["evaluation"]["command"] = f"{PY} evaluate.py --dataset {{dataset}}"
    for section, values in config_overrides.items():
        payload[section] = {**payload.get(section, {}), **values}
    resp = client.put(f"/api/projects/{project_id}/config", json=payload)
    assert resp.status_code == 200, resp.text
    return project_id


class TestProjectRegistration:
    def test_register_creates_draft_project(self, client):
        resp = client.post("/api/projects/register", json={"root_path": str(FIXTURE_ROOT)})
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "draft"
        assert body["source"] == "local"
        assert body["name"] == "test-ai-project"  # derived from the folder name

    def test_register_with_explicit_name(self, client):
        resp = client.post(
            "/api/projects/register", json={"root_path": str(FIXTURE_ROOT), "name": "My Face Model"}
        )
        assert resp.json()["name"] == "My Face Model"

    def test_list_includes_registered_project(self, client):
        project_id = _register(client)
        listed = client.get("/api/projects")
        assert listed.status_code == 200
        assert any(p["id"] == project_id for p in listed.json())

    def test_get_missing_project_returns_404(self, client):
        assert client.get("/api/projects/999999").status_code == 404


class TestScan:
    def test_scan_detects_framework_and_candidates(self, client):
        project_id = _register(client)
        resp = client.post(f"/api/projects/{project_id}/scan")
        assert resp.status_code == 200

        body = resp.json()
        assert body["framework"] == "pytorch"
        assert "train.py" in body["training_candidates"]
        assert "evaluate.py" in body["evaluation_candidates"]

        detail = client.get(f"/api/projects/{project_id}").json()
        assert detail["status"] == "scanned"
        assert detail["framework"] == "pytorch"
        assert detail["detected_config"]["framework"] == "pytorch"

    def test_scan_of_nonexistent_path_returns_422(self, client):
        project_id = _register(client, root_path="/definitely/not/a/real/path")
        resp = client.post(f"/api/projects/{project_id}/scan")
        assert resp.status_code == 422

    def test_scan_of_missing_project_returns_404(self, client):
        assert client.post("/api/projects/999999/scan").status_code == 404


class TestConfiguration:
    def test_confirm_config_marks_project_configured_with_capabilities(self, client):
        project_id = _register(client)
        resp = client.put(f"/api/projects/{project_id}/config", json=_valid_config_payload())
        assert resp.status_code == 200

        body = resp.json()
        assert body["status"] == "configured"
        assert set(body["capabilities"]) == {"train", "evaluate", "checkpoint"}
        assert body["task"] == "face_recognition"

    def test_confirmed_config_is_retrievable(self, client):
        project_id = _register(client)
        client.put(f"/api/projects/{project_id}/config", json=_valid_config_payload())

        resp = client.get(f"/api/projects/{project_id}/config")
        assert resp.status_code == 200
        assert resp.json()["training"]["command"] == "python train.py --dataset {dataset}"

    def test_get_config_before_confirmation_returns_409(self, client):
        project_id = _register(client)
        assert client.get(f"/api/projects/{project_id}/config").status_code == 409

    def test_whitespace_only_training_command_is_rejected(self, client):
        project_id = _register(client)
        payload = _valid_config_payload()
        payload["training"]["command"] = "   "
        resp = client.put(f"/api/projects/{project_id}/config", json=payload)
        assert resp.status_code == 422

    def test_invalid_curriculum_strategy_is_rejected(self, client):
        project_id = _register(client)
        payload = _valid_config_payload()
        payload["curriculum"]["strategy"] = "not_a_real_strategy"
        resp = client.put(f"/api/projects/{project_id}/config", json=payload)
        assert resp.status_code == 422

    def test_config_edit_invalidates_previous_validation_result(self, client):
        project_id = _register(client)
        client.put(f"/api/projects/{project_id}/config", json=_valid_config_payload())
        client.post(f"/api/projects/{project_id}/validate")

        client.put(f"/api/projects/{project_id}/config", json=_valid_config_payload())
        detail = client.get(f"/api/projects/{project_id}").json()
        assert detail["status"] == "configured"
        assert detail["validation_result"] is None


class TestValidation:
    def test_validate_before_config_returns_409(self, client):
        project_id = _register(client)
        assert client.post(f"/api/projects/{project_id}/validate").status_code == 409

    def test_validate_fully_configured_project_is_ok(self, client):
        project_id = _register(client)
        client.put(f"/api/projects/{project_id}/config", json=_valid_config_payload())

        resp = client.post(f"/api/projects/{project_id}/validate")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        detail = client.get(f"/api/projects/{project_id}").json()
        assert detail["status"] == "validated"
        assert detail["validation_result"]["ok"] is True

    def test_validate_flags_path_traversal_outside_project_root(self, client):
        project_id = _register(client)
        payload = _valid_config_payload()
        payload["artifact"]["path"] = "../../outside.pt"
        client.put(f"/api/projects/{project_id}/config", json=payload)

        resp = client.post(f"/api/projects/{project_id}/validate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert any(issue["field"] == "artifact.path" for issue in body["issues"])

    def test_validate_never_executes_the_training_command(self, client, tmp_path):
        # A command that would create a marker file if it were ever
        # actually run (Section 74/75: validate is side-effect-free).
        marker = tmp_path / "should_not_exist"
        project_id = _register(client)
        payload = _valid_config_payload()
        payload["training"]["command"] = f"touch {marker}"
        client.put(f"/api/projects/{project_id}/config", json=payload)

        client.post(f"/api/projects/{project_id}/validate")
        assert not marker.exists()


class TestRuns:
    def test_starting_a_run_before_configuration_returns_409(self, client):
        project_id = _register(client)
        assert client.post(f"/api/projects/{project_id}/runs").status_code == 409

    def test_full_run_trains_and_evaluates_and_archives_artifact(self, client, project_copy):
        project_id = _register_and_configure(
            client,
            project_copy,
            evaluation={"metrics_source": "file", "metrics_file": "outputs/metrics.json"},
        )

        start_resp = client.post(f"/api/projects/{project_id}/runs")
        assert start_resp.status_code == 201
        run_id = start_resp.json()["id"]
        assert start_resp.json()["run_number"] == 1
        # The response body reflects the row's state at the moment the
        # route returned -- before the background task (which completes
        # synchronously within this same TestClient call, per the
        # experiments API's own tests) finished. A follow-up GET sees the
        # final state.
        assert start_resp.json()["status"] in ("running_training", "completed")

        final = client.get(f"/api/projects/{project_id}/runs/{run_id}").json()
        assert final["status"] == "completed"
        assert final["train_result"]["status"] == "success"
        assert final["evaluation_result"]["status"] == "success"
        assert final["evaluation_result"]["metrics"] == {"accuracy": 0.968, "f1": 0.966}
        assert final["managed_artifact_path"] is not None
        assert Path(final["managed_artifact_path"]).is_file()
        assert f"projects/{project_id}/runs/1" in final["managed_artifact_path"].replace("\\", "/")

    def test_run_appears_in_the_project_runs_list(self, client, project_copy):
        project_id = _register_and_configure(client, project_copy)
        run_id = client.post(f"/api/projects/{project_id}/runs").json()["id"]

        listed = client.get(f"/api/projects/{project_id}/runs").json()
        assert any(r["id"] == run_id for r in listed)

    def test_run_numbers_increment_and_artifacts_stay_separate(self, client, project_copy):
        project_id = _register_and_configure(client, project_copy)

        first_id = client.post(f"/api/projects/{project_id}/runs").json()["id"]
        second_id = client.post(f"/api/projects/{project_id}/runs").json()["id"]

        first = client.get(f"/api/projects/{project_id}/runs/{first_id}").json()
        second = client.get(f"/api/projects/{project_id}/runs/{second_id}").json()

        assert first["run_number"] == 1
        assert second["run_number"] == 2
        assert first["managed_artifact_path"] != second["managed_artifact_path"]
        # Rule 8 / Section 40: the first run's archived copy must still be
        # intact after the second run re-executed train.py (which
        # overwrites the *live* outputs/best_model.pt in place).
        assert Path(first["managed_artifact_path"]).is_file()
        assert Path(second["managed_artifact_path"]).is_file()

    def test_run_with_no_metrics_source_reports_empty_metrics_honestly(self, client, project_copy):
        project_id = _register_and_configure(
            client, project_copy, evaluation={"metrics_source": None, "metrics_file": None}
        )
        run_id = client.post(f"/api/projects/{project_id}/runs").json()["id"]
        run = client.get(f"/api/projects/{project_id}/runs/{run_id}").json()

        assert run["status"] == "completed"
        assert run["evaluation_result"]["metrics"] == {}
        assert run["evaluation_result"]["warnings"]

    def test_failed_training_command_skips_evaluation_and_marks_run_failed(self, client, project_copy):
        project_id = _register_and_configure(
            client, project_copy, training={"command": f"{PY} -c \"import sys; sys.exit(1)\""}
        )
        run_id = client.post(f"/api/projects/{project_id}/runs").json()["id"]
        run = client.get(f"/api/projects/{project_id}/runs/{run_id}").json()

        assert run["status"] == "failed"
        assert run["train_result"]["status"] == "failed"
        assert run["evaluation_result"] is None
        assert run["managed_artifact_path"] is None

    def test_get_run_for_missing_run_returns_404(self, client, project_copy):
        project_id = _register_and_configure(client, project_copy)
        assert client.get(f"/api/projects/{project_id}/runs/999999").status_code == 404

    def test_cancel_on_a_completed_run_is_a_harmless_no_op(self, client, project_copy):
        project_id = _register_and_configure(client, project_copy)
        run_id = client.post(f"/api/projects/{project_id}/runs").json()["id"]

        resp = client.post(f"/api/projects/{project_id}/runs/{run_id}/cancel")
        assert resp.status_code == 200
        assert client.get(f"/api/projects/{project_id}/runs/{run_id}").json()["status"] == "completed"

    def test_cancel_on_a_missing_run_returns_404(self, client, project_copy):
        project_id = _register_and_configure(client, project_copy)
        assert client.post(f"/api/projects/{project_id}/runs/999999/cancel").status_code == 404

    def test_cancelling_a_slow_run_stops_it_well_before_its_full_duration(self, client, project_copy):
        project_id = _register_and_configure(
            client, project_copy, training={"command": f"{PY} -c \"import time; time.sleep(20)\""}
        )

        result: dict = {}

        def start():
            result["response"] = client.post(f"/api/projects/{project_id}/runs")

        thread = threading.Thread(target=start)
        started_at = time.monotonic()
        thread.start()

        # Poll for the run row to show up, then cancel it -- this proves
        # the cancel request reaches the same in-flight run through the
        # full route -> service -> background-task stack, not just the
        # CommandAdapter unit tested directly in test_command_adapter.py.
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
        assert elapsed < 15  # well under the command's own 20s sleep
        final = client.get(f"/api/projects/{project_id}/runs/{run_id}").json()
        assert final["status"] == "cancelled"


class TestProgress:
    def test_progress_before_any_runs_is_empty(self, client, project_copy):
        project_id = _register_and_configure(client, project_copy)
        progress = client.get(f"/api/projects/{project_id}/progress").json()
        assert progress["completed_run_count"] == 0
        assert progress["metrics"] == []

    def test_progress_for_an_unconfigured_project_does_not_error(self, client):
        project_id = _register(client)
        resp = client.get(f"/api/projects/{project_id}/progress")
        assert resp.status_code == 200
        assert resp.json()["completed_run_count"] == 0

    def test_progress_tracks_target_metric_distance_across_real_runs(self, client, project_copy):
        project_id = _register_and_configure(
            client,
            project_copy,
            evaluation={"metrics_source": "file", "metrics_file": "outputs/metrics.json"},
            optimization={"target_metric": "f1", "target_value": 0.95, "max_iterations": 10},
        )

        client.post(f"/api/projects/{project_id}/runs")
        client.post(f"/api/projects/{project_id}/runs")

        progress = client.get(f"/api/projects/{project_id}/progress").json()
        assert progress["completed_run_count"] == 2
        assert progress["target_metric"] == "f1"
        f1_trend = next(m for m in progress["metrics"] if m["metric"] == "f1")
        # the fixture's evaluate.py always reports the same f1 -- two
        # identical runs means zero delta and zero movement, which is
        # itself a real, honestly-computed fact this asserts on.
        assert f1_trend["delta_from_previous"] == 0
        assert progress["distance_to_target"] == pytest.approx(abs(0.966 - 0.95))
        assert progress["moved_closer_to_target"] is False  # unchanged distance is not "closer"
