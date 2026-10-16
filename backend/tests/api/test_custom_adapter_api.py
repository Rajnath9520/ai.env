import shutil
import sys
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "ml" / "fixtures" / "test-custom-adapter-project"
PY = sys.executable


@pytest.fixture()
def project_copy(tmp_path) -> Path:
    dest = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, dest)
    return dest


def _custom_config(**overrides) -> dict:
    base = {
        "name": "custom-adapter-api-test",
        "project": {"task": "custom_task"},
        "training": {"command": "unused-for-custom-adapters"},
        "artifact": {"path": "outputs/model.bin"},
        "adapter": {"type": "custom", "python": PY},
    }
    for section, values in overrides.items():
        base[section] = {**base.get(section, {}), **values}
    return base


def _register(client, root_path: Path) -> int:
    return client.post("/api/projects/register", json={"root_path": str(root_path)}).json()["id"]


class TestCustomAdapterConfiguration:
    def test_config_round_trips_the_adapter_section(self, client, project_copy):
        project_id = _register(client, project_copy)
        resp = client.put(f"/api/projects/{project_id}/config", json=_custom_config())
        assert resp.status_code == 200

        read_back = client.get(f"/api/projects/{project_id}/config").json()
        assert read_back["adapter"] == {"type": "custom", "entrypoint": None, "python": PY}

    def test_configuring_a_custom_adapter_does_not_eagerly_store_capabilities(self, client, project_copy):
        # Capabilities for a custom adapter come from importing the user's
        # actual module, which this project's config step deliberately
        # never does on the backend (see project_service.py) -- they
        # surface through /validate instead.
        project_id = _register(client, project_copy)
        resp = client.put(f"/api/projects/{project_id}/config", json=_custom_config())
        assert resp.json()["capabilities"] == []

    def test_invalid_adapter_type_is_rejected(self, client, project_copy):
        project_id = _register(client, project_copy)
        payload = _custom_config(adapter={"type": "not-a-real-type"})
        resp = client.put(f"/api/projects/{project_id}/config", json=payload)
        assert resp.status_code == 422


class TestCustomAdapterValidation:
    def test_validate_reports_ok_and_no_eager_capability_leak(self, client, project_copy):
        project_id = _register(client, project_copy)
        client.put(f"/api/projects/{project_id}/config", json=_custom_config())

        resp = client.post(f"/api/projects/{project_id}/validate")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_validate_reports_a_missing_adapter_file(self, client, project_copy):
        (project_copy / ".ai-env" / "adapter.py").unlink()
        project_id = _register(client, project_copy)
        client.put(f"/api/projects/{project_id}/config", json=_custom_config())

        resp = client.post(f"/api/projects/{project_id}/validate")
        assert resp.json()["ok"] is False
        assert any(i["field"] == "adapter.entrypoint" for i in resp.json()["issues"])


class TestCustomAdapterRun:
    def test_full_run_trains_and_evaluates_through_the_custom_adapter(self, client, project_copy):
        project_id = _register(client, project_copy)
        client.put(
            f"/api/projects/{project_id}/config",
            json=_custom_config(evaluation={"metrics_source": None}),  # irrelevant for custom adapters
        )

        start = client.post(f"/api/projects/{project_id}/runs")
        assert start.status_code == 201
        run_id = start.json()["id"]

        run = client.get(f"/api/projects/{project_id}/runs/{run_id}").json()
        assert run["status"] == "completed"
        assert run["train_result"]["status"] == "success"
        assert run["evaluation_result"]["status"] == "success"
        assert run["evaluation_result"]["metrics"] == {"accuracy": 0.87, "f1": 0.83}
        assert run["managed_artifact_path"] is not None
        assert Path(run["managed_artifact_path"]).read_text() == "trained-by-custom-adapter"

    def test_run_with_a_broken_custom_adapter_fails_with_a_clear_reason(self, client, project_copy):
        (project_copy / ".ai-env" / "adapter.py").write_text(
            "class Boom:\n    def train(self, config):\n        raise ValueError('deliberately broken')\n"
            "def get_adapter():\n    return Boom()\n"
        )
        project_id = _register(client, project_copy)
        client.put(f"/api/projects/{project_id}/config", json=_custom_config())

        run_id = client.post(f"/api/projects/{project_id}/runs").json()["id"]
        run = client.get(f"/api/projects/{project_id}/runs/{run_id}").json()
        assert run["status"] == "failed"
        assert "deliberately broken" in run["train_result"]["stderr_tail"]
