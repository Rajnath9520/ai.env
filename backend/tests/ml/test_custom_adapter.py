import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from app.ml.project.adapters.base import CapabilityNotSupported, RunContext
from app.ml.project.adapters.custom_adapter import CustomAdapter
from app.ml.project.capabilities import Capability
from app.ml.project.config import AIEnvConfig
from app.ml.project.security import PathTraversalError

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "test-custom-adapter-project"
PY = sys.executable


@pytest.fixture()
def project(tmp_path) -> Path:
    dest = tmp_path / "custom-adapter-project"
    shutil.copytree(FIXTURE_ROOT, dest)
    return dest


def _config(**overrides) -> dict:
    base = {
        "project": {"name": "custom-adapter-test"},
        # Required by TrainingConfig even though CustomAdapter never uses
        # it -- see AdapterConfig's docstring for why.
        "training": {"command": "unused-for-custom-adapters"},
        "artifact": {"path": "outputs/model.bin"},
        "adapter": {"type": "custom", "python": PY},
    }
    for section, values in overrides.items():
        base[section] = {**base.get(section, {}), **values}
    return base


def _adapter(project: Path, **overrides) -> CustomAdapter:
    return CustomAdapter(AIEnvConfig.from_dict(_config(**overrides)), project)


def _ctx(project: Path, **kwargs) -> RunContext:
    return RunContext(project_root=project, experiment_id="exp-1", run_id="run-1", **kwargs)


def _write_adapter(project: Path, source: str) -> None:
    (project / ".ai-env").mkdir(exist_ok=True)
    (project / ".ai-env" / "adapter.py").write_text(source)


class TestCapabilities:
    def test_discovers_all_three_methods_plus_checkpoint(self, project):
        adapter = _adapter(project)
        assert adapter.capabilities == frozenset(
            {Capability.TRAIN, Capability.EVALUATE, Capability.PREDICT, Capability.CHECKPOINT}
        )

    def test_no_artifact_path_means_no_checkpoint_capability(self, project):
        adapter = _adapter(project, artifact={"path": None})
        assert Capability.CHECKPOINT not in adapter.capabilities

    def test_partial_adapter_only_reports_what_it_defines(self, project):
        _write_adapter(
            project,
            "class OnlyTrain:\n"
            "    def train(self, config):\n"
            "        return {'ok': True}\n"
            "def get_adapter():\n"
            "    return OnlyTrain()\n",
        )
        adapter = _adapter(project, artifact={"path": None})
        assert adapter.capabilities == frozenset({Capability.TRAIN})

    def test_capabilities_are_cached_after_first_call(self, project):
        adapter = _adapter(project)
        first = adapter.capabilities
        # Delete the adapter file entirely -- if this weren't cached, the
        # second access would now see "not found" instead.
        (project / ".ai-env" / "adapter.py").unlink()
        assert adapter.capabilities == first

    def test_missing_adapter_file_yields_no_capabilities(self, project):
        (project / ".ai-env" / "adapter.py").unlink()
        adapter = _adapter(project, artifact={"path": None})
        assert adapter.capabilities == frozenset()

    def test_module_with_no_get_adapter_yields_no_capabilities(self, project):
        _write_adapter(project, "x = 1\n")
        adapter = _adapter(project, artifact={"path": None})
        assert adapter.capabilities == frozenset()

    def test_get_adapter_raising_yields_no_capabilities_not_a_crash(self, project):
        _write_adapter(project, "def get_adapter():\n    raise RuntimeError('boom')\n")
        adapter = _adapter(project, artifact={"path": None})
        assert adapter.capabilities == frozenset()

    def test_syntax_error_in_adapter_module_yields_no_capabilities(self, project):
        _write_adapter(project, "def get_adapter(:\n")  # invalid syntax
        adapter = _adapter(project, artifact={"path": None})
        assert adapter.capabilities == frozenset()


class TestValidateProject:
    def test_fully_working_adapter_is_valid(self, project):
        result = _adapter(project).validate_project()
        assert result.ok is True

    def test_missing_adapter_file_is_an_error(self, project):
        (project / ".ai-env" / "adapter.py").unlink()
        result = _adapter(project, artifact={"path": None}).validate_project()
        assert result.ok is False
        assert any(i.field == "adapter.entrypoint" for i in result.issues)

    def test_adapter_with_no_usable_methods_is_a_warning(self, project):
        _write_adapter(project, "class Empty:\n    pass\ndef get_adapter():\n    return Empty()\n")
        result = _adapter(project, artifact={"path": None}).validate_project()
        assert result.ok is True
        assert any(i.severity == "warning" and i.field == "adapter.entrypoint" for i in result.issues)

    def test_broken_adapter_module_is_an_error_with_a_useful_message(self, project):
        _write_adapter(project, "import nonexistent_module_xyz\n")
        result = _adapter(project, artifact={"path": None}).validate_project()
        assert result.ok is False
        issue = next(i for i in result.issues if i.field == "adapter.entrypoint")
        assert "nonexistent_module_xyz" in issue.message

    def test_custom_entrypoint_escaping_project_root_is_an_error(self, project):
        result = _adapter(project, adapter={"entrypoint": "../../outside.py"}).validate_project()
        assert result.ok is False
        assert any(i.field == "adapter.entrypoint" for i in result.issues)

    def test_nonexistent_project_root_is_invalid(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        result = CustomAdapter(AIEnvConfig.from_dict(_config()), missing).validate_project()
        assert result.ok is False


class TestTrain:
    def test_successful_train_produces_the_artifact(self, project):
        adapter = _adapter(project)
        result = adapter.train(_ctx(project, dataset_path="/data/v1"))
        assert result.status == "success"
        assert result.artifact is not None
        assert result.artifact.exists is True

    def test_the_variables_reach_the_adapter(self, project):
        # FixtureAdapter.train() echoes variables.dataset back into
        # outputs/model.bin's write path isn't observable directly, but a
        # custom fixture adapter can assert on it via its own file.
        _write_adapter(
            project,
            "import json\n"
            "from pathlib import Path\n"
            "class Echo:\n"
            "    def train(self, config):\n"
            "        Path('seen_dataset.json').write_text(json.dumps(config['variables']['dataset']))\n"
            "        return {'ok': True}\n"
            "def get_adapter():\n"
            "    return Echo()\n",
        )
        adapter = _adapter(project, artifact={"path": None})
        adapter.train(_ctx(project, dataset_path="/data/custom-path"))
        assert (project / "seen_dataset.json").read_text() == '"/data/custom-path"'

    def test_train_raising_is_reported_as_failed_with_the_message(self, project):
        _write_adapter(
            project,
            "class Boom:\n"
            "    def train(self, config):\n"
            "        raise ValueError('something went wrong in training')\n"
            "def get_adapter():\n"
            "    return Boom()\n",
        )
        adapter = _adapter(project, artifact={"path": None})
        result = adapter.train(_ctx(project))
        assert result.status == "failed"
        assert "something went wrong in training" in result.stderr_tail

    def test_non_json_serializable_return_value_is_reported_as_failed(self, project):
        _write_adapter(
            project,
            "class BadReturn:\n"
            "    def train(self, config):\n"
            "        return {'thing': object()}\n"
            "def get_adapter():\n"
            "    return BadReturn()\n",
        )
        adapter = _adapter(project, artifact={"path": None})
        result = adapter.train(_ctx(project))
        assert result.status == "failed"
        assert "JSON-serializable" in result.stderr_tail

    def test_train_without_the_capability_raises(self, project):
        _write_adapter(project, "class NoTrain:\n    pass\ndef get_adapter():\n    return NoTrain()\n")
        adapter = _adapter(project, artifact={"path": None})
        with pytest.raises(CapabilityNotSupported):
            adapter.train(_ctx(project))

    def test_secrets_printed_by_the_adapter_are_redacted(self, project):
        _write_adapter(
            project,
            "class Leaky:\n"
            "    def train(self, config):\n"
            "        print('OPENAI_API_KEY=sk-shouldnotappear')\n"
            "        return {'ok': True}\n"
            "def get_adapter():\n"
            "    return Leaky()\n",
        )
        adapter = _adapter(project, artifact={"path": None})
        result = adapter.train(_ctx(project))
        assert "sk-shouldnotappear" not in result.stdout_tail
        assert "REDACTED" in result.stdout_tail


class TestCancellation:
    def test_cancelling_stops_user_code_actually_sleeping(self, project):
        _write_adapter(
            project,
            "import time\n"
            "class Slow:\n"
            "    def train(self, config):\n"
            "        time.sleep(30)\n"
            "        return {'ok': True}\n"
            "def get_adapter():\n"
            "    return Slow()\n",
        )
        adapter = _adapter(project, artifact={"path": None}, training={"timeout_seconds": 60})
        cancel_event = threading.Event()
        threading.Timer(0.3, cancel_event.set).start()

        started = time.monotonic()
        result = adapter.train(_ctx(project), cancel_event=cancel_event)
        elapsed = time.monotonic() - started

        assert result.status == "cancelled"
        assert elapsed < 15


class TestEvaluate:
    def test_successful_evaluate_returns_metrics(self, project):
        adapter = _adapter(project)
        result = adapter.evaluate(_ctx(project))
        assert result.status == "success"
        assert result.metrics == {"accuracy": 0.87, "f1": 0.83}
        assert result.samples == 50

    def test_get_metrics_reflects_the_last_evaluation(self, project):
        adapter = _adapter(project)
        assert adapter.get_metrics() is None
        adapter.evaluate(_ctx(project))
        assert adapter.get_metrics() == {"accuracy": 0.87, "f1": 0.83}

    def test_non_dict_result_is_reported_as_failed(self, project):
        _write_adapter(
            project,
            "class BadEval:\n"
            "    def evaluate(self, config):\n"
            "        return 'not-a-dict'\n"
            "def get_adapter():\n"
            "    return BadEval()\n",
        )
        adapter = _adapter(project, artifact={"path": None})
        result = adapter.evaluate(_ctx(project))
        assert result.status == "failed"
        assert result.warnings

    def test_evaluate_raising_is_reported_as_failed(self, project):
        _write_adapter(
            project,
            "class Boom:\n"
            "    def evaluate(self, config):\n"
            "        raise RuntimeError('eval exploded')\n"
            "def get_adapter():\n"
            "    return Boom()\n",
        )
        adapter = _adapter(project, artifact={"path": None})
        result = adapter.evaluate(_ctx(project))
        assert result.status == "failed"
        assert any("eval exploded" in w for w in result.warnings)


class TestPredict:
    def test_predict_echoes_input_through_the_subprocess(self, project):
        adapter = _adapter(project)
        assert adapter.predict("hello") == {"echo": "hello"}

    def test_predict_without_the_capability_raises_capability_error(self, project):
        _write_adapter(project, "class NoPredict:\n    pass\ndef get_adapter():\n    return NoPredict()\n")
        adapter = _adapter(project, artifact={"path": None})
        with pytest.raises(CapabilityNotSupported):
            adapter.predict("x")

    def test_predict_raising_is_surfaced_as_a_runtime_error(self, project):
        _write_adapter(
            project,
            "class Boom:\n"
            "    def predict(self, config):\n"
            "        raise ValueError('bad input')\n"
            "def get_adapter():\n"
            "    return Boom()\n",
        )
        adapter = _adapter(project, artifact={"path": None})
        with pytest.raises(RuntimeError, match="bad input"):
            adapter.predict("x")


class TestGetArtifact:
    def test_before_training_artifact_does_not_exist(self, project):
        info = _adapter(project).get_artifact()
        assert info.exists is False

    def test_path_traversal_in_artifact_path_raises(self, project):
        adapter = _adapter(project, artifact={"path": "../../outside.bin"})
        with pytest.raises(PathTraversalError):
            adapter.get_artifact()
