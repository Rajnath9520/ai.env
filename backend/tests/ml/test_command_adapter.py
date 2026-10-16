import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from app.ml.project.adapters.base import CapabilityNotSupported, RunContext
from app.ml.project.adapters.command_adapter import CommandAdapter
from app.ml.project.capabilities import Capability
from app.ml.project.config import AIEnvConfig
from app.ml.project.security import PathTraversalError

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "test-ai-project"
PY = sys.executable  # use this interpreter, not a possibly-missing "python"/"python3" on PATH


@pytest.fixture()
def project(tmp_path) -> Path:
    """A fresh, isolated copy of the fixture project per test, so
    train.py's own `outputs/best_model.pt` writes never leak between
    tests."""
    dest = tmp_path / "test-ai-project"
    shutil.copytree(FIXTURE_ROOT, dest)
    return dest


def _config(project: Path, **overrides) -> AIEnvConfig:
    base = {
        "project": {"name": "test-ai-project", "framework": "pytorch"},
        "training": {"command": f"{PY} train.py --dataset {{dataset}}"},
        "evaluation": {"command": f"{PY} evaluate.py --dataset {{dataset}}"},
        "artifact": {"path": "outputs/best_model.pt", "type": "pytorch_checkpoint"},
        "dataset": {"path": "dataset"},
    }
    for section, values in overrides.items():
        base[section] = {**base.get(section, {}), **values}
    return AIEnvConfig.from_dict(base)


def _ctx(project: Path, **kwargs) -> RunContext:
    return RunContext(project_root=project, experiment_id="exp-1", run_id="run-1", **kwargs)


class TestCapabilities:
    def test_declares_train_evaluate_and_checkpoint_when_all_configured(self, project):
        adapter = CommandAdapter(_config(project), project)
        assert adapter.capabilities == frozenset(
            {Capability.TRAIN, Capability.EVALUATE, Capability.CHECKPOINT}
        )

    def test_declares_only_train_when_that_is_all_thats_configured(self, project):
        config = AIEnvConfig.from_dict(
            {"project": {"name": "x"}, "training": {"command": f"{PY} train.py"}}
        )
        adapter = CommandAdapter(config, project)
        assert adapter.capabilities == frozenset({Capability.TRAIN})

    def test_evaluate_without_evaluate_capability_raises(self, project):
        config = AIEnvConfig.from_dict(
            {"project": {"name": "x"}, "training": {"command": f"{PY} train.py"}}
        )
        adapter = CommandAdapter(config, project)
        with pytest.raises(CapabilityNotSupported):
            adapter.evaluate(_ctx(project))


class TestCommandSubstitution:
    def test_dataset_placeholder_is_substituted(self, project):
        adapter = CommandAdapter(_config(project), project)
        command = adapter.build_train_command(_ctx(project, dataset_path="/data/v2"))
        assert command == f"{PY} train.py --dataset /data/v2"

    def test_missing_variable_defaults_to_empty_string(self, project):
        adapter = CommandAdapter(_config(project), project)
        command = adapter.build_train_command(_ctx(project))
        assert command == f"{PY} train.py --dataset "

    def test_unsupported_variable_raises_clear_error(self, project):
        config = _config(project, training={"command": f"{PY} train.py --foo {{not_a_real_variable}}"})
        adapter = CommandAdapter(config, project)
        with pytest.raises(ValueError, match="not_a_real_variable"):
            adapter.build_train_command(_ctx(project))


class TestValidateProject:
    def test_fully_configured_fixture_project_is_valid(self, project):
        adapter = CommandAdapter(_config(project), project)
        result = adapter.validate_project()
        assert result.ok is True
        assert not any(issue.severity == "error" for issue in result.issues)

    def test_nonexistent_project_root_is_invalid(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        adapter = CommandAdapter(_config(missing), missing)
        result = adapter.validate_project()
        assert result.ok is False

    def test_missing_entrypoint_is_a_warning_not_an_error(self, project):
        config = _config(project, training={"entrypoint": "nonexistent_train.py"})
        adapter = CommandAdapter(config, project)
        result = adapter.validate_project()
        assert result.ok is True
        assert any(i.field == "training.entrypoint" and i.severity == "warning" for i in result.issues)

    def test_working_directory_escaping_project_root_is_an_error(self, project):
        config = _config(project, training={"working_directory": "../../../etc"})
        adapter = CommandAdapter(config, project)
        result = adapter.validate_project()
        assert result.ok is False
        assert any(i.field == "training.working_directory" for i in result.issues)

    def test_artifact_path_escaping_project_root_is_an_error(self, project):
        config = _config(project, artifact={"path": "../../outside.pt"})
        adapter = CommandAdapter(config, project)
        result = adapter.validate_project()
        assert result.ok is False
        assert any(i.field == "artifact.path" for i in result.issues)

    def test_evaluate_without_metrics_source_is_a_warning(self, project):
        config = _config(project, evaluation={"metrics_file": None, "metrics_source": None})
        adapter = CommandAdapter(config, project)
        result = adapter.validate_project()
        assert any(i.field == "evaluation.metrics_source" for i in result.issues)


class TestTrain:
    def test_runs_the_command_and_detects_the_artifact(self, project):
        adapter = CommandAdapter(_config(project), project)
        result = adapter.train(_ctx(project, dataset_path="dataset"))

        assert result.status == "success"
        assert result.exit_code == 0
        assert "dataset='dataset'" in result.stdout_tail
        assert result.artifact is not None
        assert result.artifact.exists is True
        assert result.artifact.size_bytes and result.artifact.size_bytes > 0

    def test_nonzero_exit_is_reported_as_failed(self, project):
        config = _config(project, training={"command": f"{PY} -c \"import sys; sys.exit(3)\""})
        adapter = CommandAdapter(config, project)
        result = adapter.train(_ctx(project))
        assert result.status == "failed"
        assert result.exit_code == 3

    def test_timeout_is_reported_honestly(self, project):
        config = _config(
            project, training={"command": f"{PY} -c \"import time; time.sleep(5)\"", "timeout_seconds": 1}
        )
        adapter = CommandAdapter(config, project)
        result = adapter.train(_ctx(project))
        assert result.status == "timeout"
        assert result.exit_code is None

    def test_secrets_are_redacted_from_captured_output(self, project):
        config = _config(
            project,
            training={"command": f"{PY} -c \"print('OPENAI_API_KEY=sk-verysecretvalue123')\""},
        )
        adapter = CommandAdapter(config, project)
        result = adapter.train(_ctx(project))
        assert "sk-verysecretvalue123" not in result.stdout_tail
        assert "REDACTED" in result.stdout_tail


class TestEvaluate:
    def test_metrics_source_file(self, project):
        config = _config(project, evaluation={"metrics_source": "file", "metrics_file": "outputs/metrics.json"})
        adapter = CommandAdapter(config, project)
        result = adapter.evaluate(_ctx(project, dataset_path="dataset"))

        assert result.status == "success"
        assert result.metrics == {"accuracy": 0.968, "f1": 0.966}
        assert result.samples == 100
        assert result.errors == 3
        assert result.artifact == "outputs/best_model.pt"

    def test_metrics_source_stdout(self, project):
        config = _config(project, evaluation={"metrics_source": "stdout"})
        adapter = CommandAdapter(config, project)
        result = adapter.evaluate(_ctx(project, dataset_path="dataset"))

        assert result.status == "success"
        assert result.metrics == {"accuracy": 0.968, "f1": 0.966}

    def test_no_metrics_source_configured_yields_honest_empty_metrics(self, project):
        config = _config(project, evaluation={"metrics_source": None, "metrics_file": None})
        adapter = CommandAdapter(config, project)
        result = adapter.evaluate(_ctx(project, dataset_path="dataset"))

        assert result.status == "success"
        assert result.metrics == {}
        assert result.warnings  # must say *why* there are no metrics

    def test_failed_command_never_produces_fabricated_metrics(self, project):
        config = _config(
            project,
            evaluation={
                "command": f"{PY} -c \"import sys; sys.exit(1)\"",
                "metrics_source": "stdout",
            },
        )
        adapter = CommandAdapter(config, project)
        result = adapter.evaluate(_ctx(project))
        assert result.status == "failed"
        assert result.metrics == {}

    def test_get_metrics_reflects_the_last_evaluation(self, project):
        config = _config(project, evaluation={"metrics_source": "file", "metrics_file": "outputs/metrics.json"})
        adapter = CommandAdapter(config, project)

        assert adapter.get_metrics() is None  # nothing has run yet -- not {}
        adapter.evaluate(_ctx(project, dataset_path="dataset"))
        assert adapter.get_metrics() == {"accuracy": 0.968, "f1": 0.966}


class TestCancellation:
    def test_setting_the_event_stops_a_running_command_early(self, project):
        config = _config(project, training={"command": f"{PY} -c \"import time; time.sleep(30)\""})
        adapter = CommandAdapter(config, project)
        cancel_event = threading.Event()

        def cancel_soon():
            time.sleep(0.3)
            cancel_event.set()

        threading.Thread(target=cancel_soon).start()
        started = time.monotonic()
        result = adapter.train(_ctx(project), cancel_event=cancel_event)
        elapsed = time.monotonic() - started

        assert result.status == "cancelled"
        assert elapsed < 5  # nowhere near the full 30s sleep

    def test_never_cancelled_command_is_unaffected_by_an_unset_event(self, project):
        adapter = CommandAdapter(_config(project), project)
        result = adapter.train(_ctx(project, dataset_path="dataset"), cancel_event=threading.Event())
        assert result.status == "success"

    def test_cancelling_kills_the_whole_process_group_not_just_the_shell(self, project, tmp_path):
        # A grandchild process that outlives a plain kill of just the
        # parent shell (i.e. it merely gets orphaned, not terminated)
        # would still write its marker file once its own short sleep
        # elapses. Process-group killing (Section 20) must take it out
        # too, before that happens.
        marker = tmp_path / "child_finished"
        child_script = tmp_path / "child_sleep.py"
        child_script.write_text(f"import time\ntime.sleep(2)\nopen({str(marker)!r}, 'w').close()\n")
        parent_script = tmp_path / "spawn_child.py"
        parent_script.write_text(
            f"import subprocess, time\n"
            f"subprocess.Popen([{PY!r}, {str(child_script)!r}])\n"
            f"time.sleep(30)\n"
        )

        config = _config(project, training={"command": f"{PY} {parent_script}"})
        adapter = CommandAdapter(config, project)
        cancel_event = threading.Event()

        def cancel_soon():
            time.sleep(0.3)
            cancel_event.set()

        threading.Thread(target=cancel_soon).start()
        result = adapter.train(_ctx(project), cancel_event=cancel_event)
        assert result.status == "cancelled"

        time.sleep(2.5)  # past the child's own 2s sleep -- it must already be dead, not merely orphaned
        assert not marker.exists()


class TestGetArtifact:
    def test_before_training_artifact_does_not_exist_yet(self, tmp_path):
        # A bare project with no outputs/ at all -- honest "not produced
        # yet", not an error.
        empty_project = tmp_path / "empty"
        empty_project.mkdir()
        config = AIEnvConfig.from_dict(
            {
                "project": {"name": "x"},
                "training": {"command": f"{PY} train.py"},
                "artifact": {"path": "outputs/best_model.pt"},
            }
        )
        adapter = CommandAdapter(config, empty_project)
        info = adapter.get_artifact()
        assert info.exists is False
        assert info.size_bytes is None

    def test_path_traversal_in_artifact_path_raises(self, project):
        config = _config(project, artifact={"path": "../../outside.pt"})
        adapter = CommandAdapter(config, project)
        with pytest.raises(PathTraversalError):
            adapter.get_artifact()
