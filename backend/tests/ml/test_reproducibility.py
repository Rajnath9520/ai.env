import subprocess
import tempfile
from pathlib import Path

import pytest

from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.manager import ExperimentManager
from app.ml.experiment.reproducibility import (
    GitInfo,
    ManifestTimer,
    ReproducibilityManifest,
    get_git_info,
    reproduce,
)

TINY_KWARGS = dict(
    agent_mode="demo_ppo",
    grid_size=5,
    obstacle_density=0.1,
    dead_end_probability=0.1,
    path_complexity=0.1,
    max_steps_per_episode=12,
    n_curriculum_steps=2,
    timesteps_per_step=256,
    eval_episodes=3,
    run_generalization=False,
)


class TestGetGitInfo:
    def test_non_git_directory_reports_unavailable(self, tmp_path):
        info = get_git_info(tmp_path)
        assert info.available is False
        assert info.commit is None
        assert info.is_dirty is None

    def test_real_git_repo_reports_commit_and_clean_state(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True, check=True)

        info = get_git_info(tmp_path)
        assert info.available is True
        assert info.commit is not None and len(info.commit) == 40
        assert info.is_dirty is False

    def test_dirty_working_tree_is_detected(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True, check=True)

        (tmp_path / "file.txt").write_text("modified")

        info = get_git_info(tmp_path)
        assert info.is_dirty is True


class TestManifestTimer:
    def test_records_start_and_end(self):
        import time

        with ManifestTimer() as timer:
            time.sleep(0.01)
        assert timer.ended_at > timer.started_at


class TestReproducibilityManifestRoundTrip:
    def test_save_load_round_trip(self):
        manifest = ReproducibilityManifest(
            experiment_name="test_exp",
            config={"seed": 1, "name": "test_exp"},
            git_info=GitInfo(available=True, commit="abc123", is_dirty=False),
            started_at=100.0,
            ended_at=142.5,
            checkpoint_path="experiments/results/checkpoints/test_exp.zip",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest.save(path)
            loaded = ReproducibilityManifest.load(path)

        assert loaded.experiment_name == "test_exp"
        assert loaded.git_info.commit == "abc123"
        assert loaded.duration_seconds == pytest.approx(42.5)
        assert loaded.checkpoint_path == manifest.checkpoint_path


class TestExperimentManagerProducesManifest:
    def test_run_result_includes_reproducibility_manifest(self):
        config = ExperimentConfig(name="repro_manifest_test", curriculum_type="random", seed=1, **TINY_KWARGS)
        result = ExperimentManager(config).run()

        assert result.reproducibility is not None
        assert result.reproducibility["experiment_name"] == "repro_manifest_test"
        assert result.reproducibility["duration_seconds"] > 0
        assert "git_info" in result.reproducibility
        assert result.reproducibility["checkpoint_path"] is not None
        assert Path(result.reproducibility["checkpoint_path"]).exists()

    def test_save_checkpoint_false_skips_checkpoint(self):
        config = ExperimentConfig(name="repro_no_checkpoint_test", curriculum_type="random", seed=1, **TINY_KWARGS)
        result = ExperimentManager(config).run(save_checkpoint=False)
        assert result.reproducibility["checkpoint_path"] is None


class TestReproduceFunction:
    def test_reproduce_matches_original_curriculum_history(self):
        config = ExperimentConfig(name="repro_full_test", curriculum_type="random", seed=3, **TINY_KWARGS)
        original_result = ExperimentManager(config).run()

        manifest_path = "experiments/results/manifests/repro_full_test.json"
        reproduced_result, warning = reproduce(manifest_path)

        original_history = [s.to_dict() for s in original_result.curriculum_history]
        reproduced_history = [s.to_dict() for s in reproduced_result.curriculum_history]
        assert original_history == reproduced_history

    def test_reproduce_returns_warning_when_git_unavailable_or_dirty(self):
        config = ExperimentConfig(name="repro_warning_test", curriculum_type="random", seed=1, **TINY_KWARGS)
        ExperimentManager(config).run()

        manifest_path = "experiments/results/manifests/repro_warning_test.json"
        _, warning = reproduce(manifest_path)
        assert warning is not None
