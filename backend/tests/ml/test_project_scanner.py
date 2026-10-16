from pathlib import Path

import pytest

from app.ml.project.scanner import ProjectScanner

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "test-ai-project"


class TestScanningTheFixtureProject:
    def test_detects_python_and_pytorch(self):
        result = ProjectScanner().scan(FIXTURE_ROOT)
        assert result.language == "python"
        assert result.framework == "pytorch"

    def test_finds_training_and_evaluation_candidates(self):
        result = ProjectScanner().scan(FIXTURE_ROOT)
        assert "train.py" in result.training_candidates
        assert "evaluate.py" in result.evaluation_candidates

    def test_finds_dataset_candidate(self):
        result = ProjectScanner().scan(FIXTURE_ROOT)
        assert "dataset/" in result.dataset_candidates

    def test_lists_dependency_files(self):
        result = ProjectScanner().scan(FIXTURE_ROOT)
        assert "requirements.txt" in result.dependency_files

    def test_never_reports_a_single_entrypoint_only_candidates(self):
        # Section 6: scan results are always candidate *lists*, even when
        # there happens to be exactly one match.
        result = ProjectScanner().scan(FIXTURE_ROOT)
        assert isinstance(result.training_candidates, list)
        assert isinstance(result.evaluation_candidates, list)

    def test_to_dict_matches_section_7_shape(self):
        result = ProjectScanner().scan(FIXTURE_ROOT)
        data = result.to_dict()
        assert set(data.keys()) == {
            "root",
            "language",
            "framework",
            "python_version",
            "gpu",
            "training_candidates",
            "evaluation_candidates",
            "dataset_candidates",
            "artifact_candidates",
            "dependency_files",
        }
        assert set(data["gpu"].keys()) == {"available", "name"}


class TestArtifactCandidateDetection:
    def test_finds_existing_checkpoint_files(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "best_model.pt").write_bytes(b"x")
        (outputs / "notes.txt").write_text("not an artifact")

        result = ProjectScanner().scan(tmp_path)
        assert "outputs/best_model.pt" in result.artifact_candidates
        assert not any(p.endswith("notes.txt") for p in result.artifact_candidates)


class TestHonestDegradation:
    """Section 85/88 -- an undocumented or empty project must degrade to
    `None`/empty lists, never a fabricated guess."""

    def test_empty_directory_yields_no_confident_guesses(self, tmp_path):
        result = ProjectScanner().scan(tmp_path)
        assert result.language is None
        assert result.framework is None
        assert result.training_candidates == []
        assert result.evaluation_candidates == []
        assert result.dataset_candidates == []
        assert result.artifact_candidates == []
        assert result.gpu_available is False
        assert result.gpu_name is None

    def test_missing_directory_raises_rather_than_returning_empty_result(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            ProjectScanner().scan(tmp_path / "does-not-exist")

    def test_does_not_descend_into_dataset_or_output_directories(self, tmp_path):
        # A stray .py file inside dataset/ (e.g. a preprocessing script
        # shipped with the data) must not count toward language/framework
        # detection or be treated as a training candidate.
        (tmp_path / "dataset").mkdir()
        (tmp_path / "dataset" / "loader.py").write_text("import tensorflow")
        (tmp_path / "train.py").write_text("import torch")

        result = ProjectScanner().scan(tmp_path)
        assert result.framework == "pytorch"
