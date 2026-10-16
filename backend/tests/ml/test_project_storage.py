from pathlib import Path

from app.ml.project.storage import archive_downloaded_artifact, archive_run_artifact


class TestArchiveRunArtifact:
    def test_missing_artifact_returns_none_and_writes_nothing(self, tmp_path):
        storage_root = tmp_path / "storage"
        result = archive_run_artifact(storage_root, project_id=1, run_number=1, artifact_path=tmp_path / "nope.pt")
        assert result is None
        assert not storage_root.exists()

    def test_copies_artifact_into_a_project_and_run_scoped_path(self, tmp_path):
        artifact = tmp_path / "outputs" / "best_model.pt"
        artifact.parent.mkdir()
        artifact.write_bytes(b"checkpoint-v1")
        storage_root = tmp_path / "storage"

        managed_path = archive_run_artifact(storage_root, project_id=7, run_number=1, artifact_path=artifact)

        assert managed_path is not None
        assert Path(managed_path).read_bytes() == b"checkpoint-v1"
        assert "projects/7/runs/1" in managed_path.replace("\\", "/")

    def test_a_second_run_does_not_overwrite_the_first(self, tmp_path):
        artifact = tmp_path / "outputs" / "best_model.pt"
        artifact.parent.mkdir()
        storage_root = tmp_path / "storage"

        artifact.write_bytes(b"checkpoint-v1")
        first = archive_run_artifact(storage_root, project_id=7, run_number=1, artifact_path=artifact)

        artifact.write_bytes(b"checkpoint-v2")  # the user's own script overwrote its own output path
        second = archive_run_artifact(storage_root, project_id=7, run_number=2, artifact_path=artifact)

        assert first != second
        assert Path(first).read_bytes() == b"checkpoint-v1"
        assert Path(second).read_bytes() == b"checkpoint-v2"

    def test_project_and_run_are_kept_separate(self, tmp_path):
        artifact = tmp_path / "best_model.pt"
        artifact.write_bytes(b"x")
        storage_root = tmp_path / "storage"

        a = archive_run_artifact(storage_root, project_id=1, run_number=1, artifact_path=artifact)
        b = archive_run_artifact(storage_root, project_id=2, run_number=1, artifact_path=artifact)
        assert a != b


class TestArchiveDownloadedArtifact:
    def test_writes_bytes_to_the_same_managed_path_shape(self, tmp_path):
        storage_root = tmp_path / "storage"
        managed_path = archive_downloaded_artifact(
            storage_root, project_id=7, run_number=1, filename="best_model.pt", content=b"from-agent"
        )
        assert Path(managed_path).read_bytes() == b"from-agent"
        assert "projects/7/runs/1" in managed_path.replace("\\", "/")

    def test_local_and_downloaded_archiving_use_the_same_directory_shape(self, tmp_path):
        local_artifact = tmp_path / "outputs" / "best_model.pt"
        local_artifact.parent.mkdir()
        local_artifact.write_bytes(b"local")
        storage_root = tmp_path / "storage"

        local_path = archive_run_artifact(storage_root, project_id=1, run_number=1, artifact_path=local_artifact)
        downloaded_path = archive_downloaded_artifact(
            storage_root, project_id=1, run_number=2, filename="best_model.pt", content=b"remote"
        )
        assert Path(local_path).parent.parent == Path(downloaded_path).parent.parent
