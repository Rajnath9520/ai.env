"""Section 15/40 -- ai.env-managed artifact storage.

A `CommandAdapter.train()` run writes its artifact wherever the user's own
project script writes it (`artifact.path` in `ai-env.yaml`), and re-running
that same script typically overwrites that path in place -- there is
nothing stopping a project's own `train.py` from clobbering
`outputs/best_model.pt` on every run. Section 40: "Never silently
overwrite the previous model." `archive_run_artifact`/
`archive_downloaded_artifact` are how ai.env keeps its own copy, indexed
by project and run number, entirely outside the user's project directory
(Section 61's LOCAL PROJECT vs. AI.env-managed storage distinction) --
never written into their repo, never touched unless a run actually
produced something to copy.

Two entry points share one managed-path convention
(`_run_storage_dir`): `archive_run_artifact` copies a local file (a
same-machine, no-agent run); `archive_downloaded_artifact` writes bytes
already fetched over HTTP (an agent-backed run, Section 5 -- the backend
has no local filesystem access to the agent's copy, only whatever
`AgentClient.download_artifact` retrieved).
"""

from __future__ import annotations

import shutil
from pathlib import Path


def _run_storage_dir(storage_root: str | Path, project_id: int, run_number: int) -> Path:
    return Path(storage_root) / "projects" / str(project_id) / "runs" / str(run_number)


def archive_run_artifact(storage_root: str | Path, project_id: int, run_number: int, artifact_path: Path) -> str | None:
    """Copies `artifact_path` into `{storage_root}/projects/{project_id}/
    runs/{run_number}/{artifact_path.name}` and returns that managed path
    as a string, or `None` if `artifact_path` doesn't exist (a failed or
    still-running training command -- honestly nothing to archive yet)."""
    if not artifact_path.is_file():
        return None

    destination_dir = _run_storage_dir(storage_root, project_id, run_number)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / artifact_path.name
    shutil.copy2(artifact_path, destination)
    return str(destination)


def archive_downloaded_artifact(
    storage_root: str | Path, project_id: int, run_number: int, filename: str, content: bytes
) -> str:
    """Same managed-path convention as `archive_run_artifact`, for bytes
    already downloaded from a remote Agent rather than a local file."""
    destination_dir = _run_storage_dir(storage_root, project_id, run_number)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    destination.write_bytes(content)
    return str(destination)
