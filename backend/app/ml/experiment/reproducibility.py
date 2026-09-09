"""Phase 25 -- Experiment Reproducibility (Section 25/41).

"Every experiment should record: random seed, code version, configuration,
algorithm, hyperparameters, environment parameters, curriculum strategy,
training duration, model checkpoint, results. Provide a way to reproduce
an experiment."

Seed/configuration/algorithm/hyperparameters/curriculum strategy are
already fully captured by `ExperimentConfig` (Phase 15) and proven
reproducible there (two runs of the same config produce byte-identical
curriculum histories). This module adds the remaining pieces: code
version, training duration, and model checkpoint, bundled into one
`ReproducibilityManifest` per run, plus a `reproduce()` function that
actually re-runs a manifest's config and hands back a fresh
`ExperimentResult` for comparison against the original.

Code version: captured via `git rev-parse HEAD` against the repository
containing this project, with an explicit `is_dirty` flag (uncommitted
changes) -- if the tree is dirty, the commit hash alone does NOT guarantee
the code used to produce the run is byte-identical to what's checked in,
and this module says so rather than silently reporting just a hash. If the
directory isn't a git repository at all (or git isn't available), this is
reported honestly as "unavailable" rather than fabricating a fake version
string.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # backend/app/ml/experiment/reproducibility.py -> project-root/


@dataclass(frozen=True)
class GitInfo:
    available: bool
    commit: str | None
    is_dirty: bool | None

    def to_dict(self) -> dict:
        return {"available": self.available, "commit": self.commit, "is_dirty": self.is_dirty}


def get_git_info(repo_dir: Path | None = None) -> GitInfo:
    repo_dir = repo_dir or REPO_ROOT
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit_result.returncode != 0:
            return GitInfo(available=False, commit=None, is_dirty=None)
        commit = commit_result.stdout.strip()

        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        is_dirty = bool(status_result.stdout.strip()) if status_result.returncode == 0 else None

        return GitInfo(available=True, commit=commit, is_dirty=is_dirty)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return GitInfo(available=False, commit=None, is_dirty=None)


@dataclass
class ReproducibilityManifest:
    experiment_name: str
    config: dict
    git_info: GitInfo
    started_at: float
    ended_at: float
    checkpoint_path: str | None

    @property
    def duration_seconds(self) -> float:
        return self.ended_at - self.started_at

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "git_info": self.git_info.to_dict(),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "checkpoint_path": self.checkpoint_path,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ReproducibilityManifest":
        data = json.loads(Path(path).read_text())
        return cls(
            experiment_name=data["experiment_name"],
            config=data["config"],
            git_info=GitInfo(**data["git_info"]),
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            checkpoint_path=data["checkpoint_path"],
        )


class ManifestTimer:
    """Small helper so ExperimentManager doesn't need to juggle raw
    `time.time()` calls itself -- just wrap the run in `with
    ManifestTimer() as timer:` and read `timer.started_at`/`ended_at`
    afterward.
    """

    def __enter__(self) -> "ManifestTimer":
        self.started_at = time.time()
        return self

    def __exit__(self, *exc_info) -> None:
        self.ended_at = time.time()


def reproduce(manifest_path: str | Path):
    """Reloads a manifest's config and re-runs it, returning a fresh
    ExperimentResult. Callers compare this against the original run's
    stored results to verify reproduction actually holds -- this function
    itself only re-runs; it doesn't diff (see tests for that comparison).

    If the manifest's git tree was dirty when the original run happened,
    this is surfaced as a warning return value rather than silently
    proceeding as if reproducibility were guaranteed.
    """
    from app.ml.experiment.config import ExperimentConfig
    from app.ml.experiment.manager import ExperimentManager

    manifest = ReproducibilityManifest.load(manifest_path)
    warning = None
    if manifest.git_info.is_dirty:
        warning = (
            "Original run's git working tree was dirty (uncommitted changes) -- "
            "exact code-level reproducibility is not guaranteed, only config-level."
        )
    elif not manifest.git_info.available:
        warning = "Original run had no git information available -- code version cannot be verified."

    config = ExperimentConfig.from_dict(manifest.config)
    result = ExperimentManager(config).run()
    return result, warning
