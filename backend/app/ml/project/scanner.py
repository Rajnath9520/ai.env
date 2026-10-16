"""Section 6/7 -- the project scanner.

Detects language/framework/candidate files from an existing AI project
directory. Every list here is a set of *candidates*: Section 6 is explicit
that "train.py" being present must never be silently treated as "the
training entrypoint" -- that confirmation is the user's, not the
scanner's. `ProjectScanner` therefore has no notion of a single "detected
entrypoint"; it always returns lists, even when a list has exactly one
item.

Detection is entirely best-effort and never raises for an unrecognized or
partially-structured project (Section 85's "Honest support model" --
"limited support" for undocumented projects is a valid, expected outcome,
not a bug). GPU/Python-version detection in particular are wrapped so a
missing `nvidia-smi` or `python3` on PATH degrades to `None`/`False`
rather than crashing the scan.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Section 6 -- candidate filenames/directories to look for. Order matters
# only for framework detection (first match wins, see `_detect_framework`);
# everywhere else every match found is kept.
_TRAINING_FILENAMES = ("train.py", "training.py", "trainer.py", "train_model.py", "main.py")
_TRAINING_SEARCH_DIRS = (".", "scripts")

_EVALUATION_FILENAMES = ("evaluate.py", "eval.py", "validation.py", "test.py", "inference.py")
_EVALUATION_SEARCH_DIRS = (".", "scripts")

_DATASET_DIRNAMES = ("dataset", "datasets", "data", "train", "validation", "val", "test")
_ARTIFACT_DIRNAMES = ("outputs", "checkpoints", "weights", "models", "artifacts")
_ARTIFACT_EXTENSIONS = (".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".keras", ".onnx", ".pkl", ".joblib")

_DEPENDENCY_FILENAMES = (
    "requirements.txt",
    "pyproject.toml",
    "environment.yml",
    "setup.py",
    "Pipfile",
    "package.json",
)

# Section 6 -- (framework name, marker strings to look for in dependency
# files or source). Checked in this order; the first framework with any
# marker present wins, since a project naming multiple ML libraries in
# requirements.txt is usually only actually built on one of them and
# reporting the most specific/heaviest match first is more useful than a
# random one. Ordering: more specific frameworks before their general
# umbrella libraries (e.g. ultralytics/stable-baselines3 before the bare
# torch they're built on).
_FRAMEWORK_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ultralytics", ("ultralytics", "yolo")),
    ("stable-baselines3", ("stable-baselines3", "stable_baselines3")),
    ("gymnasium", ("gymnasium", "gym")),
    ("huggingface_transformers", ("transformers", "huggingface")),
    ("xgboost", ("xgboost",)),
    ("lightgbm", ("lightgbm",)),
    ("scikit-learn", ("scikit-learn", "sklearn")),
    ("jax", ("jax", "jaxlib")),
    ("keras", ("keras",)),
    ("tensorflow", ("tensorflow", "tf-nightly")),
    ("onnx", ("onnx",)),
    ("pytorch", ("torch", "pytorch")),
)

_LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".kt": "kotlin",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
}

_SCAN_IGNORE_DIRNAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    ".ai-env",
    "dataset",
    "datasets",
    "data",
    "outputs",
    "checkpoints",
    "weights",
    "models",
    "artifacts",
}


@dataclass
class ScanResult:
    root: str
    language: str | None
    framework: str | None
    python_version: str | None
    gpu_available: bool
    gpu_name: str | None
    training_candidates: list[str] = field(default_factory=list)
    evaluation_candidates: list[str] = field(default_factory=list)
    dataset_candidates: list[str] = field(default_factory=list)
    artifact_candidates: list[str] = field(default_factory=list)
    dependency_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "language": self.language,
            "framework": self.framework,
            "python_version": self.python_version,
            "gpu": {"available": self.gpu_available, "name": self.gpu_name},
            "training_candidates": self.training_candidates,
            "evaluation_candidates": self.evaluation_candidates,
            "dataset_candidates": self.dataset_candidates,
            "artifact_candidates": self.artifact_candidates,
            "dependency_files": self.dependency_files,
        }


class ProjectScanner:
    """Stateless -- one call to `scan()` per invocation, no caching. The
    caller (the project service) decides whether/when a rescan is
    triggered (Section 47's "Rescan" action)."""

    def scan(self, root: str | Path) -> ScanResult:
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a directory")

        dependency_files = self._find_dependency_files(root)
        return ScanResult(
            root=str(root),
            language=self._detect_language(root),
            framework=self._detect_framework(root, dependency_files),
            python_version=self._detect_python_version(root),
            gpu_available=self._detect_gpu()[0],
            gpu_name=self._detect_gpu()[1],
            training_candidates=self._find_file_candidates(root, _TRAINING_FILENAMES, _TRAINING_SEARCH_DIRS),
            evaluation_candidates=self._find_file_candidates(
                root, _EVALUATION_FILENAMES, _EVALUATION_SEARCH_DIRS
            ),
            dataset_candidates=self._find_dir_candidates(root, _DATASET_DIRNAMES),
            artifact_candidates=self._find_artifact_candidates(root),
            dependency_files=dependency_files,
        )

    # -- language -----------------------------------------------------

    def _detect_language(self, root: Path) -> str | None:
        counts: dict[str, int] = {}
        for path in self._walk_source_files(root):
            lang = _LANGUAGE_EXTENSIONS.get(path.suffix)
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda lang: counts[lang])

    def _walk_source_files(self, root: Path):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SCAN_IGNORE_DIRNAMES for part in path.relative_to(root).parts[:-1]):
                continue
            yield path

    # -- framework ------------------------------------------------------

    def _find_dependency_files(self, root: Path) -> list[str]:
        return [name for name in _DEPENDENCY_FILENAMES if (root / name).is_file()]

    def _detect_framework(self, root: Path, dependency_files: list[str]) -> str | None:
        haystack_parts: list[str] = []
        for name in dependency_files:
            try:
                haystack_parts.append((root / name).read_text(errors="ignore").lower())
            except OSError:
                continue

        # Fall back to scanning the first-level .py files' imports when no
        # dependency file names a framework (e.g. a project with no
        # requirements.txt at all, dependencies installed ad hoc).
        for path in root.glob("*.py"):
            try:
                haystack_parts.append(path.read_text(errors="ignore").lower())
            except OSError:
                continue

        haystack = "\n".join(haystack_parts)
        for framework, markers in _FRAMEWORK_MARKERS:
            if any(marker in haystack for marker in markers):
                return framework
        return None

    # -- candidate files/dirs -------------------------------------------

    def _find_file_candidates(self, root: Path, filenames: tuple[str, ...], search_dirs: tuple[str, ...]) -> list[str]:
        found: list[str] = []
        for search_dir in search_dirs:
            base = root if search_dir == "." else root / search_dir
            if not base.is_dir():
                continue
            for filename in filenames:
                candidate = base / filename
                if candidate.is_file():
                    found.append(str(candidate.relative_to(root)))
        return found

    def _find_dir_candidates(self, root: Path, dirnames: tuple[str, ...]) -> list[str]:
        return [f"{name}/" for name in dirnames if (root / name).is_dir()]

    def _find_artifact_candidates(self, root: Path) -> list[str]:
        found: list[str] = []
        for dirname in _ARTIFACT_DIRNAMES:
            base = root / dirname
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in _ARTIFACT_EXTENSIONS:
                    found.append(str(path.relative_to(root)))
        return found

    # -- environment ------------------------------------------------------

    def _detect_python_version(self, root: Path) -> str | None:
        version_file = root / ".python-version"
        if version_file.is_file():
            return version_file.read_text().strip()
        for python in ("python3", "python"):
            try:
                proc = subprocess.run([python, "--version"], capture_output=True, text=True, timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                continue
            output = (proc.stdout or proc.stderr).strip()
            if output.lower().startswith("python"):
                return output.split()[-1]
        return None

    def _detect_gpu(self) -> tuple[bool, str | None]:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False, None
        if proc.returncode != 0:
            return False, None
        name = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else None
        return bool(name), name
