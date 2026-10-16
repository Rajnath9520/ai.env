"""Section 41 ("Model Comparison... Do NOT use fake improvement values"),
generalized to Mode B: turns a project's completed `ProjectRun`s into
metric trends, using only numbers that were actually reported by the
project's own evaluation command.

This deliberately stops short of Sections 25-30's task-specific failure
analyzers (per-condition weakness vectors like `lighting`/`occlusion`).
Those require a structured evaluation harness that breaks results down by
condition, which nothing in Mode B provides -- a `CommandAdapter`
evaluation only ever produces a flat `{metric_name: value}` dict (Section
24's contract). Inventing a weakness breakdown on top of that would be
exactly the fabrication Section 88 forbids, so this module only computes
what a flat metrics history actually supports: how each metric moved
between runs, and how far the configured target metric is from its target
value. It never guesses whether a metric is "higher is better" or "lower
is better" -- a raw trend and a target *distance* (absolute difference)
are the only claims that hold regardless of that direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence

from app.ml.project.config import OptimizationConfig


class RunLike(Protocol):
    """The minimal shape `compute_progress` needs from a run -- satisfied
    by the `ProjectRun` ORM model without this module importing it (this
    package stays free of any web/DB framework import, see
    `app/ml/project/__init__.py`'s sibling modules)."""

    run_number: int
    status: str
    evaluation_result: dict | None


@dataclass
class MetricTrendPoint:
    run_number: int
    value: float


@dataclass
class MetricTrend:
    metric: str
    points: list[MetricTrendPoint] = field(default_factory=list)
    latest_value: float = 0.0
    previous_value: float | None = None
    # latest - previous. Deliberately unlabeled as "improvement" or
    # "regression" -- that requires knowing whether this metric is
    # higher-is-better or lower-is-better, which nothing here can know.
    delta_from_previous: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProjectProgress:
    metrics: list[MetricTrend] = field(default_factory=list)
    completed_run_count: int = 0
    target_metric: str | None = None
    target_value: float | None = None
    latest_target_value: float | None = None
    # abs(latest_target_value - target_value) -- direction-agnostic, so
    # always a valid claim regardless of which way the metric is "better".
    distance_to_target: float | None = None
    # Whether the most recent run's distance-to-target is smaller than
    # the previous run's. `None` when there's no target configured, the
    # target metric was never reported, or fewer than two data points
    # exist -- never guessed.
    moved_closer_to_target: bool | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics"] = [m.to_dict() for m in self.metrics]
        return d


def compute_progress(runs: Sequence[RunLike], optimization: OptimizationConfig) -> ProjectProgress:
    completed = sorted((r for r in runs if r.status == "completed"), key=lambda r: r.run_number)
    with_metrics = [r for r in completed if r.evaluation_result and r.evaluation_result.get("metrics")]

    metric_names: list[str] = []
    for run in with_metrics:
        for name in run.evaluation_result["metrics"]:
            if name not in metric_names:
                metric_names.append(name)

    trends: list[MetricTrend] = []
    for name in metric_names:
        points = [
            MetricTrendPoint(run_number=run.run_number, value=float(run.evaluation_result["metrics"][name]))
            for run in with_metrics
            if name in run.evaluation_result["metrics"]
            and isinstance(run.evaluation_result["metrics"][name], (int, float))
            and not isinstance(run.evaluation_result["metrics"][name], bool)
        ]
        if not points:
            continue
        previous_value = points[-2].value if len(points) >= 2 else None
        trends.append(
            MetricTrend(
                metric=name,
                points=points,
                latest_value=points[-1].value,
                previous_value=previous_value,
                delta_from_previous=(points[-1].value - previous_value) if previous_value is not None else None,
            )
        )

    progress = ProjectProgress(
        metrics=trends,
        completed_run_count=len(completed),
        target_metric=optimization.target_metric,
        target_value=optimization.target_value,
    )

    if optimization.target_metric:
        matching = next((t for t in trends if t.metric == optimization.target_metric), None)
        if matching:
            progress.latest_target_value = matching.latest_value
            if optimization.target_value is not None:
                progress.distance_to_target = abs(matching.latest_value - optimization.target_value)
                if matching.previous_value is not None:
                    previous_distance = abs(matching.previous_value - optimization.target_value)
                    progress.moved_closer_to_target = progress.distance_to_target < previous_distance

    return progress
