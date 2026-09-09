"""Converts failure analysis output (Phase 4) into the normalized weakness
vector Section 12 specifies, plus a derived primary/secondary weakness
label the environment synthesizer (Phase 9) will target.

Design choice: Phase 4 detects 7 failure types, but Section 12's vector has
6 dimensions — `repeated_action_loops` has no dedicated slot. Rather than
silently dropping it or inventing a 7th vector dimension not in the spec,
it's blended into the two dimensions it behaviorally overlaps with:
oscillating back and forth is both a wasted-path-length problem
(path_efficiency) and a poor-immediate-decision problem
(short_horizon_planning). The blend weights are explicit constants below,
not hidden.

Aggregation: a weakness vector should reflect an agent's behavior across
many episodes, not one trajectory's noise. Each failure type's contribution
is a confidence-weighted average across all supplied reports of that type,
so a low-confidence outlier report (e.g. a very short episode) influences
the result less than several high-confidence ones.

Limitations: primary/secondary weakness selection is a simple thresholded
argmax over 6 independently-aggregated dimensions — it does not account for
correlations between dimensions (e.g. dead_end_recovery and long_horizon
often co-occur structurally). That correlation, if real, should show up
empirically once real experiment data exists, rather than being assumed
here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from app.ml.analysis.failure_analyzer import FailureReport

WEAKNESS_DIMENSIONS: tuple[str, ...] = (
    "obstacle_avoidance",
    "exploration",
    "dead_end_recovery",
    "path_efficiency",
    "short_horizon_planning",
    "long_horizon_planning",
)

# Documented mapping from the 7 detected failure types to the 6 weakness
# dimensions. Each dimension's value = sum(weight * aggregated_score) for
# its listed source failure types; weights per dimension sum to 1.0.
_DIMENSION_SOURCES: dict[str, dict[str, float]] = {
    "obstacle_avoidance": {"obstacle_avoidance": 1.0},
    "exploration": {"exploration": 1.0},
    "dead_end_recovery": {"dead_end_recovery": 1.0},
    "path_efficiency": {"path_inefficiency": 0.7, "repeated_action_loops": 0.3},
    "short_horizon_planning": {"short_horizon_planning": 0.6, "repeated_action_loops": 0.4},
    "long_horizon_planning": {"long_horizon_planning": 1.0},
}

SECONDARY_WEAKNESS_THRESHOLD = 0.4


@dataclass(frozen=True)
class WeaknessVector:
    obstacle_avoidance: float
    exploration: float
    dead_end_recovery: float
    path_efficiency: float
    short_horizon_planning: float
    long_horizon_planning: float
    primary_weakness: str
    secondary_weaknesses: tuple[str, ...] = field(default_factory=tuple)
    episodes_analyzed: int = 0

    def as_array(self) -> list[float]:
        return [getattr(self, dim) for dim in WEAKNESS_DIMENSIONS]

    def to_dict(self) -> dict:
        return {
            **{dim: round(getattr(self, dim), 4) for dim in WEAKNESS_DIMENSIONS},
            "primary_weakness": self.primary_weakness,
            "secondary_weaknesses": list(self.secondary_weaknesses),
            "episodes_analyzed": self.episodes_analyzed,
        }


class WeaknessVectorBuilder:
    def build(
        self,
        failure_reports: Sequence[FailureReport],
        episodes_analyzed: int | None = None,
    ) -> WeaknessVector:
        aggregated_by_type = self._aggregate_by_type(failure_reports)

        dimension_values: dict[str, float] = {}
        for dimension, sources in _DIMENSION_SOURCES.items():
            dimension_values[dimension] = sum(
                weight * aggregated_by_type.get(failure_type, 0.0)
                for failure_type, weight in sources.items()
            )

        primary = max(WEAKNESS_DIMENSIONS, key=lambda d: dimension_values[d])
        secondary = tuple(
            sorted(
                (
                    d
                    for d in WEAKNESS_DIMENSIONS
                    if d != primary and dimension_values[d] >= SECONDARY_WEAKNESS_THRESHOLD
                ),
                key=lambda d: dimension_values[d],
                reverse=True,
            )
        )

        return WeaknessVector(
            **{dim: dimension_values[dim] for dim in WEAKNESS_DIMENSIONS},
            primary_weakness=primary,
            secondary_weaknesses=secondary,
            episodes_analyzed=(
                episodes_analyzed if episodes_analyzed is not None else self._infer_episode_count(failure_reports)
            ),
        )

    @staticmethod
    def _aggregate_by_type(failure_reports: Sequence[FailureReport]) -> dict[str, float]:
        scores_by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for report in failure_reports:
            scores_by_type[report.failure_type].append((report.score, report.confidence))

        aggregated: dict[str, float] = {}
        for failure_type, pairs in scores_by_type.items():
            total_confidence = sum(c for _, c in pairs)
            if total_confidence > 0:
                aggregated[failure_type] = sum(s * c for s, c in pairs) / total_confidence
            else:
                aggregated[failure_type] = sum(s for s, _ in pairs) / len(pairs)
        return aggregated

    @staticmethod
    def _infer_episode_count(failure_reports: Sequence[FailureReport]) -> int:
        # Every episode contributes exactly one report per failure type
        # (FailureAnalyzer.analyze always returns all 7), so the max count
        # for any single failure type equals the number of episodes.
        counts: dict[str, int] = defaultdict(int)
        for report in failure_reports:
            counts[report.failure_type] += 1
        return max(counts.values(), default=0)
