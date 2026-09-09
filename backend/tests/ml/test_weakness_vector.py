import pytest

from app.ml.analysis.failure_analyzer import FailureReport
from app.ml.analysis.weakness_vector import (
    SECONDARY_WEAKNESS_THRESHOLD,
    WEAKNESS_DIMENSIONS,
    WeaknessVectorBuilder,
)

ALL_FAILURE_TYPES = (
    "obstacle_avoidance",
    "dead_end_recovery",
    "exploration",
    "repeated_action_loops",
    "path_inefficiency",
    "short_horizon_planning",
    "long_horizon_planning",
)


def _flat_episode_reports(scores: dict[str, float], confidence: float = 1.0) -> list[FailureReport]:
    """Builds one episode's worth of reports (all 7 types), defaulting
    unspecified types to a score of 0.0."""
    return [
        FailureReport(failure_type=ft, score=scores.get(ft, 0.0), confidence=confidence, features={})
        for ft in ALL_FAILURE_TYPES
    ]


class TestWeaknessVectorBuilder:
    def test_all_six_dimensions_present_and_bounded(self):
        reports = _flat_episode_reports({"obstacle_avoidance": 0.5})
        vector = WeaknessVectorBuilder().build(reports)
        for dim in WEAKNESS_DIMENSIONS:
            value = getattr(vector, dim)
            assert 0.0 <= value <= 1.0

    def test_pure_dimension_passthrough(self):
        reports = _flat_episode_reports({"long_horizon_planning": 0.9})
        vector = WeaknessVectorBuilder().build(reports)
        assert vector.long_horizon_planning == pytest.approx(0.9)
        assert vector.obstacle_avoidance == 0.0

    def test_path_efficiency_blends_inefficiency_and_loops(self):
        reports = _flat_episode_reports(
            {"path_inefficiency": 1.0, "repeated_action_loops": 1.0}
        )
        vector = WeaknessVectorBuilder().build(reports)
        assert vector.path_efficiency == pytest.approx(0.7 * 1.0 + 0.3 * 1.0)

    def test_short_horizon_blends_short_horizon_and_loops(self):
        reports = _flat_episode_reports(
            {"short_horizon_planning": 1.0, "repeated_action_loops": 1.0}
        )
        vector = WeaknessVectorBuilder().build(reports)
        assert vector.short_horizon_planning == pytest.approx(0.6 * 1.0 + 0.4 * 1.0)

    def test_primary_weakness_is_measured_not_random(self):
        reports = _flat_episode_reports({"dead_end_recovery": 0.95})
        vector = WeaknessVectorBuilder().build(reports)
        assert vector.primary_weakness == "dead_end_recovery"

    def test_secondary_weaknesses_above_threshold_only(self):
        reports = _flat_episode_reports(
            {
                "dead_end_recovery": 0.95,          # primary
                "obstacle_avoidance": SECONDARY_WEAKNESS_THRESHOLD + 0.1,  # secondary
                "exploration": SECONDARY_WEAKNESS_THRESHOLD - 0.1,        # below threshold, excluded
            }
        )
        vector = WeaknessVectorBuilder().build(reports)
        assert vector.primary_weakness == "dead_end_recovery"
        assert "obstacle_avoidance" in vector.secondary_weaknesses
        assert "exploration" not in vector.secondary_weaknesses

    def test_secondary_weaknesses_sorted_descending(self):
        reports = _flat_episode_reports(
            {
                "long_horizon_planning": 0.9,      # primary
                "obstacle_avoidance": 0.5,
                "exploration": 0.8,
            }
        )
        vector = WeaknessVectorBuilder().build(reports)
        assert vector.secondary_weaknesses[0] == "exploration"
        assert vector.secondary_weaknesses[1] == "obstacle_avoidance"

    def test_multi_episode_aggregation_is_confidence_weighted(self):
        # Two episodes report obstacle_avoidance: one low-confidence high
        # score, one high-confidence low score. The high-confidence report
        # should dominate the aggregate.
        low_conf_high_score = FailureReport("obstacle_avoidance", score=0.9, confidence=0.1, features={})
        high_conf_low_score = FailureReport("obstacle_avoidance", score=0.1, confidence=0.9, features={})
        others_ep1 = _flat_episode_reports({})[1:]  # remaining 6 types, all zero
        others_ep2 = _flat_episode_reports({})[1:]

        vector = WeaknessVectorBuilder().build(
            [low_conf_high_score, *others_ep1, high_conf_low_score, *others_ep2]
        )
        # Weighted mean: (0.9*0.1 + 0.1*0.9) / (0.1+0.9) = 0.18
        assert vector.obstacle_avoidance == pytest.approx(0.18, abs=1e-6)

    def test_episodes_analyzed_inferred_from_report_counts(self):
        reports = _flat_episode_reports({}) * 3  # simulate 3 episodes worth
        vector = WeaknessVectorBuilder().build(reports)
        assert vector.episodes_analyzed == 3

    def test_empty_reports_yields_zero_vector(self):
        vector = WeaknessVectorBuilder().build([])
        for dim in WEAKNESS_DIMENSIONS:
            assert getattr(vector, dim) == 0.0
        assert vector.episodes_analyzed == 0
        assert vector.secondary_weaknesses == ()

    def test_to_dict_and_as_array_shapes(self):
        reports = _flat_episode_reports({"exploration": 0.6})
        vector = WeaknessVectorBuilder().build(reports)
        assert len(vector.as_array()) == 6
        d = vector.to_dict()
        assert set(WEAKNESS_DIMENSIONS).issubset(d.keys())
        assert "primary_weakness" in d and "secondary_weaknesses" in d
