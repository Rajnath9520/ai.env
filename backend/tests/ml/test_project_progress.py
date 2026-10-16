from dataclasses import dataclass

import pytest

from app.ml.project.config import OptimizationConfig
from app.ml.project.progress import MetricTrendPoint, compute_progress


@dataclass
class FakeRun:
    run_number: int
    status: str
    evaluation_result: dict | None


def _optimization(**overrides) -> OptimizationConfig:
    return OptimizationConfig(**overrides)


class TestBasicTrends:
    def test_no_runs_yields_empty_progress(self):
        progress = compute_progress([], _optimization())
        assert progress.metrics == []
        assert progress.completed_run_count == 0

    def test_ignores_non_completed_runs(self):
        runs = [
            FakeRun(1, "failed", {"metrics": {"accuracy": 0.5}}),
            FakeRun(2, "cancelled", {"metrics": {"accuracy": 0.6}}),
            FakeRun(3, "running_training", None),
        ]
        progress = compute_progress(runs, _optimization())
        assert progress.completed_run_count == 0
        assert progress.metrics == []

    def test_single_completed_run_has_no_previous_value(self):
        runs = [FakeRun(1, "completed", {"metrics": {"accuracy": 0.9}})]
        progress = compute_progress(runs, _optimization())
        assert progress.completed_run_count == 1
        [trend] = progress.metrics
        assert trend.metric == "accuracy"
        assert trend.latest_value == 0.9
        assert trend.previous_value is None
        assert trend.delta_from_previous is None

    def test_tracks_delta_between_consecutive_runs(self):
        runs = [
            FakeRun(1, "completed", {"metrics": {"accuracy": 0.80}}),
            FakeRun(2, "completed", {"metrics": {"accuracy": 0.91}}),
        ]
        progress = compute_progress(runs, _optimization())
        [trend] = progress.metrics
        assert trend.latest_value == 0.91
        assert trend.previous_value == 0.80
        assert trend.delta_from_previous == pytest.approx(0.11)

    def test_runs_are_ordered_by_run_number_not_list_order(self):
        runs = [
            FakeRun(2, "completed", {"metrics": {"accuracy": 0.91}}),
            FakeRun(1, "completed", {"metrics": {"accuracy": 0.80}}),
        ]
        progress = compute_progress(runs, _optimization())
        [trend] = progress.metrics
        assert [p.run_number for p in trend.points] == [1, 2]

    def test_tracks_multiple_metrics_independently(self):
        runs = [
            FakeRun(1, "completed", {"metrics": {"accuracy": 0.8, "f1": 0.75}}),
            FakeRun(2, "completed", {"metrics": {"accuracy": 0.85, "f1": 0.70}}),
        ]
        progress = compute_progress(runs, _optimization())
        by_name = {t.metric: t for t in progress.metrics}
        assert by_name["accuracy"].delta_from_previous == pytest.approx(0.05)
        assert by_name["f1"].delta_from_previous == pytest.approx(-0.05)

    def test_a_metric_missing_from_a_later_run_keeps_its_own_history(self):
        runs = [
            FakeRun(1, "completed", {"metrics": {"accuracy": 0.8, "precision": 0.7}}),
            FakeRun(2, "completed", {"metrics": {"accuracy": 0.85}}),  # precision not reported this time
        ]
        progress = compute_progress(runs, _optimization())
        by_name = {t.metric: t for t in progress.metrics}
        assert [p.run_number for p in by_name["precision"].points] == [1]
        assert [p.run_number for p in by_name["accuracy"].points] == [1, 2]

    def test_non_numeric_metric_values_are_skipped_not_crashed_on(self):
        runs = [FakeRun(1, "completed", {"metrics": {"accuracy": 0.9, "notes": "looks good"}})]
        progress = compute_progress(runs, _optimization())
        assert [t.metric for t in progress.metrics] == ["accuracy"]

    def test_booleans_are_not_treated_as_numeric_metrics(self):
        # bool is a subclass of int in Python -- isinstance(True, int) is
        # True -- so this needs an explicit guard, not an accident of duck typing.
        runs = [FakeRun(1, "completed", {"metrics": {"accuracy": 0.9, "passed": True}})]
        progress = compute_progress(runs, _optimization())
        assert [t.metric for t in progress.metrics] == ["accuracy"]

    def test_a_run_with_empty_metrics_contributes_no_data_points(self):
        runs = [
            FakeRun(1, "completed", {"metrics": {}, "warnings": ["no metrics source configured"]}),
            FakeRun(2, "completed", {"metrics": {"accuracy": 0.9}}),
        ]
        progress = compute_progress(runs, _optimization())
        assert progress.completed_run_count == 2
        [trend] = progress.metrics
        assert trend.points == [MetricTrendPoint(2, 0.9)]


class TestTargetTracking:
    def test_no_target_configured_leaves_target_fields_none(self):
        runs = [FakeRun(1, "completed", {"metrics": {"f1": 0.9}})]
        progress = compute_progress(runs, _optimization())
        assert progress.target_metric is None
        assert progress.distance_to_target is None
        assert progress.moved_closer_to_target is None

    def test_target_metric_never_reported_leaves_distance_none(self):
        runs = [FakeRun(1, "completed", {"metrics": {"accuracy": 0.9}})]
        progress = compute_progress(runs, _optimization(target_metric="f1", target_value=0.95))
        assert progress.latest_target_value is None
        assert progress.distance_to_target is None

    def test_distance_to_target_is_absolute_and_direction_agnostic(self):
        # f1 above the target should report the same *distance* as the
        # same-magnitude undershoot would -- this module never assumes
        # higher-is-better.
        over = compute_progress(
            [FakeRun(1, "completed", {"metrics": {"f1": 0.97}})],
            _optimization(target_metric="f1", target_value=0.95),
        )
        under = compute_progress(
            [FakeRun(1, "completed", {"metrics": {"f1": 0.93}})],
            _optimization(target_metric="f1", target_value=0.95),
        )
        assert over.distance_to_target == pytest.approx(0.02)
        assert under.distance_to_target == pytest.approx(0.02)

    def test_moved_closer_to_target_true_when_distance_shrinks(self):
        runs = [
            FakeRun(1, "completed", {"metrics": {"f1": 0.80}}),
            FakeRun(2, "completed", {"metrics": {"f1": 0.90}}),
        ]
        progress = compute_progress(runs, _optimization(target_metric="f1", target_value=0.95))
        assert progress.moved_closer_to_target is True

    def test_moved_closer_to_target_false_when_distance_grows(self):
        runs = [
            FakeRun(1, "completed", {"metrics": {"f1": 0.90}}),
            FakeRun(2, "completed", {"metrics": {"f1": 0.80}}),
        ]
        progress = compute_progress(runs, _optimization(target_metric="f1", target_value=0.95))
        assert progress.moved_closer_to_target is False

    def test_moved_closer_to_target_none_with_only_one_data_point(self):
        runs = [FakeRun(1, "completed", {"metrics": {"f1": 0.90}})]
        progress = compute_progress(runs, _optimization(target_metric="f1", target_value=0.95))
        assert progress.moved_closer_to_target is None

    def test_overshooting_past_the_target_can_still_move_further_away(self):
        # Went from just below target to well above it -- distance grew,
        # even though the raw value increased. Only the distance framing
        # (not "value went up") should drive this.
        runs = [
            FakeRun(1, "completed", {"metrics": {"f1": 0.94}}),
            FakeRun(2, "completed", {"metrics": {"f1": 1.00}}),
        ]
        progress = compute_progress(runs, _optimization(target_metric="f1", target_value=0.95))
        assert progress.moved_closer_to_target is False
