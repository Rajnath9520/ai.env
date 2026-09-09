import pytest

from app.ml.curriculum.base import CurriculumStep
from app.ml.environments.config import GridWorldConfig
from app.ml.evaluation.learning_progress import LearningProgressRecord
from app.ml.evaluation.metrics import MetricsCalculator


def _step(i, success_rate, average_reward=0.0, average_steps=10.0, collision_rate=0.0) -> CurriculumStep:
    return CurriculumStep(
        step_index=i,
        environment_id=f"env_{i}",
        config=GridWorldConfig(seed=1),
        difficulty=0.3,
        success_rate=success_rate,
        average_reward=average_reward,
        average_steps=average_steps,
        collision_rate=collision_rate,
    )


class TestBasicMetrics:
    def test_pulls_final_step_values_directly(self):
        history = [
            _step(0, 0.1, average_reward=-0.5, average_steps=20.0, collision_rate=0.3),
            _step(1, 0.6, average_reward=0.4, average_steps=12.0, collision_rate=0.05),
        ]
        report = MetricsCalculator().compute(history, timesteps_per_step=1000)
        assert report.success_rate == pytest.approx(0.6)
        assert report.average_reward == pytest.approx(0.4)
        assert report.average_steps == pytest.approx(12.0)
        assert report.collision_rate == pytest.approx(0.05)
        assert report.failure_rate == pytest.approx(0.4)

    def test_empty_history_yields_none_for_direct_metrics(self):
        report = MetricsCalculator().compute([], timesteps_per_step=1000)
        assert report.success_rate is None
        assert report.failure_rate is None
        assert report.average_reward is None


class TestSampleEfficiency:
    def test_finds_first_step_crossing_threshold(self):
        history = [_step(0, 0.1), _step(1, 0.3), _step(2, 0.6), _step(3, 0.9)]
        report = MetricsCalculator(sample_efficiency_threshold=0.5).compute(history, timesteps_per_step=1000)
        assert report.sample_efficiency_timesteps == 3000

    def test_never_crossing_threshold_returns_none(self):
        history = [_step(0, 0.1), _step(1, 0.2), _step(2, 0.3)]
        report = MetricsCalculator(sample_efficiency_threshold=0.9).compute(history, timesteps_per_step=1000)
        assert report.sample_efficiency_timesteps is None

    def test_threshold_reported_for_transparency(self):
        report = MetricsCalculator(sample_efficiency_threshold=0.75).compute([], timesteps_per_step=1000)
        assert report.sample_efficiency_threshold == 0.75


class TestConvergence:
    def test_detects_stabilization_point(self):
        history = [_step(0, 0.1), _step(1, 0.3), _step(2, 0.79), _step(3, 0.80), _step(4, 0.81)]
        calc = MetricsCalculator(convergence_tolerance=0.05, convergence_window=3)
        report = calc.compute(history, timesteps_per_step=1000)
        assert report.convergence_step == 2
        assert report.convergence_timesteps == 3000

    def test_never_stabilizing_returns_none(self):
        history = [_step(0, 0.1), _step(1, 0.9), _step(2, 0.1), _step(3, 0.9)]
        calc = MetricsCalculator(convergence_tolerance=0.02, convergence_window=3)
        report = calc.compute(history, timesteps_per_step=1000)
        assert report.convergence_step is None

    def test_history_shorter_than_window_returns_none(self):
        history = [_step(0, 0.5), _step(1, 0.5)]
        calc = MetricsCalculator(convergence_window=3)
        report = calc.compute(history, timesteps_per_step=1000)
        assert report.convergence_step is None

    def test_empty_history_returns_none_not_error(self):
        report = MetricsCalculator().compute([], timesteps_per_step=1000)
        assert report.convergence_step is None
        assert report.convergence_timesteps is None


class TestLearningProgress:
    def test_mean_of_recorded_values(self):
        history = [_step(0, 0.3)]
        lp_history = [
            LearningProgressRecord("e0", None, "success_rate", 0.1, 0.3, 0.2),
            LearningProgressRecord("e1", None, "success_rate", 0.3, 0.5, 0.2),
            LearningProgressRecord("e2", None, "success_rate", 0.5, 0.3, -0.2),
        ]
        report = MetricsCalculator().compute(history, timesteps_per_step=1000, learning_progress_history=lp_history)
        assert report.learning_progress_mean == pytest.approx((0.2 + 0.2 - 0.2) / 3)

    def test_no_history_returns_none(self):
        report = MetricsCalculator().compute([_step(0, 0.3)], timesteps_per_step=1000)
        assert report.learning_progress_mean is None


class TestGeneralizationAndOOD:
    def test_generalization_score_averages_present_splits(self):
        gen = {
            "validation": {"success_rate": 0.4},
            "test": {"success_rate": 0.3},
            "ood": {"success_rate": 0.2, "average_reward": -0.1},
        }
        report = MetricsCalculator().compute([_step(0, 0.5)], timesteps_per_step=1000, generalization_results=gen)
        assert report.generalization_score == pytest.approx((0.4 + 0.3 + 0.2) / 3)
        assert report.ood_success_rate == pytest.approx(0.2)
        assert report.ood_average_reward == pytest.approx(-0.1)

    def test_missing_generalization_returns_none(self):
        report = MetricsCalculator().compute([_step(0, 0.5)], timesteps_per_step=1000)
        assert report.generalization_score is None
        assert report.ood_success_rate is None

    def test_partial_splits_only_averages_present_ones(self):
        gen = {"validation": {"success_rate": 0.6}}
        report = MetricsCalculator().compute([_step(0, 0.5)], timesteps_per_step=1000, generalization_results=gen)
        assert report.generalization_score == pytest.approx(0.6)
        assert report.ood_success_rate is None


class TestFailureRecoveryRate:
    def test_aggregates_across_steps_and_reports(self):
        failure_reports_history = [
            [
                {"failure_type": "dead_end_recovery", "features": {"dead_end_entries": 4, "successful_recoveries": 3}},
                {"failure_type": "obstacle_avoidance", "features": {}},
            ],
            [
                {"failure_type": "dead_end_recovery", "features": {"dead_end_entries": 2, "successful_recoveries": 2}},
            ],
        ]
        report = MetricsCalculator().compute(
            [_step(0, 0.5)], timesteps_per_step=1000, failure_reports_history=failure_reports_history
        )
        assert report.failure_recovery_rate == pytest.approx(5 / 6)

    def test_no_dead_end_data_returns_none(self):
        failure_reports_history = [[{"failure_type": "obstacle_avoidance", "features": {}}]]
        report = MetricsCalculator().compute(
            [_step(0, 0.5)], timesteps_per_step=1000, failure_reports_history=failure_reports_history
        )
        assert report.failure_recovery_rate is None

    def test_no_failure_data_at_all_returns_none(self):
        report = MetricsCalculator().compute([_step(0, 0.5)], timesteps_per_step=1000)
        assert report.failure_recovery_rate is None


class TestToDict:
    def test_to_dict_is_json_serializable(self):
        import json

        history = [_step(0, 0.5)]
        report = MetricsCalculator().compute(history, timesteps_per_step=1000)
        json.dumps(report.to_dict())  # should not raise
