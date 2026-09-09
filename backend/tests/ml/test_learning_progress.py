import pytest

from app.ml.environments.config import GridWorldConfig
from app.ml.evaluation.learning_progress import LearningProgressEvaluator
from app.ml.training.trainer import AgentTrainer, EvaluationResult

TINY_CONFIG = GridWorldConfig(grid_size=5, obstacle_density=0.0, max_steps=15, seed=1)


class FakeTrainer:
    """Duck-typed stand-in for AgentTrainer so learning-progress arithmetic
    and control flow can be tested without running real PPO training —
    keeps these tests fast and lets us control exactly what "before" and
    "after" performance values are."""

    def __init__(self, success_rate_sequence: list[float]):
        self.model = None
        self._sequence = list(success_rate_sequence)
        self._calls = 0
        self.train_call_count = 0
        self.evaluate_call_count = 0

    def evaluate(self, config, episodes, deterministic=True, environment_id=None):
        self.evaluate_call_count += 1
        success_rate = self._sequence[min(self._calls, len(self._sequence) - 1)]
        self._calls += 1
        return EvaluationResult(
            environment_id=environment_id,
            episodes=episodes,
            success_rate=success_rate,
            average_reward=success_rate * 2 - 1,
            average_steps=10.0,
            collision_rate=0.0,
        )

    def train(self, config, total_timesteps, environment_id=None):
        self.train_call_count += 1
        self.model = "trained"  # mark as no-longer-untrained


class TestLearningProgressEvaluatorConstruction:
    def test_rejects_unknown_metric(self):
        with pytest.raises(ValueError):
            LearningProgressEvaluator(metric="not_a_real_metric")


class TestCompute:
    def test_matches_spec_example_exactly(self):
        evaluator = LearningProgressEvaluator()
        record = evaluator.compute(
            environment_id="env_042",
            weakness_target="long_horizon_planning",
            performance_before=0.48,
            performance_after=0.67,
        )
        assert record.learning_progress == pytest.approx(0.19, abs=1e-9)
        assert record.environment_id == "env_042"
        assert record.weakness_target == "long_horizon_planning"

    def test_negative_learning_progress_when_performance_drops(self):
        evaluator = LearningProgressEvaluator()
        record = evaluator.compute("env_1", None, performance_before=0.7, performance_after=0.5)
        assert record.learning_progress == pytest.approx(-0.2)

    def test_to_dict_rounds_and_is_json_serializable(self):
        import json

        evaluator = LearningProgressEvaluator()
        record = evaluator.compute("env_1", "exploration", 0.111111, 0.222222)
        json.dumps(record.to_dict())  # should not raise
        assert record.to_dict()["learning_progress"] == pytest.approx(0.1111, abs=1e-4)


class TestEvaluatePerformance:
    def test_untrained_model_returns_zero_without_calling_evaluate(self):
        trainer = FakeTrainer(success_rate_sequence=[0.9])
        evaluator = LearningProgressEvaluator()
        performance = evaluator.evaluate_performance(trainer, TINY_CONFIG)
        assert performance == 0.0
        assert trainer.evaluate_call_count == 0

    def test_trained_model_delegates_to_trainer_evaluate(self):
        trainer = FakeTrainer(success_rate_sequence=[0.6])
        trainer.model = "already_trained"
        evaluator = LearningProgressEvaluator()
        performance = evaluator.evaluate_performance(trainer, TINY_CONFIG)
        assert performance == pytest.approx(0.6)
        assert trainer.evaluate_call_count == 1

    def test_average_reward_metric_extracts_correct_field(self):
        trainer = FakeTrainer(success_rate_sequence=[0.5])
        trainer.model = "already_trained"
        evaluator = LearningProgressEvaluator(metric="average_reward")
        performance = evaluator.evaluate_performance(trainer, TINY_CONFIG)
        assert performance == pytest.approx(0.5 * 2 - 1)


class TestMeasureAroundTraining:
    def test_full_flow_trains_between_before_and_after(self):
        trainer = FakeTrainer(success_rate_sequence=[0.3, 0.3, 0.6])
        evaluator = LearningProgressEvaluator()
        record = evaluator.measure_around_training(
            trainer,
            TINY_CONFIG,
            total_timesteps=1000,
            environment_id="env_007",
            weakness_target="exploration",
        )
        assert trainer.train_call_count == 1
        # First eval call is "before" (untrained -> 0.0 without calling evaluate),
        # train() is called, then "after" calls evaluate for the first real time.
        assert record.performance_before == 0.0
        assert record.performance_after == pytest.approx(0.3)
        assert record.learning_progress == pytest.approx(0.3)
        assert record.environment_id == "env_007"
        assert record.weakness_target == "exploration"

    def test_full_flow_with_pretrained_model_measures_real_before(self):
        trainer = FakeTrainer(success_rate_sequence=[0.4, 0.7])
        trainer.model = "already_trained"
        evaluator = LearningProgressEvaluator()
        record = evaluator.measure_around_training(
            trainer, TINY_CONFIG, total_timesteps=1000, environment_id="env_008"
        )
        assert record.performance_before == pytest.approx(0.4)
        assert record.performance_after == pytest.approx(0.7)
        assert record.learning_progress == pytest.approx(0.3)


class TestRealAgentTrainerIntegration:
    def test_runs_end_to_end_with_real_ppo_trainer(self):
        trainer = AgentTrainer(seed=1)
        evaluator = LearningProgressEvaluator()
        record = evaluator.measure_around_training(
            trainer,
            TINY_CONFIG,
            total_timesteps=256,
            environment_id="env_real",
            weakness_target="obstacle_avoidance",
            eval_episodes=3,
        )
        assert record.environment_id == "env_real"
        assert isinstance(record.learning_progress, float)
        assert -1.0 <= record.learning_progress <= 1.0
