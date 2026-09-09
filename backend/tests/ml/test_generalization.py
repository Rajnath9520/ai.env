import pytest

from app.ml.environments.config import GridWorldConfig
from app.ml.evaluation.generalization import (
    EnvironmentSplitGenerator,
    GeneralizationEvaluator,
)
from app.ml.training.trainer import AgentTrainer, EvaluationResult

BASE_CONFIG = GridWorldConfig(
    grid_size=8, obstacle_density=0.2, dead_end_probability=0.2, path_complexity=0.2, max_steps=40, seed=1
)


class FakeTrainer:
    """Duck-typed AgentTrainer stand-in: returns deterministic, config-seed
    -dependent results so evaluate_split's aggregation math is directly
    checkable without running real PPO."""

    def __init__(self):
        self.model = "trained"
        self.evaluate_calls = []

    def evaluate(self, config, episodes, deterministic=True, environment_id=None):
        self.evaluate_calls.append((config.seed, episodes, environment_id))
        # Deterministic per-seed values so aggregation can be hand-checked.
        success = config.seed % 2 == 0
        return EvaluationResult(
            environment_id=environment_id,
            episodes=episodes,
            success_rate=1.0 if success else 0.0,
            average_reward=1.0 if success else -1.0,
            average_steps=10.0,
            collision_rate=0.1,
            episode_rewards=[1.0 if success else -1.0] * episodes,
            episode_successes=[success] * episodes,
        )


class TestEnvironmentSplitGenerator:
    def test_seed_ranges_are_disjoint_by_construction(self):
        from app.ml.evaluation.generalization import SEED_RANGES

        ranges = list(SEED_RANGES.values())
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                lo_a, hi_a = ranges[i]
                lo_b, hi_b = ranges[j]
                assert hi_a <= lo_b or hi_b <= lo_a

    def test_in_distribution_configs_use_correct_seed_range(self):
        from app.ml.evaluation.generalization import SEED_RANGES

        generator = EnvironmentSplitGenerator(seed=1)
        for split in ("train", "validation", "test"):
            lo, hi = SEED_RANGES[split]
            configs = generator.in_distribution_configs(BASE_CONFIG, split, n=20)
            assert all(lo <= c.seed < hi for c in configs)
            # Same parameter distribution as the base config.
            assert all(c.obstacle_density == BASE_CONFIG.obstacle_density for c in configs)
            assert all(c.grid_size == BASE_CONFIG.grid_size for c in configs)

    def test_in_distribution_rejects_ood_split_name(self):
        generator = EnvironmentSplitGenerator(seed=1)
        with pytest.raises(ValueError):
            generator.in_distribution_configs(BASE_CONFIG, "ood", n=1)

    def test_ood_configs_differ_structurally_from_base(self):
        generator = EnvironmentSplitGenerator(seed=1)
        configs = generator.ood_configs(BASE_CONFIG, n=10)
        assert all(c.grid_size != BASE_CONFIG.grid_size for c in configs)
        assert all(c.obstacle_density != BASE_CONFIG.obstacle_density for c in configs)

    def test_build_all_splits_returns_all_four_splits_with_no_seed_overlap(self):
        generator = EnvironmentSplitGenerator(seed=1)
        splits = generator.build_all_splits(BASE_CONFIG, n_per_split=15)
        assert set(splits.keys()) == {"train", "validation", "test", "ood"}
        EnvironmentSplitGenerator.assert_no_seed_overlap(splits)  # should not raise

    def test_assert_no_seed_overlap_detects_a_real_overlap(self):
        train_configs = [GridWorldConfig(seed=999)]
        test_configs = [GridWorldConfig(seed=999)]  # deliberately overlapping
        splits = {"train": train_configs, "test": test_configs}
        with pytest.raises(AssertionError):
            EnvironmentSplitGenerator.assert_no_seed_overlap(splits)


class TestGeneralizationEvaluator:
    def test_evaluate_split_aggregates_across_configs(self):
        trainer = FakeTrainer()
        evaluator = GeneralizationEvaluator()
        configs = [
            GridWorldConfig(grid_size=6, seed=100_000),  # even -> success
            GridWorldConfig(grid_size=6, seed=100_001),  # odd -> failure
        ]
        result = evaluator.evaluate_split(trainer, configs, "validation", episodes_per_config=4)
        assert result.n_environments == 2
        assert result.episodes == 8
        assert result.success_rate == pytest.approx(0.5)
        assert result.split == "validation"

    def test_evaluate_split_empty_configs_returns_zeroed_result(self):
        trainer = FakeTrainer()
        evaluator = GeneralizationEvaluator()
        result = evaluator.evaluate_split(trainer, [], "test")
        assert result.n_environments == 0
        assert result.episodes == 0
        assert result.success_rate == 0.0

    def test_evaluate_all_splits_excludes_train(self):
        trainer = FakeTrainer()
        evaluator = GeneralizationEvaluator()
        generator = EnvironmentSplitGenerator(seed=1)
        splits = generator.build_all_splits(BASE_CONFIG, n_per_split=3)
        results = evaluator.evaluate_all_splits(trainer, splits, episodes_per_config=2)
        assert set(results.keys()) == {"validation", "test", "ood"}

    def test_evaluate_never_calls_trainer_train(self):
        """Structural guarantee: GeneralizationEvaluator has no reference
        to a train() call anywhere in its evaluate methods."""
        trainer = FakeTrainer()
        assert not hasattr(trainer, "train")  # FakeTrainer intentionally has no train()
        evaluator = GeneralizationEvaluator()
        generator = EnvironmentSplitGenerator(seed=1)
        splits = generator.build_all_splits(BASE_CONFIG, n_per_split=2)
        evaluator.evaluate_all_splits(trainer, splits, episodes_per_config=1)  # should not raise


class TestRealIntegration:
    def test_real_trainer_end_to_end(self):
        tiny_base = GridWorldConfig(grid_size=5, obstacle_density=0.0, max_steps=12, seed=1)
        trainer = AgentTrainer(seed=1)
        trainer.train(tiny_base, total_timesteps=256, environment_id="warmup")

        generator = EnvironmentSplitGenerator(seed=2)
        splits = generator.build_all_splits(tiny_base, n_per_split=2)
        EnvironmentSplitGenerator.assert_no_seed_overlap(splits)

        evaluator = GeneralizationEvaluator()
        results = evaluator.evaluate_all_splits(trainer, splits, episodes_per_config=2)
        for split_name, result in results.items():
            assert result.split == split_name
            assert 0.0 <= result.success_rate <= 1.0
