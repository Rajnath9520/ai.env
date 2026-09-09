import pytest

from app.ml.curriculum.base import CurriculumStep
from app.ml.curriculum.random_curriculum import RandomCurriculum
from app.ml.curriculum.runner import CurriculumRunner
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator
from app.ml.training.trainer import AgentTrainer


class TestRandomCurriculum:
    def test_produces_valid_configs(self):
        strategy = RandomCurriculum(grid_size=6, seed=1)
        generator = GridWorldGenerator()
        for _ in range(10):
            config = strategy.next_environment(history=[])
            # GridWorldConfig.__post_init__ already validates ranges; this
            # additionally confirms the generator can build a real layout.
            report = generator.generate(config)
            assert report.validation.is_valid

    def test_params_vary_across_calls(self):
        strategy = RandomCurriculum(grid_size=6, seed=1)
        configs = [strategy.next_environment(history=[]) for _ in range(10)]
        densities = {c.obstacle_density for c in configs}
        assert len(densities) > 1

    def test_deterministic_with_seed(self):
        strategy_a = RandomCurriculum(grid_size=6, seed=7)
        strategy_b = RandomCurriculum(grid_size=6, seed=7)
        configs_a = [strategy_a.next_environment([]) for _ in range(5)]
        configs_b = [strategy_b.next_environment([]) for _ in range(5)]
        assert [c.to_dict() for c in configs_a] == [c.to_dict() for c in configs_b]

    def test_ignores_history_by_design(self):
        # Baseline A must not adapt to performance — passing a non-empty
        # history should not change the distribution of sampled configs.
        strategy = RandomCurriculum(grid_size=6, seed=3)
        fake_history = [
            CurriculumStep(
                step_index=0,
                environment_id="e",
                config=GridWorldConfig(grid_size=6, seed=1),
                difficulty=0.9,
                success_rate=0.0,
                average_reward=-1.0,
            )
        ]
        strategy_ref = RandomCurriculum(grid_size=6, seed=3)
        config_with_history = strategy.next_environment(fake_history)
        config_without_history = strategy_ref.next_environment([])
        assert config_with_history.to_dict() == config_without_history.to_dict()


class TestCurriculumRunner:
    def test_run_produces_expected_number_of_steps(self):
        strategy = RandomCurriculum(grid_size=5, max_steps=15, seed=1)
        trainer = AgentTrainer(seed=1)
        runner = CurriculumRunner(trainer, strategy, timesteps_per_step=256, eval_episodes=3)
        history = runner.run(n_steps=3)

        assert len(history) == 3
        assert [s.step_index for s in history] == [0, 1, 2]
        for step in history:
            assert 0.0 <= step.success_rate <= 1.0
            assert step.environment_id.startswith("random_env_")
            assert 0.0 <= step.difficulty <= 1.0

    def test_policy_persists_across_curriculum_steps(self):
        strategy = RandomCurriculum(grid_size=5, max_steps=15, seed=1)
        trainer = AgentTrainer(seed=1)
        runner = CurriculumRunner(trainer, strategy, timesteps_per_step=256, eval_episodes=3)
        runner.run(n_steps=2)
        model_after_run = trainer.model
        runner.run(n_steps=1)
        assert trainer.model is model_after_run

    def test_step_dict_is_json_serializable(self):
        strategy = RandomCurriculum(grid_size=5, max_steps=15, seed=1)
        trainer = AgentTrainer(seed=1)
        runner = CurriculumRunner(trainer, strategy, timesteps_per_step=256, eval_episodes=3)
        history = runner.run(n_steps=1)
        import json

        json.dumps(history[0].to_dict())  # should not raise
