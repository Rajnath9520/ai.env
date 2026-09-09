import pytest

from app.ml.analysis.weakness_vector import WEAKNESS_DIMENSIONS
from app.ml.curriculum.failure_driven import AdaptiveCurriculumRunner
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator
from app.ml.training.trainer import AgentTrainer

TINY_BASE_CONFIG = GridWorldConfig(
    grid_size=5, obstacle_density=0.1, dead_end_probability=0.1, path_complexity=0.1, max_steps=12, seed=1
)


def _make_runner(**overrides) -> AdaptiveCurriculumRunner:
    trainer = AgentTrainer(seed=1)
    defaults = dict(
        trainer=trainer,
        base_config=TINY_BASE_CONFIG,
        timesteps_per_step=256,
        eval_episodes=3,
        n_candidates=3,
        trajectories_per_analysis=2,
    )
    defaults.update(overrides)
    return AdaptiveCurriculumRunner(**defaults)


class TestAdaptiveLoopStructure:
    def test_run_produces_expected_lengths(self):
        runner = _make_runner()
        result = runner.run(n_steps=3)
        assert len(result.history) == 3
        assert len(result.weakness_history) == 3
        assert len(result.learning_progress_history) == 3

    def test_first_step_has_no_target_weakness_and_uses_base_config(self):
        runner = _make_runner()
        result = runner.run(n_steps=2)
        first = result.history[0]
        assert first.target_weakness is None
        assert first.config.to_dict()["grid_size"] == TINY_BASE_CONFIG.grid_size
        assert first.config.to_dict()["obstacle_density"] == pytest.approx(
            TINY_BASE_CONFIG.obstacle_density
        )

    def test_subsequent_steps_have_a_valid_target_weakness(self):
        runner = _make_runner()
        result = runner.run(n_steps=3)
        for step in result.history[1:]:
            assert step.target_weakness in WEAKNESS_DIMENSIONS

    def test_weakness_history_entries_are_well_formed(self):
        runner = _make_runner()
        result = runner.run(n_steps=2)
        for weakness in result.weakness_history:
            assert weakness.primary_weakness in WEAKNESS_DIMENSIONS
            for dim in WEAKNESS_DIMENSIONS:
                assert 0.0 <= getattr(weakness, dim) <= 1.0

    def test_learning_progress_entries_reference_correct_environments(self):
        runner = _make_runner()
        result = runner.run(n_steps=3)
        env_ids_from_history = [s.environment_id for s in result.history]
        env_ids_from_lp = [lp.environment_id for lp in result.learning_progress_history]
        assert env_ids_from_history == env_ids_from_lp

    def test_step_learning_progress_matches_lp_record(self):
        runner = _make_runner()
        result = runner.run(n_steps=2)
        for step, lp in zip(result.history, result.learning_progress_history):
            assert step.learning_progress == pytest.approx(lp.learning_progress)


class TestAdaptiveLoopIntegrity:
    def test_policy_persists_across_the_full_run(self):
        runner = _make_runner()
        result_before_model = runner.trainer.model
        runner.run(n_steps=1)
        model_after_first_run = runner.trainer.model
        assert result_before_model is None
        assert model_after_first_run is not None
        runner.run(n_steps=1)
        assert runner.trainer.model is model_after_first_run

    def test_all_selected_environments_are_valid(self):
        runner = _make_runner()
        result = runner.run(n_steps=3)
        generator = GridWorldGenerator()
        for step in result.history:
            report = generator.generate(step.config)
            assert report.validation.is_valid

    def test_trajectories_are_stored(self):
        runner = _make_runner()
        runner.run(n_steps=2)
        # 2 steps * 2 trajectories_per_analysis each = 4 stored trajectories
        assert runner.trajectory_store.summary_count() == 4

    def test_result_to_dict_is_json_serializable(self):
        import json

        runner = _make_runner()
        result = runner.run(n_steps=2)
        json.dumps(result.to_dict())  # should not raise
