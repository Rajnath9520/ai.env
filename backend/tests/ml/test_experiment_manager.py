import pytest

from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.manager import ExperimentManager

TINY_KWARGS = dict(
    agent_mode="demo_ppo",
    grid_size=5,
    obstacle_density=0.1,
    dead_end_probability=0.1,
    path_complexity=0.1,
    max_steps_per_episode=12,
    n_curriculum_steps=4,
    timesteps_per_step=256,
    eval_episodes=3,
    run_generalization=True,
    generalization_envs_per_split=2,
    generalization_episodes_per_config=2,
)


class TestExperimentManagerAllCurriculumTypes:
    @pytest.mark.parametrize("curriculum_type", ["random", "fixed", "generic_adaptive", "failure_driven"])
    def test_runs_end_to_end_for_every_curriculum_type(self, curriculum_type):
        config = ExperimentConfig(
            name=f"test_{curriculum_type}",
            curriculum_type=curriculum_type,
            seed=1,
            n_candidates=2,
            trajectories_per_analysis=2,
            **TINY_KWARGS,
        )
        result = ExperimentManager(config).run()

        assert len(result.curriculum_history) == config.n_curriculum_steps
        assert result.generalization_results is not None
        assert set(result.generalization_results.keys()) == {"validation", "test", "ood"}

        if curriculum_type == "failure_driven":
            assert result.weakness_history is not None
            assert len(result.weakness_history) == config.n_curriculum_steps
            assert result.learning_progress_history is not None
        else:
            assert result.weakness_history is None
            assert result.learning_progress_history is None

    def test_result_to_dict_is_json_serializable(self):
        import json

        config = ExperimentConfig(name="test_json", curriculum_type="random", seed=1, **TINY_KWARGS)
        result = ExperimentManager(config).run()
        json.dumps(result.to_dict())  # should not raise


class TestReproducibility:
    def test_random_curriculum_is_fully_reproducible(self):
        config = ExperimentConfig(
            name="repro_random", curriculum_type="random", seed=3,
            **{**TINY_KWARGS, "run_generalization": False},
        )
        result_a = ExperimentManager(config).run()
        result_b = ExperimentManager(config).run()

        history_a = [s.to_dict() for s in result_a.curriculum_history]
        history_b = [s.to_dict() for s in result_b.curriculum_history]
        assert history_a == history_b

    def test_failure_driven_curriculum_is_fully_reproducible(self):
        config = ExperimentConfig(
            name="repro_failure_driven",
            curriculum_type="failure_driven",
            seed=3,
            n_candidates=2,
            trajectories_per_analysis=2,
            **{**TINY_KWARGS, "run_generalization": False},
        )
        result_a = ExperimentManager(config).run()
        result_b = ExperimentManager(config).run()

        history_a = [s.to_dict() for s in result_a.curriculum_history]
        history_b = [s.to_dict() for s in result_b.curriculum_history]
        assert history_a == history_b
        assert result_a.weakness_history == result_b.weakness_history
        assert result_a.learning_progress_history == result_b.learning_progress_history
