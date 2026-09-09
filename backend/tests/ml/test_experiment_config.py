import tempfile
from pathlib import Path

import pytest

from app.ml.experiment.config import ExperimentConfig


class TestExperimentConfigValidation:
    def test_defaults_are_valid(self):
        ExperimentConfig()  # should not raise

    def test_rejects_unknown_curriculum_type(self):
        with pytest.raises(ValueError):
            ExperimentConfig(curriculum_type="not_a_real_strategy")

    def test_rejects_non_ppo_algorithm(self):
        with pytest.raises(ValueError):
            ExperimentConfig(algorithm="DQN")

    def test_rejects_zero_curriculum_steps(self):
        with pytest.raises(ValueError):
            ExperimentConfig(n_curriculum_steps=0)

    def test_agent_mode_defaults_to_environment_only(self):
        # Phase 28 (Section 6): PPO/training must never be the default.
        config = ExperimentConfig()
        assert config.agent_mode == "environment_only"

    def test_rejects_unknown_agent_mode(self):
        with pytest.raises(ValueError):
            ExperimentConfig(agent_mode="secretly_train_ppo")

    def test_rejects_unknown_generation_strategy(self):
        with pytest.raises(ValueError):
            ExperimentConfig(generation_strategy="not_a_real_strategy")

    def test_algorithm_is_optional_and_does_not_gate_execution(self):
        # Section 6: `algorithm` must be optional/secondary; agent_mode is
        # what actually decides execution.
        config = ExperimentConfig(algorithm=None, agent_mode="environment_only")
        assert config.algorithm is None
        assert config.agent_mode == "environment_only"


class TestFlatRoundTrip:
    def test_to_dict_from_dict_round_trip(self):
        config = ExperimentConfig(name="my_exp", seed=7, curriculum_type="random", grid_size=12)
        restored = ExperimentConfig.from_dict(config.to_dict())
        assert restored == config

    def test_save_load_round_trip(self):
        config = ExperimentConfig(name="my_exp", seed=7, curriculum_type="fixed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.yaml"
            config.save(path)
            restored = ExperimentConfig.load(path)
        assert restored == config


class TestSpecExampleYamlShape:
    def test_loads_section_22_example_shape(self):
        yaml_text = """
algorithm: PPO

curriculum:
    type: failure_driven

environment:
    type: gridworld
    size: 10

training:
    episodes: 10000

evaluation:
    frequency: 500

seed: 42
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.yaml"
            path.write_text(yaml_text)
            config = ExperimentConfig.load(path)

        assert config.algorithm == "PPO"
        assert config.curriculum_type == "failure_driven"
        assert config.grid_size == 10
        assert config.evaluation_frequency == 500
        assert config.seed == 42
        assert config.timesteps_per_step == 10_000 // config.n_curriculum_steps

    def test_nested_shape_missing_fields_uses_defaults(self):
        yaml_text = "curriculum:\n    type: random\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.yaml"
            path.write_text(yaml_text)
            config = ExperimentConfig.load(path)
        assert config.curriculum_type == "random"
        assert config.grid_size == ExperimentConfig().grid_size
