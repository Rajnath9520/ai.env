import tempfile
from pathlib import Path

import pytest

from app.ml.environments.config import GridWorldConfig
from app.ml.training.trainer import AgentTrainer

# Small, fast-to-train configuration used across all trainer tests — the
# point of these tests is pipeline correctness, not learning quality.
TINY_CONFIG = GridWorldConfig(
    grid_size=5, obstacle_density=0.0, max_steps=15, seed=1
)


class TestAgentTrainer:
    def test_train_returns_episode_metrics(self):
        trainer = AgentTrainer(seed=1)
        result = trainer.train(TINY_CONFIG, total_timesteps=256, environment_id="env_test_1")
        assert result.total_timesteps == 256
        assert len(result.episodes) > 0
        first = result.episodes[0]
        assert first.environment_id == "env_test_1"
        assert isinstance(first.episode_reward, float)
        assert isinstance(first.success, bool)
        assert first.collision_count >= 0

    def test_policy_persists_across_multiple_train_calls(self):
        trainer = AgentTrainer(seed=1)
        trainer.train(TINY_CONFIG, total_timesteps=256, environment_id="env_a")
        model_after_first = trainer.model
        trainer.train(TINY_CONFIG, total_timesteps=256, environment_id="env_b")
        # Same underlying PPO object continues learning — not recreated —
        # which is what makes curriculum-style repeated calls "adaptive".
        assert trainer.model is model_after_first

    def test_evaluate_before_train_raises(self):
        trainer = AgentTrainer(seed=1)
        with pytest.raises(RuntimeError):
            trainer.evaluate(TINY_CONFIG, episodes=2)

    def test_evaluate_returns_well_formed_result(self):
        trainer = AgentTrainer(seed=1)
        trainer.train(TINY_CONFIG, total_timesteps=256)
        result = trainer.evaluate(TINY_CONFIG, episodes=5, environment_id="env_eval")
        assert result.episodes == 5
        assert 0.0 <= result.success_rate <= 1.0
        assert 0.0 <= result.collision_rate <= 1.0
        assert len(result.episode_rewards) == 5
        assert len(result.episode_successes) == 5

    def test_deterministic_evaluation_is_reproducible(self):
        trainer = AgentTrainer(seed=1)
        trainer.train(TINY_CONFIG, total_timesteps=256)
        result_a = trainer.evaluate(TINY_CONFIG, episodes=5, deterministic=True)
        result_b = trainer.evaluate(TINY_CONFIG, episodes=5, deterministic=True)
        assert result_a.episode_rewards == result_b.episode_rewards
        assert result_a.episode_successes == result_b.episode_successes

    def test_checkpoint_save_and_load_round_trip(self):
        trainer = AgentTrainer(seed=1)
        trainer.train(TINY_CONFIG, total_timesteps=256)
        before = trainer.evaluate(TINY_CONFIG, episodes=5, deterministic=True)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.zip"
            trainer.save_checkpoint(checkpoint_path)
            assert checkpoint_path.exists()

            reloaded_trainer = AgentTrainer(seed=1)
            reloaded_trainer.load_checkpoint(checkpoint_path, config=TINY_CONFIG)
            after = reloaded_trainer.evaluate(TINY_CONFIG, episodes=5, deterministic=True)

        # A reloaded policy evaluated deterministically on the same seeds
        # must reproduce the exact same behavior as the original.
        assert before.episode_rewards == after.episode_rewards
        assert before.episode_successes == after.episode_successes

    def test_save_checkpoint_before_train_raises(self):
        trainer = AgentTrainer(seed=1)
        with pytest.raises(RuntimeError):
            trainer.save_checkpoint("/tmp/should_not_be_created.zip")
