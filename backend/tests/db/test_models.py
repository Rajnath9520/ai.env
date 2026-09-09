import pytest
from sqlalchemy.exc import IntegrityError

from app.models.orm import (
    AgentWeakness,
    Environment,
    Episode,
    Evaluation,
    Experiment,
    Failure,
    LearningProgress,
    ModelCheckpoint,
    Trajectory,
    TrainingRun,
)


def _make_experiment(**overrides) -> Experiment:
    defaults = dict(
        name="exp_1", config_yaml="seed: 1\n", curriculum_type="failure_driven", seed=1, status="pending"
    )
    defaults.update(overrides)
    return Experiment(**defaults)


class TestExperimentAndEnvironment:
    def test_create_experiment_and_environment(self, session):
        experiment = _make_experiment()
        session.add(experiment)
        session.flush()

        env = Environment(
            experiment_id=experiment.id,
            environment_key="env_0000",
            params={"grid_size": 8, "obstacle_density": 0.2},
            difficulty=0.3,
            target_weakness="exploration",
            split="train",
        )
        session.add(env)
        session.flush()

        assert env.id is not None
        assert env.experiment.id == experiment.id
        assert experiment.environments[0].environment_key == "env_0000"

    def test_environment_key_unique_within_experiment(self, session):
        experiment = _make_experiment(name="exp_unique")
        session.add(experiment)
        session.flush()

        session.add(
            Environment(experiment_id=experiment.id, environment_key="env_dup", params={}, difficulty=0.1)
        )
        session.flush()
        session.add(
            Environment(experiment_id=experiment.id, environment_key="env_dup", params={}, difficulty=0.2)
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_deleting_experiment_cascades_to_environments(self, session):
        experiment = _make_experiment(name="exp_cascade")
        session.add(experiment)
        session.flush()
        env = Environment(experiment_id=experiment.id, environment_key="env_c", params={}, difficulty=0.1)
        session.add(env)
        session.flush()
        env_id = env.id

        session.delete(experiment)
        session.flush()

        assert session.get(Environment, env_id) is None


class TestTrainingRunEpisodeChain:
    def test_episode_trajectory_and_failures_chain(self, session):
        experiment = _make_experiment(name="exp_chain")
        session.add(experiment)
        session.flush()

        training_run = TrainingRun(experiment_id=experiment.id)
        session.add(training_run)
        session.flush()

        episode = Episode(
            training_run_id=training_run.id,
            episode_number=1,
            reward=0.9,
            steps=12,
            success=True,
            collisions=1,
        )
        session.add(episode)
        session.flush()

        session.add(
            Trajectory(
                episode_id=episode.id,
                trajectory_length=12,
                unique_states_visited=10,
                repeated_states=2,
                direction_reversals=1,
                distance_to_goal=0,
            )
        )
        session.add(
            Failure(
                episode_id=episode.id,
                failure_type="dead_end_recovery",
                score=0.4,
                confidence=0.7,
                features={"dead_end_entries": 2},
            )
        )
        session.flush()

        reloaded = session.get(Episode, episode.id)
        assert reloaded.trajectory.unique_states_visited == 10
        assert reloaded.failures[0].features["dead_end_entries"] == 2

    def test_episode_number_unique_within_training_run(self, session):
        experiment = _make_experiment(name="exp_epnum")
        session.add(experiment)
        session.flush()
        training_run = TrainingRun(experiment_id=experiment.id)
        session.add(training_run)
        session.flush()

        session.add(Episode(training_run_id=training_run.id, episode_number=1, reward=0.1, steps=5, success=False))
        session.flush()
        session.add(Episode(training_run_id=training_run.id, episode_number=1, reward=0.2, steps=6, success=True))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_deleting_training_run_cascades_to_episodes(self, session):
        experiment = _make_experiment(name="exp_tr_cascade")
        session.add(experiment)
        session.flush()
        training_run = TrainingRun(experiment_id=experiment.id)
        session.add(training_run)
        session.flush()
        episode = Episode(training_run_id=training_run.id, episode_number=1, reward=0.1, steps=5, success=False)
        session.add(episode)
        session.flush()
        episode_id = episode.id

        session.delete(training_run)
        session.flush()
        assert session.get(Episode, episode_id) is None


class TestWeaknessAndLearningProgress:
    def test_agent_weakness_row(self, session):
        experiment = _make_experiment(name="exp_weak")
        session.add(experiment)
        session.flush()
        training_run = TrainingRun(experiment_id=experiment.id)
        session.add(training_run)
        session.flush()

        weakness = AgentWeakness(
            training_run_id=training_run.id,
            obstacle_avoidance=0.1,
            exploration=0.8,
            dead_end_recovery=0.2,
            path_efficiency=0.3,
            short_horizon_planning=0.2,
            long_horizon_planning=0.4,
            primary_weakness="exploration",
            secondary_weaknesses=["path_efficiency"],
        )
        session.add(weakness)
        session.flush()

        reloaded = session.get(AgentWeakness, weakness.id)
        assert reloaded.primary_weakness == "exploration"
        assert reloaded.secondary_weaknesses == ["path_efficiency"]

    def test_learning_progress_row(self, session):
        experiment = _make_experiment(name="exp_lp")
        session.add(experiment)
        session.flush()
        env = Environment(experiment_id=experiment.id, environment_key="env_lp", params={}, difficulty=0.2)
        session.add(env)
        session.flush()

        session.add(
            LearningProgress(
                environment_id=env.id,
                metric="success_rate",
                performance_before=0.2,
                performance_after=0.5,
                learning_progress=0.3,
            )
        )
        session.flush()
        assert env.learning_progress_records[0].learning_progress == pytest.approx(0.3)


class TestEvaluationAndCheckpoint:
    def test_evaluation_row(self, session):
        experiment = _make_experiment(name="exp_eval")
        session.add(experiment)
        session.flush()
        session.add(
            Evaluation(
                experiment_id=experiment.id,
                split="test",
                success_rate=0.4,
                average_reward=0.1,
                average_steps=20.0,
                collision_rate=0.1,
            )
        )
        session.flush()
        assert experiment.evaluations[0].split == "test"

    def test_model_checkpoint_row(self, session):
        experiment = _make_experiment(name="exp_ckpt")
        session.add(experiment)
        session.flush()
        training_run = TrainingRun(experiment_id=experiment.id)
        session.add(training_run)
        session.flush()
        session.add(ModelCheckpoint(training_run_id=training_run.id, path="/tmp/ckpt.zip", episode=100))
        session.flush()
        assert training_run.checkpoints[0].path == "/tmp/ckpt.zip"
