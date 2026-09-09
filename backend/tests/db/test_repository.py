from app.database.repository import get_experiment, persist_experiment_result
from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.manager import ExperimentManager
from app.models.orm import AgentWeakness, Environment, LearningProgress, ModelCheckpoint, TrainingRun

TINY_KWARGS = dict(
    agent_mode="demo_ppo",
    grid_size=5,
    obstacle_density=0.1,
    dead_end_probability=0.1,
    path_complexity=0.1,
    max_steps_per_episode=12,
    n_curriculum_steps=2,
    timesteps_per_step=256,
    eval_episodes=3,
    n_candidates=2,
    trajectories_per_analysis=2,
    run_generalization=True,
    generalization_envs_per_split=2,
    generalization_episodes_per_config=2,
)


class TestPersistExperimentResult:
    def test_persists_failure_driven_experiment_end_to_end(self, session):
        config = ExperimentConfig(name="db_test_failure_driven", curriculum_type="failure_driven", seed=1, **TINY_KWARGS)
        result = ExperimentManager(config).run()

        experiment = persist_experiment_result(session, result)
        session.flush()

        assert experiment.id is not None
        assert experiment.curriculum_type == "failure_driven"
        assert len(experiment.environments) == config.n_curriculum_steps
        assert len(experiment.training_runs) == config.n_curriculum_steps
        assert len(experiment.evaluations) == 3

        weakness_count = (
            session.query(AgentWeakness)
            .join(TrainingRun)
            .filter(TrainingRun.experiment_id == experiment.id)
            .count()
        )
        assert weakness_count == config.n_curriculum_steps

        lp_count = (
            session.query(LearningProgress)
            .join(Environment)
            .filter(Environment.experiment_id == experiment.id)
            .count()
        )
        assert lp_count == config.n_curriculum_steps

    def test_persists_random_baseline_without_weakness_rows(self, session):
        config = ExperimentConfig(name="db_test_random", curriculum_type="random", seed=1, **TINY_KWARGS)
        result = ExperimentManager(config).run()

        experiment = persist_experiment_result(session, result)
        session.flush()

        assert len(experiment.environments) == config.n_curriculum_steps
        weakness_count = (
            session.query(AgentWeakness)
            .join(TrainingRun)
            .filter(TrainingRun.experiment_id == experiment.id)
            .count()
        )
        assert weakness_count == 0

    def test_get_experiment_retrieves_persisted_row(self, session):
        config = ExperimentConfig(name="db_test_get", curriculum_type="random", seed=1, **TINY_KWARGS)
        result = ExperimentManager(config).run()
        experiment = persist_experiment_result(session, result)
        session.flush()

        fetched = get_experiment(session, experiment.id)
        assert fetched is not None
        assert fetched.name == "db_test_get"

    def test_environment_params_roundtrip_through_jsonb(self, session):
        config = ExperimentConfig(name="db_test_params", curriculum_type="random", seed=1, **TINY_KWARGS)
        result = ExperimentManager(config).run()
        experiment = persist_experiment_result(session, result)
        session.flush()

        env = experiment.environments[0]
        assert env.params["grid_size"] == config.grid_size
        assert "obstacle_density" in env.params

    def test_evaluation_splits_are_validation_test_ood(self, session):
        config = ExperimentConfig(
            name="db_test_splits", curriculum_type="fixed", seed=1, **{**TINY_KWARGS, "n_curriculum_steps": 4}
        )
        result = ExperimentManager(config).run()
        experiment = persist_experiment_result(session, result)
        session.flush()

        splits = {e.split for e in experiment.evaluations}
        assert splits == {"validation", "test", "ood"}


class TestModelCheckpointPersistence:
    def test_checkpoint_attached_to_last_training_run(self, session):
        config = ExperimentConfig(name="db_test_checkpoint", curriculum_type="random", seed=1, **TINY_KWARGS)
        result = ExperimentManager(config).run()
        experiment = persist_experiment_result(session, result)
        session.flush()

        checkpoint_count = (
            session.query(ModelCheckpoint)
            .join(TrainingRun)
            .filter(TrainingRun.experiment_id == experiment.id)
            .count()
        )
        assert checkpoint_count == 1

        checkpoint = (
            session.query(ModelCheckpoint)
            .join(TrainingRun)
            .filter(TrainingRun.experiment_id == experiment.id)
            .one()
        )
        assert checkpoint.training_run_id == experiment.training_runs[-1].id
        assert checkpoint.path == result.reproducibility["checkpoint_path"]
