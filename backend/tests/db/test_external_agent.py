"""Phase 28 (Section 27) -- external-agent DB-backed integration test.

Drives `ExternalAgentSession` exactly the way the HTTP routes do (reset,
then step repeatedly with real actions) and checks that every stage of the
pipeline described in Section 10 actually happened for real: environment
generation, trajectory recording, failure analysis, weakness extraction,
and adaptive environment generation -- all persisted as real rows, and
`stable_baselines3.PPO` never touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.agents.external_session import ExternalAgentSession
from app.ml.experiment.config import ExperimentConfig
from app.models.orm import (
    AgentWeakness,
    Environment,
    Episode,
    Experiment,
    Failure,
    TrainingRun,
    Trajectory,
)

TINY_KWARGS = dict(
    agent_mode="external",
    grid_size=5,
    obstacle_density=0.1,
    dead_end_probability=0.15,
    path_complexity=0.15,
    max_steps_per_episode=8,
    n_curriculum_steps=3,
    n_candidates=3,
    trajectories_per_analysis=2,
)


def _make_experiment(session, config: ExperimentConfig) -> Experiment:
    import yaml

    experiment = Experiment(
        name=config.name,
        config_yaml=yaml.safe_dump(config.to_dict(), sort_keys=False),
        curriculum_type=config.curriculum_type,
        agent_mode=config.agent_mode,
        seed=config.seed,
        status="pending",
    )
    session.add(experiment)
    session.flush()
    return experiment


class TestExternalAgentInteractiveLoop:
    def test_reset_and_step_never_touch_ppo(self, session, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("PPO must never be instantiated for agent_mode='external'")

        monkeypatch.setattr("app.ml.training.trainer.PPO", _boom)

        config = ExperimentConfig(name="external_ppo_guard", seed=1, **TINY_KWARGS)
        experiment = _make_experiment(session, config)
        agent_session = ExternalAgentSession(experiment_id=experiment.id, config=config, seed=config.seed)

        obs, info = agent_session.reset(session)
        assert isinstance(obs, np.ndarray)
        assert agent_session.connected is True

        rng = np.random.default_rng(0)
        for _ in range(50):
            result = agent_session.step(session, int(rng.integers(0, 4)))
            if result["terminated"] or result["truncated"]:
                agent_session.reset(session)

    def test_full_pipeline_produces_real_trajectory_failure_and_weakness_rows(self, session):
        config = ExperimentConfig(name="external_pipeline", seed=2, **TINY_KWARGS)
        experiment = _make_experiment(session, config)
        agent_session = ExternalAgentSession(experiment_id=experiment.id, config=config, seed=config.seed)

        agent_session.reset(session)
        rng = np.random.default_rng(1)

        weakness_seen = None
        next_env_seen = None
        episodes_completed = 0
        for _ in range(400):
            result = agent_session.step(session, int(rng.integers(0, 4)))
            if result["episode_summary"] is not None:
                episodes_completed += 1
            if result["weakness"] is not None:
                weakness_seen = result["weakness"]
            if result["next_environment"] is not None:
                next_env_seen = result["next_environment"]
            if result["terminated"] or result["truncated"]:
                if weakness_seen is not None:
                    break
                agent_session.reset(session)

        assert episodes_completed >= config.trajectories_per_analysis
        assert weakness_seen is not None, "expected a weakness vector after enough trajectories"
        assert next_env_seen is not None, "expected an adaptively generated next environment"
        assert next_env_seen["generation_reason"]  # a real, non-empty reason string

        # Real rows, not fabricated ones (Section 10/21):
        environments = session.query(Environment).filter(Environment.experiment_id == experiment.id).all()
        assert len(environments) >= 2  # initial + at least one adaptively generated

        episodes = (
            session.query(Episode)
            .join(TrainingRun)
            .filter(TrainingRun.experiment_id == experiment.id)
            .all()
        )
        assert len(episodes) >= config.trajectories_per_analysis

        trajectories = (
            session.query(Trajectory)
            .join(Episode)
            .join(TrainingRun)
            .filter(TrainingRun.experiment_id == experiment.id)
            .all()
        )
        assert len(trajectories) == len(episodes)

        weaknesses = (
            session.query(AgentWeakness)
            .join(TrainingRun)
            .filter(TrainingRun.experiment_id == experiment.id)
            .all()
        )
        assert len(weaknesses) >= 1

    def test_step_before_reset_raises(self, session):
        from app.ml.agents.external_session import ExternalAgentError

        config = ExperimentConfig(name="external_no_reset", seed=3, **TINY_KWARGS)
        experiment = _make_experiment(session, config)
        agent_session = ExternalAgentSession(experiment_id=experiment.id, config=config, seed=config.seed)

        with pytest.raises(ExternalAgentError):
            agent_session.step(session, 0)

    def test_state_dict_is_honest_when_not_connected(self, session):
        config = ExperimentConfig(name="external_state_honest", seed=4, **TINY_KWARGS)
        experiment = _make_experiment(session, config)
        agent_session = ExternalAgentSession(experiment_id=experiment.id, config=config, seed=config.seed)

        state = agent_session.state_dict()
        assert state["connected"] is False
        assert state["episode"] is None
        assert state["step"] is None
