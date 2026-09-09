"""Phase 28 (Section 27) -- agent_mode tests.

The single most important assertion in this file, repeated across both
non-training modes: creating/running an environment_only or external
experiment must never construct `stable_baselines3.PPO`. We verify this
structurally (monkeypatching `PPO` itself, not just checking output shape)
so a future regression that sneaks a trainer into the wrong code path
would fail loudly here.
"""

from __future__ import annotations

import pytest

from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.manager import ExperimentManager

TINY_KWARGS = dict(
    grid_size=6,
    obstacle_density=0.15,
    dead_end_probability=0.15,
    path_complexity=0.15,
    max_steps_per_episode=20,
    n_curriculum_steps=3,
    n_candidates=3,
    trajectories_per_analysis=2,
    timesteps_per_step=256,
    eval_episodes=2,
    run_generalization=False,
)


@pytest.fixture
def forbid_ppo(monkeypatch):
    """Fails the test immediately if `stable_baselines3.PPO` (as imported
    by AgentTrainer) is ever constructed."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            "PPO was instantiated -- this must never happen outside agent_mode='demo_ppo'"
        )

    monkeypatch.setattr("app.ml.training.trainer.PPO", _boom)
    return _boom


class TestEnvironmentOnlyMode:
    def test_is_the_default_agent_mode(self):
        config = ExperimentConfig(name="default_check", **TINY_KWARGS)
        assert config.agent_mode == "environment_only"

    def test_generates_environments_without_instantiating_ppo(self, forbid_ppo):
        config = ExperimentConfig(name="env_only_no_ppo", agent_mode="environment_only", **TINY_KWARGS)
        result = ExperimentManager(config).run()

        assert result.generated_environments is not None
        assert len(result.generated_environments) == config.n_curriculum_steps
        # No agent-training artifacts should be produced in this mode.
        assert result.curriculum_history == []
        assert result.weakness_history is None
        assert result.generalization_results is None

    def test_generated_environments_have_no_agent_performance_fields(self):
        config = ExperimentConfig(name="env_only_no_fake_metrics", agent_mode="environment_only", **TINY_KWARGS)
        result = ExperimentManager(config).run()

        for record in result.generated_environments:
            assert "success_rate" not in record
            assert "average_reward" not in record
            assert "reward" not in record
            assert "episode" not in record
            # But real, environment-only structural metrics must be present.
            assert "difficulty" in record
            assert "metrics" in record
            assert "generation_reason" in record
            assert record["metrics"]["reachable_cells"] >= 0

    def test_emits_generation_events_not_training_events(self):
        config = ExperimentConfig(name="env_only_events", agent_mode="environment_only", **TINY_KWARGS)
        events: list[dict] = []
        ExperimentManager(config).run(progress_callback=events.append)

        event_types = {e["type"] for e in events}
        assert "experiment_started" in event_types
        assert "environment_selected" in event_types
        assert "generation_completed" in event_types
        # None of the training-only event vocabulary should ever appear.
        forbidden = {"training_started", "episode_updated", "training_metrics_updated", "training_completed"}
        assert forbidden.isdisjoint(event_types)

    @pytest.mark.parametrize("strategy", ["diverse", "progressive", "random"])
    def test_all_generation_strategies_run_without_ppo(self, strategy, forbid_ppo):
        config = ExperimentConfig(
            name=f"env_only_{strategy}", agent_mode="environment_only", generation_strategy=strategy, **TINY_KWARGS
        )
        result = ExperimentManager(config).run()
        assert len(result.generated_environments) == config.n_curriculum_steps


class TestExternalAgentMode:
    def test_run_raises_instead_of_silently_training(self):
        # external-agent experiments are driven via reset()/step(), never
        # ExperimentManager.run() -- see manager.py's docstring/run().
        config = ExperimentConfig(name="external_no_run", agent_mode="external", **TINY_KWARGS)
        with pytest.raises(ValueError):
            ExperimentManager(config).run()

    def test_external_session_never_instantiates_ppo(self, forbid_ppo):
        from app.ml.agents.external_session import ExternalAgentSession

        config = ExperimentConfig(name="external_session_no_ppo", agent_mode="external", **TINY_KWARGS)
        session = ExternalAgentSession(experiment_id=99999, config=config, seed=config.seed)
        # Exercise the parts of the session that don't require a DB, to
        # keep this test's PPO-guard fast and dependency-free; DB-backed
        # reset()/step() flow is covered in tests/db/test_external_agent.py.
        base_cfg = session.base_config()
        assert base_cfg.grid_size == config.grid_size


class TestDemoPPOMode:
    def test_demo_ppo_does_instantiate_ppo(self, monkeypatch):
        calls = []
        import app.ml.training.trainer as trainer_module

        original_ppo = trainer_module.PPO

        def _tracking_ppo(*args, **kwargs):
            calls.append((args, kwargs))
            return original_ppo(*args, **kwargs)

        monkeypatch.setattr(trainer_module, "PPO", _tracking_ppo)

        config = ExperimentConfig(
            name="demo_ppo_instantiates",
            agent_mode="demo_ppo",
            curriculum_type="random",
            **TINY_KWARGS,
        )
        result = ExperimentManager(config).run(save_checkpoint=False)

        assert len(calls) >= 1, "Demo Agent -- PPO must instantiate stable_baselines3.PPO"
        assert len(result.curriculum_history) == config.n_curriculum_steps

    def test_demo_ppo_produces_training_metrics(self):
        config = ExperimentConfig(
            name="demo_ppo_metrics", agent_mode="demo_ppo", curriculum_type="random", **TINY_KWARGS
        )
        result = ExperimentManager(config).run(save_checkpoint=False)
        for step in result.curriculum_history:
            step_dict = step.to_dict()
            assert "success_rate" in step_dict
            assert "average_reward" in step_dict


class TestNoHiddenCoupling:
    """Section 30 -- structural guard against re-introducing implicit PPO.
    Checks actual import/call statements, not prose (both modules'
    docstrings legitimately discuss PPO to explain why it's absent)."""

    @staticmethod
    def _code_lines(source: str) -> list[str]:
        lines = []
        in_docstring = False
        for line in source.splitlines():
            stripped = line.strip()
            triple_count = stripped.count('"""')
            if not in_docstring and stripped.startswith('"""'):
                if triple_count >= 2:
                    continue  # entire docstring is on this one line
                in_docstring = True
                continue
            if in_docstring:
                if triple_count >= 1:
                    in_docstring = False
                continue
            if stripped.startswith("#"):
                continue
            lines.append(line)
        return lines

    def test_environment_only_generator_module_has_no_ppo_reference(self):
        import inspect

        from app.ml.environments import environment_only_generator

        code = "\n".join(self._code_lines(inspect.getsource(environment_only_generator)))
        for banned in ("PPO", "AgentTrainer", "stable_baselines3", "model.learn"):
            assert banned not in code, f"{banned!r} found in executable code of environment_only_generator.py"

    def test_external_session_module_has_no_ppo_reference(self):
        import inspect

        from app.ml.agents import external_session

        code = "\n".join(self._code_lines(inspect.getsource(external_session)))
        for banned in ("PPO", "AgentTrainer", "stable_baselines3", "model.learn"):
            assert banned not in code, f"{banned!r} found in executable code of external_session.py"
