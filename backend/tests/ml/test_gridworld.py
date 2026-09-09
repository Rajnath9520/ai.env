import numpy as np
import pytest

from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GenerationError, GridWorldGenerator
from app.ml.environments.gridworld import Action, GridWorldEnv
from app.ml.environments.validator import EnvironmentValidator, GridWorldLayout


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestGridWorldConfig:
    def test_defaults_are_valid(self):
        GridWorldConfig()  # should not raise

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"grid_size": 2},
            {"obstacle_density": 1.5},
            {"obstacle_density": -0.1},
            {"dead_end_probability": 1.5},
            {"path_complexity": -0.1},
            {"goal_distance": 0},
            {"max_steps": 0},
        ],
    )
    def test_invalid_params_raise(self, kwargs):
        with pytest.raises(ValueError):
            GridWorldConfig(**kwargs)

    def test_difficulty_estimate_in_unit_range(self):
        for density in (0.0, 0.3, 0.9):
            cfg = GridWorldConfig(obstacle_density=density)
            assert 0.0 <= cfg.difficulty_estimate() <= 1.0


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class TestEnvironmentValidator:
    def setup_method(self):
        self.validator = EnvironmentValidator()

    def test_valid_open_layout(self):
        layout = GridWorldLayout(
            grid_size=5,
            agent_start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(),
        )
        result = self.validator.validate(layout)
        assert result.is_valid
        assert result.shortest_path_length == 8  # Manhattan distance, open grid

    def test_unreachable_goal_is_invalid(self):
        n = 5
        # Fully seal row y=3, isolating goal (4,4) from start (0,0).
        obstacles = {(x, 3) for x in range(n)}
        layout = GridWorldLayout(
            grid_size=n,
            agent_start=(0, 0),
            goal=(4, 4),
            obstacles=frozenset(obstacles),
        )
        result = self.validator.validate(layout)
        assert not result.is_valid
        assert "not reachable" in result.reasons[0]

    def test_start_equals_goal_is_invalid(self):
        layout = GridWorldLayout(
            grid_size=5, agent_start=(1, 1), goal=(1, 1), obstacles=frozenset()
        )
        result = self.validator.validate(layout)
        assert not result.is_valid

    def test_start_on_obstacle_is_invalid(self):
        layout = GridWorldLayout(
            grid_size=5,
            agent_start=(1, 1),
            goal=(3, 3),
            obstacles=frozenset({(1, 1)}),
        )
        result = self.validator.validate(layout)
        assert not result.is_valid

    def test_dead_end_counting(self):
        # A single-cell spur off an open room is a dead end.
        n = 5
        obstacles = {(x, y) for x in range(n) for y in range(n)}
        obstacles -= {(1, 1), (1, 2), (2, 1)}  # small open room
        obstacles -= {(3, 1)}  # spur cell reachable only via (2,1)
        layout = GridWorldLayout(
            grid_size=n,
            agent_start=(1, 1),
            goal=(1, 2),
            obstacles=frozenset(obstacles),
        )
        result = self.validator.validate(layout)
        assert result.is_valid
        assert result.dead_end_count >= 1


# ---------------------------------------------------------------------------
# Generator — the critical "never produces an invalid environment" guarantee
# ---------------------------------------------------------------------------

class TestGridWorldGenerator:
    def setup_method(self):
        self.generator = GridWorldGenerator()

    @pytest.mark.parametrize("seed", range(25))
    def test_generated_layout_always_valid_default_config(self, seed):
        config = GridWorldConfig(seed=seed)
        report = self.generator.generate(config)
        assert report.validation.is_valid

    @pytest.mark.parametrize(
        "grid_size,obstacle_density,dead_end_probability,path_complexity",
        [
            (4, 0.0, 0.0, 0.0),
            (6, 0.5, 0.5, 0.5),
            (10, 0.7, 0.8, 0.8),
            (15, 0.3, 0.1, 0.9),
            (8, 0.9, 0.9, 0.9),  # extreme combination
        ],
    )
    def test_generated_layout_valid_across_param_grid(
        self, grid_size, obstacle_density, dead_end_probability, path_complexity
    ):
        config = GridWorldConfig(
            grid_size=grid_size,
            obstacle_density=obstacle_density,
            dead_end_probability=dead_end_probability,
            path_complexity=path_complexity,
            seed=123,
        )
        report = self.generator.generate(config)
        assert report.validation.is_valid

    def test_goal_distance_respected_when_feasible(self):
        config = GridWorldConfig(grid_size=10, goal_distance=8, seed=7)
        report = self.generator.generate(config)
        start = report.layout.agent_start
        goal = report.layout.goal
        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        assert manhattan >= 8

    def test_same_seed_is_deterministic(self):
        config = GridWorldConfig(seed=42)
        report_a = self.generator.generate(config)
        report_b = self.generator.generate(config)
        assert report_a.layout == report_b.layout

    def test_different_seeds_usually_differ(self):
        layouts = {
            self.generator.generate(GridWorldConfig(seed=s)).layout for s in range(10)
        }
        assert len(layouts) > 1


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class TestGridWorldEnv:
    def _make_env(self, **config_kwargs) -> GridWorldEnv:
        config = GridWorldConfig(**{**{"seed": 1}, **config_kwargs})
        return GridWorldEnv(config=config)

    def test_reset_returns_valid_observation(self):
        env = self._make_env()
        obs, info = env.reset(seed=1)
        assert env.observation_space.contains(obs)
        assert "agent_pos" in info

    def test_action_space_has_four_discrete_actions(self):
        env = self._make_env()
        assert env.action_space.n == 4
        assert set(Action) == {Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT}

    def test_step_moves_agent_and_returns_gym_tuple(self):
        env = self._make_env(grid_size=6, obstacle_density=0.0)
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(int(Action.RIGHT))
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_collision_with_obstacle_does_not_move_agent(self):
        env = self._make_env(grid_size=5, obstacle_density=0.0)
        env.reset(seed=1)
        # Manually inject an obstacle directly to the right of the agent.
        ax, ay = env._agent_pos
        blocked = (ax + 1, ay)
        env._layout = env._layout.__class__(
            grid_size=env._layout.grid_size,
            agent_start=env._layout.agent_start,
            goal=env._layout.goal,
            obstacles=frozenset(env._layout.obstacles | {blocked}),
        )
        before = env._agent_pos
        _, reward, _, _, info = env.step(int(Action.RIGHT))
        assert env._agent_pos == before
        assert info["collided"] is True
        assert reward < 0

    def test_reaching_goal_terminates_with_positive_reward(self):
        env = self._make_env(grid_size=4, goal_distance=1, obstacle_density=0.0, path_complexity=0.0, dead_end_probability=0.0)
        env.reset(seed=3)
        # Walk directly toward the goal until termination or a step budget
        # is exhausted; grid_size=4 with no obstacles guarantees a short path.
        terminated = False
        for _ in range(50):
            layout = env._layout
            ax, ay = env._agent_pos
            gx, gy = layout.goal
            if ax < gx:
                action = Action.RIGHT
            elif ax > gx:
                action = Action.LEFT
            elif ay < gy:
                action = Action.DOWN
            elif ay > gy:
                action = Action.UP
            else:
                break
            _, reward, terminated, truncated, info = env.step(int(action))
            if terminated:
                assert info["reached_goal"] is True
                assert reward > 0
                break
        assert terminated

    def test_step_limit_truncates_episode(self):
        env = self._make_env(grid_size=8, max_steps=3, obstacle_density=0.0)
        env.reset(seed=1)
        truncated = False
        for _ in range(3):
            # Bounce against an edge so we never accidentally reach the goal.
            _, _, terminated, truncated, _ = env.step(int(Action.UP))
        assert truncated

    def test_render_ansi_returns_grid_string(self):
        env = self._make_env(grid_size=5)
        env.reset(seed=1)
        env.render_mode = "ansi"
        rendered = env.render()
        assert isinstance(rendered, str)
        lines = rendered.split("\n")
        assert len(lines) == 5
        assert all(len(line) == 5 for line in lines)

    def test_reset_without_seed_reuses_config_seed_deterministically(self):
        env_a = self._make_env(seed=99)
        env_b = self._make_env(seed=99)
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        assert np.allclose(obs_a, obs_b)
