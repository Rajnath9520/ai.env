import numpy as np
import pytest

from app.ml.analysis.trajectory import Trajectory, TrajectoryStep, summarize
from app.ml.analysis.trajectory_logger import TrajectoryLogger, TrajectoryStore
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.gridworld import Action, GridWorldEnv

_DUMMY_STATE = np.zeros(13, dtype=np.float32)


def _step(step_number, action, agent_pos, next_agent_pos, done=False, collided=False, reward=-0.01):
    return TrajectoryStep(
        step_number=step_number,
        state=_DUMMY_STATE,
        action=action,
        reward=reward,
        next_state=_DUMMY_STATE,
        done=done,
        timestamp=0.0,
        agent_pos=agent_pos,
        next_agent_pos=next_agent_pos,
        collided=collided,
    )


class TestSummarize:
    def test_empty_trajectory(self):
        traj = Trajectory(episode_id="e0", environment_id="env0", goal_pos=(3, 3), steps=[])
        summary = summarize(traj)
        assert summary.trajectory_length == 0
        assert summary.success is False
        assert summary.total_reward == 0.0

    def test_simple_successful_path_has_no_repeats_or_reversals(self):
        # RIGHT, RIGHT, DOWN, DOWN — straight path to a goal at (2,2), no backtracking.
        steps = [
            _step(0, Action.RIGHT, (0, 0), (1, 0)),
            _step(1, Action.RIGHT, (1, 0), (2, 0)),
            _step(2, Action.DOWN, (2, 0), (2, 1)),
            _step(3, Action.DOWN, (2, 1), (2, 2), done=True, reward=1.0),
        ]
        traj = Trajectory(episode_id="e1", environment_id="env1", goal_pos=(2, 2), steps=steps)
        summary = summarize(traj)

        assert summary.trajectory_length == 4
        assert summary.success is True
        assert summary.distance_to_goal == 0
        assert summary.repeated_states == 0
        assert summary.direction_reversals == 0
        assert summary.unique_states_visited == 5  # start + 4 next positions, all distinct
        assert summary.total_reward == pytest.approx(-0.03 + 1.0)

    def test_direction_reversal_detected(self):
        # RIGHT then LEFT is a direct reversal back to the start.
        steps = [
            _step(0, Action.RIGHT, (0, 0), (1, 0)),
            _step(1, Action.LEFT, (1, 0), (0, 0)),
        ]
        traj = Trajectory(episode_id="e2", environment_id="env1", goal_pos=(5, 5), steps=steps)
        summary = summarize(traj)
        assert summary.direction_reversals == 1

    def test_repeated_states_counted(self):
        # Agent returns to (0,0), a previously visited position.
        steps = [
            _step(0, Action.RIGHT, (0, 0), (1, 0)),
            _step(1, Action.LEFT, (1, 0), (0, 0)),
            _step(2, Action.RIGHT, (0, 0), (1, 0)),
        ]
        traj = Trajectory(episode_id="e3", environment_id="env1", goal_pos=(9, 9), steps=steps)
        summary = summarize(traj)
        # visited sequence: (0,0) [start], (1,0), (0,0), (1,0) -> two repeats
        assert summary.repeated_states == 2

    def test_collision_counted_but_position_unchanged(self):
        steps = [
            _step(0, Action.RIGHT, (0, 0), (0, 0), collided=True),
            _step(1, Action.DOWN, (0, 0), (0, 1)),
        ]
        traj = Trajectory(episode_id="e4", environment_id="env1", goal_pos=(0, 1), steps=steps)
        summary = summarize(traj)
        assert summary.collision_count == 1

    def test_failed_episode_reports_positive_distance_to_goal(self):
        steps = [_step(0, Action.RIGHT, (0, 0), (1, 0), done=True, reward=-0.01)]
        traj = Trajectory(episode_id="e5", environment_id="env1", goal_pos=(5, 5), steps=steps)
        summary = summarize(traj)
        assert summary.success is False
        assert summary.distance_to_goal == 9  # |5-1| + |5-0|


class TestTrajectoryLogger:
    def test_run_episode_against_real_env(self):
        config = GridWorldConfig(grid_size=6, obstacle_density=0.0, max_steps=20, seed=1)
        env = GridWorldEnv(config=config)
        logger = TrajectoryLogger()
        rng = np.random.default_rng(0)

        traj = logger.run_episode(
            env,
            policy_fn=lambda obs: int(rng.integers(4)),
            seed=1,
            episode_id="ep-1",
            environment_id="env-1",
        )

        assert traj.episode_id == "ep-1"
        assert traj.environment_id == "env-1"
        assert len(traj) > 0
        assert len(traj) <= config.max_steps
        # step numbers are sequential starting at 0
        assert [s.step_number for s in traj.steps] == list(range(len(traj)))

        summary = summarize(traj)
        assert summary.trajectory_length == len(traj)


class TestTrajectoryStore:
    def test_add_returns_summary_and_stores_raw(self):
        store = TrajectoryStore(max_raw_trajectories=10)
        traj = Trajectory(episode_id="e1", environment_id="env1", goal_pos=(1, 1), steps=[])
        summary = store.add(traj)
        assert store.get_summary("e1") is summary
        assert store.get_raw("e1") is traj
        assert store.raw_count() == 1
        assert store.summary_count() == 1

    def test_raw_trajectories_evicted_beyond_cap(self):
        store = TrajectoryStore(max_raw_trajectories=3)
        for i in range(5):
            store.add(Trajectory(episode_id=f"e{i}", environment_id="env1", goal_pos=(0, 0), steps=[]))

        assert store.raw_count() == 3
        # Oldest (e0, e1) evicted; newest three retained.
        assert store.get_raw("e0") is None
        assert store.get_raw("e1") is None
        assert store.get_raw("e4") is not None
        # All summaries retained regardless of the raw cap.
        assert store.summary_count() == 5

    def test_zero_cap_stores_no_raw_but_keeps_summaries(self):
        store = TrajectoryStore(max_raw_trajectories=0)
        store.add(Trajectory(episode_id="e0", environment_id="env1", goal_pos=(0, 0), steps=[]))
        assert store.raw_count() == 0
        assert store.summary_count() == 1

    def test_negative_cap_rejected(self):
        with pytest.raises(ValueError):
            TrajectoryStore(max_raw_trajectories=-1)
