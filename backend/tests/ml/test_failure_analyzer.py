import numpy as np
import pytest

from app.ml.analysis.failure_analyzer import FailureAnalyzer
from app.ml.analysis.trajectory import Trajectory, TrajectoryStep, summarize
from app.ml.environments.gridworld import Action
from app.ml.environments.validator import EnvironmentValidator, GridWorldLayout

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


def _open_layout(n=6, start=(0, 0), goal=(4, 4)):
    return GridWorldLayout(grid_size=n, agent_start=start, goal=goal, obstacles=frozenset())


def _reports_by_type(reports):
    return {r.failure_type: r for r in reports}


class TestFailureAnalyzerCoverage:
    def test_returns_all_seven_failure_types(self):
        layout = _open_layout()
        steps = [_step(0, Action.RIGHT, (0, 0), (1, 0), done=True)]
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=layout.goal, steps=steps)
        summary = summarize(traj)
        reports = FailureAnalyzer().analyze(traj, summary, layout)
        types = {r.failure_type for r in reports}
        assert types == {
            "obstacle_avoidance",
            "dead_end_recovery",
            "exploration",
            "repeated_action_loops",
            "path_inefficiency",
            "short_horizon_planning",
            "long_horizon_planning",
        }
        for r in reports:
            assert 0.0 <= r.score <= 1.0
            assert 0.0 <= r.confidence <= 1.0


class TestObstacleAvoidance:
    def test_high_collision_rate_yields_high_score(self):
        layout = _open_layout()
        steps = [
            _step(i, Action.RIGHT, (0, 0), (0, 0), collided=True) for i in range(10)
        ] + [_step(10, Action.DOWN, (0, 0), (0, 1), done=True)]
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=layout.goal, steps=steps)
        summary = summarize(traj)
        report = _reports_by_type(FailureAnalyzer().analyze(traj, summary, layout))["obstacle_avoidance"]
        assert report.score > 0.9

    def test_no_collisions_yields_zero_score(self):
        layout = _open_layout()
        steps = [_step(i, Action.RIGHT, (i, 0), (i + 1, 0)) for i in range(5)]
        steps[-1] = _step(4, Action.RIGHT, (4, 0), (5, 0), done=True)
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=layout.goal, steps=steps)
        summary = summarize(traj)
        report = _reports_by_type(FailureAnalyzer().analyze(traj, summary, layout))["obstacle_avoidance"]
        assert report.score == 0.0


class TestDeadEndRecoveryDetector:
    def _spur_layout(self):
        # Open 3x3 room (0,0)-(2,2) plus a spur (3,1)-(4,1).
        n = 5
        all_cells = {(x, y) for x in range(n) for y in range(n)}
        keep = {(x, y) for x in range(3) for y in range(3)} | {(3, 1), (4, 1)}
        obstacles = all_cells - keep
        return GridWorldLayout(grid_size=n, agent_start=(0, 0), goal=(2, 2), obstacles=frozenset(obstacles))

    def test_entering_and_recovering_from_spur_is_detected(self):
        layout = self._spur_layout()
        # Walk into the spur tip at (4,1) then back out to (2,1), then to goal.
        steps = [
            _step(0, Action.RIGHT, (0, 0), (1, 0)),
            _step(1, Action.RIGHT, (1, 0), (2, 0)),
            _step(2, Action.DOWN, (2, 0), (2, 1)),
            _step(3, Action.RIGHT, (2, 1), (3, 1)),   # enters dead-end zone
            _step(4, Action.RIGHT, (3, 1), (4, 1)),   # deeper into zone
            _step(5, Action.LEFT, (4, 1), (3, 1)),
            _step(6, Action.LEFT, (3, 1), (2, 1)),    # exits zone (recovery)
            _step(7, Action.DOWN, (2, 1), (2, 2), done=True),
        ]
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=layout.goal, steps=steps)
        summary = summarize(traj)
        report = _reports_by_type(FailureAnalyzer().analyze(traj, summary, layout))["dead_end_recovery"]
        assert report.features["dead_end_entries"] == 1
        assert report.features["successful_recoveries"] == 1
        assert report.score > 0.0

    def test_no_dead_end_encountered_yields_zero_score_low_confidence(self):
        layout = self._spur_layout()
        steps = [
            _step(0, Action.RIGHT, (0, 0), (1, 0)),
            _step(1, Action.DOWN, (1, 0), (1, 1)),
            _step(2, Action.DOWN, (1, 1), (1, 2), done=True),
        ]
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=layout.goal, steps=steps)
        summary = summarize(traj)
        report = _reports_by_type(FailureAnalyzer().analyze(traj, summary, layout))["dead_end_recovery"]
        assert report.score == 0.0
        assert report.confidence < 0.5


class TestExplorationDetector:
    def test_stuck_in_place_and_failed_scores_high(self):
        layout = _open_layout()
        steps = [_step(i, Action.RIGHT, (0, 0), (0, 0), collided=True) for i in range(15)]
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=layout.goal, steps=steps)
        summary = summarize(traj)
        assert not summary.success
        report = _reports_by_type(FailureAnalyzer().analyze(traj, summary, layout))["exploration"]
        assert report.score > 0.7

    def test_successful_episode_scores_low_even_with_repetition(self):
        layout = _open_layout()
        steps = [_step(i, Action.RIGHT, (0, 0), (0, 0)) for i in range(5)]
        steps.append(_step(5, Action.DOWN, (0, 0), (0, 1), done=True))
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=(0, 1), steps=steps)
        summary = summarize(traj)
        assert summary.success
        report = _reports_by_type(FailureAnalyzer().analyze(traj, summary, layout))["exploration"]
        assert report.score < 0.3


class TestRepeatedActionLoops:
    def test_oscillation_yields_high_score(self):
        layout = _open_layout()
        steps = []
        for i in range(10):
            if i % 2 == 0:
                steps.append(_step(i, Action.RIGHT, (0, 0), (1, 0)))
            else:
                steps.append(_step(i, Action.LEFT, (1, 0), (0, 0)))
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=layout.goal, steps=steps)
        summary = summarize(traj)
        report = _reports_by_type(FailureAnalyzer().analyze(traj, summary, layout))["repeated_action_loops"]
        assert report.score > 0.8

    def test_direct_path_yields_low_score(self):
        layout = _open_layout()
        steps = [_step(i, Action.RIGHT, (i, 0), (i + 1, 0)) for i in range(4)]
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=(4, 0), steps=steps)
        summary = summarize(traj)
        report = _reports_by_type(FailureAnalyzer().analyze(traj, summary, layout))["repeated_action_loops"]
        assert report.score == 0.0


class TestPathInefficiency:
    def setup_method(self):
        self.validator = EnvironmentValidator()

    def test_ratio_at_shortest_path_scores_zero(self):
        layout = _open_layout(n=5, start=(0, 0), goal=(4, 0))
        validation = self.validator.validate(layout)
        steps = [_step(i, Action.RIGHT, (i, 0), (i + 1, 0)) for i in range(4)]
        steps[-1] = _step(3, Action.RIGHT, (3, 0), (4, 0), done=True)
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=(4, 0), steps=steps)
        summary = summarize(traj)
        analyzer = FailureAnalyzer(self.validator)
        report = _reports_by_type(analyzer.analyze(traj, summary, layout, validation))["path_inefficiency"]
        assert report.score == pytest.approx(0.0, abs=1e-6)

    def test_much_longer_than_shortest_path_scores_high(self):
        layout = _open_layout(n=5, start=(0, 0), goal=(1, 0))
        validation = self.validator.validate(layout)
        # Shortest path is 1 step, but take 12 steps of back-and-forth.
        steps = []
        pos = (0, 0)
        for i in range(12):
            nxt = (1, 0) if pos == (0, 0) else (0, 0)
            steps.append(_step(i, Action.RIGHT, pos, nxt, done=(i == 11 and nxt == (1, 0))))
            pos = nxt
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=(1, 0), steps=steps)
        summary = summarize(traj)
        analyzer = FailureAnalyzer(self.validator)
        report = _reports_by_type(analyzer.analyze(traj, summary, layout, validation))["path_inefficiency"]
        assert report.score > 0.8


class TestLongHorizonPlanning:
    def setup_method(self):
        self.validator = EnvironmentValidator()

    def test_success_always_scores_zero_regardless_of_distance(self):
        layout = _open_layout(n=10, start=(0, 0), goal=(9, 9))
        validation = self.validator.validate(layout)
        steps = [_step(0, Action.RIGHT, (0, 0), (1, 0), done=True)]
        # Force success flag by matching final position to goal exactly.
        traj = Trajectory(episode_id="e", environment_id="env", goal_pos=(1, 0), steps=steps)
        summary = summarize(traj)
        analyzer = FailureAnalyzer(self.validator)
        report = _reports_by_type(analyzer.analyze(traj, summary, layout, validation))["long_horizon_planning"]
        assert report.score == 0.0

    def test_failure_on_distant_goal_scores_higher_than_failure_on_near_goal(self):
        near_layout = _open_layout(n=10, start=(0, 0), goal=(1, 0))
        far_layout = _open_layout(n=10, start=(0, 0), goal=(9, 9))
        near_validation = self.validator.validate(near_layout)
        far_validation = self.validator.validate(far_layout)

        failed_step = [_step(0, Action.UP, (0, 0), (0, 0), done=True)]  # never reaches goal
        near_traj = Trajectory(episode_id="e1", environment_id="env", goal_pos=near_layout.goal, steps=failed_step)
        far_traj = Trajectory(episode_id="e2", environment_id="env", goal_pos=far_layout.goal, steps=failed_step)

        analyzer = FailureAnalyzer(self.validator)
        near_report = _reports_by_type(
            analyzer.analyze(near_traj, summarize(near_traj), near_layout, near_validation)
        )["long_horizon_planning"]
        far_report = _reports_by_type(
            analyzer.analyze(far_traj, summarize(far_traj), far_layout, far_validation)
        )["long_horizon_planning"]

        assert far_report.score > near_report.score
