import pytest

from app.ml.analysis.weakness_vector import WEAKNESS_DIMENSIONS, WeaknessVector
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator
from app.ml.environments.synthesizer import EnvironmentSynthesizer


def _weakness(primary: str, score: float = 0.9) -> WeaknessVector:
    values = {dim: 0.0 for dim in WEAKNESS_DIMENSIONS}
    values[primary] = score
    return WeaknessVector(**values, primary_weakness=primary, secondary_weaknesses=(), episodes_analyzed=10)


BASE_CONFIG = GridWorldConfig(
    grid_size=10,
    obstacle_density=0.2,
    goal_distance=4,
    dead_end_probability=0.2,
    path_complexity=0.2,
    max_steps=60,
    seed=1,
)


class TestTargetedAdjustment:
    def test_obstacle_avoidance_only_moves_obstacle_density(self):
        weakness = _weakness("obstacle_avoidance")
        result = EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG)

        assert result.obstacle_density > BASE_CONFIG.obstacle_density
        assert result.dead_end_probability == pytest.approx(BASE_CONFIG.dead_end_probability)
        assert result.path_complexity == pytest.approx(BASE_CONFIG.path_complexity)
        assert result.grid_size == BASE_CONFIG.grid_size

    def test_path_efficiency_only_moves_path_complexity(self):
        weakness = _weakness("path_efficiency")
        result = EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG)

        assert result.path_complexity > BASE_CONFIG.path_complexity
        assert result.obstacle_density == pytest.approx(BASE_CONFIG.obstacle_density)
        assert result.dead_end_probability == pytest.approx(BASE_CONFIG.dead_end_probability)
        assert result.grid_size == BASE_CONFIG.grid_size

    def test_dead_end_recovery_moves_dead_end_and_path_complexity_not_obstacles(self):
        weakness = _weakness("dead_end_recovery")
        result = EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG)

        assert result.dead_end_probability > BASE_CONFIG.dead_end_probability
        assert result.path_complexity > BASE_CONFIG.path_complexity
        assert result.obstacle_density == pytest.approx(BASE_CONFIG.obstacle_density)
        assert result.grid_size == BASE_CONFIG.grid_size

    def test_exploration_increases_grid_size_and_goal_distance_only(self):
        weakness = _weakness("exploration")
        result = EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG)

        assert result.grid_size > BASE_CONFIG.grid_size
        assert result.goal_distance > BASE_CONFIG.goal_distance
        assert result.obstacle_density == pytest.approx(BASE_CONFIG.obstacle_density)
        assert result.dead_end_probability == pytest.approx(BASE_CONFIG.dead_end_probability)

    def test_long_horizon_planning_increases_distance_complexity_and_dead_ends(self):
        weakness = _weakness("long_horizon_planning")
        result = EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG)

        assert result.goal_distance > BASE_CONFIG.goal_distance
        assert result.path_complexity > BASE_CONFIG.path_complexity
        assert result.dead_end_probability > BASE_CONFIG.dead_end_probability
        assert result.obstacle_density == pytest.approx(BASE_CONFIG.obstacle_density)
        assert result.grid_size == BASE_CONFIG.grid_size

    def test_zero_score_produces_no_change(self):
        weakness = _weakness("obstacle_avoidance", score=0.0)
        result = EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG)
        assert result.obstacle_density == pytest.approx(BASE_CONFIG.obstacle_density)
        assert result.dead_end_probability == pytest.approx(BASE_CONFIG.dead_end_probability)
        assert result.path_complexity == pytest.approx(BASE_CONFIG.path_complexity)

    def test_higher_score_produces_larger_adjustment(self):
        weak_low = _weakness("obstacle_avoidance", score=0.2)
        weak_high = _weakness("obstacle_avoidance", score=0.9)
        low_result = EnvironmentSynthesizer(seed=1).synthesize(weak_low, BASE_CONFIG)
        high_result = EnvironmentSynthesizer(seed=1).synthesize(weak_high, BASE_CONFIG)
        assert high_result.obstacle_density > low_result.obstacle_density

    def test_explicit_target_weakness_overrides_primary(self):
        weakness = WeaknessVector(
            **{**{dim: 0.0 for dim in WEAKNESS_DIMENSIONS}, "dead_end_recovery": 0.8},
            primary_weakness="obstacle_avoidance",
            secondary_weaknesses=(),
        )
        result = EnvironmentSynthesizer(seed=1).synthesize(
            weakness, BASE_CONFIG, target_weakness="dead_end_recovery"
        )
        assert result.dead_end_probability > BASE_CONFIG.dead_end_probability
        assert result.obstacle_density == pytest.approx(BASE_CONFIG.obstacle_density)

    def test_score_override_bypasses_weakness_vector_score(self):
        weakness = _weakness("obstacle_avoidance", score=0.0)  # vector score is 0
        result = EnvironmentSynthesizer(seed=1).synthesize(
            weakness, BASE_CONFIG, score_override=0.9
        )
        assert result.obstacle_density > BASE_CONFIG.obstacle_density

    def test_unknown_target_weakness_raises(self):
        weakness = _weakness("obstacle_avoidance")
        with pytest.raises(ValueError):
            EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG, target_weakness="not_a_real_dimension")

    def test_grid_size_never_touched_for_non_exploration_weaknesses(self):
        for dim in WEAKNESS_DIMENSIONS:
            if dim == "exploration":
                continue
            weakness = _weakness(dim)
            result = EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG)
            assert result.grid_size == BASE_CONFIG.grid_size, f"grid_size changed for {dim}"


class TestSynthesizedConfigsAreValid:
    def test_all_weakness_targets_produce_valid_environments(self):
        generator = GridWorldGenerator()
        for dim in WEAKNESS_DIMENSIONS:
            weakness = _weakness(dim)
            config = EnvironmentSynthesizer(seed=1).synthesize(weakness, BASE_CONFIG)
            report = generator.generate(config)
            assert report.validation.is_valid, f"invalid environment for target {dim}"
