import pytest

from app.ml.curriculum.base import CurriculumStep
from app.ml.curriculum.generic_adaptive import GenericAdaptiveCurriculum
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator


def _fake_step(success_rate: float, i: int = 0) -> CurriculumStep:
    return CurriculumStep(
        step_index=i,
        environment_id=f"e{i}",
        config=GridWorldConfig(seed=1),
        difficulty=0.1,
        success_rate=success_rate,
        average_reward=0.0,
    )


class TestGenericAdaptiveCurriculumConstruction:
    def test_rejects_invalid_initial_difficulty(self):
        with pytest.raises(ValueError):
            GenericAdaptiveCurriculum(initial_difficulty=1.5)

    def test_rejects_decrease_threshold_above_increase_threshold(self):
        with pytest.raises(ValueError):
            GenericAdaptiveCurriculum(increase_threshold=0.4, decrease_threshold=0.5)


class TestDifficultyAdaptation:
    def test_high_success_rate_increases_difficulty(self):
        curriculum = GenericAdaptiveCurriculum(initial_difficulty=0.2, difficulty_step=0.1, seed=1)
        history = [_fake_step(success_rate=0.95, i=0)]
        assert curriculum.current_difficulty(history) == pytest.approx(0.3)

    def test_low_success_rate_decreases_difficulty(self):
        curriculum = GenericAdaptiveCurriculum(initial_difficulty=0.5, difficulty_step=0.1, seed=1)
        history = [_fake_step(success_rate=0.1, i=0)]
        assert curriculum.current_difficulty(history) == pytest.approx(0.4)

    def test_middle_success_rate_maintains_difficulty(self):
        curriculum = GenericAdaptiveCurriculum(initial_difficulty=0.5, difficulty_step=0.1, seed=1)
        history = [_fake_step(success_rate=0.6, i=0)]
        assert curriculum.current_difficulty(history) == pytest.approx(0.5)

    def test_difficulty_saturates_at_one(self):
        curriculum = GenericAdaptiveCurriculum(initial_difficulty=0.9, difficulty_step=0.2, seed=1)
        history = [_fake_step(success_rate=0.95, i=i) for i in range(5)]
        assert curriculum.current_difficulty(history) == pytest.approx(1.0)

    def test_difficulty_floors_at_zero(self):
        curriculum = GenericAdaptiveCurriculum(initial_difficulty=0.1, difficulty_step=0.2, seed=1)
        history = [_fake_step(success_rate=0.0, i=i) for i in range(5)]
        assert curriculum.current_difficulty(history) == pytest.approx(0.0)

    def test_sequence_of_mixed_signals_matches_manual_trace(self):
        curriculum = GenericAdaptiveCurriculum(initial_difficulty=0.3, difficulty_step=0.1, seed=1)
        # 0.3 -> (high) 0.4 -> (mid, maintain) 0.4 -> (low) 0.3 -> (high) 0.4
        history = [
            _fake_step(success_rate=0.9, i=0),
            _fake_step(success_rate=0.6, i=1),
            _fake_step(success_rate=0.2, i=2),
            _fake_step(success_rate=0.9, i=3),
        ]
        assert curriculum.current_difficulty(history) == pytest.approx(0.4)

    def test_next_environment_is_pure_function_of_history(self):
        curriculum = GenericAdaptiveCurriculum(initial_difficulty=0.3, difficulty_step=0.1, seed=1)
        history = [_fake_step(success_rate=0.9, i=0)]
        difficulty_a = curriculum.current_difficulty(history)
        difficulty_b = curriculum.current_difficulty(history)
        assert difficulty_a == difficulty_b


class TestGenericAdaptiveEnvironments:
    def test_generated_configs_are_valid(self):
        curriculum = GenericAdaptiveCurriculum(grid_size=8, seed=2)
        generator = GridWorldGenerator()
        history: list[CurriculumStep] = []
        for i in range(6):
            config = curriculum.next_environment(history)
            report = generator.generate(config)
            assert report.validation.is_valid
            history.append(_fake_step(success_rate=0.9, i=i))  # keep escalating

    def test_higher_difficulty_yields_higher_obstacle_density(self):
        curriculum = GenericAdaptiveCurriculum(grid_size=8, initial_difficulty=0.1, seed=2)
        low_config = curriculum.next_environment([])
        escalating_history = [_fake_step(success_rate=0.95, i=i) for i in range(5)]
        high_config = curriculum.next_environment(escalating_history)
        assert high_config.obstacle_density > low_config.obstacle_density
