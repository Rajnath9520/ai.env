import pytest

from app.ml.curriculum.base import CurriculumStep
from app.ml.curriculum.fixed_curriculum import STAGE_ORDER, STAGE_PRESETS, FixedCurriculum
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator


class TestFixedCurriculumStages:
    def test_rejects_too_few_total_steps(self):
        with pytest.raises(ValueError):
            FixedCurriculum(total_steps=2)

    def test_stage_progression_covers_all_four_stages(self):
        curriculum = FixedCurriculum(total_steps=20, seed=1)
        stages_seen = {curriculum.stage_for_step(i) for i in range(20)}
        assert stages_seen == set(STAGE_ORDER)

    def test_first_step_is_easy_last_step_is_very_hard(self):
        curriculum = FixedCurriculum(total_steps=20, seed=1)
        assert curriculum.stage_for_step(0) == "easy"
        assert curriculum.stage_for_step(19) == "very_hard"

    def test_stage_is_monotonically_non_decreasing_with_step_index(self):
        curriculum = FixedCurriculum(total_steps=20, seed=1)
        stage_indices = [STAGE_ORDER.index(curriculum.stage_for_step(i)) for i in range(20)]
        assert stage_indices == sorted(stage_indices)

    def test_stage_depends_only_on_history_length_not_performance(self):
        curriculum = FixedCurriculum(total_steps=20, seed=1)
        good_history = [
            CurriculumStep(
                step_index=i,
                environment_id=f"e{i}",
                config=GridWorldConfig(seed=1),
                difficulty=0.1,
                success_rate=0.99,  # agent doing great
                average_reward=1.0,
            )
            for i in range(5)
        ]
        bad_history = [
            CurriculumStep(
                step_index=i,
                environment_id=f"e{i}",
                config=GridWorldConfig(seed=1),
                difficulty=0.1,
                success_rate=0.0,  # agent failing constantly
                average_reward=-1.0,
            )
            for i in range(5)
        ]
        assert curriculum.next_environment(good_history).to_dict()["grid_size"] == curriculum.next_environment(
            bad_history
        ).to_dict()["grid_size"]
        stage_after_good = curriculum.stage_for_step(len(good_history))
        stage_after_bad = curriculum.stage_for_step(len(bad_history))
        assert stage_after_good == stage_after_bad == "moderate"


class TestFixedCurriculumEnvironments:
    def test_generated_configs_are_valid(self):
        curriculum = FixedCurriculum(total_steps=8, grid_size=8, seed=2)
        generator = GridWorldGenerator()
        history: list[CurriculumStep] = []
        for i in range(8):
            config = curriculum.next_environment(history)
            report = generator.generate(config)
            assert report.validation.is_valid
            history.append(
                CurriculumStep(
                    step_index=i,
                    environment_id=f"e{i}",
                    config=config,
                    difficulty=config.difficulty_estimate(),
                    success_rate=0.5,
                    average_reward=0.0,
                )
            )

    def test_difficulty_increases_across_stages_on_average(self):
        curriculum = FixedCurriculum(total_steps=40, grid_size=10, seed=3)
        history: list[CurriculumStep] = []
        difficulties_by_stage: dict[str, list[float]] = {s: [] for s in STAGE_ORDER}

        for i in range(40):
            config = curriculum.next_environment(history)
            stage = curriculum.stage_for_step(i)
            difficulties_by_stage[stage].append(config.difficulty_estimate())
            history.append(
                CurriculumStep(
                    step_index=i,
                    environment_id=f"e{i}",
                    config=config,
                    difficulty=config.difficulty_estimate(),
                    success_rate=0.5,
                    average_reward=0.0,
                )
            )

        avg_by_stage = [
            sum(difficulties_by_stage[s]) / len(difficulties_by_stage[s]) for s in STAGE_ORDER
        ]
        assert avg_by_stage == sorted(avg_by_stage)

    def test_goal_distance_scales_with_stage(self):
        curriculum = FixedCurriculum(total_steps=8, grid_size=10, seed=4)
        easy_config = curriculum.next_environment([])  # step 0 -> easy
        history = [
            CurriculumStep(
                step_index=i,
                environment_id=f"e{i}",
                config=GridWorldConfig(seed=1),
                difficulty=0.1,
                success_rate=0.5,
                average_reward=0.0,
            )
            for i in range(7)
        ]
        very_hard_config = curriculum.next_environment(history)  # step 7 -> very_hard
        assert very_hard_config.goal_distance > easy_config.goal_distance
