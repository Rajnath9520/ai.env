import pytest

from app.ml.analysis.weakness_vector import WEAKNESS_DIMENSIONS
from app.ml.curriculum.ablation import ABLATION_VARIANTS, build_ablation_runner
from app.ml.curriculum.failure_driven import _blind_weakness_vector
from app.ml.environments.config import GridWorldConfig
from app.ml.training.trainer import AgentTrainer

TINY_BASE_CONFIG = GridWorldConfig(
    grid_size=5, obstacle_density=0.1, dead_end_probability=0.1, path_complexity=0.1, max_steps=12, seed=1
)


class TestBlindWeaknessVector:
    def test_all_dimensions_equal(self):
        vector = _blind_weakness_vector()
        values = {getattr(vector, dim) for dim in WEAKNESS_DIMENSIONS}
        assert values == {0.5}

    def test_secondary_weaknesses_cover_remaining_dimensions(self):
        vector = _blind_weakness_vector()
        assert set(vector.secondary_weaknesses) == set(WEAKNESS_DIMENSIONS) - {vector.primary_weakness}

    def test_carries_episodes_analyzed_through(self):
        vector = _blind_weakness_vector(episodes_analyzed=7)
        assert vector.episodes_analyzed == 7


class TestAblationVariantDefinitions:
    def test_four_variants_defined(self):
        assert set(ABLATION_VARIANTS.keys()) == {
            "A_failure_analysis_only",
            "B_learning_progress_only",
            "C_failure_analysis_plus_lp",
            "D_full_method",
        }

    def test_variant_a_uses_failure_analysis_and_ignores_lp_and_diversity(self):
        v = ABLATION_VARIANTS["A_failure_analysis_only"]
        assert v.use_failure_analysis is True
        assert v.alpha > 0
        assert v.beta == 0
        assert v.gamma == 0

    def test_variant_b_blinds_failure_analysis_and_uses_lp(self):
        v = ABLATION_VARIANTS["B_learning_progress_only"]
        assert v.use_failure_analysis is False
        assert v.alpha == 0
        assert v.beta > 0
        assert v.gamma == 0

    def test_variant_c_uses_both_but_not_diversity(self):
        v = ABLATION_VARIANTS["C_failure_analysis_plus_lp"]
        assert v.use_failure_analysis is True
        assert v.alpha > 0
        assert v.beta > 0
        assert v.gamma == 0

    def test_variant_d_uses_all_three_components(self):
        v = ABLATION_VARIANTS["D_full_method"]
        assert v.use_failure_analysis is True
        assert v.alpha > 0
        assert v.beta > 0
        assert v.gamma > 0


class TestBuildAblationRunner:
    def test_rejects_unknown_variant(self):
        trainer = AgentTrainer(seed=1)
        with pytest.raises(ValueError):
            build_ablation_runner("not_a_real_variant", trainer, TINY_BASE_CONFIG)

    def test_builds_runner_with_correct_controller_weights(self):
        trainer = AgentTrainer(seed=1)
        runner = build_ablation_runner("D_full_method", trainer, TINY_BASE_CONFIG)
        assert runner.controller.alpha == pytest.approx(0.35)
        assert runner.controller.beta == pytest.approx(0.35)
        assert runner.controller.gamma == pytest.approx(0.15)
        assert runner.use_failure_analysis is True

    def test_builds_runner_with_blinded_targeting_for_variant_b(self):
        trainer = AgentTrainer(seed=1)
        runner = build_ablation_runner("B_learning_progress_only", trainer, TINY_BASE_CONFIG)
        assert runner.use_failure_analysis is False
        assert runner.controller.alpha == 0.0

    def test_runner_kwargs_pass_through(self):
        trainer = AgentTrainer(seed=1)
        runner = build_ablation_runner(
            "A_failure_analysis_only", trainer, TINY_BASE_CONFIG, timesteps_per_step=999, eval_episodes=7
        )
        assert runner.timesteps_per_step == 999
        assert runner.eval_episodes == 7


class TestBlindedTargetingIntegration:
    def test_variant_b_still_records_real_weakness_history(self):
        """Even with targeting blinded, the real weakness vector (from
        actual failure analysis) must still be computed and recorded --
        ablation B blinds the *decision*, not the *measurement*."""
        trainer = AgentTrainer(seed=1)
        runner = build_ablation_runner(
            "B_learning_progress_only",
            trainer,
            TINY_BASE_CONFIG,
            timesteps_per_step=256,
            eval_episodes=3,
            n_candidates=2,
            trajectories_per_analysis=2,
        )
        result = runner.run(n_steps=2)
        assert len(result.weakness_history) == 2
        # weakness_history reflects genuine analysis output, not the
        # fixed 0.5-everywhere blind vector used only for targeting.
        first_weakness = result.weakness_history[0]
        values = [getattr(first_weakness, dim) for dim in ["obstacle_avoidance", "exploration", "dead_end_recovery"]]
        assert not all(v == 0.5 for v in values)

