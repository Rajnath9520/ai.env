import pytest

from app.ml.analysis.weakness_vector import WEAKNESS_DIMENSIONS, WeaknessVector
from app.ml.environments.candidate_generator import CandidateGenerator
from app.ml.environments.config import GridWorldConfig

BASE_CONFIG = GridWorldConfig(
    grid_size=10,
    obstacle_density=0.2,
    goal_distance=4,
    dead_end_probability=0.2,
    path_complexity=0.2,
    max_steps=60,
    seed=1,
)


def _weakness(primary: str, secondary: tuple = (), score: float = 0.8) -> WeaknessVector:
    values = {dim: 0.0 for dim in WEAKNESS_DIMENSIONS}
    values[primary] = score
    for dim in secondary:
        values[dim] = 0.5
    return WeaknessVector(
        **values, primary_weakness=primary, secondary_weaknesses=secondary, episodes_analyzed=10
    )


class TestCandidateGenerationBasics:
    def test_generates_requested_number_of_candidates(self):
        weakness = _weakness("dead_end_recovery")
        candidates = CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=5)
        assert len(candidates) == 5

    def test_rejects_non_positive_n_candidates(self):
        weakness = _weakness("dead_end_recovery")
        with pytest.raises(ValueError):
            CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=0)

    def test_every_candidate_has_required_fields(self):
        weakness = _weakness("obstacle_avoidance")
        candidates = CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=3)
        for c in candidates:
            assert c.candidate_id
            assert c.target_weakness in WEAKNESS_DIMENSIONS
            assert 0.0 <= c.difficulty <= 1.0
            assert c.validity_score in (0.0, 1.0)
            assert 0.0 <= c.diversity_score <= 1.0

    def test_all_candidates_are_valid_environments(self):
        weakness = _weakness("long_horizon_planning")
        candidates = CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=5)
        assert all(c.validity_score == 1.0 for c in candidates)


class TestTargetSelection:
    def test_targets_cycle_through_primary_and_secondary(self):
        weakness = _weakness(
            "dead_end_recovery", secondary=("obstacle_avoidance", "exploration")
        )
        candidates = CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=6)
        targets = [c.target_weakness for c in candidates]
        assert targets == [
            "dead_end_recovery",
            "obstacle_avoidance",
            "exploration",
            "dead_end_recovery",
            "obstacle_avoidance",
            "exploration",
        ]

    def test_no_secondary_weaknesses_targets_primary_every_time(self):
        weakness = _weakness("path_efficiency", secondary=())
        candidates = CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=4)
        assert all(c.target_weakness == "path_efficiency" for c in candidates)


class TestDiversityScoring:
    def test_single_candidate_with_no_reference_gets_full_diversity(self):
        weakness = _weakness("obstacle_avoidance")
        candidates = CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=1)
        assert candidates[0].diversity_score == 1.0

    def test_diversity_trends_down_as_similar_candidates_accumulate(self):
        weakness = _weakness("path_efficiency")  # single target -> similar candidates
        candidates = CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=5)
        assert candidates[-1].diversity_score <= candidates[0].diversity_score

    def test_previous_configs_reduce_diversity_of_similar_candidates(self):
        weakness = _weakness("obstacle_avoidance")
        candidates_no_ref = CandidateGenerator(seed=1).generate(
            weakness, BASE_CONFIG, previous_configs=[], n_candidates=1
        )
        candidates_with_ref = CandidateGenerator(seed=1).generate(
            weakness, BASE_CONFIG, previous_configs=[BASE_CONFIG] * 5, n_candidates=1
        )
        assert candidates_with_ref[0].diversity_score <= candidates_no_ref[0].diversity_score


class TestDifficultyVariation:
    def test_severity_jitter_produces_varying_difficulty_within_batch(self):
        weakness = _weakness("obstacle_avoidance")
        candidates = CandidateGenerator(seed=1).generate(weakness, BASE_CONFIG, n_candidates=5)
        difficulties = {c.difficulty for c in candidates}
        assert len(difficulties) > 1
