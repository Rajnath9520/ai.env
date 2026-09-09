import pytest

from app.ml.analysis.weakness_vector import WEAKNESS_DIMENSIONS, WeaknessVector
from app.ml.curriculum.controller import CurriculumController
from app.ml.environments.candidate_generator import EnvironmentCandidate
from app.ml.environments.config import GridWorldConfig
from app.ml.evaluation.learning_progress import LearningProgressRecord


def _weakness(**overrides) -> WeaknessVector:
    values = {dim: 0.0 for dim in WEAKNESS_DIMENSIONS}
    values.update(overrides)
    primary = max(values, key=values.get)
    return WeaknessVector(**values, primary_weakness=primary, secondary_weaknesses=(), episodes_analyzed=5)


def _candidate(target_weakness, difficulty, diversity, candidate_id="c0"):
    return EnvironmentCandidate(
        candidate_id=candidate_id,
        config=GridWorldConfig(seed=1),
        target_weakness=target_weakness,
        difficulty=difficulty,
        validity_score=1.0,
        diversity_score=diversity,
    )


class TestWeaknessAlignment:
    def test_alignment_equals_weakness_vector_value_for_target(self):
        weakness = _weakness(obstacle_avoidance=0.7)
        candidate = _candidate("obstacle_avoidance", 0.3, 0.5)
        controller = CurriculumController()
        scored = controller.score(candidate, weakness)
        assert scored.weakness_alignment == pytest.approx(0.7)


class TestExpectedLearningProgress:
    def test_no_history_uses_default(self):
        controller = CurriculumController(default_expected_progress=0.42)
        weakness = _weakness(exploration=0.5)
        candidate = _candidate("exploration", 0.3, 0.5)
        scored = controller.score(candidate, weakness, lp_history=[])
        assert scored.expected_learning_progress == pytest.approx(0.42)

    def test_averages_matching_history_only(self):
        controller = CurriculumController()
        weakness = _weakness(exploration=0.5)
        candidate = _candidate("exploration", 0.3, 0.5)
        history = [
            LearningProgressRecord("e1", "exploration", "success_rate", 0.1, 0.3, 0.2),
            LearningProgressRecord("e2", "exploration", "success_rate", 0.2, 0.4, 0.2),
            LearningProgressRecord("e3", "obstacle_avoidance", "success_rate", 0.1, 0.9, 0.8),
        ]
        scored = controller.score(candidate, weakness, lp_history=history)
        assert scored.expected_learning_progress == pytest.approx(0.2)

    def test_negative_history_clamped_to_zero(self):
        controller = CurriculumController()
        weakness = _weakness(exploration=0.5)
        candidate = _candidate("exploration", 0.3, 0.5)
        history = [LearningProgressRecord("e1", "exploration", "success_rate", 0.6, 0.2, -0.4)]
        scored = controller.score(candidate, weakness, lp_history=history)
        assert scored.expected_learning_progress == pytest.approx(0.0)


class TestExcessiveDifficultyPenalty:
    def test_no_penalty_below_ceiling(self):
        controller = CurriculumController()
        weakness = _weakness(exploration=0.5)
        candidate = _candidate("exploration", 0.4, 0.5)
        scored = controller.score(candidate, weakness, recent_success_rate=0.8)
        assert scored.excessive_difficulty_penalty == 0.0

    def test_penalty_grows_above_ceiling_when_agent_struggling(self):
        controller = CurriculumController()
        weakness = _weakness(exploration=0.5)
        candidate = _candidate("exploration", 0.9, 0.5)
        scored = controller.score(candidate, weakness, recent_success_rate=0.0)
        assert scored.excessive_difficulty_penalty > 0.5

    def test_penalty_capped_at_one(self):
        controller = CurriculumController()
        weakness = _weakness(exploration=0.5)
        candidate = _candidate("exploration", 1.0, 0.5)
        scored = controller.score(candidate, weakness, recent_success_rate=0.0)
        assert scored.excessive_difficulty_penalty <= 1.0

    def test_strong_agent_tolerates_higher_difficulty(self):
        controller = CurriculumController()
        weakness = _weakness(exploration=0.5)
        candidate = _candidate("exploration", 0.75, 0.5)
        struggling = controller.score(candidate, weakness, recent_success_rate=0.0)
        strong = controller.score(candidate, weakness, recent_success_rate=0.9)
        assert struggling.excessive_difficulty_penalty > strong.excessive_difficulty_penalty


class TestScoreWeighting:
    def test_alpha_only_isolates_alignment(self):
        controller = CurriculumController(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
        weakness = _weakness(dead_end_recovery=0.6)
        candidate = _candidate("dead_end_recovery", 0.9, 0.1)
        scored = controller.score(candidate, weakness, recent_success_rate=0.9)
        assert scored.total_score == pytest.approx(0.6)

    def test_gamma_only_isolates_diversity(self):
        controller = CurriculumController(alpha=0.0, beta=0.0, gamma=1.0, delta=0.0)
        weakness = _weakness(dead_end_recovery=0.6)
        candidate = _candidate("dead_end_recovery", 0.9, 0.77)
        scored = controller.score(candidate, weakness)
        assert scored.total_score == pytest.approx(0.77)

    def test_delta_only_isolates_negative_difficulty_penalty(self):
        controller = CurriculumController(alpha=0.0, beta=0.0, gamma=0.0, delta=1.0)
        weakness = _weakness(dead_end_recovery=0.6)
        candidate = _candidate("dead_end_recovery", 1.0, 0.5)
        scored = controller.score(candidate, weakness, recent_success_rate=0.0)
        assert scored.total_score == pytest.approx(-scored.excessive_difficulty_penalty)
        assert scored.total_score < 0


class TestSelect:
    def test_ranks_best_first(self):
        controller = CurriculumController()
        weakness = _weakness(dead_end_recovery=0.9, obstacle_avoidance=0.1)
        good_candidate = _candidate("dead_end_recovery", 0.4, 0.8, "good")
        bad_candidate = _candidate("obstacle_avoidance", 0.95, 0.1, "bad")
        ranked = controller.select([bad_candidate, good_candidate], weakness, recent_success_rate=0.9)
        assert ranked[0].candidate.candidate_id == "good"
        assert ranked[0].total_score > ranked[1].total_score

    def test_select_returns_every_candidate_scored(self):
        controller = CurriculumController()
        weakness = _weakness(dead_end_recovery=0.5)
        candidates = [_candidate("dead_end_recovery", 0.3, 0.5, f"c{i}") for i in range(4)]
        ranked = controller.select(candidates, weakness)
        assert len(ranked) == 4
        assert {r.candidate.candidate_id for r in ranked} == {"c0", "c1", "c2", "c3"}

    def test_to_dict_includes_all_components(self):
        controller = CurriculumController()
        weakness = _weakness(dead_end_recovery=0.5)
        candidate = _candidate("dead_end_recovery", 0.3, 0.5)
        scored = controller.score(candidate, weakness)
        d = scored.to_dict()
        for key in (
            "candidate",
            "weakness_alignment",
            "expected_learning_progress",
            "diversity",
            "excessive_difficulty_penalty",
            "total_score",
        ):
            assert key in d
