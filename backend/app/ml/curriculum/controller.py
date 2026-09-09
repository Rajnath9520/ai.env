"""Phase 12 — Environment Selection (Section 19).

Score(E) = alpha * WeaknessAlignment(E)
         + beta  * ExpectedLearningProgress(E)
         + gamma * Diversity(E)
         - delta * ExcessiveDifficulty(E)

Problem solved: Phase 10 produces several valid, individually-reasonable
candidates. Something has to pick one. This module turns "reasonable" into
a single auditable score per candidate, with every component stored (not
just the winner), so a later UI or report can show *why* an environment was
chosen, not just that it was.

Term definitions (each documented since Section 19 explicitly says "do not
pretend the weights are optimal" — the same caution applies to the term
definitions themselves, which are reasonable choices, not derived facts):

- WeaknessAlignment(E): the WeaknessVector's score for whatever weakness E
  targets. A candidate targeting the agent's most severe weakness scores
  higher than one targeting a mild secondary weakness.
- ExpectedLearningProgress(E): the average *observed* learning progress
  from past environments that targeted the same weakness, from the
  learning-progress history (Phase 11). With no matching history yet, a
  configurable neutral default is used — explicitly not claimed to be a
  real prediction, just a reasonable prior before evidence exists.
- Diversity(E): the candidate's diversity_score from Phase 10, unchanged.
- ExcessiveDifficulty(E): a penalty that grows once a candidate's
  difficulty exceeds an allowed ceiling derived from the agent's recent
  success rate (a struggling agent's ceiling is lower) — meant to stop the
  controller from picking a well-aligned but currently-unproductive
  environment (e.g. maximally hard) just because its alignment score is
  high.

Limitations: alignment, expected-progress-with-no-history, and the
difficulty ceiling formula are all hand-specified heuristics, not learned
or empirically validated — flagged in each helper's docstring rather than
only here.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.analysis.weakness_vector import WeaknessVector
from app.ml.environments.candidate_generator import EnvironmentCandidate
from app.ml.evaluation.learning_progress import LearningProgressRecord


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: EnvironmentCandidate
    weakness_alignment: float
    expected_learning_progress: float
    diversity: float
    excessive_difficulty_penalty: float
    total_score: float

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "weakness_alignment": round(self.weakness_alignment, 4),
            "expected_learning_progress": round(self.expected_learning_progress, 4),
            "diversity": round(self.diversity, 4),
            "excessive_difficulty_penalty": round(self.excessive_difficulty_penalty, 4),
            "total_score": round(self.total_score, 4),
        }


class CurriculumController:
    def __init__(
        self,
        alpha: float = 0.35,
        beta: float = 0.35,
        gamma: float = 0.15,
        delta: float = 0.15,
        default_expected_progress: float = 0.3,
        default_recent_success_rate: float = 0.5,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.default_expected_progress = default_expected_progress
        self.default_recent_success_rate = default_recent_success_rate

    def score(
        self,
        candidate: EnvironmentCandidate,
        weakness: WeaknessVector,
        lp_history: list[LearningProgressRecord] = (),
        recent_success_rate: float | None = None,
    ) -> ScoredCandidate:
        alignment = getattr(weakness, candidate.target_weakness)
        expected_lp = self._expected_learning_progress(candidate, lp_history)
        diversity = candidate.diversity_score
        penalty = self._excessive_difficulty(
            candidate.difficulty,
            recent_success_rate if recent_success_rate is not None else self.default_recent_success_rate,
        )
        total = (
            self.alpha * alignment
            + self.beta * expected_lp
            + self.gamma * diversity
            - self.delta * penalty
        )
        return ScoredCandidate(
            candidate=candidate,
            weakness_alignment=alignment,
            expected_learning_progress=expected_lp,
            diversity=diversity,
            excessive_difficulty_penalty=penalty,
            total_score=total,
        )

    def select(
        self,
        candidates: list[EnvironmentCandidate],
        weakness: WeaknessVector,
        lp_history: list[LearningProgressRecord] = (),
        recent_success_rate: float | None = None,
    ) -> list[ScoredCandidate]:
        """Returns every candidate scored, ranked best-first — callers pick
        `result[0]` for the winner but keep the rest for transparency (the
        UI's candidate list, Section 31)."""
        scored = [
            self.score(c, weakness, lp_history, recent_success_rate) for c in candidates
        ]
        return sorted(scored, key=lambda s: s.total_score, reverse=True)

    # -- term helpers -----------------------------------------------------

    def _expected_learning_progress(
        self, candidate: EnvironmentCandidate, lp_history: list[LearningProgressRecord]
    ) -> float:
        """Average observed LP from past environments targeting the same
        weakness; falls back to `default_expected_progress` with no
        matching history. Negative historical LP is clamped to 0 for
        scoring purposes only (a weakness that made things worse shouldn't
        score *below* "no evidence," it should just score low) — the raw,
        unclamped value is still visible in stored LearningProgressRecords.
        """
        matching = [r.learning_progress for r in lp_history if r.weakness_target == candidate.target_weakness]
        if not matching:
            return self.default_expected_progress
        average = sum(matching) / len(matching)
        return _clamp(average)

    def _excessive_difficulty(self, difficulty: float, recent_success_rate: float) -> float:
        """Difficulty ceiling scales with recent success rate: a struggling
        agent (low success) has a low ceiling, so even moderately hard
        candidates get penalized; a strong agent's ceiling is high, so only
        near-maximum difficulty gets penalized."""
        ceiling = 0.3 + 0.7 * _clamp(recent_success_rate)
        if difficulty <= ceiling:
            return 0.0
        span = max(1e-6, 1.0 - ceiling)
        return _clamp((difficulty - ceiling) / span)
