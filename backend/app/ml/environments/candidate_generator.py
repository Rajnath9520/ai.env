"""Phase 10 — generate several candidate environments per curriculum step
instead of committing to one (Section 17).

Problem solved: a single targeted environment (Phase 9) commits to one
exact parameter setting. Generating several candidates — varying the
targeted severity and, when secondary weaknesses exist, which weakness is
targeted — gives the curriculum controller (Phase 12) something to actually
choose between, and gives the environment-generator UI panel real options
to display (Section 31).

Approach: still entirely rule-based, per Section 17's explicit instruction
not to reach for a neural generator yet. Candidates are produced by calling
`EnvironmentSynthesizer.synthesize` multiple times: targets cycle through
[primary_weakness, *secondary_weaknesses] (repeating if there are fewer
distinct weaknesses than requested candidates), and each call jitters the
severity score by a small random factor so candidates aren't identical
even when they share a target.

Scoring:
- `difficulty`: same `GridWorldConfig.difficulty_estimate()` heuristic used
  everywhere else, for comparability.
- `validity_score`: 1.0/0.0 from actually running the candidate through the
  real generator+validator (Phase 1) — not assumed, checked.
- `diversity_score`: normalized Euclidean distance, in a 5-dimensional
  parameter space (obstacle_density, dead_end_probability, path_complexity,
  goal_distance/grid_size, and normalized grid_size), averaged against a
  reference set of prior environments (and prior candidates in the same
  batch, so a batch doesn't generate five near-identical candidates). An
  empty reference set yields diversity_score=1.0 (nothing to compare
  against yet, not "not diverse").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.analysis.weakness_vector import WeaknessVector
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator
from app.ml.environments.synthesizer import MAX_GRID_SIZE, MIN_GRID_SIZE, EnvironmentSynthesizer

SEVERITY_JITTER_RANGE: tuple[float, float] = (0.75, 1.15)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class EnvironmentCandidate:
    candidate_id: str
    config: GridWorldConfig
    target_weakness: str
    difficulty: float
    validity_score: float
    diversity_score: float

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "config": self.config.to_dict(),
            "target_weakness": self.target_weakness,
            "difficulty": round(self.difficulty, 4),
            "validity_score": self.validity_score,
            "diversity_score": round(self.diversity_score, 4),
        }


class CandidateGenerator:
    def __init__(
        self,
        synthesizer: EnvironmentSynthesizer | None = None,
        generator: GridWorldGenerator | None = None,
        seed: int | None = None,
    ) -> None:
        self._synthesizer = synthesizer or EnvironmentSynthesizer(seed=seed)
        self._generator = generator or GridWorldGenerator()
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        weakness: WeaknessVector,
        base_config: GridWorldConfig,
        previous_configs: list[GridWorldConfig] | None = None,
        n_candidates: int = 5,
    ) -> list[EnvironmentCandidate]:
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")

        reference_configs = list(previous_configs or [])
        targets = self._select_targets(weakness, n_candidates)
        candidates: list[EnvironmentCandidate] = []

        for i, target in enumerate(targets):
            base_score = getattr(weakness, target)
            jitter = float(self._rng.uniform(*SEVERITY_JITTER_RANGE))
            score = _clamp(base_score * jitter)

            config = self._synthesizer.synthesize(
                weakness, base_config, target_weakness=target, score_override=score
            )
            report = self._generator.generate(config)
            diversity = self._diversity_score(config, reference_configs)

            candidates.append(
                EnvironmentCandidate(
                    candidate_id=f"cand_{i:02d}",
                    config=config,
                    target_weakness=target,
                    difficulty=config.difficulty_estimate(),
                    validity_score=1.0 if report.validation.is_valid else 0.0,
                    diversity_score=diversity,
                )
            )
            reference_configs.append(config)

        return candidates

    # -- helpers --------------------------------------------------------------

    def _select_targets(self, weakness: WeaknessVector, n_candidates: int) -> list[str]:
        targets = [weakness.primary_weakness, *weakness.secondary_weaknesses]
        if not targets:
            targets = ["obstacle_avoidance"]  # degenerate fallback; should not normally occur
        return [targets[i % len(targets)] for i in range(n_candidates)]

    def _diversity_score(
        self, config: GridWorldConfig, reference_configs: list[GridWorldConfig]
    ) -> float:
        if not reference_configs:
            return 1.0
        vector = self._param_vector(config)
        distances = [self._distance(vector, self._param_vector(c)) for c in reference_configs]
        avg_distance = sum(distances) / len(distances)
        max_possible_distance = len(vector) ** 0.5  # each dim differs by at most 1.0
        return _clamp(avg_distance / max_possible_distance)

    @staticmethod
    def _param_vector(config: GridWorldConfig) -> list[float]:
        return [
            config.obstacle_density,
            config.dead_end_probability,
            config.path_complexity,
            (config.goal_distance or 0) / config.grid_size,
            (config.grid_size - MIN_GRID_SIZE) / (MAX_GRID_SIZE - MIN_GRID_SIZE),
        ]

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
