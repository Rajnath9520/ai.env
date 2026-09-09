"""Phase 9 — the core proposed method (Section 16).

Problem solved: Baselines A-C (Phases 6-8) either ignore performance
entirely or react only to *aggregate* success rate. Neither can tell "the
agent is failing because of dead ends" apart from "failing because of
obstacles," so neither can generate an environment that specifically
exercises the weak behavior. `EnvironmentSynthesizer` closes that gap by
reading a `WeaknessVector` (Phase 5) and adjusting only the environment
parameters causally linked to the targeted weakness, per Section 16's
IF/THEN mapping — explicitly NOT increasing every parameter uniformly.

Why this approach: the mapping is small, hand-specifiable, and auditable —
each weakness dimension maps to a documented subset of parameters with
documented relative weights (`WEAKNESS_TARGET_PARAMS` below), so it's easy
to verify empirically that (e.g.) targeting `obstacle_avoidance` moves
obstacle_density and *only* obstacle_density, leaving dead-end structure
and grid size untouched — exactly the property the ablation studies
(Phase 22) will need to isolate.

Alternative approaches considered: a learned mapping (e.g. a small model
predicting parameter deltas from the weakness vector) was rejected for a
first version — it would need training data (weakness vector, resulting
learning progress) that doesn't exist yet, and would make the "is this
actually targeted, or just generically harder" question impossible to
audit directly, which Section 16 explicitly warns against.

Limitations: the mapping weights are hand-specified per the module's
documented rationale, not learned or empirically calibrated; grid_size is
only ever adjusted for the `exploration` weakness (all other weaknesses
keep grid_size fixed) to avoid every synthesized environment quietly
becoming "the same thing but bigger."
"""

from __future__ import annotations

import numpy as np

from app.ml.analysis.weakness_vector import WEAKNESS_DIMENSIONS, WeaknessVector
from app.ml.environments.config import GridWorldConfig

# Maximum possible delta applied to each parameter when a weakness's weight
# for that parameter is 1.0 and the weakness score is 1.0 (maximum
# confidence/severity). Actual delta = weight * MAX_DELTAS[param] * score.
MAX_DELTAS: dict[str, float] = {
    "obstacle_density": 0.30,
    "dead_end_probability": 0.35,
    "path_complexity": 0.35,
    "goal_distance_fraction": 0.30,
    "grid_size": 6,
}

MIN_GRID_SIZE = 4
MAX_GRID_SIZE = 24

# Section 16's IF/THEN table, translated into parameter weights in [0, 1].
# A weakness only touches the parameters listed here -- everything else is
# left exactly as it was in the base config, which is what makes this
# "targeted" rather than "generically harder."
WEAKNESS_TARGET_PARAMS: dict[str, dict[str, float]] = {
    "long_horizon_planning": {
        # "increase path length, decision points, meaningful branches, dead
        # ends, deceptive routes"
        "goal_distance_fraction": 0.7,
        "path_complexity": 0.6,
        "dead_end_probability": 0.4,
    },
    "obstacle_avoidance": {
        # "increase obstacle density, narrow corridors, constrained movement"
        "obstacle_density": 1.0,
    },
    "dead_end_recovery": {
        # "increase dead-end structures, recoverable traps, branch complexity"
        "dead_end_probability": 0.8,
        "path_complexity": 0.3,
    },
    "exploration": {
        # "increase environment size, exploration space, delayed goal visibility"
        "grid_size": 1.0,
        "goal_distance_fraction": 0.6,
    },
    "path_efficiency": {
        # "create multiple valid routes, different route lengths, meaningful shortcuts"
        "path_complexity": 1.0,
    },
    "short_horizon_planning": {
        # local/immediate decision difficulty: more nearby obstacles and
        # branch points force more frequent close-range decisions.
        "obstacle_density": 0.5,
        "path_complexity": 0.3,
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class EnvironmentSynthesizer:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def synthesize(
        self,
        weakness: WeaknessVector,
        base_config: GridWorldConfig,
        target_weakness: str | None = None,
        score_override: float | None = None,
    ) -> GridWorldConfig:
        target = target_weakness or weakness.primary_weakness
        if target not in WEAKNESS_TARGET_PARAMS:
            raise ValueError(
                f"Unknown weakness dimension {target!r}; expected one of {WEAKNESS_DIMENSIONS}"
            )
        score = score_override if score_override is not None else getattr(weakness, target)
        param_weights = WEAKNESS_TARGET_PARAMS[target]

        grid_size = self._adjust_grid_size(base_config.grid_size, param_weights, score)
        goal_distance = self._adjust_goal_distance(base_config, grid_size, param_weights, score)

        return GridWorldConfig(
            grid_size=grid_size,
            obstacle_density=self._adjust(
                base_config.obstacle_density, "obstacle_density", param_weights, score, lo=0.0, hi=0.9
            ),
            goal_distance=goal_distance,
            dead_end_probability=self._adjust(
                base_config.dead_end_probability,
                "dead_end_probability",
                param_weights,
                score,
                lo=0.0,
                hi=1.0,
            ),
            path_complexity=self._adjust(
                base_config.path_complexity, "path_complexity", param_weights, score, lo=0.0, hi=1.0
            ),
            max_steps=base_config.max_steps,
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )

    # -- helpers --------------------------------------------------------------

    def _adjust(
        self,
        base_value: float,
        param: str,
        param_weights: dict[str, float],
        score: float,
        lo: float,
        hi: float,
    ) -> float:
        weight = param_weights.get(param, 0.0)
        delta = weight * MAX_DELTAS[param] * score
        return _clamp(base_value + delta, lo, hi)

    def _adjust_grid_size(
        self, base_grid_size: int, param_weights: dict[str, float], score: float
    ) -> int:
        weight = param_weights.get("grid_size", 0.0)
        delta = round(weight * MAX_DELTAS["grid_size"] * score)
        return int(_clamp(base_grid_size + delta, MIN_GRID_SIZE, MAX_GRID_SIZE))

    def _adjust_goal_distance(
        self,
        base_config: GridWorldConfig,
        new_grid_size: int,
        param_weights: dict[str, float],
        score: float,
    ) -> int | None:
        weight = param_weights.get("goal_distance_fraction", 0.0)
        if weight == 0.0 and base_config.goal_distance is None:
            return None  # weakness doesn't touch goal distance; leave unconstrained

        base_fraction = (
            base_config.goal_distance / base_config.grid_size
            if base_config.goal_distance
            else 0.4  # reasonable default when the base config had no explicit target
        )
        delta_fraction = weight * MAX_DELTAS["goal_distance_fraction"] * score
        new_fraction = _clamp(base_fraction + delta_fraction, 0.1, 0.95)
        return max(1, round(new_fraction * new_grid_size))
