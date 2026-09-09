"""Rule-based failure analysis (Phase 4 — a CORE research component).

Design choice, stated up front per the project's research-mode rule: this
first implementation uses interpretable, deterministic behavioral features
computed directly from trajectory summaries and environment structure —
not a learned/black-box classifier — per Section 11's explicit instruction.

What problem this solves: turns a trajectory into a structured judgment
about *why* the agent failed (not just that it failed), which the
environment synthesizer (Phase 9) needs in order to target new
environments at specific weaknesses rather than generically increasing
difficulty.

Assumptions (made explicit, not hidden): each detector below uses named
threshold constants calibrated by hand for a first working version — e.g.
"a collision on 30%+ of steps maxes out the obstacle-avoidance score." Real
thresholds should eventually be calibrated against a distribution of
trajectories from real training runs; until then these are reasonable,
documented guesses, not derived from data.

Alternative approaches considered: a learned classifier over trajectory
features would adapt its own thresholds, but needs labeled failure data
that doesn't exist yet and would hide its own reasoning behind weights —
the opposite of what an interpretable first version needs. A pure
LLM-based trajectory critique was also considered and rejected for the
same reason (unauditable, and non-deterministic across runs).

Limitations: these are correlational behavioral signals, not causal
diagnoses — e.g. a high `path_inefficiency` score does not prove the agent
"doesn't understand" the shortest path, only that its route was much
longer than optimal. `confidence` is reported per-detector specifically so
downstream consumers do not treat every score as equally reliable evidence.

Evaluation: these detectors can be validated empirically by checking that
scores correlate with independently-verifiable trajectory properties (e.g.
`obstacle_avoidance` score should track collision_count, which is directly
observable) and, longer-term, against human-labeled example trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.analysis.dead_ends import dead_end_zone_cells, reachable_cells
from app.ml.analysis.trajectory import Trajectory, TrajectorySummary
from app.ml.environments.validator import EnvironmentValidator, GridWorldLayout, ValidationResult

# -- documented threshold constants (see module docstring) -------------------

OBSTACLE_COLLISION_RATE_SATURATION = 0.30  # collision on 30%+ of steps -> score 1.0
EXPLORATION_SUCCESS_DAMPING = 0.3          # exploration score damping when the episode succeeded
LOOP_REVERSAL_RATE_SATURATION = 0.40       # direction-reversal on 40%+ of transitions -> score 1.0
LOOP_REPEAT_RATE_SATURATION = 0.60         # revisiting 60%+ of visited cells -> score 1.0
PATH_INEFFICIENCY_RATIO_SATURATION = 3.0   # taking 3x the shortest path -> score 1.0
DEAD_END_STEP_FRACTION_WEIGHT = 0.5
DEAD_END_UNRECOVERED_WEIGHT = 0.5
LONG_HORIZON_PATH_LENGTH_SATURATION_FACTOR = 2.0  # shortest_path_length >= 2x grid_size -> full "distant goal" weight


@dataclass(frozen=True)
class FailureReport:
    failure_type: str
    score: float
    confidence: float
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "features": self.features,
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class FailureAnalyzer:
    """Stateless: every method is a pure function of its inputs so each
    detector can be unit-tested against hand-built trajectories/layouts
    independent of the RL agent or environment generator.
    """

    def __init__(self, validator: EnvironmentValidator | None = None) -> None:
        self._validator = validator or EnvironmentValidator()

    def analyze(
        self,
        trajectory: Trajectory,
        summary: TrajectorySummary,
        layout: GridWorldLayout,
        validation: ValidationResult | None = None,
    ) -> list[FailureReport]:
        validation = validation or self._validator.validate(layout)
        length = max(summary.trajectory_length, 1)

        return [
            self._obstacle_avoidance(summary, length),
            self._dead_end_recovery(trajectory, summary, layout, length),
            self._exploration(summary, validation, length),
            self._repeated_action_loops(summary, length),
            self._path_inefficiency(summary, validation),
            self._short_horizon_planning(summary, length),
            self._long_horizon_planning(summary, validation, layout),
        ]

    # -- individual detectors ------------------------------------------------

    def _obstacle_avoidance(self, summary: TrajectorySummary, length: int) -> FailureReport:
        collision_rate = summary.collision_count / length
        score = _clamp(collision_rate / OBSTACLE_COLLISION_RATE_SATURATION)
        confidence = _clamp(length / 15)
        return FailureReport(
            failure_type="obstacle_avoidance",
            score=score,
            confidence=confidence,
            features={
                "collision_count": summary.collision_count,
                "trajectory_length": summary.trajectory_length,
                "collision_rate": round(collision_rate, 4),
            },
        )

    def _dead_end_recovery(
        self,
        trajectory: Trajectory,
        summary: TrajectorySummary,
        layout: GridWorldLayout,
        length: int,
    ) -> FailureReport:
        zone = dead_end_zone_cells(layout, reachable_cells(layout))

        entries = 0
        recoveries = 0
        in_zone = trajectory.steps[0].agent_pos in zone if trajectory.steps else False
        recovery_run_lengths: list[int] = []
        current_run = 1 if in_zone else 0

        for step in trajectory.steps:
            now_in_zone = step.next_agent_pos in zone
            if now_in_zone and not in_zone:
                entries += 1
                current_run = 1
            elif now_in_zone and in_zone:
                current_run += 1
            elif not now_in_zone and in_zone:
                recoveries += 1
                recovery_run_lengths.append(current_run)
                current_run = 0
            in_zone = now_in_zone

        dead_end_steps = sum(1 for s in trajectory.steps if s.next_agent_pos in zone)
        recovery_rate = recoveries / entries if entries else 1.0
        avg_recovery_steps = (
            sum(recovery_run_lengths) / len(recovery_run_lengths) if recovery_run_lengths else 0.0
        )

        if entries == 0:
            score = 0.0
            confidence = 0.3  # no dead ends encountered — weak evidence either way
        else:
            score = _clamp(
                DEAD_END_STEP_FRACTION_WEIGHT * (dead_end_steps / length)
                + DEAD_END_UNRECOVERED_WEIGHT * (1 - recovery_rate)
            )
            confidence = _clamp(entries / 3)

        return FailureReport(
            failure_type="dead_end_recovery",
            score=score,
            confidence=confidence,
            features={
                "dead_end_entries": entries,
                "successful_recoveries": recoveries,
                "average_recovery_steps": round(avg_recovery_steps, 2),
                "dead_end_zone_size": len(zone),
            },
        )

    def _exploration(
        self, summary: TrajectorySummary, validation: ValidationResult, length: int
    ) -> FailureReport:
        novelty_rate = summary.unique_states_visited / length
        raw = 1 - novelty_rate
        score = raw if not summary.success else raw * EXPLORATION_SUCCESS_DAMPING
        confidence = _clamp(length / 15)
        return FailureReport(
            failure_type="exploration",
            score=_clamp(score),
            confidence=confidence,
            features={
                "unique_states_visited": summary.unique_states_visited,
                "trajectory_length": summary.trajectory_length,
                "novelty_rate": round(novelty_rate, 4),
                "reachable_cell_count": validation.reachable_cell_count,
            },
        )

    def _repeated_action_loops(self, summary: TrajectorySummary, length: int) -> FailureReport:
        reversal_rate = summary.direction_reversals / max(length - 1, 1)
        repeat_rate = summary.repeated_states / length
        score = _clamp(
            0.5 * _clamp(reversal_rate / LOOP_REVERSAL_RATE_SATURATION)
            + 0.5 * _clamp(repeat_rate / LOOP_REPEAT_RATE_SATURATION)
        )
        confidence = _clamp(length / 15)
        return FailureReport(
            failure_type="repeated_action_loops",
            score=score,
            confidence=confidence,
            features={
                "direction_reversals": summary.direction_reversals,
                "repeated_states": summary.repeated_states,
                "reversal_rate": round(reversal_rate, 4),
                "repeat_rate": round(repeat_rate, 4),
            },
        )

    def _path_inefficiency(
        self, summary: TrajectorySummary, validation: ValidationResult
    ) -> FailureReport:
        shortest = validation.shortest_path_length or 1
        ratio = summary.trajectory_length / shortest
        score = _clamp((ratio - 1) / (PATH_INEFFICIENCY_RATIO_SATURATION - 1))
        confidence = 1.0 if summary.success else 0.5
        return FailureReport(
            failure_type="path_inefficiency",
            score=score,
            confidence=confidence,
            features={
                "trajectory_length": summary.trajectory_length,
                "shortest_path_length": validation.shortest_path_length,
                "inefficiency_ratio": round(ratio, 3),
            },
        )

    def _short_horizon_planning(self, summary: TrajectorySummary, length: int) -> FailureReport:
        """Local, immediate decision quality: collisions and reversals are
        both signs of poor next-step choices rather than poor long-range
        route planning."""
        collision_rate = summary.collision_count / length
        reversal_rate = summary.direction_reversals / max(length - 1, 1)
        score = _clamp(
            0.5 * _clamp(collision_rate / OBSTACLE_COLLISION_RATE_SATURATION)
            + 0.5 * _clamp(reversal_rate / LOOP_REVERSAL_RATE_SATURATION)
        )
        confidence = _clamp(length / 15)
        return FailureReport(
            failure_type="short_horizon_planning",
            score=score,
            confidence=confidence,
            features={
                "collision_rate": round(collision_rate, 4),
                "reversal_rate": round(reversal_rate, 4),
            },
        )

    def _long_horizon_planning(
        self,
        summary: TrajectorySummary,
        validation: ValidationResult,
        layout: GridWorldLayout,
    ) -> FailureReport:
        """Whether failure correlates with the goal being structurally
        distant/complex (long shortest path) rather than nearby — a proxy
        for "did this fail because the plan needed to span many steps."""
        shortest = validation.shortest_path_length or 0
        saturation = LONG_HORIZON_PATH_LENGTH_SATURATION_FACTOR * layout.grid_size
        distance_term = _clamp(shortest / saturation) if saturation else 0.0
        score = 0.0 if summary.success else distance_term
        confidence = _clamp(shortest / 10) if shortest else 0.2
        return FailureReport(
            failure_type="long_horizon_planning",
            score=score,
            confidence=confidence,
            features={
                "shortest_path_length": validation.shortest_path_length,
                "grid_size": layout.grid_size,
                "success": summary.success,
            },
        )
