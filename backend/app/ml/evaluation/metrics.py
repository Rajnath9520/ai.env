"""Phase 23 -- Metrics (Section 23/39).

Computes every metric Section 39 lists, exclusively from real data already
produced by an experiment run (`ExperimentResult`, Phase 15) -- nothing
here is estimated, guessed, or fabricated. Where a metric genuinely cannot
be computed from the available data (e.g. sample efficiency when the
success-rate threshold was never reached), the field is `None` rather than
a made-up number, per Section 39's "Do not fabricate results."

Metric definitions (several of these are reasonable, documented choices
rather than uniquely-correct definitions -- flagged as such):

- success_rate / average_reward / average_steps / collision_rate: taken
  directly from the final curriculum step's evaluation.
- failure_rate: 1 - success_rate (the direct complement).
- sample_efficiency: total timesteps consumed up to and including the
  first curriculum step whose success rate reaches `threshold`. None if
  the threshold was never reached -- this is a real "did not converge"
  result, not something to paper over with an estimate.
- learning_progress: mean of all recorded per-step learning-progress
  values (Phase 11).
- convergence_speed: the earliest curriculum step index after which every
  subsequent step's success rate stays within `tolerance` of the final
  value, requiring at least `window` steps of that stability to avoid
  calling a single lucky late step "convergence." A documented heuristic,
  not a formal convergence test.
- generalization_score: mean success rate across whichever of
  validation/test/ood splits are present.
- ood_performance: the OOD split's success rate/reward, surfaced directly.
- failure_recovery_rate: successful_recoveries / dead_end_entries, summed
  across every recorded dead_end_recovery failure report. None if no
  dead-end encounters were ever recorded (nothing to compute a rate from).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from app.ml.curriculum.base import CurriculumStep
from app.ml.evaluation.learning_progress import LearningProgressRecord

DEFAULT_SAMPLE_EFFICIENCY_THRESHOLD = 0.5
DEFAULT_CONVERGENCE_TOLERANCE = 0.05
DEFAULT_CONVERGENCE_WINDOW = 3


@dataclass(frozen=True)
class MetricsReport:
    success_rate: float | None
    average_reward: float | None
    average_steps: float | None
    collision_rate: float | None
    failure_rate: float | None

    sample_efficiency_threshold: float
    sample_efficiency_timesteps: int | None

    learning_progress_mean: float | None

    convergence_step: int | None
    convergence_timesteps: int | None

    generalization_score: float | None
    ood_success_rate: float | None
    ood_average_reward: float | None

    failure_recovery_rate: float | None

    def to_dict(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "collision_rate": self.collision_rate,
            "failure_rate": self.failure_rate,
            "sample_efficiency_threshold": self.sample_efficiency_threshold,
            "sample_efficiency_timesteps": self.sample_efficiency_timesteps,
            "learning_progress_mean": self.learning_progress_mean,
            "convergence_step": self.convergence_step,
            "convergence_timesteps": self.convergence_timesteps,
            "generalization_score": self.generalization_score,
            "ood_success_rate": self.ood_success_rate,
            "ood_average_reward": self.ood_average_reward,
            "failure_recovery_rate": self.failure_recovery_rate,
        }


class MetricsCalculator:
    def __init__(
        self,
        sample_efficiency_threshold: float = DEFAULT_SAMPLE_EFFICIENCY_THRESHOLD,
        convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
        convergence_window: int = DEFAULT_CONVERGENCE_WINDOW,
    ) -> None:
        self.sample_efficiency_threshold = sample_efficiency_threshold
        self.convergence_tolerance = convergence_tolerance
        self.convergence_window = convergence_window

    def compute(
        self,
        curriculum_history: list[CurriculumStep],
        timesteps_per_step: int,
        learning_progress_history: list[LearningProgressRecord] | None = None,
        generalization_results: dict[str, dict] | None = None,
        failure_reports_history: list[list[dict]] | None = None,
    ) -> MetricsReport:
        final_step = curriculum_history[-1] if curriculum_history else None

        success_rate = final_step.success_rate if final_step else None
        failure_rate = (1.0 - success_rate) if success_rate is not None else None

        sample_efficiency_timesteps = self._sample_efficiency(curriculum_history, timesteps_per_step)
        convergence_step, convergence_timesteps = self._convergence(curriculum_history, timesteps_per_step)

        lp_values = [lp.learning_progress for lp in (learning_progress_history or [])]
        learning_progress_mean = statistics.mean(lp_values) if lp_values else None

        generalization_score = self._generalization_score(generalization_results)
        ood = (generalization_results or {}).get("ood")

        failure_recovery_rate = self._failure_recovery_rate(failure_reports_history)

        return MetricsReport(
            success_rate=success_rate,
            average_reward=final_step.average_reward if final_step else None,
            average_steps=final_step.average_steps if final_step else None,
            collision_rate=final_step.collision_rate if final_step else None,
            failure_rate=failure_rate,
            sample_efficiency_threshold=self.sample_efficiency_threshold,
            sample_efficiency_timesteps=sample_efficiency_timesteps,
            learning_progress_mean=learning_progress_mean,
            convergence_step=convergence_step,
            convergence_timesteps=convergence_timesteps,
            generalization_score=generalization_score,
            ood_success_rate=ood.get("success_rate") if ood else None,
            ood_average_reward=ood.get("average_reward") if ood else None,
            failure_recovery_rate=failure_recovery_rate,
        )

    # -- helpers --------------------------------------------------------------

    def _sample_efficiency(self, history: list[CurriculumStep], timesteps_per_step: int) -> int | None:
        for i, step in enumerate(history):
            if step.success_rate >= self.sample_efficiency_threshold:
                return (i + 1) * timesteps_per_step
        return None

    def _convergence(self, history: list[CurriculumStep], timesteps_per_step: int) -> tuple[int | None, int | None]:
        if not history:
            return None, None
        final_value = history[-1].success_rate

        for i in range(len(history)):
            window = history[i:]
            if len(window) < self.convergence_window:
                break
            if all(abs(s.success_rate - final_value) <= self.convergence_tolerance for s in window):
                return i, (i + 1) * timesteps_per_step
        return None, None

    def _generalization_score(self, generalization_results: dict[str, dict] | None) -> float | None:
        if not generalization_results:
            return None
        rates = [r["success_rate"] for r in generalization_results.values() if "success_rate" in r]
        return statistics.mean(rates) if rates else None

    def _failure_recovery_rate(self, failure_reports_history: list[list[dict]] | None) -> float | None:
        if not failure_reports_history:
            return None
        total_entries = 0
        total_recoveries = 0
        for step_reports in failure_reports_history:
            for report in step_reports:
                if report.get("failure_type") != "dead_end_recovery":
                    continue
                features = report.get("features", {})
                total_entries += features.get("dead_end_entries", 0)
                total_recoveries += features.get("successful_recoveries", 0)
        if total_entries == 0:
            return None
        return total_recoveries / total_entries
