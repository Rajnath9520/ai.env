"""Phase 24 -- Statistical Analysis (Section 24/40).

"When enough experiments are available: calculate mean, standard
deviation, confidence intervals where appropriate. Compare methods. Do not
claim statistical significance unless an appropriate statistical test has
actually been performed. Keep the analysis scientifically honest."

This module is the enforcement mechanism for that rule: `compare()` will
only report a p-value/significance verdict if a real test was actually run
(Welch's t-test, via scipy -- chosen because it doesn't assume equal
variance between groups, a safer default than Student's t-test when group
sizes/variances aren't known to match). Below `min_n_for_test` samples per
group, it explicitly refuses to run the test and says why, rather than
running a technically-computable-but-meaningless test on two or three data
points and reporting a p-value that looks authoritative but isn't.
"""

from __future__ import annotations

import statistics as pystats
from dataclasses import dataclass, field
from math import sqrt

from scipy import stats as scipy_stats

MIN_N_FOR_TEST = 3
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class DescriptiveStats:
    n: int
    mean: float | None
    std: float | None
    ci_low: float | None
    ci_high: float | None
    confidence_level: float
    values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence_level": self.confidence_level,
            "values": self.values,
        }


@dataclass(frozen=True)
class ComparisonResult:
    label_a: str
    label_b: str
    stats_a: DescriptiveStats
    stats_b: DescriptiveStats
    test_performed: bool
    test_name: str | None
    t_statistic: float | None
    p_value: float | None
    significant: bool | None
    alpha: float
    note: str

    def to_dict(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "stats_a": self.stats_a.to_dict(),
            "stats_b": self.stats_b.to_dict(),
            "test_performed": self.test_performed,
            "test_name": self.test_name,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "note": self.note,
        }


class StatisticalAnalyzer:
    def __init__(self, min_n_for_test: int = MIN_N_FOR_TEST, confidence_level: float = CONFIDENCE_LEVEL) -> None:
        self.min_n_for_test = min_n_for_test
        self.confidence_level = confidence_level

    def describe(self, values: list[float]) -> DescriptiveStats:
        n = len(values)
        if n == 0:
            return DescriptiveStats(
                n=0, mean=None, std=None, ci_low=None, ci_high=None, confidence_level=self.confidence_level, values=[]
            )
        mean = pystats.mean(values)
        if n < 2:
            return DescriptiveStats(
                n=n,
                mean=mean,
                std=None,
                ci_low=None,
                ci_high=None,
                confidence_level=self.confidence_level,
                values=values,
            )
        std = pystats.stdev(values)
        sem = std / sqrt(n)
        t_crit = scipy_stats.t.ppf(1 - (1 - self.confidence_level) / 2, df=n - 1)
        margin = t_crit * sem
        return DescriptiveStats(
            n=n,
            mean=mean,
            std=std,
            ci_low=mean - margin,
            ci_high=mean + margin,
            confidence_level=self.confidence_level,
            values=values,
        )

    def compare(
        self, label_a: str, values_a: list[float], label_b: str, values_b: list[float], alpha: float = 0.05
    ) -> ComparisonResult:
        stats_a = self.describe(values_a)
        stats_b = self.describe(values_b)

        if len(values_a) < self.min_n_for_test or len(values_b) < self.min_n_for_test:
            return ComparisonResult(
                label_a=label_a,
                label_b=label_b,
                stats_a=stats_a,
                stats_b=stats_b,
                test_performed=False,
                test_name=None,
                t_statistic=None,
                p_value=None,
                significant=None,
                alpha=alpha,
                note=(
                    f"Insufficient samples for a significance test (n_{label_a}={len(values_a)}, "
                    f"n_{label_b}={len(values_b)}; minimum {self.min_n_for_test} required per group). "
                    "Reporting descriptive statistics only -- no significance claim is made."
                ),
            )

        t_statistic, p_value = scipy_stats.ttest_ind(values_a, values_b, equal_var=False)
        significant = bool(p_value < alpha)
        return ComparisonResult(
            label_a=label_a,
            label_b=label_b,
            stats_a=stats_a,
            stats_b=stats_b,
            test_performed=True,
            test_name="Welch's t-test (unequal variance)",
            t_statistic=float(t_statistic),
            p_value=float(p_value),
            significant=significant,
            alpha=alpha,
            note=f"{'Significant' if significant else 'Not significant'} at alpha={alpha} (p={p_value:.4f}).",
        )
