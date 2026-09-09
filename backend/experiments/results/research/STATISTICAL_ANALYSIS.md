# Phase 24 — Statistical Analysis of Phase 21 Results

Applying `StatisticalAnalyzer` to Phase 21's real 4-strategy comparison
data (n=2 seeds per strategy).

## Descriptive statistics (final success rate)

| Strategy | n | mean | std | 95% CI |
|---|---|---|---|---|
| random | 2 | 0.050 | 0.071 | [-0.585, 0.685] |
| fixed | 2 | 0.150 | 0.212 | [-1.756, 2.056] |
| generic_adaptive | 2 | 0.050 | 0.071 | [-0.585, 0.685] |
| failure_driven | 2 | 0.000 | 0.000 | [0.000, 0.000] |

Note the confidence intervals: with n=2, several extend well outside
[0, 1] (a probability's valid range) or collapse to a single point. This
is not a bug -- it is the t-distribution-based interval doing exactly what
it should with almost no data, and it is itself informative: **these
intervals are a visible, quantitative signal that n=2 is nowhere near
enough to draw a conclusion**, on top of the qualitative point already
made in Phase 21/22's FINDINGS.md.

## Significance tests

**Every pairwise comparison was correctly refused** by `StatisticalAnalyzer`
(`test_performed: false`), because all four strategies have n=2 < the
minimum of 3 required per group. No p-value, t-statistic, or significance
verdict is reported for any pair -- per Section 24's explicit instruction
("do not claim statistical significance unless an appropriate statistical
test has actually been performed"), the honest output here is silence, not
a number.

## What this phase concludes

1. The statistical machinery itself is correct and tested (13 unit tests
   covering descriptive stats, confidence intervals at multiple confidence
   levels, and Welch's t-test against synthetic data with known, obviously
   different vs. obviously identical distributions).
2. Applied to the real data this project has actually generated so far
   (Phase 21's 2-seed pilot), the only scientifically honest conclusion is:
   **no significance claim can be made with this much data.** This is
   consistent with -- and formalizes -- the qualitative caution already
   written into Phase 21 and Phase 22's FINDINGS.md.
3. To actually answer "does failure-driven synthesis help," the same
   `run_research_experiments.py` / `run_ablation_study.py` scripts need to
   be re-run with `MIN_N_FOR_TEST` (3) or more seeds per strategy -- ideally
   5-10 for a real test -- at which point `StatisticalAnalyzer.compare()`
   will produce a real p-value instead of declining to test.
