# Research Methodology

## Research question

"Can conditioning environment synthesis on an AI agent's specific
behavioral failure modes, while optimizing for measurable learning
progress and out-of-distribution generalization, improve training
efficiency compared with random, fixed, and generic adaptive curricula?"

This is treated throughout as an experimental hypothesis, not an assumed
conclusion. This project does not claim to be the first system to
generate environments for AI training -- reinforcement learning,
automatic curriculum learning, procedural content generation (including
via RL), unsupervised environment design, and learning-progress-based
curricula all already exist. The specific combination under test here is
conditioning synthesis on *diagnosed behavioral failure type*, not just
aggregate difficulty or success rate.

## Experimental design

Four curriculum strategies are compared under **identical** conditions
(same PPO algorithm, same training budget per step, same evaluation
procedure, same base environment parameters) -- only the strategy and
random seed differ between runs:

- **A -- Random**: fully random parameters each step (`random_curriculum.py`)
- **B -- Fixed**: four predetermined difficulty stages (`fixed_curriculum.py`)
- **C -- Generic adaptive**: single scalar difficulty from overall success
  rate only (`generic_adaptive.py`)
- **D -- Failure-driven (proposed)**: the full pipeline in
  `docs/ml-pipeline.md`

Where possible, multiple seeds are used per strategy so single-seed noise
isn't mistaken for a real effect.

## Ablation design

Four variants isolate which component of the proposed method contributes
(`app/ml/curriculum/ablation.py`):

| Variant | Failure-analysis-driven targeting | Learning progress in selection | Diversity in selection |
|---|---|---|---|
| A | Yes | No | No |
| B | No (content-blinded) | Yes | No |
| C | Yes | Yes | No |
| D (full method) | Yes | Yes | Yes |

## Statistical methodology

`app/ml/evaluation/statistics_analysis.py::StatisticalAnalyzer`:

- Descriptive stats: mean, sample standard deviation, and a
  t-distribution-based confidence interval at a configurable confidence
  level (95% default).
- Significance testing: Welch's unequal-variance t-test, chosen because it
  doesn't assume the two groups have equal variance (a safer default than
  Student's t-test when that's unknown).
- **A hard floor (`min_n_for_test`, default 3) below which no test is run
  at all** -- `compare()` returns `test_performed=False` with an explicit
  reason rather than computing a technically-valid-but-meaningless p-value
  from two or three data points.

## Honesty commitments (and where they were actually exercised)

- No fabricated results, ever. Every number in every `FINDINGS.md` and
  `STATISTICAL_ANALYSIS.md` comes from an actually-executed run.
- No claims of significance without an actual test having been run and
  passing the sample-size floor.
- Negative/null results are reported as plainly as positive ones would
  be: at pilot scale (2 seeds, ~64k timesteps), **the failure-driven
  method did not show a distinguishable advantage** over the baselines on
  either the main comparison or the ablation, and every pairwise
  significance test was correctly refused for insufficient data. See
  `backend/experiments/results/research/FINDINGS.md`,
  `backend/experiments/results/ablation/FINDINGS.md`, and
  `backend/experiments/results/research/STATISTICAL_ANALYSIS.md`.
- The research question remains open. What this project delivers is a
  complete, tested, reproducible pipeline capable of answering it, run so
  far only at a scale too small to draw a conclusion from -- and that
  limitation is stated as plainly as possible rather than glossed over.

## What a real answer would require

Re-running `scripts/run_research_experiments.py` and
`scripts/run_ablation_study.py` with substantially more seeds per
condition (5-10+, per `StatisticalAnalyzer`'s own floor and general
practice) and a larger per-step training budget than this project's
development environment could afford within its compute constraints.
