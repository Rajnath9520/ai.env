# Phase 21 — Research Experiments: Pilot Results

**Status: preliminary pilot, not a conclusive test of the research hypothesis.**

## Setup

All four curriculum strategies (random, fixed, generic_adaptive, failure_driven)
were run with **identical** agent algorithm (PPO), training budget, evaluation
procedure, and base environment parameters — only the curriculum strategy and
seed differ between runs, per Section 37's requirement.

Shared configuration (`scripts/run_research_experiments.py::SHARED_KWARGS`):

| Parameter | Value |
|---|---|
| grid_size | 8 |
| base obstacle_density / dead_end_probability / path_complexity | 0.15 / 0.15 / 0.15 |
| max_steps_per_episode | 40 |
| n_curriculum_steps | 8 |
| timesteps_per_step | 8,000 (64,000 total per run) |
| eval_episodes | 10 |
| generalization envs/split, episodes/env | 6, 4 |
| seeds | 7, 13 |

**This is a small pilot**, sized to run within this environment's practical
compute budget (~100s per run x 8 runs). 64,000 timesteps is a modest budget
for PPO on even a small GridWorld -- not enough to expect strong convergence
for any strategy. Two seeds per strategy is enough to see *some* run-to-run
variance but far too few for a statistical test (that's Phase 24's job, and
it explicitly requires more seeds than this pilot has).

## Results (descriptive only -- no significance testing)

| Strategy | Final success rate (seed 7, 13) | Mean | OOD success rate (seed 7, 13) | Mean |
|---|---|---|---|---|
| random | 0.10, 0.00 | 0.05 | 0.125, 0.042 | 0.083 |
| fixed | 0.00, 0.30 | 0.15 | 0.042, 0.125 | 0.083 |
| generic_adaptive | 0.00, 0.10 | 0.05 | 0.125, 0.083 | 0.104 |
| failure_driven | 0.00, 0.00 | 0.00 | 0.167, 0.083 | 0.125 |

Full per-run data: `experiments/results/research/{strategy}_seed{seed}.json`.
Machine-readable summary: `experiments/results/research/summary.json`.

## Honest interpretation

**At this budget, no strategy shows a clear advantage on final in-distribution
success rate** -- all four are low (0-30%) and noisy across just two seeds.
`failure_driven` did not outperform the baselines on final success rate in
this pilot; its OOD generalization mean (0.125) was nominally the highest of
the four, but with n=2 seeds this is not distinguishable from noise.

This is not evidence against the research hypothesis -- it's evidence that
**64k timesteps and 2 seeds is too small a pilot to detect an effect either
way**, consistent with Section 1's instruction not to guarantee the
hypothesis is true and Section 58's ban on overclaiming. A real test of
"does failure-driven synthesis improve training efficiency and
generalization" needs substantially more training budget per curriculum step
and more seeds per strategy (Phase 24's statistical analysis phase is where
that gets formalized) -- this pilot's purpose was to prove the four-strategy
comparison pipeline runs correctly end-to-end with identical conditions, and
it does.

## What this phase actually validates

- All four curriculum strategies run to completion under identical conditions
  without errors.
- Results are stored per-run and are fully reproducible (same seed -> same
  `ExperimentConfig` -> deterministic run, per Phase 15's reproducibility
  guarantee).
- The comparison pipeline (shared config -> per-strategy runs -> aggregated
  summary) works correctly and is ready to be pointed at a much larger
  budget/seed count when more compute is available.
