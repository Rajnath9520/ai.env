# Failure-Driven Environment Synthesis for Adaptive AI Training
## Research Report

---

## 1. Abstract

This report describes and evaluates a research platform investigating
whether conditioning procedural environment generation on a reinforcement
learning agent's *specific detected behavioral failure modes* -- rather
than on aggregate difficulty or overall success rate alone -- improves
training efficiency and out-of-distribution generalization. A complete,
tested pipeline was built: a guaranteed-solvable 2D GridWorld generator,
a PPO agent, rule-based failure analysis across seven behavioral failure
types, a six-dimensional weakness vector, a targeted environment
synthesizer, a candidate-scoring curriculum controller, and a
learning-progress evaluator, wired into a full adaptive loop and compared
against three baselines (random, fixed, and generic-adaptive curricula)
under identical conditions. The proposed method (D) and an ablation
isolating its components were run at pilot scale (2 seeds, ~64,000
timesteps per run). **At this scale, no strategy or ablation variant
showed a statistically distinguishable advantage** -- a proper statistical
analysis correctly declined to test significance given the sample size.
The research question therefore remains open; this report's contribution
is a working, reproducible, honestly-evaluated pipeline and pilot data,
not a positive result.

## 2. Introduction

Reinforcement learning agents are typically trained on a fixed
environment distribution or a curriculum that increases difficulty over
time. Most curriculum learning approaches -- including automatic curriculum
learning (ACL) methods -- adjust difficulty from a single aggregate signal,
usually overall success rate, without diagnosing *why* the agent is
failing. This project investigates a more targeted alternative: detect
the agent's specific behavioral weaknesses (e.g., poor obstacle avoidance
vs. poor dead-end recovery vs. poor long-horizon planning) and generate
environments that specifically exercise the detected weakness.

## 3. Problem Statement

Given an RL agent training in a family of procedurally generated
environments, can an environment-generation policy that conditions on the
agent's diagnosed behavioral failure modes produce better sample
efficiency and generalization than one that does not?

## 4. Motivation

A generic adaptive curriculum that only tracks success rate cannot tell
"the agent keeps colliding with obstacles" apart from "the agent gets
stuck in dead ends" -- it can only make the next environment uniformly
easier or harder. If specific failure diagnosis is informative, targeting
environments at the diagnosed weakness should produce more useful learning
signal per training step than an undifferentiated difficulty increase.

## 5. Related Work

This project sits alongside, and does not claim to supersede, several
existing areas: reinforcement learning, automatic curriculum learning,
procedural content generation (including via RL), unsupervised
environment design, adversarial environment generation,
learning-progress-based curriculum learning, multi-agent curriculum
learning, and adaptive task generation. No claim is made that this is the
first system to generate environments for AI training.

## 6. Research Gap

The specific combination under test -- conditioning environment synthesis
on *diagnosed, multi-dimensional behavioral failure type* (not just a
scalar difficulty or success-rate signal) while also optimizing for
measured learning progress and diversity -- is the gap this project's
experiments target, framed as an open hypothesis rather than an assumed
result (see Section 16, Results).

## 7. Proposed Method

The failure-driven curriculum: train -> collect trajectories -> analyze
failures (7 rule-based detectors) -> aggregate into a 6-dimensional
weakness vector -> synthesize several candidate environments each targeted
at a specific weakness -> score candidates on weakness alignment, expected
learning progress, and diversity (penalizing excessive difficulty) ->
select and train -> measure learning progress -> repeat. Full detail:
`docs/ml-pipeline.md`.

## 8. Architecture

A layered system: a Next.js frontend talks to a FastAPI backend via REST
and WebSocket; the backend's service layer orchestrates a fully
web-independent ML engine; results persist to PostgreSQL. Full detail,
including known architectural limitations, in `docs/architecture.md`.

## 9. Environment Design

A 2D GridWorld with configurable grid size, obstacle density, goal
distance, dead-end probability, and path complexity. The generator is
guarantee-by-construction (carves a connecting path first, then adds
branches and obstacles), verified across 540 parameter combinations with
zero invalid environments produced. Full detail: `docs/environment-generation.md`.

## 10. RL Method

Proximal Policy Optimization (PPO) via Stable-Baselines3, with default
hyperparameters plus a slightly elevated entropy coefficient (0.01) to
favor exploration in this sparse-reward setting. No custom RL algorithm
was developed or is claimed -- the contribution under test is entirely in
the environment-generation and curriculum layer around a standard
algorithm.

## 11. Failure Analysis

Seven interpretable, deterministic detectors (obstacle avoidance,
dead-end recovery, exploration, repeated-action loops, path inefficiency,
short-horizon planning, long-horizon planning), each producing a
`{score, confidence, features}` triple from measurable trajectory
signals -- not a learned classifier, deliberately, for auditability with
no labeled failure data yet available. Full detail: `docs/failure-analysis.md`.

## 12. Environment Synthesis

Each of the six weakness dimensions maps to a specific, documented subset
of environment parameters (e.g., `obstacle_avoidance` moves only
obstacle_density). Verified directly: targeting one weakness never moves
parameters outside its documented set. Full detail: `docs/environment-generation.md`.

## 13. Learning Progress

`LearningProgress(E) = Performance_after(E) - Performance_before(E)`,
using success rate as the default performance metric. Computed for every
curriculum step of the failure-driven method.

## 14. Experimental Setup

All four curriculum strategies were run under **identical** conditions:

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| grid_size | 8 |
| Base obstacle_density / dead_end_probability / path_complexity | 0.15 / 0.15 / 0.15 |
| max_steps per episode | 40 |
| Curriculum steps | 8 |
| Timesteps per step | 8,000 (64,000 total per run) |
| Evaluation episodes | 10 |
| Generalization environments/split, episodes/environment | 6, 4 |
| Seeds | 7, 13 |

This is explicitly a **pilot-scale** budget, sized to what this project's
development environment could run in practical wall-clock time (roughly
100 seconds per run). It is not claimed to be sufficient to detect a real
effect -- see Section 16 and Section 20.

## 15. Baselines

- **Random**: uniformly random environment parameters each step.
- **Fixed**: four predetermined difficulty stages (easy -> moderate -> hard
  -> very_hard), advancing on step count alone.
- **Generic adaptive**: a single scalar difficulty, nudged up/down purely
  from overall success rate (>0.85 -> harder, <0.40 -> easier), no failure
  detail.
- **Failure-driven (proposed)**: Section 7's full loop.

## 16. Results

*All values below are real, experimentally obtained numbers from
`backend/experiments/results/research/summary.json`, generated by
`scripts/run_research_experiments.py`. No values in this section are
estimated or fabricated.*

### Final success rate (mean of 2 seeds)

| Strategy | Seed 7 | Seed 13 | Mean |
|---|---|---|---|
| Random | 0.10 | 0.00 | 0.050 |
| Fixed | 0.00 | 0.30 | 0.150 |
| Generic adaptive | 0.00 | 0.10 | 0.050 |
| Failure-driven | 0.00 | 0.00 | 0.000 |

### Generalization (mean success rate across 2 seeds)

| Strategy | Validation | Test | OOD |
|---|---|---|---|
| Random | 0.083 | 0.188 | 0.083 |
| Fixed | 0.125 | 0.083 | 0.083 |
| Generic adaptive | 0.125 | 0.083 | 0.104 |
| Failure-driven | 0.125 | 0.104 | 0.125 |

### Statistical significance

Every pairwise comparison between strategies was tested with
`StatisticalAnalyzer.compare()` (Welch's t-test, `min_n_for_test=3`). With
n=2 per strategy, **every comparison was correctly refused**
(`test_performed: False`) -- no p-value or significance verdict is
reported for any pair. Full detail:
`backend/experiments/results/research/STATISTICAL_ANALYSIS.md`.

**At this pilot scale, the failure-driven method's final success rate
(0.000) was the lowest of the four strategies**, though its OOD
generalization mean (0.125) was nominally the highest. Neither observation
is statistically meaningful with n=2.

## 17. Ablation Study

*Also real data, from `backend/experiments/results/ablation/summary.json`,
generated by `scripts/run_ablation_study.py`, same budget as Section 14.*

| Variant | Description | Seed 7 | Seed 13 | Mean |
|---|---|---|---|---|
| A | Failure analysis only | 0.20 | 0.00 | 0.100 |
| B | Learning progress only (targeting content-blinded) | 0.10 | 0.30 | 0.200 |
| C | Failure analysis + learning progress | 0.00 | 0.00 | 0.000 |
| D | Full method (+ diversity) | 0.00 | 0.00 | 0.000 |

**At this pilot scale, the full method (D) did not outperform its
simpler components.** Variant B (learning progress only, with failure
analysis deliberately blinded) had the highest mean of the four. With
n=2 seeds per variant, none of these differences can be attributed to the
ablated components rather than noise.

## 18. Generalization

Validation/test/OOD splits are strictly seed-disjoint from training by
construction (`app/ml/evaluation/generalization.py`), verified
programmatically (`assert_no_seed_overlap`), not just assumed. See
Section 16's generalization table for the real numbers obtained; all four
strategies show broadly similar, low generalization scores at this
budget, consistent with 64,000 timesteps being insufficient for strong
generalization under any strategy tested.

## 19. Discussion

The central, honest finding of this project's experimental work is
negative in the specific sense that **no evidence was found, at the scale
tested, that failure-driven environment synthesis outperforms simpler
curricula**. This is not evidence that the hypothesis is false -- with
n=2 seeds and a training budget an order of magnitude smaller than what
RL curriculum-learning literature typically uses for GridWorld-scale
tasks, the experiment as run has very low statistical power to detect an
effect of any realistic size. The confidence intervals computed in
`STATISTICAL_ANALYSIS.md` are wide enough to extend outside `[0, 1]` for
several strategies, which is itself a quantitative signal of how
underpowered this pilot is.

What can be said with confidence: the full pipeline -- generation,
training, failure analysis, weakness aggregation, synthesis, candidate
scoring, selection, learning-progress measurement, generalization
evaluation, and statistical analysis -- runs correctly end to end, is
fully reproducible (byte-identical curriculum histories from the same
seed, verified directly), and is ready to be pointed at a real-scale
experiment.

## 20. Limitations

- **Pilot-scale only**: 2 seeds and 64,000 timesteps per run, far below
  what would be needed for a well-powered test of the research question.
- **Rule-based failure detection**: thresholds are hand-calibrated, not
  fit to labeled data (none exists yet).
- **Hand-specified synthesis/scoring weights**: the weakness-to-parameter
  mapping and the controller's alpha/beta/gamma/delta weights are
  documented, reasonable choices, not learned or empirically tuned.
- **No raw trajectory persistence**: only summarized statistics are
  stored in the database, by design, which limits some possible post-hoc
  analyses.
- **Single environment family**: all experiments use 2D GridWorld;
  results may not transfer to other environment classes.
- **No multi-worker production deployment**: in-process broadcaster and
  cancellation flags don't scale past a single API process.

## 21. Future Work

1. Re-run `scripts/run_research_experiments.py` and
   `scripts/run_ablation_study.py` with 5-10+ seeds per condition and a
   substantially larger per-step training budget, then re-run
   `StatisticalAnalyzer.compare()` for a real significance test.
2. Calibrate failure-detector thresholds against a labeled dataset of
   real training trajectories, once enough exist.
3. Learn the weakness-to-parameter synthesis mapping and/or the
   controller's scoring weights from data, rather than hand-specifying
   them.
4. Extend beyond 2D GridWorld to richer environment families, per the
   project's own explicitly-deferred "optional advanced version" scope
   (neural/evolutionary environment generators, richer task
   representations).
5. Persist raw or partially-summarized trajectories for deeper post-hoc
   failure analysis, within a bounded storage budget.

## 22. Conclusion

This project delivers a complete, extensively tested (320 tests across
backend and frontend), reproducible research platform for studying
failure-driven environment synthesis in adaptive RL training, including a
working web dashboard, REST/WebSocket API, and Docker deployment
configuration. Its experimental contribution to date is a pilot-scale,
honestly-reported null result: at the tested scale, no curriculum
strategy -- including the proposed failure-driven method -- showed a
statistically distinguishable advantage. The research question posed in
Section 2 remains open and is now answerable with the tooling this project
provides, given a larger compute budget than was available during
development.

## 23. References

This project's implementation draws on well-established, standard
techniques and libraries rather than claiming novel algorithmic
contributions: Proximal Policy Optimization (Schulman et al., 2017) as
implemented in Stable-Baselines3; the Gymnasium environment interface
(the maintained successor to OpenAI Gym); Welch's t-test (Welch, 1947) for
unequal-variance significance testing; and standard breadth-first search
for maze connectivity/shortest-path computation. No claims in this report
rely on a specific citation's numeric results -- all numeric results
reported here are this project's own, obtained as described in Sections
14, 16, and 17.
