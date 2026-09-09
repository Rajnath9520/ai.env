# ML Pipeline

The full adaptive loop, module by module. Every class named here is real,
tested code -- file paths point to `backend/app/ml/`.

## 1. Environment (`environments/`)

- `config.py::GridWorldConfig` -- validated parameters (grid_size,
  obstacle_density, goal_distance, dead_end_probability, path_complexity,
  max_steps, seed).
- `generator.py::GridWorldGenerator` -- **guarantee-by-construction**
  generation: carves a connecting start->goal path first (always kept
  free), adds loop/dead-end branches per the complexity knobs, then
  scatters obstacles over the remaining cells. This makes every generated
  layout solvable *by construction*, not by reject-and-retry -- verified
  across 540+ parameter combinations with zero invalid environments.
- `validator.py::EnvironmentValidator` -- BFS reachability, shortest-path
  length, dead-end counting; used as a confirmation check, not the primary
  correctness mechanism.
- `gridworld.py::GridWorldEnv` -- the Gymnasium-compatible environment
  (4 discrete actions, position+local-obstacle-patch observation).

## 2. Agent (`training/`)

- `trainer.py::AgentTrainer` -- wraps Stable-Baselines3 PPO. One
  `AgentTrainer` instance persists its policy across repeated `train()`
  calls on different environments -- this continuity across curriculum
  steps is what makes the curriculum "adaptive" rather than
  "retrain from scratch every step."
- `metrics.py::TrainingMetricsCallback` -- per-episode metrics during
  training, with an optional `on_episode` hook used for live streaming.

## 3. Trajectory & failure analysis (`analysis/`)

- `trajectory_logger.py::TrajectoryLogger`/`TrajectoryStore` -- records
  full step-by-step trajectories (state, action, reward, done, positions,
  collisions) for a bounded number of recent episodes; everything older is
  only kept as a summary.
- `trajectory.py::summarize()` -- turns a raw trajectory into
  `TrajectorySummary` (length, total reward, success, collisions, unique
  states, repeated states, direction reversals, distance to goal).
- `dead_ends.py` -- leaf-pruning algorithm identifying full dead-end
  *zones* (not just terminal cells) via iterative degree-1 removal on the
  free-cell graph.
- `failure_analyzer.py::FailureAnalyzer` -- 7 rule-based detectors, each
  emitting `{failure_type, score, confidence, features}`:
  `obstacle_avoidance`, `dead_end_recovery`, `exploration`,
  `repeated_action_loops`, `path_inefficiency`, `short_horizon_planning`,
  `long_horizon_planning`. See `docs/failure-analysis.md` for the exact
  formulas and documented thresholds.
- `weakness_vector.py::WeaknessVectorBuilder` -- confidence-weighted
  aggregation of failure reports across episodes into a 6-dimensional
  `WeaknessVector`, with `primary_weakness`/`secondary_weaknesses` derived
  from the measured values (never chosen randomly).

## 4. Environment synthesis (`environments/synthesizer.py`, `candidate_generator.py`)

- `EnvironmentSynthesizer` -- maps each weakness dimension to the specific
  parameters it should move (e.g. `obstacle_avoidance` -> obstacle_density
  only; `exploration` -> grid_size + goal_distance only), scaled by the
  weakness's measured severity. Verified per-dimension: targeting one
  weakness never moves parameters outside its documented set.
- `CandidateGenerator` -- produces several candidates per step (cycling
  through primary + secondary weaknesses, jittering severity), each with
  `difficulty`, `validity_score` (checked against the real generator, not
  assumed), and `diversity_score` (distance in parameter space from prior
  environments).

## 5. Selection (`curriculum/controller.py`)

`CurriculumController.select()` scores every candidate:

```
Score(E) = alpha * WeaknessAlignment(E)
         + beta  * ExpectedLearningProgress(E)
         + gamma * Diversity(E)
         - delta * ExcessiveDifficulty(E)
```

All four weights are configurable, and every component of the score is
stored (`ScoredCandidate.to_dict()`), not just the winner.

## 6. Learning progress (`evaluation/learning_progress.py`)

`LearningProgressEvaluator` measures `performance_after - performance_before`
around a training call, defaulting to `success_rate` as the metric
(reward is also supported; path-efficiency/failure-reduction metrics are
explicitly not yet wired in, since they'd need trajectory-level data the
evaluator doesn't currently return).

## 7. The full loop (`curriculum/failure_driven.py::AdaptiveCurriculumRunner`)

Wires 1-6 together exactly as `docs/architecture.md`'s data-flow section
describes. Step 0 trains on a base config with no target weakness (no
trajectories exist yet to analyze); every subsequent step targets the most
recently measured weakness vector.

## 8. Baselines (`curriculum/{random,fixed,generic_adaptive}_curriculum.py`)

Three simpler strategies sharing the same `CurriculumRunner` (not the
`AdaptiveCurriculumRunner`, which needs trajectories/weakness vectors these
baselines never touch) so the four-way comparison in
`docs/research-methodology.md` is apples-to-apples:

- **Random**: samples parameters uniformly at random each step.
- **Fixed**: four predetermined difficulty stages (easy/moderate/hard/very_hard),
  advancing purely on step count.
- **Generic adaptive**: a single scalar difficulty, nudged up/down purely
  from overall success rate thresholds (>0.85 / <0.40) -- no failure-type
  detail at all.

## 9. Ablations (`curriculum/ablation.py`)

Four variants isolating which component of the proposed method matters:
A (failure analysis only), B (learning progress only, with targeting
content-blinded via `_blind_weakness_vector`), C (both, no diversity), D
(all three -- the full method). See `docs/research-methodology.md` for
results and their honest interpretation.
