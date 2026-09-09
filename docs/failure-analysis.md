# Failure Analysis

`FailureAnalyzer` (`app/ml/analysis/failure_analyzer.py`) is a **CORE
research component**. It is deliberately rule-based and deterministic for
this first version, not a learned/black-box classifier -- see the module
docstring for the full rationale (auditability, no labeled training data
yet, and the explicit requirement not to hide reasoning behind weights).

Every detector returns `{failure_type, score, confidence, features}` with
`score`/`confidence` in `[0, 1]`.

## The 7 detectors

| Detector | Signal used | Saturates at |
|---|---|---|
| `obstacle_avoidance` | collision rate | 30%+ of steps collided |
| `dead_end_recovery` | dead-end zone entries vs. successful recoveries (see below) | high unrecovered time in zone |
| `exploration` | novelty rate (unique states / steps), damped for successful episodes | low novelty on a failed episode |
| `repeated_action_loops` | direction-reversal rate + repeated-state rate | 40%/60%+ respectively |
| `path_inefficiency` | actual length vs. BFS shortest path | 3x the shortest path |
| `short_horizon_planning` | collision rate + reversal rate (local decision quality) | same saturation points as above |
| `long_horizon_planning` | whether failure correlates with a structurally distant/complex goal | shortest path >= 2x grid size |

Dead-end zones (not just terminal cells) are identified via iterative
leaf-pruning on the free-cell graph (`app/ml/analysis/dead_ends.py`) --
this correctly excludes branches that have an alternate loop route back to
the main path.

All threshold constants (30%, 40%, 60%, 3x, etc.) are named module-level
constants, documented as hand-calibrated first-pass values, not derived
from labeled data.

## Weakness aggregation

`WeaknessVectorBuilder` (`app/ml/analysis/weakness_vector.py`) aggregates
failure reports across many episodes into the 6-dimensional
`WeaknessVector` this project specifies:

- Each failure type's contribution is a **confidence-weighted average**
  across episodes, so a low-confidence short episode counts less than
  several confident ones.
- The 7 detected failure types map onto 6 vector dimensions;
  `repeated_action_loops` (no dedicated dimension) is split 30/70 into
  `path_efficiency` and 40/60 into `short_horizon_planning`, since
  oscillation is both a wasted-path problem and a poor-immediate-decision
  problem.
- `primary_weakness` = the dimension with the highest aggregated value;
  `secondary_weaknesses` = other dimensions above a 0.4 threshold, sorted
  descending. Both are derived from measured behavior, never chosen
  randomly.

## Limitations

- These are correlational behavioral signals, not causal diagnoses -- a
  high `path_inefficiency` score proves the route was much longer than
  optimal, not that the agent "doesn't understand" the shortest path.
- Thresholds are not calibrated against a labeled dataset of real
  failures; that would be a natural next step once enough real training
  runs exist to build one.
