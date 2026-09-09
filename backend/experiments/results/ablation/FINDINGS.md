# Phase 22 — Ablation Study: Pilot Results

**Status: preliminary pilot, same scale and same caveats as Phase 21's
FINDINGS.md — not a conclusive test.**

## Setup

Same shared budget as Phase 21 (grid_size=8, base density knobs 0.15, 8
curriculum steps, 8,000 timesteps/step = 64,000 total, 2 seeds), varying
only the ablation variant (`app/ml/curriculum/ablation.py`):

| Variant | Failure analysis used for targeting? | alpha (alignment) | beta (LP) | gamma (diversity) |
|---|---|---|---|---|
| A | Yes | 0.7 | 0.0 | 0.0 |
| B | No (blinded, uniform rotation) | 0.0 | 0.7 | 0.0 |
| C | Yes | 0.5 | 0.5 | 0.0 |
| D (full method) | Yes | 0.35 | 0.35 | 0.15 |

`delta` (excessive-difficulty penalty) held constant at 0.15 across all
four, since it's a safety guard, not one of the components under study.

## Results (descriptive only -- no significance testing)

| Variant | Final success rate (seed 7, 13) | Mean |
|---|---|---|
| A -- failure analysis only | 0.20, 0.00 | 0.10 |
| B -- learning progress only (blinded) | 0.10, 0.30 | 0.20 |
| C -- failure analysis + LP | 0.00, 0.00 | 0.00 |
| D -- full method | 0.00, 0.00 | 0.00 |

Full per-run data: `experiments/results/ablation/{variant}_seed{seed}.json`.

## Honest interpretation

**At this pilot scale, this ablation does not show the full method (D)
outperforming its simpler components.** If anything, B (learning-progress
only, with failure-analysis targeting deliberately blinded) had the
highest mean success rate of the four, and D/C tied at the bottom. With
n=2 seeds per variant and a 64k-timestep budget, this is consistent with
noise dominating any real effect at this scale -- the same conclusion
Phase 21 reached for the main strategy comparison, now confirmed to extend
to the ablation as well.

This result should NOT be read as "learning progress matters more than
failure analysis" or any other causal claim -- two seeds cannot support
that conclusion, and Section 1/58 explicitly prohibit overclaiming from
data like this. What it does confirm: the ablation harness itself (blinded
targeting via `_blind_weakness_vector`, controller weight variants A-D)
is wired correctly and runs to completion for all four conditions, and is
ready to be pointed at a larger seed count and training budget when more
compute is available -- which is what a real answer to "which component
matters" requires.
