# Running Experiments

## Configuration

`ExperimentConfig` (`app/ml/experiment/config.py`) fully determines one
run: algorithm (only `"PPO"` supported), seed, environment parameters,
curriculum type, training budget, and generalization-evaluation settings.

It loads YAML in two shapes:

**This project's flat canonical shape** (round-trips exactly):
```yaml
name: my_experiment
curriculum_type: failure_driven
seed: 42
grid_size: 10
n_curriculum_steps: 20
timesteps_per_step: 10000
```

**A nested shape** (mapping `training.episodes` to a total-timestep budget
spread across `n_curriculum_steps`):
```yaml
algorithm: PPO
curriculum:
    type: failure_driven
environment:
    type: gridworld
    size: 10
training:
    episodes: 10000
evaluation:
    frequency: 500
seed: 42
```

## Running one experiment

```bash
python -m app.ml.experiment.run_experiment --config path/to/experiment.yaml
```

Or via the API: `POST /api/experiments` (create, status `pending`) then
`POST /api/experiments/{id}/start` (runs in the background; poll
`GET /api/experiments/{id}/status`).

Or directly in Python:
```python
from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.manager import ExperimentManager

config = ExperimentConfig(name="my_run", curriculum_type="failure_driven", seed=42)
result = ExperimentManager(config).run()
```

Every run: (1) trains under the chosen curriculum strategy, (2) by default
evaluates generalization (validation/test/OOD splits, strictly seed-disjoint
from training -- see `app/ml/evaluation/generalization.py`), (3) saves a
checkpoint and a reproducibility manifest.

## Reproducibility

Every stochastic component (candidate generator, environment synthesizer,
split generator, PPO itself) is explicitly seeded from `config.seed` by
`ExperimentManager`. Two runs of the same config produce byte-identical
curriculum histories -- tested directly, not just assumed. See
`app/ml/experiment/reproducibility.py` and the README's reproduction
section.

## The controlled 4-strategy comparison

`scripts/run_research_experiments.py` runs random / fixed /
generic_adaptive / failure_driven under **identical** algorithm, budget,
evaluation procedure, and (varying) seed -- only the curriculum strategy
differs. Results and their honest interpretation:
`backend/experiments/results/research/FINDINGS.md`.

## The ablation study

`scripts/run_ablation_study.py` isolates which component of the proposed
method matters: failure-analysis-only, learning-progress-only (with
targeting deliberately content-blinded), both, and both+diversity (the
full method). Results: `backend/experiments/results/ablation/FINDINGS.md`.

## Honesty note

Both scripts above have only been run at **pilot scale** (2 seeds,
~64,000 timesteps per run) in this project's development environment. At
that scale, `app/ml/evaluation/statistics_analysis.py::StatisticalAnalyzer`
correctly refuses to run a significance test (its `min_n_for_test=3`
floor was not met) -- see
`backend/experiments/results/research/STATISTICAL_ANALYSIS.md`. **No
claim is made that the proposed method outperforms the baselines** at
this project's current stage of experimentation; the pipeline to properly
answer that question is built and tested, and needs a larger seed count
and training budget to run for real.
