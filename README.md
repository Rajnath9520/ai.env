# Failure-Driven Environment Synthesis for Adaptive AI Training

A research platform that investigates whether **conditioning procedural
environment generation on an RL agent's specific behavioral failure
modes** -- rather than on aggregate difficulty alone -- improves training
efficiency and out-of-distribution generalization, compared to random,
fixed, and generic-adaptive curricula.

## What this is

An RL agent (PPO) trains inside procedurally generated 2D GridWorld
mazes. After each curriculum step, the system:

1. Collects fresh trajectories from the just-trained policy.
2. Runs rule-based **failure analysis** on those trajectories (obstacle
   avoidance, dead-end recovery, exploration, path inefficiency,
   short/long-horizon planning -- see `docs/failure-analysis.md`).
3. Aggregates failures into a 6-dimensional **weakness vector**.
4. **Synthesizes candidate environments** whose parameters are adjusted
   specifically toward the detected weakness -- not just "harder" (see
   `docs/environment-generation.md`).
5. Scores candidates on weakness alignment, expected learning progress,
   and diversity, and selects one to train on next.
6. Measures learning progress, updates the curriculum, repeats.

This loop is compared against three baselines under *identical* algorithm,
training budget, and evaluation procedure: **random** environment
generation, a **fixed** easy-to-hard schedule, and a **generic adaptive**
curriculum that reacts only to overall success rate.

## Why it exists / how it differs from plain RL

This project does not propose a new RL algorithm -- it uses standard PPO
throughout. The contribution under investigation is entirely in the
**environment-generation and curriculum layer** around the agent: can the
system that decides *what the agent trains on next* be made smarter by
diagnosing failure causes, rather than just difficulty level?

## How it differs from generic automatic curriculum learning (ACL)

Most ACL approaches (including this project's own "generic adaptive"
baseline) adjust difficulty from a single aggregate signal -- usually
overall success rate -- without any model of *why* the agent is failing.
A generic ACL system can't distinguish "failing because of dead ends" from
"failing because of obstacle density"; it can only make things uniformly
easier or harder. This project's proposed method explicitly detects and
targets the *specific* weakness, which is the hypothesis under test, not
an assumed conclusion -- see `docs/research-methodology.md` and the honest,
pilot-scale results in `backend/experiments/results/*/FINDINGS.md`.

## Current status (read before trusting any numbers)

This is a working, fully-tested platform (320 tests: 312 backend + 8
frontend, all passing) -- but the research question itself has only been
run at **pilot scale** (2 seeds, ~64k timesteps per run) due to this
project's development environment's compute budget. At that scale, no
strategy showed a statistically distinguishable advantage, and a proper
statistical analysis (`app/ml/evaluation/statistics_analysis.py`)
correctly refuses to claim significance from that little data. **The
research question is not yet answered** -- the pipeline to answer it is
built, tested, and ready to be pointed at a larger seed count and training
budget. See `backend/experiments/results/research/FINDINGS.md` and
`STATISTICAL_ANALYSIS.md` for the full honest account.

## Project structure

```
project-root/
  backend/     FastAPI + ML engine (Python) -- see docs/architecture.md
  frontend/    Next.js dashboard (TypeScript/React)
  docker/      docker-compose + Dockerfiles -- see docker/README.md
  docs/        this documentation
```

## How to run it

**With Docker** (recommended once you have normal internet access to pull
base images -- see `docker/README.md` for a note on this sandbox's network
restrictions during development):

```bash
docker compose -f docker/docker-compose.yml up --build
```

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs

**Manually:**

```bash
# Backend
cd backend
pip install -r requirements.txt
export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/ailab"
python -c "from app.database.session import engine, init_db; init_db(engine)"
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 NEXT_PUBLIC_WS_BASE_URL=ws://localhost:8000 npm run dev
```

Requires a running PostgreSQL instance (`docker run -p 5432:5432 -e
POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ailab postgres:16` if you don't
have one).

## How to run experiments directly (no web UI)

```bash
cd backend
# Any one baseline / the proposed method:
python -m app.ml.curriculum.run_random_baseline
python -m app.ml.curriculum.run_fixed_baseline
python -m app.ml.curriculum.run_generic_adaptive_baseline
python -m app.ml.curriculum.run_failure_driven

# All four under identical conditions, multiple seeds:
python -m scripts.run_research_experiments

# The ablation study (which components of the method matter):
python -m scripts.run_ablation_study

# From a YAML config file:
python -m app.ml.experiment.run_experiment --config experiments/configs/your_config.yaml
```

## How to reproduce a specific experiment

Every run saves a reproducibility manifest to
`experiments/results/manifests/<name>.json` (seed, full config, git commit
+ dirty flag, training duration, checkpoint path -- see
`app/ml/experiment/reproducibility.py`). Reproduce it with:

```python
from app.ml.experiment.reproducibility import reproduce
result, warning = reproduce("experiments/results/manifests/<name>.json")
```

Two runs of the same `ExperimentConfig` produce byte-identical curriculum
histories (tested in `tests/ml/test_experiment_manager.py` and
`tests/ml/test_reproducibility.py`) -- every stochastic component is
explicitly seeded from `config.seed`.

## Tests

```bash
cd backend && python -m pytest tests/ -v          # 312 tests
cd frontend && npm test                            # 8 tests
```

See `docs/testing-audit.md` for the full coverage breakdown against the
project's required test checklist.

## Further reading

- `docs/architecture.md` -- system architecture and directory layout
- `docs/ml-pipeline.md` -- the full train-analyze-synthesize-select loop
- `docs/environment-generation.md` -- GridWorld generation and targeted synthesis
- `docs/failure-analysis.md` -- the rule-based failure detectors
- `docs/experiments.md` -- running and configuring experiments
- `docs/api.md` -- REST + WebSocket API reference
- `docs/deployment.md` -- Docker / manual deployment
- `docs/research-methodology.md` -- hypothesis, experimental design, statistical approach
