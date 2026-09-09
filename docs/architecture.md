# Architecture

## Phase 28: ai.env is a model-agnostic environment platform

**ai.env is fundamentally an environment-generation and environment-analysis
platform. It does not secretly create or require an AI model just because
an experiment is created.** Every experiment has an explicit `agent_mode`
(`app/ml/experiment/config.py`), persisted on the `Experiment` row, that
decides what runs:

```
                     ┌─────────────────────────┐
                     │          ai.env          │
                     │ AI Environment Generator │
                     └────────────┬────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
                ▼                                   ▼
      ┌───────────────────┐               ┌───────────────────┐
      │ Environment Core   │               │   Agent Layer      │
      │ (generate/validate/│               │  (optional model)  │
      │  analyze/rank)     │               └─────────┬─────────┘
      └─────────┬─────────┘               ┌──────────┼──────────┐
                │                          │          │          │
                ▼                     None (default) External  Demo PPO
          Environment                      │          │          │
                │                          └──────────┴────┬─────┘
                ▼                                     Agent Interface
          (exported, no agent involved)                     │
                                                       Trajectories
                                                              │
                                                      Failure Analysis
                                                              │
                                                       Weakness Vector
                                                              │
                                              Adaptive Environment Generator
                                                              │
                                                      New Environment
```

- **`environment_only`** (default) -- `EnvironmentOnlyGenerator`
  (`app/ml/environments/environment_only_generator.py`) generates,
  validates, and analyzes (`EnvironmentAnalyzer`,
  `app/ml/analysis/environment_analyzer.py`) a sequence of environments.
  No `AgentTrainer`, no PPO, no training, no fake reward/episode/success
  metrics. `ExperimentManager._run_environment_only()` is the only code
  path exercised.
- **`external`** -- ai.env owns the environment, trajectory collection,
  failure analysis, and adaptive generation; the external agent owns its
  own model and drives everything through a plain HTTP reset/step contract
  (`app/api/routes/external_agent.py`, backed by
  `app/ml/agents/external_session.py`). There is no "run to completion" for
  this mode -- `ExperimentManager.run()` refuses to execute it at all (see
  its docstring); `ExperimentService.start()` only generates the initial
  environment and marks the experiment `waiting_for_agent`.
- **`demo_ppo`** -- the project's original PPO/curriculum training flow,
  unchanged in behavior, but only reachable when a user explicitly selects
  "Demo Agent -- PPO". This is the *only* code path that imports
  `stable_baselines3.PPO` outside of an external agent's own process (which
  is none of ai.env's business by design). `ExperimentManager` imports
  `AgentTrainer` lazily, inside `_run_demo_ppo()`, specifically so that
  importing the manager module for the other two modes never pulls SB3 in.

`AgentInterface` (`app/ml/agents/interface.py`) is the small
reset/act/close `Protocol` any ai.env-hosted agent implements --
`DemoPPOAgent` (`app/ml/agents/demo_ppo.py`) is the only concrete
implementation today, but nothing about the environment core assumes PPO:
a future DQN/SAC/LLM/robotics/API-based agent would implement the same
interface without touching `app/ml/environments` or `app/ml/analysis` at
all. The genuinely *external* agent path deliberately does **not**
implement `AgentInterface` from ai.env's side (see
`app/ml/agents/external_adapter.py`'s docstring) -- wrapping it would mean
ai.env owns/drives the external model, which Section 9 of the spec
explicitly forbids.

Guardrails against regressing this (Section 26/30):
- `ExperimentConfig.agent_mode` defaults to `"environment_only"` and is
  validated against `VALID_AGENT_MODES`; `algorithm` is retained only for
  backward compatibility and never gates execution.
- `tests/ml/test_agent_modes.py` and `tests/db/test_external_agent.py`
  assert -- by monkeypatching `stable_baselines3.PPO` itself, not just
  checking output shape -- that neither `environment_only` nor `external`
  ever instantiates it, through both the manager/session layer and the
  full HTTP API (`tests/api/test_agent_modes_api.py`).
- `TestNoHiddenCoupling` statically scans
  `environment_only_generator.py` and `external_session.py` for any
  executable reference to `PPO`/`AgentTrainer`/`stable_baselines3`.

## Layered overview

```
Frontend (Next.js)  <-- REST + WebSocket -->  FastAPI (app/api)
                                                    |
                                          app/services (ExperimentService)
                                                    |
                              app/ml (fully independent of the web layer)
                          ExperimentManager -> curriculum runners -> PPO/env/analysis
                                                    |
                                          PostgreSQL (app/database, app/models)
```

The ML engine (`backend/app/ml/`) has **no import of FastAPI, SQLAlchemy,
or anything web-related**. It's a plain Python package that can run
standalone (`python -m app.ml.curriculum.run_failure_driven`) or be driven
by the API layer via `ExperimentService`. This separation is deliberate --
see each curriculum runner's module docstring for "keep the ML engine
independent of the web application."

## Execution model

Training (`demo_ppo`) and environment-only generation are both long-running
enough that the API never runs either synchronously inside a request.
`POST /api/experiments/{id}/start` checks `agent_mode` first:

- `environment_only` / `demo_ppo`: flips status to `running` and schedules
  the run as a FastAPI `BackgroundTask`, which opens its own database
  session (the request's session is already closed by the time the run
  would execute). Live progress streams over a WebSocket via an in-process
  pub/sub broadcaster (`app/workers/broadcaster.py`).
- `external`: has no "run to completion" -- `start()` synchronously
  generates the initial environment (fast, no model of any kind) and marks
  the experiment `waiting_for_agent`. All further progress happens through
  `POST /api/experiments/{id}/agent/reset` and `.../agent/step`, called by
  the external agent's own process, publishing the same kind of ticks over
  the same WebSocket.

## Directory structure (as built)

```
backend/
  app/
    api/
      deps.py              # DB session dependency
      routes/
        experiments.py     # REST endpoints (Section 24)
        external_agent.py  # Phase 28: reset/step/state for agent_mode="external"
        websocket.py        # live progress stream, mode-aware ticks
    core/
      config.py            # Settings (DATABASE_URL)
    database/
      session.py           # engine/session management
      repository.py        # ExperimentResult -> DB rows
    models/
      orm.py               # SQLAlchemy models (11 tables)
    schemas/
      experiment.py        # Pydantic request/response schemas
    services/
      experiment_service.py # orchestration: CRUD, start/stop, background run
    workers/
      broadcaster.py        # in-process pub/sub for live ticks
    ml/
      environments/         # GridWorld generator/validator/config/synthesizer
                            #  + environment_only_generator.py (Phase 28, agent-free)
      training/              # AgentTrainer, metrics callback, env factory (demo_ppo only)
      analysis/               # trajectory logging, failure analysis, weakness vectors
                              #  + environment_analyzer.py (Phase 28, agent-free structural metrics)
      agents/                 # Phase 28: AgentInterface, DemoPPOAgent, external_session
      curriculum/             # 4 strategies + controller + full adaptive loop + ablation
      evaluation/             # learning progress, generalization, metrics, statistics
      experiment/             # ExperimentConfig (agent_mode), ExperimentManager, reproducibility
    main.py                   # FastAPI app
  scripts/
    run_research_experiments.py
    run_ablation_study.py
  tests/
    ml/   db/   api/
  experiments/
    configs/   results/

frontend/
  app/            # Next.js App Router pages (one per dashboard screen)
  components/     # Panel/Sidebar/GridWorldRenderer/etc.
  hooks/          # useExperimentSocket, useAnimatedPath
  lib/            # api-client, grid-pathfinding
  types/          # TypeScript types mirroring the backend schemas

docker/
  backend.Dockerfile
  frontend.Dockerfile
  docker-compose.yml
```

## Data flow (one curriculum step, failure-driven method, `agent_mode="demo_ppo"`)

1. `AdaptiveCurriculumRunner` asks `CandidateGenerator` for N candidate
   environments, built from the current `WeaknessVector` via
   `EnvironmentSynthesizer`.
2. `CurriculumController` scores each candidate (alignment + expected
   learning progress + diversity - excessive-difficulty penalty) and picks
   the best.
3. `AgentTrainer.train()` trains PPO on the selected environment,
   publishing live ticks via `progress_callback` as it goes.
4. `AgentTrainer.evaluate()` measures post-training performance;
   `LearningProgressEvaluator` computes the before/after delta.
5. `TrajectoryLogger` collects fresh trajectories on similar environments;
   `FailureAnalyzer` scores each of 7 failure types; `WeaknessVectorBuilder`
   aggregates them into the next weakness vector.
6. Repeat from step 1.

`ExperimentManager` wraps this whole loop (or one of the three simpler
baseline runners) behind `_run_demo_ppo()`, only reachable when
`agent_mode="demo_ppo"`.

## Data flow (`agent_mode="environment_only"`)

1. `ExperimentManager._run_environment_only()` builds an
   `EnvironmentOnlyGenerator` with the configured `generation_strategy`
   (`diverse` / `progressive` / `random`).
2. For each of `n_curriculum_steps` environments: generate one or more
   candidate `GridWorldConfig`s (agent-free -- either jittered around the
   base config, or from the existing `FixedCurriculum`/`RandomCurriculum`
   schedules reused purely as parameter generators), run
   `GridWorldGenerator`/`EnvironmentValidator`, characterize the result with
   `EnvironmentAnalyzer`, and select the best candidate.
3. `progress_callback` emits `candidate_generated` / `candidate_validated`
   / `environment_selected` events over the same WebSocket the training
   path uses -- never the `training_started`/`episode_updated` vocabulary.
4. `attach_environment_only_results()` (`app/database/repository.py`)
   persists each selected environment as an `Environment` row only -- no
   `TrainingRun`, `Episode`, or `AgentWeakness` row is ever created for this
   mode.

## Data flow (`agent_mode="external"`)

1. `ExperimentService.start()` calls `ExternalAgentSession.prepare()`,
   which generates and persists the initial `Environment` row and sets
   status `waiting_for_agent`. No model exists yet.
2. The external agent process calls `POST .../agent/reset` ->
   `ExternalAgentSession.reset()` constructs a real `GridWorldEnv` from the
   current config and returns `(observation, info)`; status moves to
   `agent_connected` then `interacting`.
3. The external agent calls `POST .../agent/step` repeatedly ->
   `ExternalAgentSession.step()` steps the real Gymnasium environment and
   accumulates a real `TrajectoryStep` per call.
4. On episode end, the accumulated trajectory is summarized
   (`app/ml/analysis/trajectory.py`) and passed to the same
   `FailureAnalyzer` used by `demo_ppo`; results are persisted as real
   `Episode`/`Trajectory`/`Failure` rows under a `TrainingRun` linked to
   this experiment (reused as a generic "one interactive run" grouping, not
   specific to PPO).
5. Once `trajectories_per_analysis` episodes have been analyzed,
   `WeaknessVectorBuilder` produces a real `WeaknessVector`, persisted as an
   `AgentWeakness` row, and `CandidateGenerator` (the same one `demo_ppo`
   uses) generates and ranks the next environment -- persisted as a new
   `Environment` row and swapped in for the next `reset()`.
6. Steps 2-5 repeat for as long as the external agent keeps calling
   reset/step. `GET .../agent/state` reports the live, honest connection
   state (`AgentStateRead`) for the frontend's External Agent page.

## Database schema

11 tables (`app/models/orm.py`): `users`, `experiments` (now with
`agent_mode`), `environments`, `environment_parameters`, `training_runs`,
`episodes`, `trajectories`, `failures`, `agent_weaknesses`,
`learning_progress`, `evaluations`, `model_checkpoints`. `environment_only`
mode populates only `experiments`/`environments`; `external` mode populates
`experiments`/`environments`/`training_runs`/`episodes`/`trajectories`/
`failures`/`agent_weaknesses` (no `learning_progress`, `evaluations`, or
`model_checkpoints` -- those describe a trained policy, which doesn't exist
in this mode); `demo_ppo` populates all of them, unchanged from before
Phase 28. See the module's docstring for further design notes (e.g. why
raw per-step trajectories are never persisted, only summaries).

## Known architectural limitations

- No Alembic migrations yet -- tables are created via
  `Base.metadata.create_all()` (`init_db()`), sufficient for development
  but not for a production schema-migration workflow. This includes the
  Phase 28 `agent_mode` column: existing rows created before Phase 28 need
  a manual backfill (`UPDATE experiments SET agent_mode = 'demo_ppo'` for
  any that were actually trained under the old implicit-PPO behavior) if
  ever run against a pre-existing database rather than a fresh one.
- The WebSocket broadcaster, experiment cancellation flags, and Phase 28's
  `ExternalAgentManager` (in-process session registry keyed by
  experiment_id) are all in-process only -- a multi-worker deployment would
  need a shared store (e.g. Redis) for all three, not just the two that
  predate this phase.
- `Trajectory` rows for `demo_ppo` are defined in the schema but not yet
  populated by the repository layer -- only summarized `Episode`/`Failure`
  rows are written per curriculum step (see `app/database/repository.py`'s
  docstring). Phase 28's `external` mode is the exception: its
  `ExternalAgentSession` does populate real `Trajectory` rows per episode,
  since the interactive loop naturally produces one trajectory at a time
  rather than a batch per curriculum step.
- `EnvironmentOnlyGenerator`'s "diverse" strategy scores diversity purely
  from the three continuous generation parameters (obstacle density,
  dead-end probability, path complexity); it does not consider grid
  topology directly. This mirrors the same simplification already made by
  `CandidateGenerator`'s diversity scoring for `demo_ppo`/`external` modes.
- The external-agent HTTP contract assumes one agent connects to one
  experiment at a time; concurrent agents against the same experiment_id
  would race on the same in-memory `ExternalAgentSession`.
