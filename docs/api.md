# API Reference

Base URL: `http://localhost:8000` (interactive docs at `/docs`, generated
directly from the Pydantic schemas in `app/schemas/experiment.py`).

## REST endpoints

All under `/api/experiments`:

| Method & path | Purpose |
|---|---|
| `POST /` | Create an experiment (status starts `pending`). Accepts `agent_mode` (`environment_only` default / `external` / `demo_ppo`) -- see `docs/architecture.md`'s Phase 28 section. |
| `GET /` | List all experiments |
| `GET /{id}` | Get one experiment |
| `POST /{id}/start` | For `environment_only`/`demo_ppo`: begin the run in the background. For `external`: synchronously prepares the initial environment and sets status `waiting_for_agent` (409 if not `pending`). |
| `POST /{id}/stop` | Guaranteed for `pending`; best-effort for `running` (see below) |
| `GET /{id}/status` | `{id, status, message?}` -- see Section 18's expanded status vocabulary |
| `GET /{id}/metrics` | Summary metrics (`agent_mode`, latest success rate, difficulty, weakness, LP -- the agent-performance fields are always `null` for `environment_only`) |
| `GET /{id}/episodes` | Per-curriculum-step representative episode rows (`demo_ppo`) / per-interaction episode rows (`external`) -- always `[]` for `environment_only` |
| `GET /{id}/trajectories` | Real query; populated for `external` (one row per completed episode), currently always `[]` for `demo_ppo` (not yet persisted -- see `docs/architecture.md`) |
| `GET /{id}/failures` | Real detected failure reports (`demo_ppo`/`external` only) |
| `GET /{id}/weaknesses` | Weakness vector snapshots (`demo_ppo`/`external` only) |
| `GET /{id}/environments` | Every environment generated/trained on, in every mode |
| `GET /{id}/environments/{env_id}/layout` | The concrete regenerated maze (obstacles/start/goal) for one environment |
| `GET /{id}/learning-progress` | Learning-progress records per environment (`demo_ppo` only) |
| `GET /{id}/generalization` | Validation/test/OOD evaluation results (`demo_ppo` only) |
| `GET /{id}/comparison?compare_to=1,2,3` | Side-by-side comparison across experiments, now including each one's `agent_mode` |
| `POST /{id}/agent/reset` | **`external` mode only.** Resets the environment and starts a new episode; returns `(observation, info)`. 409 if `agent_mode != "external"`. |
| `POST /{id}/agent/step` | **`external` mode only.** `{action: 0-3}` -> `(observation, reward, terminated, truncated, info)`, plus `episode_summary`/`weakness`/`next_environment` once an episode completes and enough trajectories have been analyzed. |
| `GET /{id}/agent/state` | **`external` mode only.** Honest, current connection/episode state (`connected: false` when no agent has called `/reset` yet). |

Status codes: `201` on create, `200` on success, `404` for a missing
experiment/environment, `409` for an invalid state transition (e.g.
starting an already-running experiment, or calling `/agent/*` on a
non-`external` experiment), `422` for validation errors (handled
automatically by Pydantic/FastAPI).

## WebSocket

`ws://localhost:8000/ws/experiments/{id}` streams JSON ticks while an
experiment is active. The tick shape is mode-aware (Section 19) -- ticks
always carry a `type` field, and only `demo_ppo` ticks ever carry
reward/episode/success-rate fields:

```json
// demo_ppo
{
  "episode": 1250,
  "reward": 0.42,
  "success_rate": 0.74,
  "environment_id": "env_0004",
  "primary_weakness": "long_horizon_planning",
  "learning_progress": 0.08,
  "training_status": "training"
}

// environment_only
{ "type": "environment_selected", "step_index": 4, "environment_id": "env_only_0004", "record": { "...": "..." } }

// external
{ "type": "weakness_detected", "experiment_id": 12, "weakness": { "...": "..." } }
```

`training_status` is `"training"` (per-episode ticks during a curriculum
step) or `"step_completed"` (once per curriculum step, with real
`learning_progress`). The connection closes cleanly when training ends.

## Cancellation semantics (important nuance)

`POST /{id}/stop` on a `pending` experiment guarantees it transitions to
`stopped` immediately. On a `running` experiment, it sets a cooperative
cancellation flag, but the current curriculum runners execute their step
loop synchronously and don't yet poll that flag mid-run -- stopping a
genuinely in-progress run is **best-effort, not guaranteed**, and the
response's `message` field says so explicitly rather than implying success.

## Error handling

All errors return a JSON body with a `detail` field (FastAPI's default
`HTTPException` shape). The service layer (`app/services/experiment_service.py`)
translates domain exceptions (`ExperimentNotFound`, `ExperimentStateError`)
into the appropriate HTTP status in the route layer -- routes never
construct error responses by hand.
