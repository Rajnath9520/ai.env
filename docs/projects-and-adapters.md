# Projects and Training Adapters (Mode B)

This covers the first slice of the master prompt's "Mode B" -- letting a
user register **their own** AI project (built in VS Code, outside ai.env)
rather than only using the built-in GridWorld/PPO environment ("Mode A",
see `architecture.md`). It is a domain-model/execution-abstraction layer
only; see [What this phase does not include](#what-this-phase-does-not-include)
before assuming more than is actually wired up.

## The contract: `ai-env.yaml`

The one thing ai.env needs from a user's project is a confirmed
`AIEnvConfig` (`app/ml/project/config.py`), matching the nested YAML shape:

```yaml
project:
  name: face-recognition
  task: face_recognition
  framework: pytorch
training:
  entrypoint: train.py
  command: python train.py --dataset {dataset}
  working_directory: .
  timeout_seconds: 86400
evaluation:
  entrypoint: evaluate.py
  command: python evaluate.py --dataset {dataset}
  metrics_file: outputs/metrics.json
artifact:
  path: outputs/best_model.pt
  type: pytorch_checkpoint
dataset:
  path: ./dataset
curriculum:
  strategy: failure_driven
optimization:
  max_iterations: 10
  target_metric: f1
  target_value: 0.95
```

`command` strings may reference seven placeholders, substituted by
`substitute_variables()` (`app/ml/project/adapters/base.py`): `{project}`,
`{dataset}`, `{experiment}`, `{run}`, `{checkpoint}`, `{output}`,
`{config}`. Any other `{name}` raises immediately rather than executing a
mangled command.

Every dataclass validates itself in `__post_init__` (empty commands,
unknown curriculum strategies, an `evaluation.metrics_source="file"` with
no `metrics_file`, etc.) -- an invalid contract fails to construct at all,
it never gets partially stored.

## Scanning: suggestions, never facts

`ProjectScanner` (`app/ml/project/scanner.py`) inspects a project directory
and returns a `ScanResult` -- language, framework (from dependency files,
falling back to top-level source imports), and four *candidate* lists
(training/evaluation files, dataset dirs, existing artifact files). It
never picks a single "the" entrypoint; `train.py` being present is a
candidate, not a decision. An empty or unrecognized project degrades to
`None`/`[]`, never a guess (this is also why `_detect_gpu()` reports
`False, None` rather than pretending a GPU is present when `nvidia-smi`
isn't found).

`Project.detected_config` stores the raw `ScanResult.to_dict()` from the
last scan. `Project.config_yaml` is only populated once the user submits a
confirmed `AIEnvConfig` via `PUT /api/projects/{id}/config` -- routes and
the frontend must keep these visibly distinct (Section 7).

## Capabilities and `TrainingAdapter`

`TrainingAdapter` (`app/ml/project/adapters/base.py`) is the universal
interface: `validate_project`, `train`, `evaluate`, `get_metrics`,
`get_artifact`, `load_artifact`, `predict`, `cleanup`. A concrete adapter
only has to declare which `Capability` values it actually implements
(`TRAIN`, `EVALUATE`, `PREDICT`, `CHECKPOINT`, `DATASET_GENERATION`,
`ENVIRONMENT_INTERACTION`, `TRAJECTORY_COLLECTION`) -- declaring one
without overriding the matching method is a bug (raises
`NotImplementedError`), not a silently-degraded feature.

`CommandAdapter` (`app/ml/project/adapters/command_adapter.py`) is the
first, and so far only, implementation -- it runs the configured
training/evaluation commands as subprocesses:

- **Evaluation metrics** are collected exactly one of two explicit ways
  (Section 24): `metrics_source: file` reads a configured JSON
  `metrics_file`; `metrics_source: stdout` parses the last non-empty
  stdout line as JSON. Configuring neither is valid -- the command still
  runs, but the result honestly reports `metrics: {}` with a warning
  explaining why, never a fabricated value.
- **Paths** (`working_directory`, `artifact.path`) are resolved through
  `resolve_within()` (`app/ml/project/security.py`), which raises
  `PathTraversalError` if the configured relative path would land outside
  the project root.
- **Captured stdout/stderr** is passed through `redact_secrets()` before
  being stored anywhere, matching values like `API_KEY=...` or
  `token: ...`.
- `get_artifact()` reports whether the configured artifact path exists yet
  -- `exists=False` before the first successful run is the honest answer,
  not an error.

`app/ml/project/adapters/registry.py`'s `build_adapter()` picks between
`CommandAdapter` and `CustomAdapter` (Section 11: a project's own
`.ai-env/adapter.py`, run in its own subprocess, never imported) based on
`config.adapter.type` -- see [`docs/custom-adapters.md`](./custom-adapters.md)
for the full design, including a real cross-process bug its own
capability-discovery model surfaced and how it was fixed.

## API surface

`app/api/routes/projects.py`, thin controllers delegating to
`ProjectService` (`app/services/project_service.py`):

| Route | Effect |
|---|---|
| `POST /api/projects/register` | create a `Project` row (`root_path` + optional `name`/`agent_id`), status `draft` |
| `GET /api/projects` / `/{id}` | list / fetch one project |
| `PUT /api/projects/{id}/agent` | switch between direct execution (`agent_id: null`) and a registered Local Agent (Section 5, docs/local-agent.md) |
| `POST /api/projects/{id}/scan` | run `ProjectScanner`, store `detected_config`, status -> `scanned` |
| `GET`/`PUT /api/projects/{id}/config` | read / confirm the `AIEnvConfig` contract, status -> `configured` |
| `POST /api/projects/{id}/validate` | run `CommandAdapter.validate_project()` (cheap, no subprocess), status -> `validated`/`validation_failed` |
| `POST /api/projects/{id}/runs` | start a new `ProjectRun` (train, then evaluate if configured), off the request thread |
| `GET /api/projects/{id}/runs` / `/{run_id}` | list / fetch runs for a project |
| `POST /api/projects/{id}/runs/{run_id}/cancel` | best-effort cancellation of an in-flight run |

Editing the config after a validation clears `validation_result` (a stale
"ok" from before the edit must never be shown as current).

## Runs: actually training a Mode B project

`ProjectService.start_run()` mirrors `ExperimentService.start()`'s
`BackgroundTasks` pattern exactly: it creates a `ProjectRun` row (status
`running_training`), commits it, and hands `_run_project_run_in_background`
to FastAPI's `BackgroundTasks` -- the route itself never blocks on the
subprocess (Rule 11). That function opens its own DB session (the request's
session is closed by the time it runs), builds a fresh `CommandAdapter`
from the project's stored `config_yaml`, and:

1. Resolves `{dataset}` to an actual filesystem path and `{output}` to a
   fresh, ai.env-managed per-run directory before substituting them into
   the training command -- an unresolved `{dataset}` silently becoming an
   empty string was an actual bug caught by the first end-to-end test of
   this path (`shlex.split` drops a trailing empty argument, so
   `--dataset {dataset}` with no dataset configured turned into a bare,
   argument-less `--dataset` and broke argparse).
2. Runs `adapter.train()`. A non-`"success"` result ends the run there
   (`failed`, or `cancelled` if the cancel flag was set) -- evaluation
   never runs against a training command that didn't succeed.
3. If training produced an artifact, archives a copy via
   `archive_run_artifact` (`app/ml/project/storage.py`) into
   `{AI_ENV_STORAGE_ROOT}/projects/{project_id}/runs/{run_number}/` --
   outside the user's project entirely, so the *next* run overwriting the
   project's own `outputs/best_model.pt` in place never loses this one's
   copy. `ProjectRun.managed_artifact_path` is this archived path, kept
   distinct from `train_result["artifact"]["path"]` (the live, in-project
   path that may not exist by the time anyone looks at it later).
4. Runs `adapter.evaluate()` if the project's config declares
   `Capability.EVALUATE`, and sets the run's final status from whichever
   of `train`/`evaluate` actually ran, treating `cancel_event.is_set()` as
   authoritative over whatever the subprocess itself returned (avoids
   fragile "was this really a cancellation" inference from exit codes).

**Cancellation is real**, not aspirational: `CommandAdapter._run()`
(`app/ml/project/adapters/command_adapter.py`) polls the subprocess via
`Popen.communicate(timeout=...)` -- the deadlock-safe idiom the
`subprocess` docs recommend for pipes -- instead of blocking on a single
`subprocess.run(timeout=...)`, so a `threading.Event` can interrupt a run
in flight. The process starts in its own session (POSIX
`start_new_session=True`) so cancelling or timing out kills the whole
process group, not just the top-level shell -- verified by a test that
spawns a grandchild process and confirms it doesn't survive cancellation
(Section 20: "Never kill unrelated processes. Only manage processes
launched by AI.env").

## Progress: honest run-over-run trends (Section 41, generalized)

`GET /api/projects/{id}/progress` (`app/ml/project/progress.py`'s
`compute_progress`) turns a project's completed runs into metric trends,
using only numbers the project's own evaluation command actually reported.
It deliberately does not attempt Sections 25-30's task-specific weakness
vectors (`lighting`, `occlusion`, and so on) -- those need a structured,
per-condition evaluation harness that nothing in Mode B provides; a
`CommandAdapter` evaluation only ever yields a flat `{metric: value}` dict.
Inventing a breakdown on top of that would be exactly the fabrication
Section 88 forbids.

Two things it's careful never to assume:

- **Which direction is "better".** A metric could be accuracy
  (higher-is-better) or loss (lower-is-better), and nothing here can tell
  which. So it never labels a delta as an "improvement" or "regression" --
  only the raw `delta_from_previous`. The one place it *can* make a
  meaningful claim is a configured `target_value`: `distance_to_target` is
  `abs(latest - target)`, which holds regardless of direction, and
  `moved_closer_to_target` compares that distance run-over-run rather than
  the raw value.
- **That "completed" means "has metrics".** A run can complete with no
  metrics at all (Section 88's honest empty-metrics case, when neither
  `metrics_file` nor `metrics_source` is configured) -- `completed_run_count`
  still counts it, even though it contributes no points to any trend.

## Data model

Two new tables (`app/models/orm.py`):

- `projects`: `root_path`, `source` (`local`/`upload`/`built_in`),
  `language`/`framework`/`task`, `status`, `capabilities` (JSONB list of
  capability names), `detected_config`/`config_yaml`/`validation_result`.
- `project_runs`: `project_id` (FK, `CASCADE` -- a run has no meaning
  detached from the project it ran against, unlike `Experiment`'s link
  below), `run_number` (per-project, starts at 1), `status`,
  `train_result`/`evaluation_result` (JSONB -- `TrainResult`/
  `EvaluationResult.to_dict()` directly, so arbitrary per-task metrics
  live in `evaluation_result["metrics"]` per Section 23/58 rather than
  forcing GridWorld-shaped columns like `TrainingRun`/`Evaluation` have),
  `managed_artifact_path`, `error`.

`experiments.project_id` is a new, nullable FK to `projects`
(`ondelete="SET NULL"`) -- every existing GridWorld/PPO experiment has
`project_id IS NULL` and stays that way; nothing about Mode A changed.

## What this phase does not include

Sections 8-10/16/19-20 of the master prompt are implemented and working
end-to-end: a project can be registered, scanned, confirmed, validated,
**and actually trained and evaluated** through `POST /api/projects/{id}/runs`,
off the request thread, with real (process-group) cancellation and
per-run artifact archiving. What's still missing:

- **Section 5's Local Agent is implemented, in a separate doc.** A
  project can now execute either directly (backend, same machine) or
  through a registered `agent_app/` process on a different machine --
  see [`docs/local-agent.md`](./local-agent.md) for the full design and
  its own remaining gaps (no TLS, no install story).
- **Single-process, in-memory cancellation/job-tracking.** `_cancel_flags`
  (a `dict[int, threading.Event]`) doesn't survive a backend restart and
  doesn't coordinate across multiple backend processes, identical to
  `ExperimentService`'s existing limitation for GridWorld -- not a new gap
  introduced here, just not fixed either.
- **Custom adapters (Section 11) are implemented** -- see
  `docs/custom-adapters.md`. Known limitation there: `training.command`
  stays a required-but-unused field even for a pure custom adapter.
- **No failure analysis / weakness vectors / curriculum / generation for
  Mode B projects** (Sections 25-40) -- those still exist only for
  GridWorld. "Progress" (above) is the honest subset of Section 41 that a
  flat metrics history actually supports; it is not a weakness vector and
  doesn't try to be. `curriculum.strategy` and `optimization.max_iterations`
  are accepted, stored, and editable, but nothing acts on them yet -- a run
  is exactly one train+evaluate cycle, triggered by hand, not an automatic
  adaptive loop.
- **`{checkpoint}` and `{config}` command variables are always empty.**
  `{project}`, `{dataset}`, `{experiment}`, `{run}`, and `{output}` all
  resolve to real values now; there's no "previous checkpoint" or
  generated per-run config file concept yet for the other two.

## Frontend

`frontend/app/projects/` (list, register, detail-with-scan/configure/
validate/runs) and the corresponding `api.*` methods in `lib/api-client.ts`
cover the full workflow this phase's API supports, including actually
starting and watching a run. The detail page's "Configuration" panel is
deliberately a single page with labeled sections rather than Section 49's
literal numbered wizard, matching this frontend's existing convention (see
`app/experiments/new/page.tsx`); its "Prefill from scan" button turns the
latest `ScanResult` into an editable draft (`lib/project-config.ts`'s
`suggestConfigFromScan`) without ever submitting it automatically. The
"Runs" panel polls while any run is non-terminal (same pattern as
`app/experiments/[id]/page.tsx`), shows each run's status, metrics once
evaluation finishes, a stderr tail on failure, and the archived artifact
path -- and offers Cancel only while a run is actually in flight. A
"Progress" panel below it renders the target-metric trend as a `recharts`
line chart (styled like `app/dashboard/page.tsx`'s existing charts) plus a
plain table of every other metric's latest value and delta -- deltas are
shown in a single neutral color, never green/red, since this page can't
know which direction is "better" for an arbitrary metric either.

The project detail page's Overview panel also has an "Executes on" select
(Backend (local) or any registered Local Agent, with its last-known
status) that calls `PUT /api/projects/{id}/agent`; `/agents` is where
agents get registered and pinged (see [`docs/local-agent.md`](./local-agent.md)).

Section 85's honest-support-model framing applies here in full: a
registered project with a stable training command, evaluation command,
and artifact path gets orchestrated end to end -- trained, evaluated, its
artifact archived, its metrics trended across runs -- either directly or
through a Local Agent. It does not get automatic failure analysis or
adaptive dataset/environment generation; nothing here claims ai.env
"understands" an arbitrary model, only that it can run and measure one
through a stable contract.
