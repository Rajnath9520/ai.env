# Custom Adapters (Section 11)

`CustomAdapter` (`app/ml/project/adapters/custom_adapter.py`) is the
second `TrainingAdapter` implementation, alongside `CommandAdapter`
(Section 10, `docs/projects-and-adapters.md`). Where `CommandAdapter` runs
a project's own `train.py`/`evaluate.py` as configured shell commands,
`CustomAdapter` runs a project's own `.ai-env/adapter.py` -- a small
Python object with `train(config)`/`evaluate(config)`/`predict(config)`
methods.

## Why this isn't just `importlib.import_module()`

The master prompt's own sketch for this (Section 11) has the user's
adapter subclass ai.env's own `TrainingAdapter` and be imported directly.
That would mean loading and calling arbitrary user code **inside the same
process as the backend or an agent** -- a bug (or something malicious) in
that code could crash it, hang it, or read/corrupt state belonging to
ai.env itself. That's exactly what Section 59 means by "Executing
user-provided code is dangerous... Implement safeguards", and it's a
materially larger trust boundary than `CommandAdapter` already accepts by
running a project's own train.py as a *subprocess*.

So `CustomAdapter` runs `.ai-env/adapter.py` the same way `CommandAdapter`
runs train.py: as a subprocess, through the exact same `run_subprocess`
primitive (`app/ml/project/adapters/subprocess_runner.py`, extracted from
`CommandAdapter` so both share one implementation of real cancellation and
process-group-kill rather than two). Even *inspecting* which methods the
user's object defines happens in a subprocess (the `describe` action
below) -- there is no code path in `CustomAdapter` that imports the user's
module directly, not even to check what it looks like.

## The contract

`.ai-env/adapter.py` must define a module-level `get_adapter()` function
returning an object:

```python
# .ai-env/adapter.py
class MyAdapter:
    def train(self, config):
        dataset = config["variables"]["dataset"]
        # ... train the model, write a checkpoint ...
        return {"trained": True}

    def evaluate(self, config):
        # ... evaluate, however you like ...
        return {"metrics": {"accuracy": 0.94, "f1": 0.91}}

    def predict(self, config):
        return {"prediction": "..."}

def get_adapter():
    return MyAdapter()
```

Deliberately smaller than the master prompt's sketch: no ai.env imports,
no required base class, no particular method signatures beyond "takes one
JSON-serializable dict, returns one JSON-serializable value." This keeps a
custom adapter plain, dependency-free Python that works whether or not
ai.env is even installed anywhere near it.

- **Which methods you define are your capabilities.** Define only
  `train`; you get `Capability.TRAIN` and nothing else. All three are
  optional -- `_custom_adapter_runner.py`'s `describe` action checks with
  `callable(getattr(adapter, name, None))`, nothing more.
- **`config`** is `{"variables": {project, dataset, experiment, run,
  checkpoint, output}, "config": {...the full ai-env.yaml contract...}}`
  for `train`/`evaluate`; `{"input": <whatever was passed to predict()>}`
  for `predict`.
- **Return values must be JSON-serializable.** Checked explicitly
  (`json.dumps(result)`) before being reported as success, so "the
  adapter technically ran but returned something unusable" is a clear
  error, not a crash somewhere else.
- **`evaluate()` should return `{"metrics": {...}, ...}`** to show up
  anywhere metrics matter (progress trends, the Runs panel) -- see
  `CustomAdapter.evaluate()` for the exact shape expected (same fields as
  `EvaluationResult`: `metrics`, `samples`, `errors`, `artifact`,
  optionally `status`).

## Configuration

```yaml
adapter:
  type: custom            # default: "command"
  entrypoint: .ai-env/adapter.py   # default, can point elsewhere
  python: python           # must have whatever packages adapter.py imports

training:
  command: unused-for-custom-adapters   # still required -- see below
```

`AdapterConfigSchema`/`AdapterConfig` (`app/ml/project/config.py`) hold
this. `python` matters: it is **not** ai.env's own interpreter. The
backend/agent process almost certainly doesn't have the user's ML
libraries installed, so `CustomAdapter` never assumes `sys.executable` --
it always uses whatever `adapter.python` says (default `"python"`,
resolved via `PATH` at execution time, so a project's own activated venv
works if that's what's on `PATH` when the backend/agent started; an
absolute path like `/home/user/venv/bin/python` is the more reliable
choice).

**Known limitation, not fixed:** `training.command`/`training.entrypoint`
are still required, non-empty fields on `AIEnvConfig` even when
`adapter.type == "custom"`, since a pure custom-adapter project has no
shell command at all. Making that conditional was judged not worth the
risk to `TrainingConfig`'s existing validation and the ~40 tests already
covering it -- put any placeholder string there (the frontend's
Configuration form says so directly when "Custom" is selected).

## How capability discovery actually works, and the bug that shaped it

`CommandAdapter.capabilities` is a pure function of config fields (is
`training.command` set? is `evaluation.command` set?) -- it never touches
the filesystem, so it's identical whether the project ultimately executes
on this backend or through a registered Agent (`docs/local-agent.md`).

`CustomAdapter.capabilities` is **not** pure: it has to actually run the
`describe` action against the real `.ai-env/adapter.py` to know what's
there. The first version of this wired `ProjectService.update_config()`
and `start_run()` to call `build_adapter(config, project.root_path)
.capabilities`/`.supports(...)` directly on the backend, the same way it
already did for `CommandAdapter` -- which is correct for `CommandAdapter`
but silently wrong for `CustomAdapter` on an agent-backed project: the
backend would inspect `project.root_path` *on its own filesystem*, where
a remote project's files don't exist at all.

The fix (`app/services/project_service.py`): for `adapter.type ==
"custom"`, neither `update_config()` nor `start_run()` compute or
pre-check capabilities on the backend. `Project.capabilities` is simply
left empty for a custom adapter; `validate()` -- which already correctly
delegates the *entire* check to wherever the project actually lives, agent
or not -- is where a custom adapter's real capabilities surface, as
validation issues. `start_run()` skips its TRAIN pre-flight check for
custom adapters and lets actual execution (which does build the adapter
correctly, on the right machine) raise `CapabilityNotSupported` if
there's truly no `train()` -- still a clear, honest failure, just reported
as a failed run rather than a rejected request.
`tests/integration/test_local_agent_e2e.py::test_custom_adapter_runs_correctly_through_a_real_agent_process`
spawns a real separate agent process specifically to prove this -- it's
the one test that would have caught the original bug.

## Safety summary

- **Subprocess, never in-process.** A crash, hang, or infinite loop in a
  user's `adapter.py` can't take down the backend or agent process; it
  just makes that one subprocess time out or exit.
- **Real cancellation reaches user code.** Cancelling a run whose custom
  `train()` is in the middle of, say, `time.sleep(30)` or a long training
  loop actually kills it (process-group kill, same as `CommandAdapter`) --
  tested directly (`TestCancellation` in `tests/ml/test_custom_adapter.py`).
- **Secrets are redacted** from anything the adapter prints, same as
  `CommandAdapter` (`run_subprocess` handles this centrally).
- **Path traversal is still checked** -- `adapter.entrypoint` and
  `artifact.path` both resolve through `resolve_within()`.
- **Still arbitrary code execution for the project's own files.** None of
  this makes running a project's `.ai-env/adapter.py` safer than running
  its `train.py` already was -- it's exactly the same trust level as
  `CommandAdapter`, just with a different invocation shape. If you
  wouldn't run a project's own training script, don't register that
  project here either.
