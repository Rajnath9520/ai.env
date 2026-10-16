# Local Agent (Section 5)

`agent_app/` is a second, genuinely separate FastAPI application from the
main backend (`app/`) -- a different process, run on a different port,
typically on a different machine than the backend. It exists so a
project's `root_path` can refer to a path the *main backend* has no
filesystem access to at all.

**Naming note:** this "Agent" (an `Agent` database row, `agent_app/`,
`/api/agents/*`) is unrelated to the RL "external agent" concept elsewhere
in this codebase (`app.ml.agents`, `agent_mode="external"`, the
`AgentInterface`/`ExternalAgentSession` classes for GridWorld). Those
describe something that *owns a trained policy* and interacts with an
environment; this Local Agent is a *process*, and owns nothing but
whatever machine it happens to run on. Unfortunate overlap in vocabulary,
not overlap in concept.

## Running one

```bash
cd backend
AI_ENV_AGENT_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo "token: $AI_ENV_AGENT_TOKEN"   # you'll enter this when registering the agent
AI_ENV_AGENT_NAME=my-laptop AI_ENV_AGENT_OUTPUT_ROOT=./agent-data \
  uvicorn agent_app.main:app --host 0.0.0.0 --port 8765
```

An agent started **without** `AI_ENV_AGENT_TOKEN` will answer `/health` but
refuse every other request with 503. See [Authentication](#authentication).

It needs the same Python environment as the backend (it imports
`app.ml.project.*` directly -- see below) but nothing else: no
`DATABASE_URL`, no Postgres, no knowledge of experiments, projects, or
runs as domain concepts.

## Why this reuses `app.ml.project.*` instead of reimplementing it

Every filesystem/execution primitive the agent exposes --
`ProjectScanner`, `AIEnvConfig`, `build_adapter`/`CommandAdapter`,
`resolve_within`/`redact_secrets` -- is the exact same code the backend
uses for a same-machine ("no agent") project. This was possible without
any rework because `app/ml/project/` was written from the start with zero
imports of FastAPI or SQLAlchemy (see that package's own docstrings) --
Rule 2 ("Reuse existing code") applied directly, rather than "the Local
Agent phase reuses the earlier phase's code" being a retrofit. A project
behaves identically whether ai.env talks to it directly or through an
agent; only *where* the code runs differs.

`agent_app/run_manager.py` is the one genuinely new piece: in-memory
bookkeeping for in-flight runs (item 9 in Section 5's list, "process
manager"). Section 5 is explicit that "the Local Agent should NOT contain
core business logic" -- run numbering, versioning, and persistence all
stay in the backend's `ProjectRun` table; the agent doesn't have a
database at all. If the agent process restarts mid-run, that run's
bookkeeping is simply gone -- the backend, which polls this state rather
than trusting it blindly, sees an unknown run id and reports the run
failed rather than guessing at its fate.

## The HTTP surface

Every route except `/health` requires `Authorization: Bearer <token>`.

| Route | Effect |
|---|---|
| `GET /health` | `{status, agent_name, version, auth_required, token_configured}` -- **the only unauthenticated route** |
| `GET /auth-check` | succeeds only with a valid token; lets the backend verify credentials without running anything |
| `POST /scan` | wraps `ProjectScanner().scan()` |
| `POST /validate` | wraps `CommandAdapter.validate_project()` |
| `POST /runs` | starts train(+evaluate) in a background thread, returns immediately |
| `GET /runs/{id}` | current status/train_result/evaluation_result/has_artifact |
| `POST /runs/{id}/cancel` | sets that run's cancel event |
| `GET /runs/{id}/artifact` | streams the raw artifact bytes, `X-Artifact-Filename` header |

`{dataset}` is resolved to an actual path *on the agent's own machine*
inside `run_manager.py`, not by the backend -- the backend may have no
filesystem access to `root_path` at all for a genuinely remote agent, so
it cannot resolve a relative path itself the way it does for a no-agent
project. `{output}` similarly points at the agent's own local storage
(`AI_ENV_AGENT_OUTPUT_ROOT`, independent of the backend's
`AI_ENV_STORAGE_ROOT` -- a different machine entirely).

## How the backend talks to it

`app/services/agent_client.py`'s `AgentClient` is a thin `httpx` wrapper
around the table above. `ProjectService` branches on `project.agent_id`
in exactly three places -- `scan()`, `validate()`, and
`_run_project_run_in_background` (via `_execute_locally` vs.
`_execute_via_agent`) -- converting the agent's dict responses back into
the same `ScanResult`/`ValidationResult` types the no-agent path returns,
so nothing above `ProjectService` needs to know or care which case it's
in.

For a run, `_execute_via_agent`:

1. Calls `POST /runs` on the agent, records the returned `agent_run_id` on
   the `ProjectRun` row.
2. Polls `GET /runs/{agent_run_id}` every 0.5s, copying `status`/
   `train_result`/`evaluation_result` into the `ProjectRun` row as it
   goes -- so watching an agent-backed run's progress in the UI works
   identically to a local one.
3. Forwards cancellation: `ProjectService.cancel_run()` sets the same
   in-process `threading.Event` it always has; the poll loop checks it
   each cycle and, if set, calls `POST /runs/{agent_run_id}/cancel` on the
   agent before continuing to poll for the agent's own confirmation.
4. Once the agent reports `has_artifact`, downloads it via
   `GET /runs/{agent_run_id}/artifact` and archives the bytes with
   `archive_downloaded_artifact` (`app/ml/project/storage.py`) -- the same
   managed-storage path convention (`{project}/{run_number}/`) as the
   local path's `archive_run_artifact`, just from bytes instead of a local
   file.

If the agent becomes unreachable mid-run (network partition, process
crash), the poll loop catches that and marks the run `failed` with a
clear `error` message rather than hanging or silently reporting success --
proven by
`tests/integration/test_local_agent_e2e.py::test_agent_becoming_unreachable_mid_run_fails_the_run_honestly`.

## Authentication

One shared bearer token per agent, checked by `agent_app/auth.py`'s
`require_auth` dependency on every route except `/health`.

- **Fail closed.** If `AI_ENV_AGENT_TOKEN` is unset or empty, protected
  routes return **503**, not 200: an agent nobody configured a secret for
  must not silently become an open remote-execution endpoint. Running
  without a token deliberately (a throwaway local test) needs the explicit
  opt-out `AI_ENV_AGENT_ALLOW_NO_AUTH=1` -- exactly `1`, so a stray `true`
  or a forgotten variable can't switch it on.
- **Constant-time comparison** (`hmac.compare_digest`), so response timing
  doesn't reveal how many leading characters of a guess were right. A
  token that is merely a prefix of the real one is rejected.
- **Read per request**, so the token in effect is always the process
  environment's current value.
- **`/health` stays open on purpose.** It's how the backend tells
  "unreachable" from "reachable but I have the wrong token"; it exposes
  only name, version, and whether auth is required/configured.
- **A rejected request never reaches the command runner** -- tested by
  asserting an unauthenticated `POST /runs` does not execute its command.

On the backend, the token lives in `Agent.token`. It has to be stored
recoverably (the backend must present it on every call), so it sits in the
database **in plaintext**, protected only by whatever protects that
database. It is write-only through the API: `AgentRegisterRequest` accepts
it as a `SecretStr`, `AgentRead` has no `token` field (only `has_token`),
and a test asserts the token string never appears in the response of any
agent endpoint. `client_for_agent()` is the single place an `Agent` row
becomes an `AgentClient`, so every scan/validate/run presents it the same
way.

What this does and doesn't buy you: it stops anyone who can *reach* an
agent's port but doesn't hold the secret from scanning paths or running
commands. It does not protect the secret in transit (no TLS -- see below),
does not support per-user or per-project permissions, does not rotate
automatically, and does nothing about an attacker who can already read the
backend's database.

The most important limit: **this token only matters if the backend's own
API token (`AI_ENV_API_TOKEN`, `docs/deployment.md`) is also set.** The
backend attaches an agent's stored token to every call it makes on a
caller's behalf; if the backend's own `/api/projects`/`/api/agents`
routes are left unauthenticated (opted out, or simply not configured),
anyone who can reach the backend can register a project, assign it to an
agent, and start a run, agent-side auth notwithstanding. The agent token
secures the agent *against direct access to its own port*; it does not by
itself secure the agent against an open backend in front of it. Both
tokens have to be set for either one to actually mean something.

## Registering and assigning an agent

- `POST /api/agents/register {name, base_url, token}` -- always creates
  the row (a briefly-offline machine is still worth remembering), then
  immediately checks it and sets `status` honestly: `connected`,
  `unreachable`, or `unauthorized` (reachable, but the token is missing or
  wrong, or the agent itself has no token configured).
  `POST /api/agents/{id}/ping` re-checks later.
- `PUT /api/agents/{id}/token {token}` -- mirrors a token rotated on the
  agent side (or clears it with `null`) and re-checks immediately.
- `PUT /api/projects/{id}/agent {agent_id}` -- switches a project between
  agent-backed and direct execution at any time (`agent_id: null` goes
  back to the backend executing directly). Doesn't touch `config_yaml`,
  `detected_config`, or run history.
- `ProjectRegisterRequest` also accepts an optional `agent_id` up front, so
  a project can be registered already pointed at an agent.

The frontend (`app/agents/page.tsx` for registering/pinging, a select in
`app/projects/[id]/page.tsx`'s Overview panel for assignment) covers both.

## What's proven, and what's still a real gap

`tests/agent_app/` exercises the agent app in isolation (its own
`TestClient`, no separate process). `tests/integration/test_local_agent_e2e.py`
is the one place a real, separate `uvicorn agent_app.main:app` process is
spawned and talked to over an actual loopback socket -- proving the
*wiring* between backend and agent, not just each side's own logic. It
covers: registering a reachable and an unreachable agent, a wrong / missing / never-configured
token (each read as `unauthorized`, never `connected`), token rotation, the token never
appearing in any API response, the full
register -> scan -> configure -> validate -> run -> progress lifecycle
through the agent, real cross-process cancellation, and an agent
disappearing mid-run failing the run honestly rather than hanging.

What this phase does not add:

- **No TLS.** The token is a shared secret sent over plain HTTP, so on an
  untrusted network it can be read off the wire and replayed. Use a
  trusted network, or terminate TLS in front of the agent (a reverse proxy
  such as Caddy or nginx, or an SSH tunnel).
- **No agent-side install/packaging story.** "Running one" above is a raw
  `uvicorn` command; there's no installer, no systemd unit, no way for the
  agent to auto-register itself with a backend on startup (registration is
  always initiated from the backend/UI side).
- **Single agent per project, no load balancing or failover.** If the
  assigned agent goes down, runs fail (honestly) rather than retrying
  elsewhere.
- **No GPU/CPU/RAM monitoring surfaced yet** (Section 5, items 13-15) --
  `agent_app` could add this to `/runs/{id}` the same way
  `ProjectScanner._detect_gpu` already does for the one-time scan reading,
  but nothing polls it continuously during a run.
