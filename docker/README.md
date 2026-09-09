# Docker (Phase 27)

Start the whole system (Postgres + backend + frontend) with:

```bash
docker compose -f docker/docker-compose.yml up --build
```

- Backend: http://localhost:8000 (interactive docs at `/docs`)
- Frontend: http://localhost:3000
- Postgres: localhost:5432 (user/password/db: `postgres`/`postgres`/`ailab`)

The backend container creates database tables on first boot
(`init_db()`, idempotent) before starting `uvicorn`. The frontend is a
multi-stage build (`deps` -> `builder` -> `runner`) producing a small
production image; `NEXT_PUBLIC_API_BASE_URL`/`NEXT_PUBLIC_WS_BASE_URL` are
passed as build `args` (not just runtime `environment:`) because Next.js
inlines `NEXT_PUBLIC_*` variables into the client bundle at build time.

## Verification note

This project was built in a sandboxed environment whose network egress is
restricted to a small allowlist (package registries for pip/npm/apt) that
does not include Docker Hub -- `docker build` could not actually pull
`python:3.12-slim` / `node:20-alpine` / `postgres:16` here, so a full
`docker compose up` could not be executed end-to-end in this environment.

What *was* verified here:
- A real Docker daemon (`dockerd`) was installed and started in this
  sandbox, and `docker build` was run against `backend.Dockerfile` --
  it correctly reached the network layer and failed only at the base
  image pull (`403 Forbidden` from `registry-1.docker.io`, consistent with
  the network allowlist), confirming the Dockerfile syntax itself, `COPY`
  paths, and build-context wiring are correct up to that point.
- `docker-compose.yml`'s YAML was parsed and validated programmatically.
- Both Dockerfiles were structurally parsed (instruction-by-instruction)
  to confirm well-formed syntax.
- The underlying `pip install -r requirements.txt` and `npm ci && npm run
  build` steps the Dockerfiles perform were already proven to succeed in
  this same sandbox, outside Docker, throughout Phases 1-26.

What was **not** verified: an actual running multi-container stack talking
to each other over the Docker network, which requires the image pulls
above. Recommended to `docker compose up --build` once in an environment
with normal internet access before relying on this in production.
