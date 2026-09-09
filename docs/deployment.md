# Deployment

## Docker (recommended)

```bash
docker compose -f docker/docker-compose.yml up --build
```

See `docker/README.md` for full details, including an important note
about this project's development sandbox not having internet access to
pull Docker Hub base images (so full `docker compose up` was not run
end-to-end during development -- the Dockerfiles and compose config were
verified as thoroughly as the environment allowed: real `docker build`
run to the point of the (network-blocked) base-image pull, YAML validated,
Dockerfiles structurally parsed).

## Manual deployment

### Backend

```bash
cd backend
pip install -r requirements.txt
export DATABASE_URL="postgresql+psycopg2://<user>:<password>@<host>:5432/<db>"
python -c "from app.database.session import engine, init_db; init_db(engine)"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=https://your-backend-host \
NEXT_PUBLIC_WS_BASE_URL=wss://your-backend-host \
npm run build
npm start
```

Note: `NEXT_PUBLIC_*` variables are inlined into the client bundle at
**build** time, not read at container/process start -- set them before
`npm run build`, not just before `npm start`.

## Environment variables

| Variable | Where | Purpose |
|---|---|---|
| `DATABASE_URL` | backend | SQLAlchemy connection string (`postgresql+psycopg2://...`) |
| `NEXT_PUBLIC_API_BASE_URL` | frontend (build time) | REST API base URL |
| `NEXT_PUBLIC_WS_BASE_URL` | frontend (build time) | WebSocket base URL |

## Production considerations (known gaps, stated plainly)

- **No Alembic migrations** -- schema is created via
  `Base.metadata.create_all()`. Fine for a single initial deployment; a
  real production rollout with evolving schema needs real migrations.
- **In-process WebSocket broadcaster and cancellation flags** -- won't
  work correctly behind multiple API worker processes/replicas without
  moving to a shared store (Redis is the natural choice, already
  mentioned as an option in the original architecture plan).
- **Training runs as a FastAPI BackgroundTask in the same process** as the
  API server -- fine for a single-instance deployment; a real production
  setup training many concurrent experiments would want a dedicated worker
  process/queue instead.
- **No authentication/authorization** anywhere in the API -- the `users`
  table exists in the schema but nothing enforces per-user access control
  yet.
