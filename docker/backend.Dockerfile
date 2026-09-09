# Phase 27 -- Backend container.
# Build context is the project root (see docker-compose.yml), so paths
# below are relative to project-root/, not project-root/docker/.

FROM python:3.12-slim

# psycopg2 (non-binary parts) and general build tooling need these; kept
# minimal rather than pulling in a full build-essential image layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/scripts ./scripts
COPY backend/pytest.ini ./pytest.ini

# Not copying backend/tests/ or backend/experiments/ into the runtime
# image -- tests run in CI/dev against the source tree, not the built
# image, and experiment result JSON is generated at runtime, not baked in.

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
