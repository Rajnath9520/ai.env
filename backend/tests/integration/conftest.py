import pytest
from fastapi.testclient import TestClient

from app.database.session import drop_db, engine, init_db
from app.main import app

TEST_API_TOKEN = "test-api-token-xyz789"


@pytest.fixture(scope="module", autouse=True)
def _setup_schema():
    drop_db(engine)
    init_db(engine)
    yield
    drop_db(engine)


@pytest.fixture(autouse=True)
def _api_token(monkeypatch):
    monkeypatch.setenv("AI_ENV_API_TOKEN", TEST_API_TOKEN)
    monkeypatch.delenv("AI_ENV_API_ALLOW_NO_AUTH", raising=False)


@pytest.fixture
def client():
    return TestClient(app, headers={"Authorization": f"Bearer {TEST_API_TOKEN}"})
