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
    # Every test in this package exercises the *normal, authenticated*
    # path -- auth itself (missing/wrong token, fail-closed, opt-out) is
    # tested in isolation in tests/api/test_api_auth.py against a client
    # that deliberately doesn't send this header.
    monkeypatch.setenv("AI_ENV_API_TOKEN", TEST_API_TOKEN)
    monkeypatch.delenv("AI_ENV_API_ALLOW_NO_AUTH", raising=False)


@pytest.fixture
def client():
    return TestClient(app, headers={"Authorization": f"Bearer {TEST_API_TOKEN}"})


TINY_PAYLOAD = dict(
    agent_mode="demo_ppo",
    grid_size=5,
    obstacle_density=0.1,
    dead_end_probability=0.1,
    path_complexity=0.1,
    max_steps_per_episode=12,
    n_curriculum_steps=2,
    timesteps_per_step=256,
    eval_episodes=3,
    n_candidates=2,
    trajectories_per_analysis=2,
    run_generalization=True,
    generalization_envs_per_split=2,
    generalization_episodes_per_config=2,
)
