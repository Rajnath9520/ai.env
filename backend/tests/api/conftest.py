import pytest
from fastapi.testclient import TestClient

from app.database.session import drop_db, engine, init_db
from app.main import app


@pytest.fixture(scope="module", autouse=True)
def _setup_schema():
    drop_db(engine)
    init_db(engine)
    yield
    drop_db(engine)


@pytest.fixture
def client():
    return TestClient(app)


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
