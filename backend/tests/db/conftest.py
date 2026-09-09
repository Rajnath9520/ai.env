import os

import pytest
from sqlalchemy.orm import sessionmaker

from app.database.session import drop_db, init_db, make_engine

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/ailab_test"
)


@pytest.fixture(scope="module")
def engine():
    eng = make_engine(TEST_DATABASE_URL)
    drop_db(eng)
    init_db(eng)
    yield eng
    drop_db(eng)
    eng.dispose()


@pytest.fixture
def session(engine):
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    s = Session()
    try:
        yield s
        s.rollback()
    finally:
        s.close()
