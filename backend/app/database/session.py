"""Engine/session management. One engine per process, sessions created
per unit-of-work via `get_session()`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models.orm import Base


def make_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or settings.database_url, future=True)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db(bind: Engine | None = None) -> None:
    Base.metadata.create_all(bind=bind or engine)


def drop_db(bind: Engine | None = None) -> None:
    Base.metadata.drop_all(bind=bind or engine)


@contextmanager
def get_session(session_factory: sessionmaker | None = None) -> Iterator[Session]:
    factory = session_factory or SessionLocal
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
