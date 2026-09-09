"""Minimal settings loader -- deliberately not pydantic-settings, since the
only external config this project needs right now is the database URL, and
a small dataclass keeps that from becoming a new dependency for one value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/ailab"
    )


settings = Settings()
