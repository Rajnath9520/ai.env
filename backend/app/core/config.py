

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg2://postgres:amit@localhost:5432/ailab"
    )
   
    ai_env_storage_root: str = os.environ.get("AI_ENV_STORAGE_ROOT", "./data/ai-env-storage")


settings = Settings()
