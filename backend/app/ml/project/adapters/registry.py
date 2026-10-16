"""Section 9/11 -- builds the right `TrainingAdapter` for a project's
confirmed `AIEnvConfig`.

`config.adapter.type` picks the implementation: `"command"` (the default)
is `CommandAdapter` (Section 10); `"custom"` is `CustomAdapter` (Section
11), which runs a project's own `.ai-env/adapter.py` in its own
subprocess rather than being imported here -- see that module's docstring
for why. This is the one seam a future third adapter type plugs into;
nothing else has to change when one is added.
"""

from __future__ import annotations

from pathlib import Path

from app.ml.project.adapters.base import TrainingAdapter
from app.ml.project.adapters.command_adapter import CommandAdapter
from app.ml.project.adapters.custom_adapter import CustomAdapter
from app.ml.project.config import AIEnvConfig


def build_adapter(config: AIEnvConfig, project_root: str | Path) -> TrainingAdapter:
    if config.adapter.type == "custom":
        return CustomAdapter(config=config, project_root=project_root)
    return CommandAdapter(config=config, project_root=project_root)
