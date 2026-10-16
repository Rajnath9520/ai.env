

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AgentInterface(Protocol):
    """Minimal contract any ai.env-hosted agent must satisfy. Intentionally
    small -- reset/act/close -- so it does not presume any particular
    algorithm family, observation type, or training loop shape."""

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the agent's environment and return (observation, info)."""
        ...

    def act(self, observation: np.ndarray) -> int:
        """Return the next action given an observation."""
        ...

    def close(self) -> None:
        """Release any resources (model, environment) held by the agent."""
        ...
