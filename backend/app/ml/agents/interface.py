"""Phase 28 (Section 8) -- model-agnostic agent abstraction.

Problem solved: without this, every future agent implementation would need
to be shaped like PPO/Stable-Baselines3. `AgentInterface` is the one
contract ai.env's agent-connected paths depend on, so PPO, DQN, SAC, a
custom RL agent, a robotics agent, an LLM agent, or an externally-hosted
API-based agent can all be plugged in without changing the environment
generation core (`app/ml/environments`, `app/ml/analysis`) at all.

Two concrete things implement this interface in this project:
- `DemoPPOAgent` (demo_ppo.py) -- the explicitly opt-in PPO demonstration
  (Section 4), wrapping the project's existing `AgentTrainer`/SB3 code.
- The external-agent path (external_session.py) does NOT implement this
  interface from ai.env's side -- by design (Section 9), the *external*
  agent owns its own model and calls ai.env's reset/step HTTP endpoints
  directly. ai.env never wraps or drives an external agent's policy; it
  only serves the environment to it.
"""

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
