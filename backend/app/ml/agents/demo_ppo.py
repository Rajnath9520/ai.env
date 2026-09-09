"""Phase 28 (Section 4/8) -- Demo Agent: PPO.

This is the *only* legitimate path in ai.env that ever instantiates
`stable_baselines3.PPO` (Section 30's "search the project for hidden
coupling" requirement) outside of an external agent's own process, which is
none of ai.env's business by design (Section 9). It exists purely as an
optional, explicitly-selected demonstration of what an AgentInterface
implementation looks like -- it must never be created implicitly.

This wraps the project's pre-existing `AgentTrainer` (Phase 2) rather than
reimplementing PPO plumbing, since that trainer is already well-tested and
this is a thin adapter, not a rewrite.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ml.agents.interface import AgentInterface
from app.ml.environments.config import GridWorldConfig
from app.ml.training.env_factory import build_env
from app.ml.training.trainer import AgentTrainer

DEMO_AGENT_LABEL = "Demo Agent -- PPO"


class DemoPPOAgent(AgentInterface):
    """Explicit, opt-in-only PPO demonstration agent (Section 4). Callers
    must deliberately construct this class -- nothing in the environment
    generation core does so automatically.
    """

    label = DEMO_AGENT_LABEL

    def __init__(self, config: GridWorldConfig, seed: int | None = None) -> None:
        self._config = config
        self._trainer = AgentTrainer(seed=seed)
        self._env = None

    @property
    def trainer(self) -> AgentTrainer:
        """Exposes the underlying AgentTrainer so existing curriculum
        runners (which call trainer.train/evaluate directly, not
        reset/act) can keep using their well-tested training loop; this
        wrapper's reset/act/close satisfy AgentInterface for any caller
        that wants the model-agnostic surface instead."""
        return self._trainer

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._env = build_env(self._config)
        obs, info = self._env.reset()
        return obs, info

    def act(self, observation: np.ndarray) -> int:
        if self._trainer.model is None:
            # No policy trained yet -- act randomly rather than raising, so
            # a demo agent can still be exercised before its first train()
            # call, exactly like any freshly-initialized RL agent would.
            return int(np.random.randint(0, 4))
        action, _ = self._trainer.model.predict(observation, deterministic=True)
        return int(action)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
