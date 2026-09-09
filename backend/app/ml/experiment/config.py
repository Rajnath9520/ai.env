"""Phase 15 -- Experiment configuration (Section 22).

`ExperimentConfig` is the one object that fully determines an experiment:
algorithm, seed, environment parameters, curriculum strategy, and training
budget. "The entire experiment must be reproducible from configuration"
(Section 22) means every other component the experiment touches must
derive its randomness from `config.seed` -- this module doesn't enforce
that itself (ExperimentManager does, by construction), but it's the single
source of the seed every downstream component gets.

YAML shape: Section 22's example uses a nested shape (`curriculum.type`,
`environment.size`, `training.episodes`, `evaluation.frequency`). This
project's runners work in terms of curriculum *steps* and *timesteps per
step*, not a single "episodes" budget, so `training.episodes` in the YAML
is treated as the total training budget: `timesteps_per_step =
episodes // n_curriculum_steps` when both are present. This mapping is a
documented interpretation of Section 22's example, not a discovered fact --
`from_yaml_dict` also accepts the flat, unambiguous field names directly
(`n_curriculum_steps`, `timesteps_per_step`, ...) for full control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

VALID_CURRICULUM_TYPES = ("random", "fixed", "generic_adaptive", "failure_driven")

# Phase 28 (Section 6/26) -- explicit, required agent mode. This is the one
# switch that decides whether any model is ever created:
#   "environment_only" -- default. Generate/validate/analyze/export
#       environments only. No AgentTrainer, no PPO, no training.
#   "external"          -- expose the environment to an external agent via
#       the reset/step interface; ai.env never owns or creates a model.
#   "demo_ppo"           -- explicitly opt-in built-in PPO demonstration,
#       identical in spirit to this project's original curriculum/training
#       behavior, but never instantiated implicitly.
VALID_AGENT_MODES = ("environment_only", "external", "demo_ppo")

# Environment-only generation strategies (Section 13/23) -- all agent-free,
# pure functions of history/config, reusing the same rule-based building
# blocks (GridWorldGenerator, EnvironmentValidator, EnvironmentAnalyzer)
# used everywhere else in the project.
VALID_GENERATION_STRATEGIES = ("diverse", "progressive", "random")


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    # `algorithm` is retained only for backward compatibility with configs
    # saved before Phase 28 (Section 6) -- it is informational/secondary and
    # never controls execution on its own. `agent_mode` is the field that
    # actually decides what runs; see Section 6.
    algorithm: str | None = "PPO"
    agent_mode: str = "environment_only"
    curriculum_type: str = "failure_driven"
    seed: int = 42

    grid_size: int = 10
    obstacle_density: float = 0.15
    dead_end_probability: float = 0.15
    path_complexity: float = 0.15
    max_steps_per_episode: int = 60

    n_curriculum_steps: int = 20
    timesteps_per_step: int = 10_000
    eval_episodes: int = 20
    evaluation_frequency: int = 500  # reserved for Phase 18's live-eval cadence; not yet consumed here

    # failure_driven-only knobs (ignored by other curriculum types); also
    # reused, agent-free, by environment_only mode's "diverse" generation
    # strategy and by external mode's adaptive-generation loop.
    n_candidates: int = 5
    trajectories_per_analysis: int = 8

    # environment_only mode only (Section 13/15) -- which agent-free
    # strategy to use when generating the sequence of environments.
    generation_strategy: str = "diverse"

    # Phase 14 generalization evaluation, run at the end of every experiment
    # that actually trains an agent (demo_ppo only -- Section 13 environment
    # metrics are not "agent generalization", so this has no effect outside
    # demo_ppo mode).
    run_generalization: bool = True
    generalization_envs_per_split: int = 10
    generalization_episodes_per_config: int = 5

    def __post_init__(self) -> None:
        if self.curriculum_type not in VALID_CURRICULUM_TYPES:
            raise ValueError(
                f"curriculum_type must be one of {VALID_CURRICULUM_TYPES}, got {self.curriculum_type!r}"
            )
        if self.agent_mode not in VALID_AGENT_MODES:
            raise ValueError(
                f"agent_mode must be one of {VALID_AGENT_MODES}, got {self.agent_mode!r}"
            )
        if self.generation_strategy not in VALID_GENERATION_STRATEGIES:
            raise ValueError(
                f"generation_strategy must be one of {VALID_GENERATION_STRATEGIES}, "
                f"got {self.generation_strategy!r}"
            )
        if self.n_curriculum_steps < 1:
            raise ValueError("n_curriculum_steps must be >= 1")
        if self.timesteps_per_step < 1:
            raise ValueError("timesteps_per_step must be >= 1")
        # `algorithm`, when provided, is informational only (see field
        # docstring) -- only PPO is a recognized value, but it no longer
        # gates whether anything is executed. That is agent_mode's job.
        if self.algorithm not in (None, "PPO"):
            raise ValueError("only 'PPO' is a recognized 'algorithm' value in this version")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_yaml_dict(self) -> dict:
        """The project's own canonical flat shape -- round-trips exactly."""
        return self.to_dict()

    @classmethod
    def from_yaml_dict(cls, data: dict) -> "ExperimentConfig":
        # Flat, unambiguous shape (this project's canonical format).
        if "curriculum_type" in data or "n_curriculum_steps" in data:
            return cls.from_dict(data)

        # Section 22's example nested shape.
        curriculum = data.get("curriculum", {})
        environment = data.get("environment", {})
        training = data.get("training", {})
        evaluation = data.get("evaluation", {})

        kwargs: dict = {}
        if "algorithm" in data:
            kwargs["algorithm"] = data["algorithm"]
        if "seed" in data:
            kwargs["seed"] = data["seed"]
        if "type" in curriculum:
            kwargs["curriculum_type"] = curriculum["type"]
        if "size" in environment:
            kwargs["grid_size"] = environment["size"]
        if "frequency" in evaluation:
            kwargs["evaluation_frequency"] = evaluation["frequency"]

        n_steps = curriculum.get("n_steps", cls.n_curriculum_steps)
        kwargs["n_curriculum_steps"] = n_steps
        if "episodes" in training:
            # documented interpretation: total training budget, spread
            # evenly across curriculum steps (see module docstring)
            kwargs["timesteps_per_step"] = max(1, int(training["episodes"]) // int(n_steps))

        return cls.from_dict(kwargs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_yaml_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_yaml_dict(data or {})
