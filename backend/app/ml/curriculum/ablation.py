"""Phase 22 -- Ablation Studies (Section 22/38).

Defines the four ablation conditions and builds an AdaptiveCurriculumRunner
configured for each, so the study can isolate which component of the
proposed method (Phase 9-13) actually drives any observed effect:

    A. Failure analysis WITHOUT learning-progress optimization
       -> candidates are targeted at real detected weaknesses (alpha > 0),
          but selection ignores predicted learning progress (beta = 0) and
          diversity (gamma = 0).
    B. Learning-progress optimization WITHOUT detailed failure analysis
       -> candidate targeting is content-blind (see
          `_blind_weakness_vector` in failure_driven.py) so it doesn't
          reflect real detected weaknesses, but selection is driven by
          expected learning progress (beta > 0, alpha = 0).
    C. Failure analysis + learning progress
       -> both alpha and beta > 0, diversity still off (gamma = 0).
    D. Failure analysis + learning progress + diversity
       -> the full proposed method (alpha, beta, gamma all > 0) -- this is
          exactly Phase 13's default AdaptiveCurriculumRunner configuration.

`delta` (the excessive-difficulty penalty) is held constant across all four
variants -- it's a safety guard against picking unproductively-hard
environments, not one of the components under study here, so varying it
would confound the comparison Section 22 asks for.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.curriculum.controller import CurriculumController
from app.ml.curriculum.failure_driven import AdaptiveCurriculumRunner
from app.ml.environments.config import GridWorldConfig
from app.ml.training.trainer import AgentTrainer

CONSTANT_DELTA = 0.15


@dataclass(frozen=True)
class AblationVariant:
    key: str
    label: str
    use_failure_analysis: bool
    alpha: float
    beta: float
    gamma: float


ABLATION_VARIANTS: dict[str, AblationVariant] = {
    "A_failure_analysis_only": AblationVariant(
        key="A_failure_analysis_only",
        label="Failure analysis without learning-progress optimization",
        use_failure_analysis=True,
        alpha=0.7,
        beta=0.0,
        gamma=0.0,
    ),
    "B_learning_progress_only": AblationVariant(
        key="B_learning_progress_only",
        label="Learning-progress optimization without detailed failure analysis",
        use_failure_analysis=False,
        alpha=0.0,
        beta=0.7,
        gamma=0.0,
    ),
    "C_failure_analysis_plus_lp": AblationVariant(
        key="C_failure_analysis_plus_lp",
        label="Failure analysis + learning progress",
        use_failure_analysis=True,
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
    ),
    "D_full_method": AblationVariant(
        key="D_full_method",
        label="Failure analysis + learning progress + diversity (full method)",
        use_failure_analysis=True,
        alpha=0.35,
        beta=0.35,
        gamma=0.15,
    ),
}


def build_ablation_runner(
    variant_key: str,
    trainer: AgentTrainer,
    base_config: GridWorldConfig,
    **runner_kwargs,
) -> AdaptiveCurriculumRunner:
    if variant_key not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant {variant_key!r}; expected one of {list(ABLATION_VARIANTS)}")
    variant = ABLATION_VARIANTS[variant_key]

    controller = CurriculumController(
        alpha=variant.alpha, beta=variant.beta, gamma=variant.gamma, delta=CONSTANT_DELTA
    )
    return AdaptiveCurriculumRunner(
        trainer=trainer,
        base_config=base_config,
        controller=controller,
        use_failure_analysis=variant.use_failure_analysis,
        **runner_kwargs,
    )
