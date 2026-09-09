"""Bridges Phase 15's `ExperimentResult` (in-memory / JSON) to the Phase 16
database schema. This is the concrete answer to Section 23's "store
experiment metadata" -- not just table definitions nobody writes to.

Episode/Failure granularity: `CurriculumStep` records one evaluation
per curriculum step (aggregated over several episodes), not individual raw
episodes. For failure_driven experiments, this module stores exactly one
representative `Episode` row per curriculum step (reward/steps/success from
that step's evaluation) with its real `Failure` rows attached (from
`failure_reports_history`, which are genuine per-episode failure-analyzer
outputs, aggregated across the trajectories sampled for that step) --
`collisions` on that row is left at 0 since a precise collision *count* was
never retained at this granularity, rather than fabricating one.
"""

from __future__ import annotations

import yaml
from sqlalchemy.orm import Session

from app.ml.experiment.manager import ExperimentResult
from app.models.orm import (
    AgentWeakness,
    Environment,
    Episode,
    Evaluation,
    Experiment,
    Failure,
    LearningProgress,
    ModelCheckpoint,
    TrainingRun,
)


def persist_experiment_result(session: Session, result: ExperimentResult) -> Experiment:
    config = result.config

    experiment = Experiment(
        name=config.name,
        config_yaml=yaml.safe_dump(config.to_dict(), sort_keys=False),
        curriculum_type=config.curriculum_type,
        seed=config.seed,
        status="completed",
    )
    session.add(experiment)
    session.flush()

    attach_results_to_experiment(session, experiment, result)
    return experiment


def attach_results_to_experiment(
    session: Session, experiment: Experiment, result: ExperimentResult
) -> Experiment:
    """Writes an ExperimentResult's environments/training runs/weaknesses/
    learning-progress/episodes/failures/evaluations onto an *existing*
    Experiment row. Split out from `persist_experiment_result` so the API
    layer (Phase 17) can create the Experiment row up front (status=
    "pending", before anything has run) and attach results once the run
    finishes, without persist_experiment_result's assumption that the
    Experiment row doesn't exist yet.
    """

    lp_by_environment_id = {
        lp["environment_id"]: lp for lp in (result.learning_progress_history or [])
    }

    last_training_run: TrainingRun | None = None
    for i, step in enumerate(result.curriculum_history):
        env_row = Environment(
            experiment_id=experiment.id,
            environment_key=step.environment_id,
            params=step.config.to_dict(),
            difficulty=step.difficulty,
            target_weakness=step.target_weakness,
            split="train",
        )
        session.add(env_row)
        session.flush()

        training_run = TrainingRun(experiment_id=experiment.id, environment_id=env_row.id)
        session.add(training_run)
        session.flush()
        last_training_run = training_run

        lp = lp_by_environment_id.get(step.environment_id)
        if lp is not None:
            session.add(
                LearningProgress(
                    environment_id=env_row.id,
                    metric=lp["metric"],
                    performance_before=lp["performance_before"],
                    performance_after=lp["performance_after"],
                    learning_progress=lp["learning_progress"],
                )
            )

        if result.weakness_history and i < len(result.weakness_history):
            w = result.weakness_history[i]
            session.add(
                AgentWeakness(
                    training_run_id=training_run.id,
                    obstacle_avoidance=w["obstacle_avoidance"],
                    exploration=w["exploration"],
                    dead_end_recovery=w["dead_end_recovery"],
                    path_efficiency=w["path_efficiency"],
                    short_horizon_planning=w["short_horizon_planning"],
                    long_horizon_planning=w["long_horizon_planning"],
                    primary_weakness=w["primary_weakness"],
                    secondary_weaknesses=w["secondary_weaknesses"],
                )
            )

        if result.failure_reports_history and i < len(result.failure_reports_history):
            # One representative/aggregate episode row per curriculum step,
            # built from this step's evaluation -- Section 23's per-episode
            # schema exists for finer-grained data than a curriculum step
            # currently retains (see repository module docstring); this is
            # the honest middle ground rather than fabricating per-step
            # raw episode counts that were never measured.
            episode_row = Episode(
                training_run_id=training_run.id,
                episode_number=1,
                reward=step.average_reward,
                steps=round(step.average_steps) if step.average_steps is not None else step.config.max_steps,
                success=step.success_rate >= 0.5,
                collisions=0,  # not tracked at this aggregate granularity; see docstring
            )
            session.add(episode_row)
            session.flush()

            for report in result.failure_reports_history[i]:
                session.add(
                    Failure(
                        episode_id=episode_row.id,
                        failure_type=report["failure_type"],
                        score=report["score"],
                        confidence=report["confidence"],
                        features=report["features"],
                    )
                )

    if result.generalization_results:
        for split, r in result.generalization_results.items():
            session.add(
                Evaluation(
                    experiment_id=experiment.id,
                    split=split,
                    success_rate=r["success_rate"],
                    average_reward=r["average_reward"],
                    average_steps=r["average_steps"],
                    collision_rate=r["collision_rate"],
                )
            )

    # Phase 25: the final trained-policy checkpoint (if one was saved)
    # is attached to the *last* training run, since it reflects the fully
    # trained policy after every curriculum step, not any single step.
    if last_training_run is not None and result.reproducibility:
        checkpoint_path = result.reproducibility.get("checkpoint_path")
        if checkpoint_path:
            session.add(
                ModelCheckpoint(
                    training_run_id=last_training_run.id,
                    path=checkpoint_path,
                    episode=len(result.curriculum_history),
                )
            )

    session.flush()
    return experiment


def attach_environment_only_results(
    session: Session, experiment: Experiment, result: ExperimentResult
) -> Experiment:
    """Phase 28 (Section 21) -- persists an environment_only
    `ExperimentResult` as `Environment` rows only. Deliberately does not
    create any `TrainingRun`, `Episode`, `AgentWeakness`, or
    `ModelCheckpoint` row: none of those exist for this mode, and creating
    empty/fake ones would misrepresent the experiment as having trained an
    agent when it did not (Section 21: "Do not create fake training
    records for environment-only experiments.").
    """
    for record in result.generated_environments or []:
        session.add(
            Environment(
                experiment_id=experiment.id,
                environment_key=record["environment_id"],
                params={
                    **record["config"],
                    "metrics": record["metrics"],
                    "generation_reason": record["generation_reason"],
                    "generation_score": record["generation_score"],
                },
                difficulty=record["difficulty"],
                target_weakness=None,  # no agent weakness exists in this mode
                split="train",
            )
        )
    session.flush()
    return experiment


def get_experiment(session: Session, experiment_id: int) -> Experiment | None:
    return session.get(Experiment, experiment_id)
