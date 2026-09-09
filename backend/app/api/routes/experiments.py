"""Phase 17 -- experiment API routes (Section 24). Thin controllers only:
validate via Pydantic, delegate to ExperimentService, translate results/
errors to HTTP. No ML logic, no direct ORM queries beyond simple read
endpoints that don't warrant a dedicated service method.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.ml.environments.config import GridWorldConfig
from app.ml.environments.generator import GridWorldGenerator
from app.models.orm import (
    AgentWeakness,
    Environment,
    Episode,
    Evaluation,
    Experiment,
    Failure,
    LearningProgress,
    Trajectory,
    TrainingRun,
)
from app.schemas.experiment import (
    ComparisonEntry,
    EnvironmentLayoutRead,
    EnvironmentRead,
    EpisodeRead,
    ExperimentCreate,
    ExperimentRead,
    ExperimentStatusRead,
    FailureRead,
    GeneralizationRead,
    LearningProgressRead,
    MetricsRead,
    TrajectoryRead,
    WeaknessRead,
)
from app.services.experiment_service import (
    ExperimentNotFound,
    ExperimentService,
    ExperimentStateError,
)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _service(db: Session = Depends(get_db)) -> ExperimentService:
    return ExperimentService(db)


def _get_experiment_or_404(service: ExperimentService, experiment_id: int) -> Experiment:
    try:
        return service.get(experiment_id)
    except ExperimentNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreate, service: ExperimentService = Depends(_service)):
    return service.create(payload)


@router.get("", response_model=list[ExperimentRead])
def list_experiments(service: ExperimentService = Depends(_service)):
    return service.list_all()


@router.get("/{experiment_id}", response_model=ExperimentRead)
def get_experiment(experiment_id: int, service: ExperimentService = Depends(_service)):
    return _get_experiment_or_404(service, experiment_id)


@router.post("/{experiment_id}/start", response_model=ExperimentStatusRead)
def start_experiment(
    experiment_id: int,
    background_tasks: BackgroundTasks,
    service: ExperimentService = Depends(_service),
):
    try:
        experiment = service.start(experiment_id, background_tasks)
    except ExperimentNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ExperimentStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ExperimentStatusRead(id=experiment.id, status=experiment.status)


@router.post("/{experiment_id}/stop", response_model=ExperimentStatusRead)
def stop_experiment(experiment_id: int, service: ExperimentService = Depends(_service)):
    try:
        experiment = service.stop(experiment_id)
    except ExperimentNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    message = None
    if experiment.status == "running":
        message = "stop requested; cancellation is best-effort and may not take effect immediately"
    return ExperimentStatusRead(id=experiment.id, status=experiment.status, message=message)


@router.get("/{experiment_id}/status", response_model=ExperimentStatusRead)
def get_status(experiment_id: int, service: ExperimentService = Depends(_service)):
    experiment = _get_experiment_or_404(service, experiment_id)
    return ExperimentStatusRead(id=experiment.id, status=experiment.status)


@router.get("/{experiment_id}/metrics", response_model=MetricsRead)
def get_metrics(experiment_id: int, db: Session = Depends(get_db), service: ExperimentService = Depends(_service)):
    experiment = _get_experiment_or_404(service, experiment_id)

    environments = (
        db.query(Environment).filter(Environment.experiment_id == experiment_id).order_by(Environment.id).all()
    )
    latest_env = environments[-1] if environments else None

    latest_success_rate = None
    if latest_env is not None:
        latest_lp = (
            db.query(LearningProgress)
            .filter(LearningProgress.environment_id == latest_env.id)
            .order_by(LearningProgress.id.desc())
            .first()
        )
        if latest_lp is not None:
            latest_success_rate = latest_lp.performance_after

    latest_weakness = (
        db.query(AgentWeakness)
        .join(TrainingRun)
        .filter(TrainingRun.experiment_id == experiment_id)
        .order_by(AgentWeakness.id.desc())
        .first()
    )
    latest_lp_row = (
        db.query(LearningProgress)
        .join(Environment)
        .filter(Environment.experiment_id == experiment_id)
        .order_by(LearningProgress.id.desc())
        .first()
    )

    return MetricsRead(
        experiment_id=experiment_id,
        agent_mode=experiment.agent_mode,
        status=experiment.status,
        n_environments_trained=len(environments),
        latest_success_rate=latest_success_rate,
        latest_difficulty=latest_env.difficulty if latest_env else None,
        latest_primary_weakness=latest_weakness.primary_weakness if latest_weakness else None,
        latest_learning_progress=latest_lp_row.learning_progress if latest_lp_row else None,
    )


@router.get("/{experiment_id}/episodes", response_model=list[EpisodeRead])
def get_episodes(experiment_id: int, db: Session = Depends(get_db), service: ExperimentService = Depends(_service)):
    _get_experiment_or_404(service, experiment_id)
    return (
        db.query(Episode)
        .join(TrainingRun)
        .filter(TrainingRun.experiment_id == experiment_id)
        .order_by(Episode.id)
        .all()
    )


@router.get("/{experiment_id}/trajectories", response_model=list[TrajectoryRead])
def get_trajectories(
    experiment_id: int, db: Session = Depends(get_db), service: ExperimentService = Depends(_service)
):
    _get_experiment_or_404(service, experiment_id)
    # NOTE: raw/summarized per-episode Trajectory rows are not yet written
    # by the repository layer (Phase 16's documented scope) -- this query
    # is real, not a placeholder, it will simply return [] until a future
    # phase persists Trajectory rows. See app/database/repository.py.
    return (
        db.query(Trajectory)
        .join(Episode)
        .join(TrainingRun)
        .filter(TrainingRun.experiment_id == experiment_id)
        .order_by(Trajectory.id)
        .all()
    )


@router.get("/{experiment_id}/failures", response_model=list[FailureRead])
def get_failures(experiment_id: int, db: Session = Depends(get_db), service: ExperimentService = Depends(_service)):
    _get_experiment_or_404(service, experiment_id)
    return (
        db.query(Failure)
        .join(Episode)
        .join(TrainingRun)
        .filter(TrainingRun.experiment_id == experiment_id)
        .order_by(Failure.id)
        .all()
    )


@router.get("/{experiment_id}/weaknesses", response_model=list[WeaknessRead])
def get_weaknesses(
    experiment_id: int, db: Session = Depends(get_db), service: ExperimentService = Depends(_service)
):
    _get_experiment_or_404(service, experiment_id)
    return (
        db.query(AgentWeakness)
        .join(TrainingRun)
        .filter(TrainingRun.experiment_id == experiment_id)
        .order_by(AgentWeakness.id)
        .all()
    )


@router.get("/{experiment_id}/environments", response_model=list[EnvironmentRead])
def get_environments(
    experiment_id: int, db: Session = Depends(get_db), service: ExperimentService = Depends(_service)
):
    _get_experiment_or_404(service, experiment_id)
    return (
        db.query(Environment)
        .filter(Environment.experiment_id == experiment_id)
        .order_by(Environment.id)
        .all()
    )


@router.get("/{experiment_id}/learning-progress", response_model=list[LearningProgressRead])
def get_learning_progress(
    experiment_id: int, db: Session = Depends(get_db), service: ExperimentService = Depends(_service)
):
    _get_experiment_or_404(service, experiment_id)
    return (
        db.query(LearningProgress)
        .join(Environment)
        .filter(Environment.experiment_id == experiment_id)
        .order_by(LearningProgress.id)
        .all()
    )


@router.get("/{experiment_id}/generalization", response_model=list[GeneralizationRead])
def get_generalization(
    experiment_id: int, db: Session = Depends(get_db), service: ExperimentService = Depends(_service)
):
    _get_experiment_or_404(service, experiment_id)
    return (
        db.query(Evaluation)
        .filter(Evaluation.experiment_id == experiment_id)
        .order_by(Evaluation.id)
        .all()
    )


@router.get("/{experiment_id}/environments/{environment_id}/layout", response_model=EnvironmentLayoutRead)
def get_environment_layout(
    experiment_id: int,
    environment_id: int,
    db: Session = Depends(get_db),
    service: ExperimentService = Depends(_service),
):
    """Phase 20 -- regenerates and returns the concrete maze (obstacles,
    start, goal) for one environment. Not read from a stored layout table
    (Phase 16 only persists generative parameters, per its documented
    scope) -- the GridWorld generator (Phase 1) is a deterministic
    function of (grid_size, seed, density knobs), so regenerating here
    reproduces the exact same layout the agent actually trained on.
    """
    _get_experiment_or_404(service, experiment_id)
    env = db.get(Environment, environment_id)
    if env is None or env.experiment_id != experiment_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Environment {environment_id} not found")

    config = GridWorldConfig(
        grid_size=env.params["grid_size"],
        obstacle_density=env.params["obstacle_density"],
        goal_distance=env.params.get("goal_distance"),
        dead_end_probability=env.params["dead_end_probability"],
        path_complexity=env.params["path_complexity"],
        max_steps=env.params["max_steps"],
        seed=env.params.get("seed"),
    )
    report = GridWorldGenerator().generate(config)
    layout = report.layout

    return EnvironmentLayoutRead(
        environment_id=env.id,
        environment_key=env.environment_key,
        grid_size=layout.grid_size,
        agent_start=layout.agent_start,
        goal=layout.goal,
        obstacles=sorted(layout.obstacles),
    )


@router.get("/{experiment_id}/comparison", response_model=list[ComparisonEntry])
def get_comparison(
    experiment_id: int,
    compare_to: str | None = None,
    db: Session = Depends(get_db),
    service: ExperimentService = Depends(_service),
):
    """Compares `experiment_id` against other experiments. `compare_to` is
    an optional comma-separated list of experiment ids; defaults to every
    other experiment in the database (Section 33's Comparison page is
    expected to let the user narrow this down client-side or via this
    param).
    """
    _get_experiment_or_404(service, experiment_id)

    if compare_to:
        try:
            ids = {experiment_id, *(int(x) for x in compare_to.split(",") if x.strip())}
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="compare_to must be comma-separated integers") from exc
        experiments = db.query(Experiment).filter(Experiment.id.in_(ids)).order_by(Experiment.id).all()
    else:
        experiments = db.query(Experiment).order_by(Experiment.id).all()

    entries: list[ComparisonEntry] = []
    for exp in experiments:
        last_lp = (
            db.query(LearningProgress)
            .join(Environment)
            .filter(Environment.experiment_id == exp.id)
            .order_by(LearningProgress.id.desc())
            .first()
        )
        evaluations = db.query(Evaluation).filter(Evaluation.experiment_id == exp.id).all()
        entries.append(
            ComparisonEntry(
                experiment_id=exp.id,
                name=exp.name,
                agent_mode=exp.agent_mode,
                curriculum_type=exp.curriculum_type,
                status=exp.status,
                final_success_rate=last_lp.performance_after if last_lp else None,
                generalization={e.split: GeneralizationRead.model_validate(e) for e in evaluations},
            )
        )
    return entries
