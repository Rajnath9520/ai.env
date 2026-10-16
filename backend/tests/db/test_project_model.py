from app.models.orm import Experiment, Project, ProjectRun


def _make_project(**overrides) -> Project:
    defaults = dict(name="face-recognition", root_path="/projects/face-recognition", status="draft")
    defaults.update(overrides)
    return Project(**defaults)


def _make_experiment(**overrides) -> Experiment:
    defaults = dict(
        name="exp_1", config_yaml="seed: 1\n", curriculum_type="failure_driven", seed=1, status="pending"
    )
    defaults.update(overrides)
    return Experiment(**defaults)


def _make_run(project_id: int, **overrides) -> ProjectRun:
    defaults = dict(project_id=project_id, run_number=1, status="queued")
    defaults.update(overrides)
    return ProjectRun(**defaults)


class TestProjectRow:
    def test_create_project_with_defaults(self, session):
        project = _make_project()
        session.add(project)
        session.flush()

        assert project.id is not None
        assert project.source == "local"
        assert project.status == "draft"
        assert project.capabilities == []
        assert project.detected_config is None
        assert project.config_yaml is None

    def test_jsonb_fields_round_trip(self, session):
        project = _make_project(
            capabilities=["train", "evaluate", "checkpoint"],
            detected_config={"framework": "pytorch", "training_candidates": ["train.py"]},
            validation_result={"ok": True, "issues": []},
        )
        session.add(project)
        session.flush()
        session.expire(project)  # force a real reload from Postgres, not the identity map

        reloaded = session.get(Project, project.id)
        assert reloaded.capabilities == ["train", "evaluate", "checkpoint"]
        assert reloaded.detected_config == {"framework": "pytorch", "training_candidates": ["train.py"]}
        assert reloaded.validation_result == {"ok": True, "issues": []}


class TestExperimentProjectLink:
    def test_built_in_experiment_has_no_project(self, session):
        experiment = _make_experiment()
        session.add(experiment)
        session.flush()
        assert experiment.project_id is None
        assert experiment.project is None

    def test_experiment_can_reference_a_registered_project(self, session):
        project = _make_project()
        session.add(project)
        session.flush()

        experiment = _make_experiment(project_id=project.id)
        session.add(experiment)
        session.flush()

        assert experiment.project.id == project.id
        assert experiment in project.experiments

    def test_deleting_project_sets_experiment_project_id_null_not_deleting_the_experiment(self, session):
        project = _make_project()
        session.add(project)
        session.flush()

        experiment = _make_experiment(project_id=project.id)
        session.add(experiment)
        session.flush()
        experiment_id = experiment.id

        session.delete(project)
        session.flush()
        session.expire_all()

        surviving = session.get(Experiment, experiment_id)
        assert surviving is not None
        assert surviving.project_id is None


class TestProjectRunRow:
    def test_run_numbers_and_status_default(self, session):
        project = _make_project()
        session.add(project)
        session.flush()

        run = _make_run(project.id)
        session.add(run)
        session.flush()

        assert run.id is not None
        assert run.status == "queued"
        assert run.train_result is None
        assert run.evaluation_result is None
        assert run.managed_artifact_path is None

    def test_jsonb_result_fields_round_trip(self, session):
        project = _make_project()
        session.add(project)
        session.flush()

        run = _make_run(
            project.id,
            train_result={"status": "success", "exit_code": 0},
            evaluation_result={"status": "success", "metrics": {"accuracy": 0.9}},
        )
        session.add(run)
        session.flush()
        session.expire(run)

        reloaded = session.get(ProjectRun, run.id)
        assert reloaded.train_result == {"status": "success", "exit_code": 0}
        assert reloaded.evaluation_result == {"status": "success", "metrics": {"accuracy": 0.9}}

    def test_runs_are_ordered_by_run_number_on_the_project(self, session):
        project = _make_project()
        session.add(project)
        session.flush()

        session.add(_make_run(project.id, run_number=2))
        session.add(_make_run(project.id, run_number=1))
        session.flush()
        session.expire(project)

        reloaded = session.get(Project, project.id)
        assert [r.run_number for r in reloaded.runs] == [1, 2]

    def test_deleting_a_project_deletes_its_runs(self, session):
        # Unlike Experiment.project_id (SET NULL) -- a run has no meaning
        # detached from the project it ran against, so this cascades.
        project = _make_project()
        session.add(project)
        session.flush()

        run = _make_run(project.id)
        session.add(run)
        session.flush()
        run_id = run.id

        session.delete(project)
        session.flush()
        session.expire_all()

        assert session.get(ProjectRun, run_id) is None
