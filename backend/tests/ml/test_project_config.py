import pytest

from app.ml.project.config import AIEnvConfig, EvaluationConfig, TrainingConfig

SECTION_8_EXAMPLE = """
project:
  name: face-recognition
  task: face_recognition
  framework: pytorch

training:
  entrypoint: train.py
  command: python train.py --dataset {dataset}
  working_directory: .
  timeout_seconds: 86400

evaluation:
  entrypoint: evaluate.py
  command: python evaluate.py --dataset {dataset}
  metrics:
    - accuracy
    - precision
    - recall
    - f1
  metrics_file: outputs/metrics.json

artifact:
  path: outputs/best_model.pt
  type: pytorch_checkpoint

dataset:
  path: ./dataset

curriculum:
  strategy: failure_driven

optimization:
  max_iterations: 10
  target_metric: f1
  target_value: 0.95
  minimum_improvement: 0.005
"""


class TestLoadingSection8Example:
    def test_parses_every_section(self):
        config = AIEnvConfig.from_yaml(SECTION_8_EXAMPLE)

        assert config.project.name == "face-recognition"
        assert config.project.task == "face_recognition"
        assert config.project.framework == "pytorch"
        assert config.training.command == "python train.py --dataset {dataset}"
        assert config.training.timeout_seconds == 86400
        assert config.evaluation.metrics == ["accuracy", "precision", "recall", "f1"]
        assert config.evaluation.metrics_file == "outputs/metrics.json"
        assert config.artifact.path == "outputs/best_model.pt"
        assert config.dataset.path == "./dataset"
        assert config.curriculum.strategy == "failure_driven"
        assert config.optimization.target_value == 0.95

    def test_round_trips_through_yaml(self):
        config = AIEnvConfig.from_yaml(SECTION_8_EXAMPLE)
        reloaded = AIEnvConfig.from_yaml(config.to_yaml())
        assert reloaded == config

    def test_save_and_load(self, tmp_path):
        config = AIEnvConfig.from_yaml(SECTION_8_EXAMPLE)
        path = tmp_path / "ai-env.yaml"
        config.save(path)
        assert AIEnvConfig.load(path) == config


class TestValidation:
    def test_missing_project_name_is_rejected(self):
        with pytest.raises(ValueError, match="project.name"):
            AIEnvConfig.from_dict({"training": {"command": "python train.py"}})

    def test_missing_training_command_is_rejected(self):
        with pytest.raises(ValueError, match="training.command"):
            AIEnvConfig.from_dict({"project": {"name": "x"}})

    def test_empty_training_command_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            TrainingConfig(command="   ")

    def test_unknown_curriculum_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="curriculum.strategy"):
            AIEnvConfig.from_dict(
                {
                    "project": {"name": "x"},
                    "training": {"command": "python train.py"},
                    "curriculum": {"strategy": "not_a_real_strategy"},
                }
            )

    def test_metrics_source_file_requires_metrics_file(self):
        with pytest.raises(ValueError, match="metrics_file"):
            EvaluationConfig(metrics_source="file")

    def test_metrics_source_must_be_file_or_stdout(self):
        with pytest.raises(ValueError, match="metrics_source"):
            EvaluationConfig(metrics_source="somewhere_else")

    def test_defaults_are_honest_when_sections_are_omitted(self):
        config = AIEnvConfig.from_dict(
            {"project": {"name": "minimal"}, "training": {"command": "python train.py"}}
        )
        assert config.evaluation.command is None
        assert config.artifact.path is None
        assert config.dataset.path is None
        assert config.curriculum.strategy == "failure_driven"
