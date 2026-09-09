"""CLI for Phase 15: `python -m app.ml.experiment.run_experiment --config path/to/experiment.yaml`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.ml.experiment.config import ExperimentConfig
from app.ml.experiment.manager import ExperimentManager

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an experiment from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment.yaml")
    parser.add_argument(
        "--output-path", type=str, default=None, help="Defaults to experiments/results/<name>.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(args.config)
    manager = ExperimentManager(config)
    result = manager.run()

    output_path = Path(args.output_path or f"experiments/results/{config.name}.json")
    result.save(output_path)
    logging.getLogger("run_experiment").info("results_saved path=%s", output_path)


if __name__ == "__main__":
    main()
