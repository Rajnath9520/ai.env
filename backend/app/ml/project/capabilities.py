"""Section 9/84 -- the capability model.

A `TrainingAdapter` declares which of these it actually implements, rather
than every adapter being forced to implement every method of the
interface (Section 9: "Do NOT require every adapter to implement every
method. Use capability declarations."). The UI/service layer is expected
to check `supports()` before offering an action (Section 84: "If
DATASET_GENERATION is unavailable, do not show a misleading automatic
dataset-generation workflow").
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    CHECKPOINT = "checkpoint"
    DATASET_GENERATION = "dataset_generation"
    ENVIRONMENT_INTERACTION = "environment_interaction"
    TRAJECTORY_COLLECTION = "trajectory_collection"
