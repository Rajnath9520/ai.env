import { describe, expect, it } from "vitest";
import { configReadToUpdate, suggestConfigFromScan } from "@/lib/project-config";
import type { ProjectConfigRead, ScanResultRead } from "@/types/api";

const EMPTY_SCAN: ScanResultRead = {
  root: "/projects/empty",
  language: null,
  framework: null,
  python_version: null,
  gpu: { available: false, name: null },
  training_candidates: [],
  evaluation_candidates: [],
  dataset_candidates: [],
  artifact_candidates: [],
  dependency_files: [],
};

const FULL_SCAN: ScanResultRead = {
  ...EMPTY_SCAN,
  root: "/projects/face-recognition",
  language: "python",
  framework: "pytorch",
  training_candidates: ["train.py"],
  evaluation_candidates: ["evaluate.py"],
  dataset_candidates: ["dataset/"],
  artifact_candidates: ["outputs/best_model.pt"],
  dependency_files: ["requirements.txt"],
};

describe("suggestConfigFromScan", () => {
  it("never invents a command when there are no candidates", () => {
    const draft = suggestConfigFromScan(EMPTY_SCAN, "empty-project");
    expect(draft.training.command).toBe("");
    expect(draft.training.entrypoint).toBeNull();
    expect(draft.evaluation.command).toBeNull();
    expect(draft.artifact.path).toBeNull();
    expect(draft.dataset.path).toBeNull();
  });

  it("drafts an editable command from the first candidate of each kind", () => {
    const draft = suggestConfigFromScan(FULL_SCAN, "face-recognition");
    expect(draft.name).toBe("face-recognition");
    expect(draft.project.framework).toBe("pytorch");
    expect(draft.training.entrypoint).toBe("train.py");
    expect(draft.training.command).toBe("python train.py --dataset {dataset}");
    expect(draft.evaluation.entrypoint).toBe("evaluate.py");
    expect(draft.evaluation.command).toBe("python evaluate.py --dataset {dataset}");
    expect(draft.artifact.path).toBe("outputs/best_model.pt");
  });

  it("strips the trailing slash from a candidate dataset directory", () => {
    const draft = suggestConfigFromScan(FULL_SCAN, "face-recognition");
    expect(draft.dataset.path).toBe("dataset");
  });
});

describe("configReadToUpdate", () => {
  it("adds the name back onto the confirmed contract", () => {
    const config: ProjectConfigRead = {
      project: { task: "face_recognition", framework: "pytorch" },
      training: { command: "python train.py", entrypoint: null, working_directory: ".", timeout_seconds: 3600 },
      evaluation: {
        command: null,
        entrypoint: null,
        working_directory: null,
        timeout_seconds: null,
        metrics: [],
        metrics_file: null,
        metrics_source: null,
      },
      artifact: { path: null, type: null },
      dataset: { path: null },
      curriculum: { strategy: "failure_driven" },
      optimization: { max_iterations: 10, target_metric: null, target_value: null, minimum_improvement: 0.005 },
      adapter: { type: "command", entrypoint: null, python: "python" },
    };

    const update = configReadToUpdate("face-recognition", config);
    expect(update.name).toBe("face-recognition");
    expect(update.training.command).toBe("python train.py");
  });
});
