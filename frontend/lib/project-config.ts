import {
  DEFAULT_PROJECT_CONFIG_UPDATE,
  type ProjectConfigRead,
  type ProjectConfigUpdate,
  type ScanResultRead,
} from "@/types/api";

/** Turns a scan result into a *draft* config the user can edit before
 * confirming (Section 6: "Never automatically assume train.py = training
 * entrypoint. It is only a candidate."). Every field here stays fully
 * editable in the form; nothing reaches the backend until the user
 * explicitly submits via `PUT /api/projects/{id}/config`. Picking the
 * first candidate in each list (rather than leaving the field blank) is a
 * starting point to edit, not a claim that it's correct. */
export function suggestConfigFromScan(scan: ScanResultRead, projectName: string): ProjectConfigUpdate {
  const trainingEntrypoint = scan.training_candidates[0] ?? null;
  const evaluationEntrypoint = scan.evaluation_candidates[0] ?? null;
  const datasetPath = scan.dataset_candidates[0]?.replace(/\/$/, "") ?? null;
  const artifactPath = scan.artifact_candidates[0] ?? null;

  return {
    ...DEFAULT_PROJECT_CONFIG_UPDATE,
    name: projectName,
    project: { task: null, framework: scan.framework },
    training: {
      ...DEFAULT_PROJECT_CONFIG_UPDATE.training,
      entrypoint: trainingEntrypoint,
      command: trainingEntrypoint ? `python ${trainingEntrypoint} --dataset {dataset}` : "",
    },
    evaluation: {
      ...DEFAULT_PROJECT_CONFIG_UPDATE.evaluation,
      entrypoint: evaluationEntrypoint,
      command: evaluationEntrypoint ? `python ${evaluationEntrypoint} --dataset {dataset}` : null,
    },
    artifact: { path: artifactPath, type: null },
    dataset: { path: datasetPath },
  };
}

/** Converts the confirmed contract (as read back from `GET
 * /api/projects/{id}/config`) into the shape the edit form uses, so
 * reopening "Edit Configuration" on an already-configured project starts
 * from its actual saved values rather than the scan suggestions again. */
export function configReadToUpdate(name: string, config: ProjectConfigRead): ProjectConfigUpdate {
  return { name, ...config };
}
