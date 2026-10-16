"use client";

import { useCallback, useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { CapabilityBadge, ProjectStatusBadge, RunStatusBadge } from "@/components/panels/Primitives";
import { api, extractErrorMessage } from "@/lib/api-client";
import { configReadToUpdate, suggestConfigFromScan } from "@/lib/project-config";
import {
  DEFAULT_PROJECT_CONFIG_UPDATE,
  type AdapterType,
  type AgentRead,
  type CurriculumType,
  type MetricsSource,
  type ProjectConfigUpdate,
  type ProjectDetailRead,
  type ProjectProgressRead,
  type ProjectRunRead,
} from "@/types/api";

const CURRICULUM_STRATEGIES: CurriculumType[] = ["failure_driven", "generic_adaptive", "fixed", "random"];
const NON_TERMINAL_RUN_STATUSES = new Set(["queued", "running_training", "running_evaluation"]);
const RUN_POLL_INTERVAL_MS = 2000;

function TextField({
  label,
  value,
  onChange,
  placeholder,
  mono,
  required,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  mono?: boolean;
  required?: boolean;
  hint?: string;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-app-mid">{label}</span>
      <input
        required={required}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className={`bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi focus:outline-none focus:border-app-blue ${mono ? "font-mono" : ""}`}
      />
      {hint && <span className="text-[11px] text-app-low">{hint}</span>}
    </label>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  step?: number;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-app-mid">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi font-mono focus:outline-none focus:border-app-blue"
      />
    </label>
  );
}

function SelectField<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-app-mid">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="bg-app-panel2 border border-app-border rounded px-2 py-1.5 text-app-hi focus:outline-none focus:border-app-blue"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function ProjectDetailPage({ params }: { params: { id: string } }) {
  const projectId = Number(params.id);

  const [project, setProject] = useState<ProjectDetailRead | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  const [form, setForm] = useState<ProjectConfigUpdate>(DEFAULT_PROJECT_CONFIG_UPDATE);
  const [savingConfig, setSavingConfig] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [configSaved, setConfigSaved] = useState(false);

  const [validating, setValidating] = useState(false);
  const [validateError, setValidateError] = useState<string | null>(null);

  const [runs, setRuns] = useState<ProjectRunRead[] | null>(null);
  const [startingRun, setStartingRun] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [expandedRunId, setExpandedRunId] = useState<number | null>(null);
  const [cancellingRunId, setCancellingRunId] = useState<number | null>(null);
  const [progress, setProgress] = useState<ProjectProgressRead | null>(null);
  const [agents, setAgents] = useState<AgentRead[] | null>(null);
  const [assigningAgent, setAssigningAgent] = useState(false);
  const [agentError, setAgentError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const detail = await api.getProject(projectId);
    setProject(detail);

    if (detail.config_yaml) {
      const config = await api.getProjectConfig(projectId);
      setForm(configReadToUpdate(detail.name, config));
    } else if (detail.detected_config) {
      setForm(suggestConfigFromScan(detail.detected_config, detail.name));
    } else {
      setForm({ ...DEFAULT_PROJECT_CONFIG_UPDATE, name: detail.name });
    }
    return detail;
  }, [projectId]);

  useEffect(() => {
    refresh().catch((e) => setLoadError(extractErrorMessage(e, "Failed to load project")));
    api.listLocalAgents().then(setAgents).catch(() => {});
  }, [refresh]);

  async function handleAssignAgent(agentId: number | null) {
    setAssigningAgent(true);
    setAgentError(null);
    try {
      await api.assignProjectAgent(projectId, agentId);
      await refresh();
    } catch (e) {
      setAgentError(extractErrorMessage(e, "Failed to assign agent"));
    } finally {
      setAssigningAgent(false);
    }
  }

  const refreshRuns = useCallback(async () => {
    const list = await api.listProjectRuns(projectId);
    setRuns(list);
    return list;
  }, [projectId]);

  const refreshProgress = useCallback(async () => {
    setProgress(await api.getProjectProgress(projectId));
  }, [projectId]);

  useEffect(() => {
    refreshRuns().catch(() => {});
    refreshProgress().catch(() => {});
  }, [refreshRuns, refreshProgress]);

  useEffect(() => {
    if (!runs?.some((r) => NON_TERMINAL_RUN_STATUSES.has(r.status))) return;
    const interval = setInterval(() => {
      refreshRuns().catch(() => {});
      refreshProgress().catch(() => {});
    }, RUN_POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [runs, refreshRuns, refreshProgress]);

  async function handleStartRun() {
    setStartingRun(true);
    setRunError(null);
    try {
      await api.startProjectRun(projectId);
      await refreshRuns();
      await refreshProgress();
    } catch (e) {
      setRunError(extractErrorMessage(e, "Failed to start run"));
    } finally {
      setStartingRun(false);
    }
  }

  async function handleCancelRun(runId: number) {
    setCancellingRunId(runId);
    try {
      await api.cancelProjectRun(projectId, runId);
      await refreshRuns();
    } catch (e) {
      setRunError(extractErrorMessage(e, "Failed to cancel run"));
    } finally {
      setCancellingRunId(null);
    }
  }

  async function handleScan() {
    setScanning(true);
    setScanError(null);
    try {
      await api.scanProject(projectId);
      await refresh();
    } catch (e) {
      setScanError(extractErrorMessage(e, "Scan failed"));
    } finally {
      setScanning(false);
    }
  }

  function prefillFromScan() {
    if (project?.detected_config) {
      setForm(suggestConfigFromScan(project.detected_config, project.name));
      setConfigSaved(false);
    }
  }

  async function handleSaveConfig(e: React.FormEvent) {
    e.preventDefault();
    setSavingConfig(true);
    setConfigError(null);
    setConfigSaved(false);
    try {
      await api.updateProjectConfig(projectId, form);
      await refresh();
      setConfigSaved(true);
    } catch (err) {
      setConfigError(extractErrorMessage(err, "Failed to save configuration"));
    } finally {
      setSavingConfig(false);
    }
  }

  async function handleValidate() {
    setValidating(true);
    setValidateError(null);
    try {
      await api.validateProject(projectId);
      await refresh();
    } catch (e) {
      setValidateError(extractErrorMessage(e, "Validation failed"));
    } finally {
      setValidating(false);
    }
  }

  if (loadError) {
    return (
      <Shell title="PROJECT">
        <div className="text-app-red text-xs">{loadError}</div>
      </Shell>
    );
  }

  if (!project) {
    return (
      <Shell title="PROJECT">
        <div className="text-app-mid text-xs">Loading...</div>
      </Shell>
    );
  }

  const detected = project.detected_config;
  const validation = project.validation_result;

  return (
    <Shell title={project.name.toUpperCase()} subtitle={project.root_path}>
      <div className="grid grid-cols-3 gap-4">
        <Panel title="OVERVIEW" className="col-span-2">
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <div className="text-app-low text-[10px] uppercase tracking-wider">Status</div>
              <ProjectStatusBadge status={project.status} />
            </div>
            <div>
              <div className="text-app-low text-[10px] uppercase tracking-wider">Source</div>
              <div className="text-app-hi uppercase text-[11px]">{project.source}</div>
            </div>
            <div>
              <div className="text-app-low text-[10px] uppercase tracking-wider">Executes on</div>
              <select
                value={project.agent_id ?? ""}
                disabled={assigningAgent}
                onChange={(e) => handleAssignAgent(e.target.value === "" ? null : Number(e.target.value))}
                className="bg-app-panel2 border border-app-border rounded px-1.5 py-0.5 text-app-hi text-[11px] focus:outline-none focus:border-app-blue"
              >
                <option value="">Backend (local)</option>
                {agents?.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.name} ({a.status})
                  </option>
                ))}
              </select>
              {agentError && <div className="text-app-red text-[10px] mt-0.5">{agentError}</div>}
            </div>
            <div>
              <div className="text-app-low text-[10px] uppercase tracking-wider">Language</div>
              <div className="text-app-hi">{project.language ?? "--"}</div>
            </div>
            <div>
              <div className="text-app-low text-[10px] uppercase tracking-wider">Framework</div>
              <div className="text-app-hi">{project.framework ?? "--"}</div>
            </div>
            <div>
              <div className="text-app-low text-[10px] uppercase tracking-wider">Task</div>
              <div className="text-app-hi">{project.task ?? "--"}</div>
            </div>
            <div>
              <div className="text-app-low text-[10px] uppercase tracking-wider">Registered</div>
              <div className="text-app-hi">{new Date(project.created_at).toLocaleString()}</div>
            </div>
          </div>
          <div className="mt-3">
            <div className="text-app-low text-[10px] uppercase tracking-wider mb-1">Capabilities</div>
            {project.capabilities.length === 0 ? (
              <div className="text-app-low text-[11px]">
                None yet -- confirm a configuration below to declare what this project supports.
              </div>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {project.capabilities.map((c) => (
                  <CapabilityBadge key={c} capability={c} />
                ))}
              </div>
            )}
          </div>
        </Panel>

        <Panel
          title="SCAN"
          right={
            <button
              onClick={handleScan}
              disabled={scanning}
              className="px-2.5 py-1 rounded bg-app-panel2 border border-app-border text-[11px] text-app-hi hover:border-app-blue disabled:opacity-50"
            >
              {scanning ? "Scanning..." : detected ? "Rescan" : "Scan"}
            </button>
          }
        >
          {scanError && <div className="text-app-red text-[11px] mb-2">{scanError}</div>}
          {!detected ? (
            <div className="text-app-mid text-[11px]">No scan yet. Detected values are suggestions only.</div>
          ) : (
            <div className="space-y-2 text-[11px]">
              <div className="text-app-low uppercase tracking-wider text-[10px]">Detected / Suggested</div>
              <div>
                <span className="text-app-mid">Training candidates: </span>
                <span className="text-app-hi font-mono">
                  {detected.training_candidates.join(", ") || "none found"}
                </span>
              </div>
              <div>
                <span className="text-app-mid">Evaluation candidates: </span>
                <span className="text-app-hi font-mono">
                  {detected.evaluation_candidates.join(", ") || "none found"}
                </span>
              </div>
              <div>
                <span className="text-app-mid">Dataset candidates: </span>
                <span className="text-app-hi font-mono">
                  {detected.dataset_candidates.join(", ") || "none found"}
                </span>
              </div>
              <div>
                <span className="text-app-mid">Artifact candidates: </span>
                <span className="text-app-hi font-mono">
                  {detected.artifact_candidates.join(", ") || "none found"}
                </span>
              </div>
              <div>
                <span className="text-app-mid">GPU: </span>
                <span className="text-app-hi">
                  {detected.gpu.available ? (detected.gpu.name ?? "available") : "none detected (CPU)"}
                </span>
              </div>
            </div>
          )}
        </Panel>
      </div>

      <Panel
        title="CONFIGURATION"
        right={
          detected && (
            <button
              type="button"
              onClick={prefillFromScan}
              className="px-2.5 py-1 rounded bg-app-panel2 border border-app-border text-[11px] text-app-hi hover:border-app-blue"
            >
              Prefill from scan
            </button>
          )
        }
      >
        <form onSubmit={handleSaveConfig} className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <TextField label="Project name" value={form.name} onChange={(v) => setForm({ ...form, name: v })} required />
            <TextField
              label="Task"
              value={form.project.task ?? ""}
              placeholder="e.g. face_recognition"
              onChange={(v) => setForm({ ...form, project: { ...form.project, task: v || null } })}
            />
            <TextField
              label="Framework"
              value={form.project.framework ?? ""}
              placeholder="e.g. pytorch"
              onChange={(v) => setForm({ ...form, project: { ...form.project, framework: v || null } })}
            />
          </div>

          <div className="border-t border-app-border pt-4">
            <div className="text-[11px] font-semibold tracking-widest text-app-mid mb-3">ADAPTER</div>
            <div className="grid grid-cols-3 gap-3">
              <SelectField<AdapterType>
                label="Type"
                value={form.adapter.type}
                onChange={(v) => setForm({ ...form, adapter: { ...form.adapter, type: v } })}
                options={[
                  { value: "command", label: "Command (run train.py / evaluate.py as shell commands)" },
                  { value: "custom", label: "Custom (.ai-env/adapter.py with get_adapter())" },
                ]}
              />
              {form.adapter.type === "custom" && (
                <>
                  <TextField
                    label="Adapter entrypoint"
                    mono
                    value={form.adapter.entrypoint ?? ""}
                    placeholder=".ai-env/adapter.py"
                    onChange={(v) => setForm({ ...form, adapter: { ...form.adapter, entrypoint: v || null } })}
                  />
                  <TextField
                    label="Python interpreter"
                    mono
                    value={form.adapter.python}
                    onChange={(v) => setForm({ ...form, adapter: { ...form.adapter, python: v } })}
                    hint="Must have whatever packages .ai-env/adapter.py imports -- not necessarily ai.env's own interpreter."
                  />
                </>
              )}
            </div>
            {form.adapter.type === "custom" && (
              <div className="text-[11px] text-app-low mt-2">
                Runs in its own subprocess, never imported into this app -- see docs/custom-adapters.md.
                Capabilities (train/evaluate/predict) are discovered from the module, not declared here; check
                Validation below after saving.
              </div>
            )}
          </div>

          <div className="border-t border-app-border pt-4">
            <div className="text-[11px] font-semibold tracking-widest text-app-mid mb-3">
              TRAINING{form.adapter.type === "custom" && " (unused for custom adapters)"}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <TextField
                label="Command"
                mono
                required
                value={form.training.command}
                placeholder="python train.py --dataset {dataset}"
                onChange={(v) => setForm({ ...form, training: { ...form.training, command: v } })}
                hint="Placeholders: {project} {dataset} {experiment} {run} {checkpoint} {output} {config}"
              />
              <TextField
                label="Entrypoint (informational)"
                mono
                value={form.training.entrypoint ?? ""}
                onChange={(v) => setForm({ ...form, training: { ...form.training, entrypoint: v || null } })}
              />
              <TextField
                label="Working directory"
                mono
                value={form.training.working_directory}
                onChange={(v) => setForm({ ...form, training: { ...form.training, working_directory: v } })}
              />
              <NumberField
                label="Timeout (seconds)"
                min={1}
                value={form.training.timeout_seconds}
                onChange={(v) => setForm({ ...form, training: { ...form.training, timeout_seconds: v } })}
              />
            </div>
          </div>

          <div className="border-t border-app-border pt-4">
            <div className="text-[11px] font-semibold tracking-widest text-app-mid mb-3">EVALUATION</div>
            <div className="grid grid-cols-2 gap-3">
              <TextField
                label="Command"
                mono
                value={form.evaluation.command ?? ""}
                placeholder="python evaluate.py --dataset {dataset}"
                onChange={(v) => setForm({ ...form, evaluation: { ...form.evaluation, command: v || null } })}
              />
              <TextField
                label="Entrypoint (informational)"
                mono
                value={form.evaluation.entrypoint ?? ""}
                onChange={(v) => setForm({ ...form, evaluation: { ...form.evaluation, entrypoint: v || null } })}
              />
              <SelectField<MetricsSource | "none">
                label="Metrics source"
                value={form.evaluation.metrics_source ?? "none"}
                onChange={(v) =>
                  setForm({
                    ...form,
                    evaluation: { ...form.evaluation, metrics_source: v === "none" ? null : v },
                  })
                }
                options={[
                  { value: "none", label: "Not configured (no metrics collected)" },
                  { value: "file", label: "JSON file written by the evaluation command" },
                  { value: "stdout", label: "Last line of stdout is a JSON object" },
                ]}
              />
              {form.evaluation.metrics_source === "file" && (
                <TextField
                  label="Metrics file (relative to working directory)"
                  mono
                  value={form.evaluation.metrics_file ?? ""}
                  placeholder="outputs/metrics.json"
                  onChange={(v) =>
                    setForm({ ...form, evaluation: { ...form.evaluation, metrics_file: v || null } })
                  }
                />
              )}
            </div>
          </div>

          <div className="border-t border-app-border pt-4">
            <div className="text-[11px] font-semibold tracking-widest text-app-mid mb-3">ARTIFACT &amp; DATASET</div>
            <div className="grid grid-cols-3 gap-3">
              <TextField
                label="Artifact path"
                mono
                value={form.artifact.path ?? ""}
                placeholder="outputs/best_model.pt"
                onChange={(v) => setForm({ ...form, artifact: { ...form.artifact, path: v || null } })}
              />
              <TextField
                label="Artifact type"
                value={form.artifact.type ?? ""}
                placeholder="pytorch_checkpoint"
                onChange={(v) => setForm({ ...form, artifact: { ...form.artifact, type: v || null } })}
              />
              <TextField
                label="Dataset path"
                mono
                value={form.dataset.path ?? ""}
                placeholder="dataset"
                onChange={(v) => setForm({ ...form, dataset: { path: v || null } })}
              />
            </div>
          </div>

          <div className="border-t border-app-border pt-4">
            <div className="text-[11px] font-semibold tracking-widest text-app-mid mb-3">
              CURRICULUM &amp; OPTIMIZATION
            </div>
            <div className="grid grid-cols-4 gap-3">
              <SelectField<CurriculumType>
                label="Curriculum strategy"
                value={form.curriculum.strategy}
                onChange={(v) => setForm({ ...form, curriculum: { strategy: v } })}
                options={CURRICULUM_STRATEGIES.map((s) => ({ value: s, label: s.replace(/_/g, " ") }))}
              />
              <NumberField
                label="Max iterations"
                min={1}
                value={form.optimization.max_iterations}
                onChange={(v) => setForm({ ...form, optimization: { ...form.optimization, max_iterations: v } })}
              />
              <TextField
                label="Target metric"
                value={form.optimization.target_metric ?? ""}
                placeholder="f1"
                onChange={(v) =>
                  setForm({ ...form, optimization: { ...form.optimization, target_metric: v || null } })
                }
              />
              <NumberField
                label="Target value"
                step={0.01}
                value={form.optimization.target_value ?? 0}
                onChange={(v) => setForm({ ...form, optimization: { ...form.optimization, target_value: v } })}
              />
            </div>
            <div className="text-[11px] text-app-low mt-2">
              Stored as part of the contract for a future adaptive-loop phase -- nothing reads these yet (see
              docs/projects-and-adapters.md).
            </div>
          </div>

          {configError && <div className="text-app-red text-xs">{configError}</div>}
          {configSaved && !configError && <div className="text-app-green text-xs">Configuration saved.</div>}

          <button
            type="submit"
            disabled={savingConfig}
            className="px-4 py-2 rounded bg-app-blue text-white text-[12px] font-semibold hover:bg-[#2f78ea] disabled:opacity-50"
          >
            {savingConfig ? "Saving..." : "Save Configuration"}
          </button>
        </form>
      </Panel>

      <Panel
        title="VALIDATION"
        right={
          <button
            onClick={handleValidate}
            disabled={validating || !project.config_yaml}
            className="px-2.5 py-1 rounded bg-app-panel2 border border-app-border text-[11px] text-app-hi hover:border-app-blue disabled:opacity-50"
          >
            {validating ? "Validating..." : "Validate"}
          </button>
        }
      >
        <div className="text-[11px] text-app-low mb-3">
          Checks paths and configuration only (does it exist, is a command configured) -- it does not run your
          training or evaluation command.
        </div>
        {!project.config_yaml ? (
          <div className="text-app-mid text-[11px]">Save a configuration above before validating.</div>
        ) : validateError ? (
          <div className="text-app-red text-[11px]">{validateError}</div>
        ) : !validation ? (
          <div className="text-app-mid text-[11px]">Not validated yet.</div>
        ) : validation.issues.length === 0 ? (
          <div className="text-app-green text-[11px]">Valid -- no issues found.</div>
        ) : (
          <ul className="space-y-1.5 text-[11px]">
            {validation.issues.map((issue, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className={issue.severity === "error" ? "text-app-red" : "text-app-orange"}>
                  {issue.severity === "error" ? "ERROR" : "WARN"}
                </span>
                <span className="text-app-mid">
                  <span className="text-app-hi font-mono">{issue.field}</span>: {issue.message}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel
        title="RUNS"
        right={
          <button
            onClick={handleStartRun}
            disabled={startingRun || !project.config_yaml}
            className="px-2.5 py-1 rounded bg-app-blue text-white text-[11px] font-semibold hover:bg-[#2f78ea] disabled:opacity-50"
          >
            {startingRun ? "Starting..." : "Start Run"}
          </button>
        }
      >
        <div className="text-[11px] text-app-low mb-3">
          Runs your configured training command and, if configured, the evaluation command right after it.
          Artifacts are archived per run under ai.env-managed storage so a later run can never silently
          overwrite an earlier one&rsquo;s checkpoint.
        </div>
        {runError && <div className="text-app-red text-[11px] mb-2">{runError}</div>}
        {!project.config_yaml ? (
          <div className="text-app-mid text-[11px]">Save a configuration above before starting a run.</div>
        ) : !runs || runs.length === 0 ? (
          <div className="text-app-mid text-[11px]">No runs yet.</div>
        ) : (
          <div className="space-y-2">
            {runs
              .slice()
              .sort((a, b) => b.run_number - a.run_number)
              .map((run) => (
                <div key={run.id} className="border border-app-border rounded">
                  <button
                    onClick={() => setExpandedRunId(expandedRunId === run.id ? null : run.id)}
                    className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-app-panel2"
                  >
                    <div className="flex items-center gap-3 text-[11px]">
                      <span className="text-app-hi font-mono">Run {run.run_number}</span>
                      <RunStatusBadge status={run.status} />
                      {run.evaluation_result && Object.keys(run.evaluation_result.metrics).length > 0 && (
                        <span className="text-app-mid font-mono">
                          {Object.entries(run.evaluation_result.metrics)
                            .map(([k, v]) => `${k}=${v}`)
                            .join("  ")}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {NON_TERMINAL_RUN_STATUSES.has(run.status) && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelRun(run.id);
                          }}
                          disabled={cancellingRunId === run.id}
                          className="px-2 py-0.5 rounded border border-app-border text-[10px] text-app-mid hover:border-app-red hover:text-app-red disabled:opacity-50"
                        >
                          {cancellingRunId === run.id ? "Cancelling..." : "Cancel"}
                        </button>
                      )}
                      <span className="text-app-low text-[10px]">{new Date(run.started_at).toLocaleString()}</span>
                    </div>
                  </button>

                  {expandedRunId === run.id && (
                    <div className="px-3 pb-3 pt-1 space-y-3 border-t border-app-border text-[11px]">
                      {run.error && <div className="text-app-red">{run.error}</div>}

                      {run.train_result && (
                        <div>
                          <div className="text-app-low uppercase tracking-wider text-[10px] mb-1">Training</div>
                          <div className="text-app-mid">
                            status: <span className="text-app-hi">{run.train_result.status}</span> · exit code:{" "}
                            <span className="text-app-hi">{run.train_result.exit_code ?? "--"}</span> · duration:{" "}
                            <span className="text-app-hi">{run.train_result.duration_seconds.toFixed(1)}s</span>
                          </div>
                          {run.train_result.stderr_tail && (
                            <pre className="mt-1 max-h-32 overflow-auto bg-app-panel2 border border-app-border rounded p-2 font-mono text-[10px] text-app-mid whitespace-pre-wrap">
                              {run.train_result.stderr_tail}
                            </pre>
                          )}
                        </div>
                      )}

                      {run.evaluation_result && (
                        <div>
                          <div className="text-app-low uppercase tracking-wider text-[10px] mb-1">Evaluation</div>
                          <div className="text-app-mid">
                            status: <span className="text-app-hi">{run.evaluation_result.status}</span>
                          </div>
                          {Object.keys(run.evaluation_result.metrics).length > 0 && (
                            <div className="flex flex-wrap gap-3 mt-1">
                              {Object.entries(run.evaluation_result.metrics).map(([k, v]) => (
                                <div key={k}>
                                  <span className="text-app-low">{k}: </span>
                                  <span className="text-app-hi font-mono">{v}</span>
                                </div>
                              ))}
                            </div>
                          )}
                          {run.evaluation_result.warnings.length > 0 && (
                            <div className="text-app-orange mt-1">{run.evaluation_result.warnings.join(" ")}</div>
                          )}
                        </div>
                      )}

                      {run.managed_artifact_path && (
                        <div>
                          <div className="text-app-low uppercase tracking-wider text-[10px] mb-1">
                            Archived artifact
                          </div>
                          <div className="text-app-hi font-mono break-all">{run.managed_artifact_path}</div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
          </div>
        )}
      </Panel>

      <Panel title="PROGRESS">
        <div className="text-[11px] text-app-low mb-3">
          Real run-over-run metric trends only -- never a fabricated weakness score. This stops short of a
          per-condition breakdown (lighting, occlusion, etc.) since nothing here has structured evaluation data
          to break down; see docs/projects-and-adapters.md.
        </div>
        {!progress || progress.completed_run_count === 0 ? (
          <div className="text-app-mid text-[11px]">
            No completed runs with metrics yet -- start a run above to see progress here.
          </div>
        ) : (
          <div className="space-y-4">
            {progress.target_metric && (
              <div className="text-[12px]">
                <span className="text-app-low">Target: </span>
                <span className="text-app-hi font-mono">{progress.target_metric}</span>
                {progress.target_value !== null && (
                  <>
                    <span className="text-app-low"> &rarr; </span>
                    <span className="text-app-hi font-mono">{progress.target_value}</span>
                  </>
                )}
                {progress.latest_target_value === null ? (
                  <span className="text-app-mid"> -- not yet reported by any run</span>
                ) : (
                  <>
                    <span className="text-app-low"> · latest </span>
                    <span className="text-app-hi font-mono">{progress.latest_target_value}</span>
                    {progress.distance_to_target !== null && (
                      <>
                        <span className="text-app-low"> · distance </span>
                        <span className="text-app-hi font-mono">{progress.distance_to_target.toFixed(4)}</span>
                      </>
                    )}
                    {progress.moved_closer_to_target !== null && (
                      <span className={progress.moved_closer_to_target ? "text-app-green" : "text-app-orange"}>
                        {" "}
                        ({progress.moved_closer_to_target ? "closer than last run" : "not closer than last run"})
                      </span>
                    )}
                  </>
                )}
              </div>
            )}

            {progress.target_metric &&
              progress.metrics.find((m) => m.metric === progress.target_metric)?.points &&
              (progress.metrics.find((m) => m.metric === progress.target_metric)?.points.length ?? 0) > 1 && (
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={progress.metrics.find((m) => m.metric === progress.target_metric)!.points}
                      margin={{ top: 4, right: 8, left: -20, bottom: 0 }}
                    >
                      <CartesianGrid stroke="#1a2236" strokeDasharray="2 4" />
                      <XAxis dataKey="run_number" stroke="#566078" fontSize={10} />
                      <YAxis stroke="#566078" fontSize={10} />
                      <Tooltip contentStyle={{ background: "#0f1524", border: "1px solid #1f2a40", fontSize: 11 }} />
                      <Line type="monotone" dataKey="value" stroke="#3f8cff" dot strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

            <div>
              <div className="text-app-low text-[10px] uppercase tracking-wider mb-1.5">All metrics</div>
              <table className="w-full text-[11px]">
                <tbody>
                  {progress.metrics.map((m) => (
                    <tr key={m.metric} className="border-t border-[#1a2033]">
                      <td className="py-1 font-mono text-app-hi">{m.metric}</td>
                      <td className="py-1 text-app-hi font-mono text-right">{m.latest_value}</td>
                      <td className="py-1 text-app-mid font-mono text-right w-24">
                        {m.delta_from_previous === null
                          ? "--"
                          : `${m.delta_from_previous >= 0 ? "+" : ""}${m.delta_from_previous.toFixed(4)}`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Panel>
    </Shell>
  );
}
