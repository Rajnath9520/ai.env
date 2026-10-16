// Mirrors backend/app/schemas/experiment.py exactly -- keep these two in
// sync by hand for now (no generated client yet; see docs/api.md).

export type CurriculumType = "random" | "fixed" | "generic_adaptive" | "failure_driven";
export type ExperimentStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "stopped"
  // Phase 28 (Section 18) -- mode-aware statuses. Never force an
  // environment_only or external experiment through a training-shaped
  // status like "running" once it's doing something more specific.
  | "generating"
  | "validating"
  | "ready"
  | "analyzing"
  | "waiting_for_agent"
  | "agent_connected"
  | "interacting"
  | "analyzing_trajectory"
  | "adapting";

// Phase 28 (Section 6) -- the one switch that decides whether any model is
// ever created. "environment_only" is the default and must stay that way.
export type AgentMode = "environment_only" | "external" | "demo_ppo";
export type GenerationStrategy = "diverse" | "progressive" | "random";

export interface ExperimentCreate {
  name: string;
  agent_mode: AgentMode;
  curriculum_type: CurriculumType;
  seed: number;
  grid_size: number;
  obstacle_density: number;
  dead_end_probability: number;
  path_complexity: number;
  max_steps_per_episode: number;
  n_curriculum_steps: number;
  timesteps_per_step: number;
  eval_episodes: number;
  n_candidates: number;
  trajectories_per_analysis: number;
  generation_strategy: GenerationStrategy;
  run_generalization: boolean;
  generalization_envs_per_split: number;
  generalization_episodes_per_config: number;
}

export const DEFAULT_EXPERIMENT_CREATE: ExperimentCreate = {
  name: "",
  agent_mode: "environment_only",
  curriculum_type: "failure_driven",
  seed: 42,
  grid_size: 10,
  obstacle_density: 0.15,
  dead_end_probability: 0.15,
  path_complexity: 0.15,
  max_steps_per_episode: 60,
  n_curriculum_steps: 20,
  timesteps_per_step: 10000,
  eval_episodes: 20,
  n_candidates: 5,
  trajectories_per_analysis: 8,
  generation_strategy: "diverse",
  run_generalization: true,
  generalization_envs_per_split: 10,
  generalization_episodes_per_config: 5,
};

export interface ExperimentRead {
  id: number;
  name: string;
  agent_mode: AgentMode;
  curriculum_type: CurriculumType;
  seed: number;
  status: ExperimentStatus;
  created_at: string;
}

export interface ExperimentStatusRead {
  id: number;
  status: ExperimentStatus;
  message?: string | null;
}

export interface EnvironmentRead {
  id: number;
  environment_key: string;
  params: Record<string, unknown>;
  difficulty: number;
  target_weakness: string | null;
  split: string;
}

export interface EpisodeRead {
  id: number;
  training_run_id: number;
  episode_number: number;
  reward: number;
  steps: number;
  success: boolean;
  collisions: number;
}

export interface TrajectoryRead {
  id: number;
  episode_id: number;
  trajectory_length: number;
  unique_states_visited: number;
  repeated_states: number;
  direction_reversals: number;
  distance_to_goal: number;
}

export interface FailureRead {
  id: number;
  episode_id: number;
  failure_type: string;
  score: number;
  confidence: number;
  features: Record<string, unknown>;
}

export interface WeaknessRead {
  id: number;
  training_run_id: number;
  obstacle_avoidance: number;
  exploration: number;
  dead_end_recovery: number;
  path_efficiency: number;
  short_horizon_planning: number;
  long_horizon_planning: number;
  primary_weakness: string;
  secondary_weaknesses: string[];
}

export interface LearningProgressRead {
  id: number;
  environment_id: number;
  metric: string;
  performance_before: number;
  performance_after: number;
  learning_progress: number;
}

export interface GeneralizationRead {
  id: number;
  split: "validation" | "test" | "ood";
  success_rate: number;
  average_reward: number;
  average_steps: number;
  collision_rate: number;
}

export interface EnvironmentLayoutRead {
  environment_id: number;
  environment_key: string;
  grid_size: number;
  agent_start: [number, number];
  goal: [number, number];
  obstacles: [number, number][];
}

export interface MetricsRead {
  experiment_id: number;
  agent_mode: AgentMode;
  status: ExperimentStatus;
  n_environments_trained: number;
  latest_success_rate: number | null;
  latest_difficulty: number | null;
  latest_primary_weakness: string | null;
  latest_learning_progress: number | null;
}

export interface ComparisonEntry {
  experiment_id: number;
  name: string;
  agent_mode: AgentMode;
  curriculum_type: CurriculumType;
  status: ExperimentStatus;
  final_success_rate: number | null;
  generalization: Record<string, GeneralizationRead>;
}

// Phase 28 (Section 13) -- one environment produced by environment-only
// generation. No reward/episode/success-rate fields: those describe agent
// behavior, which does not exist in this mode.
export interface GeneratedEnvironmentRead {
  step_index: number;
  environment_id: string;
  grid_size: number;
  obstacle_density: number;
  dead_end_probability: number;
  path_complexity: number;
  difficulty: number;
  generation_score: number;
  generation_reason: string;
  metrics: {
    grid_size: number;
    obstacle_density_actual: number;
    dead_end_probability_config: number;
    path_complexity_config: number;
    path_length: number | null;
    reachable_cells: number;
    total_cells: number;
    dead_end_cells: number;
    branching_factor: number;
    connectivity: number;
    is_valid: boolean;
    validation_reasons: string[];
    difficulty_estimate: number;
    generation_score: number;
  };
  validation: { is_valid: boolean; reasons: string[] };
}

// Phase 28 (Section 16) -- the raw WeaknessVector shape, used for
// external-agent responses (not yet a persisted AgentWeakness row).
export interface WeaknessVectorRead {
  obstacle_avoidance: number;
  exploration: number;
  dead_end_recovery: number;
  path_efficiency: number;
  short_horizon_planning: number;
  long_horizon_planning: number;
  primary_weakness: string;
  secondary_weaknesses: string[];
  episodes_analyzed: number;
}

// Phase 28 (Section 16) -- the honest, current state of the external-agent
// connection. `connected: false` must always render as "Waiting for
// connection", never as training/interacting.
export interface AgentStateRead {
  experiment_id: number;
  agent_mode: AgentMode;
  connected: boolean;
  status: ExperimentStatus;
  environment_id: string | null;
  episode: number | null;
  step: number | null;
  last_reward: number | null;
  last_terminated: boolean | null;
  last_truncated: boolean | null;
  trajectories_recorded: number;
  latest_weakness: WeaknessVectorRead | null;
}

export interface AgentResetResponse {
  experiment_id: number;
  environment_id: string;
  episode: number;
  observation: number[];
  info: Record<string, unknown>;
}

export interface AgentStepRequest {
  action: number;
}

export interface AgentStepResponse {
  experiment_id: number;
  environment_id: string;
  episode: number;
  step: number;
  observation: number[];
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: Record<string, unknown>;
  episode_summary: Record<string, unknown> | null;
  weakness: WeaknessVectorRead | null;
  next_environment: GeneratedEnvironmentRead | null;
}

// Phase 18 WebSocket tick shape (Section 25), extended by Phase 28
// (Section 19) to also carry environment-only / external-agent event
// types. `type` distinguishes which event this is; the training-shaped
// fields below only ever appear on demo_ppo ticks.
export interface TrainingTick {
  type?: string;
  mode?: "environment_only" | "external" | "demo_ppo";
  episode?: number | null;
  reward?: number;
  success_rate?: number;
  environment_id?: string;
  primary_weakness?: string | null;
  learning_progress?: number | null;
  training_status?: "training" | "step_completed";
  // Phase 28 environment-only/external event payloads (Section 19) --
  // present only on the corresponding event `type`.
  step_index?: number;
  record?: GeneratedEnvironmentRead;
  reason?: string;
  weakness?: WeaknessVectorRead;
  terminated?: boolean;
  truncated?: boolean;
}

export const WEAKNESS_DIMENSIONS = [
  "obstacle_avoidance",
  "exploration",
  "dead_end_recovery",
  "path_efficiency",
  "short_horizon_planning",
  "long_horizon_planning",
] as const;
export type WeaknessDimension = (typeof WEAKNESS_DIMENSIONS)[number];

// Mode B -- mirrors backend/app/schemas/project.py exactly (see that
// file's own header comment: kept in sync by hand, no generated client).

export type ProjectSource = "local" | "upload" | "built_in";
export type ProjectStatus = "draft" | "scanned" | "configured" | "validated" | "validation_failed";
export type MetricsSource = "file" | "stdout";

export interface ProjectRegisterRequest {
  root_path: string;
  name?: string | null;
  agent_id?: number | null;
}

export interface ProjectRead {
  id: number;
  name: string;
  root_path: string;
  source: ProjectSource;
  agent_id: number | null;
  language: string | null;
  framework: string | null;
  task: string | null;
  status: ProjectStatus;
  capabilities: string[];
  created_at: string;
  updated_at: string;
}

export interface ScanResultRead {
  root: string;
  language: string | null;
  framework: string | null;
  python_version: string | null;
  gpu: { available: boolean; name: string | null };
  training_candidates: string[];
  evaluation_candidates: string[];
  dataset_candidates: string[];
  artifact_candidates: string[];
  dependency_files: string[];
}

export interface ProjectDetailRead extends ProjectRead {
  detected_config: ScanResultRead | null;
  config_yaml: string | null;
  validation_result: ProjectValidationRead | null;
}

// -- ai-env.yaml contract, as the nested request/response shape ----------

export interface ProjectMetaSchema {
  task?: string | null;
  framework?: string | null;
}

export interface TrainingConfigSchema {
  command: string;
  entrypoint?: string | null;
  working_directory: string;
  timeout_seconds: number;
}

export interface EvaluationConfigSchema {
  command?: string | null;
  entrypoint?: string | null;
  working_directory?: string | null;
  timeout_seconds?: number | null;
  metrics: string[];
  metrics_file?: string | null;
  metrics_source?: MetricsSource | null;
}

export interface ArtifactConfigSchema {
  path?: string | null;
  type?: string | null;
}

export interface DatasetConfigSchema {
  path?: string | null;
}

export interface CurriculumConfigSchema {
  strategy: CurriculumType;
}

export type AdapterType = "command" | "custom";

export interface AdapterConfigSchema {
  type: AdapterType;
  entrypoint?: string | null;
  python: string;
}

export interface OptimizationConfigSchema {
  max_iterations: number;
  target_metric?: string | null;
  target_value?: number | null;
  minimum_improvement: number;
}

export interface ProjectConfigUpdate {
  name: string;
  project: ProjectMetaSchema;
  training: TrainingConfigSchema;
  evaluation: EvaluationConfigSchema;
  artifact: ArtifactConfigSchema;
  dataset: DatasetConfigSchema;
  curriculum: CurriculumConfigSchema;
  optimization: OptimizationConfigSchema;
  adapter: AdapterConfigSchema;
}

export type ProjectConfigRead = Omit<ProjectConfigUpdate, "name">;

export const DEFAULT_PROJECT_CONFIG_UPDATE: ProjectConfigUpdate = {
  name: "",
  project: { task: null, framework: null },
  training: { command: "", entrypoint: null, working_directory: ".", timeout_seconds: 86400 },
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

export interface ValidationIssueRead {
  field: string;
  message: string;
  severity: "error" | "warning";
}

export interface ProjectValidationRead {
  ok: boolean;
  issues: ValidationIssueRead[];
}

// -- runs (train + evaluate cycles) ---------------------------------------

export type ProjectRunStatus =
  | "queued"
  | "running_training"
  | "running_evaluation"
  | "completed"
  | "failed"
  | "cancelled";

export interface ArtifactInfoRead {
  path: string;
  exists: boolean;
  size_bytes: number | null;
  type: string | null;
}

export interface TrainResultRead {
  status: string;
  command: string;
  exit_code: number | null;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  stdout_tail: string;
  stderr_tail: string;
  artifact: ArtifactInfoRead | null;
}

export interface EvaluationResultRead {
  status: string;
  metrics: Record<string, number>;
  samples: number | null;
  errors: number | null;
  artifact: string | null;
  warnings: string[];
}

export interface ProjectRunRead {
  id: number;
  project_id: number;
  run_number: number;
  status: ProjectRunStatus;
  started_at: string;
  ended_at: string | null;
  train_result: TrainResultRead | null;
  evaluation_result: EvaluationResultRead | null;
  managed_artifact_path: string | null;
  error: string | null;
}

// -- progress (Section 41, generalized: real run-over-run metric trends) --

export interface MetricTrendPointRead {
  run_number: number;
  value: number;
}

export interface MetricTrendRead {
  metric: string;
  points: MetricTrendPointRead[];
  latest_value: number;
  previous_value: number | null;
  delta_from_previous: number | null;
}

export interface ProjectProgressRead {
  metrics: MetricTrendRead[];
  completed_run_count: number;
  target_metric: string | null;
  target_value: number | null;
  latest_target_value: number | null;
  distance_to_target: number | null;
  moved_closer_to_target: boolean | null;
}

// -- Local Agents (Section 5) ----------------------------------------------

export type AgentStatus = "connected" | "unreachable" | "unauthorized" | "unknown";

export interface AgentRegisterRequest {
  name: string;
  base_url: string;
  // Write-only: the backend stores it to authenticate to the agent, and
  // never returns it (AgentRead only says whether one is stored).
  token?: string | null;
}

export interface AgentRead {
  id: number;
  name: string;
  base_url: string;
  has_token: boolean;
  status: AgentStatus;
  last_seen_at: string | null;
  created_at: string;
}
