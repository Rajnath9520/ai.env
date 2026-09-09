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
