import type {
  AgentRead,
  AgentRegisterRequest,
  AgentResetResponse,
  AgentStateRead,
  AgentStepResponse,
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
  ProjectConfigRead,
  ProjectConfigUpdate,
  ProjectDetailRead,
  ProjectProgressRead,
  ProjectRead,
  ProjectRegisterRequest,
  ProjectRunRead,
  ProjectValidationRead,
  ScanResultRead,
  TrajectoryRead,
  WeaknessRead,
} from "@/types/api";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_BASE_URL ?? "ws://localhost:8000";

// Mode B's command-executing routes (/api/projects, /api/agents) require
// this bearer token server-side (app/api/auth.py, AI_ENV_API_TOKEN);
// Mode A's GridWorld/PPO routes do not and ignore it. Sent on every
// request rather than only Mode B ones, since an extra header on an
// unprotected route is harmless and it keeps `request()` from needing to
// know which routes care.
//
// Honest limitation, not a real secret: NEXT_PUBLIC_* is baked into the
// client-side JS bundle, so anyone who can load this app can read it out
// of the network tab or the bundle itself. This matches
// AI_ENV_API_TOKEN's own framing (docs/deployment.md) as one shared
// operator secret for a trusted network, not a substitute for real user
// authentication -- there is none here yet.
const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN;

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: unknown,
  ) {
    super(`API request failed with status ${status}`);
  }
}

/** Pulls FastAPI's `{"detail": "..."}` out of an `ApiError`'s body when
 * present, falling back to a generic message otherwise. Shared by every
 * form that surfaces API errors (Section 90: "what happened, why, what
 * the user can do next" -- the backend's `detail` string is usually the
 * "why"). */
export function extractErrorMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    if (typeof err.body === "object" && err.body && "detail" in (err.body as Record<string, unknown>)) {
      return String((err.body as Record<string, unknown>).detail);
    }
    return `${fallback} (${err.status})`;
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {}),
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      // response wasn't JSON; leave body null
    }
    throw new ApiError(res.status, body);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createExperiment: (payload: ExperimentCreate) =>
    request<ExperimentRead>("/api/experiments", { method: "POST", body: JSON.stringify(payload) }),

  listExperiments: () => request<ExperimentRead[]>("/api/experiments"),

  getExperiment: (id: number) => request<ExperimentRead>(`/api/experiments/${id}`),

  startExperiment: (id: number) =>
    request<ExperimentStatusRead>(`/api/experiments/${id}/start`, { method: "POST" }),

  stopExperiment: (id: number) =>
    request<ExperimentStatusRead>(`/api/experiments/${id}/stop`, { method: "POST" }),

  getStatus: (id: number) => request<ExperimentStatusRead>(`/api/experiments/${id}/status`),

  getMetrics: (id: number) => request<MetricsRead>(`/api/experiments/${id}/metrics`),

  getEpisodes: (id: number) => request<EpisodeRead[]>(`/api/experiments/${id}/episodes`),

  getTrajectories: (id: number) => request<TrajectoryRead[]>(`/api/experiments/${id}/trajectories`),

  getFailures: (id: number) => request<FailureRead[]>(`/api/experiments/${id}/failures`),

  getWeaknesses: (id: number) => request<WeaknessRead[]>(`/api/experiments/${id}/weaknesses`),

  getEnvironments: (id: number) => request<EnvironmentRead[]>(`/api/experiments/${id}/environments`),

  getEnvironmentLayout: (experimentId: number, environmentId: number) =>
    request<EnvironmentLayoutRead>(`/api/experiments/${experimentId}/environments/${environmentId}/layout`),

  getLearningProgress: (id: number) =>
    request<LearningProgressRead[]>(`/api/experiments/${id}/learning-progress`),

  getGeneralization: (id: number) =>
    request<GeneralizationRead[]>(`/api/experiments/${id}/generalization`),

  getComparison: (id: number, compareTo?: number[]) => {
    const query = compareTo && compareTo.length ? `?compare_to=${compareTo.join(",")}` : "";
    return request<ComparisonEntry[]>(`/api/experiments/${id}/comparison${query}`);
  },

  // Phase 28 (Section 9) -- external-agent HTTP contract. Any external
  // agent process could call these same three endpoints directly; this
  // app's own "External Agent" page (Section 16) uses them purely to
  // *display* what's happening, and only issues reset/step itself when the
  // person explicitly drives the demo from the UI.
  resetAgent: (id: number) =>
    request<AgentResetResponse>(`/api/experiments/${id}/agent/reset`, { method: "POST" }),

  stepAgent: (id: number, action: number) =>
    request<AgentStepResponse>(`/api/experiments/${id}/agent/step`, {
      method: "POST",
      body: JSON.stringify({ action }),
    }),

  getAgentState: (id: number) => request<AgentStateRead>(`/api/experiments/${id}/agent/state`),

  // Mode B -- registered user-owned AI projects (docs/projects-and-adapters.md).
  registerProject: (payload: ProjectRegisterRequest) =>
    request<ProjectRead>("/api/projects/register", { method: "POST", body: JSON.stringify(payload) }),

  listProjects: () => request<ProjectRead[]>("/api/projects"),

  getProject: (id: number) => request<ProjectDetailRead>(`/api/projects/${id}`),

  scanProject: (id: number) => request<ScanResultRead>(`/api/projects/${id}/scan`, { method: "POST" }),

  getProjectConfig: (id: number) => request<ProjectConfigRead>(`/api/projects/${id}/config`),

  updateProjectConfig: (id: number, payload: ProjectConfigUpdate) =>
    request<ProjectRead>(`/api/projects/${id}/config`, { method: "PUT", body: JSON.stringify(payload) }),

  validateProject: (id: number) =>
    request<ProjectValidationRead>(`/api/projects/${id}/validate`, { method: "POST" }),

  startProjectRun: (id: number) =>
    request<ProjectRunRead>(`/api/projects/${id}/runs`, { method: "POST" }),

  listProjectRuns: (id: number) => request<ProjectRunRead[]>(`/api/projects/${id}/runs`),

  getProjectRun: (id: number, runId: number) =>
    request<ProjectRunRead>(`/api/projects/${id}/runs/${runId}`),

  cancelProjectRun: (id: number, runId: number) =>
    request<ProjectRunRead>(`/api/projects/${id}/runs/${runId}/cancel`, { method: "POST" }),

  getProjectProgress: (id: number) => request<ProjectProgressRead>(`/api/projects/${id}/progress`),

  assignProjectAgent: (id: number, agentId: number | null) =>
    request<ProjectRead>(`/api/projects/${id}/agent`, {
      method: "PUT",
      body: JSON.stringify({ agent_id: agentId }),
    }),

  // Local Agents (Section 5) -- a separate process, typically on the
  // user's own machine, that scan/validate/run for a project are proxied
  // to instead of the backend executing them itself.
  registerLocalAgent: (payload: AgentRegisterRequest) =>
    request<AgentRead>("/api/agents/register", { method: "POST", body: JSON.stringify(payload) }),

  listLocalAgents: () => request<AgentRead[]>("/api/agents"),

  getLocalAgent: (id: number) => request<AgentRead>(`/api/agents/${id}`),

  pingLocalAgent: (id: number) => request<AgentRead>(`/api/agents/${id}/ping`, { method: "POST" }),

  updateLocalAgentToken: (id: number, token: string | null) =>
    request<AgentRead>(`/api/agents/${id}/token`, { method: "PUT", body: JSON.stringify({ token }) }),
};

export function wsUrlForExperiment(id: number): string {
  return `${WS_BASE_URL}/ws/experiments/${id}`;
}
