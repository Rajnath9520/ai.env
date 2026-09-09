import type {
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
  TrajectoryRead,
  WeaknessRead,
} from "@/types/api";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_BASE_URL ?? "ws://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: unknown,
  ) {
    super(`API request failed with status ${status}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
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
};

export function wsUrlForExperiment(id: number): string {
  return `${WS_BASE_URL}/ws/experiments/${id}`;
}
