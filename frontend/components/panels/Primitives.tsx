import type { ReactNode } from "react";

export function Bar({
  label,
  value,
  color = "#3f8cff",
}: {
  label: string;
  value: number;
  color?: string;
}) {
  const clamped = Math.max(0, Math.min(1, value));
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-36 text-app-mid shrink-0 truncate">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-[#1a2236] overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${clamped * 100}%`, background: color }} />
      </div>
      <span className="w-10 text-right text-app-hi font-mono">{value.toFixed(2)}</span>
    </div>
  );
}

export function StatChip({
  label,
  value,
  color,
}: {
  label: string;
  value: ReactNode;
  color?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wider text-app-low">{label}</span>
      <span className="text-sm font-semibold" style={{ color: color ?? "#e6ebf5" }}>
        {value}
      </span>
    </div>
  );
}

const STATUS_COLORS: Record<string, string> = {
  pending: "#8b96ab",
  running: "#3f8cff",
  completed: "#22c55e",
  failed: "#ef4444",
  stopped: "#f97a2c",
  // Phase 28 -- environment-only + external-agent statuses (Section 18).
  generating: "#3f8cff",
  validating: "#3f8cff",
  analyzing: "#3f8cff",
  ready: "#22c55e",
  waiting_for_agent: "#8b96ab",
  agent_connected: "#3f8cff",
  interacting: "#3f8cff",
  analyzing_trajectory: "#f97a2c",
  adapting: "#f97a2c",
};

export function StatusBadge({ status }: { status: string }) {
  const color = STATUS_COLORS[status] ?? "#8b96ab";
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium" style={{ color }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {status.replace(/_/g, " ")}
    </span>
  );
}

// Phase 28 (Section 29) -- agent-mode badges for the experiment list/detail
// pages, so the mode is visible at a glance and can never be confused with
// the run status.
const AGENT_MODE_LABELS: Record<string, string> = {
  environment_only: "ENVIRONMENT ONLY",
  external: "EXTERNAL AGENT",
  demo_ppo: "DEMO PPO",
};

const AGENT_MODE_COLORS: Record<string, string> = {
  environment_only: "#3f8cff",
  external: "#a855f7",
  demo_ppo: "#f97a2c",
};

export function AgentModeBadge({ agentMode }: { agentMode: string }) {
  const label = AGENT_MODE_LABELS[agentMode] ?? agentMode.toUpperCase();
  const color = AGENT_MODE_COLORS[agentMode] ?? "#8b96ab";
  return (
    <span
      className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold tracking-wide"
      style={{ color, background: `${color}22`, border: `1px solid ${color}55` }}
    >
      {label}
    </span>
  );
}

// Mode B -- project status / capability badges (docs/projects-and-adapters.md).
const PROJECT_STATUS_COLORS: Record<string, string> = {
  draft: "#8b96ab",
  scanned: "#3f8cff",
  configured: "#3f8cff",
  validated: "#22c55e",
  validation_failed: "#ef4444",
};

export function ProjectStatusBadge({ status }: { status: string }) {
  const color = PROJECT_STATUS_COLORS[status] ?? "#8b96ab";
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium" style={{ color }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {status.replace(/_/g, " ")}
    </span>
  );
}

export function CapabilityBadge({ capability }: { capability: string }) {
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold tracking-wide text-app-mid bg-app-panel2 border border-app-border">
      {capability.replace(/_/g, " ").toUpperCase()}
    </span>
  );
}

const RUN_STATUS_COLORS: Record<string, string> = {
  queued: "#8b96ab",
  running_training: "#3f8cff",
  running_evaluation: "#3f8cff",
  completed: "#22c55e",
  failed: "#ef4444",
  cancelled: "#f97a2c",
};

export function RunStatusBadge({ status }: { status: string }) {
  const color = RUN_STATUS_COLORS[status] ?? "#8b96ab";
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium" style={{ color }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {status.replace(/_/g, " ")}
    </span>
  );
}

const AGENT_STATUS_COLORS: Record<string, string> = {
  connected: "#22c55e",
  unreachable: "#ef4444",
  unauthorized: "#f97a2c",
  unknown: "#8b96ab",
};

export function AgentStatusBadge({ status }: { status: string }) {
  const color = AGENT_STATUS_COLORS[status] ?? "#8b96ab";
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium" style={{ color }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {status}
    </span>
  );
}
